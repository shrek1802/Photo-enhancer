from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from model_manager import PhotoPerfectModelManager
from photoperfect_engine import Inspection, RepairPlan


@dataclass
class CapabilityStep:
    capability: str
    model_path: Path | None
    available: bool
    provider: str = ''
    applied: bool = False
    message: str = ''


@dataclass
class CapabilityReport:
    requested: list[str] = field(default_factory=list)
    steps: list[CapabilityStep] = field(default_factory=list)

    @property
    def applied(self) -> list[str]:
        return [step.capability for step in self.steps if step.applied]

    @property
    def missing(self) -> list[str]:
        return [step.capability for step in self.steps if not step.available]


class PhotoPerfectCapabilityRuntime:
    """Runs optional Auto Essentials ONNX capabilities with safe fallbacks."""

    WHOLE_IMAGE_CAPABILITIES = {
        'jpeg_repair', 'denoise', 'deblur', 'colour', 'lighting', 'super_resolution'
    }
    FACE_CAPABILITIES = {'face_protect', 'face_recovery'}

    def __init__(
        self,
        models_root: Path | str = 'models',
        enabled: bool = True,
        tile_size: int = 256,
    ) -> None:
        self.enabled = enabled
        self.tile_size = max(64, int(tile_size))
        self.manager = PhotoPerfectModelManager(models_root)
        self._sessions: dict[str, Any] = {}
        self._providers: dict[str, str] = {}
        self._ort = None
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if enabled:
            try:
                import onnxruntime as ort
                self._ort = ort
            except Exception:
                self._ort = None

    @staticmethod
    def requested_capabilities(inspection: Inspection, plan: RepairPlan) -> list[str]:
        requested: list[str] = []
        if inspection.is_screenshot or inspection.compression_score > 12:
            requested.append('jpeg_repair')
        if inspection.noise_score > 8.5:
            requested.append('denoise')
        if inspection.blur_score < 160:
            requested.append('deblur')
        if inspection.face_count > 0 and (
            inspection.smallest_face_ratio < 0.025
            or inspection.blur_score < 220
            or inspection.is_screenshot
        ):
            requested.append('face_recovery')
        if inspection.face_count > 0:
            requested.append('face_protect')
        if inspection.dark_fraction > 0.08:
            requested.append('lighting')
        if not inspection.is_monochrome:
            requested.append('colour')
        if inspection.is_low_resolution or inspection.is_screenshot:
            requested.append('super_resolution')
        return list(dict.fromkeys(requested))

    def _preferred_providers(self) -> list[str]:
        if self._ort is None:
            return []
        available = self._ort.get_available_providers()
        return [
            provider for provider in (
                'CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'
            ) if provider in available
        ]

    def _load_session(self, capability: str, path: Path) -> tuple[Any | None, str, str]:
        if capability in self._sessions:
            return self._sessions[capability], self._providers[capability], ''
        if self._ort is None:
            return None, '', 'ONNX Runtime unavailable'
        providers = self._preferred_providers()
        if not providers:
            return None, '', 'No compatible ONNX provider found'
        try:
            options = self._ort.SessionOptions()
            options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = self._ort.InferenceSession(str(path), sess_options=options, providers=providers)
            provider = session.get_providers()[0] if session.get_providers() else providers[-1]
            self._sessions[capability] = session
            self._providers[capability] = provider
            return session, provider, ''
        except Exception as exc:
            return None, '', f'Model load failed: {exc}'

    @staticmethod
    def _input_layout(shape: list[Any]) -> str:
        if len(shape) != 4:
            return 'nchw'
        channel_first = shape[1] in (1, 3, 4)
        channel_last = shape[-1] in (1, 3, 4)
        return 'nhwc' if channel_last and not channel_first else 'nchw'

    @staticmethod
    def _fixed_input_size(shape: list[Any], layout: str) -> tuple[int, int] | None:
        try:
            if layout == 'nhwc':
                h, w = shape[1], shape[2]
            else:
                h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return int(w), int(h)
        except Exception:
            pass
        return None

    @staticmethod
    def _normalise_output(output: np.ndarray, layout: str) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 4:
            array = array[0]
        if layout == 'nchw' and array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3:
            raise ValueError(f'Unsupported model output shape: {array.shape}')
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        array = array[..., :3]
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)

    def _run_image(self, session: Any, image_bgr: np.ndarray) -> np.ndarray:
        input_info = session.get_inputs()[0]
        output_name = session.get_outputs()[0].name
        layout = self._input_layout(list(input_info.shape))
        original_size = (image_bgr.shape[1], image_bgr.shape[0])
        fixed = self._fixed_input_size(list(input_info.shape), layout)
        prepared = cv2.resize(image_bgr, fixed, interpolation=cv2.INTER_CUBIC) if fixed else image_bgr
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = rgb[None, ...] if layout == 'nhwc' else np.transpose(rgb, (2, 0, 1))[None, ...]
        prediction = session.run([output_name], {input_info.name: tensor})[0]
        result = self._normalise_output(prediction, layout)
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        if fixed and result.shape[:2] != image_bgr.shape[:2]:
            result = cv2.resize(result, original_size, interpolation=cv2.INTER_CUBIC)
        return result

    def _run_tiled(self, session: Any, image: np.ndarray) -> np.ndarray:
        input_info = session.get_inputs()[0]
        layout = self._input_layout(list(input_info.shape))
        fixed = self._fixed_input_size(list(input_info.shape), layout)
        if fixed:
            return self._run_image(session, image)

        h, w = image.shape[:2]
        tile = self.tile_size
        overlap = min(32, tile // 8)
        step = max(32, tile - overlap)
        output: np.ndarray | None = None
        weight: np.ndarray | None = None
        scale_y = scale_x = 1.0

        for y in range(0, h, step):
            for x in range(0, w, step):
                y1, x1 = min(y + tile, h), min(x + tile, w)
                prediction = self._run_image(session, image[y:y1, x:x1])
                if output is None:
                    scale_y = prediction.shape[0] / max(y1 - y, 1)
                    scale_x = prediction.shape[1] / max(x1 - x, 1)
                    oh, ow = int(round(h * scale_y)), int(round(w * scale_x))
                    output = np.zeros((oh, ow, 3), dtype=np.float32)
                    weight = np.zeros((oh, ow, 1), dtype=np.float32)
                oy, ox = int(round(y * scale_y)), int(round(x * scale_x))
                ph, pw = prediction.shape[:2]
                oy1, ox1 = min(oy + ph, output.shape[0]), min(ox + pw, output.shape[1])
                patch = prediction[:oy1 - oy, :ox1 - ox].astype(np.float32)
                output[oy:oy1, ox:ox1] += patch
                weight[oy:oy1, ox:ox1] += 1.0

        if output is None or weight is None:
            return image
        return np.clip(output / np.maximum(weight, 1.0), 0, 255).astype(np.uint8)

    def _detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        minimum = max(28, min(image.shape[:2]) // 24)
        faces = self._face_detector.detectMultiScale(gray, 1.08, 5, minSize=(minimum, minimum))
        return [tuple(map(int, face)) for face in faces]

    @staticmethod
    def _soft_mask(width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.ellipse(
            mask,
            (width // 2, height // 2),
            (max(1, int(width * 0.46)), max(1, int(height * 0.49))),
            0, 0, 360, 1.0, -1,
        )
        sigma = max(3.0, min(width, height) * 0.08)
        return cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]

    def _run_faces(self, session: Any, image: np.ndarray, capability: str) -> tuple[np.ndarray, int]:
        faces = self._detect_faces(image)
        if not faces:
            return image, 0
        result = image.copy()
        applied = 0
        for x, y, w, h in faces:
            pad_x, pad_y = int(w * 0.28), int(h * 0.35)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y)
            roi = result[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            restored = self._run_image(session, roi)
            if restored.shape[:2] != roi.shape[:2]:
                restored = cv2.resize(restored, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_CUBIC)
            # Identity-safe blend: face recovery is deliberately restrained.
            strength = 0.42 if capability == 'face_recovery' else 0.28
            candidate = cv2.addWeighted(roi, 1.0 - strength, restored, strength, 0)
            mask = self._soft_mask(roi.shape[1], roi.shape[0])
            blended = np.clip(roi.astype(np.float32) * (1.0 - mask) + candidate.astype(np.float32) * mask, 0, 255)
            result[y0:y1, x0:x1] = blended.astype(np.uint8)
            applied += 1
        return result, applied

    def apply(
        self,
        image: np.ndarray,
        inspection: Inspection,
        plan: RepairPlan,
        allow_super_resolution: bool = True,
    ) -> tuple[np.ndarray, CapabilityReport]:
        requested = self.requested_capabilities(inspection, plan)
        if not allow_super_resolution:
            requested = [item for item in requested if item != 'super_resolution']
        report = CapabilityReport(requested=requested)
        working = image

        for capability in requested:
            path = self.manager.capability_path(capability)
            if path is None:
                report.steps.append(CapabilityStep(
                    capability, None, False, message='No installed Auto model for this capability'
                ))
                continue
            session, provider, error = self._load_session(capability, path)
            if session is None:
                report.steps.append(CapabilityStep(
                    capability, path, True, provider=provider, message=error
                ))
                continue
            try:
                if capability in self.FACE_CAPABILITIES:
                    candidate, face_count = self._run_faces(session, working, capability)
                    if face_count == 0:
                        report.steps.append(CapabilityStep(
                            capability, path, True, provider=provider,
                            message='No suitable face region found'
                        ))
                        continue
                    working = candidate
                    message = f'Applied safely to {face_count} face region(s)'
                elif capability in self.WHOLE_IMAGE_CAPABILITIES:
                    candidate = self._run_tiled(session, working)
                    if candidate.size == 0:
                        raise ValueError('Model returned an empty image')
                    working = candidate
                    message = 'Applied successfully'
                else:
                    report.steps.append(CapabilityStep(
                        capability, path, True, provider=provider,
                        message='No safe runner is registered for this capability'
                    ))
                    continue
                report.steps.append(CapabilityStep(
                    capability, path, True, provider=provider, applied=True, message=message
                ))
            except Exception as exc:
                report.steps.append(CapabilityStep(
                    capability, path, True, provider=provider,
                    message=f'Inference failed safely: {exc}'
                ))
        return working, report
