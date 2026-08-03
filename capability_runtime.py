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
    """Loads optional ONNX models by capability and applies them safely.

    Phase 3 deliberately separates *what* the repair planner requests from the
    exact model file used. Model packs can therefore be updated independently of
    the EXE. Missing or incompatible models are skipped without breaking the
    built-in photographic pipeline.
    """

    SAFE_WHOLE_IMAGE_CAPABILITIES = {
        'jpeg_repair', 'denoise', 'deblur', 'colour', 'super_resolution'
    }

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
        if not inspection.is_monochrome:
            requested.append('colour')
        if inspection.is_low_resolution or inspection.is_screenshot:
            requested.append('super_resolution')
        # Preserve ordering while removing duplicates.
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
            session = self._ort.InferenceSession(
                str(path), sess_options=options, providers=providers
            )
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
    def _normalise_output(output: np.ndarray, layout: str) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 4:
            array = array[0]
        if layout == 'nchw' and array.ndim == 3:
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        array = array[..., :3]
        if float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
        return np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)

    def _run_tile(self, session: Any, tile_bgr: np.ndarray) -> np.ndarray:
        input_info = session.get_inputs()[0]
        output_name = session.get_outputs()[0].name
        layout = self._input_layout(list(input_info.shape))
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = rgb[None, ...] if layout == 'nhwc' else np.transpose(rgb, (2, 0, 1))[None, ...]
        prediction = session.run([output_name], {input_info.name: tensor})[0]
        result = self._normalise_output(prediction, layout)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def _run_tiled(self, session: Any, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        tile = self.tile_size
        overlap = min(24, tile // 8)
        step = max(32, tile - overlap)
        output: np.ndarray | None = None
        weight: np.ndarray | None = None
        scale_y = scale_x = 1.0

        for y in range(0, h, step):
            for x in range(0, w, step):
                y1, x1 = min(y + tile, h), min(x + tile, w)
                prediction = self._run_tile(session, image[y:y1, x:x1])
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
                    capability, None, False, message='No installed model for capability'
                ))
                continue
            if capability not in self.SAFE_WHOLE_IMAGE_CAPABILITIES:
                report.steps.append(CapabilityStep(
                    capability, path, True, message='Capability requires a specialist safe region runner'
                ))
                continue
            session, provider, error = self._load_session(capability, path)
            if session is None:
                report.steps.append(CapabilityStep(
                    capability, path, True, provider=provider, message=error
                ))
                continue
            try:
                candidate = self._run_tiled(session, working)
                if candidate.size == 0:
                    raise ValueError('Model returned an empty image')
                working = candidate
                report.steps.append(CapabilityStep(
                    capability, path, True, provider=provider, applied=True,
                    message='Applied successfully'
                ))
            except Exception as exc:
                report.steps.append(CapabilityStep(
                    capability, path, True, provider=provider,
                    message=f'Inference failed safely: {exc}'
                ))
        return working, report
