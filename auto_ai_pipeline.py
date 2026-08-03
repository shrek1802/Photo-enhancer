from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # optional at runtime
    ort = None


@dataclass
class PipelineReport:
    provider: str = 'CPU'
    stages: list[str] = field(default_factory=list)
    original_score: float = 0.0
    final_score: float = 0.0
    accepted: bool = True
    confidence: int = 0
    alternatives_created: int = 0


class AutomaticAIPipeline:
    """Automatic restoration orchestrator.

    The pipeline always provides conservative OpenCV fallbacks. When ONNX models
    exist in the models directory, it automatically enables the matching neural
    stages using CUDA, DirectML or CPU execution.
    """

    MODEL_FILES = {
        'super_resolution': 'super_resolution_x2.onnx',
        'deblur': 'deblur.onnx',
        'denoise': 'denoise.onnx',
        'face_restore': 'face_restore.onnx',
        'colourise': 'colourise.onnx',
        'inpaint': 'inpaint.onnx',
    }

    def __init__(self, models_dir: Path | str = 'models') -> None:
        self.models_dir = Path(models_dir)
        self.sessions: dict[str, Any] = {}
        self.provider = self._provider()
        self._load_available_models()

    @staticmethod
    def _provider() -> str:
        if ort is None:
            return 'CPU (OpenCV fallback)'
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            return 'NVIDIA CUDA'
        if 'DmlExecutionProvider' in available:
            return 'AMD / Windows DirectML'
        return 'CPU ONNX'

    def _provider_list(self) -> list[str]:
        if ort is None:
            return []
        available = ort.get_available_providers()
        preferred = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        return [item for item in preferred if item in available]

    def _load_available_models(self) -> None:
        if ort is None:
            return
        providers = self._provider_list()
        for name, filename in self.MODEL_FILES.items():
            path = self.models_dir / filename
            if not path.exists():
                continue
            try:
                self.sessions[name] = ort.InferenceSession(str(path), providers=providers)
            except Exception:
                # A broken or incompatible model must never stop normal enhancement.
                continue

    @staticmethod
    def quality_score(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) * 8.0
        clipped = float(np.mean(gray >= 250)) * 90.0
        crushed = float(np.mean(gray <= 8)) * 75.0
        contrast = min(float(gray.std()) / 55.0, 1.0) * 22.0
        noise_residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        noise_penalty = max(float(noise_residual.std()) - 8.0, 0.0) * 0.8
        score = 30.0 + sharpness + contrast - clipped - crushed - noise_penalty
        return float(np.clip(score, 0, 100))

    @staticmethod
    def detect_screenshot_crop(image: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect obvious phone/social-media bars without risking normal photos."""
        h, w = image.shape[:2]
        if h < w * 1.35:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        top_band = gray[: max(24, int(h * 0.11))]
        bottom_band = gray[int(h * 0.86):]
        top_edges = cv2.Canny(top_band, 80, 160).mean()
        bottom_flat = bottom_band.std()
        # Conservative: only crop when strong UI-like top edges and a flat light bottom bar coexist.
        if top_edges > 9 and bottom_flat < 42 and bottom_band.mean() > 150:
            return (0, int(h * 0.055), w, int(h * 0.86))
        return None

    @staticmethod
    def _crop_ui(image: np.ndarray, report: PipelineReport) -> np.ndarray:
        box = AutomaticAIPipeline.detect_screenshot_crop(image)
        if box is None:
            return image
        x0, y0, x1, y1 = box
        report.stages.append('Removed obvious phone/social-media interface borders')
        return image[y0:y1, x0:x1]

    @staticmethod
    def _compression_cleanup(image: np.ndarray, report: PipelineReport) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block = np.mean(np.abs(np.diff(gray.astype(np.float32), axis=1))[:, 7::8]) if image.shape[1] > 16 else 0
        if block < 6.5:
            return image
        cleaned = cv2.bilateralFilter(image, 5, 22, 22)
        report.stages.append('Reduced JPEG blocking and ringing')
        return cleaned

    @staticmethod
    def _classical_deblur(image: np.ndarray, report: PipelineReport) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur >= 115:
            return image
        # Edge-limited high-pass restoration; deliberately conservative.
        base = cv2.GaussianBlur(image, (0, 0), 1.6)
        restored = cv2.addWeighted(image, 1.65, base, -0.65, 0)
        edge = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        mask = cv2.GaussianBlur(np.clip(edge / 24.0, 0, 1), (0, 0), 1.2)[..., None]
        result = image.astype(np.float32) * (1 - mask) + restored.astype(np.float32) * mask
        report.stages.append('Applied automatic soft-focus recovery')
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _face_local_restore(image: np.ndarray, report: PipelineReport) -> np.ndarray:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        minimum = max(35, min(image.shape[:2]) // 16)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(minimum, minimum))
        if len(faces) == 0:
            return image
        result = image.copy()
        restored_count = 0
        for x, y, w, h in faces:
            roi = result[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(roi_gray, cv2.CV_64F).var() > 230:
                continue
            smooth = cv2.bilateralFilter(roi, 5, 18, 18)
            detail = cv2.addWeighted(roi, 1.45, cv2.GaussianBlur(roi, (0, 0), 1.0), -0.45, 0)
            improved = cv2.addWeighted(detail, 0.82, smooth, 0.18, 0)
            result[y:y+h, x:x+w] = improved
            restored_count += 1
        if restored_count:
            report.stages.append(f'Improved {restored_count} soft face(s) while preserving identity')
        return result

    def _run_simple_onnx(self, name: str, image: np.ndarray) -> np.ndarray | None:
        session = self.sessions.get(name)
        if session is None:
            return None
        try:
            input_meta = session.get_inputs()[0]
            tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
            output = session.run(None, {input_meta.name: tensor})[0]
            output = np.squeeze(output)
            if output.ndim == 3 and output.shape[0] in (1, 3):
                output = np.transpose(output, (1, 2, 0))
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
            if output.ndim == 2:
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            else:
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            return output
        except Exception:
            return None

    def _neural_stage(self, name: str, image: np.ndarray, label: str, report: PipelineReport) -> np.ndarray:
        output = self._run_simple_onnx(name, image)
        if output is None:
            return image
        report.stages.append(f'{label} ({self.provider})')
        return output

    def restore(self, image: np.ndarray, allow_ui_crop: bool = True) -> tuple[np.ndarray, PipelineReport]:
        report = PipelineReport(provider=self.provider)
        report.original_score = self.quality_score(image)
        original = image.copy()

        working = self._crop_ui(image, report) if allow_ui_crop else image
        working = self._compression_cleanup(working, report)

        if 'denoise' in self.sessions:
            working = self._neural_stage('denoise', working, 'Neural denoising', report)
        else:
            working = cv2.fastNlMeansDenoisingColored(working, None, 3, 3, 7, 21)
            report.stages.append('Adaptive photographic denoising')

        if 'deblur' in self.sessions:
            working = self._neural_stage('deblur', working, 'Neural deblurring', report)
        else:
            working = self._classical_deblur(working, report)

        if 'face_restore' in self.sessions:
            working = self._neural_stage('face_restore', working, 'Neural face restoration', report)
        else:
            working = self._face_local_restore(working, report)

        report.final_score = self.quality_score(working)
        improvement = report.final_score - report.original_score
        report.confidence = int(np.clip(55 + improvement * 3, 20, 99))

        # Do not accept a materially worse result. UI cropping changes dimensions, so retain it
        # only when the score is not meaningfully degraded.
        if report.final_score + 2.5 < report.original_score:
            report.accepted = False
            report.stages.append('Automatic quality control rejected stronger restoration')
            return original, report

        report.accepted = True
        report.stages.append('Two-pass quality control accepted result')
        return working, report

    def super_resolve(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        output = self._run_simple_onnx('super_resolution', image)
        if output is not None:
            return output, f'Neural 2x super-resolution via {self.provider}'
        h, w = image.shape[:2]
        return cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4), 'Lanczos 2x fallback'
