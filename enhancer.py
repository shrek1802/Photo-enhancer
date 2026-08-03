from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ai_engine import NeuralEngine
from auto_ai_pipeline import PipelineReport
from identity_guard import FaceIdentityGuard
from photo_analysis import PhotoAnalysis, PhotoAnalyser
from photoperfect_engine import PhotoPerfectEngine, RepairPlan, Validation

SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED


@dataclass
class EnhanceOptions:
    preset: str = 'Auto Detect'
    strength: str = 'natural'
    upscale: str = 'Original size'
    lift_shadows: bool = True
    recover_highlights: bool = True
    reduce_flare: bool = True
    denoise: bool = True
    sharpen: bool = True
    face_aware: bool = True
    auto_rotate: bool = True
    straighten_horizon: bool = True
    portrait_finish: bool = True
    neural_ai: bool = True
    automatic_restoration: bool = True
    remove_screenshot_ui: bool = True
    identity_lock: bool = True
    good_photo_polish: bool = True
    jpeg_quality: int = 95


@dataclass
class ProcessResult:
    review_needed: bool
    analysis: PhotoAnalysis
    pipeline_report: PipelineReport | None = None
    faces_protected: int = 0
    repair_plan: RepairPlan | None = None
    validation: Validation | None = None


class PhotoEnhancer:
    """Public enhancement facade used by the Studio UI.

    Auto Detect is now routed through PhotoPerfectEngine, which inspects the
    image, classifies it, builds a repair plan and validates the result before
    optional neural upscaling. Identity Lock restores original face pixels after
    non-geometric corrections so faces are never generated or structurally changed.
    """

    def __init__(self, options: EnhanceOptions):
        self.options = options
        self.analyser = PhotoAnalyser()
        self.engine = PhotoPerfectEngine()
        self.neural = NeuralEngine(enabled=options.neural_ai)
        self.identity = FaceIdentityGuard()

    @property
    def engine_message(self) -> str:
        identity = 'Identity Lock ON' if self.options.identity_lock else 'Identity Lock OFF'
        return f'PhotoPerfect Engine v2; {self.neural.status.message}; {identity}'

    def _read(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        if self.options.auto_rotate:
            image = ImageOps.exif_transpose(image)
        return cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _mode_name(mode: str) -> str:
        aliases = {
            'Smart Auto': 'Auto Detect',
            'Event / Christening': 'Celebrations',
            'Professional Portrait': 'Portrait',
            'Old Photo Restoration': 'Auto Restore',
            'Night / Low Light': 'Low Light',
        }
        return aliases.get(mode, mode)

    @staticmethod
    def _straighten(image: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 0.45 or abs(angle) > 8:
            return image
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        crop = int(min(w, h) * min(abs(angle) / 90, 0.035))
        return rotated[crop:h-crop, crop:w-crop] if crop else rotated

    @staticmethod
    def _safe_flare_cleanup(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)
        mask = ((((value > 242) & (saturation < 58)) |
                 ((value > 232) & (saturation > 125))).astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        safe = np.zeros_like(mask)
        total = image.shape[0] * image.shape[1]
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if 10 <= area <= total * 0.006:
                safe[labels == label] = 255
        if not np.any(safe):
            return image
        safe = cv2.dilate(safe, np.ones((5, 5), np.uint8), iterations=1)
        return cv2.inpaint(image, safe, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _face_relight(image: np.ndarray, faces: list[tuple[int, int, int, int]]) -> np.ndarray:
        result = image.copy()
        for x, y, w, h in faces:
            pad = int(w * 0.10)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
            roi = result[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_l = float(l.mean())
            if mean_l >= 118:
                continue
            gain = min(1.11, 118 / max(mean_l, 1))
            new_l = np.clip(l.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            corrected = cv2.cvtColor(cv2.merge([new_l, a, b]), cv2.COLOR_LAB2BGR)
            result[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.48, corrected, 0.52, 0)
        return result

    def _upscale(self, image: np.ndarray, plan: RepairPlan) -> np.ndarray:
        h, w = image.shape[:2]
        selected = self.options.upscale
        # Screenshot Recovery and very small images need visible resolution recovery
        # even when the user left output at Original size.
        auto_2x = (
            selected == 'Original size'
            and (plan.inspection.is_screenshot or min(h, w) < 720)
        )
        if selected == '2× AI Upscale' or selected == '2× upscale' or auto_2x:
            return self.neural.upscale(image)
        if selected in {'4K AI Upscale', '4K long edge'} and max(h, w) < 3840:
            if self.neural.status.model_loaded and max(h, w) * self.neural.scale <= 4608:
                image = self.neural.upscale(image)
                h, w = image.shape[:2]
            if max(h, w) < 3840:
                scale = 3840 / max(h, w)
                image = cv2.resize(
                    image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4
                )
        return image

    def process(self, source: Path, destination: Path) -> ProcessResult:
        analysis = self.analyser.analyse(source)
        original = self._read(source)
        mode = self._mode_name(self.options.preset)

        # Detect faces before any crop or enhancement. Identity Lock is applied
        # only when the processed image keeps the same geometry.
        original_faces = self.identity.detect(original)

        processed, plan, validation = self.engine.process(original, mode)

        # Respect explicit feature switches in Advanced mode. Auto Detect normally
        # leaves these decisions to the engine.
        if self.options.reduce_flare and not plan.inspection.is_monochrome:
            processed = self._safe_flare_cleanup(processed)

        if self.options.straighten_horizon and not (
            self.options.identity_lock and original_faces
        ) and processed.shape == original.shape:
            processed = self._straighten(processed, analysis.horizon_angle)

        if self.options.face_aware:
            current_faces = self.identity.detect(processed)
            processed = self._face_relight(processed, current_faces)

        faces_protected = 0
        minimum_similarity = 1.0
        if self.options.identity_lock and processed.shape == original.shape:
            locked = self.identity.apply(original, processed)
            processed = locked.image
            faces_protected = locked.faces_protected
            minimum_similarity = locked.minimum_similarity

        processed = self._upscale(processed, plan)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(destination), processed,
            [cv2.IMWRITE_JPEG_QUALITY, int(self.options.jpeg_quality)],
        ):
            raise OSError(f'Could not write {destination}')

        review = bool(analysis.review_reason)
        if not validation.accepted:
            review = True
        if plan.inspection.quality_score < 38:
            review = True
        if self.options.identity_lock and faces_protected and minimum_similarity < 0.72:
            review = True

        return ProcessResult(
            review_needed=review,
            analysis=analysis,
            faces_protected=faces_protected,
            repair_plan=plan,
            validation=validation,
        )
