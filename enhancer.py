from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ai_engine import NeuralEngine, app_directory
from auto_ai_pipeline import PipelineReport
from auto_profile_runtime import AutoProfileRuntime
from capability_runtime import CapabilityReport, CapabilityStep, PhotoPerfectCapabilityRuntime
from identity_guard import FaceIdentityGuard
from photo_analysis import PhotoAnalysis, PhotoAnalyser
from photoperfect_engine import PhotoPerfectEngine, RepairPlan, Validation
from photoperfect_intelligence import (
    IntelligenceReport, PhotoPerfectIntelligence, PostProcessValidation,
)
from version_info import AUTO_ENGINE_VERSION

SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED


@dataclass
class EnhanceOptions:
    preset: str = 'Auto Detect'
    strength: str = 'natural'
    upscale: str = 'Original size'
    quality_target: str = 'Professional'
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
    capability_report: CapabilityReport | None = None
    intelligence_report: IntelligenceReport | None = None
    post_validation: PostProcessValidation | None = None


class PhotoEnhancer:
    """Auto Engine facade used by PhotoPerfect Studio.

    Built-in restoration always remains available. Installed profile packs tune
    the safe local processing, while neural packs are resolved by capability.
    """

    def __init__(self, options: EnhanceOptions):
        self.options = options
        self.analyser = PhotoAnalyser()
        self.engine = PhotoPerfectEngine()
        self.intelligence = PhotoPerfectIntelligence()
        self.neural = NeuralEngine(enabled=options.neural_ai)
        models_root = app_directory() / 'models'
        self.capabilities = PhotoPerfectCapabilityRuntime(
            models_root=models_root, enabled=options.neural_ai
        )
        self.profiles = AutoProfileRuntime(models_root)
        self.identity = FaceIdentityGuard()

    @property
    def engine_message(self) -> str:
        identity = 'Identity Lock ON' if self.options.identity_lock else 'Identity Lock OFF'
        installed = self.capabilities.manager.installed_capabilities()
        profile_count = len(self.profiles.installed_profiles())
        capability_text = f'{len(installed)} Auto model capability/capabilities installed'
        return (
            f'Auto Engine v{AUTO_ENGINE_VERSION}; {self.neural.status.message}; '
            f'{capability_text}; {profile_count} specialist profile(s); '
            f'target {self.options.quality_target}; {identity}'
        )

    def _read(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        if self.options.auto_rotate:
            image = ImageOps.exif_transpose(image)
        return cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _mode_name(mode: str) -> str:
        aliases = {
            'Smart Auto': 'Auto Detect',
            'Event / Christening': 'Auto Celebrations',
            'Professional Portrait': 'Auto Portrait',
            'Old Photo Restoration': 'Auto Restore',
            'Night / Low Light': 'Auto Low Light',
            'Celebrations': 'Auto Celebrations',
            'Portrait': 'Auto Portrait',
            'Landscape': 'Auto Landscape',
            'Low Light': 'Auto Low Light',
            'Screenshot Recovery': 'Auto Screenshot Recovery',
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
    def _safe_flare_cleanup(
        image: np.ndarray,
        protected_faces: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)
        mask = ((((value > 242) & (saturation < 58)) |
                 ((value > 232) & (saturation > 125))).astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        for x, y, w, h in protected_faces:
            pad_x, pad_y = int(w * 0.55), int(h * 0.70)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y)
            mask[y0:y1, x0:x1] = 0

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        safe = np.zeros_like(mask)
        total = image.shape[0] * image.shape[1]
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if 10 <= area <= total * 0.004:
                safe[labels == label] = 255
        if not np.any(safe):
            return image
        safe = cv2.dilate(safe, np.ones((5, 5), np.uint8), iterations=1)
        return cv2.inpaint(image, safe, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _face_relight(image: np.ndarray, faces: list[tuple[int, int, int, int]]) -> np.ndarray:
        result = image.copy()
        for x, y, w, h in faces:
            pad_x, pad_y = int(w * 0.22), int(h * 0.28)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y)
            roi = result[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_l = float(np.percentile(l, 55))
            if mean_l >= 126:
                continue
            target = 124.0
            gain = min(1.16, target / max(mean_l, 1.0))
            new_l = np.clip(l.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            corrected = cv2.cvtColor(cv2.merge([new_l, a, b]), cv2.COLOR_LAB2BGR)
            candidate = cv2.addWeighted(roi, 0.38, corrected, 0.62, 0)

            mask = np.zeros(roi.shape[:2], dtype=np.float32)
            cv2.ellipse(
                mask,
                (roi.shape[1] // 2, roi.shape[0] // 2),
                (max(1, int(roi.shape[1] * 0.43)), max(1, int(roi.shape[0] * 0.47))),
                0, 0, 360, 1.0, -1,
            )
            mask = cv2.GaussianBlur(mask, (0, 0), max(3.0, min(roi.shape[:2]) * 0.10))[..., None]
            blended = np.clip(
                roi.astype(np.float32) * (1.0 - mask) + candidate.astype(np.float32) * mask,
                0, 255,
            )
            result[y0:y1, x0:x1] = blended.astype(np.uint8)
        return result

    def _fallback_upscale(self, image: np.ndarray, plan: RepairPlan) -> np.ndarray:
        h, w = image.shape[:2]
        selected = self.options.upscale
        auto_2x = selected == 'Original size' and (plan.inspection.is_screenshot or min(h, w) < 720)
        if selected in {'2× AI Upscale', '2× upscale'} or auto_2x:
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
        original_faces = self.identity.detect(original)
        intelligence_report = self.intelligence.inspect(
            original, scene=analysis.scene, quality_target=self.options.quality_target
        )

        processed, plan, validation = self.engine.process(original, mode)

        wants_sr = self.options.upscale in {
            '2× AI Upscale', '2× upscale', '4K AI Upscale', '4K long edge'
        } or plan.inspection.is_screenshot or min(processed.shape[:2]) < 720
        processed, capability_report = self.capabilities.apply(
            processed,
            plan.inspection,
            plan,
            allow_super_resolution=wants_sr,
        )

        processed, profile_report = self.profiles.apply(
            processed,
            plan.inspection.image_type,
            analysis.scene,
        )
        for name, message in zip(profile_report.selected, profile_report.messages):
            capability_report.steps.append(CapabilityStep(
                capability=f'profile:{name}',
                model_path=None,
                available=True,
                applied=True,
                message=message,
            ))

        current_faces = self.identity.detect(processed)
        if self.options.reduce_flare and not plan.inspection.is_monochrome:
            processed = self._safe_flare_cleanup(processed, current_faces)

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

        post_validation = self.intelligence.validate(original, processed, intelligence_report)
        if not post_validation.accepted and processed.shape == original.shape:
            processed = original.copy()

        if 'super_resolution' not in capability_report.applied:
            processed = self._fallback_upscale(processed, plan)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(destination), processed,
            [cv2.IMWRITE_JPEG_QUALITY, int(self.options.jpeg_quality)],
        ):
            raise OSError(f'Could not write {destination}')

        review = bool(analysis.review_reason)
        if not validation.accepted or not post_validation.accepted:
            review = True
        if plan.inspection.quality_score < 38 or intelligence_report.quality_score < 38:
            review = True
        if self.options.identity_lock and faces_protected and minimum_similarity < 0.72:
            review = True
        if any(step.available and not step.applied for step in capability_report.steps):
            review = True
        if intelligence_report.warnings:
            review = review or self.options.quality_target in {'Archive', 'Museum'}

        return ProcessResult(
            review_needed=review,
            analysis=analysis,
            faces_protected=faces_protected,
            repair_plan=plan,
            validation=validation,
            capability_report=capability_report,
            intelligence_report=intelligence_report,
            post_validation=post_validation,
        )
