from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ai_engine import NeuralEngine
from auto_ai_pipeline import AutomaticAIPipeline, PipelineReport
from identity_guard import FaceIdentityGuard
from photo_analysis import PhotoAnalysis, PhotoAnalyser

SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED


@dataclass
class EnhanceOptions:
    preset: str = 'Smart Auto'
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


class PhotoEnhancer:
    GOOD_PHOTO_SCORE = 78

    def __init__(self, options: EnhanceOptions):
        self.options = options
        self.analyser = PhotoAnalyser()
        self.neural = NeuralEngine(enabled=options.neural_ai)
        self.automatic = AutomaticAIPipeline()
        self.identity = FaceIdentityGuard()
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    @property
    def engine_message(self) -> str:
        identity = 'Identity Lock ON' if self.options.identity_lock else 'Identity Lock OFF'
        return f'{self.neural.status.message}; restoration: {self.automatic.provider}; {identity}'

    def _read(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        if self.options.auto_rotate:
            image = ImageOps.exif_transpose(image)
        return cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _white_balance(image: np.ndarray, maximum_shift: float = 0.12) -> np.ndarray:
        work = image.astype(np.float32)
        means = np.array(cv2.mean(work)[:3], dtype=np.float32)
        target = float(means.mean())
        scales = target / np.maximum(means, 1.0)
        scales = np.clip(scales, 1.0 - maximum_shift, 1.0 + maximum_shift)
        return np.clip(work * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)

    def _effective_strength(self, analysis: PhotoAnalysis) -> str:
        if self.options.good_photo_polish and analysis.quality_score >= self.GOOD_PHOTO_SCORE:
            return 'polish'
        if self.options.preset != 'Smart Auto':
            return self.options.strength
        if analysis.quality_score < 48:
            return 'maximum'
        if analysis.quality_score < 72:
            return 'strong'
        return 'natural'

    def _tone(self, image: np.ndarray, strength: str) -> np.ndarray:
        settings = {
            'polish': (1.12, 0.985, 0.16, 5),
            'natural': (1.35, 0.95, 0.32, 9),
            'strong': (1.75, 0.89, 0.48, 17),
            'maximum': (2.10, 0.83, 0.62, 25),
        }
        clip, gamma, shadow_amount, highlight_amount = settings[strength]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)

        if self.options.lift_shadows:
            lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype('uint8')
            lifted = cv2.LUT(l2, lut)
            mask = cv2.GaussianBlur((255 - l).astype(np.uint8), (0, 0), 15) / 255.0
            l2 = np.clip(
                l2 * (1 - mask * shadow_amount) + lifted * mask * shadow_amount, 0, 255
            ).astype(np.uint8)

        if self.options.recover_highlights:
            high = np.clip((l2.astype(np.float32) - 190) / 65, 0, 1)
            l2 = np.clip(l2.astype(np.float32) - high * highlight_amount, 0, 255).astype(np.uint8)

        result = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        if strength == 'polish':
            # A good photo should pop, not look processed.
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s.astype(np.float32) * 1.035, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        return result

    def _preset_grade(self, image: np.ndarray, strength: str) -> np.ndarray:
        if strength == 'polish':
            return image
        if self.options.preset == 'Event / Christening':
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            a = np.clip(a.astype(np.int16) + 1, 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.int16) + 2, 0, 255).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        if self.options.preset == 'Old Photo Restoration':
            return cv2.bilateralFilter(cv2.detailEnhance(image, sigma_s=8, sigma_r=0.12), 7, 25, 25)
        if self.options.preset == 'Landscape':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s.astype(np.float32) * 1.06, 0, 255).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        return image

    def _reduce_flare(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)
        mask = ((((value > 240) & (saturation < 65)) |
                 ((value > 228) & (saturation > 110))).astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        safe = np.zeros_like(mask)
        total = image.shape[0] * image.shape[1]
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if 12 <= area <= total * 0.009:
                safe[labels == label] = 255
        if safe.mean() < 0.15:
            return image
        safe = cv2.dilate(safe, np.ones((5, 5), np.uint8), iterations=1)
        return cv2.inpaint(image, safe, 3, cv2.INPAINT_TELEA)

    def _face_light_only(self, image: np.ndarray, strength: str) -> np.ndarray:
        """Relight faces without smoothing, redrawing or changing facial geometry."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        minimum = max(40, min(image.shape[:2]) // 14)
        faces = self.face_detector.detectMultiScale(gray, 1.12, 5, minSize=(minimum, minimum))
        result = image.copy()
        targets = {'polish': 116, 'natural': 120, 'strong': 123, 'maximum': 125}
        for x, y, w, h in faces:
            pad = int(w * 0.12)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
            roi = result[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_l = float(l.mean())
            target = targets[strength]
            if mean_l < target:
                gain = min(1.10 if strength == 'polish' else 1.15, target / max(mean_l, 1))
                new_l = np.clip(l.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                corrected = cv2.cvtColor(cv2.merge([new_l, a, b]), cv2.COLOR_LAB2BGR)
                result[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.45, corrected, 0.55, 0)
        return result

    def _denoise(self, image: np.ndarray, strength: str, analysis: PhotoAnalysis) -> np.ndarray:
        if strength == 'polish' or (analysis.noise_score < 4.2 and strength == 'natural'):
            return image
        amount = {'natural': 3, 'strong': 5, 'maximum': 7}[strength]
        return cv2.fastNlMeansDenoisingColored(image, None, amount, amount, 7, 21)

    def _sharpen(self, image: np.ndarray, analysis: PhotoAnalysis, strength: str) -> np.ndarray:
        score = analysis.blur_score
        if strength == 'polish':
            amount = 0.12
        else:
            amount = 0.16 if score > 650 else 0.32 if score > 250 else 0.52 if score > 90 else 0.72
            amount *= {'natural': 0.82, 'strong': 1.0, 'maximum': 1.12}[strength]
        blurred = cv2.GaussianBlur(image, (0, 0), 1.15)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge = np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_32F)) / 30.0, 0, 1)
        edge = cv2.GaussianBlur(edge, (0, 0), 1.15)[..., None]
        return np.clip(image * (1 - edge) + sharpened * edge, 0, 255).astype(np.uint8)

    @staticmethod
    def _straighten(image: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 0.45 or abs(angle) > 8:
            return image
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        crop = int(min(width, height) * min(abs(angle) / 90, 0.035))
        return rotated[crop:height-crop, crop:width-crop] if crop > 0 else rotated

    def _upscale(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self.options.upscale == '2× upscale':
            return self.neural.upscale(image)
        if self.options.upscale == '4K long edge' and max(h, w) < 3840:
            if self.neural.status.model_loaded and max(h, w) * self.neural.scale <= 4608:
                image = self.neural.upscale(image)
                h, w = image.shape[:2]
            if max(h, w) < 3840:
                scale = 3840 / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        return image

    def process(self, source: Path, destination: Path) -> ProcessResult:
        analysis = self.analyser.analyse(source)
        strength = self._effective_strength(analysis)
        image = self._read(source)
        pipeline_report: PipelineReport | None = None
        faces_in_original = self.identity.detect(image)
        good_photo = strength == 'polish'

        # Identity Lock disables generative face restoration and risky automatic crops.
        removed_face_session = None
        if self.options.identity_lock:
            removed_face_session = self.automatic.sessions.pop('face_restore', None)

        try:
            if self.options.automatic_restoration and not good_photo:
                image, pipeline_report = self.automatic.restore(
                    image,
                    allow_ui_crop=self.options.remove_screenshot_ui and not bool(faces_in_original),
                )
        finally:
            if removed_face_session is not None:
                self.automatic.sessions['face_restore'] = removed_face_session

        identity_reference = image.copy()

        # Avoid geometric face movement. Horizon straightening is safe for scenes,
        # but is skipped automatically when faces are present and Identity Lock is on.
        if self.options.straighten_horizon and not (self.options.identity_lock and faces_in_original):
            image = self._straighten(image, analysis.horizon_angle)
            identity_reference = self._straighten(identity_reference, analysis.horizon_angle)

        balance_shift = 0.07 if good_photo else 0.12
        image = self._tone(self._white_balance(image, balance_shift), strength)
        image = self._preset_grade(image, strength)

        if self.options.reduce_flare and not good_photo:
            image = self._reduce_flare(image)
        if self.options.face_aware:
            image = self._face_light_only(image, strength)
        if self.options.denoise and not self.options.automatic_restoration:
            image = self._denoise(image, strength, analysis)
        if self.options.sharpen:
            image = self._sharpen(image, analysis, strength)

        faces_protected = 0
        identity_similarity = 1.0
        if self.options.identity_lock and identity_reference.shape == image.shape:
            locked = self.identity.apply(identity_reference, image)
            image = locked.image
            faces_protected = locked.faces_protected
            identity_similarity = locked.minimum_similarity

        # Good photographs receive only a restrained professional polish.
        if good_photo and identity_reference.shape == image.shape:
            image = cv2.addWeighted(identity_reference, 0.38, image, 0.62, 0)

        image = self._upscale(image)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, self.options.jpeg_quality]):
            raise OSError(f'Could not write {destination}')

        review = bool(analysis.review_reason)
        if pipeline_report and pipeline_report.confidence < 45:
            review = True
        if self.options.identity_lock and faces_protected and identity_similarity < 0.72:
            review = True

        return ProcessResult(review, analysis, pipeline_report, faces_protected)
