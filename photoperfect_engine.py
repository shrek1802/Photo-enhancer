from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Inspection:
    image_type: str
    quality_score: int
    is_screenshot: bool
    is_monochrome: bool
    is_low_resolution: bool
    face_count: int
    smallest_face_ratio: float
    blur_score: float
    noise_score: float
    compression_score: float
    dark_fraction: float
    highlight_fraction: float
    contrast: float
    problems: list[str] = field(default_factory=list)


@dataclass
class RepairPlan:
    name: str
    stages: list[str]
    requested_mode: str
    confidence: int
    inspection: Inspection


@dataclass
class Validation:
    before_score: float
    after_score: float
    accepted: bool
    improvement: float


class PhotoPerfectEngine:
    """Deterministic Auto Detect engine used before specialist ONNX models.

    It classifies the input, builds a repair plan, applies safe photographic
    corrections and validates that the result is not materially worse. It does
    not generate or replace faces.
    """

    def __init__(self) -> None:
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    @staticmethod
    def _quality(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharp = min(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) * 8.0, 40.0)
        contrast = min(float(gray.std()) / 55.0, 1.0) * 23.0
        clipped = float(np.mean(gray >= 250)) * 95.0
        crushed = float(np.mean(gray <= 8)) * 80.0
        residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        noise = max(float(residual.std()) - 8.0, 0.0) * 0.9
        return float(np.clip(35.0 + sharp + contrast - clipped - crushed - noise, 0, 100))

    @staticmethod
    def _compression_score(gray: np.ndarray) -> float:
        if min(gray.shape) < 24:
            return 0.0
        vertical = np.abs(np.diff(gray.astype(np.float32), axis=1))
        horizontal = np.abs(np.diff(gray.astype(np.float32), axis=0))
        v_bound = float(vertical[:, 7::8].mean()) if vertical.shape[1] > 8 else 0.0
        h_bound = float(horizontal[7::8, :].mean()) if horizontal.shape[0] > 8 else 0.0
        v_all = float(vertical.mean()) + 1e-6
        h_all = float(horizontal.mean()) + 1e-6
        ratio = ((v_bound / v_all) + (h_bound / h_all)) / 2.0
        return float(np.clip((ratio - 0.9) * 100.0, 0, 100))

    @staticmethod
    def _screenshot_likelihood(image: np.ndarray) -> bool:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        portrait_phone = h / max(w, 1) > 1.55
        top = gray[: max(28, int(h * 0.10))]
        bottom = gray[int(h * 0.84):]
        edge_density = float(cv2.Canny(top, 70, 150).mean())
        flat_bottom = float(bottom.std()) < 48
        light_or_dark_bar = float(bottom.mean()) > 145 or float(bottom.mean()) < 70
        return bool(portrait_phone and edge_density > 5.5 and flat_bottom and light_or_dark_bar)

    def inspect(self, image: np.ndarray) -> Inspection:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        is_monochrome = float(np.percentile(saturation, 90)) < 18
        is_screenshot = self._screenshot_likelihood(image)
        is_low_resolution = min(h, w) < 1000 or h * w < 1_500_000
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        noise = float(residual.std())
        compression = self._compression_score(gray)
        dark = float(np.mean(gray < 35))
        highlights = float(np.mean(gray > 248))
        contrast = float(gray.std())
        minimum = max(28, min(h, w) // 20)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(minimum, minimum))
        face_ratios = [(fw * fh) / float(h * w) for _, _, fw, fh in faces]
        smallest_face = min(face_ratios) if face_ratios else 0.0

        problems: list[str] = []
        if is_screenshot:
            problems.append('social-media or phone screenshot interface')
        if is_low_resolution:
            problems.append('low resolution')
        if compression > 16:
            problems.append('JPEG compression artefacts')
        if blur < 80:
            problems.append('soft focus or blur')
        if noise > 8.5:
            problems.append('visible noise')
        if dark > 0.18:
            problems.append('deep shadows')
        if highlights > 0.07:
            problems.append('clipped highlights')
        if contrast < 34:
            problems.append('low contrast')
        if faces and smallest_face < 0.012:
            problems.append('small face detail')
        if is_monochrome:
            problems.append('black and white / monochrome image')

        if is_screenshot:
            image_type = 'Screenshot Recovery'
        elif is_monochrome:
            image_type = 'Black & White Restore'
        elif len(faces) > 0:
            image_type = 'Portrait / People'
        elif dark > 0.18:
            image_type = 'Low Light'
        else:
            image_type = 'General Photograph'

        return Inspection(
            image_type=image_type,
            quality_score=int(round(self._quality(image))),
            is_screenshot=is_screenshot,
            is_monochrome=is_monochrome,
            is_low_resolution=is_low_resolution,
            face_count=len(faces),
            smallest_face_ratio=smallest_face,
            blur_score=blur,
            noise_score=noise,
            compression_score=compression,
            dark_fraction=dark,
            highlight_fraction=highlights,
            contrast=contrast,
            problems=problems,
        )

    def plan(self, inspection: Inspection, requested_mode: str) -> RepairPlan:
        mode = requested_mode or 'Auto Detect'
        if mode == 'Auto Detect':
            if inspection.is_screenshot:
                name = 'Screenshot Recovery'
            elif inspection.is_monochrome:
                name = 'Black & White Restore'
            elif inspection.quality_score >= 78:
                name = 'Professional Light Polish'
            elif inspection.face_count:
                name = 'Identity-Safe Portrait Recovery'
            else:
                name = 'Automatic Photo Recovery'
        else:
            name = mode

        stages: list[str] = []
        if inspection.is_screenshot:
            stages += ['crop screenshot interface', 'repair JPEG compression']
        elif inspection.compression_score > 16:
            stages.append('repair JPEG compression')
        if inspection.noise_score > 8.5:
            stages.append('adaptive denoise')
        if inspection.blur_score < 140:
            stages.append('edge-limited detail recovery')
        if inspection.is_monochrome:
            stages += ['restore monochrome contrast', 'recover local tonal detail']
        if inspection.dark_fraction > 0.10:
            stages.append('recover shadows')
        if inspection.highlight_fraction > 0.04:
            stages.append('compress highlights')
        if inspection.face_count:
            stages.append('identity-safe face lighting')
        stages += ['professional colour and lighting', 'quality validation']
        confidence = int(np.clip(60 + len(inspection.problems) * 5, 60, 96))
        return RepairPlan(name, stages, mode, confidence, inspection)

    @staticmethod
    def _crop_screenshot(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        # Conservative crop removes common top status/header and bottom action bars.
        top = int(h * 0.055)
        bottom = int(h * 0.84)
        if bottom - top < h * 0.65:
            return image
        return image[top:bottom, 0:w]

    @staticmethod
    def _repair_compression(image: np.ndarray, strong: bool) -> np.ndarray:
        diameter, sigma = (7, 30) if strong else (5, 22)
        cleaned = cv2.bilateralFilter(image, diameter, sigma, sigma)
        # Recover edges after deblocking without restoring block boundaries.
        soft = cv2.GaussianBlur(cleaned, (0, 0), 0.9)
        return cv2.addWeighted(cleaned, 1.22, soft, -0.22, 0)

    @staticmethod
    def _restore_monochrome(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 4, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=1.75, tileGridSize=(8, 8)).apply(denoised)
        # Blend global and local tone to avoid the harsh processed result seen before.
        toned = cv2.addWeighted(denoised, 0.28, clahe, 0.72, 0)
        detail = cv2.addWeighted(toned, 1.34, cv2.GaussianBlur(toned, (0, 0), 1.15), -0.34, 0)
        return cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _recover_lighting(image: np.ndarray, inspection: Inspection) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        local = cv2.createCLAHE(clipLimit=1.45, tileGridSize=(8, 8)).apply(l)
        if inspection.dark_fraction > 0.10:
            gamma = 0.88
            lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
            lifted = cv2.LUT(local, lut)
            shadow = cv2.GaussianBlur((255 - l).astype(np.uint8), (0, 0), 17).astype(np.float32) / 255.0
            local = np.clip(local * (1 - shadow * 0.48) + lifted * shadow * 0.48, 0, 255).astype(np.uint8)
        if inspection.highlight_fraction > 0.04:
            high = np.clip((local.astype(np.float32) - 188) / 67, 0, 1)
            local = np.clip(local.astype(np.float32) - high * 16, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([local, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _professional_finish(image: np.ndarray, light: bool) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean = np.array(cv2.mean(image)[:3], dtype=np.float32)
        target = mean.mean()
        shift = 0.07 if light else 0.13
        scales = np.clip(target / np.maximum(mean, 1.0), 1 - shift, 1 + shift)
        balanced = np.clip(image.astype(np.float32) * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * (1.035 if light else 1.075), 0, 255).astype(np.uint8)
        finished = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(image, 0.18 if not light else 0.34, finished, 0.82 if not light else 0.66, 0)

    def execute(self, image: np.ndarray, plan: RepairPlan) -> tuple[np.ndarray, Validation]:
        before = self._quality(image)
        working = image.copy()
        inspection = plan.inspection
        light = plan.name == 'Professional Light Polish'

        if inspection.is_screenshot:
            working = self._crop_screenshot(working)
        if inspection.compression_score > 10 or inspection.is_screenshot:
            working = self._repair_compression(working, strong=inspection.is_screenshot)
        if inspection.is_monochrome:
            working = self._restore_monochrome(working)
        else:
            if inspection.noise_score > 8.5 and not light:
                working = cv2.fastNlMeansDenoisingColored(working, None, 4, 4, 7, 21)
            if inspection.blur_score < 140 and not light:
                base = cv2.GaussianBlur(working, (0, 0), 1.35)
                working = cv2.addWeighted(working, 1.48, base, -0.48, 0)
            working = self._recover_lighting(working, inspection)
            working = self._professional_finish(working, light=light)

        after = self._quality(working)
        # The generic score can undervalue deliberate screenshot cropping; permit a small
        # score drop there, but reject clear degradation for ordinary photographs.
        tolerance = 4.0 if inspection.is_screenshot else 1.5
        accepted = after + tolerance >= before
        if not accepted:
            working = image.copy()
            after = before
        return working, Validation(before, after, accepted, after - before)

    def process(self, image: np.ndarray, requested_mode: str) -> tuple[np.ndarray, RepairPlan, Validation]:
        inspection = self.inspect(image)
        plan = self.plan(inspection, requested_mode)
        result, validation = self.execute(image, plan)
        return result, plan, validation
