from __future__ import annotations

from dataclasses import dataclass, field

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
    text_edge_score: float = 0.0
    social_ui_score: float = 0.0
    problems: list[str] = field(default_factory=list)


@dataclass
class RepairPlan:
    name: str
    stages: list[str]
    requested_mode: str
    confidence: int
    inspection: Inspection
    strategy: str = 'balanced'


@dataclass
class Validation:
    before_score: float
    after_score: float
    accepted: bool
    improvement: float
    attempts: int = 1
    selected_strategy: str = 'balanced'
    reasons: list[str] = field(default_factory=list)


class PhotoPerfectEngine:
    """Phase 2 adaptive engine.

    It inspects, classifies, builds a dynamic repair plan, tries several safe
    strategies, and keeps the highest-scoring result. It never generates faces.
    """

    def __init__(self) -> None:
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    @staticmethod
    def _metrics(image: np.ndarray) -> dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        return {
            'sharpness': float(lap),
            'contrast': float(gray.std()),
            'noise': float(residual.std()),
            'dark': float(np.mean(gray <= 8)),
            'clipped': float(np.mean(gray >= 250)),
        }

    @classmethod
    def _quality(cls, image: np.ndarray) -> float:
        m = cls._metrics(image)
        sharp = min(np.log1p(m['sharpness']) * 8.2, 41.0)
        contrast = min(m['contrast'] / 55.0, 1.0) * 23.0
        clipped = m['clipped'] * 95.0
        crushed = m['dark'] * 80.0
        noise = max(m['noise'] - 8.0, 0.0) * 0.9
        return float(np.clip(35.0 + sharp + contrast - clipped - crushed - noise, 0, 100))

    @staticmethod
    def _compression_score(gray: np.ndarray) -> float:
        if min(gray.shape) < 24:
            return 0.0
        vertical = np.abs(np.diff(gray.astype(np.float32), axis=1))
        horizontal = np.abs(np.diff(gray.astype(np.float32), axis=0))
        vb = float(vertical[:, 7::8].mean()) if vertical.shape[1] > 8 else 0.0
        hb = float(horizontal[7::8, :].mean()) if horizontal.shape[0] > 8 else 0.0
        va = float(vertical.mean()) + 1e-6
        ha = float(horizontal.mean()) + 1e-6
        return float(np.clip((((vb / va) + (hb / ha)) / 2.0 - 0.88) * 120.0, 0, 100))

    @staticmethod
    def _ui_scores(image: np.ndarray) -> tuple[float, float]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        top_h = max(30, int(h * 0.12))
        bottom_y = int(h * 0.80)
        top_gray = gray[:top_h]
        bottom_gray = gray[bottom_y:]
        middle_gray = gray[top_h:bottom_y]
        top_edges = edges[:top_h]
        bottom_edges = edges[bottom_y:]

        text_edge = float(np.mean(edges > 0) * 100.0)
        top_density = float(np.mean(top_edges > 0) * 100.0)
        bottom_density = float(np.mean(bottom_edges > 0) * 100.0)
        flat_top = max(0.0, 20.0 - float(top_gray.std()))
        flat_bottom = max(0.0, 20.0 - float(bottom_gray.std()))
        middle_mean = float(middle_gray.mean()) if middle_gray.size else float(gray.mean())
        band_contrast = min(
            abs(float(top_gray.mean()) - middle_mean) +
            abs(float(bottom_gray.mean()) - middle_mean),
            80.0,
        ) / 4.0
        portrait_bonus = 16.0 if h / max(w, 1) > 1.5 else 0.0
        social = np.clip(
            top_density * 1.8 + bottom_density * 1.5 +
            flat_top * 0.35 + flat_bottom * 0.55 +
            band_contrast + portrait_bonus,
            0,
            100,
        )
        return text_edge, float(social)

    def inspect(self, image: np.ndarray) -> Inspection:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        is_monochrome = float(np.percentile(hsv[:, :, 1], 90)) < 18
        text_edge, social_ui = self._ui_scores(image)
        aspect = h / max(w, 1)
        is_screenshot = (
            social_ui >= 34
            or (aspect > 1.5 and social_ui >= 26)
            or (aspect > 1.75 and social_ui >= 22 and text_edge >= 1.0)
        )
        is_low_resolution = min(h, w) < 1000 or h * w < 1_500_000
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        noise = float(residual.std())
        compression = self._compression_score(gray)
        dark = float(np.mean(gray < 35))
        highlights = float(np.mean(gray > 248))
        contrast = float(gray.std())
        minimum = max(28, min(h, w) // 22)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(minimum, minimum))
        ratios = [(fw * fh) / float(h * w) for _, _, fw, fh in faces]
        smallest_face = min(ratios) if ratios else 0.0

        problems: list[str] = []
        if is_screenshot:
            problems.append('social-media or phone screenshot interface')
        if is_low_resolution:
            problems.append('low resolution')
        if compression > 12:
            problems.append('JPEG compression artefacts')
        if blur < 120:
            problems.append('soft focus or blur')
        if noise > 8.5:
            problems.append('visible noise')
        if dark > 0.14:
            problems.append('deep shadows')
        if highlights > 0.05:
            problems.append('clipped highlights')
        if contrast < 36:
            problems.append('low contrast')
        if faces and smallest_face < 0.015:
            problems.append('small face detail')
        if is_monochrome:
            problems.append('black and white / monochrome image')

        if is_screenshot:
            image_type = 'Screenshot Recovery'
        elif is_monochrome:
            image_type = 'Black & White Restore'
        elif len(faces) > 0:
            image_type = 'Portrait / People'
        elif dark > 0.16:
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
            text_edge_score=text_edge,
            social_ui_score=social_ui,
            problems=problems,
        )

    @staticmethod
    def _read_exif_hint(_image: np.ndarray) -> bool:
        return False

    def plan(self, inspection: Inspection, requested_mode: str) -> RepairPlan:
        mode = requested_mode or 'Auto Detect'
        if mode == 'Auto Detect':
            if inspection.is_screenshot:
                name = 'Screenshot Recovery'
            elif inspection.is_monochrome:
                name = 'Black & White Restore'
            elif inspection.quality_score >= 80:
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
        elif inspection.compression_score > 12:
            stages.append('repair JPEG compression')
        if inspection.noise_score > 8.5:
            stages.append('adaptive denoise')
        if inspection.blur_score < 160:
            stages.append('edge-limited detail recovery')
        if inspection.is_monochrome:
            stages += ['restore monochrome contrast', 'recover local tonal detail']
        if inspection.dark_fraction > 0.08:
            stages.append('recover shadows')
        if inspection.highlight_fraction > 0.03:
            stages.append('compress highlights')
        if inspection.face_count:
            stages.append('identity-safe face lighting')
        stages += ['professional colour and lighting', 'quality validation with retry']
        confidence = int(np.clip(62 + len(inspection.problems) * 5, 62, 97))
        return RepairPlan(name, stages, mode, confidence, inspection)

    @staticmethod
    def _crop_screenshot(image: np.ndarray, strategy: str) -> np.ndarray:
        h, w = image.shape[:2]
        if strategy == 'strong':
            top, bottom = int(h * 0.075), int(h * 0.82)
        else:
            top, bottom = int(h * 0.05), int(h * 0.86)
        return image[top:bottom, 0:w] if bottom - top >= h * 0.64 else image

    @staticmethod
    def _repair_compression(image: np.ndarray, strategy: str) -> np.ndarray:
        settings = {'gentle': (5, 18, 0.14), 'balanced': (7, 26, 0.22), 'strong': (9, 34, 0.28)}
        diameter, sigma, amount = settings[strategy]
        cleaned = cv2.bilateralFilter(image, diameter, sigma, sigma)
        soft = cv2.GaussianBlur(cleaned, (0, 0), 0.9)
        return cv2.addWeighted(cleaned, 1 + amount, soft, -amount, 0)

    @staticmethod
    def _restore_monochrome(image: np.ndarray, strategy: str) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h = {'gentle': 2, 'balanced': 4, 'strong': 6}[strategy]
        clip = {'gentle': 1.25, 'balanced': 1.65, 'strong': 2.0}[strategy]
        detail_amount = {'gentle': 0.18, 'balanced': 0.30, 'strong': 0.42}[strategy]
        denoised = cv2.fastNlMeansDenoising(gray, None, h, 7, 21)
        local = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(denoised)
        toned = cv2.addWeighted(denoised, 0.30, local, 0.70, 0)
        blurred = cv2.GaussianBlur(toned, (0, 0), 1.1)
        detail = cv2.addWeighted(toned, 1 + detail_amount, blurred, -detail_amount, 0)
        return cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _recover_lighting(image: np.ndarray, inspection: Inspection, strategy: str) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = {'gentle': 1.18, 'balanced': 1.42, 'strong': 1.70}[strategy]
        local = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        if inspection.dark_fraction > 0.08:
            gamma = {'gentle': 0.95, 'balanced': 0.89, 'strong': 0.83}[strategy]
            amount = {'gentle': 0.22, 'balanced': 0.42, 'strong': 0.58}[strategy]
            lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
            lifted = cv2.LUT(local, lut)
            shadow = cv2.GaussianBlur((255 - l).astype(np.uint8), (0, 0), 17).astype(np.float32) / 255.0
            local = np.clip(local * (1 - shadow * amount) + lifted * shadow * amount, 0, 255).astype(np.uint8)
        if inspection.highlight_fraction > 0.03:
            amount = {'gentle': 8, 'balanced': 15, 'strong': 22}[strategy]
            high = np.clip((local.astype(np.float32) - 188) / 67, 0, 1)
            local = np.clip(local.astype(np.float32) - high * amount, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([local, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _professional_finish(image: np.ndarray, strategy: str, monochrome: bool) -> np.ndarray:
        if monochrome:
            return image
        mean = np.array(cv2.mean(image)[:3], dtype=np.float32)
        target = mean.mean()
        shift = {'gentle': 0.06, 'balanced': 0.10, 'strong': 0.14}[strategy]
        scales = np.clip(target / np.maximum(mean, 1.0), 1 - shift, 1 + shift)
        balanced = np.clip(image.astype(np.float32) * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        saturation = {'gentle': 1.035, 'balanced': 1.065, 'strong': 1.09}[strategy]
        s = np.clip(s.astype(np.float32) * saturation, 0, 255).astype(np.uint8)
        finished = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
        blend = {'gentle': 0.62, 'balanced': 0.78, 'strong': 0.88}[strategy]
        return cv2.addWeighted(image, 1 - blend, finished, blend, 0)

    def _execute_strategy(self, image: np.ndarray, plan: RepairPlan, strategy: str) -> np.ndarray:
        inspection = plan.inspection
        working = image.copy()
        if inspection.is_screenshot:
            working = self._crop_screenshot(working, strategy)
        if inspection.compression_score > 8 or inspection.is_screenshot:
            working = self._repair_compression(working, strategy)
        if inspection.is_monochrome:
            working = self._restore_monochrome(working, strategy)
        else:
            if inspection.noise_score > 8.5 and strategy != 'gentle':
                h = 3 if strategy == 'balanced' else 5
                working = cv2.fastNlMeansDenoisingColored(working, None, h, h, 7, 21)
            if inspection.blur_score < 180:
                amount = {'gentle': 0.22, 'balanced': 0.42, 'strong': 0.60}[strategy]
                base = cv2.GaussianBlur(working, (0, 0), 1.3)
                working = cv2.addWeighted(working, 1 + amount, base, -amount, 0)
            working = self._recover_lighting(working, inspection, strategy)
            working = self._professional_finish(working, strategy, monochrome=False)
        return working

    def execute(self, image: np.ndarray, plan: RepairPlan) -> tuple[np.ndarray, Validation]:
        before = self._quality(image)
        strategies = ['gentle', 'balanced', 'strong']
        if plan.name == 'Professional Light Polish':
            strategies = ['gentle', 'balanced']
        candidates: list[tuple[float, str, np.ndarray]] = []
        for strategy in strategies:
            candidate = self._execute_strategy(image, plan, strategy)
            score = self._quality(candidate)
            candidates.append((score, strategy, candidate))
        after, selected, result = max(candidates, key=lambda item: item[0])
        tolerance = 4.0 if plan.inspection.is_screenshot else 1.5
        accepted = after + tolerance >= before
        reasons = [f'{strategy}: {score:.1f}' for score, strategy, _ in candidates]
        if not accepted:
            result = image.copy()
            after = before
            reasons.append('All candidate pipelines were rejected; original retained')
        plan.strategy = selected
        return result, Validation(
            before_score=before,
            after_score=after,
            accepted=accepted,
            improvement=after - before,
            attempts=len(candidates),
            selected_strategy=selected,
            reasons=reasons,
        )

    def process(self, image: np.ndarray, requested_mode: str) -> tuple[np.ndarray, RepairPlan, Validation]:
        inspection = self.inspect(image)
        plan = self.plan(inspection, requested_mode)
        result, validation = self.execute(image, plan)
        return result, plan, validation
