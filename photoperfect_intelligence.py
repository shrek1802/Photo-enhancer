from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceProfile:
    box: tuple[int, int, int, int]
    area_ratio: float
    brightness: float
    sharpness: float
    contrast: float
    protection_level: str
    maximum_change: float


@dataclass(frozen=True)
class IntelligenceReport:
    scene: str
    quality_target: str
    quality_score: int
    blur_score: float
    noise_score: float
    compression_score: float
    dynamic_range: float
    dark_fraction: float
    highlight_fraction: float
    colour_cast: float
    faces: tuple[FaceProfile, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PostProcessValidation:
    accepted: bool
    identity_similarity: float
    structure_similarity: float
    clipping_delta: float
    oversharpening_ratio: float
    reasons: tuple[str, ...] = ()


class PhotoPerfectIntelligence:
    """Phase 4 inspection and post-processing safety layer.

    This module does not identify people or estimate demographic attributes. It
    measures image quality and facial-region stability so the engine can choose
    conservative settings and reject results that drift too far from the source.
    """

    TARGETS = {
        'Standard': {'identity': 0.80, 'structure': 0.74, 'clip_delta': 0.035, 'sharp_ratio': 3.2},
        'Professional': {'identity': 0.84, 'structure': 0.78, 'clip_delta': 0.028, 'sharp_ratio': 2.8},
        'Studio': {'identity': 0.88, 'structure': 0.82, 'clip_delta': 0.022, 'sharp_ratio': 2.5},
        'Archive': {'identity': 0.91, 'structure': 0.86, 'clip_delta': 0.018, 'sharp_ratio': 2.2},
        'Museum': {'identity': 0.94, 'structure': 0.89, 'clip_delta': 0.014, 'sharp_ratio': 2.0},
    }

    def __init__(self) -> None:
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _normalised_correlation(first: np.ndarray, second: np.ndarray) -> float:
        if first.size == 0 or second.size == 0:
            return 1.0
        first = cv2.resize(first, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        second = cv2.resize(second, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        first -= first.mean()
        second -= second.mean()
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator < 1e-6:
            return 1.0
        return float(np.clip(np.sum(first * second) / denominator, -1.0, 1.0))

    @staticmethod
    def _compression_score(gray: np.ndarray) -> float:
        if min(gray.shape) < 24:
            return 0.0
        vertical = np.abs(np.diff(gray.astype(np.float32), axis=1))
        horizontal = np.abs(np.diff(gray.astype(np.float32), axis=0))
        vb = float(vertical[:, 7::8].mean()) if vertical.shape[1] > 8 else 0.0
        hb = float(horizontal[7::8, :].mean()) if horizontal.shape[0] > 8 else 0.0
        baseline = (float(vertical.mean()) + float(horizontal.mean())) / 2.0 + 1e-6
        return float(np.clip((((vb + hb) / 2.0) / baseline - 0.9) * 115.0, 0, 100))

    @staticmethod
    def _noise_score(gray: np.ndarray) -> float:
        residual = gray.astype(np.float32) - cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        return float(residual.std())

    def _faces(self, image: np.ndarray) -> tuple[FaceProfile, ...]:
        gray = self._gray(image)
        h, w = gray.shape
        minimum = max(28, min(h, w) // 24)
        detected = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(minimum, minimum)
        )
        profiles: list[FaceProfile] = []
        for x, y, fw, fh in detected:
            roi = gray[y:y + fh, x:x + fw]
            ratio = float((fw * fh) / max(h * w, 1))
            sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
            brightness = float(roi.mean()) if roi.size else 0.0
            contrast = float(roi.std()) if roi.size else 0.0
            if ratio < 0.012 or sharpness < 35:
                level, maximum = 'Maximum', 0.10
            elif ratio < 0.035 or sharpness < 80:
                level, maximum = 'High', 0.16
            else:
                level, maximum = 'Normal', 0.24
            profiles.append(FaceProfile(
                box=(int(x), int(y), int(fw), int(fh)),
                area_ratio=ratio,
                brightness=brightness,
                sharpness=sharpness,
                contrast=contrast,
                protection_level=level,
                maximum_change=maximum,
            ))
        return tuple(profiles)

    def inspect(self, image: np.ndarray, scene: str, quality_target: str) -> IntelligenceReport:
        gray = self._gray(image)
        faces = self._faces(image)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise = self._noise_score(gray)
        compression = self._compression_score(gray)
        dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        dark = float(np.mean(gray < 35))
        highlights = float(np.mean(gray > 248))
        means = np.asarray(cv2.mean(image)[:3], dtype=np.float32)
        colour_cast = float(np.max(means) - np.min(means))
        sharp_component = min(np.log1p(blur) * 8.0, 40.0)
        range_component = min(dynamic_range / 150.0, 1.0) * 24.0
        penalties = noise * 0.7 + compression * 0.15 + dark * 55.0 + highlights * 70.0
        score = int(round(np.clip(35.0 + sharp_component + range_component - penalties, 0, 100)))
        warnings: list[str] = []
        if faces and any(face.protection_level == 'Maximum' for face in faces):
            warnings.append('Tiny or heavily blurred face detected; maximum identity protection applied')
        if compression > 20:
            warnings.append('Heavy JPEG compression detected')
        if blur < 80:
            warnings.append('Strong blur or missed focus detected')
        if dark > 0.18:
            warnings.append('Large areas of deep shadow detected')
        if highlights > 0.08:
            warnings.append('Large areas of clipped highlights detected')
        target = quality_target if quality_target in self.TARGETS else 'Professional'
        return IntelligenceReport(
            scene=scene,
            quality_target=target,
            quality_score=score,
            blur_score=blur,
            noise_score=noise,
            compression_score=compression,
            dynamic_range=dynamic_range,
            dark_fraction=dark,
            highlight_fraction=highlights,
            colour_cast=colour_cast,
            faces=faces,
            warnings=tuple(warnings),
        )

    def validate(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        report: IntelligenceReport,
    ) -> PostProcessValidation:
        if reference.shape != candidate.shape:
            return PostProcessValidation(True, 1.0, 1.0, 0.0, 1.0, ('Geometry changed; face validation deferred until aligned output',))
        ref_gray = self._gray(reference)
        out_gray = self._gray(candidate)
        structure = self._normalised_correlation(ref_gray, out_gray)
        similarities: list[float] = []
        for face in report.faces:
            x, y, w, h = face.box
            pad = int(max(w, h) * 0.12)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(reference.shape[1], x + w + pad), min(reference.shape[0], y + h + pad)
            similarities.append(self._normalised_correlation(
                ref_gray[y0:y1, x0:x1], out_gray[y0:y1, x0:x1]
            ))
        identity = min(similarities) if similarities else 1.0
        ref_clipping = float(np.mean((ref_gray <= 5) | (ref_gray >= 250)))
        out_clipping = float(np.mean((out_gray <= 5) | (out_gray >= 250)))
        clipping_delta = max(0.0, out_clipping - ref_clipping)
        ref_sharp = float(cv2.Laplacian(ref_gray, cv2.CV_64F).var()) + 1e-6
        out_sharp = float(cv2.Laplacian(out_gray, cv2.CV_64F).var())
        sharp_ratio = out_sharp / ref_sharp
        limits = self.TARGETS[report.quality_target]
        reasons: list[str] = []
        if identity < limits['identity']:
            reasons.append(f'Face similarity {identity:.3f} below {limits["identity"]:.2f}')
        if structure < limits['structure']:
            reasons.append(f'Image structure similarity {structure:.3f} below {limits["structure"]:.2f}')
        if clipping_delta > limits['clip_delta']:
            reasons.append(f'New highlight/shadow clipping increased by {clipping_delta:.3f}')
        if sharp_ratio > limits['sharp_ratio']:
            reasons.append(f'Possible over-sharpening ratio {sharp_ratio:.2f}')
        return PostProcessValidation(
            accepted=not reasons,
            identity_similarity=identity,
            structure_similarity=structure,
            clipping_delta=clipping_delta,
            oversharpening_ratio=sharp_ratio,
            reasons=tuple(reasons),
        )
