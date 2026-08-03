from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PhotoAnalysis:
    filename: str
    quality_score: int
    scene: str
    blur_score: float
    brightness: float
    dark_fraction: float
    clipped_highlights: float
    contrast: float
    noise_score: float
    face_count: int
    horizon_angle: float
    review_reason: str = ''


def read_bgr(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def perceptual_hash(path: Path, size: int = 16) -> str:
    image = ImageOps.exif_transpose(Image.open(path)).convert('L').resize((size, size))
    data = np.asarray(image, dtype=np.float32)
    dct = cv2.dct(data)
    low = dct[:8, :8]
    median = np.median(low[1:])
    bits = (low > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def estimate_horizon(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90,
                            minLineLength=max(80, gray.shape[1] // 5), maxLineGap=20)
    if lines is None:
        return 0.0
    angles: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        if abs(angle) <= 12:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


class PhotoAnalyser:
    def __init__(self) -> None:
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def analyse(self, path: Path) -> PhotoAnalysis:
        image = read_bgr(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = gray.shape
        preview_scale = min(1.0, 1400 / max(height, width))
        if preview_scale < 1:
            preview = cv2.resize(gray, None, fx=preview_scale, fy=preview_scale,
                                 interpolation=cv2.INTER_AREA)
        else:
            preview = gray

        blur = float(cv2.Laplacian(preview, cv2.CV_64F).var())
        brightness = float(gray.mean())
        dark = float(np.mean(gray < 35))
        clipped = float(np.mean(gray > 250))
        contrast = float(gray.std())
        smooth = cv2.GaussianBlur(preview, (3, 3), 0)
        noise = float(np.std(preview.astype(np.float32) - smooth.astype(np.float32)))
        minimum = max(36, min(preview.shape[:2]) // 16)
        faces = self.face_detector.detectMultiScale(preview, 1.12, 5,
                                                     minSize=(minimum, minimum))
        face_count = len(faces)
        saturation = float(hsv[:, :, 1].mean())

        if face_count >= 2:
            scene = 'Group portrait'
        elif face_count == 1:
            scene = 'Portrait'
        elif brightness < 75:
            scene = 'Night / low light'
        elif saturation < 35:
            scene = 'Old or faded photo'
        elif width > height * 1.25:
            scene = 'Landscape / event'
        else:
            scene = 'General photo'

        penalties = 0.0
        penalties += max(0, 70 - min(70, blur)) * 0.38
        penalties += dark * 34
        penalties += clipped * 45
        penalties += max(0, noise - 5) * 1.3
        penalties += max(0, 34 - contrast) * 0.45
        if brightness < 55 or brightness > 215:
            penalties += 9
        score = int(np.clip(round(100 - penalties), 1, 100))

        reasons: list[str] = []
        if blur < 35:
            reasons.append('severe blur')
        elif blur < 75:
            reasons.append('soft focus')
        if dark > 0.28:
            reasons.append('very dark')
        if clipped > 0.10:
            reasons.append('blown highlights')
        if noise > 10:
            reasons.append('heavy noise')
        if score < 48:
            reasons.append('low quality score')

        return PhotoAnalysis(
            filename=path.name,
            quality_score=score,
            scene=scene,
            blur_score=round(blur, 2),
            brightness=round(brightness, 2),
            dark_fraction=round(dark, 4),
            clipped_highlights=round(clipped, 4),
            contrast=round(contrast, 2),
            noise_score=round(noise, 2),
            face_count=face_count,
            horizon_angle=round(estimate_horizon(preview), 2),
            review_reason=', '.join(reasons),
        )


def write_report(path: Path, rows: list[PhotoAnalysis]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
