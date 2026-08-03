from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED


@dataclass
class EnhanceOptions:
    strength: str = 'natural'
    upscale: str = 'Original size'
    lift_shadows: bool = True
    recover_highlights: bool = True
    reduce_flare: bool = True
    denoise: bool = True
    sharpen: bool = True
    face_aware: bool = True
    auto_rotate: bool = True
    jpeg_quality: int = 95


@dataclass
class ProcessResult:
    review_needed: bool
    blur_score: float
    clipped_highlights: float
    dark_fraction: float


class PhotoEnhancer:
    def __init__(self, options: EnhanceOptions):
        self.options = options
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _read(self, path: Path) -> np.ndarray:
        image = Image.open(path)
        if self.options.auto_rotate:
            image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _white_balance(image: np.ndarray) -> np.ndarray:
        work = image.astype(np.float32)
        means = np.array(cv2.mean(work)[:3], dtype=np.float32)
        target = float(means.mean())
        scales = np.clip(target / np.maximum(means, 1.0), 0.85, 1.18)
        return np.clip(work * scales.reshape(1, 1, 3), 0, 255).astype(np.uint8)

    def _tone(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = {'natural': 1.4, 'strong': 1.8, 'maximum': 2.2}[self.options.strength]
        l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)

        if self.options.lift_shadows:
            gamma = {'natural': 0.94, 'strong': 0.88, 'maximum': 0.82}[self.options.strength]
            lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype('uint8')
            lifted = cv2.LUT(l2, lut)
            mask = cv2.GaussianBlur((255 - l).astype(np.uint8), (0, 0), 15) / 255.0
            amount = {'natural': 0.35, 'strong': 0.50, 'maximum': 0.65}[self.options.strength]
            l2 = np.clip(l2 * (1 - mask * amount) + lifted * mask * amount, 0, 255).astype(np.uint8)

        if self.options.recover_highlights:
            high = np.clip((l2.astype(np.float32) - 190) / 65, 0, 1)
            compression = {'natural': 10, 'strong': 18, 'maximum': 26}[self.options.strength]
            l2 = np.clip(l2.astype(np.float32) - high * compression, 0, 255).astype(np.uint8)

        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    def _reduce_flare(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, saturation, value = cv2.split(hsv)
        mask = ((((value > 238) & (saturation < 70)) | ((value > 225) & (saturation > 95))).astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        safe = np.zeros_like(mask)
        total = image.shape[0] * image.shape[1]
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if 12 <= area <= total * 0.012:
                safe[labels == label] = 255
        if safe.mean() < 0.2:
            return image
        safe = cv2.dilate(safe, np.ones((5, 5), np.uint8), iterations=1)
        return cv2.inpaint(image, safe, 3, cv2.INPAINT_TELEA)

    def _faces(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        minimum = max(40, min(image.shape[:2]) // 14)
        faces = self.face_detector.detectMultiScale(gray, 1.12, 5, minSize=(minimum, minimum))
        result = image.copy()
        for x, y, w, h in faces:
            pad = int(w * 0.12)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
            roi = result[y0:y1, x0:x1]
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_l = float(l.mean())
            if mean_l < 122:
                gain = min(1.16, 122 / max(mean_l, 1))
                l = np.clip(l.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            corrected = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            result[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.35, corrected, 0.65, 0)
        return result, len(faces)

    @staticmethod
    def _blur_score(image: np.ndarray) -> float:
        return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        h = {'natural': 3, 'strong': 5, 'maximum': 7}[self.options.strength]
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)

    def _sharpen(self, image: np.ndarray, blur_score: float) -> np.ndarray:
        if blur_score > 650:
            amount = 0.20
        elif blur_score > 250:
            amount = 0.38
        elif blur_score > 90:
            amount = 0.58
        else:
            amount = 0.80
        amount *= {'natural': 0.8, 'strong': 1.0, 'maximum': 1.15}[self.options.strength]
        blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge = np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_32F)) / 30.0, 0, 1)
        edge = cv2.GaussianBlur(edge, (0, 0), 1.2)[..., None]
        return np.clip(image * (1 - edge) + sharpened * edge, 0, 255).astype(np.uint8)

    def _upscale(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if self.options.upscale == '2× upscale':
            return cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        if self.options.upscale == '4K long edge' and max(h, w) < 3840:
            scale = 3840 / max(h, w)
            return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        return image

    def process(self, source: Path, destination: Path) -> ProcessResult:
        image = self._read(source)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = self._blur_score(image)
        clipped = float(np.mean(gray > 250))
        dark = float(np.mean(gray < 35))
        image = self._tone(self._white_balance(image))
        if self.options.reduce_flare:
            image = self._reduce_flare(image)
        face_count = 0
        if self.options.face_aware:
            image, face_count = self._faces(image)
        if self.options.denoise:
            image = self._denoise(image)
        if self.options.sharpen:
            image = self._sharpen(image, blur)
        image = self._upscale(image)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, self.options.jpeg_quality]):
            raise OSError(f'Could not write {destination}')
        review = blur < 35 or clipped > 0.10 or dark > 0.30
        return ProcessResult(review, blur, clipped, dark)
