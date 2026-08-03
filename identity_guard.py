from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class IdentityResult:
    image: np.ndarray
    faces_protected: int
    minimum_similarity: float


class FaceIdentityGuard:
    """Preserve captured facial structure while allowing lighting and colour polish.

    This guard never generates facial features. It detects faces in a reference
    frame, transfers only safe luminance/colour changes, and blends the original
    high-frequency facial structure back into the processed result.
    """

    def __init__(self) -> None:
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        minimum = max(36, min(image.shape[:2]) // 18)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(minimum, minimum)
        )
        return [tuple(map(int, face)) for face in faces]

    @staticmethod
    def _similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
        if reference.shape != candidate.shape or reference.size == 0:
            return 0.0
        ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cand = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref = cv2.resize(ref, (96, 96), interpolation=cv2.INTER_AREA)
        cand = cv2.resize(cand, (96, 96), interpolation=cv2.INTER_AREA)
        ref -= ref.mean()
        cand -= cand.mean()
        denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
        if denominator < 1e-6:
            return 1.0
        return float(np.clip(np.sum(ref * cand) / denominator, -1.0, 1.0))

    @staticmethod
    def _protect_face(reference: np.ndarray, processed: np.ndarray) -> np.ndarray:
        # Keep the processed image's improved overall lighting and colour, but
        # restore the real high-frequency facial structure from the reference.
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        out_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        ref_l, ref_a, ref_b = cv2.split(ref_lab)
        out_l, out_a, out_b = cv2.split(out_lab)

        ref_low = cv2.GaussianBlur(ref_l, (0, 0), 2.0)
        ref_detail = ref_l.astype(np.float32) - ref_low.astype(np.float32)
        safe_l = np.clip(out_l.astype(np.float32) + ref_detail * 0.88, 0, 255).astype(np.uint8)

        # Colour may improve, but limit skin-colour movement to avoid changing identity.
        safe_a = np.clip(out_a.astype(np.float32) * 0.55 + ref_a.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
        safe_b = np.clip(out_b.astype(np.float32) * 0.55 + ref_b.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
        protected = cv2.cvtColor(cv2.merge([safe_l, safe_a, safe_b]), cv2.COLOR_LAB2BGR)

        h, w = reference.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        rx, ry = max(w * 0.49, 1), max(h * 0.52, 1)
        alpha = 1.0 - np.clip(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2, 0, 1)
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 5)[..., None]
        blended = processed.astype(np.float32) * (1 - alpha) + protected.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    def apply(self, reference: np.ndarray, processed: np.ndarray) -> IdentityResult:
        if reference.shape != processed.shape:
            return IdentityResult(processed, 0, 0.0)

        result = processed.copy()
        similarities: list[float] = []
        protected_count = 0
        for x, y, w, h in self.detect(reference):
            pad = int(max(w, h) * 0.14)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(reference.shape[1], x + w + pad), min(reference.shape[0], y + h + pad)
            ref_roi = reference[y0:y1, x0:x1]
            out_roi = result[y0:y1, x0:x1]
            if ref_roi.size == 0 or out_roi.size == 0:
                continue
            similarity = self._similarity(ref_roi, out_roi)
            similarities.append(similarity)
            result[y0:y1, x0:x1] = self._protect_face(ref_roi, out_roi)
            protected_count += 1

        minimum = min(similarities) if similarities else 1.0
        return IdentityResult(result, protected_count, minimum)
