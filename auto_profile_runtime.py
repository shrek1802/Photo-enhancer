from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ProfileApplicationReport:
    selected: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class AutoProfileRuntime:
    """Applies conservative settings from installed processing-profile packs.

    Profile packs improve and specialise the built-in Auto Engine. They are
    separate from ONNX neural weights and therefore always retain safe local
    processing and identity protection.
    """

    def __init__(self, models_root: Path | str) -> None:
        self.packs_root = Path(models_root) / 'packs'

    def installed_profiles(self) -> list[dict]:
        profiles: list[dict] = []
        if not self.packs_root.exists():
            return profiles
        for path in sorted(self.packs_root.glob('*/profile.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                if payload.get('pack_type') == 'processing-profile':
                    profiles.append(payload)
            except Exception:
                continue
        return profiles

    @staticmethod
    def _matches(profile: dict, image_type: str, scene: str) -> bool:
        haystack = f'{image_type} {scene}'.lower()
        return any(str(hint).lower() in haystack for hint in profile.get('scene_hints', []))

    @staticmethod
    def _soft_detail(image: np.ndarray, amount: float) -> np.ndarray:
        amount = float(np.clip(amount, 0.0, 0.65))
        if amount <= 0:
            return image
        base = cv2.GaussianBlur(image, (0, 0), 1.15)
        return cv2.addWeighted(image, 1.0 + amount, base, -amount, 0)

    @staticmethod
    def _denoise(image: np.ndarray, amount: float) -> np.ndarray:
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount < 0.12:
            return image
        h = max(2, int(round(2 + amount * 5)))
        cleaned = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        blend = min(0.72, 0.20 + amount * 0.52)
        return cv2.addWeighted(image, 1.0 - blend, cleaned, blend, 0)

    @staticmethod
    def _lighting(image: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lum = l.astype(np.float32) / 255.0
        shadow_mask = np.clip((0.52 - lum) / 0.52, 0.0, 1.0)
        highlight_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.0)
        adjusted = lum.copy()
        adjusted += shadow_mask * float(np.clip(shadows, 0, 1)) * 0.16
        adjusted -= highlight_mask * float(np.clip(highlights, 0, 1)) * 0.10
        l2 = np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _colour(image: np.ndarray, amount: float) -> np.ndarray:
        amount = float(np.clip(amount, 0.0, 0.55))
        if amount <= 0:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s2 = np.clip(s.astype(np.float32) * (1.0 + amount * 0.20), 0, 255).astype(np.uint8)
        candidate = cv2.cvtColor(cv2.merge([h, s2, v]), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(image, 0.60, candidate, 0.40, 0)

    def apply(self, image: np.ndarray, image_type: str, scene: str) -> tuple[np.ndarray, ProfileApplicationReport]:
        report = ProfileApplicationReport()
        working = image
        matches = [p for p in self.installed_profiles() if self._matches(p, image_type, scene)]
        # Limit stacking so multiple overlapping packs cannot over-process a photo.
        for profile in matches[:2]:
            settings = profile.get('settings', {})
            before = working
            working = self._denoise(working, float(settings.get('denoise', 0.0)))
            working = self._lighting(
                working,
                float(settings.get('shadow_recovery', settings.get('face_relight', 0.0) * 0.45)),
                float(settings.get('highlight_protection', 0.0)),
            )
            working = self._soft_detail(
                working,
                float(settings.get('detail_recovery', settings.get('text_sharpen', 0.0))),
            )
            working = self._colour(working, float(settings.get('colour', 0.0)))
            # Profile processing remains restrained even when several settings are active.
            working = cv2.addWeighted(before, 0.28, working, 0.72, 0)
            report.selected.append(str(profile.get('name', profile.get('pack_id', 'Auto profile'))))
            report.messages.append(f"Applied installed profile: {report.selected[-1]}")
        return working, report
