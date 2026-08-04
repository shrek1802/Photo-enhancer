from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ProfileApplicationReport:
    selected: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class AutoProfileRuntime:
    """Applies conservative settings from installed Auto processing packs.

    Profile packs specialise the built-in Auto Engine. Hybrid core packs may
    additionally provide shared neural capabilities. Dependencies are checked
    before a specialist profile is applied.
    """

    SUPPORTED_TYPES = {'processing-profile', 'hybrid-core'}

    def __init__(self, models_root: Path | str) -> None:
        self.packs_root = Path(models_root) / 'packs'

    def installed_profiles(self) -> list[dict]:
        profiles: list[dict] = []
        if not self.packs_root.exists():
            return profiles
        for path in sorted(self.packs_root.glob('*/profile.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                if payload.get('pack_type') in self.SUPPORTED_TYPES:
                    payload['_path'] = str(path)
                    profiles.append(payload)
            except Exception:
                continue
        return profiles

    def installed_pack_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self.packs_root.exists():
            return ids
        for directory in self.packs_root.iterdir():
            if not directory.is_dir():
                continue
            for candidate in (directory / 'manifest.json', directory / 'profile.json'):
                if not candidate.exists():
                    continue
                try:
                    payload = json.loads(candidate.read_text(encoding='utf-8'))
                    pack_id = str(payload.get('pack_id', '')).strip()
                    if pack_id:
                        ids.add(pack_id)
                        break
                except Exception:
                    continue
        return ids

    @staticmethod
    def _matches(profile: dict, image_type: str, scene: str) -> bool:
        hints = profile.get('scene_hints', [])
        if not hints:
            return profile.get('pack_type') == 'hybrid-core'
        haystack = f'{image_type} {scene}'.lower()
        return any(str(hint).lower() in haystack for hint in hints)

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
    def _jpeg_repair(image: np.ndarray, amount: float) -> np.ndarray:
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount < 0.10:
            return image
        sigma = 0.55 + amount * 0.85
        softened = cv2.GaussianBlur(image, (0, 0), sigma)
        detail = cv2.addWeighted(image, 1.0 + amount * 0.12, softened, -amount * 0.12, 0)
        return cv2.addWeighted(image, 0.58, detail, 0.42, 0)

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

    @staticmethod
    def _blur_metrics(image: np.ndarray) -> tuple[float, float, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        horizontal = float(np.mean(np.abs(gx)))
        vertical = float(np.mean(np.abs(gy)))
        return lap, horizontal, vertical

    @classmethod
    def _motion_recovery(cls, image: np.ndarray, amount: float, halo_limit: float) -> tuple[np.ndarray, str]:
        amount = float(np.clip(amount, 0.0, 0.85))
        if amount < 0.10:
            return image, 'Motion recovery not required'
        lap, horizontal, vertical = cls._blur_metrics(image)
        if lap > 360:
            return image, 'Motion recovery skipped: image already sharp'

        # Prefer a directional high-pass when one edge direction is noticeably weaker.
        ratio = (horizontal + 1e-4) / (vertical + 1e-4)
        if ratio < 0.72:
            kernel = np.array([[0, -0.18, 0], [0, 1.36, 0], [0, -0.18, 0]], dtype=np.float32)
            blur_type = 'horizontal motion softness'
        elif ratio > 1.38:
            kernel = np.array([[0, 0, 0], [-0.18, 1.36, -0.18], [0, 0, 0]], dtype=np.float32)
            blur_type = 'vertical motion softness'
        else:
            kernel = np.array([[0, -0.08, 0], [-0.08, 1.32, -0.08], [0, -0.08, 0]], dtype=np.float32)
            blur_type = 'general camera softness'

        directional = cv2.filter2D(image, -1, kernel)
        gaussian = cv2.GaussianBlur(image, (0, 0), 1.0)
        unsharp = cv2.addWeighted(image, 1.0 + amount * 0.42, gaussian, -amount * 0.42, 0)
        candidate = cv2.addWeighted(directional, 0.45, unsharp, 0.55, 0)

        # Suppress halos by limiting per-channel changes around strong edges.
        limit = max(5.0, 10.0 + float(np.clip(halo_limit, 0, 1)) * 18.0)
        delta = np.clip(candidate.astype(np.float32) - image.astype(np.float32), -limit, limit)
        result = np.clip(image.astype(np.float32) + delta * min(0.82, 0.38 + amount * 0.52), 0, 255)
        return result.astype(np.uint8), f'Applied {blur_type} recovery'

    @staticmethod
    def _night_recovery(image: np.ndarray, denoise: float, shadows: float, highlights: float, colour: float) -> tuple[np.ndarray, str]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_fraction = float(np.mean(gray < 58))
        if dark_fraction < 0.18:
            return image, 'Night recovery skipped: scene is not sufficiently dark'

        # Reduce chroma noise more strongly than luminance noise to preserve detail.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        chroma_strength = max(3, int(round(3 + np.clip(denoise, 0, 1) * 7)))
        a2 = cv2.bilateralFilter(a, 7, chroma_strength * 2, 21)
        b2 = cv2.bilateralFilter(b, 7, chroma_strength * 2, 21)
        l2 = cv2.fastNlMeansDenoising(l, None, max(2, int(2 + denoise * 4)), 7, 21)
        working = cv2.cvtColor(cv2.merge([l2, a2, b2]), cv2.COLOR_LAB2BGR)

        # Gray-world correction is deliberately capped to avoid false skin tones.
        means = np.mean(working.reshape(-1, 3), axis=0)
        target = float(np.mean(means))
        gains = np.clip(target / np.maximum(means, 1.0), 0.88, 1.12)
        balanced = np.clip(working.astype(np.float32) * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
        working = cv2.addWeighted(working, 0.55, balanced, 0.45, 0)

        working = AutoProfileRuntime._lighting(working, shadows, highlights)
        working = AutoProfileRuntime._colour(working, colour)

        # Protect lamps, windows and stage lights from clipping after shadow lift.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        highlight_mask = cv2.GaussianBlur((hsv[..., 2] > 220).astype(np.float32), (0, 0), 3.0)[..., None]
        result = working.astype(np.float32) * (1.0 - highlight_mask) + image.astype(np.float32) * highlight_mask
        return np.clip(result, 0, 255).astype(np.uint8), 'Applied low-light chroma denoise, colour balance and protected shadow recovery'

    def apply(self, image: np.ndarray, image_type: str, scene: str) -> tuple[np.ndarray, ProfileApplicationReport]:
        report = ProfileApplicationReport()
        working = image
        installed_ids = self.installed_pack_ids()
        profiles = self.installed_profiles()
        matches = [profile for profile in profiles if self._matches(profile, image_type, scene)]
        matches.sort(key=lambda item: 0 if item.get('pack_type') == 'hybrid-core' else 1)

        applied_count = 0
        for profile in matches:
            dependencies = [str(item) for item in profile.get('depends_on', [])]
            missing = [item for item in dependencies if item not in installed_ids]
            name = str(profile.get('name', profile.get('pack_id', 'Auto profile')))
            if missing:
                report.skipped.append(name)
                report.messages.append(f"Skipped {name}: install {', '.join(missing)} first")
                continue
            if applied_count >= 2:
                report.skipped.append(name)
                report.messages.append(f'Skipped {name}: safe profile stacking limit reached')
                continue

            settings = profile.get('settings', {})
            pack_id = str(profile.get('pack_id', ''))
            before = working

            if pack_id == 'auto-night-recovery':
                working, message = self._night_recovery(
                    working,
                    float(settings.get('denoise', 0.72)),
                    float(settings.get('shadow_recovery', 0.70)),
                    float(settings.get('highlight_protection', 0.62)),
                    float(settings.get('colour', 0.28)),
                )
                report.messages.append(message)
            elif pack_id == 'auto-motion-recovery':
                working, message = self._motion_recovery(
                    working,
                    float(settings.get('deblur', 0.58)),
                    float(settings.get('halo_limit', 0.18)),
                )
                working = self._denoise(working, float(settings.get('denoise', 0.18)))
                report.messages.append(message)
            else:
                working = self._jpeg_repair(working, float(settings.get('jpeg_repair', 0.0)))
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

            working = cv2.addWeighted(before, 0.24, working, 0.76, 0)
            report.selected.append(name)
            report.messages.append(f'Applied installed profile: {name}')
            applied_count += 1
        return working, report
