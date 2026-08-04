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
    """Apply installed Auto processing profiles with conservative safeguards."""

    SUPPORTED_TYPES = {"processing-profile", "hybrid-core"}

    def __init__(self, models_root: Path | str) -> None:
        self.packs_root = Path(models_root) / "packs"

    @staticmethod
    def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.size == 0:
            raise ValueError("Image is empty")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        elif array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Unsupported image shape: {array.shape}")
        return np.ascontiguousarray(array)

    def installed_profiles(self) -> list[dict]:
        profiles: list[dict] = []
        if not self.packs_root.exists():
            return profiles
        for path in sorted(self.packs_root.glob("*/profile.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("pack_type") in self.SUPPORTED_TYPES:
                    payload["_path"] = str(path)
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
            for candidate in (directory / "manifest.json", directory / "profile.json"):
                if not candidate.exists():
                    continue
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                    pack_id = str(payload.get("pack_id", "")).strip()
                    if pack_id:
                        ids.add(pack_id)
                        break
                except Exception:
                    continue
        return ids

    @staticmethod
    def _matches(profile: dict, image_type: str, scene: str) -> bool:
        hints = profile.get("scene_hints", [])
        if not hints:
            return profile.get("pack_type") == "hybrid-core"
        haystack = f"{image_type} {scene}".lower()
        return any(str(hint).lower() in haystack for hint in hints)

    @staticmethod
    def _soft_detail(image: np.ndarray, amount: float) -> np.ndarray:
        image = AutoProfileRuntime._ensure_bgr_u8(image)
        amount = float(np.clip(amount, 0.0, 0.65))
        if amount <= 0:
            return image
        base = cv2.GaussianBlur(image, (0, 0), 1.15)
        return cv2.addWeighted(image, 1.0 + amount, base, -amount, 0)

    @staticmethod
    def _denoise(image: np.ndarray, amount: float) -> np.ndarray:
        image = AutoProfileRuntime._ensure_bgr_u8(image)
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount < 0.12:
            return image
        h = max(2, int(round(2 + amount * 5)))
        cleaned = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        blend = min(0.72, 0.20 + amount * 0.52)
        return cv2.addWeighted(image, 1.0 - blend, cleaned, blend, 0)

    @staticmethod
    def _jpeg_repair(image: np.ndarray, amount: float) -> np.ndarray:
        image = AutoProfileRuntime._ensure_bgr_u8(image)
        amount = float(np.clip(amount, 0.0, 1.0))
        if amount < 0.10:
            return image
        sigma = 0.55 + amount * 0.85
        softened = cv2.GaussianBlur(image, (0, 0), sigma)
        detail = cv2.addWeighted(image, 1.0 + amount * 0.12, softened, -amount * 0.12, 0)
        return cv2.addWeighted(image, 0.58, detail, 0.42, 0)

    @staticmethod
    def _lighting(image: np.ndarray, shadows: float, highlights: float) -> np.ndarray:
        image = AutoProfileRuntime._ensure_bgr_u8(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lum = l.astype(np.float32) / 255.0
        shadow_mask = np.clip((0.52 - lum) / 0.52, 0.0, 1.0)
        highlight_mask = np.clip((lum - 0.72) / 0.28, 0.0, 1.0)
        adjusted = lum + shadow_mask * float(np.clip(shadows, 0, 1)) * 0.16
        adjusted -= highlight_mask * float(np.clip(highlights, 0, 1)) * 0.10
        l2 = np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _colour(image: np.ndarray, amount: float) -> np.ndarray:
        image = AutoProfileRuntime._ensure_bgr_u8(image)
        amount = float(np.clip(amount, 0.0, 0.55))
        if amount <= 0:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s2 = np.clip(s.astype(np.float32) * (1.0 + amount * 0.20), 0, 255).astype(np.uint8)
        candidate = cv2.cvtColor(cv2.merge([h, s2, v]), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(image, 0.60, candidate, 0.40, 0)

    @staticmethod
    def _heritage_restore(image: np.ndarray, settings: dict) -> np.ndarray:
        image = AutoProfileRuntime._ensure_bgr_u8(image)
        gray_delta = float(np.mean(np.ptp(image.astype(np.int16), axis=2)))
        if gray_delta < 9.0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            background = cv2.GaussianBlur(gray, (0, 0), max(9.0, min(gray.shape) / 42.0))
            corrected = cv2.divide(gray, np.maximum(background, 1), scale=150)
            corrected = cv2.normalize(corrected, None, 8, 247, cv2.NORM_MINMAX)
            median = cv2.medianBlur(corrected, 3)
            dust = cv2.absdiff(corrected, median)
            dust_mask = (dust > 28).astype(np.uint8) * 255
            dust_mask = cv2.morphologyEx(dust_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            dust_ratio = float(np.mean(dust_mask > 0))
            if 0.0 < dust_ratio < 0.012:
                corrected = cv2.inpaint(corrected, dust_mask, 2, cv2.INPAINT_TELEA)
            restored = cv2.createCLAHE(clipLimit=1.65, tileGridSize=(8, 8)).apply(corrected)
            restored = cv2.addWeighted(restored, 1.10, cv2.GaussianBlur(restored, (0, 0), 0.9), -0.10, 0)
            return cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a = np.clip(a.astype(np.int16) + int(np.clip(128 - np.median(a), -8, 8)), 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.int16) + int(np.clip(128 - np.median(b), -8, 8)), 0, 255).astype(np.uint8)
        l = cv2.createCLAHE(clipLimit=1.45, tileGridSize=(8, 8)).apply(l)
        restored = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.addWeighted(image, 0.28, restored, 0.72, 0)

    @staticmethod
    def _social_recover(image: np.ndarray, settings: dict) -> np.ndarray:
        working = AutoProfileRuntime._ensure_bgr_u8(image)
        h, w = working.shape[:2]
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        text_mask = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        smooth = cv2.bilateralFilter(working, 7, 28, 28)
        repaired = cv2.addWeighted(smooth, 0.72, cv2.GaussianBlur(smooth, (0, 0), 0.55), 0.28, 0)
        mask3 = (text_mask.astype(np.float32) / 255.0)[..., None]
        working = np.clip(repaired.astype(np.float32) * (1.0 - mask3) + working.astype(np.float32) * mask3, 0, 255).astype(np.uint8)
        detail = cv2.addWeighted(working, 1.22, cv2.GaussianBlur(working, (0, 0), 1.0), -0.22, 0)
        delta = np.clip(detail.astype(np.int16) - working.astype(np.int16), -18, 18)
        working = np.clip(working.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        if bool(settings.get("auto_upscale", False)) and max(h, w) < 1400:
            scale = 2.0 if max(h, w) < 900 else 1.5
            working = cv2.resize(working, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LANCZOS4)
        return working

    @staticmethod
    def _motion_recovery(image: np.ndarray, deblur: float = 0.48, halo_limit: float = 0.18) -> tuple[np.ndarray, str]:
        image = AutoProfileRuntime._ensure_bgr_u8(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness >= 260:
            return image.copy(), "Motion recovery skipped: image is already sharp"
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ex, ey = float(np.mean(np.abs(gx))), float(np.mean(np.abs(gy)))
        sigma_x, sigma_y = (1.8, 0.7) if ex < ey * 0.78 else ((0.7, 1.8) if ey < ex * 0.78 else (1.15, 1.15))
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
        amount = float(np.clip(deblur, 0.0, 0.62))
        candidate = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        limit = int(round(10 + float(np.clip(halo_limit, 0, 1)) * 45))
        delta = np.clip(candidate.astype(np.int16) - image.astype(np.int16), -limit, limit)
        result = np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        return result, "Applied direction-aware motion/softness recovery"

    @staticmethod
    def _motion_recover(image: np.ndarray, settings: dict) -> np.ndarray:
        return AutoProfileRuntime._motion_recovery(image, float(settings.get("deblur", 0.48)), float(settings.get("halo_limit", 0.18)))[0]

    @staticmethod
    def _night_recovery(image: np.ndarray, denoise: float = 0.72, shadow_recovery: float = 0.65, highlight_protection: float = 0.62, colour: float = 0.28) -> tuple[np.ndarray, str]:
        image = AutoProfileRuntime._ensure_bgr_u8(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        if float(np.median(l)) > 112 and float(np.mean(l < 55)) < 0.16:
            return image.copy(), "Night recovery skipped: scene is not sufficiently dark"
        # bilateralFilter does not support a merged two-channel image; process channels separately.
        a = cv2.bilateralFilter(a, 7, 34, 34)
        b = cv2.bilateralFilter(b, 7, 34, 34)
        lum = l.astype(np.float32) / 255.0
        shadow = np.clip((0.62 - lum) / 0.62, 0.0, 1.0)
        highlights = np.clip((lum - 0.72) / 0.28, 0.0, 1.0)
        lum = lum + shadow * float(np.clip(shadow_recovery, 0, 1)) * 0.20
        lum -= highlights * float(np.clip(highlight_protection, 0, 1)) * 0.08
        l2 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        h_luma = max(2, int(round(2 + float(np.clip(denoise, 0, 1)) * 3)))
        h_colour = max(3, int(round(3 + float(np.clip(denoise, 0, 1)) * 5)))
        cleaned = cv2.fastNlMeansDenoisingColored(result, None, h_luma, h_colour, 7, 21)
        blend = 0.48 + float(np.clip(denoise, 0, 1)) * 0.20
        output = cv2.addWeighted(result, 1.0 - blend, cleaned, blend, 0)
        return output, "Applied low-light denoise, shadow recovery and highlight protection"

    @staticmethod
    def _night_recover(image: np.ndarray, settings: dict) -> np.ndarray:
        return AutoProfileRuntime._night_recovery(image, float(settings.get("denoise", 0.72)), float(settings.get("shadow_recovery", 0.65)), float(settings.get("highlight_protection", 0.62)), float(settings.get("colour", 0.28)))[0]

    def apply(self, image: np.ndarray, image_type: str, scene: str) -> tuple[np.ndarray, ProfileApplicationReport]:
        report = ProfileApplicationReport()
        working = self._ensure_bgr_u8(image)
        installed_ids = self.installed_pack_ids()
        matches = [p for p in self.installed_profiles() if self._matches(p, image_type, scene)]
        matches.sort(key=lambda item: 0 if item.get("pack_type") == "hybrid-core" else 1)
        applied_count = 0
        for profile in matches:
            dependencies = [str(item) for item in profile.get("depends_on", [])]
            missing = [item for item in dependencies if item not in installed_ids]
            name = str(profile.get("name", profile.get("pack_id", "Auto profile")))
            pack_id = str(profile.get("pack_id", ""))
            if missing:
                report.skipped.append(name)
                report.messages.append(f"Skipped {name}: install {', '.join(missing)} first")
                continue
            if applied_count >= 2:
                report.skipped.append(name)
                report.messages.append(f"Skipped {name}: safe profile stacking limit reached")
                continue
            settings = profile.get("settings", {})
            before = working
            dedicated_message = ""
            if pack_id == "auto-motion-recovery":
                working, dedicated_message = self._motion_recovery(working, float(settings.get("deblur", 0.48)), float(settings.get("halo_limit", 0.18)))
            elif pack_id == "auto-night-recovery":
                working, dedicated_message = self._night_recovery(working, float(settings.get("denoise", 0.72)), float(settings.get("shadow_recovery", 0.65)), float(settings.get("highlight_protection", 0.62)), float(settings.get("colour", 0.28)))
            elif pack_id == "auto-heritage":
                working = self._heritage_restore(working, settings)
            elif pack_id == "auto-social-recovery":
                working = self._social_recover(working, settings)
            else:
                working = self._jpeg_repair(working, float(settings.get("jpeg_repair", 0.0)))
                working = self._denoise(working, float(settings.get("denoise", 0.0)))
                working = self._lighting(working, float(settings.get("shadow_recovery", settings.get("face_relight", 0.0) * 0.45)), float(settings.get("highlight_protection", 0.0)))
                working = self._soft_detail(working, float(settings.get("detail_recovery", settings.get("text_sharpen", 0.0))))
                working = self._colour(working, float(settings.get("colour", 0.0)))
            if working.shape == before.shape:
                working = cv2.addWeighted(before, 0.24, working, 0.76, 0)
            report.selected.append(name)
            report.messages.append(dedicated_message or f"Applied installed profile: {name}")
            applied_count += 1
        return working, report
