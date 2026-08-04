import json
from pathlib import Path

import cv2
import numpy as np

from auto_profile_runtime import AutoProfileRuntime


def _install_profile(root: Path, pack_id: str, name: str, hints: list[str], settings: dict) -> None:
    directory = root / 'packs' / pack_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'profile.json').write_text(json.dumps({
        'schema_version': 1,
        'pack_id': pack_id,
        'name': name,
        'version': '1.0.0',
        'pack_type': 'processing-profile',
        'scene_hints': hints,
        'settings': settings,
    }), encoding='utf-8')


def test_motion_recovery_preserves_shape_and_limits_changes(tmp_path: Path):
    _install_profile(tmp_path, 'auto-motion-recovery', 'Auto Motion Recovery', ['General Photograph'], {
        'deblur': 0.58,
        'detail_recovery': 0.46,
        'halo_limit': 0.18,
    })
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(image, (45, 35), (175, 125), (210, 210, 210), -1)
    image = cv2.GaussianBlur(image, (0, 0), 3.0)

    result, report = AutoProfileRuntime(tmp_path).apply(image, 'General Photograph', 'General Photograph')

    assert result.shape == image.shape
    assert 'Auto Motion Recovery' in report.selected
    assert np.max(np.abs(result.astype(np.int16) - image.astype(np.int16))) <= 28
    assert any('motion' in message.lower() or 'softness' in message.lower() for message in report.messages)


def test_motion_recovery_skips_already_sharp_image():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[::2, ::2] = 255
    result, message = AutoProfileRuntime._motion_recovery(image, 0.58, 0.18)
    assert result.shape == image.shape
    assert 'already sharp' in message.lower()


def test_night_recovery_lifts_dark_midtones_without_clipping_lamps(tmp_path: Path):
    _install_profile(tmp_path, 'auto-night-recovery', 'Auto Night Recovery', ['Low Light'], {
        'denoise': 0.72,
        'shadow_recovery': 0.70,
        'highlight_protection': 0.62,
        'colour': 0.28,
    })
    rng = np.random.default_rng(7)
    image = np.full((140, 200, 3), (24, 30, 38), dtype=np.uint8)
    noise = rng.normal(0, 7, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(image, (155, 42), 14, (245, 245, 250), -1)
    before_lamp = image[42, 155].copy()

    result, report = AutoProfileRuntime(tmp_path).apply(image, 'Low Light', 'Low Light')

    assert result.shape == image.shape
    assert 'Auto Night Recovery' in report.selected
    assert float(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).mean()) > float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    assert np.max(np.abs(result[42, 155].astype(np.int16) - before_lamp.astype(np.int16))) < 12
    assert any('low-light' in message.lower() for message in report.messages)


def test_night_recovery_skips_bright_scene():
    image = np.full((80, 100, 3), 180, dtype=np.uint8)
    result, message = AutoProfileRuntime._night_recovery(image, 0.72, 0.70, 0.62, 0.28)
    assert np.array_equal(result, image)
    assert 'not sufficiently dark' in message.lower()
