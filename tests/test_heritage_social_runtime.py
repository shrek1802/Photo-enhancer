import json
from pathlib import Path

import cv2
import numpy as np

from auto_profile_runtime import AutoProfileRuntime


def _install_profile(root: Path, pack_id: str, name: str, scene_hints: list[str], settings: dict) -> None:
    pack = root / 'packs' / pack_id
    pack.mkdir(parents=True, exist_ok=True)
    (pack / 'profile.json').write_text(json.dumps({
        'schema_version': 1,
        'pack_id': pack_id,
        'name': name,
        'version': '1.0.0',
        'pack_type': 'processing-profile',
        'scene_hints': scene_hints,
        'settings': settings,
    }), encoding='utf-8')


def test_heritage_monochrome_remains_monochrome_and_preserves_size(tmp_path: Path) -> None:
    runtime = AutoProfileRuntime(tmp_path)
    _install_profile(tmp_path, 'auto-heritage', 'Auto Heritage', ['Black & White Restore'], {})

    base = np.tile(np.linspace(35, 205, 180, dtype=np.uint8), (120, 1))
    image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    image[20, 30] = (255, 255, 255)
    result, report = runtime.apply(image, 'Black & White Restore', 'Old Photo')

    assert result.shape == image.shape
    assert 'Auto Heritage' in report.selected
    assert float(np.mean(np.ptp(result.astype(np.int16), axis=2))) < 2.0
    assert float(result.std()) > 8.0


def test_heritage_colour_restore_does_not_create_extreme_cast(tmp_path: Path) -> None:
    runtime = AutoProfileRuntime(tmp_path)
    _install_profile(tmp_path, 'auto-heritage', 'Auto Heritage', ['Old Photo'], {})

    image = np.full((100, 140, 3), (72, 105, 145), dtype=np.uint8)
    cv2.circle(image, (70, 50), 25, (90, 125, 168), -1)
    result, _ = runtime.apply(image, 'Old Photo', 'Old Photo')

    channel_means = result.reshape(-1, 3).mean(axis=0)
    assert result.shape == image.shape
    assert float(channel_means.max() - channel_means.min()) < 95.0


def test_social_recovery_protects_text_edges_and_upscales_small_image(tmp_path: Path) -> None:
    runtime = AutoProfileRuntime(tmp_path)
    _install_profile(
        tmp_path,
        'auto-social-recovery',
        'Auto Social Recovery',
        ['Screenshot Recovery'],
        {'auto_upscale': True},
    )

    image = np.full((240, 320, 3), 185, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (319, 36), (35, 45, 70), -1)
    cv2.putText(image, 'FACEBOOK', (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    # Add blocky compression-like pattern.
    for y in range(48, 220, 16):
        for x in range(8, 312, 16):
            value = 130 + ((x + y) // 16 % 3) * 18
            image[y:y + 16, x:x + 16] = (value, value + 4, value + 8)

    before_edges = cv2.Canny(cv2.cvtColor(image[:40], cv2.COLOR_BGR2GRAY), 70, 150).mean()
    result, report = runtime.apply(image, 'Screenshot Recovery', 'Screenshot Recovery')
    after_top = cv2.resize(result[:80], (320, 40), interpolation=cv2.INTER_AREA)
    after_edges = cv2.Canny(cv2.cvtColor(after_top, cv2.COLOR_BGR2GRAY), 70, 150).mean()

    assert result.shape[0] == 480
    assert result.shape[1] == 640
    assert 'Auto Social Recovery' in report.selected
    assert after_edges >= before_edges * 0.60


def test_social_recovery_never_downsizes_large_image(tmp_path: Path) -> None:
    runtime = AutoProfileRuntime(tmp_path)
    _install_profile(
        tmp_path,
        'auto-social-recovery',
        'Auto Social Recovery',
        ['Screenshot Recovery'],
        {'auto_upscale': True},
    )
    image = np.full((1200, 1600, 3), 128, dtype=np.uint8)
    result, _ = runtime.apply(image, 'Screenshot Recovery', 'Screenshot Recovery')
    assert result.shape == image.shape
