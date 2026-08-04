from __future__ import annotations

import json

import numpy as np

from auto_profile_runtime import AutoProfileRuntime


def test_profile_pack_is_discovered_and_applied(tmp_path):
    pack = tmp_path / 'models' / 'packs' / 'auto-night-recovery'
    pack.mkdir(parents=True)
    (pack / 'profile.json').write_text(
        json.dumps({
            'pack_type': 'processing-profile',
            'pack_id': 'auto-night-recovery',
            'name': 'Auto Night Recovery',
            'scene_hints': ['night', 'low light'],
            'settings': {
                'denoise': 0.45,
                'shadow_recovery': 0.55,
                'highlight_protection': 0.35,
                'detail_recovery': 0.18,
                'colour': 0.20,
            },
        }),
        encoding='utf-8',
    )

    runtime = AutoProfileRuntime(tmp_path / 'models')
    image = np.full((96, 96, 3), 24, dtype=np.uint8)
    image[20:76, 20:76] = 55

    output, report = runtime.apply(image, 'Low Resolution Photo', 'Night / low light')

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert report.selected == ['Auto Night Recovery']
    assert float(output.mean()) > float(image.mean())


def test_profile_stack_is_limited_to_two(tmp_path):
    packs_root = tmp_path / 'models' / 'packs'
    for index in range(3):
        pack = packs_root / f'pack-{index}'
        pack.mkdir(parents=True)
        (pack / 'profile.json').write_text(
            json.dumps({
                'pack_type': 'processing-profile',
                'pack_id': f'pack-{index}',
                'name': f'Pack {index}',
                'scene_hints': ['portrait'],
                'settings': {'detail_recovery': 0.10},
            }),
            encoding='utf-8',
        )

    runtime = AutoProfileRuntime(tmp_path / 'models')
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    _, report = runtime.apply(image, 'Portrait', 'Portrait')

    assert len(report.selected) == 2
