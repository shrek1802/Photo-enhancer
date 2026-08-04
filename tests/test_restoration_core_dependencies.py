from __future__ import annotations

import json

import numpy as np

from auto_profile_runtime import AutoProfileRuntime


def _write_pack(root, pack_id: str, profile: dict) -> None:
    pack = root / 'packs' / pack_id
    pack.mkdir(parents=True, exist_ok=True)
    (pack / 'manifest.json').write_text(json.dumps({
        'schema_version': 1,
        'pack_id': pack_id,
        'name': profile['name'],
        'version': '1.0.0',
        'files': [],
    }), encoding='utf-8')
    (pack / 'profile.json').write_text(json.dumps(profile), encoding='utf-8')


def test_specialist_profile_waits_for_shared_core(tmp_path):
    specialist = {
        'schema_version': 1,
        'pack_id': 'auto-night-recovery',
        'name': 'Auto Night Recovery',
        'pack_type': 'processing-profile',
        'scene_hints': ['night'],
        'depends_on': ['auto-restoration-core'],
        'settings': {'denoise': 0.5},
    }
    _write_pack(tmp_path, 'auto-night-recovery', specialist)
    runtime = AutoProfileRuntime(tmp_path)
    image = np.full((48, 48, 3), 40, dtype=np.uint8)

    result, report = runtime.apply(image, 'Photo', 'night')

    assert report.selected == []
    assert report.skipped == ['Auto Night Recovery']
    assert np.array_equal(result, image)


def test_shared_core_enables_specialist_profile(tmp_path):
    core = {
        'schema_version': 1,
        'pack_id': 'auto-restoration-core',
        'name': 'Auto Restoration Core',
        'pack_type': 'hybrid-core',
        'scene_hints': ['night'],
        'settings': {'denoise': 0.2, 'jpeg_repair': 0.2},
    }
    specialist = {
        'schema_version': 1,
        'pack_id': 'auto-night-recovery',
        'name': 'Auto Night Recovery',
        'pack_type': 'processing-profile',
        'scene_hints': ['night'],
        'depends_on': ['auto-restoration-core'],
        'settings': {'denoise': 0.4, 'shadow_recovery': 0.4},
    }
    _write_pack(tmp_path, 'auto-restoration-core', core)
    _write_pack(tmp_path, 'auto-night-recovery', specialist)
    runtime = AutoProfileRuntime(tmp_path)
    image = np.full((48, 48, 3), 40, dtype=np.uint8)

    result, report = runtime.apply(image, 'Photo', 'night')

    assert report.selected == ['Auto Restoration Core', 'Auto Night Recovery']
    assert result.shape == image.shape
