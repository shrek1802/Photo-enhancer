from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from model_manager import ModelPackManifest, PhotoPerfectModelManager
from photoperfect_engine import PhotoPerfectEngine


def synthetic_screenshot() -> np.ndarray:
    image = np.full((1600, 800, 3), 210, dtype=np.uint8)
    image[90:1260] = (90, 110, 130)
    cv2.rectangle(image, (0, 0), (799, 90), (245, 245, 245), -1)
    cv2.putText(image, 'Facebook', (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)
    cv2.rectangle(image, (0, 1260), (799, 1599), (250, 250, 250), -1)
    for x in (100, 300, 500, 700):
        cv2.circle(image, (x, 1420), 28, (60, 60, 60), 3)
    return image


def synthetic_monochrome() -> np.ndarray:
    base = np.tile(np.linspace(65, 170, 900, dtype=np.uint8), (1200, 1))
    cv2.circle(base, (450, 470), 180, 120, -1)
    cv2.rectangle(base, (260, 660), (640, 1050), 105, -1)
    noise = np.random.default_rng(42).normal(0, 5, base.shape)
    gray = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_auto_detect_routes_screenshot_and_retries() -> None:
    engine = PhotoPerfectEngine()
    image = synthetic_screenshot()
    result, plan, validation = engine.process(image, 'Auto Detect')
    assert plan.name == 'Screenshot Recovery'
    assert plan.inspection.is_screenshot
    assert validation.attempts >= 2
    assert validation.selected_strategy in {'gentle', 'balanced', 'strong'}
    assert result.shape[0] < image.shape[0]


def test_monochrome_uses_dedicated_restore_without_colour_cast() -> None:
    engine = PhotoPerfectEngine()
    image = synthetic_monochrome()
    result, plan, validation = engine.process(image, 'Auto Detect')
    assert plan.name == 'Black & White Restore'
    assert plan.inspection.is_monochrome
    b, g, r = cv2.split(result)
    assert np.max(np.abs(b.astype(np.int16) - g.astype(np.int16))) == 0
    assert np.max(np.abs(g.astype(np.int16) - r.astype(np.int16))) == 0
    assert validation.accepted


def test_good_photo_uses_light_polish() -> None:
    engine = PhotoPerfectEngine()
    rng = np.random.default_rng(7)
    image = rng.integers(40, 220, size=(1200, 1600, 3), dtype=np.uint8)
    inspection = engine.inspect(image)
    plan = engine.plan(inspection, 'Auto Enhance')
    assert plan.name == 'Auto Enhance'
    assert 'professional colour and lighting' in plan.stages


def test_model_manifest_validation(tmp_path: Path) -> None:
    models_root = tmp_path / 'models'
    manager = PhotoPerfectModelManager(models_root)
    pack = models_root / 'packs' / 'essentials'
    pack.mkdir(parents=True)
    model = pack / 'super_resolution_x2.onnx'
    model.write_bytes(b'test-model')
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    payload = {
        'pack_id': 'essentials',
        'name': 'Essentials',
        'version': '1.0.0',
        'minimum_app_version': '2.0.0',
        'archive_url': '',
        'archive_sha256': '',
        'files': [{
            'capability': 'super_resolution',
            'filename': model.name,
            'sha256': digest,
            'size': model.stat().st_size,
            'providers': ['CPUExecutionProvider'],
            'required': True,
        }],
    }
    (pack / 'manifest.json').write_text(json.dumps(payload), encoding='utf-8')
    installed = manager.installed('essentials')
    assert installed is not None and installed.valid
    assert manager.capability_path('super_resolution') == model


def test_model_manager_rejects_bad_checksum(tmp_path: Path) -> None:
    manager = PhotoPerfectModelManager(tmp_path / 'models')
    pack = manager.packs_root / 'bad'
    pack.mkdir(parents=True)
    (pack / 'model.onnx').write_bytes(b'wrong')
    manifest = ModelPackManifest.from_dict({
        'pack_id': 'bad',
        'name': 'Bad',
        'version': '1.0.0',
        'files': [{
            'capability': 'deblur',
            'filename': 'model.onnx',
            'sha256': '0' * 64,
            'size': 5,
            'required': True,
        }],
    })
    validated = manager.validate(manifest, pack)
    assert not validated.valid
    assert any('Checksum failed' in error for error in validated.errors)
