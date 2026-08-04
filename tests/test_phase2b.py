from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from capability_runtime import PhotoPerfectCapabilityRuntime
from model_manager import PhotoPerfectModelManager
from photoperfect_engine import PhotoPerfectEngine
from version_info import APP_VERSION, AUTO_ENGINE_VERSION


def test_versions_are_phase_2b() -> None:
    assert APP_VERSION == '2.3.0'
    assert AUTO_ENGINE_VERSION == '2.2.0'


def test_requested_capabilities_include_face_recovery_for_soft_portrait() -> None:
    engine = PhotoPerfectEngine()
    image = np.full((900, 700, 3), 125, dtype=np.uint8)
    inspection = engine.inspect(image)
    inspection.face_count = 1
    inspection.smallest_face_ratio = 0.01
    inspection.blur_score = 80.0
    plan = engine.plan(inspection, 'Auto Detect')
    requested = PhotoPerfectCapabilityRuntime.requested_capabilities(inspection, plan)
    assert 'face_recovery' in requested
    assert 'face_protect' in requested


def test_local_model_pack_install_and_checksum_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        payload = b'not-a-real-model-but-valid-test-bytes'
        digest = hashlib.sha256(payload).hexdigest()
        pack = root / 'pack'
        (pack / 'models').mkdir(parents=True)
        (pack / 'models' / 'test.onnx').write_bytes(payload)
        manifest = {
            'schema_version': 2,
            'pack_id': 'auto-test',
            'name': 'Auto Test',
            'version': '1.0.0',
            'minimum_app_version': '2.3.0',
            'archive_url': '',
            'archive_sha256': '',
            'files': [{
                'capability': 'denoise',
                'filename': 'models/test.onnx',
                'sha256': digest,
                'size': len(payload),
                'providers': ['CPUExecutionProvider'],
                'required': True,
            }],
        }
        (pack / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        archive = root / 'auto-test.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            for path in pack.rglob('*'):
                if path.is_file():
                    zipped.write(path, path.relative_to(pack))

        manager = PhotoPerfectModelManager(root / 'models-root')
        installed = manager.install_local_archive(archive)
        assert installed.valid
        assert manager.capability_path('denoise') is not None


def test_face_relight_mask_keeps_image_shape() -> None:
    from enhancer import PhotoEnhancer, EnhanceOptions

    image = np.full((400, 400, 3), 80, dtype=np.uint8)
    result = PhotoEnhancer._face_relight(image, [(120, 100, 120, 140)])
    assert result.shape == image.shape
    assert result.dtype == np.uint8
