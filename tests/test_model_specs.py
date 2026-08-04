import json
from pathlib import Path

import numpy as np

from capability_runtime import PhotoPerfectCapabilityRuntime
from model_specs import ModelSpec, load_model_spec


def test_loads_sidecar_model_spec(tmp_path: Path):
    model = tmp_path / 'deblur.onnx'
    model.write_bytes(b'placeholder')
    model.with_suffix('.onnx.json').write_text(json.dumps({
        'layout': 'nhwc',
        'colour_order': 'bgr',
        'input_range': 'minus_one_one',
        'output_range': 'minus_one_one',
        'tile_size': 384,
        'overlap': 48,
        'preserve_size': True,
        'blend_strength': 0.65,
    }), encoding='utf-8')

    spec = load_model_spec(model, 'deblur')
    assert spec.layout == 'nhwc'
    assert spec.colour_order == 'bgr'
    assert spec.tile_size == 384
    assert spec.preserve_size is True
    assert spec.blend_strength == 0.65


def test_default_spec_is_backwards_compatible(tmp_path: Path):
    model = tmp_path / 'super_resolution.onnx'
    model.write_bytes(b'placeholder')
    spec = load_model_spec(model, 'super_resolution')
    assert spec.layout == 'auto'
    assert spec.colour_order == 'rgb'
    assert spec.input_range == 'zero_one'


def test_minus_one_one_round_trip():
    image = np.array([[[0, 127, 255], [32, 64, 128]]], dtype=np.uint8)
    spec = ModelSpec(
        capability='denoise', colour_order='bgr',
        input_range='minus_one_one', output_range='minus_one_one'
    )
    encoded = PhotoPerfectCapabilityRuntime._encode_input(image, spec)
    decoded = PhotoPerfectCapabilityRuntime._decode_output(encoded, 'nhwc', spec)
    assert np.max(np.abs(decoded.astype(int) - image.astype(int))) <= 1


def test_tile_weights_fade_edges():
    weights = PhotoPerfectCapabilityRuntime._tile_weight(64, 64, 8)
    assert weights.shape == (64, 64, 1)
    assert weights[0, 0, 0] < weights[32, 32, 0]
    assert weights[32, 32, 0] == 1.0
