from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_CAPABILITIES = {
    'jpeg_repair', 'denoise', 'deblur', 'colour', 'lighting',
    'super_resolution', 'face_protect', 'face_recovery', 'inpaint',
}
ALLOWED_PROVIDERS = {
    'CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'
}
ALLOWED_LAYOUTS = {'auto', 'nchw', 'nhwc'}
ALLOWED_COLOURS = {'rgb', 'bgr', 'gray', 'grey', 'y'}
ALLOWED_RANGES = {'zero_one', 'minus_one_one', 'zero_255'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().lower()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def validate_spec(path: Path, capability: str) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        errors.append(f'Missing model specification: {path.name}')
        return errors
    try:
        spec = read_json(path)
    except Exception as exc:
        return [f'Invalid JSON in {path.name}: {exc}']

    if str(spec.get('layout', 'auto')).lower() not in ALLOWED_LAYOUTS:
        errors.append(f'{path.name}: unsupported layout')
    if str(spec.get('colour_order', 'rgb')).lower() not in ALLOWED_COLOURS:
        errors.append(f'{path.name}: unsupported colour_order')
    if str(spec.get('input_range', 'zero_one')).lower() not in ALLOWED_RANGES:
        errors.append(f'{path.name}: unsupported input_range')
    if str(spec.get('output_range', 'zero_one')).lower() not in ALLOWED_RANGES:
        errors.append(f'{path.name}: unsupported output_range')
    tile = int(spec.get('tile_size', 256))
    overlap = int(spec.get('overlap', 32))
    if tile < 64:
        errors.append(f'{path.name}: tile_size must be at least 64')
    if overlap < 0 or overlap >= tile // 2:
        errors.append(f'{path.name}: overlap must be non-negative and less than half tile_size')
    blend = float(spec.get('blend_strength', 1.0))
    if not 0.0 <= blend <= 1.0:
        errors.append(f'{path.name}: blend_strength must be between 0 and 1')
    if capability in {'face_recovery', 'face_protect'} and blend > 0.50:
        errors.append(f'{path.name}: face model blend_strength must not exceed 0.50')
    return errors


def validate_pack(root: Path, check_onnx: bool = True) -> list[str]:
    errors: list[str] = []
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        return ['Missing manifest.json']
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return [f'Invalid manifest.json: {exc}']

    files = manifest.get('files', [])
    if not files:
        errors.append('Neural pack manifest contains no model files')
        return errors

    try:
        import onnx
        import onnxruntime as ort
    except Exception as exc:
        if check_onnx:
            return [f'ONNX validation dependencies unavailable: {exc}']
        onnx = ort = None

    seen: set[str] = set()
    for item in files:
        capability = str(item.get('capability', ''))
        filename = str(item.get('filename', ''))
        if capability not in ALLOWED_CAPABILITIES:
            errors.append(f'Unsupported capability: {capability}')
        if capability in seen:
            errors.append(f'Duplicate capability: {capability}')
        seen.add(capability)
        if not filename:
            errors.append(f'{capability}: missing filename')
            continue
        model = (root / filename).resolve()
        if root.resolve() not in model.parents:
            errors.append(f'{capability}: unsafe filename path')
            continue
        if not model.exists():
            errors.append(f'{capability}: missing model file {filename}')
            continue
        expected_size = int(item.get('size', 0))
        if expected_size and model.stat().st_size != expected_size:
            errors.append(f'{capability}: file size mismatch')
        expected_hash = str(item.get('sha256', '')).lower()
        if not expected_hash:
            errors.append(f'{capability}: SHA-256 is required')
        elif sha256(model) != expected_hash:
            errors.append(f'{capability}: SHA-256 mismatch')

        providers = set(map(str, item.get('providers', [])))
        if not providers:
            errors.append(f'{capability}: provider list is required')
        unknown = providers - ALLOWED_PROVIDERS
        if unknown:
            errors.append(f'{capability}: unsupported providers {sorted(unknown)}')
        if 'CPUExecutionProvider' not in providers:
            errors.append(f'{capability}: CPU fallback is required')

        spec_path = model.with_suffix(model.suffix + '.json')
        if not spec_path.exists():
            spec_path = model.with_suffix('.json')
        errors.extend(validate_spec(spec_path, capability))

        if check_onnx and onnx is not None and ort is not None:
            try:
                onnx.checker.check_model(onnx.load(str(model)))
                session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
                if not session.get_inputs() or not session.get_outputs():
                    errors.append(f'{capability}: model has no usable input/output')
            except Exception as exc:
                errors.append(f'{capability}: ONNX validation failed: {exc}')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate a PhotoPerfect Auto neural pack')
    parser.add_argument('pack_directory', type=Path)
    parser.add_argument('--skip-onnx', action='store_true')
    args = parser.parse_args()
    errors = validate_pack(args.pack_directory, check_onnx=not args.skip_onnx)
    if errors:
        for error in errors:
            print(f'ERROR: {error}')
        return 1
    print(f'Validated neural pack: {args.pack_directory}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
