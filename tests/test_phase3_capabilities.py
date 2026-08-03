from __future__ import annotations

import numpy as np

from capability_runtime import PhotoPerfectCapabilityRuntime
from photoperfect_engine import Inspection, RepairPlan


def inspection(**overrides) -> Inspection:
    values = dict(
        image_type='General Photograph',
        quality_score=55,
        is_screenshot=False,
        is_monochrome=False,
        is_low_resolution=False,
        face_count=0,
        smallest_face_ratio=0.0,
        blur_score=300.0,
        noise_score=4.0,
        compression_score=0.0,
        dark_fraction=0.0,
        highlight_fraction=0.0,
        contrast=50.0,
        problems=[],
    )
    values.update(overrides)
    return Inspection(**values)


def plan_for(item: Inspection) -> RepairPlan:
    return RepairPlan(
        name=item.image_type,
        stages=[],
        requested_mode='Auto Detect',
        confidence=90,
        inspection=item,
    )


def test_screenshot_requests_repair_and_resolution_capabilities(tmp_path) -> None:
    item = inspection(
        image_type='Screenshot Recovery',
        is_screenshot=True,
        is_low_resolution=True,
        compression_score=45.0,
        blur_score=90.0,
        noise_score=12.0,
    )
    runtime = PhotoPerfectCapabilityRuntime(tmp_path / 'models', enabled=False)
    requested = runtime.requested_capabilities(item, plan_for(item))
    assert requested == [
        'jpeg_repair', 'denoise', 'deblur', 'colour', 'super_resolution'
    ]


def test_monochrome_never_requests_colour_model(tmp_path) -> None:
    item = inspection(
        image_type='Black & White Restore',
        is_monochrome=True,
        blur_score=100.0,
        noise_score=10.0,
    )
    runtime = PhotoPerfectCapabilityRuntime(tmp_path / 'models', enabled=False)
    requested = runtime.requested_capabilities(item, plan_for(item))
    assert 'colour' not in requested
    assert 'deblur' in requested
    assert 'denoise' in requested


def test_missing_models_fall_back_without_changing_image(tmp_path) -> None:
    item = inspection(
        is_low_resolution=True,
        compression_score=30.0,
        blur_score=100.0,
        noise_score=11.0,
    )
    runtime = PhotoPerfectCapabilityRuntime(tmp_path / 'models', enabled=False)
    image = np.full((160, 240, 3), 127, dtype=np.uint8)
    result, report = runtime.apply(
        image, item, plan_for(item), allow_super_resolution=False
    )
    assert np.array_equal(result, image)
    assert report.applied == []
    assert set(report.missing) == {'jpeg_repair', 'denoise', 'deblur', 'colour'}


def test_good_monochrome_photo_requests_no_specialist_models(tmp_path) -> None:
    item = inspection(
        image_type='Black & White Restore',
        is_monochrome=True,
        quality_score=88,
        blur_score=400.0,
        noise_score=3.0,
    )
    runtime = PhotoPerfectCapabilityRuntime(tmp_path / 'models', enabled=False)
    assert runtime.requested_capabilities(item, plan_for(item)) == []
