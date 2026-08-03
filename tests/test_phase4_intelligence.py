from __future__ import annotations

import cv2
import numpy as np

from photoperfect_intelligence import PhotoPerfectIntelligence


def synthetic_portrait() -> np.ndarray:
    image = np.full((900, 700, 3), 145, dtype=np.uint8)
    cv2.ellipse(image, (350, 360), (125, 165), 0, 0, 360, (165, 180, 205), -1)
    cv2.circle(image, (310, 330), 12, (45, 45, 45), -1)
    cv2.circle(image, (390, 330), 12, (45, 45, 45), -1)
    cv2.ellipse(image, (350, 415), (45, 20), 0, 0, 180, (55, 55, 85), 4)
    return image


def test_phase4_report_contains_quality_metrics() -> None:
    engine = PhotoPerfectIntelligence()
    image = synthetic_portrait()
    report = engine.inspect(image, scene='Portrait', quality_target='Studio')
    assert report.quality_target == 'Studio'
    assert 0 <= report.quality_score <= 100
    assert report.dynamic_range >= 0
    assert report.blur_score >= 0
    assert report.compression_score >= 0


def test_identical_candidate_passes_museum_validation() -> None:
    engine = PhotoPerfectIntelligence()
    image = synthetic_portrait()
    report = engine.inspect(image, scene='Portrait', quality_target='Museum')
    validation = engine.validate(image, image.copy(), report)
    assert validation.accepted
    assert validation.identity_similarity == 1.0
    assert validation.structure_similarity > 0.999


def test_structurally_changed_candidate_is_rejected() -> None:
    engine = PhotoPerfectIntelligence()
    image = synthetic_portrait()
    report = engine.inspect(image, scene='Portrait', quality_target='Professional')
    changed = np.full_like(image, 245)
    validation = engine.validate(image, changed, report)
    assert not validation.accepted
    assert validation.reasons


def test_quality_targets_become_more_conservative() -> None:
    targets = PhotoPerfectIntelligence.TARGETS
    assert targets['Standard']['identity'] < targets['Professional']['identity']
    assert targets['Professional']['identity'] < targets['Studio']['identity']
    assert targets['Studio']['identity'] < targets['Archive']['identity']
    assert targets['Archive']['identity'] < targets['Museum']['identity']
    assert targets['Standard']['sharp_ratio'] > targets['Museum']['sharp_ratio']
