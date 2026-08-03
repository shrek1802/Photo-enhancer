from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from generative_reconstruction import (
    GenerativeReconstructionEngine, OpenAIImageEditClient, ReconstructionSettings,
)


def encode_png(image: np.ndarray) -> str:
    ok, data = cv2.imencode('.png', image)
    assert ok
    return base64.b64encode(data.tobytes()).decode('ascii')


def portrait() -> np.ndarray:
    image = np.full((700, 520, 3), 120, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (519, 80), (25, 25, 25), -1)
    cv2.ellipse(image, (260, 310), (105, 145), 0, 0, 360, (165, 180, 205), -1)
    cv2.circle(image, (225, 285), 10, (35, 35, 35), -1)
    cv2.circle(image, (295, 285), 10, (35, 35, 35), -1)
    cv2.ellipse(image, (260, 365), (38, 16), 0, 0, 180, (55, 55, 80), 3)
    return image


def test_client_parses_base64_candidates(tmp_path: Path) -> None:
    source = tmp_path / 'source.png'
    image = portrait()
    cv2.imwrite(str(source), image)

    def transport(_request):
        return json.dumps({'data': [{'b64_json': encode_png(image)}]}).encode()

    client = OpenAIImageEditClient('test-key', transport=transport)
    candidates = client.edit(source, ReconstructionSettings(candidates=1))
    assert len(candidates) == 1
    assert candidates[0].shape == image.shape


def test_safe_candidate_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / 'source.png'
    image = portrait()
    cv2.imwrite(str(source), image)

    class Client:
        def edit(self, _path, _settings):
            return [image.copy()]

    result = GenerativeReconstructionEngine(Client()).reconstruct(
        source, ReconstructionSettings(candidates=1, quality_target='Professional')
    )
    assert result.accepted
    assert result.image is not None
    assert result.accepted_candidates == 1


def test_identity_drift_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / 'source.png'
    image = portrait()
    cv2.imwrite(str(source), image)
    changed = np.full_like(image, 245)

    class Client:
        def edit(self, _path, _settings):
            return [changed]

    result = GenerativeReconstructionEngine(Client()).reconstruct(
        source, ReconstructionSettings(candidates=1, quality_target='Professional')
    )
    assert not result.accepted
    assert result.image is None
    assert result.accepted_candidates == 0
