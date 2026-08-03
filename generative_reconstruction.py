from __future__ import annotations

import base64
import json
import mimetypes
import os
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from photoperfect_intelligence import PhotoPerfectIntelligence, PostProcessValidation


DEFAULT_PROMPT = (
    'Faithfully reconstruct this photograph at professional photographic quality. '
    'Preserve the same people, facial identity, expression, pose, body proportions, '
    'clothing, visible text, objects, camera viewpoint and scene layout. Remove only '
    'compression, blur, noise, screenshot artefacts and poor lighting. Do not beautify, '
    'change age, change hairstyle, change clothing, add objects or invent a different face.'
)


@dataclass(frozen=True)
class ReconstructionSettings:
    model: str = 'gpt-image-1.5'
    quality: str = 'high'
    input_fidelity: str = 'high'
    output_format: str = 'png'
    size: str = 'auto'
    candidates: int = 3
    quality_target: str = 'Professional'
    prompt: str = DEFAULT_PROMPT


@dataclass(frozen=True)
class ReconstructionCandidate:
    image: np.ndarray
    validation: PostProcessValidation
    score: float


@dataclass
class ReconstructionResult:
    image: np.ndarray | None
    accepted: bool
    attempted: int
    accepted_candidates: int
    best_validation: PostProcessValidation | None = None
    messages: list[str] = field(default_factory=list)


class ReconstructionError(RuntimeError):
    pass


class OpenAIImageEditClient:
    """Small dependency-free client for the OpenAI Images edit endpoint.

    The API key is read at call time and is never written to disk by this class.
    A custom transport may be injected for deterministic tests.
    """

    endpoint = 'https://api.openai.com/v1/images/edits'

    def __init__(self, api_key: str | None = None, transport: Callable | None = None):
        self.api_key = api_key
        self.transport = transport

    @staticmethod
    def _multipart(fields: dict[str, str], image_path: Path) -> tuple[bytes, str]:
        boundary = '----PhotoPerfect' + secrets.token_hex(16)
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend([
                f'--{boundary}\r\n'.encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode('utf-8'), b'\r\n',
            ])
        mime = mimetypes.guess_type(image_path.name)[0] or 'application/octet-stream'
        chunks.extend([
            f'--{boundary}\r\n'.encode(),
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode(),
            f'Content-Type: {mime}\r\n\r\n'.encode(),
            image_path.read_bytes(), b'\r\n',
            f'--{boundary}--\r\n'.encode(),
        ])
        return b''.join(chunks), boundary

    def edit(self, image_path: Path, settings: ReconstructionSettings) -> list[np.ndarray]:
        key = (self.api_key or os.getenv('OPENAI_API_KEY', '')).strip()
        if not key:
            raise ReconstructionError('No OpenAI API key is configured.')
        fields = {
            'model': settings.model,
            'prompt': settings.prompt,
            'quality': settings.quality,
            'input_fidelity': settings.input_fidelity,
            'output_format': settings.output_format,
            'size': settings.size,
            'n': str(max(1, min(settings.candidates, 4))),
        }
        body, boundary = self._multipart(fields, image_path)
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method='POST',
            headers={
                'Authorization': f'Bearer {key}',
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'User-Agent': 'PhotoPerfect-Studio/5',
            },
        )
        try:
            if self.transport:
                payload = self.transport(request)
            else:
                with urllib.request.urlopen(request, timeout=300) as response:
                    payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise ReconstructionError(f'Image reconstruction request failed ({exc.code}): {detail}') from exc
        except OSError as exc:
            raise ReconstructionError(f'Image reconstruction request failed: {exc}') from exc
        decoded = json.loads(payload.decode('utf-8'))
        images: list[np.ndarray] = []
        for item in decoded.get('data', []):
            encoded = item.get('b64_json')
            if not encoded:
                continue
            raw = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
        if not images:
            raise ReconstructionError('The reconstruction service returned no usable image.')
        return images


class GenerativeReconstructionEngine:
    """Phase 5 candidate generation plus Phase 4 safety validation."""

    def __init__(self, client: OpenAIImageEditClient | None = None):
        self.client = client or OpenAIImageEditClient()
        self.intelligence = PhotoPerfectIntelligence()

    @staticmethod
    def _fit(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        h, w = reference.shape[:2]
        return cv2.resize(candidate, (w, h), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _candidate_score(validation: PostProcessValidation) -> float:
        penalty = validation.clipping_delta * 8.0 + max(0.0, validation.oversharpening_ratio - 1.0) * 0.03
        return validation.identity_similarity * 0.62 + validation.structure_similarity * 0.38 - penalty

    def reconstruct(self, source: Path, settings: ReconstructionSettings) -> ReconstructionResult:
        reference = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if reference is None:
            raise ReconstructionError(f'Could not read {source}')
        report = self.intelligence.inspect(reference, 'Generative Reconstruction', settings.quality_target)
        generated = self.client.edit(source, settings)
        accepted: list[ReconstructionCandidate] = []
        messages: list[str] = []
        best_failed: ReconstructionCandidate | None = None
        for index, image in enumerate(generated, start=1):
            fitted = self._fit(image, reference)
            validation = self.intelligence.validate(reference, fitted, report)
            candidate = ReconstructionCandidate(fitted, validation, self._candidate_score(validation))
            messages.append(
                f'Candidate {index}: identity={validation.identity_similarity:.3f}, '
                f'structure={validation.structure_similarity:.3f}, accepted={validation.accepted}'
            )
            if validation.accepted:
                accepted.append(candidate)
            elif best_failed is None or candidate.score > best_failed.score:
                best_failed = candidate
        if not accepted:
            return ReconstructionResult(
                image=None, accepted=False, attempted=len(generated), accepted_candidates=0,
                best_validation=best_failed.validation if best_failed else None,
                messages=messages + ['No candidate passed identity and structure validation.'],
            )
        best = max(accepted, key=lambda item: item.score)
        return ReconstructionResult(
            image=best.image, accepted=True, attempted=len(generated),
            accepted_candidates=len(accepted), best_validation=best.validation,
            messages=messages,
        )
