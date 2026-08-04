from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    capability: str
    layout: str = "auto"
    colour_order: str = "rgb"
    input_range: str = "zero_one"
    output_range: str = "zero_one"
    input_name: str = ""
    output_name: str = ""
    tile_size: int = 256
    overlap: int = 32
    preserve_size: bool = False
    blend_strength: float = 1.0

    @classmethod
    def from_dict(cls, capability: str, payload: dict[str, Any]) -> "ModelSpec":
        return cls(
            capability=capability,
            layout=str(payload.get("layout", "auto")).lower(),
            colour_order=str(payload.get("colour_order", "rgb")).lower(),
            input_range=str(payload.get("input_range", "zero_one")).lower(),
            output_range=str(payload.get("output_range", "zero_one")).lower(),
            input_name=str(payload.get("input_name", "")),
            output_name=str(payload.get("output_name", "")),
            tile_size=max(64, int(payload.get("tile_size", 256))),
            overlap=max(0, int(payload.get("overlap", 32))),
            preserve_size=bool(payload.get("preserve_size", False)),
            blend_strength=max(0.0, min(1.0, float(payload.get("blend_strength", 1.0)))),
        )


def load_model_spec(model_path: Path, capability: str) -> ModelSpec:
    """Load optional metadata beside an ONNX file.

    Supported locations, in priority order:
    1. ``model.onnx.json``
    2. ``model.json``
    3. ``model-specs.json`` in the pack directory, keyed by capability

    Models without metadata keep the legacy RGB 0..1 behaviour.
    """
    candidates = [
        model_path.with_suffix(model_path.suffix + ".json"),
        model_path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            return ModelSpec.from_dict(capability, payload)
        except Exception:
            continue

    for parent in (model_path.parent, model_path.parent.parent):
        catalogue = parent / "model-specs.json"
        if not catalogue.exists():
            continue
        try:
            payload = json.loads(catalogue.read_text(encoding="utf-8"))
            entry = payload.get(capability, {})
            if isinstance(entry, dict):
                return ModelSpec.from_dict(capability, entry)
        except Exception:
            continue
    return ModelSpec(capability=capability)
