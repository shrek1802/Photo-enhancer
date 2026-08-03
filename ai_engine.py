from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class EngineStatus:
    available: bool
    provider: str
    display_name: str
    model_loaded: bool
    message: str


def app_directory() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


class NeuralEngine:
    """Optional ONNX restoration engine supporting CUDA, DirectML and CPU.

    Models are kept outside the EXE so releases stay manageable. Place a compatible
    NCHW RGB super-resolution ONNX model at models/super_resolution_x2.onnx.
    The engine uses tiled inference to avoid exhausting GPU memory on large photos.
    """

    MODEL_NAME = 'super_resolution_x2.onnx'

    def __init__(self, enabled: bool = True, tile_size: int = 256):
        self.enabled = enabled
        self.tile_size = max(64, int(tile_size))
        self.session = None
        self.input_name = ''
        self.output_name = ''
        self.scale = 2
        self.status = self._initialise()

    @property
    def model_path(self) -> Path:
        configured = os.environ.get('PHOTOPERFECT_MODEL_DIR')
        root = Path(configured) if configured else app_directory() / 'models'
        return root / self.MODEL_NAME

    def _initialise(self) -> EngineStatus:
        if not self.enabled:
            return EngineStatus(False, 'Disabled', 'Neural AI disabled', False,
                                'GPU neural processing is turned off.')
        try:
            import onnxruntime as ort
        except Exception:
            return EngineStatus(False, 'Unavailable', 'CPU photographic processing', False,
                                'ONNX Runtime is not included in this build.')

        available = ort.get_available_providers()
        preferred = [
            ('CUDAExecutionProvider', 'NVIDIA CUDA'),
            ('DmlExecutionProvider', 'AMD/NVIDIA DirectML'),
            ('CPUExecutionProvider', 'ONNX CPU'),
        ]
        chosen = next(((provider, name) for provider, name in preferred if provider in available), None)
        if chosen is None:
            return EngineStatus(False, 'Unavailable', 'CPU photographic processing', False,
                                f'No supported ONNX provider was found. Available: {available}')

        provider, display = chosen
        if not self.model_path.exists():
            return EngineStatus(True, provider, display, False,
                                f'{display} detected. Add {self.MODEL_NAME} to the models folder to enable neural upscaling.')
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                                providers=[provider, 'CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            shape_in = self.session.get_inputs()[0].shape
            shape_out = self.session.get_outputs()[0].shape
            if len(shape_in) == 4 and len(shape_out) == 4 and isinstance(shape_in[-1], int) and isinstance(shape_out[-1], int):
                self.scale = max(1, int(round(shape_out[-1] / max(shape_in[-1], 1))))
            return EngineStatus(True, provider, display, True,
                                f'{display} active with {self.MODEL_NAME}.')
        except Exception as exc:
            self.session = None
            return EngineStatus(True, provider, display, False,
                                f'{display} detected, but the neural model could not load: {exc}')

    def upscale(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.session is None:
            h, w = image_bgr.shape[:2]
            return cv2.resize(image_bgr, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        height, width = image_bgr.shape[:2]
        scale = self.scale or 2
        output = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
        weight = np.zeros((height * scale, width * scale, 1), dtype=np.float32)
        overlap = 16
        step = max(32, self.tile_size - overlap)

        for y in range(0, height, step):
            for x in range(0, width, step):
                y1 = min(y + self.tile_size, height)
                x1 = min(x + self.tile_size, width)
                tile = image_bgr[y:y1, x:x1]
                rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
                prediction = self.session.run([self.output_name], {self.input_name: tensor})[0]
                prediction = np.squeeze(prediction, axis=0)
                prediction = np.transpose(prediction, (1, 2, 0))
                prediction = np.clip(prediction, 0.0, 1.0)
                prediction = cv2.cvtColor((prediction * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)

                oy, ox = y * scale, x * scale
                ph, pw = prediction.shape[:2]
                output[oy:oy + ph, ox:ox + pw] += prediction
                weight[oy:oy + ph, ox:ox + pw] += 1.0

        return np.clip(output / np.maximum(weight, 1.0), 0, 255).astype(np.uint8)


def detect_engine() -> EngineStatus:
    return NeuralEngine(enabled=True).status
