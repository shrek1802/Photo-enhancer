# Optional AI Models

PhotoPerfect Batch AI runs without these files using conservative OpenCV restoration. Adding compatible ONNX models enables the stronger automatic neural stages.

Place models in this folder using these exact names:

- `super_resolution_x2.onnx` — 2x general photo super-resolution
- `deblur.onnx` — motion/defocus deblurring
- `denoise.onnx` — photographic denoising and JPEG cleanup
- `face_restore.onnx` — identity-preserving face restoration
- `colourise.onnx` — black-and-white colourisation
- `inpaint.onnx` — large-mask object, tear and flare removal

## Expected simple image-model format

The automatic runner supports models with one NCHW float RGB input and one image output. Input pixels are normalised to 0–1. Output pixels are expected in 0–1. Dynamic image dimensions are recommended.

Models with specialised inputs, masks, latent controls or multiple outputs require a dedicated adapter before they can be enabled safely.

## GPU editions

- AMD/Universal build: ONNX Runtime DirectML, with CPU fallback.
- NVIDIA build: ONNX Runtime CUDA, with CPU fallback.

The application checks model compatibility at startup. A missing, corrupt or incompatible model is skipped instead of stopping normal photo processing.

Only use model files whose licences permit your intended use and redistribution. Model weights are not committed to this repository.
