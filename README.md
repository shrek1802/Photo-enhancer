# PhotoPerfect Batch AI

Windows 11 photo analysis, enhancement and manual repair application.

## GPU editions

Every release now provides two separate Windows builds:

- **PhotoPerfect-Batch-AI-AMD-DirectML.exe** — intended for AMD Radeon cards such as the RX 6900 XT. It uses ONNX Runtime DirectML and can also run on other Windows GPUs.
- **PhotoPerfect-Batch-AI-NVIDIA-CUDA.exe** — intended for NVIDIA cards such as the RTX 2060. It uses ONNX Runtime CUDA when the compatible NVIDIA driver/runtime is available.

Both editions retain the existing CPU photographic pipeline, so batch enhancement still works when GPU neural inference is unavailable.

## Neural model support

The app now contains a tiled ONNX inference engine for large photographs. It automatically prefers:

1. NVIDIA CUDA
2. Windows DirectML
3. ONNX CPU

To enable true neural 2× super-resolution, place a compatible NCHW RGB ONNX model here beside the EXE:

```text
models/super_resolution_x2.onnx
```

The release ZIP creates the `models` folder for you. If the model is missing or cannot load, the app safely uses high-quality Lanczos upscaling instead of failing. Model files are kept separate because they are large and their licences must be checked before redistribution.

## Current features

- Smart analysis and quality scoring for every photograph
- Blur, noise, exposure, highlight and contrast detection
- Portrait, event/christening, old-photo, landscape and low-light presets
- Face-aware exposure correction and natural portrait finishing
- Shadow and highlight recovery
- Conservative lens-flare correction
- Intelligent sharpening and noise reduction
- Horizon straightening
- Original, 2× and 4K output options
- Duplicate and near-duplicate detection
- Best-photo selection and CSV reports
- Before/after comparison viewer
- Manual brush repair for flare, scratches, shadows and unwanted objects
- Crop presets, undo, reset and separate repaired output
- Original photographs are never overwritten

## Download

Open **Releases** and choose the newest AMD DirectML or NVIDIA CUDA build for the computer being used.

## Important limitation

A neural model can reconstruct plausible detail, but no system can guarantee recovery of facial detail that was completely destroyed by flare, blur, clipping or missing pixels. Difficult results remain reviewable rather than silently replacing someone’s identity.
