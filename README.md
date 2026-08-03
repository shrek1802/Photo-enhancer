# PhotoPerfect Studio

Professional photo enhancement and restoration for Windows 11.

## Simple input options

At the top of the app you can:

- Select one photo
- Select multiple photos
- Select a complete folder
- Drag and drop photos or a folder onto the window

Original photographs are never overwritten. Finished copies are saved inside a `Professionally Enhanced` folder beside the selected photos or inside the selected source folder.

## Enhancement modes

- **Auto Detect** — recommended; analyses every photograph and chooses the safest processing plan automatically
- **Auto Enhance** — lightly improves already-good photographs
- **Auto Restore** — stronger repair for faded, blurred, noisy or compressed photographs
- **Family**
- **Portrait**
- **Celebrations**
- **Landscape**
- **Low Light**
- **Screenshot Recovery**
- **Advanced**

## PhotoPerfect Engine Phase 2

Auto Detect performs a proper inspection before processing. It measures screenshot and social-media UI likelihood, monochrome content, resolution, face size, JPEG blocking, blur, noise, exposure, highlights and contrast.

For each image it builds a dynamic repair plan, tries Gentle, Balanced and Strong safe strategies, then keeps the highest-scoring accepted result. If every candidate scores worse, the original is retained.

## PhotoPerfect Engine Phase 3

Phase 3 adds a real neural capability layer between the repair planner and the model packs.

The engine can now request these capabilities automatically:

- `jpeg_repair`
- `denoise`
- `deblur`
- `colour`
- `super_resolution`

Each capability is resolved through the installed model-pack manifests. The runtime:

- Selects CUDA, DirectML or CPU automatically
- Uses tiled ONNX inference for large photographs
- Supports common NCHW and NHWC image-model layouts
- Keeps model files outside the EXE so they can update independently
- Skips missing or incompatible models safely
- Falls back to the built-in Phase 2 photographic engine if inference fails
- Reports which capabilities were requested, applied or missing

Super-resolution remains connected to the output-size controls so an installed model cannot unexpectedly change image dimensions.

`face_protect` and `inpaint` remain reserved specialist capabilities. They are not run over whole photographs until a safe region-based implementation and suitable validated models are available.

## Face Identity Lock

Face Identity Lock is enabled by default. It preserves real facial features and expressions, avoids generative face replacement and prefers a less dramatic faithful result when stronger processing could alter somebody's appearance.

## Versioned model packs

Packs are separate from app releases and can be updated independently. A manifest records the pack version, minimum app version, capabilities, provider support, file sizes, SHA-256 checksums and archive details.

Installations are staged, checksum verified and validated before replacing the current pack. A failed installation retains or restores the previous working pack.

The first pack template is:

`model_packs/essentials-manifest.template.json`

No unlicensed, fake or empty neural weights are bundled. Genuine PhotoPerfect model packs will be released separately after model compatibility, output quality and licensing have been verified.

## GPU editions

Every release provides two Windows builds:

- **PhotoPerfect-Studio-AMD-DirectML.exe** — for AMD Radeon cards such as the RX 6900 XT
- **PhotoPerfect-Studio-NVIDIA-CUDA.exe** — for NVIDIA cards such as the RTX 2060

Both editions retain CPU fallback processing.

## Quality gates

GitHub Actions runs source compilation plus Phase 2 and Phase 3 regression tests before either Windows executable is built. Tests cover screenshot routing, monochrome restoration, candidate retries, model-pack checksum validation, capability planning and safe missing-model fallback.

## Download

Open the repository's **Releases** section and download the newest AMD DirectML or NVIDIA CUDA edition.

## Important limitation

No software can guarantee recovery of detail that was never captured or was completely destroyed. PhotoPerfect Studio uses conservative processing, Face Identity Lock and quality validation rather than silently inventing a different face.
