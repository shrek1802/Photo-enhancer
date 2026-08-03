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

The engine can request these capabilities automatically:

- `jpeg_repair`
- `denoise`
- `deblur`
- `colour`
- `super_resolution`

Each capability is resolved through the installed model-pack manifests. The runtime selects CUDA, DirectML or CPU automatically, uses tiled ONNX inference, skips missing models safely and falls back to the built-in engine if inference fails.

## PhotoPerfect Engine Phase 4

Phase 4 adds the intelligence and safety layer needed before generative reconstruction is introduced.

It now measures:

- Blur and local sharpness
- Noise and JPEG compression
- Dynamic range
- Deep shadows and clipped highlights
- Colour cast
- Whole-image structural similarity
- Per-face size, brightness, contrast and sharpness

Every detected face receives a protection level:

- Normal
- High
- Maximum

Tiny or heavily blurred faces automatically receive stronger protection and a lower permitted change amount.

After processing, the candidate is checked for:

- Face-region similarity
- Whole-image structural similarity
- New shadow or highlight clipping
- Excessive sharpening

Unsafe same-geometry results are rejected and the original is retained.

The engine API supports five quality targets:

- Standard
- Professional — current default
- Studio
- Archive
- Museum

Higher targets use progressively stricter identity, structure, clipping and sharpening limits. The selectable interface for these targets will be exposed alongside the future reconstruction controls; current automatic processing uses Professional.

Phase 4 face analysis protects appearance. It does not identify people and does not infer age, ethnicity, gender or other demographic attributes.

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

GitHub Actions runs source compilation plus Phase 2, Phase 3 and Phase 4 regression tests before either Windows executable is built. Tests cover screenshot routing, monochrome restoration, model-pack validation, capability fallback, quality inspection and rejection of structurally unsafe results.

## Download

Open the repository's **Releases** section and download the newest AMD DirectML or NVIDIA CUDA edition.

## Important limitation

No software can guarantee recovery of detail that was never captured or was completely destroyed. PhotoPerfect Studio uses conservative processing, Face Identity Lock and quality validation rather than silently inventing a different face.
