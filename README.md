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

Auto Detect now performs a proper inspection before processing. It measures screenshot and social-media UI likelihood, monochrome content, resolution, face size, JPEG blocking, blur, noise, exposure, highlights and contrast.

For each image it builds a dynamic repair plan, then tries multiple safe strategies:

- Gentle
- Balanced
- Strong

The engine scores every candidate and keeps the best accepted result. If every candidate scores worse, it keeps the original rather than silently saving a degraded image.

Dedicated processing is included for:

- Screenshot and social-media recovery
- Black-and-white tonal restoration
- Identity-safe portraits
- Low-light photographs
- Professional light polish for already-good images

## Face Identity Lock

Face Identity Lock is enabled by default. It preserves real facial features and expressions, avoids generative face replacement and prefers a less dramatic faithful result when stronger processing could alter somebody's appearance.

## Versioned model packs

Phase 2 adds a real model-pack system. Packs are separate from app releases and can be updated independently.

A pack manifest records:

- Pack ID and version
- Minimum app version
- Model capability names
- File names and sizes
- Supported execution providers
- SHA-256 checksums
- Archive download URL and checksum

Installations are staged in a temporary folder, checksum verified and validated before replacing the current pack. A failed installation automatically retains or restores the previous working pack.

The first pack template is:

`model_packs/essentials-manifest.template.json`

Supported capability names currently include:

- `jpeg_repair`
- `deblur`
- `denoise`
- `super_resolution`
- `face_protect`
- `colour`
- `inpaint`

No unlicensed or fake neural weights are bundled. The built-in photographic engine remains usable when no optional model pack is installed.

## GPU editions

Every release provides two Windows builds:

- **PhotoPerfect-Studio-AMD-DirectML.exe** — for AMD Radeon cards such as the RX 6900 XT
- **PhotoPerfect-Studio-NVIDIA-CUDA.exe** — for NVIDIA cards such as the RTX 2060

Both editions retain CPU fallback processing.

## Quality gates

GitHub Actions now runs source compilation and Phase 2 regression tests before either Windows executable is built. Tests cover screenshot routing, monochrome restoration, candidate pipeline retries and model-pack checksum validation.

## Download

Open the repository's **Releases** section and download the newest AMD DirectML or NVIDIA CUDA edition.

## Important limitation

No software can guarantee recovery of detail that was never captured or was completely destroyed. PhotoPerfect Studio uses conservative processing, Face Identity Lock and quality validation rather than silently inventing a different face.
