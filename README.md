# PhotoPerfect Studio

Professional AI photo enhancement and restoration for Windows 11.

## Simple input options

At the top of the app you can:

- Select one photo
- Select multiple photos
- Select a complete folder
- Drag and drop photos or a folder onto the window

Original photographs are never overwritten. Finished copies are saved inside a `Professionally Enhanced` folder beside the selected photos or inside the selected source folder.

## Enhancement modes

- **Auto Detect** — recommended; analyses every photograph and chooses the safest processing plan automatically
- **Auto Enhance** — lightly improves already-good photographs with better lighting, colour, contrast and natural sharpness
- **Auto Restore** — stronger repair for faded, damaged, blurred, noisy or compressed photographs
- **Family** — natural family-photo enhancement with protected faces
- **Portrait** — professional portrait lighting and colour
- **Celebrations** — consistent results for weddings, birthdays, christenings, parties and other occasions
- **Landscape** — scenery, nature and travel photographs
- **Low Light** — dark indoor, evening and night photographs
- **Screenshot Recovery** — compressed shared images and obvious phone or social-media borders
- **Advanced** — exposes the detailed processing controls

## Face Identity Lock

Face Identity Lock is enabled by default. It preserves real facial features and expressions, avoids generative face replacement and prefers a less dramatic faithful result when stronger processing could alter somebody's appearance.

## GPU editions

Every release provides two Windows builds:

- **PhotoPerfect-Studio-AMD-DirectML.exe** — for AMD Radeon cards such as the RX 6900 XT
- **PhotoPerfect-Studio-NVIDIA-CUDA.exe** — for NVIDIA cards such as the RTX 2060

Both editions retain CPU fallback processing.

## Current features

- Automatic photo analysis and quality scoring
- Per-photo processing decisions
- Shadow and highlight recovery
- White-balance and colour correction
- Conservative lens-flare correction
- Noise and compression cleanup
- Intelligent sharpening
- Screenshot-border detection
- Horizon straightening
- Original, 2× and 4K output
- Duplicate and near-duplicate detection
- Best-photo selection
- Review folders and CSV reports
- Before/after comparison
- Manual repair brush for marks, flare, scratches, shadows and unwanted objects
- Crop presets, undo and reset

## Optional model packs

The app works without external model files. The release ZIP includes a `models` folder reserved for versioned PhotoPerfect model packs. A future model manager will download, verify, install and update those packs automatically.

## Download

Open the repository's **Releases** section and download the newest AMD DirectML or NVIDIA CUDA edition.

## Important limitation

No software can guarantee recovery of detail that was never captured or was completely destroyed. PhotoPerfect Studio uses conservative processing and review checks rather than silently inventing a different face.
