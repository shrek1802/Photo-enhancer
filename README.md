# PhotoPerfect Batch AI

Windows 11 desktop application for analysing, repairing and professionally enhancing complete folders of photographs while preserving every original.

## Current version: 0.2.0

### Smart processing

- Analyses every photograph before processing it
- Generates a quality score from 1–100
- Detects soft focus, severe blur, darkness, clipped highlights, noise and low contrast
- Classifies portraits, group portraits, events, landscapes, low-light and old/faded photos
- Smart Auto chooses Natural, Strong or Maximum enhancement separately for each image
- Produces `Photo Analysis Report.csv` with the measurements and review reasons

### Enhancement

- Automatic white balance, exposure and colour correction
- Selective shadow lifting and highlight recovery
- Conservative lens-flare and coloured-glare reduction
- Noise and compression cleanup
- Face-aware exposure correction and natural portrait finishing
- Photo-specific intelligent sharpening
- Automatic correction of slightly crooked horizons
- Original-size, 2× upscale or 4K-long-edge output

### Presets

- Smart Auto
- Event / Christening
- Professional Portrait
- Old Photo Restoration
- Landscape
- Night / Low Light

### Sorting and review

- Processes complete folders including subfolders
- Keeps every original photograph untouched
- Creates `Professionally Enhanced` inside the selected folder
- Places difficult photographs and explanation files into `Review Needed`
- Detects exact and visually similar duplicates
- Places duplicate groups into `Duplicate Review`
- Marks the highest-scoring image in each group as `_BEST`
- Copies the enhanced winners into `Best Photos`

## Downloading the Windows EXE

Open the repository's **Releases** section and download the newest:

- `PhotoPerfect-Batch-AI.exe`
- or `PhotoPerfect-Batch-AI-Windows.zip`

A new release is built automatically after changes to the main branch.

## Local development

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Build locally

```powershell
pyinstaller --noconfirm --clean --onefile --windowed --name "PhotoPerfect-Batch-AI" --collect-all cv2 --collect-all PySide6 app.py
```

The executable is written to `dist/PhotoPerfect-Batch-AI.exe`.

## Planned capability packs

- Before/after comparison viewer with linked zoom
- Manual repair brush and object removal
- Neural super-resolution, deblurring and face restoration using ONNX/DirectML for AMD GPUs
- Scratch, tear and stain restoration for scanned photographs
- Colourisation controls for black-and-white photos
- Face grouping and searchable people albums
- Automatic crop presets for prints and social media
- One-click updater using GitHub Releases

## Repair limitation

Large lens flare, deep shadow or severe blur covering important facial detail cannot always be reconstructed faithfully because the original pixels may not contain that detail. The app makes conservative repairs and flags difficult images instead of silently inventing a different face.
