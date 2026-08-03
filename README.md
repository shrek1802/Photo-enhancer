# PhotoPerfect Batch AI

Windows 11 desktop application for analysing, repairing and professionally enhancing complete folders of photographs while preserving every original.

## Current version: 0.3.0

### Smart batch processing

- Analyses every photograph before processing it
- Generates a quality score from 1–100
- Detects soft focus, severe blur, darkness, clipped highlights, noise and low contrast
- Classifies portraits, group portraits, events, landscapes, low-light and old/faded photos
- Smart Auto chooses Natural, Strong or Maximum enhancement separately for each image
- Produces `Photo Analysis Report.csv` with measurements and review reasons
- Detects exact and visually similar duplicates
- Marks the strongest image in each duplicate group as `_BEST`

### Automatic enhancement

- Automatic white balance, exposure and colour correction
- Selective shadow lifting and highlight recovery
- Conservative lens-flare and coloured-glare reduction
- Noise and compression cleanup
- Face-aware exposure correction and natural portrait finishing
- Photo-specific intelligent sharpening
- Automatic correction of slightly crooked horizons
- Original-size, 2× upscale or 4K-long-edge output

### Processing presets

- Smart Auto
- Event / Christening
- Professional Portrait
- Old Photo Restoration
- Landscape
- Night / Low Light

### Manual Repair Studio

The main app now contains a separate hands-on editor for difficult photographs:

- Open one photograph without changing the original
- Before/after split comparison slider
- Mouse-wheel zoom up to 600%
- Adjustable repair brush
- Brush over lens flare, scratches, unwanted objects or marks and inpaint them
- Brush over unwanted shadows and selectively lift them
- Selective softening for damaged or noisy areas
- Automatic dust and bright-spot detection
- Undo history and complete reset
- Centre crop presets for 1:1, 4:5, 3:2 and 16:9
- Save a separate repaired JPG or PNG
- Check the newest GitHub Release from inside the editor

### Output folders

- `Professionally Enhanced`
- `Review Needed`
- `Duplicate Review`
- `Best Photos`
- `Photo Analysis Report.csv`

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

## Remaining neural-AI capability packs

- ONNX/DirectML neural super-resolution and deblurring for AMD GPUs
- Dedicated scratch, tear and stain segmentation for badly damaged scans
- Neural black-and-white photograph colourisation
- Face embeddings for searchable people albums
- Subject-aware crop positioning rather than centre-only crops
- Signed one-click installer/updater

These require separately distributed AI model files. They are not represented by inactive or misleading buttons in the current build.

## Repair limitation

Large lens flare, deep shadow or severe blur covering important facial detail cannot always be reconstructed faithfully because the original pixels may not contain that detail. The app makes conservative repairs and flags difficult images instead of silently inventing a different face.
