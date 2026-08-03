# PhotoPerfect Batch AI

Windows 11 desktop application for batch-enhancing complete folders of photographs while preserving the originals.

## Features

- Select one folder and process all supported photos, including subfolders
- Creates `Professionally Enhanced` inside the selected folder
- Keeps every original photograph untouched
- Automatic white balance, exposure and colour correction
- Shadow lifting and highlight recovery
- Conservative lens-flare and coloured-glare reduction
- Noise and compression cleanup
- Face-aware exposure correction
- Photo-specific intelligent sharpening
- Original-size, 2× upscale or 4K-long-edge output
- Places difficult results in `Review Needed`
- Fully local processing

## Downloading the Windows EXE

Open the repository's **Actions** tab, select **Build Windows EXE**, open the newest successful run, and download the `PhotoPerfect-Batch-AI-Windows` artifact.

The artifact contains:

- `PhotoPerfect-Batch-AI.exe`
- `PhotoPerfect-Batch-AI-Windows.zip`

A version tag such as `v0.1.0` also creates a normal GitHub Release with the EXE attached.

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

## Important limitation

Large lens flare or deep shadow covering important facial detail cannot always be reconstructed faithfully because the original pixels may not contain that detail. The app uses conservative repairs and flags difficult images rather than inventing a different face.
