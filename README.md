# PhotoPerfect Studio

Professional automatic photo enhancement and restoration for Windows 11.

## Inputs

- Select one photo
- Select multiple photos
- Select a complete folder
- Drag and drop photos or a folder

Original photographs are never overwritten. Finished copies are saved in a `Professionally Enhanced` folder.

## Automatic modes

- **Auto Detect** — recommended; analyses each photograph and builds its own repair plan
- **Auto Enhance** — light professional finishing for already-good photographs
- **Auto Restore** — stronger repair for faded, blurred, noisy or compressed photographs
- **Auto Portrait** — identity-safe portrait lighting and face recovery
- **Auto Celebrations** — weddings, christenings, birthdays and family occasions
- **Auto Landscape**
- **Auto Low Light**
- **Auto Screenshot Recovery**
- **Advanced**

## Phase 2B

PhotoPerfect Studio v2.3 adds the real Auto Essentials runtime and model-management interface.

The Auto Engine can now request and run these independently updateable capabilities:

- Auto JPEG Recovery
- Auto Denoise
- Auto Deblur
- Auto Face Recovery
- Auto Face Protect
- Auto Lighting
- Auto Colour
- Auto Super Resolution

Whole-image models use tiled ONNX inference. Fixed-size ONNX inputs are also supported. Face models run only on feathered face regions and are blended conservatively to protect identity.

The app shows the detected image type, problems found, chosen pipeline, stages, installed models and before/after quality result for every photograph.

## Auto Model Manager

Open **Auto Model Manager** inside the app to:

- View installed model packs and capabilities
- Install a local model-pack ZIP
- Install from a manifest URL
- Validate file sizes and SHA-256 checksums
- Keep the previous working pack if an update fails
- Open the local model folder

Model packs are versioned separately from the Windows application, allowing restoration quality to improve without rebuilding the complete app.

The pack template is:

`model_packs/essentials-manifest.template.json`

No fake or unlicensed neural weights are bundled. Built-in restoration remains available when no external models are installed.

## Face Identity Lock

Face Identity Lock is enabled by default. It protects facial features and expressions, uses restrained face-model blending and rejects unsafe same-geometry results.

Flare cleanup now excludes faces, hair and nearby clothing so bright subjects are not accidentally inpainted or partially removed.

## GPU editions

Every release provides:

- **PhotoPerfect-Studio-AMD-DirectML.exe** — AMD Radeon cards such as RX 6900 XT
- **PhotoPerfect-Studio-NVIDIA-CUDA.exe** — NVIDIA cards such as RTX 2060

Both editions retain CPU fallback processing.

## Quality gates

GitHub Actions compiles the source and runs regression tests before building either executable. Tests cover automatic routing, model-pack installation, checksums, face-capability selection and identity-safe processing.

## Limitation

Reconstruction models infer plausible missing detail; they cannot prove what destroyed pixels originally contained. PhotoPerfect therefore favours faithful identity-preserving results over aggressive face invention.
