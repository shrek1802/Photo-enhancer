from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelFile:
    capability: str
    filename: str
    sha256: str
    size: int
    providers: tuple[str, ...] = ()
    required: bool = True


@dataclass(frozen=True)
class ModelPackManifest:
    pack_id: str
    name: str
    version: str
    minimum_app_version: str
    archive_url: str
    archive_sha256: str
    files: tuple[ModelFile, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'ModelPackManifest':
        files = tuple(
            ModelFile(
                capability=str(item['capability']),
                filename=str(item['filename']),
                sha256=str(item.get('sha256', '')).lower(),
                size=int(item.get('size', 0)),
                providers=tuple(str(value) for value in item.get('providers', [])),
                required=bool(item.get('required', True)),
            )
            for item in payload.get('files', [])
        )
        return cls(
            pack_id=str(payload['pack_id']),
            name=str(payload['name']),
            version=str(payload['version']),
            minimum_app_version=str(payload.get('minimum_app_version', '0.0.0')),
            archive_url=str(payload.get('archive_url', '')),
            archive_sha256=str(payload.get('archive_sha256', '')).lower(),
            files=files,
        )


@dataclass
class InstalledPack:
    manifest: ModelPackManifest
    directory: Path
    valid: bool
    errors: list[str] = field(default_factory=list)


class ModelPackError(RuntimeError):
    pass


class PhotoPerfectModelManager:
    """Secure installer and registry for independently versioned Auto model packs."""

    def __init__(self, models_root: Path | str = 'models') -> None:
        self.models_root = Path(models_root)
        self.packs_root = self.models_root / 'packs'
        self.packs_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest().lower()

    @staticmethod
    def load_manifest(path: Path | str) -> ModelPackManifest:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return ModelPackManifest.from_dict(payload)

    @staticmethod
    def fetch_manifest(url: str, timeout: int = 30) -> ModelPackManifest:
        request = urllib.request.Request(url, headers={'User-Agent': 'PhotoPerfect-Studio'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8'))
        return ModelPackManifest.from_dict(payload)

    def pack_directory(self, pack_id: str) -> Path:
        return self.packs_root / pack_id

    def installed_manifest_path(self, pack_id: str) -> Path:
        return self.pack_directory(pack_id) / 'manifest.json'

    def installed(self, pack_id: str) -> InstalledPack | None:
        manifest_path = self.installed_manifest_path(pack_id)
        if not manifest_path.exists():
            return None
        try:
            manifest = self.load_manifest(manifest_path)
        except Exception as exc:
            return InstalledPack(
                manifest=ModelPackManifest(pack_id, pack_id, '0.0.0', '0.0.0', '', ''),
                directory=manifest_path.parent,
                valid=False,
                errors=[f'Invalid installed manifest: {exc}'],
            )
        return self.validate(manifest, manifest_path.parent)

    def installed_packs(self) -> list[InstalledPack]:
        packs: list[InstalledPack] = []
        for manifest_path in sorted(self.packs_root.glob('*/manifest.json')):
            try:
                manifest = self.load_manifest(manifest_path)
                packs.append(self.validate(manifest, manifest_path.parent))
            except Exception as exc:
                fallback = ModelPackManifest(
                    manifest_path.parent.name, manifest_path.parent.name,
                    '0.0.0', '0.0.0', '', '',
                )
                packs.append(InstalledPack(fallback, manifest_path.parent, False, [str(exc)]))
        return packs

    def validate(self, manifest: ModelPackManifest, directory: Path) -> InstalledPack:
        errors: list[str] = []
        for model in manifest.files:
            path = directory / model.filename
            if not path.exists():
                if model.required:
                    errors.append(f'Missing required model: {model.filename}')
                continue
            actual_size = path.stat().st_size
            if model.size and actual_size != model.size:
                errors.append(f'Wrong size for {model.filename}: expected {model.size}, got {actual_size}')
                continue
            if model.sha256 and self.sha256(path) != model.sha256:
                errors.append(f'Checksum failed for {model.filename}')
        return InstalledPack(manifest, directory, not errors, errors)

    @staticmethod
    def _safe_extract(archive: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive) as zipped:
            root = destination.resolve()
            for member in zipped.infolist():
                target = (destination / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ModelPackError('Unsafe path found in model archive')
            zipped.extractall(destination)

    @staticmethod
    def _download(url: str, destination: Path, timeout: int = 300) -> None:
        request = urllib.request.Request(url, headers={'User-Agent': 'PhotoPerfect-Studio'})
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open('wb') as output:
            shutil.copyfileobj(response, output)

    def _activate_candidate(self, manifest: ModelPackManifest, candidate: Path) -> InstalledPack:
        manifest_path = candidate / 'manifest.json'
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(self.to_dict(manifest), indent=2), encoding='utf-8')

        validated = self.validate(manifest, candidate)
        if not validated.valid:
            raise ModelPackError('; '.join(validated.errors))

        target = self.pack_directory(manifest.pack_id)
        backup = target.with_name(f'{target.name}.backup')
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        try:
            shutil.copytree(candidate, target)
            final = self.validate(manifest, target)
            if not final.valid:
                raise ModelPackError('; '.join(final.errors))
        except Exception:
            if target.exists():
                shutil.rmtree(target)
            if backup.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return self.validate(manifest, target)

    def install(self, manifest: ModelPackManifest) -> InstalledPack:
        if not manifest.archive_url:
            raise ModelPackError('The model pack manifest has no archive URL')
        with tempfile.TemporaryDirectory(prefix='photoperfect-models-') as temporary:
            root = Path(temporary)
            archive = root / 'pack.zip'
            extracted = root / 'extracted'
            extracted.mkdir()
            self._download(manifest.archive_url, archive)
            if manifest.archive_sha256 and self.sha256(archive) != manifest.archive_sha256:
                raise ModelPackError('Model pack archive checksum verification failed')
            self._safe_extract(archive, extracted)
            candidate = extracted / manifest.pack_id if (extracted / manifest.pack_id).is_dir() else extracted
            return self._activate_candidate(manifest, candidate)

    def install_local_archive(self, archive: Path | str) -> InstalledPack:
        archive_path = Path(archive)
        if not archive_path.exists():
            raise ModelPackError('The selected model pack does not exist')
        with tempfile.TemporaryDirectory(prefix='photoperfect-local-models-') as temporary:
            extracted = Path(temporary) / 'extracted'
            extracted.mkdir()
            self._safe_extract(archive_path, extracted)
            manifests = list(extracted.rglob('manifest.json'))
            if len(manifests) != 1:
                raise ModelPackError('The archive must contain exactly one manifest.json')
            manifest = self.load_manifest(manifests[0])
            return self._activate_candidate(manifest, manifests[0].parent)

    def install_from_manifest_url(self, url: str) -> InstalledPack:
        return self.install(self.fetch_manifest(url))

    def remove(self, pack_id: str) -> None:
        target = self.pack_directory(pack_id)
        if target.exists():
            shutil.rmtree(target)

    def capability_path(self, capability: str) -> Path | None:
        for installed in self.installed_packs():
            if not installed.valid:
                continue
            for model in installed.manifest.files:
                if model.capability == capability:
                    path = installed.directory / model.filename
                    if path.exists():
                        return path
        legacy_names = {
            'super_resolution': 'super_resolution_x2.onnx',
            'jpeg_repair': 'jpeg_repair.onnx',
            'deblur': 'deblur.onnx',
            'denoise': 'denoise.onnx',
            'face_protect': 'face_protect.onnx',
            'face_recovery': 'face_recovery.onnx',
            'colour': 'colour.onnx',
            'lighting': 'lighting.onnx',
            'inpaint': 'inpaint.onnx',
        }
        name = legacy_names.get(capability)
        candidate = self.models_root / name if name else None
        return candidate if candidate and candidate.exists() else None

    def installed_capabilities(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for capability in (
            'super_resolution', 'jpeg_repair', 'deblur', 'denoise',
            'face_protect', 'face_recovery', 'colour', 'lighting', 'inpaint',
        ):
            path = self.capability_path(capability)
            if path:
                result[capability] = path
        return result

    @staticmethod
    def to_dict(manifest: ModelPackManifest) -> dict[str, Any]:
        return {
            'schema_version': 2,
            'pack_id': manifest.pack_id,
            'name': manifest.name,
            'version': manifest.version,
            'minimum_app_version': manifest.minimum_app_version,
            'archive_url': manifest.archive_url,
            'archive_sha256': manifest.archive_sha256,
            'files': [
                {
                    'capability': model.capability,
                    'filename': model.filename,
                    'sha256': model.sha256,
                    'size': model.size,
                    'providers': list(model.providers),
                    'required': model.required,
                }
                for model in manifest.files
            ],
        }
