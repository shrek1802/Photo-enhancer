from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTextEdit, QVBoxLayout,
)

from ai_engine import app_directory
from model_manager import ModelPackError, PhotoPerfectModelManager
from version_info import AUTO_ESSENTIALS_VERSION


class AutoModelManagerDialog(QDialog):
    """Small, safe UI for viewing and installing Auto model packs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Auto Model Manager')
        self.resize(680, 480)
        self.manager = PhotoPerfectModelManager(app_directory() / 'models')

        layout = QVBoxLayout(self)
        heading = QLabel('Auto Essentials and specialist Auto model packs')
        heading.setStyleSheet('font-size: 18px; font-weight: 700;')
        layout.addWidget(heading)

        note = QLabel(
            'Model packs are installed separately from the app. Every archive is checked '
            'against its manifest and SHA-256 values before it is activated.'
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        layout.addWidget(self.status, 1)

        controls = QHBoxLayout()
        refresh = QPushButton('Refresh')
        install_archive = QPushButton('Install Pack ZIP')
        install_manifest = QPushButton('Install from Manifest URL')
        open_folder = QPushButton('Open Models Folder')
        refresh.clicked.connect(self.refresh)
        install_archive.clicked.connect(self.install_archive)
        install_manifest.clicked.connect(self.install_manifest_url)
        open_folder.clicked.connect(self.open_folder)
        controls.addWidget(refresh)
        controls.addWidget(install_archive)
        controls.addWidget(install_manifest)
        controls.addWidget(open_folder)
        layout.addLayout(controls)

        self.refresh()

    def refresh(self) -> None:
        lines = [f'Auto Essentials framework: v{AUTO_ESSENTIALS_VERSION}', '']
        packs = self.manager.installed_packs()
        if not packs:
            lines.append('No Auto model packs are installed yet.')
        for pack in packs:
            state = 'Ready' if pack.valid else 'Invalid'
            lines.append(f'{pack.manifest.name} v{pack.manifest.version} — {state}')
            lines.append(f'  Location: {pack.directory}')
            for model in pack.manifest.files:
                lines.append(f'  • {model.capability}: {model.filename}')
            for error in pack.errors:
                lines.append(f'  ERROR: {error}')
            lines.append('')
        capabilities = self.manager.installed_capabilities()
        lines.append(f'Available capabilities: {len(capabilities)}')
        for name, path in sorted(capabilities.items()):
            lines.append(f'  ✓ {name}: {path.name}')
        self.status.setPlainText('\n'.join(lines))

    def install_archive(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, 'Select Auto model pack', '', 'ZIP archives (*.zip)'
        )
        if not selected:
            return
        try:
            pack = self.manager.install_local_archive(Path(selected))
        except Exception as exc:
            QMessageBox.critical(self, 'Auto Model Manager', str(exc))
            return
        QMessageBox.information(
            self, 'Auto Model Manager',
            f'Installed {pack.manifest.name} v{pack.manifest.version}.',
        )
        self.refresh()

    def install_manifest_url(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        url, ok = QInputDialog.getText(
            self,
            'Install Auto model pack',
            'Manifest URL:',
        )
        if not ok or not url.strip():
            return
        try:
            pack = self.manager.install_from_manifest_url(url.strip())
        except (ModelPackError, OSError, ValueError) as exc:
            QMessageBox.critical(self, 'Auto Model Manager', str(exc))
            return
        QMessageBox.information(
            self, 'Auto Model Manager',
            f'Installed {pack.manifest.name} v{pack.manifest.version}.',
        )
        self.refresh()

    def open_folder(self) -> None:
        folder = self.manager.models_root.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception:
            QMessageBox.information(self, 'Models folder', str(folder))
