from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)

from generative_reconstruction import (
    GenerativeReconstructionEngine, OpenAIImageEditClient, ReconstructionSettings,
)


class ReconstructionSignals(QObject):
    finished = Signal(str, str)
    failed = Signal(str)


class ReconstructionWorker(threading.Thread):
    def __init__(self, source: Path, destination: Path, api_key: str, settings: ReconstructionSettings, signals: ReconstructionSignals):
        super().__init__(daemon=True)
        self.source = source
        self.destination = destination
        self.api_key = api_key
        self.settings = settings
        self.signals = signals

    def run(self) -> None:
        try:
            engine = GenerativeReconstructionEngine(OpenAIImageEditClient(self.api_key))
            result = engine.reconstruct(self.source, self.settings)
            if not result.accepted or result.image is None:
                self.signals.failed.emit('\n'.join(result.messages))
                return
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.destination), result.image, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                raise OSError(f'Could not write {self.destination}')
            details = '\n'.join(result.messages)
            self.signals.finished.emit(str(self.destination), details)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class ReconstructionWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source: Path | None = None
        self.worker: ReconstructionWorker | None = None
        self.signals = ReconstructionSignals()
        self.signals.finished.connect(self.on_finished)
        self.signals.failed.connect(self.on_failed)
        self.setWindowTitle('PhotoPerfect Generative Reconstruction')
        self.resize(720, 620)
        self.build_ui()

    def build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        title = QLabel('Generative Reconstruction')
        title.setStyleSheet('font-size: 25px; font-weight: 700;')
        info = QLabel(
            'For severely blurred, compressed or damaged photographs. This feature creates '
            'plausible missing detail, then rejects results that fail identity or structure checks.'
        )
        info.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(info)

        source_box = QGroupBox('1. Photograph')
        source_layout = QHBoxLayout(source_box)
        self.source_label = QLabel('No photograph selected')
        choose = QPushButton('Select Photograph')
        choose.clicked.connect(self.select_photo)
        source_layout.addWidget(self.source_label, 1)
        source_layout.addWidget(choose)
        layout.addWidget(source_box)

        settings_box = QGroupBox('2. Reconstruction settings')
        form = QFormLayout(settings_box)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText('OpenAI API key (used for this session only)')
        self.api_key.setText(os.getenv('OPENAI_API_KEY', ''))
        self.quality_target = QComboBox()
        self.quality_target.addItems(['Professional', 'Studio', 'Archive', 'Museum'])
        self.candidates = QComboBox()
        self.candidates.addItems(['2', '3', '4'])
        self.candidates.setCurrentText('3')
        self.confirm = QCheckBox(
            'I understand this mode generates plausible detail and is not a purely factual restoration.'
        )
        form.addRow('API key:', self.api_key)
        form.addRow('Validation target:', self.quality_target)
        form.addRow('Candidates:', self.candidates)
        form.addRow(self.confirm)
        layout.addWidget(settings_box)

        self.start = QPushButton('Reconstruct Photograph')
        self.start.setMinimumHeight(48)
        self.start.clicked.connect(self.start_reconstruction)
        layout.addWidget(self.start)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        privacy = QLabel('This feature requires internet access and sends the selected photograph to the configured image-editing service.')
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        self.setCentralWidget(root)

    def select_photo(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, 'Select photograph', '', 'Images (*.jpg *.jpeg *.png *.webp)'
        )
        if selected:
            self.source = Path(selected)
            self.source_label.setText(self.source.name)

    def start_reconstruction(self) -> None:
        if not self.source:
            QMessageBox.information(self, 'PhotoPerfect', 'Select a photograph first.')
            return
        if not self.api_key.text().strip():
            QMessageBox.information(self, 'PhotoPerfect', 'Enter an OpenAI API key.')
            return
        if not self.confirm.isChecked():
            QMessageBox.information(self, 'PhotoPerfect', 'Confirm that you understand generative reconstruction first.')
            return
        output = self.source.parent / 'Generative Reconstruction' / f'{self.source.stem}_reconstructed.jpg'
        settings = ReconstructionSettings(
            candidates=int(self.candidates.currentText()),
            quality_target=self.quality_target.currentText(),
        )
        self.start.setEnabled(False)
        self.log.setPlainText('Generating and validating candidates...')
        self.worker = ReconstructionWorker(
            self.source, output, self.api_key.text().strip(), settings, self.signals
        )
        self.worker.start()

    def on_finished(self, path: str, details: str) -> None:
        self.start.setEnabled(True)
        self.log.setPlainText(details)
        QMessageBox.information(self, 'PhotoPerfect', f'Reconstruction saved to:\n{path}')

    def on_failed(self, message: str) -> None:
        self.start.setEnabled(True)
        self.log.setPlainText(message)
        QMessageBox.critical(self, 'PhotoPerfect', 'No safe reconstruction was produced. See the log for details.')


def main() -> None:
    import sys
    app = QApplication(sys.argv)
    window = ReconstructionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
