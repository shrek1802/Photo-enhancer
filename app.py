from __future__ import annotations

import shutil
import sys
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget
)

from enhancer import EnhanceOptions, PhotoEnhancer, supported_image

APP_NAME = 'PhotoPerfect Batch AI'
OUTPUT_FOLDER = 'Professionally Enhanced'


class WorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, int, str)
    failed = Signal(str)


class BatchWorker(threading.Thread):
    def __init__(self, folder: Path, options: EnhanceOptions, signals: WorkerSignals):
        super().__init__(daemon=True)
        self.folder = folder
        self.options = options
        self.signals = signals
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def photos(self) -> list[Path]:
        output = self.folder / OUTPUT_FOLDER
        found: list[Path] = []
        for path in self.folder.rglob('*'):
            if not path.is_file() or not supported_image(path):
                continue
            try:
                path.relative_to(output)
                continue
            except ValueError:
                found.append(path)
        return sorted(found)

    def run(self) -> None:
        try:
            photos = self.photos()
            if not photos:
                self.signals.failed.emit('No supported photographs were found in the selected folder.')
                return

            output_root = self.folder / OUTPUT_FOLDER
            review_root = output_root / 'Review Needed'
            output_root.mkdir(exist_ok=True)
            enhancer = PhotoEnhancer(self.options)
            completed = 0
            review_count = 0

            for index, source in enumerate(photos, start=1):
                if self.cancel_requested:
                    break
                relative = source.relative_to(self.folder)
                destination = output_root / relative.parent / f'{source.stem}_enhanced.jpg'
                self.signals.progress.emit(index - 1, len(photos), source.name)
                try:
                    result = enhancer.process(source, destination)
                    if result.review_needed:
                        review = review_root / relative.parent / destination.name
                        review.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(destination, review)
                        review_count += 1
                    completed += 1
                except Exception:
                    error = review_root / relative.parent / f'{source.stem}_ERROR.txt'
                    error.parent.mkdir(parents=True, exist_ok=True)
                    error.write_text(traceback.format_exc(), encoding='utf-8')
                    review_count += 1
                self.signals.progress.emit(index, len(photos), source.name)

            status = 'Cancelled' if self.cancel_requested else 'Complete'
            self.signals.finished.emit(completed, review_count, status)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.folder: Path | None = None
        self.worker: BatchWorker | None = None
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.on_progress)
        self.signals.finished.connect(self.on_finished)
        self.signals.failed.connect(self.on_failed)
        self.setWindowTitle(APP_NAME)
        self.resize(780, 680)
        self.build_ui()

    def build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        title = QLabel(APP_NAME)
        title.setStyleSheet('font-size: 25px; font-weight: 700;')
        subtitle = QLabel('Select a folder and automatically repair and enhance every photograph while keeping the originals untouched.')
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        folder_box = QGroupBox('1. Select photographs')
        folder_layout = QHBoxLayout(folder_box)
        self.folder_label = QLabel('No folder selected')
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        choose = QPushButton('Select Photo Folder')
        choose.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label, 1)
        folder_layout.addWidget(choose)
        layout.addWidget(folder_box)

        settings = QGroupBox('2. Enhancement settings')
        form = QFormLayout(settings)
        self.strength = QComboBox()
        self.strength.addItems(['Natural', 'Strong', 'Maximum'])
        self.upscale = QComboBox()
        self.upscale.addItems(['Original size', '2× upscale', '4K long edge'])
        self.shadow = self.checkbox('Lift unwanted shadows and brighten dark faces', True)
        self.highlight = self.checkbox('Recover harsh highlights where possible', True)
        self.flare = self.checkbox('Reduce small lens flare spots and coloured glare', True)
        self.denoise = self.checkbox('Remove noise and compression damage', True)
        self.sharpen = self.checkbox('Sharpen each photograph only where needed', True)
        self.faces = self.checkbox('Face-aware exposure correction and identity protection', True)
        self.rotate = self.checkbox('Correct orientation from photo metadata', True)
        self.quality = QSpinBox()
        self.quality.setRange(85, 100)
        self.quality.setValue(95)
        self.quality.setSuffix('%')
        form.addRow('Enhancement strength:', self.strength)
        form.addRow('Output resolution:', self.upscale)
        for widget in [self.shadow, self.highlight, self.flare, self.denoise, self.sharpen, self.faces, self.rotate]:
            form.addRow(widget)
        form.addRow('JPEG quality:', self.quality)
        layout.addWidget(settings)

        note = QLabel('Finished photographs are saved in a new <b>Professionally Enhanced</b> folder inside the selected folder. Difficult photographs are copied into <b>Review Needed</b>.')
        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.start_button = QPushButton('Start Batch Enhancement')
        self.start_button.setMinimumHeight(44)
        self.start_button.clicked.connect(self.start_batch)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_batch)
        controls.addWidget(self.start_button, 1)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)

        self.progress = QProgressBar()
        self.status = QLabel('Ready')
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.log)
        privacy = QLabel('Local processing: photographs stay on this PC and originals are never overwritten.')
        privacy.setStyleSheet('color: #555;')
        layout.addWidget(privacy)
        self.setCentralWidget(central)

    @staticmethod
    def checkbox(text: str, checked: bool) -> QCheckBox:
        box = QCheckBox(text)
        box.setChecked(checked)
        return box

    def select_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, 'Select photo folder')
        if selected:
            self.folder = Path(selected)
            self.folder_label.setText(str(self.folder))
            self.log.append(f'Selected: {self.folder}')

    def options(self) -> EnhanceOptions:
        return EnhanceOptions(
            strength=self.strength.currentText().lower(),
            upscale=self.upscale.currentText(),
            lift_shadows=self.shadow.isChecked(),
            recover_highlights=self.highlight.isChecked(),
            reduce_flare=self.flare.isChecked(),
            denoise=self.denoise.isChecked(),
            sharpen=self.sharpen.isChecked(),
            face_aware=self.faces.isChecked(),
            auto_rotate=self.rotate.isChecked(),
            jpeg_quality=self.quality.value(),
        )

    def start_batch(self) -> None:
        if not self.folder:
            QMessageBox.information(self, APP_NAME, 'Please select a photo folder first.')
            return
        output = self.folder / OUTPUT_FOLDER
        if output.exists() and QMessageBox.question(
            self, APP_NAME, 'The output folder already exists. Existing matching files may be replaced. Continue?'
        ) != QMessageBox.Yes:
            return
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.log.append('Starting batch...')
        self.worker = BatchWorker(self.folder, self.options(), self.signals)
        self.worker.start()

    def cancel_batch(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText('Stopping safely after the current photograph...')

    def on_progress(self, current: int, total: int, filename: str) -> None:
        self.progress.setValue(int(current / max(total, 1) * 100))
        self.status.setText(f'Processing {current}/{total}: {filename}')

    def on_finished(self, completed: int, review: int, status: str) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if status == 'Complete':
            self.progress.setValue(100)
        message = f'{status}: {completed} processed, {review} flagged for review.'
        self.status.setText(message)
        self.log.append(message)
        QMessageBox.information(self, APP_NAME, message)

    def on_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.status.setText('Processing failed')
        self.log.append(message)
        QMessageBox.critical(self, APP_NAME, message)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
