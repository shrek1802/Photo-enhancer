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
from photo_analysis import exact_hash, hamming, perceptual_hash, write_report
from repair_editor import RepairEditor

APP_NAME = 'PhotoPerfect Batch AI'
APP_VERSION = '0.3.0'
OUTPUT_FOLDER = 'Professionally Enhanced'


class WorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, int, int, str)
    failed = Signal(str)


class BatchWorker(threading.Thread):
    def __init__(self, folder: Path, options: EnhanceOptions, find_duplicates: bool,
                 signals: WorkerSignals):
        super().__init__(daemon=True)
        self.folder = folder
        self.options = options
        self.find_duplicates = find_duplicates
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

    def organise_duplicates(self, photos: list[Path], scores: dict[Path, int], output: Path) -> int:
        if not self.find_duplicates or self.cancel_requested:
            return 0
        exact_seen: dict[str, Path] = {}
        groups: list[list[tuple[Path, str]]] = []
        duplicate_count = 0
        for path in photos:
            if self.cancel_requested:
                break
            digest = exact_hash(path)
            if digest in exact_seen:
                groups.append([(exact_seen[digest], perceptual_hash(exact_seen[digest])),
                               (path, perceptual_hash(path))])
                duplicate_count += 1
                continue
            exact_seen[digest] = path
            phash = perceptual_hash(path)
            matched = False
            for group in groups:
                if hamming(phash, group[0][1]) <= 7:
                    if all(existing[0] != path for existing in group):
                        group.append((path, phash))
                        duplicate_count += 1
                    matched = True
                    break
            if not matched:
                groups.append([(path, phash)])

        useful = [group for group in groups if len(group) > 1]
        duplicate_root = output / 'Duplicate Review'
        best_root = output / 'Best Photos'
        for index, group in enumerate(useful, start=1):
            group_dir = duplicate_root / f'Group {index:03d}'
            group_dir.mkdir(parents=True, exist_ok=True)
            unique_paths = list(dict.fromkeys(item[0] for item in group))
            best = max(unique_paths, key=lambda item: scores.get(item, 0))
            for source in unique_paths:
                marker = '_BEST' if source == best else ''
                shutil.copy2(source, group_dir / f'{source.stem}{marker}{source.suffix}')
            best_root.mkdir(parents=True, exist_ok=True)
            enhanced = output / best.relative_to(self.folder).parent / f'{best.stem}_enhanced.jpg'
            if enhanced.exists():
                shutil.copy2(enhanced, best_root / enhanced.name)
        return duplicate_count

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
            analyses = []
            scores: dict[Path, int] = {}

            for index, source in enumerate(photos, start=1):
                if self.cancel_requested:
                    break
                relative = source.relative_to(self.folder)
                destination = output_root / relative.parent / f'{source.stem}_enhanced.jpg'
                self.signals.progress.emit(index - 1, len(photos), source.name)
                try:
                    result = enhancer.process(source, destination)
                    result.analysis.filename = str(relative)
                    analyses.append(result.analysis)
                    scores[source] = result.analysis.quality_score
                    if result.review_needed:
                        review = review_root / relative.parent / destination.name
                        review.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(destination, review)
                        review.with_suffix('.txt').write_text(
                            f'Quality score: {result.analysis.quality_score}/100\n'
                            f'Scene: {result.analysis.scene}\n'
                            f'Reason: {result.analysis.review_reason}\n', encoding='utf-8')
                        review_count += 1
                    completed += 1
                except Exception:
                    error = review_root / relative.parent / f'{source.stem}_ERROR.txt'
                    error.parent.mkdir(parents=True, exist_ok=True)
                    error.write_text(traceback.format_exc(), encoding='utf-8')
                    review_count += 1
                self.signals.progress.emit(index, len(photos), source.name)

            write_report(output_root / 'Photo Analysis Report.csv', analyses)
            duplicate_count = self.organise_duplicates(photos, scores, output_root)
            status = 'Cancelled' if self.cancel_requested else 'Complete'
            self.signals.finished.emit(completed, review_count, duplicate_count, status)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.folder: Path | None = None
        self.worker: BatchWorker | None = None
        self.editor_windows: list[RepairEditor] = []
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.on_progress)
        self.signals.finished.connect(self.on_finished)
        self.signals.failed.connect(self.on_failed)
        self.setWindowTitle(f'{APP_NAME} v{APP_VERSION}')
        self.resize(840, 830)
        self.build_ui()

    def build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(11)
        title = QLabel(f'{APP_NAME}  v{APP_VERSION}')
        title.setStyleSheet('font-size: 25px; font-weight: 700;')
        subtitle = QLabel('Smart batch enhancement plus a hands-on repair studio for flare, shadows, scratches and unwanted objects.')
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

        settings = QGroupBox('2. Smart processing settings')
        form = QFormLayout(settings)
        self.preset = QComboBox()
        self.preset.addItems(['Smart Auto', 'Event / Christening', 'Professional Portrait',
                              'Old Photo Restoration', 'Landscape', 'Night / Low Light'])
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
        self.portrait = self.checkbox('Natural portrait finishing without replacing faces', True)
        self.straighten = self.checkbox('Automatically straighten slightly crooked horizons', True)
        self.rotate = self.checkbox('Correct orientation from photo metadata', True)
        self.duplicates = self.checkbox('Find duplicates and near-duplicates and select the best', True)
        self.quality = QSpinBox()
        self.quality.setRange(85, 100)
        self.quality.setValue(95)
        self.quality.setSuffix('%')
        form.addRow('Processing preset:', self.preset)
        form.addRow('Enhancement strength:', self.strength)
        form.addRow('Output resolution:', self.upscale)
        for widget in [self.shadow, self.highlight, self.flare, self.denoise,
                       self.sharpen, self.faces, self.portrait, self.straighten,
                       self.rotate, self.duplicates]:
            form.addRow(widget)
        form.addRow('JPEG quality:', self.quality)
        layout.addWidget(settings)

        note = QLabel('Batch outputs include <b>Professionally Enhanced</b>, <b>Review Needed</b>, '
                      '<b>Duplicate Review</b>, <b>Best Photos</b> and a CSV quality report.')
        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.start_button = QPushButton('Analyse and Enhance Folder')
        self.start_button.setMinimumHeight(46)
        self.start_button.clicked.connect(self.start_batch)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_batch)
        controls.addWidget(self.start_button, 1)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)

        editor_box = QGroupBox('3. Manual repair and comparison')
        editor_layout = QHBoxLayout(editor_box)
        editor_text = QLabel('Open one difficult photograph to brush away flare, scratches, unwanted objects or shadows, compare before/after and crop it.')
        editor_text.setWordWrap(True)
        editor_button = QPushButton('Open Manual Repair Studio')
        editor_button.setMinimumHeight(42)
        editor_button.clicked.connect(self.open_repair_editor)
        editor_layout.addWidget(editor_text, 1)
        editor_layout.addWidget(editor_button)
        layout.addWidget(editor_box)

        self.progress = QProgressBar()
        self.status = QLabel('Ready')
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
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

    def open_repair_editor(self) -> None:
        editor = RepairEditor(parent=self)
        editor.show()
        self.editor_windows.append(editor)
        editor.destroyed.connect(lambda: self.editor_windows.remove(editor) if editor in self.editor_windows else None)

    def options(self) -> EnhanceOptions:
        return EnhanceOptions(
            preset=self.preset.currentText(), strength=self.strength.currentText().lower(),
            upscale=self.upscale.currentText(), lift_shadows=self.shadow.isChecked(),
            recover_highlights=self.highlight.isChecked(), reduce_flare=self.flare.isChecked(),
            denoise=self.denoise.isChecked(), sharpen=self.sharpen.isChecked(),
            face_aware=self.faces.isChecked(), portrait_finish=self.portrait.isChecked(),
            straighten_horizon=self.straighten.isChecked(), auto_rotate=self.rotate.isChecked(),
            jpeg_quality=self.quality.value())

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
        self.log.append('Starting smart analysis and enhancement...')
        self.worker = BatchWorker(self.folder, self.options(), self.duplicates.isChecked(), self.signals)
        self.worker.start()

    def cancel_batch(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText('Stopping safely after the current photograph...')

    def on_progress(self, current: int, total: int, filename: str) -> None:
        self.progress.setValue(int(current / max(total, 1) * 100))
        self.status.setText(f'Processing {current}/{total}: {filename}')

    def on_finished(self, completed: int, review: int, duplicates: int, status: str) -> None:
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if status == 'Complete':
            self.progress.setValue(100)
        message = (f'{status}: {completed} processed, {review} flagged for review, '
                   f'{duplicates} duplicate/near-duplicate matches found.')
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
