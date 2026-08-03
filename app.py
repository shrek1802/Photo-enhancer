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
APP_VERSION = '0.5.0'
OUTPUT_FOLDER = 'Professionally Enhanced'


class WorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, int, int, str)
    failed = Signal(str)


class BatchWorker(threading.Thread):
    def __init__(self, photos: list[Path], output_root: Path, common_root: Path,
                 options: EnhanceOptions, find_duplicates: bool,
                 signals: WorkerSignals):
        super().__init__(daemon=True)
        self.selected_photos = sorted(dict.fromkeys(photos))
        self.output_root = output_root
        self.common_root = common_root
        self.options = options
        self.find_duplicates = find_duplicates
        self.signals = signals
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def relative_path(self, source: Path) -> Path:
        try:
            return source.relative_to(self.common_root)
        except ValueError:
            return Path(source.name)

    def organise_duplicates(self, photos: list[Path], scores: dict[Path, int]) -> int:
        if not self.find_duplicates or self.cancel_requested or len(photos) < 2:
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
        duplicate_root = self.output_root / 'Duplicate Review'
        best_root = self.output_root / 'Best Photos'
        for index, group in enumerate(useful, start=1):
            group_dir = duplicate_root / f'Group {index:03d}'
            group_dir.mkdir(parents=True, exist_ok=True)
            unique_paths = list(dict.fromkeys(item[0] for item in group))
            best = max(unique_paths, key=lambda item: scores.get(item, 0))
            for source in unique_paths:
                marker = '_BEST' if source == best else ''
                shutil.copy2(source, group_dir / f'{source.stem}{marker}{source.suffix}')
            best_root.mkdir(parents=True, exist_ok=True)
            relative = self.relative_path(best)
            enhanced = self.output_root / relative.parent / f'{best.stem}_enhanced.jpg'
            if enhanced.exists():
                shutil.copy2(enhanced, best_root / enhanced.name)
        return duplicate_count

    def run(self) -> None:
        try:
            photos = [path for path in self.selected_photos if path.is_file() and supported_image(path)]
            if not photos:
                self.signals.failed.emit('No supported photographs were selected.')
                return

            review_root = self.output_root / 'Review Needed'
            self.output_root.mkdir(parents=True, exist_ok=True)
            enhancer = PhotoEnhancer(self.options)
            completed = 0
            review_count = 0
            analyses = []
            scores: dict[Path, int] = {}

            for index, source in enumerate(photos, start=1):
                if self.cancel_requested:
                    break
                relative = self.relative_path(source)
                destination = self.output_root / relative.parent / f'{source.stem}_enhanced.jpg'
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
                        reason = result.analysis.review_reason or 'Automatic quality check requested review.'
                        review.with_suffix('.txt').write_text(
                            f'Quality score: {result.analysis.quality_score}/100\n'
                            f'Scene: {result.analysis.scene}\n'
                            f'Reason: {reason}\n', encoding='utf-8')
                        review_count += 1
                    completed += 1
                except Exception:
                    error = review_root / relative.parent / f'{source.stem}_ERROR.txt'
                    error.parent.mkdir(parents=True, exist_ok=True)
                    error.write_text(traceback.format_exc(), encoding='utf-8')
                    review_count += 1
                self.signals.progress.emit(index, len(photos), source.name)

            write_report(self.output_root / 'Photo Analysis Report.csv', analyses)
            duplicate_count = self.organise_duplicates(photos, scores)
            status = 'Cancelled' if self.cancel_requested else 'Complete'
            self.signals.finished.emit(completed, review_count, duplicate_count, status)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_photos: list[Path] = []
        self.output_root: Path | None = None
        self.common_root: Path | None = None
        self.worker: BatchWorker | None = None
        self.editor_windows: list[RepairEditor] = []
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.on_progress)
        self.signals.finished.connect(self.on_finished)
        self.signals.failed.connect(self.on_failed)
        self.setWindowTitle(f'{APP_NAME} v{APP_VERSION}')
        self.resize(860, 850)
        self.build_ui()

    def build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(11)
        title = QLabel(f'{APP_NAME}  v{APP_VERSION}')
        title.setStyleSheet('font-size: 25px; font-weight: 700;')
        subtitle = QLabel('Choose one photo, several photos, or a complete folder. Face Identity Lock and Smart Auto remain enabled by default.')
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        input_box = QGroupBox('1. Choose photos')
        input_layout = QVBoxLayout(input_box)
        button_row = QHBoxLayout()
        one_button = QPushButton('Load One Photo')
        one_button.setMinimumHeight(44)
        one_button.clicked.connect(self.select_one_photo)
        many_button = QPushButton('Load Multiple Photos')
        many_button.setMinimumHeight(44)
        many_button.clicked.connect(self.select_multiple_photos)
        folder_button = QPushButton('Load Photo Folder')
        folder_button.setMinimumHeight(44)
        folder_button.clicked.connect(self.select_folder)
        button_row.addWidget(one_button)
        button_row.addWidget(many_button)
        button_row.addWidget(folder_button)
        self.selection_label = QLabel('Nothing selected')
        self.selection_label.setWordWrap(True)
        self.selection_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        input_layout.addLayout(button_row)
        input_layout.addWidget(self.selection_label)
        layout.addWidget(input_box)

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
        self.faces = self.checkbox('Face Identity Lock — preserve the original face exactly', True)
        self.portrait = self.checkbox('Light professional portrait polish without replacing faces', True)
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

        note = QLabel('For individual or selected photos, the app creates <b>Professionally Enhanced</b> beside those photos. For a folder, it creates the output folder inside the selected folder.')
        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.start_button = QPushButton('Analyse and Enhance Selection')
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
        self.log.setMaximumHeight(105)
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

    @staticmethod
    def image_filter() -> str:
        return 'Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff);;All files (*.*)'

    def apply_file_selection(self, files: list[str]) -> None:
        photos = [Path(item) for item in files if supported_image(Path(item))]
        if not photos:
            return
        self.selected_photos = photos
        parents = {path.parent for path in photos}
        if len(parents) == 1:
            self.common_root = next(iter(parents))
            self.output_root = self.common_root / OUTPUT_FOLDER
        else:
            self.common_root = Path(Path.commonpath([str(path.parent) for path in photos]))
            self.output_root = self.common_root / OUTPUT_FOLDER
        if len(photos) == 1:
            self.selection_label.setText(f'1 photo selected: {photos[0]}')
        else:
            self.selection_label.setText(f'{len(photos)} photos selected. Output: {self.output_root}')
        self.log.append(self.selection_label.text())

    def select_one_photo(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, 'Select one photo', '', self.image_filter())
        if filename:
            self.apply_file_selection([filename])

    def select_multiple_photos(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Select one or more photos', '', self.image_filter())
        if filenames:
            self.apply_file_selection(filenames)

    def select_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, 'Select photo folder')
        if not selected:
            return
        folder = Path(selected)
        output = folder / OUTPUT_FOLDER
        photos: list[Path] = []
        for path in folder.rglob('*'):
            if not path.is_file() or not supported_image(path):
                continue
            try:
                path.relative_to(output)
                continue
            except ValueError:
                photos.append(path)
        self.selected_photos = sorted(photos)
        self.common_root = folder
        self.output_root = output
        self.selection_label.setText(f'Folder selected: {folder} — {len(photos)} supported photo(s) found')
        self.log.append(self.selection_label.text())

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
        if not self.selected_photos or not self.output_root or not self.common_root:
            QMessageBox.information(self, APP_NAME, 'Please load one photo, multiple photos, or a folder first.')
            return
        if self.output_root.exists() and QMessageBox.question(
            self, APP_NAME, 'The output folder already exists. Existing matching files may be replaced. Continue?'
        ) != QMessageBox.Yes:
            return
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.log.append(f'Starting smart analysis for {len(self.selected_photos)} photo(s)...')
        self.worker = BatchWorker(
            self.selected_photos, self.output_root, self.common_root,
            self.options(), self.duplicates.isChecked(), self.signals)
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
                   f'{duplicates} duplicate/near-duplicate matches found.\nOutput: {self.output_root}')
        self.status.setText(message.replace('\n', ' '))
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
