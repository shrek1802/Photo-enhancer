from __future__ import annotations

import os
import shutil
import sys
import threading
import traceback
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget
)

from enhancer import EnhanceOptions, PhotoEnhancer, supported_image
from photo_analysis import exact_hash, hamming, perceptual_hash, write_report
from repair_editor import RepairEditor

APP_NAME = 'PhotoPerfect Studio'
APP_VERSION = '1.0.0'
OUTPUT_FOLDER = 'Professionally Enhanced'

MODE_MAP = {
    'Auto Detect': 'Smart Auto',
    'Auto Enhance': 'Smart Auto',
    'Auto Restore': 'Old Photo Restoration',
    'Family': 'Professional Portrait',
    'Portrait': 'Professional Portrait',
    'Celebrations': 'Event / Christening',
    'Landscape': 'Landscape',
    'Low Light': 'Night / Low Light',
    'Screenshot Recovery': 'Smart Auto',
    'Advanced': 'Smart Auto',
}


class WorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, int, int, str)
    failed = Signal(str)


class SelectionWorker(threading.Thread):
    def __init__(
        self,
        sources: list[Path],
        folder_root: Path | None,
        options: EnhanceOptions,
        find_duplicates: bool,
        signals: WorkerSignals,
    ) -> None:
        super().__init__(daemon=True)
        self.sources = sources
        self.folder_root = folder_root
        self.options = options
        self.find_duplicates = find_duplicates
        self.signals = signals
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def output_for(self, source: Path) -> tuple[Path, Path]:
        if self.folder_root is not None:
            output_root = self.folder_root / OUTPUT_FOLDER
            relative = source.relative_to(self.folder_root)
            destination = output_root / relative.parent / f'{source.stem}_enhanced.jpg'
            return output_root, destination
        output_root = source.parent / OUTPUT_FOLDER
        destination = output_root / f'{source.stem}_enhanced.jpg'
        return output_root, destination

    def organise_duplicates(self, scores: dict[Path, int], output_root: Path) -> int:
        if not self.find_duplicates or self.cancel_requested or len(self.sources) < 2:
            return 0

        exact_seen: dict[str, Path] = {}
        groups: list[list[tuple[Path, str]]] = []
        duplicate_count = 0

        for path in self.sources:
            if self.cancel_requested:
                break
            digest = exact_hash(path)
            if digest in exact_seen:
                groups.append([
                    (exact_seen[digest], perceptual_hash(exact_seen[digest])),
                    (path, perceptual_hash(path)),
                ])
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
        duplicate_root = output_root / 'Duplicate Review'
        best_root = output_root / 'Best Photos'

        for index, group in enumerate(useful, start=1):
            group_dir = duplicate_root / f'Group {index:03d}'
            group_dir.mkdir(parents=True, exist_ok=True)
            unique_paths = list(dict.fromkeys(item[0] for item in group))
            best = max(unique_paths, key=lambda item: scores.get(item, 0))
            for source in unique_paths:
                marker = '_BEST' if source == best else ''
                shutil.copy2(source, group_dir / f'{source.stem}{marker}{source.suffix}')
            _, enhanced = self.output_for(best)
            if enhanced.exists():
                best_root.mkdir(parents=True, exist_ok=True)
                shutil.copy2(enhanced, best_root / enhanced.name)

        return duplicate_count

    def run(self) -> None:
        try:
            if not self.sources:
                self.signals.failed.emit('No supported photographs were selected.')
                return

            enhancer = PhotoEnhancer(self.options)
            completed = 0
            review_count = 0
            analyses_by_root: dict[Path, list] = defaultdict(list)
            scores: dict[Path, int] = {}
            first_output_root: Path | None = None

            for index, source in enumerate(self.sources, start=1):
                if self.cancel_requested:
                    break

                output_root, destination = self.output_for(source)
                first_output_root = first_output_root or output_root
                review_root = output_root / 'Review Needed'
                output_root.mkdir(parents=True, exist_ok=True)
                self.signals.progress.emit(index - 1, len(self.sources), source.name)

                try:
                    result = enhancer.process(source, destination)
                    result.analysis.filename = source.name if self.folder_root is None else str(source.relative_to(self.folder_root))
                    analyses_by_root[output_root].append(result.analysis)
                    scores[source] = result.analysis.quality_score

                    if result.review_needed:
                        review = review_root / destination.name
                        review.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(destination, review)
                        details = [
                            f'Quality score: {result.analysis.quality_score}/100',
                            f'Scene: {result.analysis.scene}',
                            f'Reason: {result.analysis.review_reason or "Automatic confidence check"}',
                        ]
                        if getattr(result, 'pipeline_report', None):
                            details.append(f'Confidence: {result.pipeline_report.confidence}%')
                            details.append('Stages: ' + ', '.join(result.pipeline_report.stages))
                        review.with_suffix('.txt').write_text('\n'.join(details) + '\n', encoding='utf-8')
                        review_count += 1
                    completed += 1
                except Exception:
                    error = review_root / f'{source.stem}_ERROR.txt'
                    error.parent.mkdir(parents=True, exist_ok=True)
                    error.write_text(traceback.format_exc(), encoding='utf-8')
                    review_count += 1

                self.signals.progress.emit(index, len(self.sources), source.name)

            for output_root, analyses in analyses_by_root.items():
                write_report(output_root / 'Photo Analysis Report.csv', analyses)

            duplicate_count = 0
            if first_output_root is not None:
                duplicate_count = self.organise_duplicates(scores, first_output_root)

            status = 'Cancelled' if self.cancel_requested else 'Complete'
            self.signals.finished.emit(completed, review_count, duplicate_count, status)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class DropPanel(QFrame):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.selected_sources: list[Path] = []
        self.folder_root: Path | None = None
        self.worker: SelectionWorker | None = None
        self.editor_windows: list[RepairEditor] = []
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.on_progress)
        self.signals.finished.connect(self.on_finished)
        self.signals.failed.connect(self.on_failed)

        self.setAcceptDrops(True)
        self.setWindowTitle(f'{APP_NAME} v{APP_VERSION}')
        self.resize(930, 860)
        self.build_ui()

    def build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QLabel(APP_NAME)
        title.setStyleSheet('font-size: 30px; font-weight: 750;')
        subtitle = QLabel('Professional AI photo enhancement and restoration with Face Identity Lock.')
        subtitle.setStyleSheet('font-size: 14px; color: #666;')
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        input_box = QGroupBox('1. Choose photos')
        input_layout = QVBoxLayout(input_box)
        buttons = QHBoxLayout()
        one = QPushButton('Select Photo')
        many = QPushButton('Select Multiple Photos')
        folder = QPushButton('Select Folder')
        for button in (one, many, folder):
            button.setMinimumHeight(44)
        one.clicked.connect(self.select_one_photo)
        many.clicked.connect(self.select_multiple_photos)
        folder.clicked.connect(self.select_folder)
        buttons.addWidget(one)
        buttons.addWidget(many)
        buttons.addWidget(folder)
        input_layout.addLayout(buttons)

        self.drop_panel = DropPanel()
        self.drop_panel.setFrameShape(QFrame.StyledPanel)
        self.drop_panel.setStyleSheet(
            'QFrame { border: 2px dashed #888; border-radius: 10px; padding: 12px; }'
            'QFrame:hover { border-color: #444; }'
        )
        drop_layout = QVBoxLayout(self.drop_panel)
        drop_title = QLabel('Drag and drop a photo, several photos, or a folder here')
        drop_title.setAlignment(Qt.AlignCenter)
        drop_title.setStyleSheet('font-size: 14px; font-weight: 600;')
        self.selection_label = QLabel('Nothing selected')
        self.selection_label.setAlignment(Qt.AlignCenter)
        self.selection_label.setWordWrap(True)
        self.selection_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        drop_layout.addWidget(drop_title)
        drop_layout.addWidget(self.selection_label)
        self.drop_panel.clicked.connect(self.select_one_photo)
        input_layout.addWidget(self.drop_panel)
        layout.addWidget(input_box)

        mode_box = QGroupBox('2. Enhancement mode')
        mode_form = QFormLayout(mode_box)
        self.mode = QComboBox()
        self.mode.addItems([
            'Auto Detect', 'Auto Enhance', 'Auto Restore', 'Family', 'Portrait',
            'Celebrations', 'Landscape', 'Low Light', 'Screenshot Recovery', 'Advanced'
        ])
        self.mode.setCurrentText('Auto Detect')
        self.mode.currentTextChanged.connect(self.mode_changed)
        self.mode_help = QLabel()
        self.mode_help.setWordWrap(True)
        self.mode_help.setStyleSheet('color: #666;')
        mode_form.addRow('Mode:', self.mode)
        mode_form.addRow(self.mode_help)
        layout.addWidget(mode_box)

        essentials = QGroupBox('3. Essential settings')
        essentials_form = QFormLayout(essentials)
        self.upscale = QComboBox()
        self.upscale.addItems(['Original size', '2× upscale', '4K long edge'])
        self.identity_lock = self.checkbox(
            'Face Identity Lock — preserve real facial features and expressions', True
        )
        self.duplicates = self.checkbox('Find duplicates and select the best photograph', True)
        self.auto_restore = self.checkbox('Use automatic restoration when the photo needs it', True)
        self.remove_ui = self.checkbox('Remove obvious screenshot and social-media borders', True)
        essentials_form.addRow('Output resolution:', self.upscale)
        essentials_form.addRow(self.identity_lock)
        essentials_form.addRow(self.duplicates)
        essentials_form.addRow(self.auto_restore)
        essentials_form.addRow(self.remove_ui)
        layout.addWidget(essentials)

        self.advanced_box = QGroupBox('Advanced controls')
        advanced_form = QFormLayout(self.advanced_box)
        self.strength = QComboBox()
        self.strength.addItems(['Natural Finish', 'Professional Finish', 'Maximum Recovery'])
        self.shadow = self.checkbox('Lift unwanted shadows and brighten dark faces', True)
        self.highlight = self.checkbox('Recover harsh highlights where possible', True)
        self.flare = self.checkbox('Reduce small lens flare spots and coloured glare', True)
        self.denoise = self.checkbox('Remove noise and compression damage', True)
        self.sharpen = self.checkbox('Sharpen each photograph only where needed', True)
        self.straighten = self.checkbox('Straighten slightly crooked horizons', True)
        self.quality = QSpinBox()
        self.quality.setRange(85, 100)
        self.quality.setValue(95)
        self.quality.setSuffix('%')
        advanced_form.addRow('Finish:', self.strength)
        for widget in (self.shadow, self.highlight, self.flare, self.denoise, self.sharpen, self.straighten):
            advanced_form.addRow(widget)
        advanced_form.addRow('JPEG quality:', self.quality)
        self.advanced_box.setVisible(False)
        layout.addWidget(self.advanced_box)

        controls = QHBoxLayout()
        self.start_button = QPushButton('Start Enhancement')
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet('font-size: 15px; font-weight: 700;')
        self.start_button.clicked.connect(self.start_processing)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setMinimumHeight(50)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_processing)
        controls.addWidget(self.start_button, 1)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)

        tools = QHBoxLayout()
        repair = QPushButton('Open Repair Studio')
        repair.clicked.connect(self.open_repair_editor)
        tools.addWidget(repair)
        tools.addStretch(1)
        layout.addLayout(tools)

        self.progress = QProgressBar()
        self.status = QLabel('Ready')
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.log)

        privacy = QLabel('Local processing: originals are never overwritten and photographs stay on this PC.')
        privacy.setStyleSheet('color: #666;')
        layout.addWidget(privacy)

        self.setCentralWidget(central)
        self.mode_changed('Auto Detect')

    @staticmethod
    def checkbox(text: str, checked: bool) -> QCheckBox:
        box = QCheckBox(text)
        box.setChecked(checked)
        return box

    def mode_changed(self, mode: str) -> None:
        descriptions = {
            'Auto Detect': 'Recommended. Analyses each photograph and automatically chooses the safest, most suitable processing plan.',
            'Auto Enhance': 'For already-good photographs that need better colour, lighting, contrast and a natural professional finish.',
            'Auto Restore': 'For faded, damaged, blurred, noisy, scanned or heavily compressed photographs.',
            'Family': 'Natural enhancement for everyday family photographs, with gentle skin tones and protected faces.',
            'Portrait': 'Professional lighting and colour for individual or group portraits.',
            'Celebrations': 'Consistent colour and lighting for weddings, birthdays, christenings, parties and other special occasions.',
            'Landscape': 'Improves scenery, nature, travel photographs and skies without over-processing.',
            'Low Light': 'Reduces noise and improves dark indoor, evening and night photographs.',
            'Screenshot Recovery': 'Repairs compressed images and removes obvious phone or social-media interface borders.',
            'Advanced': 'Shows full manual controls while retaining Face Identity Lock.',
        }
        self.mode_help.setText(descriptions.get(mode, ''))
        self.advanced_box.setVisible(mode == 'Advanced')

    @staticmethod
    def image_filter() -> str:
        return 'Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)'

    def select_one_photo(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, 'Select a photograph', '', self.image_filter())
        if selected:
            self.set_file_selection([Path(selected)])

    def select_multiple_photos(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(self, 'Select photographs', '', self.image_filter())
        if selected:
            self.set_file_selection([Path(item) for item in selected])

    def select_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, 'Select photo folder')
        if selected:
            self.set_folder_selection(Path(selected))

    def set_file_selection(self, paths: list[Path]) -> None:
        valid = sorted({path.resolve() for path in paths if path.is_file() and supported_image(path)})
        self.selected_sources = valid
        self.folder_root = None
        if len(valid) == 1:
            text = f'1 photo selected: {valid[0].name}'
        else:
            text = f'{len(valid)} photos selected'
        self.selection_label.setText(text)
        self.log.append(text)

    def set_folder_selection(self, folder: Path) -> None:
        output = folder / OUTPUT_FOLDER
        photos: list[Path] = []
        for path in folder.rglob('*'):
            if not path.is_file() or not supported_image(path):
                continue
            try:
                path.relative_to(output)
                continue
            except ValueError:
                photos.append(path.resolve())
        self.selected_sources = sorted(photos)
        self.folder_root = folder.resolve()
        text = f'Folder selected: {folder.name} — {len(photos)} supported photos'
        self.selection_label.setText(text)
        self.log.append(text)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        folders = [path for path in paths if path.is_dir()]
        files = [path for path in paths if path.is_file() and supported_image(path)]
        if len(folders) == 1 and not files:
            self.set_folder_selection(folders[0])
        elif files:
            self.set_file_selection(files)
        event.acceptProposedAction()

    def open_repair_editor(self) -> None:
        editor = RepairEditor(parent=self)
        editor.show()
        self.editor_windows.append(editor)
        editor.destroyed.connect(
            lambda: self.editor_windows.remove(editor) if editor in self.editor_windows else None
        )

    def options(self) -> EnhanceOptions:
        finish_map = {
            'Natural Finish': 'natural',
            'Professional Finish': 'strong',
            'Maximum Recovery': 'maximum',
        }
        selected_mode = self.mode.currentText()
        strength = finish_map[self.strength.currentText()]
        if selected_mode == 'Auto Enhance':
            strength = 'natural'
        elif selected_mode == 'Auto Restore':
            strength = 'maximum'
        elif selected_mode in {'Family', 'Portrait', 'Celebrations'}:
            strength = 'strong'

        return EnhanceOptions(
            preset=MODE_MAP[selected_mode],
            strength=strength,
            upscale=self.upscale.currentText(),
            lift_shadows=self.shadow.isChecked(),
            recover_highlights=self.highlight.isChecked(),
            reduce_flare=self.flare.isChecked(),
            denoise=self.denoise.isChecked(),
            sharpen=self.sharpen.isChecked(),
            face_aware=self.identity_lock.isChecked(),
            portrait_finish=not self.identity_lock.isChecked(),
            straighten_horizon=self.straighten.isChecked(),
            auto_rotate=True,
            neural_ai=True,
            automatic_restoration=self.auto_restore.isChecked() or selected_mode in {'Auto Detect', 'Auto Restore', 'Screenshot Recovery'},
            remove_screenshot_ui=self.remove_ui.isChecked() or selected_mode == 'Screenshot Recovery',
            jpeg_quality=self.quality.value(),
        )

    def start_processing(self) -> None:
        if not self.selected_sources:
            QMessageBox.information(self, APP_NAME, 'Select a photo, several photos, or a folder first.')
            return

        existing_roots = {self.output_root_for(source) for source in self.selected_sources}
        if any(root.exists() for root in existing_roots):
            answer = QMessageBox.question(
                self,
                APP_NAME,
                'A Professionally Enhanced folder already exists. Existing matching files may be replaced. Continue?',
            )
            if answer != QMessageBox.Yes:
                return

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.log.append(f'Starting {self.mode.currentText()} for {len(self.selected_sources)} photo(s)...')
        self.worker = SelectionWorker(
            self.selected_sources,
            self.folder_root,
            self.options(),
            self.duplicates.isChecked(),
            self.signals,
        )
        self.worker.start()

    def output_root_for(self, source: Path) -> Path:
        return (self.folder_root / OUTPUT_FOLDER) if self.folder_root else (source.parent / OUTPUT_FOLDER)

    def cancel_processing(self) -> None:
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
        message = (
            f'{status}: {completed} processed, {review} flagged for review, '
            f'{duplicates} duplicate/near-duplicate matches found.'
        )
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
