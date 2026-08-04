from __future__ import annotations

import shutil
import sys
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from enhancer import EnhanceOptions, PhotoEnhancer, supported_image
from model_manager import PhotoPerfectModelManager
from repair_editor import RepairEditor

APP_NAME = 'PhotoPerfect Studio'
APP_VERSION = '2.2.0'
AUTO_ENGINE_VERSION = '2.1.0'
AUTO_ESSENTIALS_PACK_ID = 'auto-essentials'
OUTPUT_FOLDER = 'Professionally Enhanced'


class WorkerSignals(QObject):
    progress = Signal(int, int, str)
    report = Signal(str)
    finished = Signal(int, int, str)
    failed = Signal(str)


class SelectionWorker(threading.Thread):
    def __init__(self, sources: list[Path], folder_root: Path | None,
                 options: EnhanceOptions, signals: WorkerSignals) -> None:
        super().__init__(daemon=True)
        self.sources = sources
        self.folder_root = folder_root
        self.options = options
        self.signals = signals
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def destination_for(self, source: Path) -> tuple[Path, Path]:
        if self.folder_root:
            root = self.folder_root / OUTPUT_FOLDER
            relative = source.relative_to(self.folder_root)
            return root, root / relative.parent / f'{source.stem}_enhanced.jpg'
        root = source.parent / OUTPUT_FOLDER
        return root, root / f'{source.stem}_enhanced.jpg'

    def run(self) -> None:
        try:
            enhancer = PhotoEnhancer(self.options)
            self.signals.report.emit(f'Auto Engine: {enhancer.engine_message}')
            completed = 0
            review_count = 0
            for index, source in enumerate(self.sources, 1):
                if self.cancel_requested:
                    break
                root, destination = self.destination_for(source)
                root.mkdir(parents=True, exist_ok=True)
                self.signals.progress.emit(index - 1, len(self.sources), source.name)
                try:
                    result = enhancer.process(source, destination)
                    plan = getattr(result, 'repair_plan', None)
                    validation = getattr(result, 'validation', None)
                    if plan:
                        inspection = plan.inspection
                        self.signals.report.emit(
                            f'{source.name}\n'
                            f'  Detected: {inspection.image_type}\n'
                            f'  Pipeline: {plan.name} ({plan.strategy})\n'
                            f'  Problems: {", ".join(inspection.problems) or "none"}\n'
                            f'  Stages: {" → ".join(plan.stages)}'
                        )
                    if validation:
                        self.signals.report.emit(
                            f'  Auto Quality: {validation.before_score:.1f} → '
                            f'{validation.after_score:.1f} | '
                            f'{"Accepted" if validation.accepted else "Original retained"}'
                        )
                    if result.review_needed:
                        review = root / 'Review Needed' / destination.name
                        review.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(destination, review)
                        review_count += 1
                    completed += 1
                except Exception:
                    error = root / 'Review Needed' / f'{source.stem}_ERROR.txt'
                    error.parent.mkdir(parents=True, exist_ok=True)
                    error.write_text(traceback.format_exc(), encoding='utf-8')
                    self.signals.report.emit(f'ERROR processing {source.name}: see {error.name}')
                    review_count += 1
                self.signals.progress.emit(index, len(self.sources), source.name)
            status = 'Cancelled' if self.cancel_requested else 'Complete'
            self.signals.finished.emit(completed, review_count, status)
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
        self.signals.report.connect(self.on_report)
        self.signals.finished.connect(self.on_finished)
        self.signals.failed.connect(self.on_failed)
        self.setAcceptDrops(True)
        self.setWindowTitle(f'{APP_NAME} v{APP_VERSION}')
        self.resize(960, 900)
        self.build_ui()

    def auto_pack_status(self) -> str:
        try:
            manager = PhotoPerfectModelManager(Path(sys.executable).resolve().parent / 'models' if getattr(sys, 'frozen', False) else 'models')
            pack = manager.installed(AUTO_ESSENTIALS_PACK_ID)
            if pack and pack.valid:
                capabilities = ', '.join(sorted(manager.installed_capabilities())) or 'manifest only'
                return f'Auto Essentials {pack.manifest.version} installed ({capabilities})'
            if pack and not pack.valid:
                return 'Auto Essentials installed but invalid: ' + '; '.join(pack.errors)
        except Exception as exc:
            return f'Auto Essentials status unavailable: {exc}'
        return 'Auto Essentials not installed — built-in Auto restoration active'

    def build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(11)

        title = QLabel(f'{APP_NAME}  v{APP_VERSION}')
        title.setStyleSheet('font-size: 29px; font-weight: 750;')
        layout.addWidget(title)
        layout.addWidget(QLabel('Automatic professional photo enhancement and restoration.'))

        status_box = QGroupBox('System status')
        status_form = QFormLayout(status_box)
        status_form.addRow('Auto Engine:', QLabel(f'v{AUTO_ENGINE_VERSION} — loaded'))
        self.pack_status_label = QLabel(self.auto_pack_status())
        self.pack_status_label.setWordWrap(True)
        status_form.addRow('Model pack:', self.pack_status_label)
        status_form.addRow('Face protection:', QLabel('Face Identity Lock enabled by default'))
        layout.addWidget(status_box)

        input_box = QGroupBox('1. Choose photos')
        input_layout = QVBoxLayout(input_box)
        row = QHBoxLayout()
        one = QPushButton('Select Photo')
        many = QPushButton('Select Multiple Photos')
        folder = QPushButton('Select Folder')
        one.clicked.connect(self.select_one)
        many.clicked.connect(self.select_many)
        folder.clicked.connect(self.select_folder)
        for button in (one, many, folder):
            button.setMinimumHeight(42)
            row.addWidget(button)
        input_layout.addLayout(row)
        self.drop_panel = DropPanel()
        self.drop_panel.setFrameShape(QFrame.StyledPanel)
        self.drop_panel.setStyleSheet('QFrame { border: 2px dashed #888; border-radius: 10px; padding: 14px; }')
        drop_layout = QVBoxLayout(self.drop_panel)
        drop = QLabel('Drag and drop a photo, several photos, or a folder here')
        drop.setAlignment(Qt.AlignCenter)
        self.selection_label = QLabel('Nothing selected')
        self.selection_label.setAlignment(Qt.AlignCenter)
        self.selection_label.setWordWrap(True)
        drop_layout.addWidget(drop)
        drop_layout.addWidget(self.selection_label)
        self.drop_panel.clicked.connect(self.select_one)
        input_layout.addWidget(self.drop_panel)
        layout.addWidget(input_box)

        mode_box = QGroupBox('2. Automatic mode')
        mode_form = QFormLayout(mode_box)
        self.mode = QComboBox()
        self.mode.addItems(['Auto Detect', 'Auto Enhance', 'Auto Restore', 'Auto Portrait',
                            'Auto Celebrations', 'Auto Landscape', 'Auto Low Light',
                            'Auto Screenshot Recovery', 'Advanced'])
        self.mode.currentTextChanged.connect(self.mode_changed)
        self.mode_help = QLabel()
        self.mode_help.setWordWrap(True)
        mode_form.addRow('Mode:', self.mode)
        mode_form.addRow(self.mode_help)
        layout.addWidget(mode_box)

        settings = QGroupBox('3. Output and protection')
        form = QFormLayout(settings)
        self.upscale = QComboBox()
        self.upscale.addItems(['Original size', '2× upscale', '4K long edge'])
        self.identity_lock = QCheckBox('Face Identity Lock — preserve facial features and expressions')
        self.identity_lock.setChecked(True)
        self.remove_ui = QCheckBox('Remove obvious screenshot and social-media interface borders')
        self.remove_ui.setChecked(True)
        form.addRow('Output resolution:', self.upscale)
        form.addRow(self.identity_lock)
        form.addRow(self.remove_ui)
        layout.addWidget(settings)

        self.advanced = QGroupBox('Advanced controls')
        advanced_form = QFormLayout(self.advanced)
        self.strength = QComboBox()
        self.strength.addItems(['Natural Finish', 'Professional Finish', 'Maximum Recovery'])
        self.flare = QCheckBox('Reduce small flare spots and coloured glare')
        self.flare.setChecked(True)
        self.straighten = QCheckBox('Straighten slightly crooked horizons')
        self.straighten.setChecked(True)
        self.quality = QSpinBox()
        self.quality.setRange(85, 100)
        self.quality.setValue(95)
        advanced_form.addRow('Finish:', self.strength)
        advanced_form.addRow(self.flare)
        advanced_form.addRow(self.straighten)
        advanced_form.addRow('JPEG quality:', self.quality)
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)

        row = QHBoxLayout()
        self.start = QPushButton('Start Auto Enhancement')
        self.start.setMinimumHeight(48)
        self.start.clicked.connect(self.start_processing)
        self.cancel = QPushButton('Cancel')
        self.cancel.setMinimumHeight(48)
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_processing)
        row.addWidget(self.start, 1)
        row.addWidget(self.cancel)
        layout.addLayout(row)

        repair = QPushButton('Open Repair Studio')
        repair.clicked.connect(self.open_repair)
        layout.addWidget(repair)

        self.progress = QProgressBar()
        self.status = QLabel('Ready')
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(QLabel('Auto processing report'))
        layout.addWidget(self.log)
        layout.addWidget(QLabel('Originals are never overwritten. Processing remains local on this PC.'))
        self.setCentralWidget(central)
        self.mode_changed('Auto Detect')

    def mode_changed(self, mode: str) -> None:
        descriptions = {
            'Auto Detect': 'Recommended. Inspects every image and automatically builds the safest repair plan.',
            'Auto Enhance': 'Light professional colour, lighting and contrast for already-good photographs.',
            'Auto Restore': 'Stronger restoration for damaged, faded, compressed, noisy or blurred photographs.',
            'Auto Portrait': 'Face-aware lighting and detail recovery with Face Identity Lock.',
            'Auto Celebrations': 'Consistent lighting and colour for weddings, christenings, birthdays and parties.',
            'Auto Landscape': 'Natural scenery and travel enhancement.',
            'Auto Low Light': 'Noise reduction and shadow recovery for dark photographs.',
            'Auto Screenshot Recovery': 'Repairs compressed shared images and removes obvious interface borders.',
            'Advanced': 'Shows additional controls while keeping automatic analysis active.',
        }
        self.mode_help.setText(descriptions.get(mode, ''))
        self.advanced.setVisible(mode == 'Advanced')

    @staticmethod
    def image_filter() -> str:
        return 'Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)'

    def select_one(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Select a photo', '', self.image_filter())
        if path:
            self.set_files([Path(path)])

    def select_many(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, 'Select photos', '', self.image_filter())
        if paths:
            self.set_files([Path(path) for path in paths])

    def select_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Select photo folder')
        if path:
            folder = Path(path).resolve()
            output = folder / OUTPUT_FOLDER
            files = []
            for candidate in folder.rglob('*'):
                if not candidate.is_file() or not supported_image(candidate):
                    continue
                try:
                    candidate.relative_to(output)
                    continue
                except ValueError:
                    files.append(candidate.resolve())
            self.selected_sources = sorted(files)
            self.folder_root = folder
            self.selection_label.setText(f'{folder.name}: {len(files)} photo(s) selected')

    def set_files(self, paths: list[Path]) -> None:
        self.selected_sources = sorted({p.resolve() for p in paths if p.is_file() and supported_image(p)})
        self.folder_root = None
        self.selection_label.setText(
            self.selected_sources[0].name if len(self.selected_sources) == 1
            else f'{len(self.selected_sources)} photos selected'
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        folders = [p for p in paths if p.is_dir()]
        files = [p for p in paths if p.is_file() and supported_image(p)]
        if len(folders) == 1 and not files:
            folder = folders[0].resolve()
            output = folder / OUTPUT_FOLDER
            self.selected_sources = [p.resolve() for p in folder.rglob('*') if p.is_file() and supported_image(p) and output not in p.parents]
            self.folder_root = folder
            self.selection_label.setText(f'{folder.name}: {len(self.selected_sources)} photo(s) selected')
        elif files:
            self.set_files(files)
        event.acceptProposedAction()

    def options(self) -> EnhanceOptions:
        mode_alias = {
            'Auto Detect': 'Auto Detect', 'Auto Enhance': 'Auto Enhance',
            'Auto Restore': 'Auto Restore', 'Auto Portrait': 'Portrait',
            'Auto Celebrations': 'Celebrations', 'Auto Landscape': 'Landscape',
            'Auto Low Light': 'Low Light', 'Auto Screenshot Recovery': 'Screenshot Recovery',
            'Advanced': 'Auto Detect',
        }
        strength = {'Natural Finish': 'natural', 'Professional Finish': 'strong',
                    'Maximum Recovery': 'maximum'}[self.strength.currentText()]
        if self.mode.currentText() == 'Auto Restore':
            strength = 'maximum'
        return EnhanceOptions(
            preset=mode_alias[self.mode.currentText()], strength=strength,
            upscale=self.upscale.currentText(), lift_shadows=True,
            recover_highlights=True, reduce_flare=self.flare.isChecked(),
            denoise=True, sharpen=True, face_aware=True, auto_rotate=True,
            straighten_horizon=self.straighten.isChecked(), portrait_finish=True,
            neural_ai=True, automatic_restoration=True,
            remove_screenshot_ui=self.remove_ui.isChecked(),
            identity_lock=self.identity_lock.isChecked(), good_photo_polish=True,
            jpeg_quality=self.quality.value(),
        )

    def start_processing(self) -> None:
        if not self.selected_sources:
            QMessageBox.information(self, APP_NAME, 'Select a photo, several photos, or a folder first.')
            return
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        self.progress.setValue(0)
        self.log.clear()
        self.log.append(f'{APP_NAME} v{APP_VERSION}')
        self.log.append(f'Auto Engine v{AUTO_ENGINE_VERSION}')
        self.log.append(self.auto_pack_status())
        self.log.append(f'Starting {self.mode.currentText()} for {len(self.selected_sources)} photo(s)...')
        self.worker = SelectionWorker(self.selected_sources, self.folder_root, self.options(), self.signals)
        self.worker.start()

    def cancel_processing(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.cancel.setEnabled(False)

    def on_progress(self, current: int, total: int, filename: str) -> None:
        self.progress.setValue(int(current / max(total, 1) * 100))
        self.status.setText(f'Processing {current}/{total}: {filename}')

    def on_report(self, text: str) -> None:
        self.log.append(text)

    def on_finished(self, completed: int, review: int, status: str) -> None:
        self.start.setEnabled(True)
        self.cancel.setEnabled(False)
        if status == 'Complete':
            self.progress.setValue(100)
        message = f'{status}: {completed} processed, {review} placed in Review Needed.'
        self.status.setText(message)
        self.log.append(message)
        QMessageBox.information(self, APP_NAME, message)

    def on_failed(self, message: str) -> None:
        self.start.setEnabled(True)
        self.cancel.setEnabled(False)
        self.status.setText('Processing failed')
        self.log.append(message)
        QMessageBox.critical(self, APP_NAME, message)

    def open_repair(self) -> None:
        editor = RepairEditor(parent=self)
        editor.show()
        self.editor_windows.append(editor)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
