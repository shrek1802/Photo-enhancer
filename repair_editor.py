from __future__ import annotations

import json
import urllib.request
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget
)

RELEASES_URL = 'https://github.com/shrek1802/Photo-enhancer/releases'
LATEST_API = 'https://api.github.com/repos/shrek1802/Photo-enhancer/releases/latest'


def cv_to_qimage(image: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, channels = rgb.shape
    return QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888).copy()


class ImageCanvas(QWidget):
    mask_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.brush_size = 34
        self.zoom = 1.0
        self.compare_percent = 100
        self.drawing = False
        self.last_point: QPoint | None = None
        self.setMinimumSize(720, 480)
        self.setMouseTracking(True)

    def set_images(self, original: np.ndarray, current: np.ndarray | None = None) -> None:
        self.original = original.copy()
        self.current = (current if current is not None else original).copy()
        self.mask = np.zeros(original.shape[:2], np.uint8)
        self.fit_image()
        self.update()

    def fit_image(self) -> None:
        if self.current is None:
            return
        h, w = self.current.shape[:2]
        self.zoom = min(max(self.width(), 1) / w, max(self.height(), 1) / h, 1.0)

    def image_rect(self) -> QRect:
        if self.current is None:
            return QRect()
        h, w = self.current.shape[:2]
        dw, dh = int(w * self.zoom), int(h * self.zoom)
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)

    def widget_to_image(self, point: QPoint) -> tuple[int, int] | None:
        if self.current is None:
            return None
        rect = self.image_rect()
        if not rect.contains(point):
            return None
        x = int((point.x() - rect.x()) / self.zoom)
        y = int((point.y() - rect.y()) / self.zoom)
        h, w = self.current.shape[:2]
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.current is None or self.original is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, 'Open a photograph to begin')
            return

        split = int(self.current.shape[1] * self.compare_percent / 100)
        display = self.original.copy()
        display[:, :split] = self.current[:, :split]
        image = cv_to_qimage(display)
        rect = self.image_rect()
        painter.drawImage(rect, image)

        if self.mask is not None and np.any(self.mask):
            overlay = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
            overlay[self.mask > 0] = (255, 70, 70, 120)
            rgba = QImage(overlay.data, overlay.shape[1], overlay.shape[0], overlay.strides[0], QImage.Format_RGBA8888).copy()
            painter.drawImage(rect, rgba)

        if 0 < self.compare_percent < 100:
            x = rect.x() + int(rect.width() * self.compare_percent / 100)
            painter.setPen(QPen(Qt.white, 2))
            painter.drawLine(x, rect.y(), x, rect.bottom())
            painter.drawText(x + 7, rect.y() + 22, 'Enhanced')
            painter.drawText(max(rect.x() + 7, x - 65), rect.y() + 22, 'Original')

    def _draw_to(self, point: QPoint) -> None:
        if self.mask is None:
            return
        mapped = self.widget_to_image(point)
        if mapped is None:
            return
        x, y = mapped
        radius = max(2, int(self.brush_size / max(self.zoom, 0.05) / 2))
        if self.last_point is not None:
            last = self.widget_to_image(self.last_point)
            if last:
                cv2.line(self.mask, last, (x, y), 255, radius * 2, cv2.LINE_AA)
        cv2.circle(self.mask, (x, y), radius, 255, -1, cv2.LINE_AA)
        self.last_point = point
        self.mask_changed.emit()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = None
            self._draw_to(event.position().toPoint())

    def mouseMoveEvent(self, event) -> None:
        if self.drawing:
            self._draw_to(event.position().toPoint())

    def mouseReleaseEvent(self, event) -> None:
        self.drawing = False
        self.last_point = None

    def wheelEvent(self, event) -> None:
        self.zoom = max(0.08, min(6.0, self.zoom * (1.15 if event.angleDelta().y() > 0 else 0.87)))
        self.update()


class RepairEditor(QMainWindow):
    def __init__(self, initial_path: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('PhotoPerfect Manual Repair Studio')
        self.resize(1100, 780)
        self.path: Path | None = None
        self.original: np.ndarray | None = None
        self.history: list[np.ndarray] = []
        self.build_ui()
        if initial_path and initial_path.exists():
            self.load_path(initial_path)

    def build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        open_button = QPushButton('Open Photo')
        open_button.clicked.connect(self.open_photo)
        save_button = QPushButton('Save Repaired Copy')
        save_button.clicked.connect(self.save_photo)
        undo_button = QPushButton('Undo')
        undo_button.clicked.connect(self.undo)
        reset_button = QPushButton('Reset')
        reset_button.clicked.connect(self.reset)
        update_button = QPushButton('Check for Updates')
        update_button.clicked.connect(self.check_updates)
        for button in (open_button, save_button, undo_button, reset_button, update_button):
            top.addWidget(button)
        top.addStretch(1)
        layout.addLayout(top)

        tools = QHBoxLayout()
        tools.addWidget(QLabel('Brush:'))
        self.brush = QSpinBox()
        self.brush.setRange(4, 250)
        self.brush.setValue(34)
        self.brush.valueChanged.connect(self.set_brush)
        tools.addWidget(self.brush)

        clear_mask = QPushButton('Clear Brush Mask')
        clear_mask.clicked.connect(self.clear_mask)
        repair = QPushButton('Repair Brushed Area')
        repair.clicked.connect(self.inpaint)
        auto_spots = QPushButton('Detect Dust / Bright Spots')
        auto_spots.clicked.connect(self.detect_spots)
        lighten = QPushButton('Lift Brushed Shadow')
        lighten.clicked.connect(self.lift_shadow)
        soften = QPushButton('Soften Brushed Area')
        soften.clicked.connect(self.soften_area)
        for button in (clear_mask, repair, auto_spots, lighten, soften):
            tools.addWidget(button)
        layout.addLayout(tools)

        crop_row = QHBoxLayout()
        crop_row.addWidget(QLabel('Crop:'))
        self.crop_mode = QComboBox()
        self.crop_mode.addItems(['Keep current', 'Square 1:1', 'Portrait 4:5', 'Photo 3:2', 'Widescreen 16:9'])
        crop_row.addWidget(self.crop_mode)
        crop_button = QPushButton('Centre Crop')
        crop_button.clicked.connect(self.crop_image)
        crop_row.addWidget(crop_button)
        crop_row.addStretch(1)
        crop_row.addWidget(QLabel('Compare:'))
        self.compare = QSlider(Qt.Horizontal)
        self.compare.setRange(0, 100)
        self.compare.setValue(100)
        self.compare.setMinimumWidth(240)
        self.compare.valueChanged.connect(self.set_compare)
        crop_row.addWidget(self.compare)
        layout.addLayout(crop_row)

        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)
        hint = QLabel('Paint over lens flare, unwanted shadows, scratches or objects, then choose a repair action. Mouse wheel zooms. Originals are never overwritten.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.setCentralWidget(root)

    def open_photo(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, 'Open photograph', '', 'Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)')
        if filename:
            self.load_path(Path(filename))

    def load_path(self, path: Path) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.critical(self, 'Open failed', 'The photograph could not be opened.')
            return
        self.path = path
        self.original = image.copy()
        self.history.clear()
        self.canvas.set_images(image)
        self.setWindowTitle(f'PhotoPerfect Manual Repair Studio — {path.name}')

    def push_history(self) -> bool:
        if self.canvas.current is None:
            return False
        self.history.append(self.canvas.current.copy())
        if len(self.history) > 20:
            self.history.pop(0)
        return True

    def set_brush(self, value: int) -> None:
        self.canvas.brush_size = value

    def set_compare(self, value: int) -> None:
        self.canvas.compare_percent = value
        self.canvas.update()

    def clear_mask(self) -> None:
        if self.canvas.mask is not None:
            self.canvas.mask.fill(0)
            self.canvas.update()

    def inpaint(self) -> None:
        if self.canvas.current is None or self.canvas.mask is None or not np.any(self.canvas.mask):
            QMessageBox.information(self, 'Repair', 'Brush over the area to repair first.')
            return
        self.push_history()
        mask = cv2.dilate(self.canvas.mask, np.ones((3, 3), np.uint8), iterations=1)
        self.canvas.current = cv2.inpaint(self.canvas.current, mask, 4, cv2.INPAINT_TELEA)
        self.clear_mask()

    def detect_spots(self) -> None:
        if self.canvas.current is None:
            return
        image = self.canvas.current
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        local = cv2.medianBlur(gray, 21)
        difference = cv2.absdiff(gray, local)
        mask = ((difference > 38) & ((gray > 220) | (gray < 25))).astype(np.uint8) * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        safe = np.zeros_like(mask)
        total = gray.size
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if 4 <= area <= total * 0.0015:
                safe[labels == label] = 255
        self.canvas.mask = cv2.dilate(safe, np.ones((3, 3), np.uint8), iterations=1)
        self.canvas.update()

    def lift_shadow(self) -> None:
        if self.canvas.current is None or self.canvas.mask is None or not np.any(self.canvas.mask):
            QMessageBox.information(self, 'Shadow repair', 'Brush over the unwanted shadow first.')
            return
        self.push_history()
        image = self.canvas.current
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lifted = np.clip(l.astype(np.float32) * 1.18 + 5, 0, 255).astype(np.uint8)
        corrected = cv2.cvtColor(cv2.merge([lifted, a, b]), cv2.COLOR_LAB2BGR)
        alpha = cv2.GaussianBlur(self.canvas.mask, (0, 0), 12).astype(np.float32) / 255.0
        alpha = alpha[..., None]
        self.canvas.current = np.clip(image * (1 - alpha) + corrected * alpha, 0, 255).astype(np.uint8)
        self.clear_mask()

    def soften_area(self) -> None:
        if self.canvas.current is None or self.canvas.mask is None or not np.any(self.canvas.mask):
            QMessageBox.information(self, 'Soften', 'Brush over the area first.')
            return
        self.push_history()
        blurred = cv2.bilateralFilter(self.canvas.current, 9, 35, 35)
        alpha = cv2.GaussianBlur(self.canvas.mask, (0, 0), 8).astype(np.float32) / 255.0
        alpha = (alpha * 0.55)[..., None]
        self.canvas.current = np.clip(self.canvas.current * (1 - alpha) + blurred * alpha, 0, 255).astype(np.uint8)
        self.clear_mask()

    def crop_image(self) -> None:
        if self.canvas.current is None or self.crop_mode.currentText() == 'Keep current':
            return
        ratios = {'Square 1:1': 1.0, 'Portrait 4:5': 4 / 5, 'Photo 3:2': 3 / 2, 'Widescreen 16:9': 16 / 9}
        target = ratios[self.crop_mode.currentText()]
        image = self.canvas.current
        h, w = image.shape[:2]
        current = w / h
        if current > target:
            new_w = int(h * target)
            x = (w - new_w) // 2
            cropped = image[:, x:x + new_w]
        else:
            new_h = int(w / target)
            y = (h - new_h) // 2
            cropped = image[y:y + new_h, :]
        self.push_history()
        self.canvas.current = cropped.copy()
        self.canvas.mask = np.zeros(cropped.shape[:2], np.uint8)
        self.canvas.fit_image()
        self.canvas.update()

    def undo(self) -> None:
        if self.history:
            self.canvas.current = self.history.pop()
            self.canvas.mask = np.zeros(self.canvas.current.shape[:2], np.uint8)
            self.canvas.update()

    def reset(self) -> None:
        if self.original is not None:
            self.history.clear()
            self.canvas.set_images(self.original)

    def save_photo(self) -> None:
        if self.canvas.current is None:
            return
        suggested = 'repaired_photo.jpg' if self.path is None else f'{self.path.stem}_repaired.jpg'
        filename, _ = QFileDialog.getSaveFileName(self, 'Save repaired copy', str((self.path.parent if self.path else Path.home()) / suggested), 'JPEG (*.jpg);;PNG (*.png)')
        if filename:
            ext = Path(filename).suffix.lower()
            params = [cv2.IMWRITE_JPEG_QUALITY, 96] if ext in {'.jpg', '.jpeg'} else []
            if not cv2.imwrite(filename, self.canvas.current, params):
                QMessageBox.critical(self, 'Save failed', 'The repaired photograph could not be saved.')

    def check_updates(self) -> None:
        try:
            request = urllib.request.Request(LATEST_API, headers={'User-Agent': 'PhotoPerfect-Batch-AI'})
            with urllib.request.urlopen(request, timeout=8) as response:
                data = json.loads(response.read().decode('utf-8'))
            tag = data.get('tag_name', 'latest')
            answer = QMessageBox.question(self, 'Update check', f'Latest published release: {tag}\n\nOpen the Releases page?')
            if answer == QMessageBox.Yes:
                webbrowser.open(RELEASES_URL)
        except Exception as exc:
            QMessageBox.warning(self, 'Update check', f'Could not check GitHub Releases.\n\n{exc}')
