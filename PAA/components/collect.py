import sys
import os
import cv2
import numpy as np

from ultralytics import YOLO

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QAction,
    QPixmap,
    QImage,
    QPainter,
    QPen,
    QColor,
    QKeySequence
)
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsObject
)
from datetime import datetime


class HumanBBoxItem(QGraphicsObject):

    def __init__(self, rect: QRectF, parent=None):
        super().__init__(parent)

        self.rect = rect
        self.close_rect = QRectF(
            rect.right() - 20, rect.top(), 20, 20)

        self.removed = False
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIsSelectable, True)

    def boundingRect(self):
        return self.rect.adjusted(-2, -2, 2, 2)

    def paint(self, painter: QPainter, option, widget=None):

        # Draw bbox
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawRect(self.rect)

        # Draw close button
        painter.setBrush(QColor(255, 0, 0))
        painter.drawRect(self.close_rect)

        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawText(self.close_rect, Qt.AlignmentFlag.AlignCenter, "X")

    def mousePressEvent(self, event):

        if self.close_rect.contains(event.pos()):
            scene = self.scene()
            scene.removeItem(self)
            self.removed = True
            return

        super().mousePressEvent(event)

    def get_xyxy(self):
        return (
            int(self.rect.left()),
            int(self.rect.top()),
            int(self.rect.right()),
            int(self.rect.bottom())
        )


class AnnotatorWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PAR Annotator")
        self.resize(1400, 900)

        self.model = YOLO("yolo26l.pt")

        self.image_np = None
        self.pixmap_item = None
        self.save_dir = "./crops"
        self.bbox_items : list[HumanBBoxItem] = []

        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()

        self.paste_btn = QPushButton("Paste")
        self.detect_btn = QPushButton("Detect Human")
        self.export_btn = QPushButton("Export Crops")
        self.savedir_btn = QPushButton("Set Save Dir")

        toolbar.addWidget(self.paste_btn)
        toolbar.addWidget(self.detect_btn)
        toolbar.addWidget(self.export_btn)
        toolbar.addWidget(self.savedir_btn)

        layout.addLayout(toolbar)

        # Graphics view
        self.scene = QGraphicsScene()

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout.addWidget(self.view)

        # Connect signals
        self.paste_btn.clicked.connect(self.paste_image)
        self.detect_btn.clicked.connect(self.detect_human)
        self.export_btn.clicked.connect(self.export_crops)
        self.savedir_btn.clicked.connect(self.set_savedir)

        # Ctrl+V shortcut
        paste_action = QAction(self)
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(self.paste_image)

        self.addAction(paste_action)

    def paste_image(self):

        clipboard = QApplication.clipboard()

        if not clipboard.mimeData().hasImage():
            print("Clipboard has no image")
            return

        qimage = clipboard.image()

        if qimage.isNull():
            print("Invalid image")
            return

        self.scene.clear()
        self.bbox_items.clear()

        pixmap = QPixmap.fromImage(qimage)

        self.pixmap_item = self.scene.addPixmap(pixmap)

        self.image_np = self.qimage_to_numpy(qimage)

        self.scene.setSceneRect(self.pixmap_item.boundingRect())

    def detect_human(self):

        if self.image_np is None:
            return

        results = self.model(self.image_np)

        for item in self.bbox_items:
            self.scene.removeItem(item)

        self.bbox_items.clear()

        for box in results[0].boxes:

            cls_id = int(box.cls[0])

            # COCO class 0 = person
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            rect = QRectF(
                x1,
                y1,
                x2 - x1,
                y2 - y1
            )

            bbox_item = HumanBBoxItem(rect)

            self.scene.addItem(bbox_item)

            self.bbox_items.append(bbox_item)

    def export_crops(self):

        if self.image_np is None:
            return

        if not self.save_dir:
            return
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, item in enumerate(self.bbox_items):
            if item.removed:
                continue

            x1, y1, x2, y2 = item.get_xyxy()

            crop = self.image_np[y1:y2, x1:x2]

            save_path = os.path.join(
                self.save_dir, f"{timestamp}_crop_{idx}.png")

            cv2.imwrite(save_path, crop)

        print("Export done")

    def set_savedir(self):
        self.save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory"
        )


    def qimage_to_numpy(self, qimage):

        qimage = qimage.convertToFormat(
            QImage.Format.Format_RGBA8888
        )

        width = qimage.width()
        height = qimage.height()

        ptr = qimage.bits()
        ptr.setsize(height * width * 4)

        arr = np.frombuffer(ptr, np.uint8).reshape(
            (height, width, 4)
        )

        # Convert RGB -> BGR for OpenCV/YOLO
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        return arr.copy()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = AnnotatorWidget()
    window.show()

    sys.exit(app.exec())