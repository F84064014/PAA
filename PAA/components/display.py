from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPaintEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel
from pathlib import Path

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.start = None
        self.end   = None
        self.pixmap_image = None
        self.pixmap_w = 256
        self.pixmap_h = 512
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def loadImage(self, image_path):
        if not Path(image_path).exists():
            print(f"[ERROR] ImagePath={image_path} not exist")
            return
        pixmap_image = QPixmap(image_path)
        scaled_pixmap = pixmap_image.scaled(
            self.pixmap_w, self.pixmap_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled_pixmap)

    def resetBBox(self):
        self.start = None
        self.end   = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = event.pos()
            self.end   = self.start
        if event.button() == Qt.MouseButton.RightButton:
            self.start = None
            self.end = None
        self.update()

    def mouseMoveEvent(self, event):
        if self.start:
            self.end = event.pos()
            self.update()

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.GlobalColor.red, 2)
        painter.setPen(pen)

        if self.start and self.end:
            rect = QRect(self.start, self.end)
            painter.drawRect(rect.normalized())

        painter.end()

    @property
    def xyxyn(self):
        if self.start and self.end:
            x1, y1 = self.start.x(), self.start.y()
            x2, y2 = self.end.x(), self.end.y()

            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            img_w = self.width()  #self.pixmap().width()
            img_h = self.height() #self.pixmap().height()

            bbox = (x_min / img_w, y_min / img_h, x_max / img_w, y_max / img_h)
        else:
            bbox = (0.0, 0.0, 1.0, 1.0)

        return bbox
