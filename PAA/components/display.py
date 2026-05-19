from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QPen, QImage
from PyQt6.QtWidgets import (
    QWidget,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel
)
import cv2
import numpy as np
from ultralytics import YOLO

class ImageLabel(QWidget):
    
    def __init__(self) -> None:
        super().__init__()

        self.image_view = ImageView()
        self.main_layout = QVBoxLayout(self)

        self.size_btn = QPushButton("Resize")
        self.size_btn.clicked.connect(self.image_view.toggle_size_mode)

        self.brightness_label  = QLabel("Brightness")
        self.brightness_label.setStyleSheet('color: white')
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(30)
        self.brightness_slider.setMaximum(200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(
            self.image_view.set_brightness
        )
        self.brightness_layout = QHBoxLayout()
        self.brightness_layout.addWidget(self.brightness_label)
        self.brightness_layout.addWidget(self.brightness_slider)

        self.mask_btn = QPushButton("Mask")
        self.mask_btn.clicked.connect(self.image_view.toggle_mask_mode)

        self.face_btn = QPushButton("Face")
        self.face_btn.clicked.connect(self.image_view.toggle_face_mode)

        self.main_layout.addWidget(self.image_view)
        self.main_layout.addWidget(self.size_btn)
        self.main_layout.addLayout(self.brightness_layout)
        self.main_layout.addWidget(self.mask_btn)
        self.main_layout.addWidget(self.face_btn)

    def loadImage(self, image_path):
        self.image_view.loadImage(image_path)

    def loadMask(self, mask_path):
        self.image_view.loadMask(mask_path)

    def loadFace(self, face):
        self.image_view.loadFace(face)

class ImageView(QGraphicsView):

    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = None
        self.origional_image: np.ndarray = None
        self.origional_mask : np.ndarray = None
        self.origional_face : np.ndarray = None

        self.fixed_size_mode = False
        self.brightness = 1.0
        self.mask_item: QGraphicsPixmapItem = None
        self.face_item  = None

        self.start_pos = None
        self.current_rect = None

        self.setRenderHints(self.renderHints())
        self.setMouseTracking(True)
        self.setStyleSheet("""
                           background-color: #2b2b2b}
                           """)

    def loadImage(self, image_path):

        self.origional_image = cv2.imread(image_path)
        self.update_pixmap()

    def loadMask(self, mask_path=None):
        if mask_path is None:
            return
        self.origional_mask = cv2.imread(mask_path).any(-1)
        self.update_mask()

    def loadFace(self, face: np.ndarray):
        if face is None:
            return
        self.origional_face = face
        self.update_face()

    def update_mask(self):
        mask = self.origional_mask
        if mask is None:
            return

        H, W  = self.origional_image.shape[:2]
        size  = (512, 256) if self.fixed_size_mode else (H, W)

        if self.fixed_size_mode:
            mask = (cv2.resize(mask.astype(np.uint8), size[::-1]) > 0.5)

        # RGBA overlay
        overlay = np.zeros((*size, 4), dtype=np.uint8)
        overlay[mask] = [255, 0, 0, 120]

        qimage = QImage(overlay.data, size[1], size[0], 4 * size[1],
                        QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)

        if self.mask_item is not None:
            try:
                self.scene.removeItem(self.mask_item)
            except RuntimeError:
                pass
            self.mask_item = None

        self.mask_item = QGraphicsPixmapItem(pixmap)
        self.mask_item.setZValue(10)
        self.scene.addItem(self.mask_item)

    def update_face(self):
        face = self.origional_face

        if self.face_item is not None:
            try:
                self.scene.removeItem(self.face_item)
            except RuntimeError:
                pass
            self.face_item = None

        self.face_item = QGraphicsRectItem(
            face[0], face[1], face[2], face[3])
        self.face_item.setPen(QPen(Qt.GlobalColor.green, 2))
        self.scene.addItem(self.face_item)

    def mousePressEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:

            self.start_pos = self.mapToScene(event.pos())

            self.current_rect = QGraphicsRectItem()
            self.current_rect.setPen(QPen(Qt.GlobalColor.red, 2))

            self.scene.addItem(self.current_rect)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):

        if self.start_pos and self.current_rect:

            current_pos = self.mapToScene(event.pos())

            rect = QRectF(self.start_pos, current_pos).normalized()

            self.current_rect.setRect(rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton:

            self.start_pos = None

        super().mouseReleaseEvent(event)

    def toggle_size_mode(self):
        self.fixed_size_mode = not self.fixed_size_mode
        self.update_pixmap()
        self.update_mask()

    def toggle_mask_mode(self):
        if self.mask_item is not None:
            self.mask_item.setVisible(not self.mask_item.isVisible())

    def toggle_face_mode(self):
        if self.face_item is not None:
            self.face_item.setVisible(not self.face_item.isVisible())


    def set_brightness(self, value):
        self.brightness = value / 100.0
        self.update_pixmap()

    def update_pixmap(self):

        if self.origional_image is None:
            return

        image = self.origional_image.copy()

        image = cv2.convertScaleAbs(
            image, alpha=self.brightness, beta=0)

        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape

        qimage = QImage(image.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        if self.fixed_size_mode:

            pixmap = pixmap.scaled(
                256, 512,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        self.scene.clear()

        self.pixmap_item = QGraphicsPixmapItem(
            pixmap
        )

        self.scene.addItem(
            self.pixmap_item
        )

        self.setSceneRect(
            QRectF(pixmap.rect())
        )