from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QApplication, QMainWindow, QLineEdit,
    QLabel, QHBoxLayout, QVBoxLayout
)
from .components import (
    ImageLabel, AttributeLabel
)
from .dataset import (
    load_data
)
from datetime import datetime

class Annotator(QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PAA Tool")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setObjectName("MainWindow")
        self.setStyleSheet("""
            QWidget#MainWindow {
                           background-color: #2b2b2b}
                           """)
        self.setFixedSize(1000, 800)

        # Loading dataset
        self.dataset = load_data("data/dataset_all.pkl")
        # self.dataset = load_data("data/2026_04_18_19_26_temp.csv")

        self.imageLabel     :ImageLabel
        self.attributeLabel :AttributeLabel

        self.index_edit = QLineEdit(str(0))
        self.index_edit.setFixedWidth(50)
        self.index_edit.setStyleSheet("color: white; background-color: #3b3b3b;")
        self.index_edit.returnPressed.connect(self.load_image)

        self.total_label = QLabel(f" / {len(self.dataset)}")
        self.total_label.setStyleSheet("color: white;")

        top_layout = QHBoxLayout()
        top_layout.addStretch()
        top_layout.addWidget(self.index_edit)
        top_layout.addWidget(self.total_label)
        top_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(self.buildImageAndAttribute())

        self.setLayout(main_layout)

        self.load_image()

    @property
    def cur_index(self):
        return int(self.index_edit.text())

    @cur_index.setter
    def cur_index(self, value: int):
        value = min(max(value, 0), len(self.dataset)-1)
        self.index_edit.setText(str(value))

    def load_image(self):
        if self.cur_index < 0:
            self.cur_index = 0
        if self.cur_index >= len(self.dataset):
            self.cur_index = len(self.dataset)-1
        self.imageLabel.loadImage(self.dataset.get_image(self.cur_index))
        self.attributeLabel.loadLabel(self.dataset.labels[self.cur_index])

    def save_dataset(self):
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
        self.dataset.save_csv(f"{timestamp}_temp.csv")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_N:
            self.cur_index += 1
            self.load_image()
        if event.key() == Qt.Key.Key_U:
            self.cur_index -= 1
            self.load_image()
        if event.key() == Qt.Key.Key_S:
            self.save_dataset()

    def buildImageAndAttribute(self) -> QHBoxLayout:

        self.imageLabel     = ImageLabel()
        self.attributeLabel = AttributeLabel(self.dataset.attriubte_names)

        layout = QHBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.attributeLabel)

        return layout

    def buildMenuBar():
        pass