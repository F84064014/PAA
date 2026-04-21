from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QGridLayout,
    QGroupBox, QLabel, QVBoxLayout
)
from PAA.dataset import (
    Attributes
)
import numpy as np

class FilterPanel(QWidget):
    def __init__(self, attributes: Attributes):
        super().__init__()
        self.cbs    :list[QCheckBox]    = list()
        self.layout :QVBoxLayout        = QVBoxLayout(self)

        self.setWindowFlags(Qt.WindowType.Popup)
        self.setStyleSheet("background-color: silver; color: black")

        for group_box in self.build_attributes_cbs(attributes):
            self.layout.addWidget(group_box)

    def build_attributes_cbs(self, attributes: Attributes) -> list[QGroupBox]:
        group_boxes = []
        for group_name, attr_list in attributes.group():
            group_box = QGroupBox(group_name)
            group_layout = QGridLayout()

            for i, (idx, attr) in enumerate(attr_list):
                cb = QCheckBox(attr)
                cb.setCheckState(Qt.CheckState.Unchecked)
                self.cbs.append(cb)

                row, col = divmod(i, 4)
                group_layout.addWidget(cb, row, col)

            group_box.setLayout(group_layout)
            group_boxes.append(group_box)
        return group_boxes

    @property
    def mask(self) -> np.ndarray:
        return np.array([cb.isChecked() for cb in self.cbs], dtype=int)