from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QGridLayout,
    QGroupBox, QLabel, QVBoxLayout,
    QHBoxLayout, QComboBox
)
from PAA.backend import (
    Attributes
)
import numpy as np

class FilterPanel(QWidget):
    def __init__(self, attributes: Attributes, splits: list[str]):
        super().__init__()
        self.attr_cbs   :list[QCheckBox]    = list()
        self.split_cbs  :list[QCheckBox]    = list()
        self.layout     :QVBoxLayout        = QVBoxLayout(self)

        self.setWindowFlags(Qt.WindowType.Popup)
        self.setStyleSheet("background-color: silver; color: black")

        for group_box in self.build_attributes_cbs(attributes):
            self.layout.addWidget(group_box)
        self.layout.addWidget(self.build_split_cbs(splits))

    def build_attributes_cbs(self, attributes: Attributes) -> list[QGroupBox]:
        group_boxes = []
        for group_name, attr_list in attributes.group():
            group_box = QGroupBox(group_name)
            group_layout = QGridLayout()

            for i, (idx, attr) in enumerate(attr_list):
                cb = QCheckBox(attr)
                cb.setCheckState(Qt.CheckState.Unchecked)
                self.attr_cbs.append(cb)

                row, col = divmod(i, 4)
                group_layout.addWidget(cb, row, col)

            group_box.setLayout(group_layout)
            group_boxes.append(group_box)
        return group_boxes

    def build_split_cbs(self, splits: list[str]):
        box = QGroupBox("Partition")
        layout = QHBoxLayout(box)
        for split in splits:
            cb = QCheckBox(split)
            cb.setChecked(True)
            self.split_cbs.append(cb)
            layout.addWidget(cb)
        return box

    @property
    def mask(self) -> np.ndarray:
        return np.array([cb.isChecked() for cb in self.attr_cbs], dtype=int)
    
    @property
    def split(self) -> set:
        return set([cb.text() for cb in self.split_cbs if cb.isChecked()])