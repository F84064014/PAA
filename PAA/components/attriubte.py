from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QCheckBox,
    QScrollArea, QGridLayout, QHBoxLayout, QComboBox
)
from PAA.backend import (
    Attributes
)
import numpy as np

class AttributeLabel(QScrollArea):
    def __init__(self, attributes: Attributes, splits: list[str]):
        super().__init__()
        container:          QWidget               = QWidget()
        self.layout:        QVBoxLayout           = QVBoxLayout(container)
        self.cbs:           dict[str, QCheckBox]  = {}
        self.split_combo:   QComboBox             = None
        self.index_cbs:     list[QCheckBox]       = [None] * len(attributes)
        self.cols:          int                   = 4

        self.setWidgetResizable(True)

        for group_box in self.build_attributes_cbs(attributes):
            self.layout.addWidget(group_box)
        self.layout.addWidget(self.build_split_selector(splits))

        self.layout.addStretch()
        self.setWidget(container)
    
    def loadLabel(self, label: np.ndarray, split: str):
        for l, cb in zip(label, self.index_cbs):
            cb.setChecked(l > 0)
        self.split_combo.setCurrentText(split)

    def getLabel(self) -> tuple[np.ndarray, str]:
        label = np.array([cb.isChecked() for cb in self.index_cbs], dtype=int)
        split = self.split_combo.currentText()
        return (label, split)

    def setCheckboxStyle(self, name: str, cb: QCheckBox):
        if name in ['Black', 'Blue', 'Brown', 'Green',
                    'Grey', 'Orange', 'Pink', 'Purple',
                    'Red', 'White', 'Yellow']:
            drak_color = ['Black', 'Blue', 'Purple', 'Red', 'Brown', 'Green']
            cb.setStyleSheet(f"""QCheckBox{{
                                background-color: {name};
                                font-weight: bold;
                                color: {'White' if name in drak_color else 'Black'};
                                padding: 6px;
                             }}
                             QCheckBox:checked {{
                                border: 2px solid {'White' if name in drak_color else 'Black'};
                             }}
                             """)

    def build_split_selector(self, splits: list[str]) -> QGroupBox:
        box = QGroupBox("Partition")
        layout = QHBoxLayout(box)

        self.split_combo = QComboBox()
        self.split_combo.addItems(splits)

        layout.addWidget(self.split_combo)
        return box

    def build_attributes_cbs(self, attributes: Attributes) -> list[QGroupBox]:
        group_boxes = []
        for group_name, attr_list in attributes.group():
            group_box = QGroupBox(group_name)
            group_layout = QGridLayout()

            for i, (idx, attr) in enumerate(attr_list):
                cb = QCheckBox(attr)

                self.setCheckboxStyle(attr, cb)
                cb_key = (group_name, attr, idx)
                self.cbs[cb_key] = cb
                self.index_cbs[idx] = cb

                row, col = divmod(i, self.cols)
                group_layout.addWidget(cb, row, col)

            group_box.setLayout(group_layout)
            group_boxes.append(group_box)
        return group_boxes

    @property
    def label_array(self) -> np.ndarray:
        arr = np.array(len(self.cbs), dtype=int)
        for (_, _, idx), cb in self.cbs.items():
            arr[idx] = cb.isChecked()
        return arr