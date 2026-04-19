from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QCheckBox,
    QScrollArea, QGridLayout
)
import numpy as np

class AttributeLabel(QScrollArea):
    def __init__(self, attributes):
        super().__init__()
        self.setWidgetResizable(True)

        container:      QWidget               = QWidget()
        self.layout:    QVBoxLayout           = QVBoxLayout(container)
        self.cbs:       dict[str, QCheckBox]  = {}
        self.index_cbs: list[QCheckBox]       = [None] * len(attributes)
        self.cols:      int                   = 4

        grouped_attributes = self.getAttributesGroup(attributes)
        for group_name, attr_list in grouped_attributes.items():
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
            self.layout.addWidget(group_box)

        self.layout.addStretch()
        self.setWidget(container)

    def getAttributesGroup(self, attributes: list[str]):
        grouped_attributes = dict()
        for idx, attribute in enumerate(attributes):
            group, attr = attribute.rsplit('-', 1)
            if group not in grouped_attributes:
                grouped_attributes[group] = list()
            grouped_attributes[group].append((idx, attr))
        return grouped_attributes
    
    def loadLabel(self, arr: np.ndarray):
        for l, cb in zip(arr, self.index_cbs):
            cb.setChecked(l > 0)

    def setCheckboxStyle(self, name: str, cb: QCheckBox):
        if name in ['Black', 'Blue', 'Brown', 'Green',
                    'Grey', 'Orange', 'Pink', 'Purple',
                    'Red', 'White', 'Yellow']:
            cb.setStyleSheet(f"""QCheckBox{{
                                background-color: {name};
                                font-weight: bold;
                                color: {'White' if name in ['Black'] else 'Black'}
                             }}""")

    @property
    def label_array(self) -> np.ndarray:
        arr = np.array(len(self.cbs), dtype=int)
        for (_, _, idx), cb in self.cbs.items():
            arr[idx] = cb.isChecked()
        return arr