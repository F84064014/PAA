from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QGridLayout
)
import numpy as np

class FilterPanel(QWidget):
    def __init__(self, attributes: list[str]):
        super().__init__()
        self.cbs :list[QCheckBox] = list()

        self.setWindowFlags(Qt.WindowType.Popup)
        self.setStyleSheet("background-color: silver; color: black")

        layout = QGridLayout(self)

        for i, name in enumerate(attributes):
            cb = QCheckBox(name)
            cb.setTristate(False)
            cb.setCheckState(Qt.CheckState.Unchecked)

            row, col = divmod(i, 3)
            layout.addWidget(cb, row, col)
            self.cbs.append(cb)

    @property
    def mask(self) -> np.ndarray:
        return np.array([cb.isChecked() for cb in self.cbs], dtype=int)