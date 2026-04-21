from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QGridLayout,
    QVBoxLayout, QLabel, QGroupBox,
    QHBoxLayout
)
from PAA.backend import (
    Attributes, Model
)
import numpy as np

class ModelPanel(QWidget):
    def __init__(self, model: str, attributes: Attributes):
        super().__init__()
        self.pred_lbs   :list[QLabel] = list()
        self.diff_lbs   :list[QLabel] = list()
        self.layout     :QHBoxLayout  = QHBoxLayout(self)
        self.model      :Model        = Model(model)

        self.setWindowTitle("Model Panel")
        self.setStyleSheet("background-color: #3b3b3b")

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Predict"))
        for group_box in self.build_attributes_lbs(attributes, self.pred_lbs):
            left_layout.addWidget(group_box)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("L1 dist"))
        for group_box in self.build_attributes_lbs(attributes, self.diff_lbs):
            right_layout.addWidget(group_box)

        self.layout.addLayout(left_layout)
        self.layout.addLayout(right_layout)

    def build_attributes_lbs(self, attributes: Attributes, lbs: list) -> list[QGroupBox]:
        group_boxes = []
        for group_name, attr_list in attributes.group():
            group_box = QGroupBox(group_name)
            group_box.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    margin-top: 8px;
                }""")
            group_layout = QGridLayout()

            for i, (idx, attr) in enumerate(attr_list):
                lb = QLabel('0.0000')
                lbs.append(lb)

                row, col = divmod(i, 4)
                group_layout.addWidget(lb, row, col)

            group_box.setLayout(group_layout)
            group_boxes.append(group_box)
        return group_boxes

    def updatePredict(self, img: str, labels: np.ndarray=None):
        probs = self.model(img)
        for lb, prob in zip(self.pred_lbs, probs):
            color = ModelPanel.prob_to_color(prob)
            lb.setText(f"{prob:.4f}")
            lb.setStyleSheet(f"color: {color};")

        if labels is not None:
            self.updateDiff(probs, labels)

    def updateDiff(self, probs, labels):
        for lb, diff in zip(self.diff_lbs, probs - labels):
            color = ModelPanel.prob_to_color(1-np.abs(diff))
            lb.setText(f"{diff:+.4f}")
            lb.setStyleSheet(f"color: {color};")

    def prob_to_color(prob: float) -> str:
        prob = max(0.0, min(1.0, prob))
        r = int(255 * (1 - prob))
        g = int(255 * prob)
        b = 0
        return f"rgb({r}, {g}, {b})"