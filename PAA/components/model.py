from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QGridLayout,
    QVBoxLayout, QLabel, QGroupBox
)
from PAA.dataset import (
    Attributes
)
import cv2
import numpy as np
import onnxruntime as ort

class ModelPanel(QWidget):
    def __init__(self, model: str, attributes: Attributes):
        super().__init__()
        self.lbs        :list[QLabel] = list()
        self.layout     :QVBoxLayout  = QVBoxLayout(self)

        self.model = ort.InferenceSession(
            model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.setWindowTitle("Model Panel")
        self.setStyleSheet("background-color: #3b3b3b")

        for group_box in self.build_attributes_lbs(attributes):
            self.layout.addWidget(group_box)

    def build_attributes_lbs(self, attributes: Attributes) -> list[QGroupBox]:
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
                self.lbs.append(lb)

                row, col = divmod(i, 4)
                group_layout.addWidget(lb, row, col)

            group_box.setLayout(group_layout)
            group_boxes.append(group_box)
        return group_boxes

    def updatePredict(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        probs = self.inference(img)
        for lb, prob in zip(self.lbs, probs):
            color = ModelPanel.prob_to_color(prob)
            lb.setText(f"{prob:.4f}")
            lb.setStyleSheet(f"color: {color};")

    def prob_to_color(prob: float) -> str:
        prob = max(0.0, min(1.0, prob))
        r = int(255 * (1 - prob))
        g = int(255 * prob)
        b = 0
        return f"rgb({r}, {g}, {b})"

    def inference(self, img):
        input_size = self.model.get_inputs()[0].shape[-2:][::-1]
        rgb = cv2.resize(np.array(img), input_size)
        rgb = rgb.astype(np.uint8)
        rgb = rgb.transpose(2, 0, 1)[None]
        probs = self.model.run(None, {'input': rgb})[0][0]
        return probs