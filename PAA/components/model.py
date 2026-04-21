from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QCheckBox, QGridLayout
)
import cv2
import numpy as np
import onnxruntime as ort

class ModelPanel(QWidget):
    def __init__(self, model: str, attributes: list[str]):
        super().__init__()

        self.model = ort.InferenceSession(
            model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        
    def inference(self, img):
        input_size = self.model.get_inputs()[0].shape[-2:][::-1]
        rgb = cv2.resize(np.array(img), input_size)
        rgb = rgb.astype(np.uint8)
        rgb = rgb.transpose(2, 0, 1)[None]
        probs = self.model.run(None, {'input': rgb})[0][0]
        return probs