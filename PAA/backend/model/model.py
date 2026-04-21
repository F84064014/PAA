import cv2
import numpy as np
import onnxruntime as ort

class Model:
    def __init__(self, path: str) -> None:
        
        self.model = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def __call__(self, img: str) -> np.ndarray:
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError("img should be int type = str")
        probs = self.infer(img)

        # Place postprocessing here

        return probs

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        input_size = self.model.get_inputs()[0].shape[-2:][::-1]
        rgb = cv2.resize(rgb, input_size)
        rgb = rgb.astype(np.uint8)
        rgb = rgb.transpose(2, 0, 1)[None]
        probs = self.model.run(None, {'input': rgb})[0][0]
        return probs