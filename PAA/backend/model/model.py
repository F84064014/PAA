import cv2
import tqdm
import numpy as np
import onnxruntime as ort

class Model:
    def __init__(self, path: str) -> None:
        
        self.model = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def __call__(self, imgs: str | list) -> np.ndarray:
        if isinstance(imgs, str):
            imgs = [imgs]
        elif isinstance(imgs, list):
            imgs = tqdm.tqdm(imgs)
        assert len(imgs)

        res = []
        for img in imgs:
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res.append(self.infer(img))

        return np.stack(res, axis=0) if len(res) > 1 else res[0]

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        input_size = self.model.get_inputs()[0].shape[-2:][::-1]
        rgb = cv2.resize(rgb, input_size)
        rgb = rgb.astype(np.uint8)
        rgb = rgb.transpose(2, 0, 1)[None]
        probs = self.model.run(None, {'input': rgb})[0][0]
        return probs