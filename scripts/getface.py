import sys
import cv2
sys.path.append("/home/plchu/Experiments/yoloface")
import pandas as pd
from face_detector import YoloDetector

face_model = YoloDetector(
    weights_name="/home/plchu/Experiments/yoloface/yolov5m-face.pt",
    config_name="/home/plchu/Experiments/yoloface/models/yolov5m.yaml",
    target_size=640, device="cpu", min_face=5)

file_names = pd.read_csv(
    "/home/plchu/Labeler/data/ExportData_2026_0513.csv"
)['file_path'].tolist()

output_csv = list()

for file_name in file_names:
    img = cv2.imread(file_name)
    h, w, _ = img.shape
    bboxes, points = face_model(img)
    if not len(bboxes[0]):
        output_csv.append({
            "file_path": file_name,
            "x": 0, "y": 0, "w": 0, "h": 0
        })
        continue
    xyxy = bboxes[0][0]
    output_csv.append({
        "file_path": file_name,
        "x": xyxy[0], "w": (xyxy[2]-xyxy[0]),
        "y": xyxy[1], "h": (xyxy[3]-xyxy[1]),
    })

pd.DataFrame(output_csv).to_csv("face.csv", index=False)