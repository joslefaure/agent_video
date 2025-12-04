from ultralytics import YOLO
import logging
from typing import List, Dict, Any

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            logging.info(f"Loading detector: {self.model_name}")
            # This will download the model if not present
            self.model = YOLO(self.model_name)

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        self._load_model()
        try:
            results = self.model(image_path, verbose=False)
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        "class": result.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()[0] # [x1, y1, x2, y2]
                    })
            return detections
        except Exception as e:
            logging.error(f"Detection failed for {image_path}: {e}")
            return []
