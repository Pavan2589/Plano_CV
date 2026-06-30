"""
YOLOv8 product detector.

Single responsibility: find bounding boxes for "product"-like objects
in a shelf image. Does NOT identify what the product is — that's the
embedder + nearest-neighbour lookup's job.
"""
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

logger = logging.getLogger("plano_cv.detector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)


def preprocess_with_clahe(image: Image.Image) -> Image.Image:
    """
    Lighting normalization. Helps detection/embedding consistency across
    shelves with uneven store lighting (e.g. darker bottom rows).
    """
    img_array = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


class ProductDetector:
    def __init__(self):
        self.model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        self.confidence = float(os.getenv("YOLO_CONFIDENCE", "0.20"))

        logger.info(f"Loading YOLOv8 model from {self.model_path}...")
        start = time.time()
        self.model = YOLO(self.model_path)
        logger.info(f"YOLOv8 model loaded in {time.time() - start:.3f}s")

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        start = time.time()
        results = self.model(image, conf=self.confidence, verbose=False)

        detections = []
        if len(results) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = xyxy

                if x1 < 0 or y1 < 0 or x2 < x1 or y2 < y1:
                    continue
                if not (0.0 <= conf <= 1.0):
                    continue

                detections.append({
                    "bbox": [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)],
                    "confidence": round(conf, 4),
                })

        logger.info(f"YOLO detection: {len(detections)} boxes in {time.time() - start:.3f}s "
                     f"(confidence threshold={self.confidence})")
        return detections

    def save_debug_image(self, image: Image.Image, detections: List[Dict[str, Any]],
                          labels: List[str] = None, output_path: str = "debug_detections.jpg") -> str:
        """
        Draw boxes + labels with collision-avoidance so dense/touching products
        don't overwrite each other's labels (this bit us hard on tightly-packed
        Coke bottles earlier).
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        drawn_label_positions = []

        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            conf = det.get("confidence")
            label = labels[idx] if labels and idx < len(labels) else (
                f"{conf:.2f}" if isinstance(conf, (int, float)) else "product"
            )

            draw.rectangle(bbox, outline="yellow", width=3)

            label_x = bbox[0]
            label_y = max(0, bbox[1] - 14)

            for prev_x, prev_y in drawn_label_positions:
                if abs(label_x - prev_x) < 70 and abs(label_y - prev_y) < 14:
                    label_y = max(0, label_y - 14)

            draw.text((label_x, label_y), label, fill="yellow", font=font)
            drawn_label_positions.append((label_x, label_y))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path, format="JPEG")
        return output_path


detector_instance = ProductDetector()
