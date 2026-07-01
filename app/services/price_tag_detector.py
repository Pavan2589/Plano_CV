import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger("plano_cv.price_tag_detector")
logger.setLevel(logging.INFO)


class PriceTagDetector:
    def __init__(self):
        # Yellow HSV range - tune based on actual tag color samples if needed
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])
        self.min_area_fraction = 0.001
        self.max_area_fraction = 0.03
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 5.0

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        img_array = np.array(image.convert("RGB"))
        img_height, img_width = img_array.shape[:2]
        total_area = img_width * img_height

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tags = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_fraction = (w * h) / total_area
            aspect_ratio = w / h if h > 0 else 0

            if (self.min_area_fraction <= area_fraction <= self.max_area_fraction
                    and self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                tags.append({
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "area_fraction": round(area_fraction, 5),
                    "aspect_ratio": round(aspect_ratio, 2),
                })

        logger.info(f"Price tag detection: {len(tags)} candidate tags found "
                     f"(from {len(contours)} raw contours)")
        return tags

    def save_debug_mask(self, image: Image.Image, output_path: str = "debug/price_mask_trace.jpg") -> str:
        img_array = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, mask)
        return output_path


price_tag_detector_instance = PriceTagDetector()
