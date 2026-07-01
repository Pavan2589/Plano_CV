import logging
from typing import List, Dict, Any
from PIL import Image, ImageDraw

logger = logging.getLogger("plano_cv.shelf_rail_detector")
logger.setLevel(logging.INFO)


class ShelfRailDetector:
    def __init__(self):
        # Minimum vertical gap between two consecutive bbox Y-extents
        # to be considered a shelf boundary. In pixels.
        # Set to 30px default — adjust if shelves are very close together
        # or products overlap vertically.
        self.min_gap_px = 30

    def detect_from_bboxes(self, detections: List[Dict[str, Any]],
                            image_height: float) -> List[float]:
        if not detections:
            return []

        # Compute center_y for every detection
        center_ys = sorted([
            (det["bbox"][1] + det["bbox"][3]) / 2.0
            for det in detections
            if len(det.get("bbox", [])) >= 4
        ])

        if not center_ys:
            return []

        # Find the largest gap between consecutive center_y values
        # That gap = the empty space between shelf rows
        max_gap = 0
        rail_y = None
        for i in range(len(center_ys) - 1):
            gap = center_ys[i + 1] - center_ys[i]
            if gap > max_gap:
                max_gap = gap
                rail_y = (center_ys[i] + center_ys[i + 1]) / 2.0

        if rail_y is None or max_gap < self.min_gap_px:
            logger.warning(f"No clear row boundary found. Largest gap was "
                           f"{max_gap:.1f}px at y={rail_y}")
            return []

        logger.info(f"Row boundary detected at y={rail_y:.1f} "
                    f"(gap={max_gap:.1f}px between centroids)")
        return [rail_y]

    def detect(self, image: Image.Image) -> List[float]:
        """
        Legacy method kept for API compatibility with the debug/grid
        endpoint which calls detect() without bbox data. Returns empty
        list — callers should use detect_from_bboxes() instead.
        """
        logger.warning("detect() called without bbox data — use "
                       "detect_from_bboxes(detections, image_height) instead")
        return []

    def save_debug_image(self, image: Image.Image,
                         rail_y_positions: List[float],
                         output_path: str = "debug/shelf_rails.jpg") -> str:
        import os
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        img_width = image.size[0]
        for y in rail_y_positions:
            draw.line(
                [(0, int(y)), (img_width, int(y))],
                fill="cyan", width=4
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        annotated.save(output_path, format="JPEG")
        return output_path


shelf_rail_detector_instance = ShelfRailDetector()
