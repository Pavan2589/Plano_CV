import cv2
import numpy as np
import logging
from typing import Dict, Any
from PIL import Image

logger = logging.getLogger("plano_cv.sanity_check")
logger.setLevel(logging.INFO)


class PlanogramSanityChecker:
    def __init__(self):
        self.visual_similarity_threshold = 0.35  # below this, shelves look unrelated
        self.min_count_ratio = 0.3   # check has fewer than 30% of expected detections
        self.max_count_ratio = 3.0   # check has more than 300% of expected detections
        self.compare_size = (256, 256)

    def compute_visual_similarity(self, reference_image: Image.Image,
                                   check_image: Image.Image) -> float:
        """
        Resizes both images to a common small size and computes
        normalized cross-correlation via template matching. This is a
        coarse global similarity score, not product-level comparison -
        it's meant to catch 'completely different shelf' cases cheaply.
        """
        ref_small = np.array(reference_image.convert("RGB").resize(self.compare_size))
        chk_small = np.array(check_image.convert("RGB").resize(self.compare_size))

        ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_RGB2GRAY)
        chk_gray = cv2.cvtColor(chk_small, cv2.COLOR_RGB2GRAY)

        result = cv2.matchTemplate(chk_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
        score = float(result[0][0])

        return max(0.0, score)  # clamp negative correlation to 0

    def check_detection_count(self, expected_count: int, detected_count: int) -> Dict[str, Any]:
        if expected_count == 0:
            return {"ratio": None, "within_range": True}

        ratio = detected_count / expected_count
        within_range = self.min_count_ratio <= ratio <= self.max_count_ratio

        return {"ratio": round(ratio, 3), "within_range": within_range}

    def run(self, reference_image: Image.Image, check_image: Image.Image,
            expected_item_count: int, detected_item_count: int) -> Dict[str, Any]:
        visual_similarity = self.compute_visual_similarity(reference_image, check_image)
        visual_ok = visual_similarity >= self.visual_similarity_threshold

        count_check = self.check_detection_count(expected_item_count, detected_item_count)

        passed = visual_ok and count_check["within_range"]

        reasons = []
        if not visual_ok:
            reasons.append(f"Visual similarity too low ({visual_similarity:.3f}, "
                           f"threshold {self.visual_similarity_threshold})")
        if not count_check["within_range"]:
            reasons.append(f"Detection count ratio out of range "
                           f"({count_check['ratio']}, expected between "
                           f"{self.min_count_ratio} and {self.max_count_ratio})")

        logger.info(f"Sanity check: passed={passed}, visual_similarity={visual_similarity:.3f}, "
                     f"count_ratio={count_check['ratio']}")

        return {
            "passed": passed,
            "visual_similarity": round(visual_similarity, 4),
            "detection_count_ratio": count_check["ratio"],
            "reasons": reasons,
        }


sanity_checker_instance = PlanogramSanityChecker()
