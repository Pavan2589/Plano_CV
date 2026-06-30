"""
Full-shelf bipartite matching using the Hungarian algorithm.

Replaces the old row-clustering + Needleman-Wunsch approach. No row
pre-clustering, no facing_count expansion needed — every expected item
vs every detected item is compared at once using a combined cost of
(1 - embedding similarity) and normalized spatial distance.
"""
import os
import math
import logging
from typing import List, Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("plano_cv.matcher")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

SENTINEL_COST = 999.0


def calculate_similarity(embedding_a: List[float], embedding_b: List[float]) -> float:
    if not embedding_a or not embedding_b:
        return 0.0
    a = np.array(embedding_a, dtype=np.float32)
    b = np.array(embedding_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class HungarianMatcher:
    def __init__(self):
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.60"))
        self.alpha = float(os.getenv("MATCH_ALPHA", "0.7"))  # embedding weight
        self.beta = float(os.getenv("MATCH_BETA", "0.3"))    # spatial weight
        logger.info(f"HungarianMatcher initialized: threshold={self.similarity_threshold}, "
                     f"alpha={self.alpha}, beta={self.beta}")

    def _normalized_center(self, bbox: List[float], width: float, height: float):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx / width if width else 0.0), (cy / height if height else 0.0)

    def match_full_shelf(
        self,
        expected_items: List[Dict[str, Any]],
        detected_items: List[Dict[str, Any]],
        expected_image_width: float,
        expected_image_height: float,
        shelf_image_width: float,
        shelf_image_height: float,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        expected_items: [{ id, product_id, product_name, embedding, bbox }]
                         (bbox is from the REFERENCE image, used only for spatial cost)
        detected_items: [{ bbox, embedding }]
                         (from the SHELF image being checked)

        Returns: { matched: [...], missing: [...], wrong_product: [...], unexpected: [...] }
        """
        n = len(expected_items)
        m = len(detected_items)

        matched, missing, wrong_product, unexpected = [], [], [], []

        if n == 0 and m == 0:
            return {"matched": matched, "missing": missing, "wrong_product": wrong_product, "unexpected": unexpected}

        if n == 0:
            for det in detected_items:
                unexpected.append({"bbox": det["bbox"], "note": "no expected items defined"})
            return {"matched": matched, "missing": missing, "wrong_product": wrong_product, "unexpected": unexpected}

        if m == 0:
            for exp in expected_items:
                missing.append({"product_id": exp["product_id"], "product_name": exp["product_name"], "bbox": exp.get("bbox")})
            return {"matched": matched, "missing": missing, "wrong_product": wrong_product, "unexpected": unexpected}

        # Precompute normalized spatial centers
        exp_centers = [self._normalized_center(e["bbox"], expected_image_width, expected_image_height) for e in expected_items]
        det_centers = [self._normalized_center(d["bbox"], shelf_image_width, shelf_image_height) for d in detected_items]

        size = max(n, m)
        cost_matrix = np.full((size, size), SENTINEL_COST, dtype=np.float64)

        for i, exp in enumerate(expected_items):
            for j, det in enumerate(detected_items):
                sim = calculate_similarity(exp["embedding"], det["embedding"])
                embedding_cost = 1.0 - sim
                ex, ey = exp_centers[i]
                dx, dy = det_centers[j]
                spatial_dist = math.sqrt((ex - dx) ** 2 + (ey - dy) ** 2)
                cost_matrix[i][j] = self.alpha * embedding_cost + self.beta * spatial_dist

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        assigned_expected = set()
        assigned_detected = set()

        for i, j in zip(row_idx, col_idx):
            if i >= n or j >= m:
                # padded sentinel row/col, no real item on one side
                continue

            assigned_expected.add(i)
            assigned_detected.add(j)

            exp = expected_items[i]
            det = detected_items[j]
            sim = calculate_similarity(exp["embedding"], det["embedding"])

            if sim >= self.similarity_threshold:
                matched.append({
                    "product_id": exp["product_id"],
                    "product_name": exp["product_name"],
                    "expected_bbox": exp.get("bbox"),
                    "detected_bbox": det["bbox"],
                    "similarity": round(sim, 4),
                })
            else:
                wrong_product.append({
                    "product_id": exp["product_id"],
                    "expected_product_name": exp["product_name"],
                    "detected_bbox": det["bbox"],
                    "similarity": round(sim, 4),
                })

        for i, exp in enumerate(expected_items):
            if i not in assigned_expected:
                missing.append({
                    "product_id": exp["product_id"],
                    "product_name": exp["product_name"],
                    "bbox": exp.get("bbox"),
                })

        for j, det in enumerate(detected_items):
            if j not in assigned_detected:
                unexpected.append({"bbox": det["bbox"]})

        logger.info(f"Hungarian match: {len(matched)} matched, {len(missing)} missing, "
                     f"{len(wrong_product)} wrong_product, {len(unexpected)} unexpected "
                     f"(expected={n}, detected={m})")

        return {"matched": matched, "missing": missing, "wrong_product": wrong_product, "unexpected": unexpected}


matcher_instance = HungarianMatcher()
