"""
Reference-image-driven planogram generation.

Replaces manual row/position cell entry entirely. Upload a photo of a
correctly-stocked shelf, and this produces a flat list of expected
items: one entry per detected product, each carrying its own bbox and
matched reference product. No grouping, no facing_count collapsing,
no row/position numbers — those caused real bugs in the old system and
are deliberately not reintroduced here. Hungarian matching in matcher.py
doesn't need them; it matches purely on embedding + spatial position.
"""
import logging
from typing import List, Dict, Any
from PIL import Image

from app.services.detector import detector_instance, preprocess_with_clahe
from app.services.embedder import embedder_instance
from app.services.matcher import calculate_similarity
from app import db

logger = logging.getLogger("plano_cv.planogram_generator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

REFERENCE_MATCH_THRESHOLD_DEFAULT = 0.55


class PlanogramGenerator:

    async def generate_from_reference(
        self, image: Image.Image, match_threshold: float = REFERENCE_MATCH_THRESHOLD_DEFAULT
    ) -> Dict[str, Any]:
        image = preprocess_with_clahe(image)
        width, height = image.size

        # Stage 1: raw detection
        detections = detector_instance.detect(image)
        logger.info(f"[Stage 1] Raw YOLO detections: {len(detections)}")

        # Stage 2: embed every detection + look up nearest reference product
        reference_products = await db.get_all_reference_products_with_embeddings()

        accepted_items = []
        rejected_items = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
            embedding = embedder_instance.generate_embedding(crop).tolist()

            best_product = None
            best_similarity = -1.0
            for ref in reference_products:
                sim = calculate_similarity(embedding, ref["embedding"])
                if sim > best_similarity:
                    best_similarity = sim
                    best_product = ref

            record = {
                "bbox": det["bbox"],
                "embedding": embedding,
                "confidence": det["confidence"],
                "matched_product_id": best_product["id"] if best_product else None,
                "matched_product_name": best_product["name"] if best_product else None,
                "similarity": round(best_similarity, 4) if best_product else 0.0,
            }

            if best_product is not None and best_similarity >= match_threshold:
                accepted_items.append(record)
            else:
                rejected_items.append(record)

        logger.info(f"[Stage 2] Accepted: {len(accepted_items)}, Rejected (below threshold "
                     f"{match_threshold} or no reference products): {len(rejected_items)}")

        # Stage 3: build flat expected_items list — ONE entry per accepted detection.
        # No grouping, no collapsing. This is deliberately the simplest possible
        # mapping to avoid the data-loss bug from the row/position approach.
        expected_items = []
        for idx, item in enumerate(accepted_items):
            expected_items.append({
                "id": f"item_{idx}",
                "product_id": item["matched_product_id"],
                "product_name": item["matched_product_name"],
                "embedding": item["embedding"],
                "bbox": item["bbox"],
                "similarity": item["similarity"],
            })

        logger.info(f"[Stage 3] Final expected_items: {len(expected_items)} "
                     f"(should equal accepted count: {len(accepted_items)})")

        return {
            "ref_image_width": width,
            "ref_image_height": height,
            "total_yolo_detections": len(detections),
            "accepted_count": len(accepted_items),
            "rejected_count": len(rejected_items),
            "expected_items": expected_items,
            "rejected_items": [
                {"bbox": r["bbox"], "best_guess": r["matched_product_name"], "similarity": r["similarity"]}
                for r in rejected_items
            ],
        }


planogram_generator_instance = PlanogramGenerator()
