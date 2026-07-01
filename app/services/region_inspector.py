import logging
from typing import List, Dict, Any
from PIL import Image

from app.services.embedder import embedder_instance
from app.services.matcher import calculate_similarity

logger = logging.getLogger("plano_cv.region_inspector")
logger.setLevel(logging.INFO)


class RegionInspector:
    def __init__(self):
        self.similarity_threshold = 0.55  # reuse same default as before, tune as needed

    def inspect(self, expected_items: List[Dict[str, Any]],
                aligned_check_image: Image.Image,
                reference_products: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        For each expected_item, crop the SAME bbox region directly from
        the aligned check image (no detection, no matching — just look
        at what's physically there in that exact rectangle) and compare
        its embedding against the expected product's embedding.

        This eliminates assignment/matching entirely. Each slot is judged
        independently using only its own expected coordinates.
        """
        check_width, check_height = aligned_check_image.size

        matched = []
        wrong_product = []
        missing = []

        for item in expected_items:
            x1, y1, x2, y2 = item["bbox"]
            x1 = max(0, min(x1, check_width))
            x2 = max(0, min(x2, check_width))
            y1 = max(0, min(y1, check_height))
            y2 = max(0, min(y2, check_height))

            if x2 <= x1 or y2 <= y1:
                print(f"[DIAGNOSTIC] {item['product_name']} | bbox={item['bbox']} | "
                      f"DROPPED: region_out_of_bounds (clamped to x1={x1},y1={y1},x2={x2},y2={y2}, "
                      f"image={check_width}x{check_height})")
                missing.append({
                    "product_id": item["product_id"],
                    "product_name": item["product_name"],
                    "bbox": item["bbox"],
                    "reason": "region_out_of_bounds_after_alignment"
                })
                continue

            crop = aligned_check_image.crop((int(x1), int(y1), int(x2), int(y2)))

            # Detect near-empty/blank region (likely genuinely missing product)
            crop_array = list(crop.convert("L").getdata())
            avg_brightness = sum(crop_array) / len(crop_array) if crop_array else 0
            brightness_variance = sum((p - avg_brightness) ** 2 for p in crop_array) / len(crop_array) if crop_array else 0

            if brightness_variance < 50:  # very flat/uniform region, likely empty shelf
                print(f"[DIAGNOSTIC] {item['product_name']} | bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] | "
                      f"DROPPED: region_appears_empty (variance={brightness_variance:.2f}, avg_brightness={avg_brightness:.1f})")
                missing.append({
                    "product_id": item["product_id"],
                    "product_name": item["product_name"],
                    "bbox": item["bbox"],
                    "reason": "region_appears_empty"
                })
                continue

            actual_embedding = embedder_instance.generate_embedding(crop).tolist()
            sim_to_expected = calculate_similarity(item["embedding"], actual_embedding)

            if sim_to_expected >= self.similarity_threshold:
                print(f"[DIAGNOSTIC] {item['product_name']} | bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] | "
                      f"MATCHED (sim={sim_to_expected:.4f})")
                matched.append({
                    "product_id": item["product_id"],
                    "product_name": item["product_name"],
                    "bbox": item["bbox"],
                    "similarity": round(sim_to_expected, 4),
                })
            else:
                best_alt_product = None
                best_alt_similarity = -1.0
                for ref in reference_products:
                    sim = calculate_similarity(actual_embedding, ref["embedding"])
                    if sim > best_alt_similarity:
                        best_alt_similarity = sim
                        best_alt_product = ref

                print(f"[DIAGNOSTIC] {item['product_name']} | bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] | "
                      f"WRONG_PRODUCT (sim_to_expected={sim_to_expected:.4f}, "
                      f"best_alt={best_alt_product['name'] if best_alt_product else 'none'} "
                      f"sim={best_alt_similarity:.4f})")
                wrong_product.append({
                    "expected_product_id": item["product_id"],
                    "expected_product_name": item["product_name"],
                    "bbox": item["bbox"],
                    "expected_similarity": round(sim_to_expected, 4),
                    "detected_product_name": best_alt_product["name"] if best_alt_product else "unknown",
                    "detected_similarity": round(best_alt_similarity, 4) if best_alt_product else 0.0,
                })

        logger.info(f"Region inspection: {len(matched)} matched, {len(wrong_product)} wrong_product, "
                     f"{len(missing)} missing (out of {len(expected_items)} expected slots)")

        return {"matched": matched, "wrong_product": wrong_product, "missing": missing}


region_inspector_instance = RegionInspector()
