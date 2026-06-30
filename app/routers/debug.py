"""
Single diagnostic endpoint that traces every pipeline stage for one
image in one response. Built specifically to stop the back-and-forth
guessing of "is detection broken, or embedding, or row logic" — this
shows every stage's counts and data side by side.
"""
import io
import logging
from fastapi import APIRouter, UploadFile, File
from PIL import Image

from app.services.detector import detector_instance, preprocess_with_clahe
from app.services.embedder import embedder_instance
from app.services.matcher import calculate_similarity
from app import db

logger = logging.getLogger("plano_cv.router.debug")
router = APIRouter()


@router.post("/debug/trace")
async def debug_trace(image: UploadFile = File(...), match_threshold: float = 0.55):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    pil_image = preprocess_with_clahe(pil_image)
    width, height = pil_image.size

    # Stage 1: raw YOLO
    detections = detector_instance.detect(pil_image)

    # Stage 2: embed + nearest neighbour for every detection
    reference_products = await db.get_all_reference_products_with_embeddings()

    stage_2_matches = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = pil_image.crop((int(x1), int(y1), int(x2), int(y2)))
        embedding = embedder_instance.generate_embedding(crop).tolist()

        best_product = None
        best_similarity = -1.0
        for ref in reference_products:
            sim = calculate_similarity(embedding, ref["embedding"])
            if sim > best_similarity:
                best_similarity = sim
                best_product = ref

        stage_2_matches.append({
            "bbox": det["bbox"],
            "confidence": det["confidence"],
            "matched_product": best_product["name"] if best_product else None,
            "similarity": round(best_similarity, 4) if best_product else 0.0,
            "accepted": bool(best_product and best_similarity >= match_threshold),
        })

    accepted = [m for m in stage_2_matches if m["accepted"]]
    rejected = [m for m in stage_2_matches if not m["accepted"]]

    # Per-product counts (pure count, no spatial logic at all)
    from collections import Counter
    product_counts = Counter(m["matched_product"] for m in accepted if m["matched_product"])

    return {
        "image_size": {"width": width, "height": height},
        "match_threshold_used": match_threshold,
        "stage_1_yolo_raw_count": len(detections),
        "stage_1_yolo_boxes": detections,
        "stage_2_embedding_matches": stage_2_matches,
        "stage_2_accepted_count": len(accepted),
        "stage_2_rejected_count": len(rejected),
        "stage_2_rejected_items": rejected,
        "product_counts": dict(product_counts),
        "summary": (
            f"YOLO found {len(detections)} boxes. "
            f"{len(accepted)} matched a reference product above threshold {match_threshold}. "
            f"{len(rejected)} were rejected (no match or below threshold)."
        ),
    }
