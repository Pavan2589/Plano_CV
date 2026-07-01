"""
Single diagnostic endpoint that traces every pipeline stage for one
image in one response. Built specifically to stop the back-and-forth
guessing of "is detection broken, or embedding, or row logic" — this
shows every stage's counts and data side by side.
"""
import io
import base64
import logging
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image

from app.services.detector import detector_instance, preprocess_with_clahe
from app.services.price_tag_detector import price_tag_detector_instance
from app.services.embedder import embedder_instance
from app.services.matcher import calculate_similarity
from app.services.alignment import aligner_instance
from app.services.shelf_rail_detector import shelf_rail_detector_instance
from app.services.position_assigner import position_assigner_instance
from app.routers.planogram import PLANOGRAM_STORE
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


@router.post("/debug/price-tags")
async def debug_price_tags(image: UploadFile = File(...)):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))

    tags = price_tag_detector_instance.detect(pil_image)
    debug_path = price_tag_detector_instance.save_debug_mask(pil_image)

    return {
        "tags_found": len(tags),
        "tags": tags,
        "debug_mask_saved_to": debug_path
    }


@router.post("/debug/alignment")
async def debug_alignment(planogram_id: str = Form(...), image: UploadFile = File(...)):
    if planogram_id not in PLANOGRAM_STORE:
        return {"error": "planogram not found"}

    planogram = PLANOGRAM_STORE[planogram_id]
    reference_image = Image.open(io.BytesIO(planogram["ref_image_bytes"]))

    contents = await image.read()
    check_image = Image.open(io.BytesIO(contents))

    aligned, success, match_count = aligner_instance.align(reference_image, check_image)

    buf = io.BytesIO()
    aligned.save(buf, format="JPEG")
    aligned_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "alignment_success": success,
        "match_count": match_count,
        "aligned_image_base64": aligned_b64
    }


@router.post("/debug/grid")
async def debug_grid(image: UploadFile = File(...), match_threshold: float = 0.55):
    """
    Takes any shelf image and returns:
    1. Detected shelf rail Y positions
    2. Every detected product with its assigned (row, position)
    3. A debug image with rails drawn as cyan lines + product labels

    Use this to verify rail detection and position assignment are
    correct BEFORE implementing compliance comparison.
    """
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    pil_image_clahe = preprocess_with_clahe(pil_image)
    img_width, img_height = pil_image_clahe.size

    # Step 1: shelf rails will be detected after enriched_detections is built

    # Step 2: detect products
    detections = detector_instance.detect(pil_image_clahe)

    # Step 3: embed + match each detection to nearest reference product
    reference_products = await db.get_all_reference_products_with_embeddings()
    enriched_detections = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = pil_image_clahe.crop((int(x1), int(y1), int(x2), int(y2)))
        embedding = embedder_instance.generate_embedding(crop).tolist()

        best_product = None
        best_similarity = -1.0
        for ref in reference_products:
            sim = calculate_similarity(embedding, ref["embedding"])
            if sim > best_similarity:
                best_similarity = sim
                best_product = ref

        if best_product and best_similarity >= match_threshold:
            enriched_detections.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "matched_product": best_product["name"],
                "product_id": best_product["id"],
                "similarity": round(best_similarity, 4),
                "embedding": embedding,
            })

    # Step 4: detect shelf rails from bbox gaps, then assign row + position
    rail_y_positions = shelf_rail_detector_instance.detect_from_bboxes(
        detections=enriched_detections,
        image_height=float(img_height),
    )
    grid = position_assigner_instance.assign(
        detections=enriched_detections,
        rail_y_positions=rail_y_positions,
        image_height=float(img_height),
    )

    # Step 5: draw rails + product labels on annotated image
    from PIL import ImageDraw, ImageFont
    annotated = pil_image_clahe.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for rail_y in rail_y_positions:
        draw.line([(0, int(rail_y)), (img_width, int(rail_y))], fill="cyan", width=3)

    drawn_label_positions = []
    for item in grid:
        bbox = item["bbox"]
        label = f"R{item['row']}P{item['position']} {item['matched_product']}"
        draw.rectangle(bbox, outline="green", width=2)
        label_x = bbox[0]
        label_y = max(0, bbox[1] - 14)
        for prev_x, prev_y in drawn_label_positions:
            if abs(label_x - prev_x) < 70 and abs(label_y - prev_y) < 14:
                label_y = max(0, label_y - 14)
        draw.text((label_x, label_y), label, fill="green", font=font)
        drawn_label_positions.append((label_x, label_y))

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image_size": {"width": img_width, "height": img_height},
        "rail_y_positions": [round(y, 1) for y in rail_y_positions],
        "total_rails_detected": len(rail_y_positions),
        "total_products_detected": len(grid),
        "grid": [
            {
                "row": item["row"],
                "position": item["position"],
                "product": item["matched_product"],
                "similarity": item["similarity"],
                "bbox": item["bbox"],
            }
            for item in grid
        ],
        "annotated_image_base64": annotated_b64,
    }
