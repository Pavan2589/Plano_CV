import io
import os
import base64
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image, ImageDraw, ImageFont

from app.services.detector import detector_instance, preprocess_with_clahe
from app.services.embedder import embedder_instance
from app.services.matcher import calculate_similarity
from app.services.shelf_rail_detector import shelf_rail_detector_instance
from app.services.position_assigner import position_assigner_instance
from app.services.compliance_comparator import compliance_comparator_instance
from app.routers.planogram import PLANOGRAM_STORE
from app import db

logger = logging.getLogger("plano_cv.router.compliance")
router = APIRouter()

MATCH_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))


@router.post("/compliance/check")
async def run_compliance_check(
        planogram_id: str = Form(...), image: UploadFile = File(...)):

    if planogram_id not in PLANOGRAM_STORE:
        raise HTTPException(
            status_code=404,
            detail="Planogram not found. Generate one first via "
                   "/planogram/generate"
        )

    planogram = PLANOGRAM_STORE[planogram_id]
    reference_items = planogram["expected_items"]

    contents = await image.read()
    try:
        shelf_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    shelf_image = preprocess_with_clahe(shelf_image)
    shelf_width, shelf_height = shelf_image.size

    # Step 1: detect products on shelf-check image
    detections = detector_instance.detect(shelf_image)

    # Step 2: embed + identify each detection
    reference_products = await db.get_all_reference_products_with_embeddings()
    shelf_items_raw = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = shelf_image.crop((int(x1), int(y1), int(x2), int(y2)))
        embedding = embedder_instance.generate_embedding(crop).tolist()

        best_product = None
        best_similarity = -1.0
        for ref in reference_products:
            sim = calculate_similarity(embedding, ref["embedding"])
            if sim > best_similarity:
                best_similarity = sim
                best_product = ref

        if best_product and best_similarity >= MATCH_THRESHOLD:
            shelf_items_raw.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "product_name": best_product["name"],
                "product_id": best_product["id"],
                "similarity": round(best_similarity, 4),
                "embedding": embedding,
            })

    # Step 3: assign row + position to shelf detections
    shelf_rail_positions = shelf_rail_detector_instance.detect_from_bboxes(
        detections=shelf_items_raw,
        image_height=float(shelf_height),
    )

    shelf_items = position_assigner_instance.assign(
        detections=shelf_items_raw,
        rail_y_positions=shelf_rail_positions,
        image_height=float(shelf_height),
    )

    # Step 4: run compliance comparison
    comparison = compliance_comparator_instance.compare(
        reference_items=reference_items,
        shelf_items=shelf_items,
    )

    # Step 5: annotate shelf image with results
    annotated = shelf_image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Draw shelf rails
    for rail_y in shelf_rail_positions:
        draw.line(
            [(0, int(rail_y)), (shelf_width, int(rail_y))],
            fill="cyan", width=3
        )

    # Build violation lookup by (row, position) for fast annotation
    violation_lookup = {}
    for v in comparison["sequence_violations"]:
        key = (v["row"], v["position"])
        violation_lookup[key] = v

    drawn_label_positions = []

    def draw_label(bbox, label, color):
        draw.rectangle(bbox, outline=color, width=3)
        label_x = bbox[0]
        label_y = max(0, bbox[1] - 14)
        for prev_x, prev_y in drawn_label_positions:
            if abs(label_x - prev_x) < 70 and abs(label_y - prev_y) < 14:
                label_y = max(0, label_y - 14)
        draw.text((label_x, label_y), label, fill=color, font=font)
        drawn_label_positions.append((label_x, label_y))

    for item in shelf_items:
        bbox = item["bbox"]
        row = item.get("row")
        position = item.get("position")
        product = item.get("product_name", "?")
        sim = item.get("similarity", 0.0)
        key = (row, position)

        if key in violation_lookup:
            v = violation_lookup[key]
            if v["type"] == "wrong_product":
                label = (f"R{row}P{position} Expected "
                         f"{v['expected_product']} got {product}")
                draw_label(bbox, label, "red")
            else:
                draw_label(bbox, f"R{row}P{position} {product}", "green")
        else:
            draw_label(bbox, f"R{row}P{position} {product} "
                             f"{int(sim*100)}%", "green")

    # Draw missing products legend
    missing = [v for v in comparison["sequence_violations"]
               if v["type"] == "missing_product"]
    if missing:
        legend_x, legend_y = 10, 10
        line_h = 16
        box_h = 30 + len(missing) * line_h
        draw.rectangle(
            [legend_x, legend_y, legend_x + 350, legend_y + box_h],
            fill=(0, 0, 0, 180), outline="orange", width=2
        )
        draw.text((legend_x + 10, legend_y + 8),
                  "Missing Products:", fill="white", font=font)
        for idx, m in enumerate(missing):
            draw.text(
                (legend_x + 10, legend_y + 26 + idx * line_h),
                f"- R{m['row']}P{m['position']} {m['expected_product']}",
                fill="orange", font=font
            )

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "planogram_id": planogram_id,
        "overall_score": comparison["overall_score"],
        "count_score": comparison["count_score"],
        "sequence_score": comparison["sequence_score"],
        "shelf_rail_positions": [round(y, 1) for y in shelf_rail_positions],
        "total_expected": len(reference_items),
        "total_detected_on_shelf": len(shelf_items),
        "count_violations": comparison["count_violations"],
        "sequence_violations": comparison["sequence_violations"],
        "unexpected_products": comparison["unexpected_products"],
        "row_results": comparison["row_results"],
        "annotated_image_base64": annotated_b64,
    }
