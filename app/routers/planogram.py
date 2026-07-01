"""
Planogram generation endpoint + a tiny in-memory store.

No `planograms` DB table in this minimal build — a single generated
planogram is kept in process memory (PLANOGRAM_STORE) so it can be
referenced by the compliance check endpoint. This is intentionally
throwaway; once the CV logic is verified, this gets backed by Postgres.
"""
import io
import base64
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from app.services.planogram_generator import planogram_generator_instance
from app.services.shelf_rail_detector import shelf_rail_detector_instance
from app.services.position_assigner import position_assigner_instance
from app.services.detector import preprocess_with_clahe

logger = logging.getLogger("plano_cv.router.planogram")
router = APIRouter()

# In-memory store: { planogram_id: { expected_items, ref_image_width, ref_image_height } }
PLANOGRAM_STORE: dict = {}


@router.post("/planogram/generate")
async def generate_planogram(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = await planogram_generator_instance.generate_from_reference(pil_image)

    clahe_image = preprocess_with_clahe(pil_image)
    img_width, img_height = clahe_image.size

    rail_y_positions = shelf_rail_detector_instance.detect_from_bboxes(
        detections=result["expected_items"],
        image_height=float(result["ref_image_height"]),
    )

    enriched_items = position_assigner_instance.assign(
        detections=result["expected_items"],
        rail_y_positions=rail_y_positions,
        image_height=float(img_height),
    )

    planogram_id = str(uuid.uuid4())
    PLANOGRAM_STORE[planogram_id] = {
        "expected_items": enriched_items,
        "ref_image_width": result["ref_image_width"],
        "ref_image_height": result["ref_image_height"],
        "ref_image_bytes": contents,
        "rail_y_positions": rail_y_positions,
    }

    from PIL import ImageDraw, ImageFont
    annotated = clahe_image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for rail_y in rail_y_positions:
        draw.line(
            [(0, int(rail_y)), (img_width, int(rail_y))],
            fill="cyan", width=4
        )

    drawn_label_positions = []
    for item in enriched_items:
        bbox = item["bbox"]
        row = item.get("row", "?")
        position = item.get("position", "?")
        product = item.get("product_name", "?")
        sim = item.get("similarity", 0.0)
        label = f"R{row}P{position} {product} {int(sim*100)}%"

        color = "green" if sim >= 0.75 else "orange"
        draw.rectangle(bbox, outline=color, width=3)

        label_x = bbox[0]
        label_y = max(0, bbox[1] - 14)
        for prev_x, prev_y in drawn_label_positions:
            if abs(label_x - prev_x) < 70 and abs(label_y - prev_y) < 14:
                label_y = max(0, label_y - 14)
        draw.text((label_x, label_y), label, fill=color, font=font)
        drawn_label_positions.append((label_x, label_y))

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "planogram_id": planogram_id,
        "ref_image_width": result["ref_image_width"],
        "ref_image_height": result["ref_image_height"],
        "total_yolo_detections": result["total_yolo_detections"],
        "accepted_count": result["accepted_count"],
        "rejected_count": result["rejected_count"],
        "expected_items": enriched_items,
        "rejected_items": result["rejected_items"],
        "rail_y_positions": [round(y, 1) for y in rail_y_positions],
        "annotated_image_base64": annotated_b64,
    }


@router.get("/planogram/{planogram_id}")
async def get_planogram(planogram_id: str):
    if planogram_id not in PLANOGRAM_STORE:
        raise HTTPException(status_code=404, detail="Planogram not found")
    return PLANOGRAM_STORE[planogram_id]


@router.get("/planogram")
async def list_planograms():
    return [
        {"planogram_id": pid, "item_count": len(data["expected_items"])}
        for pid, data in PLANOGRAM_STORE.items()
    ]
