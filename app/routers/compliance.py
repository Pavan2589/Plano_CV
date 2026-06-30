"""
Compliance check: upload a real shelf photo, compare it against a
previously generated planogram using full-shelf Hungarian matching.
"""
import io
import base64
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from app.services.detector import detector_instance, preprocess_with_clahe
from app.services.embedder import embedder_instance
from app.services.matcher import matcher_instance
from app.services.annotator import annotator_instance
from app.routers.planogram import PLANOGRAM_STORE

logger = logging.getLogger("plano_cv.router.compliance")
router = APIRouter()


@router.post("/compliance/check")
async def run_compliance_check(planogram_id: str = Form(...), image: UploadFile = File(...)):
    if planogram_id not in PLANOGRAM_STORE:
        raise HTTPException(status_code=404, detail="Planogram not found. Generate one first via /planogram/generate")

    planogram = PLANOGRAM_STORE[planogram_id]
    expected_items = planogram["expected_items"]
    ref_width = planogram["ref_image_width"]
    ref_height = planogram["ref_image_height"]

    contents = await image.read()
    try:
        shelf_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    shelf_image = preprocess_with_clahe(shelf_image)
    shelf_width, shelf_height = shelf_image.size

    detections = detector_instance.detect(shelf_image)
    logger.info(f"Compliance check: {len(detections)} raw detections on shelf image")

    detected_items = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = shelf_image.crop((int(x1), int(y1), int(x2), int(y2)))
        embedding = embedder_instance.generate_embedding(crop).tolist()
        detected_items.append({"bbox": det["bbox"], "embedding": embedding})

    match_result = matcher_instance.match_full_shelf(
        expected_items=expected_items,
        detected_items=detected_items,
        expected_image_width=ref_width,
        expected_image_height=ref_height,
        shelf_image_width=shelf_width,
        shelf_image_height=shelf_height,
    )

    annotated = annotator_instance.annotate(shelf_image, match_result)
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    total_expected = len(expected_items)
    correct = len(match_result["matched"])
    score = round((correct / total_expected) * 100, 2) if total_expected > 0 else 0.0

    return {
        "planogram_id": planogram_id,
        "overall_score": score,
        "total_expected": total_expected,
        "total_detected_on_shelf": len(detected_items),
        "matched": match_result["matched"],
        "missing": match_result["missing"],
        "wrong_product": match_result["wrong_product"],
        "unexpected": match_result["unexpected"],
        "annotated_image_base64": annotated_b64,
    }
