"""
Planogram generation endpoint + a tiny in-memory store.

No `planograms` DB table in this minimal build — a single generated
planogram is kept in process memory (PLANOGRAM_STORE) so it can be
referenced by the compliance check endpoint. This is intentionally
throwaway; once the CV logic is verified, this gets backed by Postgres.
"""
import io
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from app.services.planogram_generator import planogram_generator_instance

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

    planogram_id = str(uuid.uuid4())
    PLANOGRAM_STORE[planogram_id] = {
        "expected_items": result["expected_items"],
        "ref_image_width": result["ref_image_width"],
        "ref_image_height": result["ref_image_height"],
    }

    return {
        "planogram_id": planogram_id,
        **result,
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
