import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from app.services.embedder import embedder_instance
from app import db

router = APIRouter()


@router.post("/reference-products")
async def add_reference_product(name: str = Form(...), image: UploadFile = File(...)):
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    embedding = embedder_instance.generate_embedding(pil_image).tolist()
    result = await db.insert_reference_product(name=name, embedding=embedding)
    return result


@router.get("/reference-products")
async def list_reference_products():
    return await db.list_reference_products()


@router.delete("/reference-products/{product_id}")
async def delete_reference_product(product_id: str):
    deleted = await db.delete_reference_product(product_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Reference product not found")
    return {"status": "deleted", "id": product_id}
