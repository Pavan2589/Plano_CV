import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.routers import reference_products, planogram, compliance, debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plano_cv.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Plano CV minimal service...")
    await db.init_db()
    yield
    await db.close_db()


app = FastAPI(title="Plano CV - Minimal Build", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reference_products.router, prefix="/api")
app.include_router(planogram.router, prefix="/api")
app.include_router(compliance.router, prefix="/api")
app.include_router(debug.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve the throwaway static frontend (index.html) at the root.
# Mounted LAST so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
