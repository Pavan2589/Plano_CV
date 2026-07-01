"""
Minimal DB layer for Plano CV.

Scope is intentionally tiny: ONE table (reference_products) storing
2048-d ResNet50 embeddings. No clients/stores/sections/planograms/jobs
schema yet — that gets reintroduced once the CV logic is proven correct.

SQLite backend: pgvector's native <=> operator doesn't exist here, so
embeddings are stored as float32 BLOBs and nearest-neighbour search is
done in-process with numpy. Fine at this table's scale (a handful of
reference SKUs); revisit if this ever needs to scale to thousands of
products.
"""
import os
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import aiosqlite
import numpy as np

logger = logging.getLogger("plano_cv.db")
logger.setLevel(logging.INFO)

DATABASE_PATH = os.getenv("DATABASE_PATH", "./plano_cv.db")

_conn: Optional[aiosqlite.Connection] = None
# aiosqlite serializes access over one connection fine for reads, but
# concurrent writes can still hit "database is locked" — cheap insurance.
_write_lock = asyncio.Lock()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS reference_products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);
"""


async def init_db():
    """Open the connection and ensure schema exists. Call once at startup."""
    global _conn
    logger.info(f"Connecting to SQLite database at {DATABASE_PATH}...")
    _conn = await aiosqlite.connect(DATABASE_PATH)
    # Sane defaults for a local single-file dev DB.
    await _conn.execute("PRAGMA journal_mode=WAL;")
    await _conn.execute("PRAGMA foreign_keys=ON;")
    await _conn.execute(SCHEMA_SQL)
    await _conn.commit()
    logger.info("Database initialized: reference_products table ready.")


def get_conn() -> aiosqlite.Connection:
    if _conn is None:
        raise RuntimeError("DB connection not initialized. Call init_db() at startup.")
    return _conn


async def close_db():
    global _conn
    if _conn:
        await _conn.close()
        _conn = None


# ── Encoding helpers ─────────────────────────────────────────────────────────

def _encode_embedding(embedding: List[float]) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _decode_embedding(blob: bytes) -> List[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


# ── Reference Products ──────────────────────────────────────────────────────

async def insert_reference_product(name: str, embedding: List[float]) -> Dict[str, Any]:
    conn = get_conn()
    product_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    async with _write_lock:
        await conn.execute(
            "INSERT INTO reference_products (id, name, embedding, created_at) VALUES (?, ?, ?, ?)",
            (product_id, name, _encode_embedding(embedding), created_at),
        )
        await conn.commit()
    return {"id": product_id, "name": name, "created_at": created_at}


async def list_reference_products() -> List[Dict[str, Any]]:
    conn = get_conn()
    async with conn.execute(
        "SELECT id, name, created_at FROM reference_products ORDER BY created_at DESC"
    ) as cursor:
        rows = await cursor.fetchall()
    return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


async def get_all_reference_products_with_embeddings() -> List[Dict[str, Any]]:
    """Used by detection/matching code to do nearest-neighbour lookups in-process."""
    conn = get_conn()
    async with conn.execute(
        "SELECT id, name, embedding FROM reference_products"
    ) as cursor:
        rows = await cursor.fetchall()
    return [
        {"id": r[0], "name": r[1], "embedding": _decode_embedding(r[2])}
        for r in rows
    ]


async def find_nearest_reference_product(embedding: List[float]) -> Optional[Dict[str, Any]]:
    """
    Nearest-neighbour lookup by cosine similarity, computed in Python since
    SQLite has no vector distance operator. Returns the single closest
    reference product + its similarity (1 - cosine distance).
    """
    products = await get_all_reference_products_with_embeddings()
    if not products:
        return None

    query = np.asarray(embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return None

    best = None
    best_similarity = -1.0
    for p in products:
        vec = np.asarray(p["embedding"], dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            continue
        similarity = float(np.dot(query, vec) / (query_norm * vec_norm))
        if similarity > best_similarity:
            best_similarity = similarity
            best = p

    if best is None:
        return None
    return {"id": best["id"], "name": best["name"], "similarity": best_similarity}


async def delete_reference_product(product_id: str) -> bool:
    conn = get_conn()
    async with _write_lock:
        cursor = await conn.execute(
            "DELETE FROM reference_products WHERE id = ?", (product_id,)
        )
        await conn.commit()
        deleted = cursor.rowcount > 0
    return deleted
