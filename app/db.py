"""
Minimal DB layer for Plano CV.

Scope is intentionally tiny: ONE table (reference_products) storing
2048-d ResNet50 embeddings via pgvector. No clients/stores/sections/
planograms/jobs schema yet — that gets reintroduced once the CV logic
is proven correct.
"""
import os
import logging
import asyncpg
from typing import List, Dict, Any, Optional

logger = logging.getLogger("plano_cv.db")
logger.setLevel(logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/plano_cv")

_pool: Optional[asyncpg.Pool] = None

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS reference_products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR NOT NULL,
    embedding VECTOR(2048) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


async def init_db():
    """Create the connection pool and ensure schema exists. Call once at startup."""
    global _pool
    logger.info(f"Connecting to database...")
    _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)

    async with _pool.acquire() as conn:
        # Register pgvector codec so we can pass/receive python lists directly
        await conn.execute(SCHEMA_SQL)
        await _register_vector_codec(conn)

    logger.info("Database initialized: reference_products table ready.")


async def _register_vector_codec(conn: asyncpg.Connection):
    """Teach asyncpg how to encode/decode the VECTOR(2048) pgvector type."""
    def encode_vector(value: List[float]) -> str:
        return "[" + ",".join(str(float(x)) for x in value) + "]"

    def decode_vector(value: str) -> List[float]:
        value = value.strip("[]")
        if not value:
            return []
        return [float(x) for x in value.split(",")]

    await conn.set_type_codec(
        "vector",
        encoder=encode_vector,
        decoder=decode_vector,
        schema="public",
        format="text",
    )


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialized. Call init_db() at startup.")
    return _pool


async def close_db():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ── Reference Products ──────────────────────────────────────────────────────

async def insert_reference_product(name: str, embedding: List[float]) -> Dict[str, Any]:
    pool = get_pool()
    async with pool.acquire() as conn:
        await _register_vector_codec(conn)
        row = await conn.fetchrow(
            """
            INSERT INTO reference_products (name, embedding)
            VALUES ($1, $2)
            RETURNING id, name, created_at
            """,
            name,
            embedding,
        )
        return dict(row)


async def list_reference_products() -> List[Dict[str, Any]]:
    pool = get_pool()
    async with pool.acquire() as conn:
        await _register_vector_codec(conn)
        rows = await conn.fetch(
            "SELECT id, name, created_at FROM reference_products ORDER BY created_at DESC"
        )
        return [dict(r) for r in rows]


async def get_all_reference_products_with_embeddings() -> List[Dict[str, Any]]:
    """Used by detection/matching code to do nearest-neighbour lookups in-process."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await _register_vector_codec(conn)
        rows = await conn.fetch(
            "SELECT id, name, embedding FROM reference_products"
        )
        return [
            {"id": str(r["id"]), "name": r["name"], "embedding": list(r["embedding"])}
            for r in rows
        ]


async def find_nearest_reference_product(embedding: List[float]) -> Optional[Dict[str, Any]]:
    """
    Pure SQL nearest-neighbour lookup using pgvector's cosine distance operator (<=>).
    Returns the single closest reference product + its similarity (1 - distance).
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        await _register_vector_codec(conn)
        row = await conn.fetchrow(
            """
            SELECT id, name, 1 - (embedding <=> $1) AS similarity
            FROM reference_products
            ORDER BY embedding <=> $1
            LIMIT 1
            """,
            embedding,
        )
        if row is None:
            return None
        return {"id": str(row["id"]), "name": row["name"], "similarity": float(row["similarity"])}


async def delete_reference_product(product_id: str) -> bool:
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM reference_products WHERE id = $1", product_id
        )
        return result.endswith("1")
