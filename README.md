# Plano CV — Minimal Build

A stripped-down rebuild of Plano's computer vision pipeline, scoped to
**prove the detection → embedding → matching logic is correct in
isolation** before reconnecting any backend (Node/BullMQ/MinIO) or
frontend (Angular) layers.

## What this replaces from the old system

| Old approach | New approach |
|---|---|
| Manual `planogram_cells` entry (row, position typed by hand) | Upload one reference shelf photo — items auto-detected and embedded |
| Row clustering (`cluster_rows`, fixed 50px tolerance) + Needleman-Wunsch sequence alignment | Full-shelf Hungarian matching (embedding similarity + spatial position combined), no row dependency at all |
| 13-table Postgres schema | 1 table: `reference_products(id, name, embedding)` |
| MinIO, BullMQ, multi-tenant auth | Local file uploads, synchronous requests, no auth |
| Angular admin/agent/client-manager apps | One static HTML page with vanilla JS + canvas overlays |

## What this deliberately does NOT include yet

- Facing count grouping (every detection is one independent item)
- Row/position numbers (matching is spatial, not grid-based)
- Multi-store / multi-client support
- Job queues, background workers
- Image object storage (MinIO) — uploads are processed in-memory only
- Historical tracking, scoring trends, flagging rules
- Authentication

## Setup

### 1. Start Postgres (with pgvector)

```bash
docker compose up -d
```

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# edit .env if needed — defaults work for local docker-compose setup
```

### 4. Run the service

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser — this serves the debug
console (`static/index.html`) directly, no separate frontend build step.

## Usage flow

1. **Add Reference Products** — upload a clean photo of each SKU (Coke,
   Pepsi, Lays...) with its name. This computes and stores its 2048-d
   ResNet50 embedding.

2. **Generate Planogram** — upload a photo of the *correctly stocked*
   shelf. The system detects every product, matches each to its nearest
   reference product, and returns a flat list (`expected_items`) — no
   manual row/position entry, no facing_count collapsing. Copy the
   returned `planogram_id`.

3. **Run Compliance Check** — paste the `planogram_id`, upload a photo
   of an actual shelf to audit. Returns an annotated image plus
   matched / missing / wrong_product / unexpected breakdowns, using
   full-shelf Hungarian assignment (no row clustering).

4. **Debug Trace** — for any image, see exact counts at every pipeline
   stage (raw YOLO boxes → embedding matches → accepted/rejected) in
   one JSON response. Built specifically to avoid guessing where
   detections get lost between stages.

## Known carry-over issues to verify on your own data

These were observed bugs/risks in the original system — verify they
don't resurface here before reconnecting backend/frontend:

- **Embedding quality gap between products**: if one reference product
  (e.g. Coke) consistently scores 10-15% lower similarity than another
  (e.g. Pepsi) under the same shelf conditions, the reference image for
  that product likely needs to be retaken under more representative
  lighting/angle — this is a data quality issue, not a code bug.
- **Low-light detection misses**: if YOLO fails to detect items in
  poorly-lit shelf rows, lower `YOLO_CONFIDENCE` in `.env` as a test,
  and confirm CLAHE preprocessing (already applied by default here) is
  sufficient, or whether the reference YOLO model itself needs
  retraining/fine-tuning for low-light retail conditions.
- **Threshold tuning**: `SIMILARITY_THRESHOLD` and
  `REFERENCE_MATCH_THRESHOLD` in `.env` are starting points (0.60,
  0.55). Tune per-deployment based on actual embedding score
  distributions seen via the Debug Trace endpoint.

## Next steps (once CV logic is verified correct)

1. Reintroduce Postgres tables for `clients`, `stores`, `sections`,
   `compliance_jobs`, `compliance_results` — the matching/generation
   logic in `services/` doesn't need to change, only the persistence
   layer around it.
2. Swap local file handling for MinIO.
3. Wrap synchronous compliance checks in BullMQ for async processing.
4. Build the real Angular frontend, replacing `static/index.html`.
5. Add auth + role-based access.
