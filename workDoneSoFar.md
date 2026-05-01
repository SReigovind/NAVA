# Work Done So Far

# Date - 1 May 2026

## Planned Today
- Stabilize Mizhi (EfficientNet) inference with Grad-CAM.
- Add VNIR (Thanal) monitoring with HSV isolation and checkpoint logic.
- Build a minimal FastAPI backend and a lightweight web UI to test endpoints.

## Completed Today
- Added EfficientNet inference + Grad-CAM utilities with a robust label loader.
- Built a smoke test that samples multiple images, supports seeding, and outputs side-by-side source/Grad-CAM panels with metadata.
- Implemented VNIR pipeline with HSV isolation, ONNX inference, ratio tracking, and alert logic.
- Added VNIR panel renderer and a VNIR smoke test with ordered time-series input.
- Implemented rolling averages, baseline, global average, and comparisons in VNIR stats.
- Added FastAPI endpoints for diagnose and VNIR upload, plus health check.
- Added a minimal UI served from the API, with:
  - Diagnose view: original image + Grad-CAM (hidden on unreliable results)
  - VNIR view: HSV/VNIR images, comparisons, and plant history controls
- Added VNIR history management: list existing plant IDs and delete history from UI.
- Improved UI formatting for VNIR comparison percentages and image placeholders.
- Added config handling and shared schemas/utils for API responses.

## Next Steps
- RAG ingestion: collect PDFs in knowledge_base/sources and build ChromaDB index.
- Add /api/chat endpoint with RAG grounding and citation checks.
- Integrate Bhashini translation for Malayalam/Tamil/Hindi with entity preservation.
- Add SQLite-backed memory summaries with APScheduler jobs.
- Extend UI with a chat tab and source display.
