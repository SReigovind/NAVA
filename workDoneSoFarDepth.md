# Work Done So Far

# Date - 1 May 2026

## Planned Today
- Stabilize Mizhi (EfficientNet) inference with Grad-CAM and reproducible tests.
- Implement VNIR (Thanal) monitoring with HSV isolation, checkpoint logic, and clear outputs.
- Stand up FastAPI endpoints with a minimal UI to test both pipelines.

## Completed Today
- Repository structure and model assets
  - Created modular package layout under nava_core with Mizhi, Mozhi, Yukthi, Gathi, and shared.
  - Added a clear folder guide and aligned model placement.
  - Loaded EfficientNet-B0 checkpoint from models/mizhi and Thanal ONNX from models/thanal.
  - Added label map file for 34 classes and wired label loading.

- Mizhi detection pipeline
  - Implemented EfficientNet-B0 inference with confidence threshold and reliability flag.
  - Added Grad-CAM generator on EfficientNet features layer.
  - Built a robust smoke test that:
    - Samples multiple images randomly (seed supported).
    - Generates side-by-side source + Grad-CAM output.
    - Renders labels, confidence, reliability, and source folder in output image.
    - Enforces max output size and consistent layout.

- VNIR monitoring pipeline
  - Implemented HSV leaf isolation logic from prior Thanal code (green / yellow-brown / none).
  - Added ONNX VNIR inference wrapper (CPU) for ThanalModel.onnx.
  - Added VNIR stats tracking with:
    - Initial baseline (first 5 scans)
    - Rolling 5-scan average
    - Previous checkpoint (baseline until scan 10)
    - Global average
    - Comparisons: vs baseline, vs rolling 5, vs previous checkpoint, vs global
  - Added warnings for >= 15 percent drop vs baseline.
  - Built VNIR smoke test that processes time-series images and saves labeled panels.

- API + UI
  - Added FastAPI app with endpoints:
    - GET /api/health
    - POST /api/diagnose
    - POST /api/vnir-upload
    - GET /api/vnir-plants
    - POST /api/vnir-clear
  - Added config settings for model paths and thresholds via environment variables.
  - Added shared Pydantic schemas and image helpers.
  - Built a minimal UI served at / with two tabs:
    - Diagnose: original image + Grad-CAM side by side (hidden when unreliable)
    - VNIR: HSV + VNIR output, metrics, and plant history controls
  - Added plant history list, use-selected, and delete history behavior.
  - Cleaned UI placeholders and formatting; compact percent display.

## Next Steps
- RAG ingestion
  - Collect authoritative PDFs in knowledge_base/sources.
  - Build ingestion script and ChromaDB index.
  - Validate retrieval and citation safety.

- Chat endpoint
  - Implement /api/chat with RAG grounding and refusal on unsupported dosages.
  - Add response schema with sources.

- Multilingual layer
  - Integrate Bhashini translation (Malayalam, Tamil, Hindi).
  - Preserve chemical and disease entities in translation.

- Memory system
  - Add SQLite tables for chat logs and summaries.
  - Add APScheduler jobs for daily and weekly summaries.

- UI updates
  - Add chat tab with language selector and source display.
  - Optional: add summary timeline panel for VNIR history.
