# Detailed Research Diary

# Date - 1 May 2026

## Plan for the Day
- Stabilize Mizhi (EfficientNet) inference and Grad-CAM so outputs are reliable and reproducible.
- Implement VNIR (Thanal) monitoring with HSV leaf isolation, checkpoint logic, and readable metrics.
- Stand up FastAPI endpoints and a minimal UI to exercise both pipelines end to end.
- Keep a clean, reproducible project structure with CPU-only runtime and editable installs.

## Work Completed (Detailed)

### 1) Repository Structure and Project Setup
- Created a modular layout under nava_core with Mizhi, Mozhi, Yukthi, Gathi, and shared components so each domain stays isolated.
- Added project folder guide and aligned file placement: models/ for checkpoints, data/ for datasets, logs/ for outputs, knowledge_base/ for documents.
- Added pyproject.toml to make editable installs work and stop import issues when running scripts directly.
- Updated requirements.txt to enforce CPU-only PyTorch wheels (avoids accidental CUDA installs on machines without GPU).
- Confirmed model artifact placement:
  - EfficientNet-B0 checkpoint in models/mizhi.
  - Thanal ONNX model in models/thanal (onnx and onnx.data).

### 2) Mizhi (Disease Detection) Pipeline
Goal: A stable, reproducible inference pipeline with explainability.

What was implemented:
- EfficientNet-B0 predictor in nava_core/mizhi/detection/inference.py with image preprocessing, confidence scoring, and a reliability flag based on threshold.
- Grad-CAM integration in nava_core/mizhi/detection/gradcam.py to produce heatmaps and overlays.
- Label handling via nava_core/mizhi/detection/labels.py with a simple text label file (one class per line).

What we tried and adjusted:
- Initial runs had import errors when invoking scripts directly (ModuleNotFoundError). Fix: add pyproject.toml and use editable install so nava_core resolves without sys.path hacks.
- GPU wheels were pulled on earlier installs. Fix: pin CPU-only torch index in requirements.txt to prevent CUDA packages.
- Grad-CAM was too expensive or misleading on low-confidence predictions. Fix: only compute Grad-CAM when confidence exceeds a threshold and mark reliability in outputs.

Smoke test improvements:
- Built scripts/mizhi_smoke_test.py to sample test images and create side-by-side original + Grad-CAM panels.
- Added --count and --seed for reproducibility.
- Enforced max output size and consistent 1:1 layout.
- Overlaid label, confidence, reliability status, and source folder on the output image.

### 3) VNIR (Thanal) Monitoring Pipeline
Goal: Early stress detection from VNIR segmentation with clear history-based metrics.

What was implemented:
- ONNX inference wrapper in nava_core/mizhi/vnir/inference.py using onnxruntime CPU backend.
- Leaf isolation logic in nava_core/mizhi/vnir/pipeline.py using HSV masks with a green and yellow-brown cascade (matching the earlier Thanal pipeline behavior).
- VNIR analytics in nava_core/mizhi/vnir/analyzer.py tracking:
  - Baseline (first 5 scans)
  - Rolling 5-scan average
  - Previous checkpoint (baseline until scan 10)
  - Global average
  - Percent deltas vs baseline, rolling, previous checkpoint, and global
- Stress warnings when a >= 15 percent drop vs baseline is detected.

What we tried and adjusted:
- For plants with fewer than 10 scans, previous checkpoint was not meaningful. Fix: keep previous checkpoint equal to baseline until scan 10.
- Needed clearer VNIR reporting to compare current signal across multiple references. Fix: added explicit percent comparisons and included them in outputs and render panel.

Smoke test and visualization:
- Built scripts/vnir_smoke_test.py to process a time-series folder of images per plant and save panel outputs under logs/vnir.
- Implemented nava_core/mizhi/vnir/render.py to generate a clean side-by-side image: HSV isolate + VNIR map + text metrics block.

### 4) API Layer and Minimal UI
Goal: A working demo to drive both pipelines from a browser.

What was implemented:
- FastAPI app in nava_core/gathi/api/main.py with endpoints:
  - GET /api/health
  - POST /api/diagnose
  - POST /api/vnir-upload
  - GET /api/vnir-plants
  - POST /api/vnir-clear
- Shared config in nava_core/shared/config/settings.py for model paths and thresholds via env vars.
- Pydantic response schemas in nava_core/shared/schemas/api.py.
- Image helpers in nava_core/shared/utils/image.py for consistent base64 encoding.
- Minimal UI at nava_core/gathi/ui/index.html served from /.

UI design changes and fixes:
- Two tabs: Diagnose (Mizhi) and VNIR (Thanal).
- Added empty placeholder frames so the layout is stable before uploads.
- Reduced placeholder size and cleaned whitespace to avoid awkward blank panels.
- Hid Grad-CAM panel if prediction reliability is low to avoid misleading explanations.
- VNIR percent values formatted for clarity and consistency.
- Added plant history list and delete history controls (list + clear) connected to API.

### 5) Documentation and Diary Updates
- Updated projectFolderStructure.md to reflect actual layout.
- Created workDoneSoFar.md (short summary) and workDoneSoFarDepth.md (detailed diary).
- Renamed detailedWorkDone.md to workDoneSoFarDepth.md to match naming convention.

## Failures, Issues, and Fixes (Explicit Log)
- Import errors when running scripts directly.
  - Cause: package not installed in editable mode.
  - Fix: add pyproject.toml and use pip install -e .

- CUDA packages installed on CPU-only environment.
  - Cause: default PyTorch index.
  - Fix: set CPU-only index in requirements.txt.

- Grad-CAM being generated for low-confidence predictions.
  - Cause: no reliability gating.
  - Fix: skip Grad-CAM when confidence below threshold; return only original image and reliability flag.

- VNIR checkpoint logic unclear for early scans.
  - Cause: less than 10 scans made checkpoint unstable.
  - Fix: keep previous checkpoint equal to baseline until scan 10.

- UI showed broken image placeholders before upload.
  - Cause: missing placeholder strategy.
  - Fix: use empty data URLs and minimal frame boxes.

## Design Decisions Captured Today
- Keep Mizhi in PyTorch for Grad-CAM support (do not convert to ONNX yet).
- Keep VNIR in ONNX for lightweight CPU inference.
- Separate concerns: detection, VNIR, API, and UI are modular to allow independent testing.
- Enforce CPU-only environment to simplify deployment on typical lab machines.
- Use simple, explicit CSV history for VNIR per plant to keep auditing easy.

## Next Steps (Planned)
- RAG ingestion
  - Collect authoritative PDFs in knowledge_base/sources.
  - Build ingestion script and ChromaDB index.
  - Validate retrieval quality and citation safety.

- Chat endpoint
  - Implement /api/chat with RAG grounding and refusal for unsupported dosages.
  - Return responses with citations.

- Multilingual layer
  - Integrate Bhashini translation for Malayalam, Tamil, Hindi.
  - Preserve chemical and disease entities in translation.

- Memory system
  - Add SQLite tables for chat logs and summaries.
  - Add APScheduler jobs for daily and weekly summaries.

- UI updates
  - Add a chat tab with language selector and source display.
  - Optional: add a VNIR history timeline panel.
