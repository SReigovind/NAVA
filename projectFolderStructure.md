# NAVA Project Folder Structure

This file explains what each folder is for and where to place artifacts, data, and code. It is meant to keep the repo organized and reduce confusion during development and demo prep.

## Root Layout

- .nava/
  - Python virtual environment (local only). Do not commit.
- nava_core/
  - Application source code. Organized by the four modules plus shared code.
- models/
  - Model artifacts and weights. Keep large binaries here and gitignore them.
- data/
  - Raw and processed datasets used for experimentation and evaluation.
- knowledge_base/
  - Curated PDF sources and the local vector index for RAG.
- scripts/
  - One-off utilities such as ingestion, indexing, or data prep scripts.
- tests/
  - Unit and integration tests.
- notebooks/
  - Research and experiment notebooks.
- assets/
  - Static images, diagrams, UI mockups, and demo visuals.
- logs/
  - Runtime logs (local only).
- docs/
  - Project reports and design documents.

## Application Code Layout (nava_core)

- nava_core/mizhi/
  - Disease detection and VNIR monitoring logic.
  - nava_core/mizhi/detection/ for EfficientNet inference and Grad-CAM.
  - nava_core/mizhi/vnir/ for Thanal ONNX inference and checkpoint logic.
- nava_core/mozhi/
  - Chat, memory, and translation pipeline.
  - nava_core/mozhi/chat/ for LLM API calls and chat handlers.
  - nava_core/mozhi/memory/ for summarization and storage.
  - nava_core/mozhi/translation/ for Bhashini integration and entity handling.
- nava_core/yukthi/
  - RAG pipeline and explainability integration.
  - nava_core/yukthi/rag/ for ingestion, indexing, and retrieval.
  - nava_core/yukthi/xai/ for Grad-CAM helpers if separated from Mizhi.
- nava_core/gathi/
  - Orchestration and UI layer.
  - nava_core/gathi/api/ for FastAPI endpoints and request schemas.
  - nava_core/gathi/ui/ for the web UI.
- nava_core/shared/
  - Shared config, schemas, and utilities.
  - nava_core/shared/config/ for settings and env parsing.
  - nava_core/shared/schemas/ for Pydantic models and API contracts.
  - nava_core/shared/utils/ for common helpers.

## Model Artifacts (models)

- models/mizhi/
  - EfficientNet checkpoint(s), for example: best_model.pth
- models/thanal/
  - VNIR ONNX model, for example: ThanalModel.onnx
- models/mozhi/
  - Optional local summarizer models or tokenizer assets.
- models/yukthi/
  - Optional local embedding models or cached weights.

## Data (data)

- data/raw/
  - Original datasets, as downloaded.
- data/processed/
  - Preprocessed or augmented datasets.
- data/cache/
  - Intermediate artifacts or temporary files.

## Knowledge Base (knowledge_base)

- knowledge_base/sources/
  - Authoritative PDFs (KAU, ICAR, state pesticide rules).
- knowledge_base/index/
  - ChromaDB or other vector index files.

## Notes

- Keep large artifacts out of git. Use .gitignore and document expected paths.
- All file names should be ASCII to avoid cross-platform issues.
- If you add new folders, update this document so the team stays aligned.
