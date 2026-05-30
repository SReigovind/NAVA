# NAVA — Next-gen Agricultural Virtual Assistant

> *From diagnostic tool to digital agronomist.*

**NAVA** is a comprehensive, end-to-end AI platform designed to democratize expert-level agricultural guidance for smallholder farmers — particularly in regions like Kerala, India, where access to timely agronomic expertise is critically scarce. It combines computer vision, large language models, retrieval-augmented generation, and virtual spectral analysis into a single, accessible web application that acts as a trusted digital partner across the entire crop lifecycle.

---

## The Problem

Crop diseases account for an estimated **20–30% reduction in global agricultural yields annually**. For smallholder farmers, this is not a statistic — it is an economic crisis. The traditional response is manual visual inspection, either by the farmer themselves or through a local extension officer. This process is:

- **Slow** — by the time an expert is consulted, infection has often spread
- **Subjective and error-prone** — human diagnosis is inconsistent, especially for early-stage or overlapping diseases
- **Inaccessible** — rural farming ecosystems face a severe shortage of agronomic expertise, particularly in local languages
- **Reactive, not proactive** — standard RGB-based tools can only detect disease after visible lesions have formed, which is frequently too late to prevent significant crop loss

Existing AI tools do not adequately solve this. Most deep learning models are trained on controlled laboratory datasets and fail under real-world field conditions — variable lighting, complex backgrounds, multiple concurrent stressors. Generic large language models, when used for agricultural advice, are prone to hallucinating chemical dosages and misidentifying crop-specific entities, creating genuine safety risks for farmers who act on that advice. And virtually no existing solution integrates early stress detection, verified advisory generation, contextual memory, and local language support into a single platform accessible on a basic smartphone.

---

## The Solution

NAVA is built around four principles:

**1. Early detection, not just diagnosis.**
By estimating virtual near-infrared (VNIR) spectral data from standard smartphone RGB images, NAVA detects physiological plant stress before visible lesions form — enabling intervention weeks earlier than conventional tools.

**2. Safe, grounded advice.**
All treatment recommendations are tethered to a verified knowledge base of agricultural extension documents and chemical regulations via Retrieval-Augmented Generation (RAG). The LLM cannot hallucinate dosages it must cite from source.

**3. Transparent reasoning.**
Explainable AI (XAI) using Grad-CAM visually highlights the exact leaf regions that triggered a diagnosis. Farmers see *why* the model flagged their crop, not just *what* it flagged — building the trust necessary for them to act on AI advice.

**4. Contextual and persistent.**
NAVA remembers a farm's full history across an entire growing season. Its chat assistant maintains a multi-level memory hierarchy — recent messages, rolling summaries, and long-term rollups — so context is never lost even in extended sessions.

---

## Phase 1 — Validated Diagnostic Pipeline

Phase 1 established and validated the core dual-model architecture: a computer vision (CV) module for disease identification paired with a natural language processing (NLP) module for treatment prescription generation.

**Dataset — the Superset strategy:**
Rather than relying on a single controlled dataset, we aggregated data from multiple open-source repositories — PlantVillage, PlantWild (V1 & V2), PlantDoc, PaddyDoctor, ASDID, and Kaggle competition datasets — to cover **34 disease classes across 7 major crops**: Rice, Corn, Tomato, Soybean, Cassava, Banana, and Cucumber (including healthy class variants for each crop). A strict 300–700 filtering rule was applied to address severe class imbalance, followed by augmentation using Albumentations (geometric transforms, brightness contrast, RGB shift, Gaussian blur) to simulate real-world field conditions. The final dataset comprises **20,400 training/validation samples** and **4,089 test samples**.

**Model selection — comparison study:**
Three architectures were trained and compared under identical conditions:

| Model | Best Validation Accuracy | Training Time |
|---|---|---|
| ResNet-50 | 85.39% | 5 min 00 sec |
| MobileNetV2 | 83.53% | 4 min 34 sec |
| EfficientNet-B0 | **94.54%** | 4 min 38 sec |

EfficientNet-B0 was selected as the production backbone — it achieved the highest accuracy at comparable speed to the lightweight MobileNetV2, confirming that compound scaling outperforms both depth-only and width-only scaling for this task.

---

### Thanal — VNIR Estimation Engine *(competition-built, competition-validated)*

Thanal is a dedicated virtual near-infrared estimation model developed as part of a national-level competition, now integrated as NAVA's early stress detection engine.

**Architecture:** UNet with Attention Gates. Leaf-region isolation is performed via HSV multi-cascade filtering before NIR estimation, ensuring the model focuses on plant tissue rather than background noise.

**Performance:** 28 dB PSNR · 0.85 SSIM on held-out validation data.

**Deployment:** Exported to ONNX runtime format, validated and deployed on a Raspberry Pi 4 — confirming the model is viable on edge hardware with no GPU.

**Monitoring logic:** Rather than relying on absolute NIR/Green ratio thresholds, Thanal uses a **rolling checkpoint strategy**: the first 5 scans of a monitored plant establish a personalised baseline; every subsequent scan is compared against this baseline and a rolling average. A significant drop in the NIR/Green ratio triggers a tiered stress alert (WARNING or CRITICAL).

---

## Phase 2 — The Full Digital Agronomist

Phase 2 transforms NAVA from a validated diagnostic pipeline into a proactive, intelligent, and accessible agricultural ecosystem. It is now **complete** and structured around five named modules.

---

### Module: Gathi — API Server & Frontend

**FastAPI** backend serving a **React SPA** (Vite, React Router v6). All modules are integrated via a dependency-injection system (`deps.py`) with singleton preloading at startup.

- REST API with Bearer token authentication (bcrypt password hashing, `secrets.token_hex` session tokens)
- FastAPI lifespan context preloads EfficientNet-B0, Thanal ONNX, and ChromaDB in background threads at boot
- React SPA with dark-mode design system, glassmorphism cards, word-by-word chat animation, and per-message RAG carousel
- Full farm management UI: field creation, crop tracking, per-plant disease and VNIR history panels
- Grad-CAM heatmap rendering inline with diagnosis results
- Collapsible knowledge source carousel on every RAG-grounded chat response

---

### Module: Mizhi — Disease Detection & VNIR Monitoring

**Disease detection pipeline:**
- EfficientNet-B0 (PyTorch) fine-tuned on the 34-class Superset; confidence threshold gate at 80% (below → `UNRELIABLE`, no Grad-CAM generated)
- Grad-CAM explainability layer returns a heatmap overlay alongside the prediction
- Results stored as `diagnose` events per plant; field-wide context auto-regenerates after each scan

**VNIR monitoring pipeline:**
- HSV leaf isolator filters background before Thanal ONNX inference
- Rolling checkpoint analysis: baseline (first 5 scans) → per-scan ratio comparison vs. baseline, global average, rolling average, and previous checkpoint
- Results stored as `vnir` events and in a dedicated `vnir_history` timeseries table per plant
- Status tiers: CALIBRATING → OK → WARNING → CRITICAL

---

### Module: Mozhi — Conversational AI & Memory

**ChatService orchestration:**
- Multi-level memory: last 12 messages (unsummarised) + level-1 summaries (per 14-message batch) + level-2 rollups (every 5 level-1 summaries)
- Farm context injected as structured system prompt: field metadata, all sibling crops, per-plant recent diagnose and VNIR events
- Smart crop notes: after each level-1 summary, a lightweight LLM pass extracts explicit farmer decisions and appends them as timestamped auto-notes to the crop's notes field
- Context-aware RAG routing: router receives the last assistant reply as context, enabling short follow-ups like "yes, tell me more" to correctly trigger retrieval
- Word-by-word typewriter animation on assistant messages; per-session carousel state

**Session management:**
- Sessions stored in localStorage (UUID hex, timestamp label); full history fetched from DB on session switch
- `chat_context` table anchors sessions to a specific field+crop so context is always grounded
- `chat_state` pointer tracks last summarised message ID; `chat_summaries` stores both levels

---

### Module: Yukthi — RAG Knowledge Retrieval

**Ingestion pipeline** (offline, `python ingest.py`):
- Per-crop subfolder layout in `ragsource/` — drop files, run ingest
- Section-aware chunker for `.txt` (PlantVillage disease entry format); PyMuPDF block-level chunker for `.pdf`
- BAAI/bge-small-en-v1.5 embedding (384-dim); ChromaDB persistent collections per crop (`nava_{crop}`)
- Deterministic upsert IDs; `--force` flag wipes and rebuilds from scratch

**Retrieval pipeline** (online, per chat request):
1. **QueryRouter** — Llama-3.1-8B-Instruct at `temperature=0.0, max_tokens=5`; outputs `RETRIEVE` or `SKIP`. Context-aware: receives last assistant reply for ambiguous follow-up resolution.
2. **KeywordExtractor** — same 8B model extracts 3 agronomic search terms from the enriched query (crop + detected condition + user message)
3. **Hybrid retriever** — 5 semantic candidates (cosine similarity) + ~5 keyword-filtered candidates (ChromaDB `where_document`) → merge, deduplicate, rerank (`0.7 × cosine + 0.3 × keyword_overlap`) → top 3 chunks
4. **Context injection** — chunks appended as `AGRONOMIC REFERENCE — VERIFIED SOURCE MATERIAL` system block before main LLM call

---

### Module: Shared — Foundation Layer

- `UserStore` (global `users.db`): user registry, session tokens
- `FieldStore` (per-user `user_{hash}.db`): fields, crops, plants, events, VNIR history; WAL mode; non-destructive schema migrations via `PRAGMA table_info`
- `SessionStore`: chat messages, summaries, state, and context — co-located in the per-user DB
- `Settings` frozen dataclass loaded from `.env`; all modules read from a single `get_settings()` singleton
- Pydantic v2 request/response schemas for all API endpoints

---

## Qualitative Testing Suite

A complete automated testing framework in `tests/` generates Markdown reports with embedded images and detailed internal logs, then exports PDF copies.

| Script | What it tests | Output |
|--------|--------------|--------|
| `test_disease_advanced.py` | 7 crops × (1 healthy + 1 diseased) image; GradCAM decoding; reliability gate | `disease_report.md` + `tests/outputs/` |
| `test_vnir_advanced.py` | 5-image baseline calibration + 3 stress images on Banana | `vnir_report.md` |
| `test_chat_advanced.py` | Routing decisions, RAG retrieval logs, LLM context payload, summary trigger, auto-notes extraction | `chat_report.md` |
| `export_pdfs.py` | Converts all three Markdown reports to styled A4 PDFs via `weasyprint` | `disease_report.pdf`, `vnir_report.pdf`, `chat_report.pdf` |

---

## In Progress

| Feature | Status |
|---------|--------|
| Season dropdown (Kerala 3-season calendar) | Planned — frontend only |
| Geo-weather context injection into chat | Planned — Open-Meteo API, stdlib only |

---

## Future Work

- **Multilingual support:** DeepL API Free tier for Malayalam translation (ML → EN for RAG input, LLM instructed to respond in Malayalam); EN/ML toggle in ChatPanel
- **IoT sensor fusion:** Hardware-agnostic ingestion endpoint for field sensor nodes (soil moisture, weather stations, ESP32 cameras)
- **Multi-label disease detection:** Sigmoid multi-label output head for concurrent pathology detection
- **Expanded crop coverage:** Additional regional crops relevant to Kerala and broader South Asian farming
- **On-device LLM:** Quantised (4-bit) small language model for fully offline advisory generation

---

## Project Structure

```
NAVA-AG/
├── nava_core/
│   ├── gathi/          # FastAPI server, routers, React SPA
│   ├── mizhi/          # EfficientNet-B0, Grad-CAM, Thanal VNIR pipeline
│   ├── mozhi/          # ChatService, ChatClient, SessionStore, memory
│   ├── yukthi/         # RAG: chunker, store, pipeline, router, keywords, retriever
│   └── shared/         # Config, schemas, UserStore, FieldStore, utilities
├── models/             # EfficientNet-B0.pth, ThanalModel.onnx, labels
├── ragsource/          # Per-crop knowledge base subfolders
├── logs/               # users.db, user_{hash}.db files, chroma/ vector store
├── tests/              # Advanced qualitative test suites
├── documentation/      # Documentation
├── ingest.py           # Offline RAG ingestion CLI
├── run.py              # Server entry point
├── implementation_plan.md
└── worklog.md
```

---

## Project Information

- **Degree:** M.Sc. Artificial Intelligence and Machine Learning (2024–2026)
- **Institution:** School of Artificial Intelligence and Robotics, Mahatma Gandhi University, Kottayam, Kerala
- **Team:** Dhanus VS (MG24C3135006) · Sreegovind S (MG24C3135011)
- **Internal Guide:** Ms. Mintu Movi, Assistant Professor, School of AI & Robotics, MGU

---