# NAVA — Next-gen Agricultural Virtual Assistant

> *From diagnostic tool to digital agronomist.*

**NAVA** is a comprehensive, end-to-end AI platform designed to democratize expert-level agricultural guidance for smallholder farmers — particularly in regions like Kerala, India, where access to timely agronomic expertise is critically scarce. It combines computer vision, large language models, retrieval-augmented generation, and virtual spectral analysis into a single, accessible mobile application that acts as a trusted digital partner across the entire crop lifecycle.

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

**4. Contextual, multilingual, and persistent.**
NAVA remembers a farm's history across an entire growing season. It speaks to farmers in their regional language. And it works — at a basic level — without internet connectivity.

---

## What We Have Built So Far

### Phase 1 — Validated Diagnostic Pipeline

Phase 1 established and validated the core dual-model architecture: a computer vision (CV) module for disease identification paired with a natural language processing (NLP) module for treatment prescription generation.

**Dataset — the Superset strategy:**
Rather than relying on a single controlled dataset, we aggregated data from multiple open-source repositories — PlantVillage, PlantWild (V1 & V2), PlantDoc, PaddyDoctor, ASDID, and Kaggle competition datasets — to cover **34 disease classes across 7 major crops**: Rice, Corn, Tomato, Soybean, Cassava, Banana, and Cucumber. A strict 300–700 filtering rule was applied to address severe class imbalance, followed by augmentation using Albumentations (geometric transforms, brightness contrast, RGB shift, Gaussian blur) to simulate real-world field conditions. The final dataset comprises **20,400 training/validation samples** and **4,089 test samples**.

**Model selection — ablation study:**
Three architectures were trained and compared under identical conditions:

| Model | Best Validation Accuracy | Training Time |
|---|---|---|
| ResNet-50 | 85.39% | 5 min 00 sec |
| MobileNetV2 | 83.53% | 4 min 34 sec |
| EfficientNet-B0 | **94.54%** | 4 min 38 sec |

EfficientNet-B0 was selected as the production backbone — it achieved the highest accuracy at comparable speed to the lightweight MobileNetV2, confirming that compound scaling outperforms both depth-only and width-only scaling for this task.

**Pipeline architecture:**
The system is orchestrated by three Python modules:
- `cv.py` — loads the trained EfficientNet-B0 checkpoint, applies preprocessing transforms, runs inference, and returns a predicted class label with a confidence score
- `nlp.py` — receives the predicted label, constructs a context-aware prompt by parsing crop and disease tokens, queries the Llama 3.1 8B model via the Hugging Face Inference API, and returns both a rule-based short prescription and a detailed LLM-generated treatment plan
- `app.py` — orchestrates the full pipeline via a Gradio web interface, implementing a confidence threshold safety gate (≥ 0.85) that halts prescription generation for low-confidence predictions to prevent misinformation

**Safety mechanisms:**
- Predictions below 0.85 confidence are flagged as `UNRELIABLE` — no specific prescription is generated
- A static rule-based dictionary provides immediate, pre-verified short prescriptions for all 34 disease classes before the LLM response arrives
- Healthy plant detection returns maintenance advice rather than triggering the prescription pipeline

---

### Thanal — VNIR Estimation Engine *(competition-built, competition-validated)*

Thanal is a dedicated virtual near-infrared estimation model developed as part of a national-level competition, now being integrated as NAVA's early stress detection engine.

**Architecture:** UNet with Attention Gates. Leaf-region isolation is performed via HSV multi-cascade filtering before NIR estimation, ensuring the model focuses on plant tissue rather than background noise.

**Performance:** 28 dB PSNR · 0.85 SSIM on held-out validation data.

**Deployment:** Exported to ONNX runtime format, validated and deployed on a Raspberry Pi 4 — confirming the model is viable on edge hardware with no GPU.

**Monitoring logic:** Rather than relying on absolute NIR/Green ratio thresholds (which would require crop-specific calibration data that does not exist at sufficient scale), Thanal uses a **rolling checkpoint strategy**:
- The first 5 scans of a monitored plant establish a personalised baseline checkpoint
- Every subsequent 5 scans form a new checkpoint
- Each new checkpoint is compared to the previous one — a significant drop in the NIR/Green ratio triggers a stress alert
- This approach is **crop-agnostic** by design: the comparison is relative to the plant's own history, not a fixed species threshold

**Existing deployment (Pi/IoT):** Thanal currently supports multi-camera continuous monitoring on Raspberry Pi 4, with one camera per plant and configurable scan intervals. This remains available as a hardware deployment option for farm-scale use.

---

## Phase 2 — The Full Digital Agronomist

Phase 2 transforms NAVA from a validated diagnostic pipeline into a proactive, intelligent, and accessible agricultural ecosystem. It is structured around four named modules, each addressing a specific gap identified in Phase 1.

### Confirmed Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| CV training | PyTorch + `timm` (EfficientNet-B4) | Best-in-class accuracy, compound scaling |
| Augmentation | Albumentations | Fastest, most flexible augmentation library |
| XAI | `pytorch-grad-cam` | Minimal overhead, direct EfficientNet compatibility |
| VNIR engine | Thanal (ONNX runtime) | Already built, Pi-validated, cross-platform |
| VNIR storage | SQLite | Zero-cost, zero-server, adequate for per-plant scan history |
| RAG orchestration | LangChain | Most mature, best community support |
| Vector store | ChromaDB (local) | No server required, fully free, persistent |
| Embeddings | `BAAI/bge-m3` | Best free multilingual embedding model, runs locally |
| Doc ingestion | PyMuPDF / pdfplumber | For parsing agricultural extension PDFs |
| LLM serving | Ollama + Llama 3.1 8B | Local inference, zero API cost, strong reasoning |
| Summarisation LLM | Ollama + Qwen2.5-1.5B | Fast, cheap, for memory cron jobs |
| Multilingual | Bhashini API | Free, Government of India, Malayalam/Tamil/Hindi |
| Backend | FastAPI (Python) | Async, Python-native, minimal overhead |
| Task scheduling | APScheduler | Lightweight cron, no Redis/Celery required |
| Mobile app | Flutter (Dart) | Single codebase, Android + iOS |
| Edge inference | `tflite_flutter` + `onnxruntime_flutter` | On-device CV and VNIR with zero connectivity |
| Training hardware | NVIDIA A100 (JupyterHub) | Remote access for model training runs |
| Inference hardware | Mac M4 Pro (Ollama) | Unified memory, efficient LLM serving |
| Dev hardware | Debian Linux, GTX 1650 Ti | FastAPI development and integration |

---

### Module 1 — Mizhi: Multi-Disease Detection & VNIR Monitoring

**What it solves:** Phase 1 EfficientNet-B0 performs single-label classification — it predicts one disease per image. Real crops frequently present multiple concurrent pathologies. Additionally, RGB-only detection is inherently reactive, identifying disease only after visible lesions appear.

**Component A — Multi-label disease classifier:**
- Backbone upgraded from EfficientNet-B0 to **EfficientNet-B4** via `timm` for higher capacity
- Output head changed from softmax (single-label) to **sigmoid with BCEWithLogitsLoss** — each disease class gets an independent probability score, enabling simultaneous detection of multiple conditions
- Superset dataset extended with multi-label annotations for co-occurring disease pairs, with synthetic co-occurrence augmentation
- Confidence threshold logic retained from Phase 1, now applied per-class

**Component B — Thanal VNIR integration:**
Thanal is integrated as a named sub-system within Mizhi, with two distinct entry points into the same pipeline:

*Direct monitoring flow:* The user creates a Plant ID for each plant they wish to monitor. Flutter sends three scheduled local notifications daily (morning / afternoon / evening — suggested times aligned with natural light consistency). The user taps the notification, photographs the plant, and Thanal processes the image. No automated camera hardware is required.

*Disease detection cross-feed:* When a user photographs a leaf for disease detection, they are prompted — "Add this scan to [Plant Name]'s VNIR history?" — allowing disease detection sessions to contribute bonus scans to the monitoring timeline without any additional action.

Both paths write to the same SQLite schema with a `source` field (`direct` | `disease_detection_crossfeed`), keeping the pipeline unified.

**SQLite schema (VNIR):**
```sql
CREATE TABLE plants (
    plant_id    TEXT PRIMARY KEY,
    name        TEXT,
    created_at  INTEGER
);

CREATE TABLE vnir_scans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id        TEXT,
    timestamp       INTEGER,
    nir_green_ratio REAL,
    source          TEXT,       -- 'direct' | 'disease_detection_crossfeed'
    checkpoint_id   INTEGER     -- null until assigned to a checkpoint
);

CREATE TABLE vnir_checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id        TEXT,
    timestamp       INTEGER,
    avg_ratio       REAL,
    scan_count      INTEGER,
    alert_triggered INTEGER DEFAULT 0
);
```

**Checkpoint logic:**
```python
scans = get_scans_since_last_checkpoint(plant_id)
if len(scans) >= 5:
    new_checkpoint_ratio = mean([s.nir_green_ratio for s in scans])
    delta = previous_checkpoint.avg_ratio - new_checkpoint_ratio
    if delta > ALERT_THRESHOLD:
        trigger_stress_alert(plant_id)
    save_checkpoint(plant_id, new_checkpoint_ratio, scans)
```

If the user misses a scan, the checkpoint simply takes longer to form. No broken state, no error — the UI shows "4/5 scans collected for next checkpoint."

---

### Module 2 — Mozhi: Multilingual Chatbot & Contextual Memory

**What it solves:** Phase 1 has no memory — every session starts fresh. The LLM has no knowledge of what disease this farm had last week, what treatment was applied, or whether it worked. For NAVA to function as a real agronomist it must remember.

**Hierarchical memory architecture:**
Rather than keeping raw chat logs in the LLM context window (which would exhaust the context limit within days), Mozhi uses a four-level compression strategy:

```
Level 1 — Live context window
  Current session raw messages (~8K tokens, Llama 3.1 native context)

Level 2 — Daily summary  [cron job, runs at midnight]
  APScheduler triggers a Qwen2.5-1.5B summarisation of the past 24h
  Raw messages → compressed daily digest → stored in SQLite
  Raw logs cleared after summarisation

Level 3 — Weekly digest  [cron job, runs Sunday midnight]
  Daily summaries for the week → aggregated weekly chronicle
  Target: < 500 tokens per week per plant

Level 4 — Season chronicle  [persistent]
  Full growing-season history: diseases encountered, treatments
  applied, outcomes observed — injected into every LLM system prompt
```

**Context injection (assembled per request):**
```
[Season chronicle for this plant]
[This week's daily summaries]
[Last 5 raw messages from current session]
[RAG-retrieved context chunks for this query]
[User message]
```

**SQLite schema (memory):**
```sql
CREATE TABLE sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id    TEXT,
    timestamp   INTEGER,
    role        TEXT,   -- 'user' | 'assistant'
    content     TEXT
);

CREATE TABLE daily_summaries (
    date        TEXT,
    plant_id    TEXT,
    summary     TEXT,
    PRIMARY KEY (date, plant_id)
);

CREATE TABLE season_chronicle (
    plant_id    TEXT,
    week        TEXT,
    digest      TEXT,
    PRIMARY KEY (plant_id, week)
);
```

**Multilingual strategy — Bhashini translation bridge:**
Rather than using a natively multilingual LLM (which tends to have weak agricultural domain vocabulary in Indian languages), Mozhi uses Bhashini as a translation layer:
```
Regional language input → Bhashini → English → Llama 3.1 → English response → Bhashini → Regional language output
```
This preserves the full reasoning capability of Llama 3.1 while delivering responses in Malayalam, Tamil, Hindi, and other supported languages. Agricultural entities (pesticide names, disease terms) that must not be translated remain in their standard form.

---

### Module 3 — Yukthi: RAG Pipeline, Explainable AI & IoT Fusion

**What it solves:** LLM hallucination of chemical dosages is the most critical safety risk in agricultural AI. Yukthi grounds every prescription in verified source documents, makes the model's visual reasoning transparent to the farmer, and integrates real-world environmental data to make advice hyper-local.

**RAG pipeline:**
```
User query
  → BAAI/bge-m3 embedding
  → ChromaDB top-5 similarity search
  → Retrieved chunks injected into Llama system prompt
  → Response generated with mandatory source citation
  → Citations displayed alongside prescription in UI
```

Knowledge base sources: Kerala Agricultural University extension bulletins, ICAR crop disease management guidelines, state-level pesticide regulation documents. All ingested as PDFs via PyMuPDF, chunked at 512 tokens with 64-token overlap (preserves context around chemical names), embedded with `BAAI/bge-m3`, stored in a persistent local ChromaDB instance.

Keyword-weighted loss (KAIT-inspired): agricultural entities — pesticide names, disease identifiers, dosage values — are flagged as high-precision tokens during fine-tuning, reducing the probability of factual errors on the terms that matter most.

**Explainable AI — Grad-CAM:**
`pytorch-grad-cam` generates heatmap overlays on the final convolutional layer of the EfficientNet backbone. The output is an annotated image showing precisely which leaf regions — lesion patterns, colour changes, texture anomalies — contributed most to the diagnosis. This is returned alongside every disease detection result. For environmental-factor-based advice (humidity, temperature), SHAP values provide a text justification: "High humidity (>85%) in the past 48h increases blast risk by X."

**IoT sensor fusion:**
A single agnostic endpoint `/api/iot-ingest` accepts JSON-formatted readings from any sensor node — ESP32-CAM, weather station, soil moisture probe, or external weather API. The latest reading for a given location is retrieved and injected into the LLM system prompt before inference, so disease likelihood advice dynamically reflects current field conditions.

Virtual sensor fallback: if physical IoT data is unavailable, a gradient boosting model infers missing environmental parameters (soil moisture, leaf wetness duration) from basic weather API data (temperature, humidity, rainfall) — ensuring hyper-local advice even for farms with no sensor hardware.

---

### Module 4 — Gathi: Flutter App, FastAPI Backend & Edge Deployment

**What it solves:** Phase 1 is a Gradio web interface — functional for demonstration but not deployable for a Kerala farmer with an Android phone and intermittent connectivity. Gathi delivers NAVA as a real mobile application with offline fallback.

**Backend — FastAPI:**
```
POST /api/diagnose       →  CV inference + Grad-CAM overlay
POST /api/chat           →  Mozhi memory assembly + Llama 3.1
POST /api/rag            →  Yukthi RAG retrieval
POST /api/vnir-upload    →  Thanal processing + checkpoint logic
POST /api/iot-ingest     →  Sensor data ingestion
```
Ollama serves Llama 3.1 8B locally on the M4 Pro using unified memory — no recurring API costs. FastAPI handles async orchestration, routing each request to the appropriate module and aggregating the response.

**Mobile app — Flutter:**
Core screens:
- **Diagnose** — camera capture, disease result with Grad-CAM overlay, confidence score, short prescription, detailed LLM treatment plan
- **Chat** — conversational interface with farm history context, multilingual toggle
- **Monitor** — VNIR plant list, individual plant timeline, checkpoint history, stress alerts
- **Dashboard** — IoT sensor readings, weather context, environmental risk summary

**Edge deployment — offline fallback:**
EfficientNet-B0 (Phase 1 model, kept for edge use due to size) exported via `torch → ONNX → TensorFlow Lite`. The `.tflite` file is bundled in the Flutter app. `tflite_flutter` runs inference entirely on-device. Thanal's ONNX model is also bundled via `onnxruntime_flutter`.

When connectivity is unavailable:
```
Offline mode:
  Camera → TFLite EfficientNet-B0 → disease class + static prescription
  Camera → ONNX Thanal → NIR/Green ratio → local SQLite checkpoint update
  [No LLM, no RAG, no Bhashini — core functionality preserved]

On reconnect:
  Queued scans sync to backend
  Full RAG + LLM pipeline resumes
```

---

## System Architecture — Full Stack

```
┌─────────────────────────────────────────────────────┐
│                    USER LAYER                        │
│  Flutter App (Android / iOS)                        │
│  Diagnose · Chat · Monitor · Dashboard              │
└──────────────┬──────────────────────────────────────┘
               │ HTTPS / local network
┌──────────────▼──────────────────────────────────────┐
│                APPLICATION LAYER                     │
│  FastAPI Backend                                    │
│  /diagnose  /chat  /rag  /vnir-upload  /iot-ingest  │
│  Ollama (Llama 3.1 8B — M4 Pro unified memory)      │
│  Confidence threshold gate · Rule-based triage      │
└──┬──────────┬───────────┬───────────┬───────────────┘
   │          │           │           │
┌──▼──┐  ┌───▼───┐  ┌────▼───┐  ┌───▼────┐
│MIZHI│  │THANAL │  │YUKTHI  │  │ MOZHI  │
│ B4  │  │ ONNX  │  │  RAG   │  │ Memory │
│ CV  │  │ VNIR  │  │Grad-CAM│  │Bhashini│
└──┬──┘  └───┬───┘  └────┬───┘  └───┬────┘
   │          │           │           │
┌──▼──────────▼───────────▼───────────▼────────────────┐
│                    DATA LAYER                         │
│  Superset Dataset (20,400 samples)                   │
│  ChromaDB (vector store — RAG knowledge base)        │
│  SQLite (VNIR scans · checkpoints · chat memory)     │
│  Bhashini API · IoT ingest endpoint                  │
└───────────────────────────────────────────────────────┘

Edge (offline):
  TFLite EfficientNet-B0 + ONNX Thanal → bundled in Flutter APK
```

---

## Development Roadmap

### April (current)
- Finalise confirmed architecture and tech stack
- Set up FastAPI project skeleton with stub endpoints
- Integrate Thanal ONNX into Python backend (`/api/vnir-upload`)
- Define SQLite schemas for VNIR and memory storage

### May — Weeks 1–2
- Mizhi: multi-label dataset annotation + EfficientNet-B4 training on A100
- Yukthi: RAG pipeline — ingest KAU/ICAR extension documents into ChromaDB
- Yukthi: implement Grad-CAM overlay endpoint

### May — Weeks 3–4
- Mozhi: hierarchical memory layer + APScheduler cron jobs
- Mozhi: Bhashini API integration and translation bridge
- Yukthi: IoT ingest endpoint + virtual sensor fallback model
- Gathi: Flutter UI scaffolding — all core screens

### June 1–15
- Gathi: TFLite + ONNX export for offline edge fallback
- Gathi: end-to-end integration — all modules connected through app
- Performance benchmarking across all modules
- Report finalisation and demonstration preparation

---

## Project Information

**Degree:** M.Sc. Artificial Intelligence and Machine Learning (2024–2026)
**Institution:** School of Artificial Intelligence and Robotics, Mahatma Gandhi University, Kottayam, Kerala
**Team:** Dhanus VS (MG24C3135006) · Sreegovind S (MG24C3135011)
**Internal Guide:** Ms. Mintu Movi, Assistant Professor, School of AI & Robotics, MGU
**External Guide:** Dr. Hsing-Kuo Pao, Professor, Department of Computer Science and Information Engineering, National Taiwan University of Science and Technology

---
