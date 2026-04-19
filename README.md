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
Rather than relying on a single controlled dataset, we aggregated data from multiple open-source repositories — PlantVillage, PlantWild (V1 & V2), PlantDoc, PaddyDoctor, ASDID, and Kaggle competition datasets — to cover **34 disease classes across 7 major crops**: Rice, Corn, Tomato, Soybean, Cassava, Banana, and Cucumber (including healthy class variants for each crop). A strict 300–700 filtering rule was applied to address severe class imbalance, followed by augmentation using Albumentations (geometric transforms, brightness contrast, RGB shift, Gaussian blur) to simulate real-world field conditions. The final dataset comprises **20,400 training/validation samples** and **4,089 test samples**.

**Model selection — comparison study:**
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
| CV model | EfficientNet-B0 (PyTorch → ONNX) | Phase 1 production model, converted for cross-platform serving |
| XAI | `pytorch-grad-cam` | Minimal overhead, direct EfficientNet compatibility |
| VNIR engine | Thanal (ONNX runtime) | Already built, Pi-validated, cross-platform |
| VNIR + memory storage | SQLite | Zero-cost, zero-server, sufficient for per-plant scan history |
| RAG orchestration | LangChain | Most mature, best community support |
| Vector store | ChromaDB (local) | No server required, fully free, persistent |
| Embeddings | `BAAI/bge-m3` | Best free multilingual embedding model, runs locally |
| Doc ingestion | PyMuPDF / pdfplumber | For parsing agricultural extension PDFs |
| LLM serving | Ollama + Llama 3.1 8B | Local inference, zero API cost, strong reasoning |
| Summarisation LLM | Ollama + Qwen2.5-1.5B | Fast, lightweight, for memory cron summarisation jobs |
| Multilingual | Bhashini API | Free, Government of India, Malayalam/Tamil/Hindi |
| Backend | FastAPI (Python) | Async, Python-native, minimal overhead |
| Task scheduling | APScheduler | Lightweight cron, no Redis/Celery required |
| Mobile app | Flutter (Dart) | Single codebase, Android + iOS |
| Edge inference | `onnxruntime_flutter` | On-device CV and VNIR with zero connectivity |
| Training hardware | NVIDIA A100 (JupyterHub) | Remote access for model training runs |
| Inference hardware | Mac M4 Pro (Ollama) | Unified memory, efficient local LLM serving |
| Dev hardware | Debian Linux, GTX 1650 Ti | FastAPI development and integration work |

---

### Module 1 — Mizhi: Disease Detection & VNIR Monitoring

**What it addresses:** Phase 1 established a strong disease classifier and Thanal validated VNIR-based stress estimation independently. Mizhi brings both together in a unified module — converting the existing EfficientNet-B0 model to ONNX for cross-platform serving, adding Grad-CAM visual explainability to every diagnosis, and integrating Thanal for proactive early stress monitoring through a practical smartphone-based workflow.

**Component A — EfficientNet-B0 in ONNX:**
The Phase 1 EfficientNet-B0 model (34 classes across 7 crops, 94.54% validation accuracy) is converted from its `.pth` PyTorch checkpoint to ONNX format. This single conversion enables the model to be served by the FastAPI backend via ONNX Runtime and bundled directly in the Flutter app for offline edge inference — without maintaining two separate model codebases. The confidence threshold safety gate (≥ 0.85) and rule-based prescription dictionary from Phase 1 are retained unchanged.

**Component B — Thanal VNIR integration:**
Thanal is integrated into the module with two natural entry points into the same underlying pipeline:

*Direct monitoring flow:* The user creates a named Plant ID for each plant they wish to monitor. The Flutter app schedules three local notifications daily — morning, afternoon, and evening — prompting the user to photograph that plant. Tapping a notification opens the camera directly to that plant's monitoring view. No automated camera hardware is required. The VNIR monitoring screen displays each plant's ratio timeline, current checkpoint progress (e.g. "3 of 5 scans for next checkpoint"), and any active stress alerts.

*Disease detection cross-feed:* When a user photographs a leaf for disease detection, they are prompted — "Add this scan to [Plant Name]'s VNIR history?" — allowing disease detection sessions to contribute bonus scans to the monitoring timeline as a natural side effect of normal app use. Both flows write to the same data store and are processed by the same Thanal pipeline.

---

### Module 2 — Mozhi: Multilingual Chatbot & Contextual Memory

**What it addresses:** Phase 1 has no conversational memory — every session starts fresh. A genuine digital agronomist must remember what disease appeared last week, what treatment was applied, and whether it resolved. Mozhi adds persistent contextual memory and regional language accessibility on top of the existing LLM pipeline.

**Hierarchical memory architecture:**
Rather than keeping raw chat logs in the LLM context window — which would exhaust the context limit within days of active use — Mozhi uses a four-level compression strategy to preserve farm history across an entire growing season while keeping active token usage minimal:

- **Level 1 — Live context window:** Current session raw messages, held in memory for the duration of the session
- **Level 2 — Daily summary:** A background cron job runs at midnight via APScheduler, passing the past 24 hours of raw messages to a small, fast model (Qwen2.5-1.5B via Ollama) for summarisation. The compressed daily digest is stored persistently and the raw log is cleared
- **Level 3 — Weekly digest:** Daily summaries are aggregated weekly into a concise chronicle, targeting under 500 tokens per plant per week
- **Level 4 — Season chronicle:** The full growing-season record — diseases encountered, treatments applied, outcomes observed — stored persistently and injected into every LLM system prompt

When a new message arrives, the system prompt assembled for Llama 3.1 contains the season chronicle, recent daily summaries, the last few raw messages from the current session, and any RAG-retrieved context for the current query. The LLM receives a rich, compressed history of the farm without ever exceeding its context window.

**Multilingual strategy — Bhashini translation bridge:**
Rather than using a natively multilingual LLM (which tends to have weak agricultural domain vocabulary in Indian regional languages), Mozhi uses Bhashini as a translation layer around the English-language Llama 3.1 pipeline:

```
Regional language input → Bhashini → English → Llama 3.1 → English response → Bhashini → Regional language output
```

This preserves the full reasoning and domain capability of Llama 3.1 while delivering responses in Malayalam, Tamil, Hindi, and other Bhashini-supported languages. Agricultural entities — pesticide names, disease identifiers, chemical compounds — that must not be mistranslated are handled through term-level protection in the translation pipeline.

---

### Module 3 — Yukthi: RAG Pipeline & Explainable AI

**What it addresses:** LLM hallucination of chemical dosages is the most critical safety risk in agricultural AI advisory. Yukthi grounds every prescription in verified source documents and makes the model's visual reasoning transparent to the farmer — two measures that together address both safety and trust.

**RAG-grounded advisory pipeline:**
```
User query
  → BAAI/bge-m3 embedding
  → ChromaDB top-5 similarity search
  → Retrieved chunks injected into Llama 3.1 system prompt
  → Response generated with source attribution
  → Citations displayed alongside prescription in the app UI
```

The knowledge base is populated from freely available, authoritative sources: Kerala Agricultural University extension bulletins, ICAR crop disease management guidelines, and state-level pesticide regulation documents. All documents are ingested as PDFs, chunked at 512 tokens with 64-token overlap (preserving context around chemical names and dosage values), embedded with `BAAI/bge-m3`, and stored in a persistent local ChromaDB instance. No cloud vector database is required.

**Explainable AI — Grad-CAM:**
`pytorch-grad-cam` generates heatmap overlays on the final convolutional layer of EfficientNet-B0 after every disease detection. The output is an annotated image highlighting precisely which regions of the leaf — lesion patterns, colour changes, texture anomalies — contributed most to the diagnosis. This overlay is returned alongside every detection result in the app, giving the farmer a visual justification for the AI's conclusion rather than a black-box label.

---

### Module 4 — Gathi: Mobile App & Backend Orchestration

**What it addresses:** Phase 1 is a Gradio web interface — adequate for demonstration but not deployable for a farmer in the field. Gathi is the integration and delivery layer that brings all of NAVA's capabilities together through a single mobile application, orchestrates communication between all modules via a FastAPI backend, and ensures the system remains functional under low or no connectivity.

**Backend — FastAPI:**
FastAPI serves as the central orchestration layer, routing requests between the Flutter app and the underlying modules:

```
POST /api/diagnose       →  ONNX EfficientNet inference + Grad-CAM overlay
POST /api/chat           →  Mozhi memory assembly + Llama 3.1 via Ollama
POST /api/rag            →  Yukthi ChromaDB retrieval + augmented generation
POST /api/vnir-upload    →  Thanal ONNX processing + checkpoint logic
```

Ollama serves Llama 3.1 8B locally on the M4 Pro using unified memory — no recurring API costs. APScheduler manages the Mozhi memory cron jobs within the same process.

**Mobile app — Flutter:**
A single Flutter codebase targeting Android and iOS. Core screens:
- **Diagnose** — camera capture, disease result with Grad-CAM overlay, confidence score, short rule-based prescription, full LLM-generated treatment plan
- **Chat** — conversational interface with full farm history context, regional language toggle via Bhashini
- **Monitor** — VNIR plant list, individual plant ratio timeline, checkpoint progress, stress alerts
- **Settings** — plant ID management, notification preferences, language selection

**Edge deployment — offline fallback:**
Both the EfficientNet-B0 and Thanal models are in ONNX format, allowing them to be bundled directly in the Flutter app via `onnxruntime_flutter`. When connectivity is unavailable, the app runs both models entirely on-device:

```
Offline mode:
  Camera → ONNX EfficientNet → disease class + static rule-based prescription
  Camera → ONNX Thanal → NIR/Green ratio → local SQLite checkpoint update

On reconnect:
  Full RAG + LLM pipeline resumes
  Any scans taken offline are synced and processed
```

This ensures that the two most critical capabilities — disease identification and VNIR stress monitoring — remain available regardless of network conditions.

---

## Development Roadmap

### April (current)
- Finalise architecture and confirmed tech stack
- Set up FastAPI project skeleton with all endpoints as stubs
- Convert EfficientNet-B0 `.pth` checkpoint to ONNX format
- Integrate Thanal ONNX into the FastAPI backend
- Mizhi: wire ONNX EfficientNet inference into `/api/diagnose`
- Yukthi: ingest KAU and ICAR extension documents into ChromaDB
- Yukthi: implement Grad-CAM overlay and return annotated image via API

### May
- Mozhi: build hierarchical memory layer and APScheduler cron summarisation
- Mozhi: integrate Bhashini API translation bridge
- Gathi: Flutter UI — all core screens scaffolded and connected to the API
- Gathi: bundle ONNX models into Flutter for offline edge fallback
- End-to-end integration testing across all modules
- Performance benchmarking, report finalisation, demonstration preparation

---

## Future Work

The following directions are identified as meaningful extensions to NAVA beyond the current project scope:

- **IoT sensor fusion:** A hardware-agnostic ingestion endpoint that accepts real-time readings from field sensor nodes (soil moisture probes, weather stations, ESP32-based cameras) and injects environmental context into LLM advisory — making disease likelihood assessment hyper-local and dynamic
- **Multi-label disease detection:** Upgrading the classifier to detect multiple concurrent pathologies per image, requiring a larger annotated dataset with co-occurrence labels and a sigmoid multi-label output head replacing the current softmax
- **Expanded crop coverage:** Extending the Superset dataset to include additional regional crops relevant to Kerala and broader South Asian farming contexts
- **On-device LLM:** Deploying a quantised small language model (4-bit) directly on the mobile device for fully offline advisory generation, removing the dependency on a local inference server

---

## Project Information

- **Degree:** M.Sc. Artificial Intelligence and Machine Learning (2024–2026)
- **Institution:** School of Artificial Intelligence and Robotics, Mahatma Gandhi University, Kottayam, Kerala
- **Team:** Dhanus VS (MG24C3135006) · Sreegovind S (MG24C3135011)
- **Internal Guide:** Ms. Mintu Movi, Assistant Professor, School of AI & Robotics, MGU
- **External Guide:** Dr. Hsing-Kuo Pao, Professor, Department of Computer Science and Information Engineering, National Taiwan University of Science and Technology

---