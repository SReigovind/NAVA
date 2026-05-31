# Project Evolution: Phase 1 to Phase 2

> **Subfolder:** `non_technical/`
> **Cross-references:** [01_problem_and_vision.md](01_problem_and_vision.md) | [06_model_comparison_study.md](06_model_comparison_study.md) | [technical/01_system_architecture.md](../technical/01_system_architecture.md)

---

## Phase 1: Establishing the Core Science

Phase 1 of NAVA was a research project, not a product. Its goal was to validate the technical foundations before committing to a full system implementation.

### What Phase 1 Delivered

**The Model Comparison Study**
The formal, reproducible comparison of ResNet-50, MobileNetV2, and EfficientNet-B0 on the Superset. This established not just which architecture to use, but a rigorous methodology: same dataset, same training configuration, same hardware, same evaluation protocol. The results (EfficientNet-B0 at 94.54%) gave confidence that the diagnostic capability was solid enough to build a product around.

**The Superset Construction**
The six-dataset aggregation, 300–700 class balance rule, and augmentation pipeline. This was the most labour-intensive part of Phase 1: sourcing datasets, writing ingestion scripts, applying the balance filter, building the augmentation pipeline, and verifying the class distribution. The dataset is the single most important artifact from Phase 1 because it determines the model's real-world generalisability.

**RAG Pipeline Validation**
A preliminary evaluation of the RAG system's effectiveness at reducing hallucination. The evaluation compared:
- Baseline LLM (Llama-3.1-8B) answering agricultural questions without retrieval
- RAG-augmented LLM answering the same questions with retrieved KAU Package of Practices passages

The RAG system reduced factual errors significantly on the evaluated questions, validating the architectural choice.

**The Thanal VNIR Model**
Training and validation of the UNet+Attention model on competition data, producing the 28 dB PSNR / 0.85 SSIM benchmark. The competition validation confirmed that estimating NIR from RGB is feasible with a well-designed architecture.

### What Phase 1 Did Not Include

- No web application
- No user accounts or farm management
- No chat interface (only offline evaluation of the RAG pipeline)
- No deployment infrastructure
- No frontend
- The VNIR system was validated but not productised

---

## The Gap Between Phase 1 and Phase 2

Phase 1 proved that the core capabilities work. Phase 2 asked a harder question: can these capabilities be assembled into a system that a real farmer can use?

This required addressing several non-research challenges:

**Authentication and multi-user data isolation.** Multiple users sharing the same server must have completely isolated data. A bug that leaks one user's farm data to another is not just a technical failure — it's a trust failure.

**The farm data model.** Fields contain crops. Crops contain plants. Plants accumulate scan histories. This hierarchical model has to be stored, queried, and managed. The schema design and storage layer (see [technical/08_database_design.md](../technical/08_database_design.md)) was a significant new engineering effort.

**Real-time chat with farm context.** The Phase 1 RAG evaluation was offline: a fixed set of questions, a fixed retrieval pipeline, manual evaluation. Phase 2's chat interface required session management, real-time LLM calls, memory persistence, context assembly from the live farm database, and a UI that makes the RAG citations visible and navigable.

**Frontend engineering.** NAVA Phase 2 is a full-stack web application: React SPA, routing, authentication context, responsive layout, dark-mode design system (~80KB of CSS). None of this existed in Phase 1.

**Weather and geocoding.** The decision to include ambient weather context in every chat response required integration with Nominatim (geocoding) and Open-Meteo (weather), a DB persistence strategy for weather data, and background task management to keep weather fresh without blocking the API.

---

## Key Pivots and Design Decisions Made During Phase 2

### Pivot: SQLite per User, Not a Shared Database

The initial design used a single shared SQLite database for all farm data. Early testing revealed that concurrent access from multiple users in the same process created WAL contention and occasional lock errors.

The solution: per-user SQLite databases, keyed to a hash of the user's ID. Each user's data lives in a completely separate file. There is no shared state between users at the database layer. This solved the concurrency problem and added a security benefit: you can delete a user's entire data by deleting one file.

### Pivot: DB-Backed Weather Instead of Live API Calls in Chat

The first implementation of weather context fetched live from Open-Meteo on every chat request, using a `ThreadPoolExecutor` to avoid blocking the async API. This added 1–2 seconds of latency to every chat message and was fragile: if the geocoding service was slow, the chat request would time out.

The replacement architecture: weather is stored in the fields table, refreshed by background tasks at login and field creation. Chat requests read from the DB (a 1-millisecond operation), not from a remote API. Weather is fresh (updated at login) but not live-at-request-time. For a farm management tool where weather context is directional rather than minute-precise, this tradeoff is entirely acceptable.

### Pivot: Two-Level VNIR Alerts Instead of a Single Threshold

The first VNIR implementation used a single threshold: if the current checkpoint average dropped more than 15% relative to the previous checkpoint, trigger a WARNING. Testing revealed two problems:

1. The single comparison didn't distinguish between a *deteriorating trend* (current ratio is falling relative to recent scans) and a *significant departure from the healthy baseline* (current ratio has fallen far from the first five scans). These are qualitatively different situations.
2. The zero-ratio guard was absent. A scan where no leaf was detected (ratio = 0) would corrupt the baseline and trigger spurious warnings on every subsequent scan.

The two-level system (WARNING at 10% rolling drop, CRITICAL at 15% baseline drop) and the zero-ratio guard were added as a result of this evaluation.

### Pivot: Background Task for Geocoding on Field Creation

The initial design required explicit lat/lon entry from the user. This was immediately identified as an unrealistic UX requirement: most farmers don't know their coordinates, and the ones who do would find the entry friction frustrating.

The replacement: when a field is created with a location name ("Wayanad, Kerala"), a background task fires asynchronously, calling Nominatim to resolve the name to coordinates, then calling Open-Meteo to fetch the weather. The user sees their field data immediately; the weather data appears when the background task completes (typically within 2–3 seconds). This is the standard progressive loading pattern in modern web applications.

---

## Where NAVA Stands Now

Phase 2 delivers:
- Disease detection with Grad-CAM for 34 classes across 7 crops (94.54% accuracy)
- VNIR stress monitoring with two-level alerts and calibration-aware messaging
- Grounded conversational AI with persistent hierarchical memory
- Full farm management: fields, crops, plants, scan histories, notes, weather
- User authentication with per-user isolated data
- A full-stack web application deployable on a single server

---

## Future Directions

See [futureWork.md](../../futureWork.md) for the documented roadmap. The highlights:
- Multilingual native UI (Malayalam, Hindi) — currently LLM handles non-English queries but the UI is English-only
- Native iOS/Android app
- Expanded crop and disease coverage
- Satellite/drone imagery integration for field-level monitoring
- Ground-truth validation study for Thanal's NIR-from-RGB estimation
- Production infrastructure (Docker, object storage, migration framework)
