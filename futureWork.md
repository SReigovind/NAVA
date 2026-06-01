# NAVA — Future Work

> Ideas and extensions that are outside the current MSc project scope but represent natural next steps for production deployment or continued research.

---

## 1. Multilingual Interface & Voice Support

NAVA currently operates entirely in English. Kerala's farming community predominantly communicates in Malayalam. The natural next step is a Malayalam-first UI with:

- **UI translation** — all labels, button text, and status messages localised to Malayalam
- **Malayalam chat** — `Mozhi`'s prompt chain adjusted to request Malayalam-language responses from the LLM
- **Voice input/output** — Web Speech API for dictation and TTS readback, making NAVA accessible to users with low literacy
- **Transliteration** — supporting both Malayalam script and Manglish (romanised Malayalam) input

---

## 2. Native Mobile Application

NAVA runs as a web app. A native iOS/Android wrapper (React Native or Flutter) would unlock:

- **Offline disease scanning** — ship a lightweight quantised version of EfficientNet-B0 (Core ML / TFLite) that runs without internet
- **Camera integration** — direct native camera access with guided framing for leaf capture (bounding box overlay)
- **Push notifications** — alert farmers when a stress warning crosses CRITICAL threshold
- **Background sync** — queue scans offline, sync to the server when connectivity returns

---

## 3. Expanded Crop & Disease Coverage

The current model covers **34 disease classes across 7 crops**. Production scale would require:

- Additional crops: pepper, coconut, cardamom, ginger, arecanut (all important Kerala crops not currently covered)
- Expanded disease classes per crop: current banana coverage has 3 diseases; PlantVillage lists 12+ banana conditions
- Continuous retraining pipeline: auto-ingestion of new labelled samples from field scans flagged as low-confidence
- Multi-label classification: many plants exhibit more than one disease simultaneously; the current single-label head cannot represent this


---

## 4. Multi-User Farm Collaboration

NAVA is currently single-user (one account = one farm). Multi-user support would enable:

- **Field sharing** — a farm owner can invite a hired worker or extension officer to view (read-only) or contribute scans to their fields
- **Extension officer dashboard** — a supervisor view showing aggregated health status across multiple registered farmers' fields in a geographic cluster
- **Role-based access** — Owner / Manager / Observer permission levels per field

---

## 5. Automated Field Health Reports

Generating a weekly or seasonal PDF report summarising:

- All disease detections (dates, classes, confidence, affected plants)
- VNIR stress trend per plant (chart)
- NAVA chat-extracted auto-notes (actions taken by the farmer)
- Weather summary (temperature, precipitation over the period)

This report could be submitted to crop insurance schemes or bank loan officers as evidence of good farm management practice.

---

## 6. Crop Insurance & Market Price Integration

- **Insurance claims** — link a disease detection event to a pre-registered insurance policy; auto-populate a claim form with the scan evidence
- **Market price feed** — display current mandated price (MSP) and local mandi price for the detected crop at harvest stage
- **Yield prediction** — given current stage, stress history, and weather, estimate expected yield using a regression model trained on historical yield + stress data

---

## 7. Real VNIR Ground Truth Validation

Thanal (the VNIR estimator) was trained and validated on competition data. A rigorous field validation study would:

- Co-locate smartphone RGB captures with a handheld NIR spectrometer reading for the same leaf
- Measure actual correlation between Thanal's estimated NIR/Green ratio and the ground-truth spectrometer NIR/Green ratio
- Establish per-crop calibration offsets for the warning thresholds (current thresholds are empirically derived from the competition validation set)

---

## 8. Production Infrastructure

For deployment beyond the MSc demo:

- **Container image** — Dockerfile and docker-compose for reproducible deployment
- **Object storage** — replace local `logs/` directory with S3-compatible storage (MinIO or AWS S3) for model files, ChromaDB snapshots, and Grad-CAM image cache
- **Database migration framework** — replace the manual `PRAGMA table_info` approach with Alembic for tracked, reversible schema migrations
- **Monitoring** — Prometheus metrics endpoint + Grafana dashboard for request latency, model inference time, and LLM API error rate
- **Rate limiting** — per-user rate limits on LLM endpoints to control Hugging Face API costs at scale
