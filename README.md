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


---

### Module 1 — Mizhi: Disease Detection & VNIR Monitoring

### Module 2 — Mozhi: Multilingual Chatbot & Contextual Memory

### Module 3 — Yukthi: RAG Pipeline & Explainable AI

### Module 4 — Gathi: Mobile App & Backend Orchestration

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