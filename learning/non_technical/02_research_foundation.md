# Research Foundations

> **Subfolder:** `non_technical/`
> **Cross-references:** [01_problem_and_vision.md](01_problem_and_vision.md) | [06_model_comparison_study.md](06_model_comparison_study.md) | [technical/02_disease_detection_pipeline.md](../technical/02_disease_detection_pipeline.md) | [technical/04_vnir_stress_monitoring.md](../technical/04_vnir_stress_monitoring.md) | [technical/05_rag_and_knowledge_grounding.md](../technical/05_rag_and_knowledge_grounding.md)

---

## Why a Research Foundation Matters

NAVA is not a collection of interesting demos glued together. Every component rests on a body of published research that established what is possible, what works, and crucially — what doesn't. Understanding the research context explains not just what NAVA does but *why it was built this way* and *which choices were forced by the state of the science*.

This document surveys the research basis for each of NAVA's four main capabilities.

---

## 1. Plant Disease Detection from Images

### The Core Idea

The idea of identifying plant diseases from photographs predates deep learning. Agronomists and plant pathologists have long used visual symptoms — the shape, colour, and distribution of lesions — as primary diagnostic criteria. The question was whether a machine could learn those visual patterns automatically, from data, without hand-coded rules.

The seminal paper that established deep learning as viable for this task was **"Using Deep Learning for Image-Based Plant Disease Detection"** (Mohanty, Hughes, Salathé, 2016, *Frontiers in Plant Science*). The paper trained a deep convolutional neural network on the PlantVillage dataset (54,000 images, 26 diseases across 14 crops) and demonstrated 99.35% accuracy. This result galvanised the field.

### What the Mohanty et al. Paper Got Wrong

The 99.35% accuracy figure was obtained on a test set drawn from the same controlled-photography conditions as the training data: single leaves against uniform backgrounds under controlled lighting. The model did not generalise to real field photographs. Subsequent studies (Barbedo, 2018; Ramcharan et al., 2019) documented accuracy drops of 30–60 percentage points when the same models were applied to images taken in the field.

This is the generalisation gap problem, and it is the primary reason NAVA's dataset strategy (see [03_dataset_and_training_strategy.md](03_dataset_and_training_strategy.md)) explicitly prioritises field-condition datasets and augmentation strategies.

### Transfer Learning

Training a convolutional neural network from scratch for plant disease detection is impractical. You would need millions of high-quality labelled images and weeks of GPU compute. Transfer learning solves this.

The key insight: a model pre-trained on ImageNet (1.4 million natural images, 1,000 classes) has learned general visual features in its early layers — edges, textures, colours, shapes — that are directly relevant to plant disease detection. The final layers of the network, which encode high-level semantic categories, can be replaced and retrained on the smaller domain-specific dataset while keeping the pre-trained visual features frozen or fine-tuned.

This technique reduces the amount of domain data needed by an order of magnitude and dramatically accelerates training. All three architectures tested in NAVA's comparison study (ResNet-50, MobileNetV2, EfficientNet-B0) use ImageNet pre-training.

### EfficientNet and Compound Scaling

In 2019, Tan and Le at Google Brain published **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"**. The central observation was that existing approaches to improving CNN accuracy (making the network deeper, wider, or training on higher-resolution images) were done independently and somewhat arbitrarily. EfficientNet proposed *compound scaling*: scaling all three dimensions simultaneously in a principled ratio derived by neural architecture search.

The result was a family of models (B0 through B7) that achieved better accuracy than larger predecessors (ResNet, VGG, Inception) at significantly lower computational cost. EfficientNet-B0, the smallest member, has 5.3M parameters and achieves 77.1% top-1 accuracy on ImageNet — competitive with ResNet-50 (25.6M parameters, 76.1%). For a deployment target that must run on CPU without a GPU, this efficiency-accuracy tradeoff is exactly what NAVA needs.

### Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) was introduced by Selvaraju et al. in **"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization"** (2017, ICCV). The idea: instead of asking which input pixels the network used, ask which features in the final convolutional layer were most important for the predicted class, by backpropagating the class score through the network and examining the gradient magnitudes at the target layer.

This produces a coarse spatial map (the resolution of the final convolutional layer's feature maps, not the full input resolution) that, when upsampled and overlaid on the input image, highlights the regions the model attended to. For a disease classifier, those highlighted regions should correspond to the visible lesion — and if they don't, that's a signal that the model's reasoning is suspect.

---

## 2. VNIR Stress Monitoring from RGB Images

### The Biology: Why NIR Matters

Near-infrared (NIR) reflectance is a well-established proxy for plant physiological health, used commercially in satellite-based NDVI (Normalised Difference Vegetation Index) measurements, drones, and hyperspectral cameras. The biology is straightforward:

Healthy plant cells have a tightly organised mesophyll (the spongy middle layer of a leaf). This structure reflects strongly in the near-infrared band (approximately 700–1300 nm wavelength) due to internal cell wall reflections. When a plant is under stress — nutrient deficiency, water stress, early-stage pathogen infection — the mesophyll structure degrades before visible symptoms appear. The leaf becomes less internally reflective in NIR, even while still appearing normal to the naked eye.

This means NIR reflectance can detect physiological stress days to weeks before visible symptoms form. A farmer who acts on that early signal can intervene before the disease becomes visible — and visible disease means the infection has already progressed significantly.

### The Problem: NIR Cameras Are Expensive

Dedicated multispectral cameras that measure NIR reflectance start at several hundred dollars for basic models and run into the thousands for research-grade instruments. A smallholder farmer in Kerala is not buying one.

Standard smartphone cameras capture only the visible spectrum (Red, Green, Blue). They do not capture NIR — their sensors are physically filtered to exclude it.

### The Thanal Approach: Estimating NIR from RGB

The central research question underlying NAVA's VNIR module is: *can a deep learning model learn to estimate NIR reflectance from an RGB image, with sufficient accuracy to be useful for early stress detection?*

The answer, from the competition on which Thanal was trained, is yes — with caveats. The model learns to use the spectral and textural patterns in the visible RGB channels that correlate with NIR reflectance in healthy versus stressed leaf tissue. It is not measuring NIR directly; it is making an informed estimate based on what it has learned about the relationship between visible-spectrum appearance and NIR reflectance.

The architecture used is a **UNet with Attention Gates**. UNet (Ronneberger et al., 2015) is a pixel-to-pixel encoder-decoder architecture originally developed for medical image segmentation. It preserves spatial resolution through skip connections between encoder and decoder layers. Attention gates (Oktay et al., 2018) allow the decoder to selectively focus on relevant spatial regions when upsampling, suppressing background noise.

This architecture is appropriate because NIR estimation is fundamentally a pixel-wise regression task: for every pixel in the input RGB image, predict the NIR intensity value. UNet's structure handles this well.

### The Ratio, Not the Absolute Value

NAVA does not use the absolute NIR estimate as a health indicator. Instead, it computes the ratio of mean estimated NIR intensity to mean green channel intensity over the leaf region. This ratio is more meaningful than either value alone because:

1. It is partially normalised against lighting variation (both numerator and denominator scale with illumination)
2. It captures the relative balance between leaf photosynthetic activity (green channel proxy) and cellular structure (NIR proxy)
3. It provides a basis for the rolling comparison strategy — tracking whether the ratio is declining relative to the plant's own history, rather than requiring a fixed absolute threshold

---

## 3. Retrieval-Augmented Generation (RAG)

### The Hallucination Problem

Large language models (LLMs) like Llama-3 are trained on vast quantities of internet text. They are extraordinarily good at producing fluent, grammatically correct, contextually appropriate text. They are not good at reliably producing *factually correct* text, especially for narrow, specialised domains where their training data may be sparse, conflicting, or out of date.

For agriculture, this is a serious problem. An LLM asked "what is the correct dosage of Carbendazim fungicide for banana Sigatoka management?" might produce a plausible-sounding answer that is wrong by 50%. A farmer who applies half the recommended dose may not treat the disease. A farmer who applies double the dose may damage the crop or contaminate the soil.

### The RAG Solution

Retrieval-Augmented Generation was introduced formally by Lewis et al. at Meta AI (2020, *Advances in Neural Information Processing Systems*). The core idea: when an LLM needs to answer a factual question, first *retrieve* relevant passages from a curated knowledge base, then *generate* the answer using those passages as grounding material injected into the context.

This approach has several structural advantages:
- The knowledge base can be updated independently of the model (no retraining required)
- The retrieved passages can be shown to the user as citations (source attribution)
- The LLM is explicitly instructed to use the retrieved material as authoritative — it cannot easily contradict a passage that is right there in its context window
- Factual errors become auditable: if the answer is wrong, you can check whether the retrieved passage was wrong or whether the model deviated from it

### Embedding Models and Vector Search

For retrieval to work, the knowledge base must be searchable by semantic meaning, not just keyword. This requires embedding the text passages into dense vector representations, where similar meanings map to nearby vectors in a high-dimensional space.

NAVA uses the `BAAI/bge-small-en-v1.5` model for embedding — a 33M-parameter bi-encoder trained specifically for semantic retrieval. It produces 384-dimensional vectors. ChromaDB provides the persistent vector store and approximate nearest-neighbour search.

---

## 4. Conversational Memory

### Why Memory Matters for Agricultural AI

A farmer's conversation with NAVA is not a one-time transaction. It unfolds over months. Questions build on previous answers. Treatments are tried, evaluated, and adjusted. A new symptom that appears in October only makes sense in the context of the diagnosis made in August.

Without memory, every conversation starts fresh. The LLM knows nothing about the farm, the previous discussions, or the treatments already tried. It cannot give advice that improves over time.

### The Context Window Limit

All LLMs have a fixed context window — the maximum number of tokens they can attend to at once. For a chat interface, this means there is a finite amount of conversation history that can be included in each new request. Once the conversation grows beyond that window, old messages must be discarded.

The hierarchical summarisation approach NAVA uses — compressing old messages into rolling summaries (L1: recent exchanges; L2: long-term rollup) — is a practical engineering solution to this constraint. It ensures that the LLM always has both the broad arc of the conversation history and the recent specific detail, without overflowing the context window. See [technical/07_hierarchical_memory.md](../technical/07_hierarchical_memory.md) for the full treatment.

---

## Summary

| NAVA Capability | Research Basis |
|----------------|----------------|
| Disease detection | Transfer learning from ImageNet; CNN-based plant pathology literature (Mohanty et al., 2016) |
| Compound scaling model | EfficientNet (Tan & Le, 2019) |
| Explainability | Grad-CAM (Selvaraju et al., 2017) |
| NIR from RGB | UNet+Attention pixel-to-pixel regression; plant NIR reflectance biology |
| Knowledge grounding | RAG (Lewis et al., 2020); bi-encoder embedding models |
| Conversation memory | Hierarchical summarisation; LLM context window management |
