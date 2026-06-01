# NAVA — Next-gen Agricultural Virtual Assistant
## Final Project Report

### Abstract

Crop diseases and delayed agronomic interventions cause significant reductions in agricultural yields, frequently affecting smallholder farmers who lack consistent access to expert guidance. Existing agricultural AI solutions often face challenges in real-world conditions due to reactive diagnosis and the tendency of Large Language Models (LLMs) to generate inaccurate chemical dosages or hallucinate advice. This project presents NAVA (Next-gen Agricultural Virtual Assistant), an end-to-end digital agronomist platform developed to provide localized, reliable agricultural assistance directly through a mobile-friendly web interface.

The proposed system employs a dual-model vision architecture. It utilizes an EfficientNet-B0 classifier for disease identification across multiple crops, achieving a 94.54% validation accuracy. This is paired with Thanal, a virtual near-infrared (VNIR) estimation model that proactively identifies physiological plant stress from standard smartphone RGB images by evaluating scans against a rolling baseline. To provide actionable and safe treatment recommendations, NAVA integrates a conversational agent powered by Llama-3 with a hybrid Retrieval-Augmented Generation (RAG) pipeline. The RAG system grounds the LLM’s responses by dynamically routing queries, extracting key agronomic terms, and retrieving relevant guidelines from a verified database of institutional agricultural documents.

Additionally, the platform incorporates Grad-CAM heatmaps for visual explainability, localized weather data injection, and a multi-level conversational memory that automatically tracks the farmer's actions. Experimental evaluations demonstrate that this integrated approach improves factual reliability compared to baseline parametric models, offering a practical, context-aware tool for proactive crop management.

---

### Table of Contents

**1. Introduction**
*   1.1 Motivation and Background
*   1.2 Problem Statement
*   1.3 Project Objectives
*   1.4 Scope of the Project

**2. Literature Review**
*   2.1 *(Subsections to be populated by authors)*

**3. Methodology**
*   3.1 NAVA System Architecture Overview
*   3.2 Data Collection and Preprocessing
    *   3.2.1 Disease Detection
    *   3.2.2 NIR Prediction
*   3.3 Gathi: API Server and Frontend Integration
    *   3.3.1 FastAPI Backend Framework
    *   3.3.2 React Single Page Application (SPA)
*   3.4 Mizhi: Vision Pipeline and Proactive Monitoring
    *   3.4.1 Disease Classification with EfficientNet-B0
    *   3.4.2 Explainability Layer using Grad-CAM
    *   3.4.3 VNIR Estimation Engine (Thanal)
    *   3.4.4 Rolling Baseline Stress Detection Logic
*   3.5 Mozhi: Conversational AI and Memory Management
    *   3.5.1 Llama-3 Chat Integration
    *   3.5.2 Multi-Level Memory Hierarchy
    *   3.5.3 Dynamic Context Injection (Farm, Crop, Weather)
    *   3.5.4 Automated Agronomic Note Extraction
*   3.6 Yukthi: Knowledge Retrieval Pipeline
    *   3.6.1 Document Ingestion and Chunking
    *   3.6.2 Query Routing Classifier
    *   3.6.3 Agronomic Keyword Extraction
    *   3.6.4 Hybrid Retrieval Strategy (Semantic and Keyword)
*   3.7 Shared Foundation Layer
    *   3.7.1 Localized SQLite Storage (Field and Session Stores)
    *   3.7.2 Geo-Coordinate Resolution
    *   3.7.3 Open-Meteo Weather Integration

**4. Experimental Results and Analysis**
*   4.1 Introduction
*   4.2 Testing Methodology and Setup
    *   4.2.1 Qualitative Testing Framework
    *   4.2.2 Evaluation Criteria
*   4.3 Quantitative Analysis
    *   4.3.1 Vision Model Training, Comparison and Validation
    *   4.3.2 VNIR Model Performance Metrics (PSNR and SSIM)
*   4.4 Qualitative Analysis
    *   4.4.1 Chat Context and Memory Evaluation
    *   4.4.2 RAG System Evaluation (With vs. Without RAG)
    *   4.4.2 Grad-CAM Heatmap Analysis
    *   4.4.4 VNIR Monitoring Analysis
*   4.5 Summary of Findings

**5. Future Scope**

**6. Conclusion**

**7. References**

**8. Appendix**
