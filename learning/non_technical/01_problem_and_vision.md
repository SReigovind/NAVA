# The Problem NAVA Was Built to Solve

> **Subfolder:** `non_technical/`
> **Cross-references:** [02_research_foundation.md](02_research_foundation.md) | [04_design_philosophy.md](04_design_philosophy.md) | [05_market_and_differentiation.md](05_market_and_differentiation.md)

---

## The Global Crop Disease Problem

The world produces enough food to feed everyone alive today. And yet, between 20 and 40 percent of all harvested crops are lost to plant diseases, pests, and post-harvest deterioration before they reach a plate. That is not a supply problem — it is a knowledge problem. The diseases that cause those losses are mostly well understood. The treatments exist. The agricultural science has been done. What is missing is a reliable channel to get that knowledge to the farmer, at the field, in time to act.

A farmer who discovers dark necrotic spots on a rice leaf has, at most, a few days to intervene before an infection like rice blast spreads to neighbouring plants. In that window, the farmer needs to know: What disease is this? Is it fungal or bacterial? What product treats it? At what dose? Are there any resistance concerns? What should I do with the harvest if I apply fungicide now?

In most high-income countries, this information arrives through a smartphone app or a direct line to an extension officer. In most of the world, it doesn't arrive at all. The farmer guesses. Asks a neighbour. Applies what they used last season, right or wrong.

---

## The Kerala Context

NAVA was designed with a specific agricultural region in mind: Kerala, a state in southern India with a dense and economically significant smallholder farming sector.

Kerala's primary crops include:
- **Rice** (*Oryza sativa*) — the cultural staple, grown across three distinct seasons (Virippu, Mundakan, Puncha) in paddy fields
- **Banana** (*Musa spp.*) — particularly Nendran, the large cooking banana unique to Kerala, and Robusta for export; susceptible to Banana Fusarium Wilt (Panama Disease), Sigatoka, and Moko disease
- **Coconut** — ubiquitous in the landscape though not in NAVA's current classification scope
- **Vegetables** — tomato, chilli, and other crops grown on small plots

The typical Kerala smallholder farms 0.5 to 2 acres. They have smartphones. They have intermittent internet. They do not have a plant pathologist on speed dial.

Kerala's agricultural extension service — the Kerala Agricultural University (KAU) — produces excellent guidance material: the *Package of Practices* (PoP), crop-specific disease management guides, and research publications. But this material sits in PDFs in university libraries, not in the hands of farmers at the moment of need.

This is the precise gap NAVA targets: the distance between verified agricultural knowledge and the farmer who needs it.

---

## Why Existing Solutions Fall Short

Before building NAVA, it is worth understanding why existing tools were insufficient. The failure modes break into four categories.

### 1. Accuracy Under Real-World Conditions

Most plant disease AI models are trained on clean, controlled, lab-quality images: a single leaf against a white background, photographed under uniform lighting by a research assistant. These models can achieve very high accuracy on their test sets. They collapse the moment you photograph a leaf in a paddy field at noon with uneven lighting, partial occlusion, a muddy hand in the frame, and a camera that cost $80.

Real field photography is ugly. NAVA's training data addresses this directly (see [03_dataset_and_training_strategy.md](03_dataset_and_training_strategy.md)) by aggregating multiple real-world datasets, applying aggressive augmentation, and enforcing class balance rules that prevent any single controlled-environment dataset from dominating the training signal.

### 2. No Explanation

Even when a model produces a correct prediction, showing only "Disease X detected: 87% confidence" is not useful enough. A farmer needs to understand *why* the model reached that conclusion in order to decide whether to trust it. An unexplained AI output is an oracle: you either believe it or you don't. NAVA's Grad-CAM heatmaps (see [technical/03_gradcam_explainability.md](../technical/03_gradcam_explainability.md)) change this — they show exactly which region of the leaf the model attended to, turning a black-box result into a visual argument.

### 3. No Context — Generic, Not Personalised Advice

Apps that offer chat interfaces to an LLM give the same answer to every farmer: generic, internet-derived, often inconsistent advice about a disease. They do not know which farm the question comes from, what crops are growing there, what the plant's history looks like, or what the farmer tried last month.

NAVA knows all of this. Every chat response is constructed from:
- The field's metadata (location, soil type, area)
- The specific crop and plant being discussed
- The full scan history for that plant
- The farmer's notes from previous seasons
- The current weather at the field's location

The result is advice that sounds like a real agronomist who actually visited the farm, not a generic FAQ.

### 4. Hallucinated Advice

LLMs can produce fluent, confident, and completely wrong agronomic advice. Telling a farmer to apply the wrong fungicide can cause crop damage, environmental harm, and financial loss. Existing chat-based agricultural apps that run queries directly through an LLM with no knowledge grounding have no structural mechanism to prevent this.

NAVA's Retrieval-Augmented Generation (RAG) system addresses this at the architecture level: when a question requires factual knowledge about pest management, fertiliser dosage, or cultivation practice, NAVA retrieves the relevant passage from a verified document (Kerala Agricultural University's Package of Practices, peer-reviewed disease management guides) and injects it into the LLM's context before answering. The sources are shown to the farmer. The LLM cannot deviate from them without a visible contradiction.

---

## The Vision: A Digital Agronomist in Every Pocket

NAVA's vision is not to replace human agronomists. It is to make the knowledge of a good agronomist available to every farmer, at any time, without requiring the farmer to have the agronomist's phone number or the money to pay for a farm visit.

Specifically, NAVA aims to:

1. **Detect disease early** — before symptoms are severe enough that amateur identification is possible, let alone before a phone consultation can be arranged
2. **Monitor plant stress continuously** — tracking physiological change over time, not just point-in-time snapshots
3. **Ground advice in verified knowledge** — not the general internet, but specifically agricultural extension documents written for the crops and conditions in question
4. **Remember the farm** — accumulating a history of every scan, every treatment, every conversation, so the advice always improves over time
5. **Know the environment** — incorporating real weather data so recommendations factor in current field conditions

The system is designed to be deployable on a modest server (even a Raspberry Pi), to run all ML inference on CPU without a GPU, and to function in environments with limited but not zero internet connectivity.

---

## What Success Looks Like

A farmer using NAVA would:

1. Create an account and add their fields (location, size, soil type)
2. Add their crops and track individual plants across a growing season
3. When a leaf looks unusual, photograph it and upload it — receiving a disease diagnosis with a heatmap and a plain-language explanation in seconds
4. Check back weekly with VNIR scans to monitor stress before visual symptoms appear
5. Ask the chat assistant anything: treatment schedules, fertiliser rates, pest management, variety selection — and receive verified, source-attributed answers that the farmer can trace back to authoritative documents
6. Build a growing seasonal archive of their farm's health history, which informs every future conversation

This is not a future vision — every one of these capabilities is fully implemented and working in NAVA Phase 2.

---

## Further Reading

- The research foundations behind each capability → [02_research_foundation.md](02_research_foundation.md)
- How the training dataset was built to handle real-world conditions → [03_dataset_and_training_strategy.md](03_dataset_and_training_strategy.md)
- Why NAVA shows uncertainty rather than hiding it → [04_design_philosophy.md](04_design_philosophy.md)
- How NAVA compares to Plantix and other tools → [05_market_and_differentiation.md](05_market_and_differentiation.md)
