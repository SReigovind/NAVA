# Dataset Construction and Training Strategy

> **Subfolder:** `non_technical/`
> **Cross-references:** [02_research_foundation.md](02_research_foundation.md) | [06_model_comparison_study.md](06_model_comparison_study.md) | [technical/02_disease_detection_pipeline.md](../technical/02_disease_detection_pipeline.md)

---

## Why the Dataset is a Core Engineering Decision

In applied machine learning, the dataset is not a data concern — it is an engineering concern. A model is bounded by the quality and representativeness of its training data. No architectural cleverness can compensate for a dataset that does not represent the real-world distribution the model will encounter at inference time.

NAVA's training dataset — the Superset — was not simply downloaded from one place. It was consciously constructed from six sources, curated, balanced, and augmented with specific design goals in mind. Understanding those decisions explains much of NAVA's real-world accuracy and robustness.

---

## The Six Source Datasets

The Superset aggregates images from these publicly available repositories:

### 1. PlantVillage
The original large-scale plant disease dataset, compiled by Penn State researchers. Contains ~54,000 images of healthy and diseased leaves against uniform grey backgrounds under controlled lighting. Covers 26 diseases across 14 crops. High image quality, clean labels, well-documented. The founding dataset of the field.

**Limitation:** Nearly all images are lab-controlled. Leaf texture and colour are clean. There is no background clutter, no inconsistent lighting, no partial occlusion. Models trained only on PlantVillage will perform well on PlantVillage and poorly in a rice paddy.

### 2. PlantWild V1 and V2
A deliberate counterpoint to PlantVillage. Images collected in actual field conditions — on the plant, not as isolated leaves. Contains background foliage, inconsistent lighting, partial occlusion, and the kind of visual complexity that a farmer's smartphone produces. Dramatically increases the robustness of models trained on it.

**Why it matters:** PlantWild's inclusion is the primary driver of NAVA's ability to handle real-world uploads. Without it, the model would memorise PlantVillage's controlled appearance.

### 3. PlantDoc
A diverse, crowdsourced dataset specifically designed to address the generalization gap. Images come from multiple photographers in multiple environments. Useful for evaluating and improving cross-domain robustness.

### 4. PaddyDoctor
A rice-specific dataset from India, covering rice diseases under Indian field conditions. Critically relevant for NAVA's target context: rice cultivation in Kerala. Provides coverage of Blast, Sheath Blight, Brown Spot, and Tungro under conditions representative of South Asian paddy farming.

**Why it matters:** Without a South Asian rice dataset, NAVA's rice disease predictions would be calibrated to American or European rice cultivation photography. Lighting, soil colour, background foliage, and leaf presentation are all different.

### 5. ASDID (Annotated Soybean Disease Image Dataset)
High-quality soybean disease imagery. Expands crop coverage and provides additional class diversity.

### 6. Kaggle Competition Datasets
Several Kaggle competition datasets contributed images for crops not well-covered by the above — including cassava and banana — along with additional corn and tomato images.

---

## The 300–700 Class Balance Rule

After aggregating all six sources, the raw class distribution was severely imbalanced. Some disease classes (Tomato Late Blight, Corn Common Rust) had thousands of images from multiple sources. Others (Banana Fusarium Wilt, Cassava Mosaic) had fewer than 100.

Training on this imbalanced distribution produces a biased model: it learns that the majority classes are "more likely" and adjusts its predictions accordingly. A disease that appears rarely in the training data will be under-predicted, even when it is present.

The solution was a strict class filter with two rules:

**Rule 1: Minimum 300 images per class.**
Any class with fewer than 300 images was excluded entirely. The reason is not arbitrary — it is based on the practical observation that with fewer than 300 examples, any reasonable train/validation split produces a validation set too small to give reliable accuracy estimates. The model may be memorising the limited examples rather than learning general features. Exclusion is safer than inclusion of noise.

**Rule 2: Maximum 700 images per class.**
Any class exceeding 700 images was downsampled by random sampling to exactly 700. This prevents majority classes from contributing disproportionately large gradient updates during training, which would cause the model to optimise primarily for those classes at the expense of rarer ones.

The 300–700 range was selected empirically based on the distribution of available data. It is not a universal constant — it is a NAVA-specific decision calibrated to produce a balanced, trainable dataset from the available sources.

**Result:** 34 disease classes across 7 crops (Rice, Corn, Tomato, Soybean, Cassava, Banana, Cucumber), including a healthy class for each crop. Each class has between 300 and 700 images. Total: approximately 20,400 training/validation images.

---

## Augmentation Strategy

Even after balancing, 300–700 images per class is a relatively small dataset by deep learning standards. Augmentation artificially expands the training set by applying random transformations to each image, creating new "virtual" training examples. This forces the model to learn features that are invariant to the applied transformations.

NAVA used the **Albumentations** library for augmentation. The augmentation pipeline was designed to simulate the specific sources of variability in real-world field photography:

### Geometric Transforms
- **Horizontal and vertical flip** — a leaf photographed from the left versus from the right should produce the same diagnosis
- **Random rotation** (up to ±30°) — farmers don't hold phones perfectly level
- **Elastic distortion** — simulates the non-rigid deformation of leaves in wind or when the photograph is slightly blurred due to movement

### Lighting Variation
- **Random brightness and contrast adjustment** — field photography happens at different times of day with different sun angles
- **Random gamma** — corrects for different camera sensor responses

### Colour Variation
- **Random RGB channel shift** — different smartphone models have different colour science (white balance, sensor spectral response). A disease that looks yellow-green on one phone might look more orange-green on another. Channel shifts force the model to be robust to these differences.

### Blur
- **Gaussian blur** — simulates camera shake, out-of-focus shots, or low-resolution images from older phones

### What Was Not Augmented
- **Hue shifts beyond a small range** — extreme colour changes could make a diseased leaf look healthy or vice versa, and the model needs to learn colour as a feature
- **Cutout or random erasing** — removing parts of the leaf image could hide the very lesions the model needs to learn to detect

---

## The Test Set

4,089 test images were held out from augmentation and training entirely. These images were not used in any hyperparameter decision during training. They represent the gold standard for evaluating the model's generalisation.

**Final test set accuracy: 94.54%** across all 34 classes.

This figure was obtained on images that the model had never seen and that were not augmented. It is a conservative, honest estimate of real-world performance — not an inflated training-set accuracy.

---

## Why This Matters for a Practitioner

If you were building a similar system and skipped the dataset engineering:

- Downloading only PlantVillage and training on it would likely produce 92–99% accuracy on PlantVillage and 50–70% accuracy on real field photos
- Skipping class balancing would produce a model that is good at diagnosing common diseases and poor at diagnosing rare ones — the exact opposite of what a farmer needs (rare diseases are precisely those where a second opinion is most valuable)
- Skipping augmentation would produce a model that is brittle to lighting changes, phone orientation, and camera models — all of which vary constantly in real-world use

Each of these decisions — source selection, class balancing, augmentation design — is a direct response to a documented failure mode in the plant disease AI literature.

---

## The Superset in Numbers

| Metric | Value |
|--------|-------|
| Source datasets | 6 |
| Disease classes | 34 |
| Crops covered | 7 (Rice, Corn, Tomato, Soybean, Cassava, Banana, Cucumber) |
| Minimum images per class | 300 |
| Maximum images per class | 700 |
| Total training/validation images | ~20,400 |
| Total test images | 4,089 |
| Test set accuracy | 94.54% |
