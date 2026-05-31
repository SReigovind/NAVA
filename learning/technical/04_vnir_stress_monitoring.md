# VNIR Stress Monitoring

> **Subfolder:** `technical/`
> **Cross-references:** [non_technical/02_research_foundation.md](../non_technical/02_research_foundation.md) | [02_disease_detection_pipeline.md](02_disease_detection_pipeline.md) | [code/07_mizhi_vnir.md](../code/07_mizhi_vnir.md)

---

## The Biology: Why NIR Is a Leading Indicator of Plant Stress

When a plant is under physiological stress — nutrient deficiency, water stress, or early-stage pathogenic infection — its internal cellular structure changes before any visible symptoms appear. Specifically:

The **mesophyll layer** (the spongy middle layer of a leaf) is densely packed with cells in a healthy plant. This packing creates multiple air-cell interfaces inside the leaf that act as mirrors for near-infrared radiation (approximately 700–1300 nm), reflecting it strongly outward. When the plant is stressed, cells lose turgor, membranes degrade, and intercellular spaces change — reducing the internal reflection. The leaf becomes less reflective in NIR.

This NIR reflectance decrease happens at the cellular level, days to weeks before the plant shows visible symptoms like chlorosis (yellowing) or necrosis (brown lesions). A sensor that can detect NIR reflectance can therefore detect stress before any visual examination could.

This is the biological basis for NDVI (Normalised Difference Vegetation Index), used commercially in satellite remote sensing. NAVA brings the same principle to consumer smartphone cameras.

---

## The Technical Challenge: Smartphones Don't See NIR

Standard smartphone cameras capture only the visible spectrum (Red ≈ 400–700 nm, Green ≈ 500–600 nm, Blue ≈ 400–500 nm). Dedicated camera sensors that can detect NIR cost hundreds to thousands of dollars and are not owned by smallholder farmers.

Thanal's insight: **can a neural network learn to estimate the NIR reflectance of a leaf from the visible RGB channels alone?**

The answer is a qualified yes. The visible and NIR channels of leaf images are not independent — they share underlying information about the leaf's physiological state. The model learns the statistical relationship between visible-channel appearance and NIR reflectance from paired RGB-NIR training examples. At inference, given only the RGB image, it predicts what the NIR channel would look like.

This is not a measurement of actual NIR — it is a learned approximation. The validation metrics (28 dB PSNR, 0.85 SSIM) quantify how accurately the model recovers the ground-truth NIR values. 28 dB PSNR corresponds to visually indistinguishable images in many imaging contexts; 0.85 SSIM indicates high structural similarity.

---

## Stage 1: HSV Leaf Isolation

Before the VNIR model can estimate NIR reflectance, it needs to process only the leaf tissue — not soil, hands, pots, or background objects. Background pixels would contribute noise to the average NIR estimate.

The isolation uses **HSV (Hue-Saturation-Value) colour space** rather than RGB. HSV is more robust for plant segmentation because:
- Hue separates chromatic information from intensity — a green leaf in shadow and a green leaf in sunlight have different RGB values but similar hue values
- Saturation filters out near-grey pixels (background objects often have low saturation)
- Value can be used to exclude very dark (shadow) or very bright (glare) pixels

Two masks are applied:

**Green mask** (for healthy leaf tissue):
- Hue in the range 30–90° (the green portion of the HSV wheel)
- Saturation ≥ 40 (filters out grey/white backgrounds)
- Value ≥ 40 (filters out shadows)

**Yellow-brown mask** (for stressed/dying tissue):
- Hue in the range 15–30° (yellow-orange, typical of chlorotic tissue)
- Saturation ≥ 40

Each mask is cleaned up with morphological operations (close followed by open, using elliptical kernels). Closing fills small holes inside the mask (areas temporarily lost due to lighting variation). Opening removes small noise regions (isolated pixels that passed the colour filter but are not leaf tissue).

**State determination from contour analysis:**
- If the largest green contour area ≥ the largest yellow-brown contour AND ≥ 5% of the frame: `leaf_state = "GREEN"` → healthy enough for VNIR estimation
- If the yellow-brown contour dominates: `leaf_state = "YELLOW_BROWN"` → immediate `CRITICAL: Visual Stress` without running the model
- If neither meets the 5% threshold: `leaf_state = "NONE"` → `"No Leaf Detected"`

The 5% threshold prevents the system from treating a tiny patch of leaf visible at the corner of the frame as sufficient for VNIR estimation.

**Why dual masks instead of a single green mask?**
A single green mask would fail to detect or characterise yellowing leaves. By tracking both green tissue and yellow-brown tissue, the system can make an immediate visual diagnosis for advanced stress cases without even running the computationally expensive ONNX model. This shortcircuits the pipeline for the cases where stress is already visually obvious.

---

## Stage 2: ONNX Inference

For `GREEN` leaf states, the masked RGB image (leaf pixels visible, background zeroed) is passed to the `VNIREngine`.

**Why ONNX instead of PyTorch?**
The Thanal model was trained in PyTorch but deployed via ONNX Runtime. ONNX (Open Neural Network Exchange) is a standardised model format that separates the model's computational graph from the training framework. ONNX Runtime then provides optimised CPU inference without requiring a full PyTorch installation.

Benefits for NAVA:
- Smaller deployment footprint (no PyTorch required on the inference server)
- ONNX Runtime is validated on Raspberry Pi 4 — confirming edge deployment viability
- Potential future optimisations (quantisation, operator fusion) without retraining

**The preprocessing:**
1. Resize to 256×256 (the model's expected input size)
2. Convert to float32, normalise to [0, 1] (divide by 255)
3. Transpose from HWC (Height×Width×Channels) to NCHW (Batch×Channels×Height×Width) — ONNX Runtime expects NCHW format
4. Add batch dimension (unsqueeze)

**The output:**
The model returns a (1, 1, H, W) tensor — a single-channel grayscale prediction. The output is extracted as `outputs[0][0, 0]` — a (H, W) numpy array where each value represents the estimated NIR intensity at that pixel.

---

## Stage 3: Ratio Computation and Two-Level Alert System

The raw NIR estimate is a spatial image. The `VNIRAnalyzer` converts it into a single scalar metric: the NIR/Green ratio.

**Ratio computation:**
For each pixel within the leaf mask (green or yellow-brown tissue area), two values are extracted:
- `avg_g` — mean green channel intensity (from the original RGB, not the masked version) over the leaf mask
- `avg_vnir` — mean estimated NIR intensity over the same mask

`ratio = avg_vnir / avg_g`

The ratio is more meaningful than either value alone because it is partially normalised against illumination variation. In brighter light, both the NIR estimate and the green channel will be higher — the ratio remains more stable.

**The zero-ratio guard:**
If either `avg_g` or the leaf mask area is zero (no leaf detected, or a completely masked image), `ratio = 0.0`. This value is stored in the history record but **excluded from all statistical calculations** — baseline building, rolling window, comparison. Without this guard, a single failed scan (no leaf in frame) would set the baseline to near-zero and trigger CRITICAL on every subsequent healthy scan.

---

## The Two-Level Alert System

Rather than comparing against a fixed absolute threshold (which would require per-crop calibration data that doesn't exist for this application), NAVA uses a **relative comparison against the plant's own history**.

The history is a timeseries of ratio values — one per scan, chronological. Valid ratios (ratio > 0) are used for all statistics.

**Level 1: WARNING — Rolling window comparison**
Compares the current ratio against the mean of the last 5 valid scans (the rolling mean). If the current ratio is ≥10% below the rolling mean: `WARNING: Stress detected`.

*What it signals:* The plant's ratio has been declining in recent scans. The trend is downward. This is an early notice — the decline may not yet have crossed the baseline threshold, but something is changing.

**Level 2: CRITICAL — Baseline comparison**
Compares the current ratio against the baseline mean (the mean of the first 5 valid scans, assumed to be healthy). If the current ratio is ≥15% below the baseline mean: `CRITICAL: Significant stress vs. baseline`.

*What it signals:* The plant has significantly departed from its own healthy starting point. This is not just a recent fluctuation — the ratio has moved far from the reference established when the plant was first monitored.

**Priority:** If both thresholds are breached simultaneously, CRITICAL takes precedence. A CRITICAL status is more actionable than a WARNING, so it should always be surfaced when present.

**Why different thresholds (10% vs 15%)?**
The rolling mean is computed over recent scans and is itself moving. A 10% drop from a recent mean can happen due to measurement variation alone (a slightly different photo angle, lighting change). A 15% drop from the baseline is harder to explain away as noise — the baseline is stable (5 scans taken early in the monitoring period).

**Calibration phase:**
Until 5 valid scans have been recorded, the system outputs `"Calibrating: N scans remaining"`. This prevents false alerts before the baseline is established. A new plant should not trigger CRITICAL on its second scan.

---

## Why Not Absolute Thresholds?

An absolute threshold (e.g., "ratio < 0.6 = stressed") would require:
- Per-crop calibration data (a healthy banana ratio is different from a healthy tomato ratio)
- Per-environment calibration (a plant in full sun has a different absolute ratio than one in partial shade)
- Per-camera calibration (different smartphone camera spectral responses produce different RGB values)

A relative threshold sidesteps all of these requirements. It says: "I don't know what an absolutely healthy ratio looks like for your plant in your conditions — but I know that a 15% drop from *your plant's own healthy baseline* is significant."

This is the same principle used in clinical medicine for vital signs: a heart rate of 90 bpm is normal for one patient and elevated for another. What matters is *your baseline* and how far you've deviated from it.
