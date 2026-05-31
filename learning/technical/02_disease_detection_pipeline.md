# Disease Detection Pipeline

> **Subfolder:** `technical/`
> **Cross-references:** [non_technical/02_research_foundation.md](../non_technical/02_research_foundation.md) | [non_technical/06_model_comparison_study.md](../non_technical/06_model_comparison_study.md) | [03_gradcam_explainability.md](03_gradcam_explainability.md) | [code/06_mizhi_detection.md](../code/06_mizhi_detection.md)

---

## What the Pipeline Must Do

A farmer uploads a leaf photograph. Within 1–2 seconds, they should receive:
1. A disease class name (or "Healthy")
2. A confidence level expressed in plain language
3. A Grad-CAM heatmap showing which region of the leaf the model attended to
4. A reliability verdict (RELIABLE / UNRELIABLE)

That's the user-facing output. Behind it is a carefully designed pipeline with multiple stages.

---

## Stage 1: Preprocessing

The uploaded image (a PIL Image, any size, any orientation) is transformed into a tensor that matches the ImageNet preprocessing used during EfficientNet-B0's pre-training. This is critical: the model's learned features are calibrated to a specific input distribution. Feeding it non-normalised inputs produces undefined behaviour.

**The preprocessing steps:**

1. **Resize to 256px on the shorter side** — ensures the subsequent center crop captures a consistent proportion of the image
2. **Center crop to 224×224** — the standard ImageNet input resolution for EfficientNet-B0
3. **Convert to tensor** — transforms the PIL image from 0–255 uint8 to 0.0–1.0 float32
4. **Normalise with ImageNet mean and std** — subtracts `[0.485, 0.456, 0.406]` from each channel and divides by `[0.229, 0.224, 0.225]`. These values are the channel-wise mean and standard deviation of the ImageNet training set. The normalisation brings the input distribution in line with what the model expects.

The preprocessor also retains the 224×224 cropped image (before normalisation) as the base for the Grad-CAM overlay.

**Why 256 then crop vs. direct 224 resize?** Resizing directly to 224 distorts the aspect ratio for non-square images and can stretch leaf shapes. Resizing the shorter side to 256 and then center-cropping to 224 preserves the aspect ratio for most leaf images and consistently centres the main leaf subject.

---

## Stage 2: Fast Inference (`predict()`)

The preprocessed tensor passes through the full EfficientNet-B0 network in a `torch.no_grad()` context, producing a 34-dimensional logit vector (one logit per disease class). Softmax converts the logits to probabilities (0 to 1, summing to 1 across all classes). `torch.max()` extracts the highest probability and its class index.

**Why `torch.no_grad()`?** During inference, PyTorch does not need to compute or store gradients. The forward pass is faster (no gradient tape overhead) and uses less memory. The `no_grad()` context is standard practice for any inference-only forward pass.

The confidence (the maximum softmax probability) is compared against the configured threshold (default 0.80). If it falls below the threshold, the result is tagged `UNRELIABLE` and the pipeline terminates here. No Grad-CAM is computed.

**Why terminate early for UNRELIABLE?** Computing Grad-CAM for an unreliable prediction would produce a heatmap that is equally unreliable — the gradient activations for a low-confidence prediction are noisy. Showing a noisy heatmap implies a confidence in the visual explanation that isn't warranted. The UNRELIABLE label is more informative and more honest.

---

## Stage 3: Grad-CAM Computation (`predict_with_cam()`)

For RELIABLE predictions, the pipeline runs a second, gradient-enabled forward pass. Note that this is not double inference — the fast `predict()` call is only used to determine whether the prediction is reliable. If reliable, `predict_with_cam()` runs the full forward pass including Grad-CAM in one step. See [03_gradcam_explainability.md](03_gradcam_explainability.md) for the complete Grad-CAM technical treatment.

**Why not always run Grad-CAM?** Grad-CAM requires keeping the full computation graph in memory for backpropagation. This uses more memory and is slower than the `no_grad()` forward pass. For unreliable predictions (which happen more frequently for unusual images or disease categories near the boundary), the extra cost is unnecessary.

---

## Stage 4: Result Persistence

The prediction is written to the user's FieldStore as an event record:

```python
field_store.add_event(
    field_id=field_id,
    crop_id=crop_id,
    plant_id=plant_id,
    event_type="diagnose",
    payload={
        "class_label": result.class_label,
        "confidence": result.confidence,
        "reliability": result.reliability,
        "original_image": base64_original,
        "gradcam_image": base64_cam or None,
    }
)
```

The images are stored as base64-encoded strings in the JSON payload. This trades storage efficiency for query simplicity — no separate image file system management is needed.

After adding the event, `_refresh_field_context()` regenerates the `shared_context` text for the field. This ensures that the next chat request about this field/crop will see the updated scan history in its context.

---

## The Confidence Gate in Detail

The 0.80 threshold is configurable via the `NAVA_CONFIDENCE_THRESHOLD` environment variable. The default of 0.80 was chosen to balance two failure modes:

**Setting the threshold too low (e.g., 0.60):**
The system presents more confident diagnoses. This increases the number of cases where a wrong diagnosis is presented as reliable. A farmer who receives "Late Blight — RELIABLE" when the actual disease is Early Blight may apply the wrong treatment.

**Setting the threshold too high (e.g., 0.95):**
The system flags most predictions as UNRELIABLE. This is safe but useless — if the system always says "consult an expert," farmers will stop using it.

**0.80 as the default:**
At 94.54% test accuracy, the majority of predictions are RELIABLE. The 0.80 threshold acts as a secondary filter that catches the cases where the model's softmax distribution is spread across multiple classes (indicating confusion between similar disease categories), even if the top class probability is above 50%.

The threshold can be tuned without retraining the model — it only affects how predictions are categorised after inference.

---

## The Two-Path Architecture: Why Not Always Use Grad-CAM?

The two-path design (fast `predict()` to check reliability, then `predict_with_cam()` only if reliable) might seem complex. Why not run Grad-CAM on every prediction?

**Memory:** A Grad-CAM computation holds the full intermediate activation tensors from all convolutional layers in memory throughout the forward pass. For a batch size of 1 on CPU, this is manageable — but it adds memory pressure compared to `no_grad()`.

**Latency:** The Grad-CAM backward pass adds 10–30% to inference time on CPU. For UNRELIABLE predictions (which don't need a heatmap), this latency is wasted.

**Correctness:** As noted above, a Grad-CAM heatmap for an UNRELIABLE prediction would be noisy and potentially misleading. Not generating it is the correct behaviour.

The two-path design is a deliberate performance and correctness optimisation. It has no impact on the farmer's experience for RELIABLE predictions.

---

## EfficientNet-B0 Architecture Summary

For completeness:

- **Input:** 224×224×3 RGB tensor (normalised)
- **Architecture:** Mobile inverted bottleneck convolution (MBConv) blocks with squeeze-and-excitation (SE) modules, arranged in 7 stages
- **Feature pyramid:** The model builds increasingly abstract representations at decreasing spatial resolutions through each stage
- **Head:** Global average pooling → dropout (0.2) → linear layer (34 outputs for 34 disease classes)
- **Parameters:** ~5.3M
- **Output:** 34-dimensional logit vector → softmax probabilities

The final convolutional feature block (`model.features[-1]`) is the Grad-CAM target layer — it contains the highest-level semantic feature maps before the global average pool, making it the most informative layer for class-specific localisation.

---

## Model Loading and the Checkpoint Format Problem

PyTorch models can be saved in multiple formats, and loading them requires handling all variants:

1. **Full serialised model** — the entire `nn.Module` object was pickled and saved
2. **State dict** — only the weights (parameter tensors), keyed by layer name
3. **Nested state dict** — state dict stored under a `'state_dict'`, `'model_state_dict'`, or `'model'` key in a larger checkpoint dict
4. **DataParallel prefix** — weights saved from a multi-GPU training run have `'module.'` prepended to every key

NAVA's `_extract_state_dict()` function handles all four formats transparently. This robustness is important because the EfficientNet checkpoint might be re-trained by other researchers using different saving conventions.
