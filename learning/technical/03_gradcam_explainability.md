# Grad-CAM Explainability

> **Subfolder:** `technical/`
> **Cross-references:** [02_disease_detection_pipeline.md](02_disease_detection_pipeline.md) | [non_technical/04_design_philosophy.md](../non_technical/04_design_philosophy.md) | [code/06_mizhi_detection.md](../code/06_mizhi_detection.md)

---

## Why Explainability Is a Safety Feature

An AI system that says "this is Late Blight" without any indication of *why* it reached that conclusion asks the user to trust it unconditionally. For a farmer making a treatment decision — choosing a fungicide, timing a spray, deciding whether to isolate plants — that unconditional trust is dangerous. The model may be pattern-matching on a shadow, a soil particle, or a reflection rather than on the actual disease symptom.

Grad-CAM (Gradient-weighted Class Activation Mapping) transforms the model from an oracle into an expert who shows their work. The resulting heatmap overlay answers the question: "Which part of the leaf is the model looking at to make this prediction?" If the highlighted region corresponds to a visible lesion, the farmer has visual confirmation that the model's reasoning is sound. If it highlights the background or an unrelated region, the prediction should be treated with skepticism.

This is not just an academic nicety — it is a practical trust mechanism that requires zero technical knowledge to interpret.

---

## What Grad-CAM Computes

Grad-CAM computes a spatial importance map for a specific class prediction by combining two signals:

1. **Feature maps** — the activations at a chosen convolutional layer (the final block, `model.features[-1]` in EfficientNet)
2. **Gradients** — the gradient of the predicted class score with respect to those activations

The process:

**Step 1: Forward pass**
Run the input image through the full network. Capture the activation tensors at the target layer (a hook registered on the layer stores them).

**Step 2: Backward pass**
Backpropagate the score for the *predicted class* (not the loss, but the raw class score — a single scalar). Capture the gradient tensors flowing back through the target layer (a second hook stores them).

**Step 3: Global average pool the gradients**
For each channel in the gradient tensor, average across the spatial dimensions. This produces a weight vector: one scalar weight per channel of the feature map. The weight represents "how much did changes in this channel's activations affect the predicted class score?"

**Step 4: Weighted sum of feature maps**
Multiply each channel of the feature map by its corresponding gradient weight, then sum across channels. The result is a single spatial map — one value per spatial location in the feature map.

**Step 5: ReLU**
Apply ReLU to the weighted sum. This zeroes out negative values, retaining only the spatial locations that have a *positive* contribution to the predicted class score. Negative contributions (locations that suppressed the predicted class) are removed — they are not relevant to explaining why the model chose this class.

**Step 6: Upsampling and overlay**
The raw Grad-CAM output has the spatial resolution of the feature map (approximately 7×7 for EfficientNet-B0's last block at 224×224 input). This is upsampled to 224×224 using bilinear interpolation. The upsampled map is normalised to [0, 1] and then converted to a colour heatmap using OpenCV's `COLORMAP_JET` (blue = low activation, red = high activation). The heatmap is composited over the original cropped image as a semi-transparent overlay.

---

## Why the Last Convolutional Block?

The choice of target layer matters. Grad-CAM can be applied to any convolutional layer in the network, but earlier layers produce spatially diffuse, texture-focused maps, while later layers produce class-discriminative, semantically meaningful maps.

The last convolutional feature block (`model.features[-1]`) is chosen because:
- It contains the highest-level semantic features — the model has, by this point, encoded "this region looks like a fungal lesion" rather than "this region has a particular texture"
- It has the spatial resolution closest to the input (though still coarser) — the upsampled map is a reasonable approximation of where in the input the model is looking
- After this layer, global average pooling discards all spatial information — you cannot apply Grad-CAM to layers after the pooling

---

## The Hook Mechanism

PyTorch's hook system allows you to intercept tensors flowing through a network without modifying the network's forward or backward pass. Two types of hooks are used:

**Forward hook** (registered with `layer.register_forward_hook`):
Fires after the layer's forward pass completes. Captures the output activation tensor.

**Backward hook** (registered with `layer.register_full_backward_hook`):
Fires after gradients have been computed for the layer. Captures the gradient tensor with respect to the layer's output.

Both hooks store their tensors in the `GradCamGenerator` instance's state. After the forward and backward passes complete, the stored tensors are used to compute the weighted sum.

**Why not just access the gradients directly?** PyTorch does not expose intermediate gradients without retaining them explicitly. Hooks are the standard mechanism for capturing intermediate values during the forward and backward passes.

---

## The Critical Absence of `torch.no_grad()`

Grad-CAM requires gradients. If `torch.no_grad()` is active during the forward pass, PyTorch does not build the computation graph. When you subsequently call `.backward()` on the class score, there is no graph to backpropagate through — the gradient tensors at the hook location will be zeros or undefined.

This is why `predict_with_cam()` does not use `torch.no_grad()`. The forward pass is slower (the computation graph is maintained in memory), but gradients are available for the backward pass.

In practice, this makes `predict_with_cam()` approximately 20–30% slower than `predict()` on the same hardware. The extra cost is worthwhile for the explainability it provides.

---

## Limitations of Grad-CAM

Grad-CAM is not perfect. Understanding its limitations is part of using it responsibly.

**Low resolution:** The base heatmap is at the spatial resolution of the last convolutional layer — approximately 7×7 for EfficientNet-B0. Upsampling to 224×224 smooths this out, but the result is coarse. The heatmap can highlight a region that is correct but cannot pinpoint exactly which pixels matter.

**Single class:** Grad-CAM highlights what the model uses to discriminate the predicted class from all others. For a disease class that is visually similar to a healthy class, the highlighted region may be the subtle difference between them — which might not look like the most obvious feature to a human observer.

**Layer choice sensitivity:** If a different target layer is chosen, the heatmap changes. The last block is the most semantically meaningful choice, but this is a design decision, not an absolute truth.

**Not causal:** Grad-CAM shows correlation between spatial regions and the class prediction — regions that, when modified, would most affect the class score. It does not establish that the model is "looking at" those regions in any deep cognitive sense. It is a diagnostic tool, not a window into the model's "reasoning."

These limitations are not arguments against using Grad-CAM — they are arguments for presenting it honestly. NAVA's UI positions the heatmap as "where the model is looking," not "where the disease definitely is." The farmer retains the judgement call.
