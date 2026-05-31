# The Model Comparison Study

> **Subfolder:** `non_technical/`
> **Cross-references:** [02_research_foundation.md](02_research_foundation.md) | [03_dataset_and_training_strategy.md](03_dataset_and_training_strategy.md) | [technical/02_disease_detection_pipeline.md](../technical/02_disease_detection_pipeline.md) | [code/06_mizhi_detection.md](../code/06_mizhi_detection.md)

---

## Why a Comparison Study?

Choosing a neural network architecture is a design decision, not an arbitrary preference. It should be made empirically, with evidence. The architecture determines the accuracy ceiling, the inference speed, the memory footprint, and the deployment cost — all of which matter for NAVA's target use case.

NAVA's Phase 1 contribution was a formal, reproducible comparison study: three architectures, same dataset, same training configuration, same hardware, same evaluation protocol. The winner would be the backbone of NAVA's disease detection system.

---

## The Three Candidates

### ResNet-50 (Depth Scaling)
Residual Networks were introduced by He et al. at Microsoft Research in 2015 and won the ImageNet competition that year. The key innovation is the **residual connection** (skip connection): instead of learning the desired output directly, each block learns the *residual* — the difference between the input and the desired output. This allows very deep networks (ResNet goes up to 152 layers) to be trained without the vanishing gradient problem that plagued earlier deep networks.

ResNet-50 has 25.6 million parameters. At the time of its publication, it represented the state of the art in accuracy through depth — the idea that making networks deeper, with the right architecture, keeps improving performance.

**Why test it:** ResNet-50 is the benchmark architecture for transfer learning in applied computer vision. Any serious comparison must include it.

### MobileNetV2 (Width Scaling — Efficiency-Focused)
MobileNetV2 was designed by Google specifically for deployment on mobile devices (hence the name). The key technique is **depthwise separable convolutions**: instead of a single 3D convolution that mixes spatial and channel information simultaneously, the operation is split into a depthwise convolution (one filter per input channel, spatial mixing only) and a pointwise convolution (1x1 convolution, channel mixing only). This reduces the computational cost by a factor of 8–9x compared to standard convolutions.

MobileNetV2 has 3.4 million parameters — much smaller than ResNet-50. It is designed to be fast on CPU and deployable on phones.

**Why test it:** NAVA needs CPU-deployable inference. MobileNetV2 represents the efficiency-first end of the spectrum.

### EfficientNet-B0 (Compound Scaling)
EfficientNet (Tan and Le, Google Brain, 2019) starts from a different premise: instead of scaling one dimension of the network at a time (depth like ResNet, width like MobileNet), scale all three dimensions — depth, width, and input resolution — simultaneously, according to a principled ratio derived from neural architecture search.

The key insight from the paper: when you increase the input resolution, the network also needs more layers (more depth) to capture the larger receptive field needed, and more channels (more width) to capture the additional fine-grained features visible in higher-resolution images. Scaling them independently misses these interactions.

EfficientNet-B0 has 5.3 million parameters — between MobileNetV2 and ResNet-50 in size — but achieves 77.1% top-1 ImageNet accuracy, higher than ResNet-50 (76.1%) and much higher than MobileNetV2 (72.0%).

**Why test it:** If compound scaling is as effective as claimed, EfficientNet-B0 should be the best option for NAVA's constraints: it should be more accurate than MobileNetV2 while remaining small enough for CPU inference.

---

## The Experimental Setup

All three models were trained under identical conditions to ensure the comparison is fair:

| Setting | Value |
|---------|-------|
| Dataset | NAVA Superset (20,400 images, 34 classes) |
| Pretrained weights | ImageNet (for all three) |
| Optimiser | Adam |
| Learning rate | 1e-4 (fine-tuning) |
| Batch size | 32 |
| Epochs | 20 with early stopping |
| Hardware | Same GPU (single training run each) |
| Evaluation | Held-out test set, 4,089 images |

The only variable is the architecture. Everything else is constant.

---

## Results

| Model | Architecture Principle | Val. Accuracy | Test Accuracy | Training Time |
|-------|----------------------|--------------|---------------|---------------|
| ResNet-50 | Depth scaling | 85.39% | 85.31% | 5 min 00 sec |
| MobileNetV2 | Width scaling (efficiency) | 83.53% | 83.41% | 4 min 34 sec |
| **EfficientNet-B0** | **Compound scaling** | **94.54%** | **94.47%** | **4 min 38 sec** |

EfficientNet-B0 outperforms both alternatives by a significant margin — approximately 9 percentage points over ResNet-50 and 11 over MobileNetV2. And it does so in approximately the same training time as MobileNetV2.

---

## Interpreting the Results

### Why Such a Large Gap?

The performance gap of 9–11 percentage points is not subtle. It is large enough that it cannot be explained by random variation or hyperparameter differences. It reflects a genuine architectural advantage.

The explanation is compound scaling: when you have a 256×256 input image of a leaf that contains subtle textural differences between disease categories, a model that scales depth, width, and resolution together can capture both the fine-grained local patterns (visible at high resolution) and the structural relationships across the leaf (requiring sufficient depth and width) simultaneously. A model that only scales depth (ResNet) or only designed for efficiency (MobileNet) misses part of this optimisation.

### Why Is MobileNetV2 Worse Than ResNet-50?

MobileNetV2 was designed for speed and small size on mobile devices. Its depthwise separable convolutions are computationally efficient but sacrifice some representational capacity compared to full convolutions. For a task that requires distinguishing 34 similar-looking disease categories — many of which differ only in subtle textural or colour patterns — that representational capacity matters.

MobileNetV2 is the right choice if deployment speed is the primary constraint and a few percentage points of accuracy can be sacrificed. For NAVA, where a wrong diagnosis can cause real harm, accuracy takes priority.

### Is 94.54% Good Enough?

94.54% test accuracy across 34 classes means the model is wrong approximately 1 in 18 predictions. This is not perfect — and NAVA's confidence gate exists precisely to handle those errors. When the model is uncertain (softmax < 0.80), it flags the prediction as `UNRELIABLE` rather than presenting a confident-but-wrong diagnosis.

The important comparison is not "is 94.54% perfect?" but "is 94.54% better than the alternative?" The alternative for most smallholder farmers in Kerala is no diagnostic assistance at all — visual inspection with no training, the neighbour's advice, or waiting for an extension officer who may not come for weeks.

---

## The Decision

EfficientNet-B0 was selected as NAVA's disease detection backbone based on:

1. **Accuracy:** 94.54% — highest among the three candidates
2. **Training efficiency:** Near-identical training time to the faster MobileNetV2
3. **Model size:** 5.3M parameters — small enough for CPU inference without significant latency
4. **Research validation:** A well-documented architecture with extensive peer-reviewed evidence of performance across vision tasks
5. **Transfer learning compatibility:** ImageNet pre-training provides a strong foundation for the Superset fine-tuning

The decision is documented and reproducible: anyone can replicate the comparison study by running the training notebooks in the `notebooks/` directory with the Superset dataset.
