# Mizhi Detection: `inference.py` and `gradcam.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/02_disease_detection_pipeline.md](../technical/02_disease_detection_pipeline.md) | [technical/03_gradcam_explainability.md](../technical/03_gradcam_explainability.md) | [05_gathi_routers_weather_diagnose_vnir_chat.md](05_gathi_routers_weather_diagnose_vnir_chat.md)

**Source files:**
- [`inference.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mizhi/detection/inference.py)
- [`gradcam.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mizhi/detection/gradcam.py)
- [`labels.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mizhi/detection/labels.py)

---

## `PredictionResult` Dataclass

```python
@dataclass
class PredictionResult:
    class_index: int
    class_label: str
    confidence: float
    reliability: str
```

A simple value object. Using a `@dataclass` rather than a dict gives type safety, autocompletion, and makes the code self-documenting (the caller knows exactly what fields to expect). `reliability` is either `"RELIABLE"` or `"UNRELIABLE"` — a string rather than a bool to make logging and API serialisation clear.

---

## Model Builder Functions

**`_build_model(num_classes: int) -> torch.nn.Module`**
```python
def _build_model(num_classes: int) -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model
```

`weights=None` creates the model architecture with randomly initialised weights. The checkpoint weights are loaded separately. This avoids downloading ImageNet weights from the internet at inference time.

`model.classifier[1]` is the final linear layer of EfficientNet-B0's classifier head — the layer that maps from the feature dimension (1280) to the number of output classes. The original ImageNet model has 1000 outputs. NAVA replaces this with a linear layer of `num_classes` (34) outputs, matching the disease class count.

**`_extract_state_dict(checkpoint) -> Tuple[Optional[Module], Optional[dict]]`**
```python
def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint, None       # Saved as full model
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                return None, checkpoint[key]   # Nested dict
        return None, checkpoint       # Flat state dict
    return None, None                 # Unknown format
```

Handles four checkpoint formats transparently. The function returns either a ready-to-use `Module` (if the checkpoint is a full model) or a state dict (to be loaded via `load_state_dict`). Returns `(None, None)` for unrecognised formats, which causes `ValueError` in the caller.

**`_clean_state_dict(state_dict: dict) -> dict`**
```python
def _clean_state_dict(state_dict: dict) -> dict:
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
```

Removes the `"module."` prefix that `torch.nn.DataParallel` adds to every key when a model is trained on multiple GPUs. Without this cleaning, `load_state_dict()` would fail with key mismatches because the current single-GPU model's keys don't have this prefix.

---

## `EfficientNetB0Predictor` Class

### `__init__`

```python
class EfficientNetB0Predictor:
    def __init__(
        self,
        model_path=None,
        labels_path=None,
        device="cpu",
        confidence_threshold=0.80,
    ):
        self.device = torch.device(device)
        self.labels = load_labels(self.labels_path)
        self.confidence_threshold = confidence_threshold

        self.model = _build_model(num_classes=len(self.labels))
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model_obj, state_dict = _extract_state_dict(checkpoint)

        if model_obj is not None:
            self.model = model_obj
        elif state_dict is not None:
            self.model.load_state_dict(_clean_state_dict(state_dict), strict=True)
        else:
            raise ValueError("Unsupported checkpoint format")

        self.model.to(self.device)
        self.model.eval()

        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._cam = GradCamGenerator(self.model, self.model.features[-1])
```

The constructor:
1. Resolves the number of classes from the labels file (so the model architecture matches the checkpoint)
2. Loads the checkpoint using `torch.load(map_location="cpu")` — always loads to CPU first, even if the target device is a CUDA GPU (avoids CUDA OOM if weights are large)
3. Calls `model.to(self.device)` — moves the model to the configured device
4. Calls `model.eval()` — puts the model in evaluation mode (disables dropout, sets BatchNorm to use running statistics)
5. Constructs the preprocessing transform pipeline
6. Creates a `GradCamGenerator` targeting `model.features[-1]`

**`model.eval()` is critical.** EfficientNet-B0 uses Batch Normalisation, which behaves differently during training (uses batch statistics) and evaluation (uses running statistics accumulated during training). Not calling `.eval()` would produce inconsistent predictions on small batch sizes (batch size 1 in inference).

### `_preprocess`

```python
def _preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
    image = image.convert("RGB")
    cropped = transforms.CenterCrop(224)(transforms.Resize(256)(image))
    tensor = self._transform(image)
    return tensor, cropped
```

Returns both the normalised tensor (for model input) and the cropped PIL image (for Grad-CAM overlay). The cropped image is 224×224 in PIL format, not normalised — the Grad-CAM overlay is generated by compositing the heatmap over this original-colour cropped image.

`image.convert("RGB")` handles greyscale, RGBA, or palette-mode images by converting them all to 3-channel RGB. Without this, the network's 3-channel input would receive the wrong number of channels.

### `predict()`

```python
def predict(self, image: Image.Image) -> PredictionResult:
    tensor, _ = self._preprocess(image)
    input_tensor = tensor.unsqueeze(0).to(self.device)
    with torch.no_grad():
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, class_index = torch.max(probs, dim=1)
    
    conf = float(confidence.item())
    reliability = "RELIABLE" if conf >= self.confidence_threshold else "UNRELIABLE"
    return PredictionResult(...)
```

`tensor.unsqueeze(0)` adds the batch dimension: `[3, 224, 224]` → `[1, 3, 224, 224]`. The model expects `[batch_size, channels, height, width]`.

`torch.no_grad()` disables the gradient computation graph. `confidence.item()` converts the single-element tensor to a Python float.

### `predict_with_cam()`

```python
def predict_with_cam(self, image: Image.Image) -> Tuple[PredictionResult, Image.Image]:
    tensor, cropped = self._preprocess(image)
    input_tensor = tensor.unsqueeze(0).to(self.device)

    # No torch.no_grad() — gradients required for Grad-CAM
    logits = self.model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    confidence, class_index = torch.max(probs, dim=1)

    cam_image = self._cam.generate(input_tensor, cropped, idx)
    result = PredictionResult(...)
    return result, cam_image
```

Identical to `predict()` except: no `torch.no_grad()` wrapper, and `self._cam.generate()` is called with the input tensor, original cropped image, and the predicted class index. The `GradCamGenerator` uses `input_tensor` (not a copy) — the same computation graph that PyTorch built during the forward pass is available for backpropagation.

---

## `GradCamGenerator` Class

```python
class GradCamGenerator:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._cam: Optional[GradCAM] = None

    def _ensure_cam(self) -> None:
        if self._cam is None:
            self._cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def generate(self, input_tensor, original_image, class_index) -> Image.Image:
        self._ensure_cam()
        targets = [ClassifierOutputTarget(class_index)]
        grayscale_cam = self._cam(input_tensor=input_tensor, targets=targets)[0]

        rgb_img = np.array(original_image).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return Image.fromarray(overlay)
```

`GradCamGenerator` wraps `pytorch_grad_cam.GradCAM`, which handles the hook registration, backward pass, and activation/gradient aggregation described in [technical/03_gradcam_explainability.md](../technical/03_gradcam_explainability.md).

**Lazy `_cam` construction (`_ensure_cam`):**
`GradCAM` is constructed on first use rather than in `__init__`. This is because `GradCAM` registers hooks on the model at construction time. If the model is not yet loaded when `GradCamGenerator.__init__` runs (e.g., due to an import ordering issue), the hooks would attach to an uninitialised model. Lazy construction ensures hooks are attached to the fully-initialised model on first `generate()` call.

**`ClassifierOutputTarget(class_index)`:**
This tells `pytorch_grad_cam` which class to use as the score for backpropagation — specifically the predicted class (not the ground truth, which is unknown at inference time). This is the standard Grad-CAM usage: explain the model's actual prediction.

**`show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)`:**
`pytorch_grad_cam`'s utility function that:
1. Applies the JET colour map to the grayscale heatmap (0=blue, 1=red)
2. Composites the coloured heatmap over the original image with alpha blending
3. Returns the composite as a uint8 numpy array

`rgb_img` must be in `[0, 1]` float32 range — hence the `/255.0` division. `use_rgb=True` specifies RGB channel order (matching PIL). Without this, channels would be swapped (PIL is RGB, OpenCV default is BGR).

---

## `labels.py` — Label Loading

The labels file (`EfficientNet-B0-labels.txt`) is a plain text file with one class name per line. The index of each line corresponds to the class index output by the model's final layer. `load_labels()` reads this file and returns a list of strings.

The label format: `banana_black_sigatoka`, `rice_blast`, `tomato_late_blight`, etc. The underscore separation makes it easy to parse the crop name (`banana`) from the disease name (`black_sigatoka`) if needed for filtering.
