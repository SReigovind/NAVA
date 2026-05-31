# Mizhi VNIR: `pipeline.py`, `inference.py`, `analyzer.py`, `validation.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/04_vnir_stress_monitoring.md](../technical/04_vnir_stress_monitoring.md) | [05_gathi_routers_weather_diagnose_vnir_chat.md](05_gathi_routers_weather_diagnose_vnir_chat.md) | [12_shared_storage.md](12_shared_storage.md)

**Source files:**
- [`pipeline.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mizhi/vnir/pipeline.py)
- [`inference.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mizhi/vnir/inference.py)
- [`analyzer.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mizhi/vnir/analyzer.py)
- [`validation.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mizhi/vnir/validation.py)

---

## Overview: Three-Layer Architecture

The VNIR pipeline is split into three separate classes with distinct concerns:

1. **`VNIRPipeline`** — orchestration: receives a PIL image, coordinates the other two classes, returns `VNIRStats` and visualisation images
2. **`VNIREngine`** — ONNX inference: takes a masked RGB image, returns a predicted NIR grayscale image
3. **`VNIRAnalyzer`** — statistics and alerting: takes the raw values and the history, returns computed statistics and the two-level alert status

This separation makes each class independently testable and replaceable.

---

## `VNIRStats` Dataclass

```python
@dataclass
class VNIRStats:
    status: str
    avg_g: float = 0.0
    avg_vnir: float = 0.0
    ratio: float = 0.0
    baseline: float | None = None
    rolling_avg: float | None = None
    prev_checkpoint_avg: float | None = None
    global_avg: float | None = None
    vs_baseline: float | None = None
    vs_global: float | None = None
    vs_rolling: float | None = None
    vs_prev_checkpoint: float | None = None
    ready: bool = False
    leaf_state: str = "NONE"
    scan_index: int = 0
```

`VNIRStats` is the single result object returned by the full pipeline. All fields default to `0.0` or `None` — which allows construction with just `status` for early-return cases (`VNIRStats(status="No Leaf Detected")`).

The `None` defaults for comparison fields are important: they tell the frontend that these values are not yet available (during calibration) rather than that they are zero.

---

## `VNIRPipeline` — The Orchestrator

### `__init__`

```python
class VNIRPipeline:
    def __init__(self, model_path=None, stress_threshold_pct=15.0, warning_threshold_pct=10.0):
        self.engine = VNIREngine(model_path=model_path)
        self.analyzer = VNIRAnalyzer(
            stress_threshold_pct=stress_threshold_pct,
            warning_threshold_pct=warning_threshold_pct,
        )
```

The two thresholds (15% CRITICAL, 10% WARNING) are configurable through `Settings` and passed from `get_vnir_pipeline()` in `deps.py`. This allows tuning without code changes.

### `isolate_leaf()` — HSV Segmentation in Detail

```python
def isolate_leaf(self, frame_bgr: np.ndarray) -> LeafIsolationResult:
    frame_256 = cv2.resize(frame_bgr, (256, 256))
    hsv_frame = cv2.cvtColor(frame_256, cv2.COLOR_BGR2HSV)
    total_pixels = 256 * 256

    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Green mask: H=[30°,90°], S≥40, V≥40
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_hsv_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green_hsv_mask = cv2.morphologyEx(green_hsv_mask, cv2.MORPH_CLOSE, kernel_large)
    green_hsv_mask = cv2.morphologyEx(green_hsv_mask, cv2.MORPH_OPEN, kernel_small)
    ...
```

**OpenCV's HSV range:** OpenCV represents Hue in [0, 179] (not [0, 360]). So "30° to 90°" in standard HSV maps to values 30 to 90 in OpenCV's range (conveniently the same numbers). Saturation and Value are in [0, 255].

**Elliptical kernels:** `cv2.MORPH_ELLIPSE` creates an elliptical structuring element. Ellipses are preferred over rectangles for morphological operations on organic shapes (leaves) because they don't introduce rectangular artifacts at the mask boundary.

**Close then Open sequence:**
- `MORPH_CLOSE` (dilation then erosion): fills holes in the mask — gaps in the leaf caused by specular highlights or partially transparent areas
- `MORPH_OPEN` (erosion then dilation): removes isolated small regions — background noise pixels that passed the colour filter

Using large kernel (11×11) for close and small kernel (5×5) for open is intentional: large kernel fills bigger holes, small kernel removes only small noise.

**Contour analysis and 5% area threshold:**
```python
min_area = total_pixels * 0.05  # 5% of 256×256 = 3,276 pixels
if max_green_area >= max_yellow_area and max_green_area >= min_area:
    leaf_state = "GREEN"
elif max_yellow_area > max_green_area and max_yellow_area >= min_area:
    leaf_state = "YELLOW_BROWN"
```

Finding the largest contour (`max(..., key=cv2.contourArea)`) picks the primary leaf object, ignoring smaller background noise. The 5% threshold prevents a tiny leaf fragment at the edge of the frame from triggering the full VNIR pipeline.

**`cv2.drawContours(..., -1, 255, -1)`:** Draws the filled interior (not just the border) of the selected contour into `contour_bound`. `thickness=-1` means fill. This filled mask is then AND-ed with the HSV mask to produce the final `leaf_mask` — the intersection of the HSV colour filter and the contour interior.

### `process_image()` — Full Pipeline

```python
def process_image(self, image, plant_id, history_ratios):
    frame_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    isolation = self.isolate_leaf(frame_bgr)

    hsv_image = Image.fromarray(cv2.cvtColor(isolation.hsv_visual, cv2.COLOR_BGR2RGB))
    vnir_image = Image.new("L", (256, 256), color=0)  # default: black

    if isolation.leaf_state == "GREEN":
        vnir_image = self.engine.predict(Image.fromarray(isolation.masked_rgb))
        vnir_array = np.array(vnir_image).astype(np.float32)
        stats = self.analyzer.analyze(isolation.masked_rgb, vnir_array, isolation.leaf_mask, history_ratios)
    elif isolation.leaf_state == "YELLOW_BROWN":
        stats = VNIRStats(status="CRITICAL: Visual Stress")
    else:
        stats = VNIRStats(status="No Leaf Detected")

    stats.leaf_state = isolation.leaf_state
    return stats, hsv_image, vnir_image
```

Three code paths based on `leaf_state`:
- **GREEN:** Run full ONNX inference + statistics analysis
- **YELLOW_BROWN:** Immediate CRITICAL (visual stress already obvious) — no ONNX call needed
- **NONE:** No leaf detected

The `vnir_image` default is a black (zero) image for non-GREEN states. The frontend displays this image as a visual alongside the HSV isolation view.

---

## `VNIREngine` — ONNX Inference

```python
class VNIREngine:
    def __init__(self, model_path=None):
        path = str(model_path or models_dir() / "thanal_vnir.onnx")
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image) -> Image.Image:
        img = image.resize((256, 256)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0   # [0,1]
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]    # NCHW
        output = self.session.run(None, {self.input_name: arr})
        pred = output[0][0, 0]                           # (H, W) float32
        pred_uint8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(pred_uint8, mode="L")
```

**`ort.InferenceSession`:** Loads the ONNX model and prepares an optimised inference graph. ONNX Runtime selects the best backend (CPU in this deployment) automatically.

**`self.session.get_inputs()[0].name`:** ONNX models have named inputs. The name depends on how the model was exported from PyTorch. Fetching it at load time instead of hardcoding `"input"` makes the engine robust to different export conventions.

**`arr.transpose(2, 0, 1)[np.newaxis, ...]`:** Transforms from `(H, W, C)` numpy HWC format to `(1, C, H, W)` NCHW format. ONNX Runtime expects NCHW. The `[np.newaxis, ...]` adds the batch dimension.

**`np.clip(pred, 0, 1)`:** The model output may contain values slightly outside [0, 1] due to floating-point arithmetic. Clipping prevents overflow when converting to uint8.

---

## `VNIRAnalyzer.analyze()` — The Statistics Engine

```python
def analyze(self, rgb_image, vnir_image, leaf_mask, history_ratios):
    g_channel = rgb_image[:, :, 1].astype(np.float32)
    leaf_g = g_channel[leaf_mask > 0]       # green pixels only
    leaf_vnir = vnir_image[leaf_mask > 0]   # same pixels in NIR prediction

    avg_g = float(np.mean(leaf_g))
    avg_vnir = float(np.mean(leaf_vnir))
    current_ratio = float(avg_vnir / (avg_g + 1e-5))  # zero-division guard
```

**Channel indexing:** `rgb_image[:, :, 1]` extracts the Green channel (channel index 1 in RGB). `leaf_mask > 0` creates a boolean index that selects only the pixels inside the leaf area.

**`avg_g + 1e-5`:** The `1e-5` addition prevents division by zero if the green channel mean is exactly 0 (e.g., a completely dark image). This is the arithmetic zero-division guard (distinct from the zero-ratio guard in the router, which guards against failed scans corrupting the history timeseries).

```python
all_ratios = history_ratios + [current_ratio]
total_scans = len(all_ratios)

if total_scans < 5:
    stats.status = f"Calibrating ({total_scans}/5)"
else:
    baseline = float(np.mean(all_ratios[0:5]))  # first 5 scans
    current_5_avg = float(np.mean(all_ratios[-5:]))  # last 5 scans (rolling)

    if total_scans >= 10:
        prev_checkpoint_avg = float(np.mean(all_ratios[-10:-5]))
    else:
        prev_checkpoint_avg = baseline

    # Two-tier comparison
    if stats.vs_baseline <= -self.stress_threshold_pct:
        stats.status = "CRITICAL: STRESS"
    elif stats.vs_rolling <= -self.warning_threshold_pct:
        stats.status = "WARNING: STRESS"
    else:
        stats.status = "OK"
```

**`all_ratios = history_ratios + [current_ratio]`:** The current scan is conceptually the last in the history for statistics purposes. Appending it to the end before computing means `all_ratios[-5:]` always includes the current scan as the most recent point.

**`_safe_pct(new_val, old_val)`:**
```python
def _safe_pct(new_val: float, old_val: float) -> float:
    denom = old_val if abs(old_val) > 1e-6 else 1e-6
    return ((new_val - old_val) / denom) * 100
```
Computes the percentage change from `old_val` to `new_val`. The guard `abs(old_val) > 1e-6` prevents division by near-zero baselines. A negative result means the current value is below the reference — which is what triggers stress alerts.

---

## `validation.py` — Plant ID Validation

```python
def validate_plant_id(plant_id: str | int) -> str:
    pid = str(plant_id).strip()
    if not pid:
        raise ValueError("plant_id cannot be empty")
    return pid
```

A simple utility function that normalises the plant ID to a string and validates it's not empty. Called in the pipeline to ensure a valid identifier is available for logging. The plant ID is used only for log messages — the actual database interaction uses the integer `plant_id` from the route handler.
