# VNIR Stress Monitoring Qualitative Report

## Phase 1: Baseline Calibration (Banana)

Uploading 5 healthy banana images to establish the baseline.

| Scan | Status | Ratio | Baseline | Image | HSV Isolate | NIR Predicted |
|------|--------|-------|----------|-------|-------------|---------------|
| 1 | Calibrating (1/5) | 0.5832 | N/A | <img src='./outputs/vnir_calib_0_orig.jpg' width='100'/> | <img src='./outputs/vnir_calib_0_hsv.jpg' width='100'/> | <img src='./outputs/vnir_calib_0_nir.jpg' width='100'/> |
| 2 | Calibrating (2/5) | 0.8618 | N/A | <img src='./outputs/vnir_calib_1_orig.jpg' width='100'/> | <img src='./outputs/vnir_calib_1_hsv.jpg' width='100'/> | <img src='./outputs/vnir_calib_1_nir.jpg' width='100'/> |
| 3 | Calibrating (3/5) | 1.5054 | N/A | <img src='./outputs/vnir_calib_2_orig.jpg' width='100'/> | <img src='./outputs/vnir_calib_2_hsv.jpg' width='100'/> | <img src='./outputs/vnir_calib_2_nir.jpg' width='100'/> |
| 4 | Calibrating (4/5) | 0.3910 | N/A | <img src='./outputs/vnir_calib_3_orig.jpg' width='100'/> | <img src='./outputs/vnir_calib_3_hsv.jpg' width='100'/> | <img src='./outputs/vnir_calib_3_nir.jpg' width='100'/> |
| 5 | OK | 0.7794 | 0.8241669285434821 | <img src='./outputs/vnir_calib_4_orig.jpg' width='100'/> | <img src='./outputs/vnir_calib_4_hsv.jpg' width='100'/> | <img src='./outputs/vnir_calib_4_nir.jpg' width='100'/> |

<div style='page-break-before: always;'></div>

## Phase 2: Stress Detection (Banana Sigatoka)

Testing with 3 diseased (Sigatoka) images.

| Scan | Status | Ratio | Vs Baseline | Image | HSV Isolate | NIR Predicted |
|------|--------|-------|-------------|-------|-------------|---------------|
| 1 | **CRITICAL: Visual Stress** | 0.0000 | N/A | <img src='./outputs/vnir_stress_0_orig.jpg' width='100'/> | <img src='./outputs/vnir_stress_0_hsv.jpg' width='100'/> | <img src='./outputs/vnir_stress_0_nir.jpg' width='100'/> |
| 2 | **No Leaf Detected** | 0.0000 | N/A | <img src='./outputs/vnir_stress_1_orig.jpg' width='100'/> | <img src='./outputs/vnir_stress_1_hsv.jpg' width='100'/> | <img src='./outputs/vnir_stress_1_nir.jpg' width='100'/> |
| 3 | **WARNING: STRESS** | 0.2580 | -68.70% | <img src='./outputs/vnir_stress_2_orig.jpg' width='100'/> | <img src='./outputs/vnir_stress_2_hsv.jpg' width='100'/> | <img src='./outputs/vnir_stress_2_nir.jpg' width='100'/> |
