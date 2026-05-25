# Disease Detection Qualitative Report

## Banana

| Condition | True Label | Image | GradCAM | Predicted Class | Confidence | Reliable |
|-----------|------------|-------|---------|-----------------|------------|----------|
| Healthy | **banana_healthy** | <img src='./outputs/banana_healthy_orig.jpg' width='150'/> | <img src='./outputs/banana_healthy_cam.jpg' width='150'/> | **banana_healthy** | 0.9997 | **RELIABLE** |
| Diseased | **banana_sigatoka** | <img src='./outputs/banana_diseased_orig.jpg' width='150'/> | <img src='./outputs/banana_diseased_cam.jpg' width='150'/> | **banana_sigatoka** | 0.9996 | **RELIABLE** |

<div style='page-break-before: always;'></div>

## Cassava

| Condition | True Label | Image | GradCAM | Predicted Class | Confidence | Reliable |
|-----------|------------|-------|---------|-----------------|------------|----------|
| Healthy | **cassava_healthy** | <img src='./outputs/cassava_healthy_orig.jpg' width='150'/> | <img src='./outputs/cassava_healthy_cam.jpg' width='150'/> | **cassava_healthy** | 1.0000 | **RELIABLE** |
| Diseased | **cassava_blight** | <img src='./outputs/cassava_diseased_orig.jpg' width='150'/> | <img src='./outputs/cassava_diseased_cam.jpg' width='150'/> | **cassava_blight** | 1.0000 | **RELIABLE** |

<div style='page-break-before: always;'></div>

## Corn

| Condition | True Label | Image | GradCAM | Predicted Class | Confidence | Reliable |
|-----------|------------|-------|---------|-----------------|------------|----------|
| Healthy | **corn_healthy** | <img src='./outputs/corn_healthy_orig.jpg' width='150'/> | <img src='./outputs/corn_healthy_cam.jpg' width='150'/> | **corn_healthy** | 0.9961 | **RELIABLE** |
| Diseased | **corn_cercospora_leaf_spot** | <img src='./outputs/corn_diseased_orig.jpg' width='150'/> | <img src='./outputs/corn_diseased_cam.jpg' width='150'/> | **corn_common_rust** | 0.8169 | **RELIABLE** |

<div style='page-break-before: always;'></div>

## Cucumber

| Condition | True Label | Image | GradCAM | Predicted Class | Confidence | Reliable |
|-----------|------------|-------|---------|-----------------|------------|----------|
| Healthy | **cucumber_healthy** | <img src='./outputs/cucumber_healthy_orig.jpg' width='150'/> | <img src='./outputs/cucumber_healthy_cam.jpg' width='150'/> | **cucumber_healthy** | 1.0000 | **RELIABLE** |
| Diseased | **cucumber_angular_leaf_spot** | <img src='./outputs/cucumber_diseased_orig.jpg' width='150'/> | <img src='./outputs/cucumber_diseased_cam.jpg' width='150'/> | **cucumber_angular_leaf_spot** | 1.0000 | **RELIABLE** |

<div style='page-break-before: always;'></div>

## Rice

| Condition | True Label | Image | GradCAM | Predicted Class | Confidence | Reliable |
|-----------|------------|-------|---------|-----------------|------------|----------|
| Healthy | **rice_healthy** | <img src='./outputs/rice_healthy_orig.jpg' width='150'/> | <img src='./outputs/rice_healthy_cam.jpg' width='150'/> | **rice_healthy** | 0.9985 | **RELIABLE** |
| Diseased | **rice_blast** | <img src='./outputs/rice_diseased_orig.jpg' width='150'/> | <img src='./outputs/rice_diseased_cam.jpg' width='150'/> | **rice_blast** | 0.9992 | **RELIABLE** |

<div style='page-break-before: always;'></div>

## Soybean

| Condition | True Label | Image | GradCAM | Predicted Class | Confidence | Reliable |
|-----------|------------|-------|---------|-----------------|------------|----------|
| Healthy | **soybean_healthy** | <img src='./outputs/soybean_healthy_orig.jpg' width='150'/> | N/A | **tomato_late_blight** | 0.2508 | **UNRELIABLE** |
| Diseased | **soybean_bacterial_blight** | <img src='./outputs/soybean_diseased_orig.jpg' width='150'/> | <img src='./outputs/soybean_diseased_cam.jpg' width='150'/> | **soybean_bacterial_blight** | 1.0000 | **RELIABLE** |

<div style='page-break-before: always;'></div>

## Tomato

| Condition | True Label | Image | GradCAM | Predicted Class | Confidence | Reliable |
|-----------|------------|-------|---------|-----------------|------------|----------|
| Healthy | **tomato_healthy** | <img src='./outputs/tomato_healthy_orig.jpg' width='150'/> | N/A | **tomato_mosaic_virus** | 0.7775 | **UNRELIABLE** |
| Diseased | **tomato_bacterial_leaf_spot** | <img src='./outputs/tomato_diseased_orig.jpg' width='150'/> | N/A | **tomato_early_blight** | 0.7812 | **UNRELIABLE** |

