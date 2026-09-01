# SMARTFALL AI — PHASE 10 PHONE ERROR & CONFUSION ANALYSIS

Detailed evaluation of the Top 3 fall-detection models on PHONE to understand missed falls, high-motion ADL false alarms, and class confusions.

## 1. `HistGradientBoosting` Error Profile

- **Test Fall Recall**: **`77.71%`** (Missed: `245` / `1099` falls)
- **Test False Positive Rate (FPR)**: **`21.21%`** (False Alarms: `197` / `929` ADLs)
- **Binary Fall F1**: **`0.7944`**

### Per-Class Fall Detection Sensitivity

| Fall Class | Test Support | Precision | Recall (Sensitivity) | F1-Score | Status |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 182 | 0.7399 | **90.66%** | 0.8148 | EXCELLENT |
| `FALL_FORWARD` | 320 | 0.4965 | **66.88%** | 0.5699 | ATTENTION |
| `FALL_FROM_SITTING` | 72 | 0.2996 | **98.61%** | 0.4595 | EXCELLENT |
| `FALL_LEFT` | 424 | 0.9752 | **27.83%** | 0.4330 | ATTENTION |
| `FALL_RIGHT` | 101 | 0.5385 | **20.79%** | 0.3000 | ATTENTION |

### High-Motion Normal Activity (ADL) Specificity

| High-Motion ADL | Test Support | Precision | Recall (Specificity) | F1-Score | False Fall Risk |
|---|---|---|---|---|---|
| `JUMPING` | 83 | 0.9878 | **97.59%** | 0.9818 | LOW |
| `RUNNING` | 174 | 0.8308 | **95.98%** | 0.8907 | LOW |
| `SIT_DOWN` | 42 | 0.5667 | **40.48%** | 0.4722 | MEDIUM |
| `STAND_UP` | 66 | 0.5750 | **34.85%** | 0.4340 | MEDIUM |
| `PICKING_UP_OBJECT` | 18 | 0.3333 | **11.11%** | 0.1667 | MEDIUM |

---

## 2. `Gradient Boosting` Error Profile

- **Test Fall Recall**: **`77.98%`** (Missed: `242` / `1099` falls)
- **Test False Positive Rate (FPR)**: **`21.21%`** (False Alarms: `197` / `929` ADLs)
- **Binary Fall F1**: **`0.7961`**

### Per-Class Fall Detection Sensitivity

| Fall Class | Test Support | Precision | Recall (Sensitivity) | F1-Score | Status |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 182 | 0.7443 | **89.56%** | 0.8130 | EXCELLENT |
| `FALL_FORWARD` | 320 | 0.5448 | **68.44%** | 0.6066 | ATTENTION |
| `FALL_FROM_SITTING` | 72 | 0.2857 | **91.67%** | 0.4356 | EXCELLENT |
| `FALL_LEFT` | 424 | 0.9624 | **30.19%** | 0.4596 | ATTENTION |
| `FALL_RIGHT` | 101 | 0.5942 | **40.59%** | 0.4824 | ATTENTION |

### High-Motion Normal Activity (ADL) Specificity

| High-Motion ADL | Test Support | Precision | Recall (Specificity) | F1-Score | False Fall Risk |
|---|---|---|---|---|---|
| `JUMPING` | 83 | 0.9872 | **92.77%** | 0.9565 | LOW |
| `RUNNING` | 174 | 0.8446 | **93.68%** | 0.8883 | LOW |
| `SIT_DOWN` | 42 | 0.3103 | **21.43%** | 0.2535 | MEDIUM |
| `STAND_UP` | 66 | 0.3500 | **21.21%** | 0.2642 | MEDIUM |
| `PICKING_UP_OBJECT` | 18 | 0.0556 | **5.56%** | 0.0556 | MEDIUM |

---

## 3. `Random Forest` Error Profile

- **Test Fall Recall**: **`74.16%`** (Missed: `284` / `1099` falls)
- **Test False Positive Rate (FPR)**: **`24.22%`** (False Alarms: `225` / `929` ADLs)
- **Binary Fall F1**: **`0.7620`**

### Per-Class Fall Detection Sensitivity

| Fall Class | Test Support | Precision | Recall (Sensitivity) | F1-Score | Status |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 182 | 0.7403 | **93.96%** | 0.8281 | EXCELLENT |
| `FALL_FORWARD` | 320 | 0.5132 | **72.81%** | 0.6021 | GOOD |
| `FALL_FROM_SITTING` | 72 | 0.4068 | **100.00%** | 0.5783 | EXCELLENT |
| `FALL_LEFT` | 424 | 0.9643 | **31.84%** | 0.4787 | ATTENTION |
| `FALL_RIGHT` | 101 | 0.5526 | **20.79%** | 0.3022 | ATTENTION |

### High-Motion Normal Activity (ADL) Specificity

| High-Motion ADL | Test Support | Precision | Recall (Specificity) | F1-Score | False Fall Risk |
|---|---|---|---|---|---|
| `JUMPING` | 83 | 0.9759 | **97.59%** | 0.9759 | LOW |
| `RUNNING` | 174 | 0.7778 | **96.55%** | 0.8615 | LOW |
| `SIT_DOWN` | 42 | 0.5000 | **50.00%** | 0.5000 | MEDIUM |
| `STAND_UP` | 66 | 0.4583 | **33.33%** | 0.3860 | MEDIUM |
| `PICKING_UP_OBJECT` | 18 | 0.0000 | **0.00%** | 0.0000 | MEDIUM |

---
