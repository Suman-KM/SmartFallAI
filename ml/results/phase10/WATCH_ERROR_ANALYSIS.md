# SMARTFALL AI — PHASE 10 WATCH ERROR & CONFUSION ANALYSIS

Detailed evaluation of the Top 3 fall-detection models on WATCH to understand missed falls, high-motion ADL false alarms, and class confusions.

## 1. `CNN-BiLSTM Hybrid` Error Profile

- **Test Fall Recall**: **`92.43%`** (Missed: `69` / `911` falls)
- **Test False Positive Rate (FPR)**: **`24.62%`** (False Alarms: `423` / `1718` ADLs)
- **Binary Fall F1**: **`0.7739`**

### Per-Class Fall Detection Sensitivity

| Fall Class | Test Support | Precision | Recall (Sensitivity) | F1-Score | Status |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 324 | 0.8508 | **77.47%** | 0.8110 | GOOD |
| `FALL_FORWARD` | 531 | 0.5876 | **83.99%** | 0.6915 | GOOD |
| `FALL_FROM_SITTING` | 0 | 0.0000 | **0.00%** | 0.0000 | ATTENTION |
| `FALL_LEFT` | 0 | 0.0000 | **0.00%** | 0.0000 | ATTENTION |
| `FALL_RIGHT` | 56 | 0.1932 | **30.36%** | 0.2361 | ATTENTION |

### High-Motion Normal Activity (ADL) Specificity

| High-Motion ADL | Test Support | Precision | Recall (Specificity) | F1-Score | False Fall Risk |
|---|---|---|---|---|---|
| `JUMPING` | 123 | 0.9435 | **95.12%** | 0.9474 | LOW |
| `RUNNING` | 292 | 0.9275 | **83.22%** | 0.8773 | LOW |
| `SIT_DOWN` | 43 | 0.5556 | **23.26%** | 0.3279 | MEDIUM |
| `STAND_UP` | 32 | 0.4375 | **65.62%** | 0.5250 | MEDIUM |
| `PICKING_UP_OBJECT` | 46 | 0.0909 | **2.17%** | 0.0351 | MEDIUM |

---

## 2. `Bi-LSTM` Error Profile

- **Test Fall Recall**: **`84.19%`** (Missed: `144` / `911` falls)
- **Test False Positive Rate (FPR)**: **`23.46%`** (False Alarms: `403` / `1718` ADLs)
- **Binary Fall F1**: **`0.7371`**

### Per-Class Fall Detection Sensitivity

| Fall Class | Test Support | Precision | Recall (Sensitivity) | F1-Score | Status |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 324 | 0.8996 | **71.91%** | 0.7993 | GOOD |
| `FALL_FORWARD` | 531 | 0.5973 | **82.11%** | 0.6915 | GOOD |
| `FALL_FROM_SITTING` | 0 | 0.0000 | **0.00%** | 0.0000 | ATTENTION |
| `FALL_LEFT` | 0 | 0.0000 | **0.00%** | 0.0000 | ATTENTION |
| `FALL_RIGHT` | 56 | 0.1918 | **25.00%** | 0.2171 | ATTENTION |

### High-Motion Normal Activity (ADL) Specificity

| High-Motion ADL | Test Support | Precision | Recall (Specificity) | F1-Score | False Fall Risk |
|---|---|---|---|---|---|
| `JUMPING` | 123 | 0.9000 | **87.80%** | 0.8889 | LOW |
| `RUNNING` | 292 | 0.8935 | **80.48%** | 0.8468 | LOW |
| `SIT_DOWN` | 43 | 0.5263 | **23.26%** | 0.3226 | MEDIUM |
| `STAND_UP` | 32 | 0.2857 | **62.50%** | 0.3922 | MEDIUM |
| `PICKING_UP_OBJECT` | 46 | 0.2000 | **2.17%** | 0.0392 | MEDIUM |

---

## 3. `1D-CNN` Error Profile

- **Test Fall Recall**: **`83.42%`** (Missed: `151` / `911` falls)
- **Test False Positive Rate (FPR)**: **`23.34%`** (False Alarms: `401` / `1718` ADLs)
- **Binary Fall F1**: **`0.7336`**

### Per-Class Fall Detection Sensitivity

| Fall Class | Test Support | Precision | Recall (Sensitivity) | F1-Score | Status |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 324 | 0.8508 | **77.47%** | 0.8110 | GOOD |
| `FALL_FORWARD` | 531 | 0.5965 | **83.24%** | 0.6950 | GOOD |
| `FALL_FROM_SITTING` | 0 | 0.0000 | **0.00%** | 0.0000 | ATTENTION |
| `FALL_LEFT` | 0 | 0.0000 | **0.00%** | 0.0000 | ATTENTION |
| `FALL_RIGHT` | 56 | 0.1351 | **17.86%** | 0.1538 | ATTENTION |

### High-Motion Normal Activity (ADL) Specificity

| High-Motion ADL | Test Support | Precision | Recall (Specificity) | F1-Score | False Fall Risk |
|---|---|---|---|---|---|
| `JUMPING` | 123 | 0.9748 | **94.31%** | 0.9587 | LOW |
| `RUNNING` | 292 | 0.9496 | **83.90%** | 0.8909 | LOW |
| `SIT_DOWN` | 43 | 1.0000 | **2.33%** | 0.0455 | MEDIUM |
| `STAND_UP` | 32 | 0.3600 | **56.25%** | 0.4390 | MEDIUM |
| `PICKING_UP_OBJECT` | 46 | 0.0000 | **0.00%** | 0.0000 | MEDIUM |

---
