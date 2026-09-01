# SMARTFALL AI — PHASE 11 HIGH-MOTION FALSE POSITIVE EVALUATION

Evaluation of high-acceleration daily activities (`JUMPING`, `RUNNING`, `SIT_DOWN`, `STAND_UP`, `PICKING_UP_OBJECT`) to guarantee false alarm immunity.

## WATCH High-Motion False Fall Predictions

| Model Family | JUMPING False Falls | RUNNING False Falls | SIT_DOWN False Falls | STAND_UP False Falls | PICKING_UP False Falls | Total False Alarms | False Fall Risk |
|---|---|---|---|---|---|---|---|
| **`Random Forest (Champion)`** | `117` | `18` | `7` | `5` | `10` | **`157`** | ELEVATED |
| **`CNN-BiLSTM Hybrid`** | `0` | `10` | `5` | `6` | `17` | **`38`** | ELEVATED |
| **`1D-CNN`** | `0` | `5` | `3` | `7` | `19` | **`34`** | ELEVATED |
| **`Bi-LSTM`** | `1` | `9` | `17` | `8` | `13` | **`48`** | ELEVATED |

---

## PHONE High-Motion False Fall Predictions

| Model Family | JUMPING False Falls | RUNNING False Falls | SIT_DOWN False Falls | STAND_UP False Falls | PICKING_UP False Falls | Total False Alarms | False Fall Risk |
|---|---|---|---|---|---|---|---|
| **`1D-CNN (Champion)`** | `0` | `5` | `10` | `27` | `13` | **`55`** | ELEVATED |
| **`Gradient Boosting`** | `1` | `7` | `15` | `11` | `5` | **`39`** | ELEVATED |
| **`HistGradientBoosting`** | `0` | `4` | `11` | `12` | `7` | **`34`** | ELEVATED |
| **`CNN-BiLSTM Hybrid`** | `0` | `7` | `16` | `15` | `6` | **`44`** | ELEVATED |
| **`Bi-LSTM`** | `1` | `6` | `12` | `13` | `3` | **`35`** | ELEVATED |

---
