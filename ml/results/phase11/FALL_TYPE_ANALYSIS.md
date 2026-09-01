# SMARTFALL AI — PHASE 11 PER-FALL-TYPE SENSITIVITY BREAKDOWN

Evaluation of sensitivity across all 5 fall classes to identify directional patterns and hardest fall dynamics.

## WATCH — `Random Forest (Champion)` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 324 | 0.9180 | **72.53%** | 0.8103 | GOOD |
| `FALL_FORWARD` | 531 | 0.5808 | **83.24%** | 0.6842 | GOOD |
| `FALL_FROM_SITTING` | 0 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |
| `FALL_LEFT` | 0 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |
| `FALL_RIGHT` | 56 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |

---

## WATCH — `CNN-BiLSTM Hybrid` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 324 | 0.8654 | **69.44%** | 0.7705 | CHALLENGING |
| `FALL_FORWARD` | 531 | 0.5583 | **82.11%** | 0.6646 | GOOD |
| `FALL_FROM_SITTING` | 0 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |
| `FALL_LEFT` | 0 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |
| `FALL_RIGHT` | 56 | 0.2125 | **30.36%** | 0.2500 | CHALLENGING |

---

## WATCH — `1D-CNN` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 324 | 0.9210 | **82.72%** | 0.8715 | GOOD |
| `FALL_FORWARD` | 531 | 0.5949 | **82.67%** | 0.6919 | GOOD |
| `FALL_FROM_SITTING` | 0 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |
| `FALL_LEFT` | 0 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |
| `FALL_RIGHT` | 56 | 0.1707 | **25.00%** | 0.2029 | CHALLENGING |

---

## WATCH — `Bi-LSTM` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 324 | 0.8380 | **73.46%** | 0.7829 | GOOD |
| `FALL_FORWARD` | 531 | 0.5992 | **82.49%** | 0.6941 | GOOD |
| `FALL_FROM_SITTING` | 0 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |
| `FALL_LEFT` | 0 | 0.0000 | **0.00%** | 0.0000 | CHALLENGING |
| `FALL_RIGHT` | 56 | 0.1778 | **28.57%** | 0.2192 | CHALLENGING |

---

## PHONE — `1D-CNN (Champion)` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 182 | 0.5578 | **92.86%** | 0.6969 | EXCELLENT |
| `FALL_FORWARD` | 320 | 0.5430 | **75.00%** | 0.6299 | GOOD |
| `FALL_FROM_SITTING` | 72 | 0.2353 | **27.78%** | 0.2548 | CHALLENGING |
| `FALL_LEFT` | 424 | 0.7931 | **5.42%** | 0.1015 | CHALLENGING |
| `FALL_RIGHT` | 101 | 0.3265 | **15.84%** | 0.2133 | CHALLENGING |

---

## PHONE — `Gradient Boosting` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 182 | 0.7443 | **89.56%** | 0.8130 | EXCELLENT |
| `FALL_FORWARD` | 320 | 0.5448 | **68.44%** | 0.6066 | CHALLENGING |
| `FALL_FROM_SITTING` | 72 | 0.2857 | **91.67%** | 0.4356 | EXCELLENT |
| `FALL_LEFT` | 424 | 0.9624 | **30.19%** | 0.4596 | CHALLENGING |
| `FALL_RIGHT` | 101 | 0.5942 | **40.59%** | 0.4824 | CHALLENGING |

---

## PHONE — `HistGradientBoosting` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 182 | 0.7399 | **90.66%** | 0.8148 | EXCELLENT |
| `FALL_FORWARD` | 320 | 0.4965 | **66.88%** | 0.5699 | CHALLENGING |
| `FALL_FROM_SITTING` | 72 | 0.2996 | **98.61%** | 0.4595 | EXCELLENT |
| `FALL_LEFT` | 424 | 0.9752 | **27.83%** | 0.4330 | CHALLENGING |
| `FALL_RIGHT` | 101 | 0.5385 | **20.79%** | 0.3000 | CHALLENGING |

---

## PHONE — `CNN-BiLSTM Hybrid` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 182 | 0.7269 | **90.66%** | 0.8068 | EXCELLENT |
| `FALL_FORWARD` | 320 | 0.3699 | **45.31%** | 0.4073 | CHALLENGING |
| `FALL_FROM_SITTING` | 72 | 0.4487 | **97.22%** | 0.6140 | EXCELLENT |
| `FALL_LEFT` | 424 | 0.8462 | **20.75%** | 0.3333 | CHALLENGING |
| `FALL_RIGHT` | 101 | 0.2471 | **20.79%** | 0.2258 | CHALLENGING |

---

## PHONE — `Bi-LSTM` Per-Fall Breakdown

| Fall Direction / Type | Test Support | Precision | Recall (Sensitivity) | F1-Score | Detection Rating |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | 182 | 0.6478 | **87.91%** | 0.7459 | EXCELLENT |
| `FALL_FORWARD` | 320 | 0.4255 | **43.75%** | 0.4314 | CHALLENGING |
| `FALL_FROM_SITTING` | 72 | 0.3209 | **95.83%** | 0.4808 | EXCELLENT |
| `FALL_LEFT` | 424 | 0.7826 | **16.98%** | 0.2791 | CHALLENGING |
| `FALL_RIGHT` | 101 | 0.5437 | **55.45%** | 0.5490 | CHALLENGING |

---
