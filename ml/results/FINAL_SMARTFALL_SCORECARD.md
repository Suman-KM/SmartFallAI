# SMARTFALL AI — FINAL RESEARCH SCORECARD

| Device | Preprocessing Pipeline | Model Architecture | Test Fall Recall | Binary Fall F1 | Macro-F1 | Specificity | False Positive Rate | P95 Latency | Model Size | Final Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| **WATCH (`SM-R870`)** | `02_robust_scaling` | **`Random Forest (100 Trees)`** | **`84.08%`** | **`0.6960`** | **`0.1112`** | **`69.50%`** | **`30.50%`** (1.45% w/ 2W) | **`0.22 ms`** | **`9.98 MB`** | **`KEEP CURRENT (Champion)`** |
| **PHONE (`SM-A507FN`)** | `02_robust_scaling` | **`1D-CNN (3-Stage ConvNet)`** | **`63.42%`** | **`0.7019`** | **`0.4901`** | **`79.55%`** | **`20.45%`** (1.55% w/ 2W) | **`0.03 ms`** | **`164.7 KB`** | **`KEEP CURRENT (Champion)`** |

---

### WATCH FINAL DECISION: **`KEEP CURRENT MODEL (P02 Robust Scaling + Random Forest)`**
* **Deployment Format**: Pure Kotlin decision tree ensemble (`trees.bin`, zero GC allocation).
* **Physical Validation**: 100% detection across safe controlled physical falls (Phase 9).
* **Duty Cycle**: 0.22 ms P95 inference latency vs 1,000 ms budget (< 0.03% CPU duty cycle).

### PHONE FINAL DECISION: **`KEEP CURRENT MODEL (P02 Robust Scaling + 1D-CNN)`**
* **Deployment Format**: Self-contained ONNX runtime (`model.onnx` via `PhoneOnnxEngine.kt`).
* **Physical Validation**: 100% detection across safe controlled physical falls (Phase 9).
* **Duty Cycle**: 0.03 ms P95 inference latency vs 1,000 ms budget (< 0.003% CPU duty cycle).
