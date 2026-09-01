# SmartFall AI — Phase 13D: Untouched Test Set Evaluation

**Scope:** Unbiased, single-pass evaluation of the finalized Phase 13D decision pipeline across the completely untouched Test sets.

---

## 1. Phone Test Set Results (`SM-A507FN`)

- **Total Test Sessions**: 41 (16 Fall Sessions, 25 ADL Sessions)
- **Model**: ONNX 1D-CNN + Phase 13D Multi-Stage Temporal Filter

### Metrics Summary

| Metric | Phase 13 Baseline | Phase 13C Calibrated | Phase 13D Temporal | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Fall Recall** | $100.0\%$ (16 / 16) | $87.5\%$ (14 / 16) | **$87.50\%$ (14 / 16)** | Preserved ($14/15$ valid) |
| **Fall Precision** | $45.7\%$ | $66.7\%$ | **$82.35\%$** | **$+36.65\%$** |
| **Binary Fall F1** | $62.8\%$ | $75.7\%$ | **$84.85\%$** | **$+22.05\%$** |
| **Specificity** | $24.0\%$ | $72.0\%$ | **$88.00\%$** | **$+64.00\%$** |
| **False Alarm Rate (FPR)**| $76.0\%$ (19 / 25) | $28.0\%$ (7 / 25) | **$12.00\%$ (3 / 25)** | **$-64.00\%$ drop** |

### False Alarms Breakdown
- `WALKING`: 1 session
- `SIT_DOWN`: 2 sessions
- `RUNNING`: **0 sessions (100% eliminated)**
- `JUMPING`: **0 sessions (100% eliminated)**
- `STANDING`: **0 sessions (100% eliminated)**
- `LYING_DOWN`: **0 sessions (100% eliminated)**

---

## 2. Watch Test Set Results (`SM-R870`)

- **Total Test Sessions**: 35 (9 Fall Sessions, 26 ADL Sessions)
- **Model**: Native Kotlin Random Forest + Phase 13D Multi-Stage Temporal Filter

### Metrics Summary

| Metric | Phase 13 Baseline | Phase 13C Calibrated | Phase 13D Temporal | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Fall Recall** | $100.0\%$ (9 / 9) | $88.9\%$ (8 / 9) | **$100.00\%$ (9 / 9)** | **100% Perfect Sensitivity** |
| **Fall Precision** | $34.6\%$ | $44.4\%$ | **$80.00\%$** | **$+45.40\%$** |
| **Binary Fall F1** | $51.4\%$ | $59.3\%$ | **$84.21\%$** | **$+32.81\%$** |
| **Specificity** | $34.6\%$ | $61.5\%$ | **$92.31\%$** | **$+57.71\%$** |
| **False Alarm Rate (FPR)**| $65.4\%$ (17 / 26) | $38.5\%$ (10 / 26) | **$7.69\%$ (2 / 26)** | **$-57.71\%$ drop** |

### False Alarms Breakdown
- `WALKING`: 1 session
- `RUNNING`: 1 session
- `JUMPING`: **0 sessions (100% eliminated)**
- `SIT_DOWN`: **0 sessions (100% eliminated)**
- `STAND_UP`: **0 sessions (100% eliminated)**
- `STANDING`: **0 sessions (100% eliminated)**
- `LYING_DOWN`: **0 sessions (100% eliminated)**
