# SMARTFALL AI — FINAL MODEL & PREPROCESSING SELECTION

## Selected Configurations

### 1. WATCH WINNER
- **Device**: Samsung Galaxy Watch 4 (`SM-R870`)
- **Selected Preprocessing**: **P02 — Robust Scaling (`02_robust_scaling`)**
- **Selected Model Architecture**: **Random Forest (`RandomForestClassifier`, 100 estimators, max depth 20)**
- **Selection Basis (Validation Set)**:
  - **Validation Macro-F1**: `0.6158`
  - **Validation Accuracy**: `0.7523`
  - **Validation Fall Recall**: `0.7918`
- **Untouched Final Test Set Performance**:
  - **Test Accuracy**: `0.6729`
  - **Test Macro-F1**: `0.5285`
  - **Test Fall Recall (Sensitivity)**: `0.8408`
  - **Test Fall Precision**: `0.6569`
  - **Test Binary Fall F1**: `0.7376`
- **Operational Metrics**:
  - **Model File Size**: `44041.2 KB`
  - **Inference Latency**: `0.207 ms / window`

---

### 2. PHONE WINNER
- **Device**: Samsung Galaxy A50s (`SM-A507FN`)
- **Selected Preprocessing**: **P02 — Robust Scaling (`02_robust_scaling`)**
- **Selected Model Architecture**: **1D-CNN (3-stage Temporal Convolution)**
- **Selection Basis (Validation Set)**:
  - **Validation Macro-F1**: `0.4929`
  - **Validation Accuracy**: `0.5767`
  - **Validation Fall Recall**: `0.7569`
- **Untouched Final Test Set Performance**:
  - **Test Accuracy**: `0.5182`
  - **Test Macro-F1**: `0.4901`
  - **Test Fall Recall (Sensitivity)**: `0.6342`
  - **Test Fall Precision**: `0.7858`
  - **Test Binary Fall F1**: `0.7019`
- **Operational Metrics**:
  - **Total Parameters**: `38734`
  - **Model File Size**: `158.9 KB`
  - **Inference Latency**: `0.019 ms / window`
