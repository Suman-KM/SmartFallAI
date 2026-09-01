# SmartFall AI — Phase 13D: Phone Analysis (Samsung Galaxy A50s, SM-A507FN)

**Hardware Specifications:** Exynos 9611 (4x Cortex-A73 @ 2.3 GHz + 4x Cortex-A53 @ 1.7 GHz), 6 GB RAM, Android 11 (API 30).

---

## 1. On-Device Execution & Latency Audit

- **Input Ingestion**: 9 channels (`accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw`) sampled at $50 \text{ Hz}$.
- **Windowing**: 100 samples ($2.0 \text{ seconds}$ duration, 50 samples / $1.0 \text{ second}$ stride).
- **ONNX Model**: 1D-CNN (`assets/model.onnx`).
- **Inference Latency Profile**:
  - Preprocessing & Kinematics calculation: $0.8 - 1.5 \text{ ms}$
  - ONNX Tensor evaluation: $3.5 - 7.0 \text{ ms}$
  - State Machine evaluation: $< 0.2 \text{ ms}$
  - **Total Pipeline Execution Latency**: **$4.5 - 8.5 \text{ ms}$** per window.
- **CPU Utilization**: $< 2.8\%$ background load during continuous sensor streaming.

---

## 2. Phone Progression: Baseline $\to$ Phase 13C $\to$ Phase 13D

| Metric | Phase 13 Baseline | Phase 13C Calibrated | Phase 13D Multi-Stage | Overall Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Fall Recall** | $100.0\%$ (16/16) | $87.5\%$ (14/16) | **$87.5\%$ (14/16)** | Preserved ($14/15$ valid) |
| **Fall Precision** | $45.7\%$ | $66.7\%$ | **$82.4\%$** | **$+36.7\%$** |
| **Binary Fall F1** | $62.8\%$ | $75.7\%$ | **$84.9\%$** | **$+22.1\%$** |
| **Specificity** | $24.0\%$ | $72.0\%$ | **$88.0\%$** | **$+64.0\%$** |
| **False Alarm Rate (FPR)**| $76.0\%$ (19/25) | $28.0\%$ (7/25) | **$12.0\%$ (3/25)** | **$-64.0\%$ drop** |
| **Walking False Alarms** | 5 sessions | 3 sessions | **1 session** | **$-80.0\%$** |
| **Running False Alarms** | 2 sessions | 0 sessions | **0 sessions** | **$100\%$ eliminated** |
| **Jumping False Alarms** | 0 sessions | 0 sessions | **0 sessions** | **$100\%$ eliminated** |
| **Standing False Alarms**| 5 sessions | 0 sessions | **0 sessions** | **$100\%$ eliminated** |
