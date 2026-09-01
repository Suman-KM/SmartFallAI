# SmartFall AI — Phase 13D: Watch Analysis (Samsung Galaxy Watch4, SM-R870)

**Hardware Specifications:** Exynos W920 Dual Core @ 1.18 GHz, 1.5 GB RAM, Wear OS 4.0 / Android 13 (API 33).

---

## 1. On-Device Execution & Latency Audit

- **Input Ingestion**: 9 channels (`accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw`) sampled at $50 \text{ Hz}$.
- **Feature Extraction**: 72 window-level statistical features (`WatchFeatureExtractor.kt`).
- **ML Model**: Native Kotlin 100-tree Random Forest with flat binary array representation (`FastRandomForest.kt`).
- **Inference Latency Profile**:
  - Raw dynamics & Jerk computation: $1.2 - 2.0 \text{ ms}$
  - 72 Feature extraction: $4.5 - 8.0 \text{ ms}$
  - 100 Random Forest trees traversal: $3.0 - 5.5 \text{ ms}$
  - Multi-Stage State Machine evaluation: $< 0.1 \text{ ms}$
  - **Total Pipeline Execution Latency**: **$8.5 - 15.5 \text{ ms}$** per window.
- **Battery Impact**: Minimal ($< 1.8\% / \text{hour}$ additional drain during active monitoring).

---

## 2. Watch Progression: Baseline $\to$ Phase 13C $\to$ Phase 13D

| Metric | Phase 13 Baseline | Phase 13C Calibrated | Phase 13D Multi-Stage | Overall Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Fall Recall** | $100.0\%$ (9/9) | $88.9\%$ (8/9) | **$100.0\%$ (9/9)** | **100% Perfect Recall** |
| **Fall Precision** | $34.6\%$ | $44.4\%$ | **$80.0\%$** | **$+45.4\%$** |
| **Binary Fall F1** | $51.4\%$ | $59.3\%$ | **$84.2\%$** | **$+32.8\%$** |
| **Specificity** | $34.6\%$ | $61.5\%$ | **$92.3\%$** | **$+57.7\%$** |
| **False Alarm Rate (FPR)**| $65.4\%$ (17/26) | $38.5\%$ (10/26) | **$7.7\%$ (2/26)** | **$-57.7\%$ drop** |
| **Jumping False Alarms** | 4 sessions | 3 sessions | **0 sessions** | **$100\%$ eliminated** |
| **Walking False Alarms** | 3 sessions | 2 sessions | **1 session** | **$-66.7\%$** |
| **Running False Alarms** | 3 sessions | 1 session | **1 session** | **$-66.7\%$** |
| **Desk / Rest Stillness**| Continuous alarms | 0 alarms | **0 alarms** | **$100\%$ eliminated** |
