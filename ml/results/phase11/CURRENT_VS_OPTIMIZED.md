# SMARTFALL AI — PHASE 11 CURRENT VS OPTIMIZED MODEL COMPARISON

## 1. WATCH Comparison & Final Decision

### Current Deployed Model:
- **Architecture**: `Random Forest (100 estimators, max_depth=20)`
- **Preprocessing**: `P02 — Robust Scaling`
- **Val Fall Recall**: `79.18%` | **Test Fall Recall**: `84.08%`
- **Test Binary Fall F1**: `0.7376` | **Test FPR**: `1.45%`
- **P95 Latency**: `0.22 ms` | **Model Storage**: `9.98 MB` flat binary (`trees.bin`)
- **Runtime Feasibility**: **100% Native Kotlin, ZERO-GC heap allocation, standalone offline on Wear OS.**

### Candidate Model: `CNN-BiLSTM Hybrid`
- **Architecture**: 1D-CNN Feature Extractor + Bidirectional LSTM Sequence Model
- **Val Fall Recall**: `84.50%` | **Test Fall Recall**: `92.43%`
- **Test Binary Fall F1**: `0.7810` | **Test FPR**: `1.57%`
- **P95 Latency**: `0.76 ms` | **Model Storage**: `307.0 KB`
- **Trade-offs & Analysis**:
  1. While `CNN-BiLSTM` shows a higher test recall on the offline batch test set (+8.35%), it introduces substantial sequential recurrent state management and requires an ONNX/TFLite C++ runtime layer on Wear OS.
  2. The current `Random Forest` tree engine runs natively in pure Kotlin flat primitive arrays with 0.22 ms latency, zero native JNI overhead, and zero memory leak risks on the Samsung Galaxy Watch 4.
  3. In real physical device testing (Phase 9), the deployed Random Forest achieved **100% detection on all 5 physical fall simulations with 0 false alarms**.

### Watch Decision: **KEEP CURRENT (P02 Robust Scaling + Random Forest)**

---

## 2. PHONE Comparison & Final Decision

### Current Deployed Model:
- **Architecture**: `1D-CNN (3-Stage Temporal Convolutional Network)`
- **Preprocessing**: `P02 — Robust Scaling`
- **Val Fall Recall**: `76.76%` | **Test Fall Recall**: `77.19%`
- **Test Binary Fall F1**: `0.7019` | **Test FPR**: `1.55%`
- **P95 Latency**: `0.03 ms` | **Model Storage**: `164.7 KB` self-contained `model.onnx`
- **Runtime Feasibility**: **Microsoft ONNX Runtime Android, 0.03 ms latency (< 0.003% duty cycle).**

### Candidate Model: `Gradient Boosting / HistGradientBoosting`
- **Val Fall Recall**: `88.35%` | **Test Fall Recall**: `77.98%`
- **Test Binary Fall F1**: `0.6910` | **Test FPR**: `2.27%`
- **P95 Latency**: `0.41 ms` | **Model Storage**: `5,242.9 KB`
- **Trade-offs & Analysis**:
  1. Gradient Boosting achieves a comparable test recall (`77.98%` vs `77.19%`), but has a **higher false positive rate (`2.27%` vs `1.55%`)** and a lower Binary Fall F1 (`0.6910` vs `0.7019`).
  2. Gradient Boosting requires evaluating 100 boosted trees sequentially, which requires 32x more storage (5.2 MB vs 164 KB) and 13x higher latency than the 1D-CNN.
  3. 1D-CNN spatial filters naturally model continuous temporal IMU correlations across pocket-worn orientations.

### Phone Decision: **KEEP CURRENT (P02 Robust Scaling + 1D-CNN)**
