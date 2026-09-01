# SMARTFALL AI — PHASE 10 DEPLOYMENT & ON-DEVICE FEASIBILITY MATRIX

| Model Family | WATCH Deployment Feasibility | PHONE Deployment Feasibility | Offline Capable | Max P95 Latency | Real-Time Suitable (< 1,000 ms) |
|---|---|---|---|---|---|
| **`Random Forest`** | **EXCELLENT** (Native Kotlin Decision Ensemble, < 10 MB binary) | **EXCELLENT** (Fast tree evaluation) | **YES** | **< 1.0 ms** | **YES (> 99.9% budget margin)** |
| **`Extra Trees`** | **EXCELLENT** (Native Kotlin Decision Ensemble) | **EXCELLENT** (Fast tree evaluation) | **YES** | **< 1.0 ms** | **YES (> 99.9% budget margin)** |
| **`HistGradientBoosting`** | **GOOD** (Tree traversal in Kotlin) | **EXCELLENT** (Fast tree evaluation) | **YES** | **< 2.0 ms** | **YES (> 99.8% budget margin)** |
| **`Gradient Boosting`** | **GOOD** (Tree traversal in Kotlin) | **EXCELLENT** (Fast tree evaluation) | **YES** | **< 2.0 ms** | **YES (> 99.8% budget margin)** |
| **`RBF SVM`** | **POOR** (Requires evaluating kernel against 5,000+ support vectors) | **MODERATE** (High memory & compute) | **YES** | **> 15.0 ms** | **YES (Marginal)** |
| **`Logistic Regression`** | **EXCELLENT** (Linear dot product in Kotlin) | **EXCELLENT** (Linear dot product) | **YES** | **< 0.05 ms** | **YES (> 99.99% budget margin)** |
| **`1D-CNN`** | **MODERATE** (Requires lightweight ONNX/TFLite runtime) | **EXCELLENT** (Microsoft ONNX Runtime, 16.3 KB) | **YES** | **< 0.1 ms** | **YES (> 99.99% budget margin)** |
| **`Bi-LSTM`** | **POOR** (Recurrent step-by-step latency, memory intensive) | **GOOD** (ONNX Runtime) | **YES** | **~1.5 ms** | **YES (> 99.8% budget margin)** |
| **`GRU`** | **POOR** (Recurrent step-by-step latency) | **GOOD** (ONNX Runtime) | **YES** | **~1.2 ms** | **YES (> 99.8% budget margin)** |
| **`CNN-BiLSTM Hybrid`** | **POOR** (Heavy multi-layer recurrent overhead) | **GOOD** (ONNX Runtime) | **YES** | **~2.0 ms** | **YES (> 99.8% budget margin)** |
