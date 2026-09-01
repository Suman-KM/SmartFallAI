# SMARTFALL AI — PHASE 11 FINAL MODEL OPTIMIZATION SCORECARD

| Device | Model Family | Preprocessing | Val Fall Recall | Test Fall Recall | Fall Precision | Binary Fall F1 | Specificity | FPR | Macro-F1 | P95 Latency | Size | Deployment Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **WATCH** | **`Random Forest (Champion)`** | `02_robust_scaling` | 79.18% | **84.08%** | 0.5938 | **0.6960** | 69.50% | **30.50%** | 0.1112 | 13.68 ms | 44041.2 KB | **RETAIN (Champion)** |
| **WATCH** | **`CNN-BiLSTM Hybrid`** | `02_robust_scaling` | 78.09% | **88.04%** | 0.6390 | **0.7405** | 73.63% | **26.37%** | 0.4934 | 0.74 ms | 307.0 KB | **Candidate** |
| **WATCH** | **`1D-CNN`** | `02_robust_scaling` | 76.63% | **89.57%** | 0.6689 | **0.7658** | 76.48% | **23.52%** | 0.5305 | 0.20 ms | 161.1 KB | **Candidate** |
| **WATCH** | **`Bi-LSTM`** | `02_robust_scaling` | 82.20% | **82.33%** | 0.6345 | **0.7167** | 74.85% | **25.15%** | 0.4904 | 2.40 ms | 551.4 KB | **Candidate** |
| **PHONE** | **`1D-CNN (Champion)`** | `02_robust_scaling` | 80.31% | **64.97%** | 0.7863 | **0.7115** | 79.12% | **20.88%** | 0.4181 | 0.20 ms | 161.5 KB | **RETAIN (Champion)** |
| **PHONE** | **`Gradient Boosting`** | `02_robust_scaling` | 88.35% | **77.98%** | 0.8131 | **0.7961** | 78.79% | **21.21%** | 0.4643 | 0.42 ms | 5242.9 KB | **Candidate** |
| **PHONE** | **`HistGradientBoosting`** | `02_robust_scaling` | 88.41% | **77.71%** | 0.8126 | **0.7944** | 78.79% | **21.21%** | 0.4933 | 20.78 ms | 5066.9 KB | **Candidate** |
| **PHONE** | **`CNN-BiLSTM Hybrid`** | `02_robust_scaling` | 82.72% | **63.97%** | 0.7293 | **0.6815** | 71.91% | **28.09%** | 0.4546 | 0.74 ms | 307.0 KB | **Candidate** |
| **PHONE** | **`Bi-LSTM`** | `02_robust_scaling` | 76.89% | **71.88%** | 0.8012 | **0.7578** | 78.90% | **21.10%** | 0.4491 | 2.44 ms | 551.4 KB | **Candidate** |