# SMARTFALL AI — PHASE 10 WATCH MODEL LEADERBOARD

Ranking priority: 1. Fall Recall (Sensitivity), 2. False Positive Rate (FPR), 3. Binary Fall F1, 4. Macro-F1, 5. Latency & Footprint.

| Rank | Model Family | Preprocessing | Val Fall Recall | Val Binary F1 | Val Macro-F1 | Val FPR | Test Fall Recall | Test Binary F1 | P95 Latency | Size |
|---|---|---|---|---|---|---|---|---|---|---|
| **#1** | **`CNN-BiLSTM Hybrid`** | `02_robust_scaling` | **84.50%** | 0.7799 | 0.6212 | 15.71% | **92.43%** | 0.7739 | 0.76 ms | 307.0 KB |
| **#2** | **`Bi-LSTM`** | `02_robust_scaling` | **81.60%** | 0.7778 | 0.5925 | 13.76% | **84.19%** | 0.7371 | 2.48 ms | 551.4 KB |
| **#3** | **`1D-CNN`** | `02_robust_scaling` | **81.48%** | 0.7718 | 0.6264 | 14.47% | **83.42%** | 0.7336 | 0.20 ms | 161.1 KB |
| **#4** | **`Random Forest`** | `02_robust_scaling` | **79.18%** | 0.7578 | 0.1331 | 14.53% | **84.08%** | 0.6960 | 13.71 ms | 44041.2 KB |
| **#5** | **`HistGradientBoosting`** | `02_robust_scaling` | **77.00%** | 0.7378 | 0.1334 | 15.48% | **89.24%** | 0.7085 | 19.99 ms | 4095.1 KB |
| **#6** | **`GRU`** | `02_robust_scaling` | **75.67%** | 0.7942 | 0.6057 | 7.27% | **79.03%** | 0.7076 | 2.41 ms | 416.8 KB |
| **#7** | **`Extra Trees`** | `02_robust_scaling` | **75.67%** | 0.7267 | 0.1305 | 15.89% | **83.86%** | 0.7180 | 13.70 ms | 87845.3 KB |
| **#8** | **`Gradient Boosting`** | `02_robust_scaling` | **74.58%** | 0.7142 | 0.1271 | 16.72% | **88.36%** | 0.7413 | 0.40 ms | 4940.0 KB |
| **#9** | **`Logistic Regression`** | `02_robust_scaling` | **72.03%** | 0.5822 | 0.1115 | 36.80% | **81.34%** | 0.6369 | 0.05 ms | 4.6 KB |
| **#10** | **`RBF SVM`** | `02_robust_scaling` | **69.61%** | 0.5799 | 0.1155 | 34.38% | **81.78%** | 0.6232 | 0.60 ms | 8385.3 KB |