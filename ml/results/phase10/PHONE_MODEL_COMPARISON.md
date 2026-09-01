# SMARTFALL AI — PHASE 10 PHONE MODEL LEADERBOARD

Ranking priority: 1. Fall Recall (Sensitivity), 2. False Positive Rate (FPR), 3. Binary Fall F1, 4. Macro-F1, 5. Latency & Footprint.

| Rank | Model Family | Preprocessing | Val Fall Recall | Val Binary F1 | Val Macro-F1 | Val FPR | Test Fall Recall | Test Binary F1 | P95 Latency | Size |
|---|---|---|---|---|---|---|---|---|---|---|
| **#1** | **`HistGradientBoosting`** | `02_robust_scaling` | **88.41%** | 0.8475 | 0.5050 | 22.04% | **77.71%** | 0.7944 | 21.46 ms | 5066.9 KB |
| **#2** | **`Gradient Boosting`** | `02_robust_scaling` | **88.35%** | 0.8447 | 0.4880 | 22.70% | **77.98%** | 0.7961 | 0.41 ms | 5242.9 KB |
| **#3** | **`Random Forest`** | `02_robust_scaling` | **85.60%** | 0.8304 | 0.4736 | 22.41% | **74.16%** | 0.7620 | 13.72 ms | 37934.6 KB |
| **#4** | **`Extra Trees`** | `02_robust_scaling` | **85.13%** | 0.8417 | 0.5002 | 18.69% | **71.97%** | 0.7526 | 13.68 ms | 74538.9 KB |
| **#5** | **`Logistic Regression`** | `02_robust_scaling` | **83.99%** | 0.8028 | 0.3979 | 27.52% | **71.43%** | 0.7265 | 0.05 ms | 4.9 KB |
| **#6** | **`RBF SVM`** | `02_robust_scaling` | **82.65%** | 0.7472 | 0.2491 | 42.04% | **80.71%** | 0.7884 | 0.47 ms | 6503.3 KB |
| **#7** | **`CNN-BiLSTM Hybrid`** | `02_robust_scaling` | **79.04%** | 0.8158 | 0.4826 | 16.06% | **68.15%** | 0.7230 | 0.78 ms | 307.0 KB |
| **#8** | **`Bi-LSTM`** | `02_robust_scaling` | **78.97%** | 0.7977 | 0.4959 | 20.73% | **78.89%** | 0.7943 | 2.44 ms | 551.4 KB |
| **#9** | **`GRU`** | `02_robust_scaling` | **78.43%** | 0.8010 | 0.4592 | 18.98% | **75.52%** | 0.7875 | 2.40 ms | 416.8 KB |
| **#10** | **`1D-CNN`** | `02_robust_scaling` | **76.76%** | 0.8039 | 0.4821 | 15.47% | **72.52%** | 0.7623 | 0.20 ms | 161.1 KB |