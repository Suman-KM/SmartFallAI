# SMARTFALL AI — PHASE 10 EXTENDED MULTI-MODEL BENCHMARK REPORT

## 1. Experimental Overview

Phase 10 evaluates 10 diverse machine learning model architectures on 2 independent edge computing platforms:

1. **WATCH (`SM-R870` — Wear OS)**
2. **PHONE (`SM-A507FN` — Android)**

All experiments strictly adhere to the frozen Phase 5 dataset and session split (70% train / 15% validation / 15% test). GPS, heart rate, session IDs, and timestamps are strictly excluded from input tensors $X$.

## 2. Complete Watch Benchmark Results (10 Models)

| Model | Type | Val Fall Recall | Val Binary F1 | Val Macro-F1 | Val FPR | Test Fall Recall | Test Binary F1 | Latency (P95) | Model Size |
|---|---|---|---|---|---|---|---|---|---|
| **`Random Forest`** | Classical (72 Feats) | **79.18%** | 0.7578 | 0.1331 | 14.53% | **84.08%** | 0.6960 | 13.71 ms | 44041.2 KB |
| **`Extra Trees`** | Classical (72 Feats) | **75.67%** | 0.7267 | 0.1305 | 15.89% | **83.86%** | 0.7180 | 13.70 ms | 87845.3 KB |
| **`HistGradientBoosting`** | Classical (72 Feats) | **77.00%** | 0.7378 | 0.1334 | 15.48% | **89.24%** | 0.7085 | 19.99 ms | 4095.1 KB |
| **`Gradient Boosting`** | Classical (72 Feats) | **74.58%** | 0.7142 | 0.1271 | 16.72% | **88.36%** | 0.7413 | 0.40 ms | 4940.0 KB |
| **`RBF SVM`** | Classical (72 Feats) | **69.61%** | 0.5799 | 0.1155 | 34.38% | **81.78%** | 0.6232 | 0.60 ms | 8385.3 KB |
| **`Logistic Regression`** | Classical (72 Feats) | **72.03%** | 0.5822 | 0.1115 | 36.80% | **81.34%** | 0.6369 | 0.05 ms | 4.6 KB |
| **`1D-CNN`** | Deep Temporal (100x9) | **81.48%** | 0.7718 | 0.6264 | 14.47% | **83.42%** | 0.7336 | 0.20 ms | 161.1 KB |
| **`Bi-LSTM`** | Deep Temporal (100x9) | **81.60%** | 0.7778 | 0.5925 | 13.76% | **84.19%** | 0.7371 | 2.48 ms | 551.4 KB |
| **`GRU`** | Deep Temporal (100x9) | **75.67%** | 0.7942 | 0.6057 | 7.27% | **79.03%** | 0.7076 | 2.41 ms | 416.8 KB |
| **`CNN-BiLSTM Hybrid`** | Deep Temporal (100x9) | **84.50%** | 0.7799 | 0.6212 | 15.71% | **92.43%** | 0.7739 | 0.76 ms | 307.0 KB |

## 3. Complete Phone Benchmark Results (10 Models)

| Model | Type | Val Fall Recall | Val Binary F1 | Val Macro-F1 | Val FPR | Test Fall Recall | Test Binary F1 | Latency (P95) | Model Size |
|---|---|---|---|---|---|---|---|---|---|
| **`Random Forest`** | Classical (72 Feats) | **85.60%** | 0.8304 | 0.4736 | 22.41% | **74.16%** | 0.7620 | 13.72 ms | 37934.6 KB |
| **`Extra Trees`** | Classical (72 Feats) | **85.13%** | 0.8417 | 0.5002 | 18.69% | **71.97%** | 0.7526 | 13.68 ms | 74538.9 KB |
| **`HistGradientBoosting`** | Classical (72 Feats) | **88.41%** | 0.8475 | 0.5050 | 22.04% | **77.71%** | 0.7944 | 21.46 ms | 5066.9 KB |
| **`Gradient Boosting`** | Classical (72 Feats) | **88.35%** | 0.8447 | 0.4880 | 22.70% | **77.98%** | 0.7961 | 0.41 ms | 5242.9 KB |
| **`RBF SVM`** | Classical (72 Feats) | **82.65%** | 0.7472 | 0.2491 | 42.04% | **80.71%** | 0.7884 | 0.47 ms | 6503.3 KB |
| **`Logistic Regression`** | Classical (72 Feats) | **83.99%** | 0.8028 | 0.3979 | 27.52% | **71.43%** | 0.7265 | 0.05 ms | 4.9 KB |
| **`1D-CNN`** | Deep Temporal (100x9) | **76.76%** | 0.8039 | 0.4821 | 15.47% | **72.52%** | 0.7623 | 0.20 ms | 161.1 KB |
| **`Bi-LSTM`** | Deep Temporal (100x9) | **78.97%** | 0.7977 | 0.4959 | 20.73% | **78.89%** | 0.7943 | 2.44 ms | 551.4 KB |
| **`GRU`** | Deep Temporal (100x9) | **78.43%** | 0.8010 | 0.4592 | 18.98% | **75.52%** | 0.7875 | 2.40 ms | 416.8 KB |
| **`CNN-BiLSTM Hybrid`** | Deep Temporal (100x9) | **79.04%** | 0.8158 | 0.4826 | 16.06% | **68.15%** | 0.7230 | 0.78 ms | 307.0 KB |