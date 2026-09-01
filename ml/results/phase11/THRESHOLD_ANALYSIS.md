# SMARTFALL AI — PHASE 11 VALIDATION THRESHOLD OPTIMIZATION REPORT

Evaluation of probability classification thresholds $\theta \in [0.30, 0.80]$ on **VALIDATION SET ONLY** to identify the optimal safety-to-false-alarm trade-off.

## WATCH — `Random Forest (Champion)` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **91.77%** | 0.6153 | 0.7366 | 72.00% | 28.00% |
| $\theta = 0.40$ | **84.87%** | 0.6989 | 0.7665 | 82.16% | 17.84% |
| $\theta = 0.50$ | **77.72%** | 0.7580 | 0.7675 | 87.89% | 12.11% |
| $\theta = 0.60$ | **67.68%** | 0.7775 | 0.7236 | 90.55% | 9.45% |
| $\theta = 0.70$ | **56.17%** | 0.7720 | 0.6503 | 91.91% | 8.09% |
| $\theta = 0.80$ | **44.92%** | 0.7794 | 0.5699 | 93.80% | 6.20% |

---

## WATCH — `CNN-BiLSTM Hybrid` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **84.87%** | 0.6479 | 0.7348 | 77.50% | 22.50% |
| $\theta = 0.40$ | **81.23%** | 0.6918 | 0.7472 | 82.34% | 17.66% |
| $\theta = 0.50$ | **76.51%** | 0.7215 | 0.7427 | 85.59% | 14.41% |
| $\theta = 0.60$ | **69.85%** | 0.7168 | 0.7075 | 86.53% | 13.47% |
| $\theta = 0.70$ | **65.50%** | 0.7213 | 0.6865 | 87.66% | 12.34% |
| $\theta = 0.80$ | **62.95%** | 0.7334 | 0.6775 | 88.84% | 11.16% |

---

## WATCH — `1D-CNN` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **82.81%** | 0.6634 | 0.7367 | 79.50% | 20.50% |
| $\theta = 0.40$ | **79.06%** | 0.7239 | 0.7558 | 85.29% | 14.71% |
| $\theta = 0.50$ | **75.18%** | 0.7573 | 0.7546 | 88.25% | 11.75% |
| $\theta = 0.60$ | **72.28%** | 0.7960 | 0.7576 | 90.96% | 9.04% |
| $\theta = 0.70$ | **69.49%** | 0.8858 | 0.7788 | 95.63% | 4.37% |
| $\theta = 0.80$ | **66.59%** | 0.9615 | 0.7868 | 98.70% | 1.30% |

---

## WATCH — `Bi-LSTM` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **86.80%** | 0.7660 | 0.8138 | 87.06% | 12.94% |
| $\theta = 0.40$ | **83.66%** | 0.8101 | 0.8231 | 90.43% | 9.57% |
| $\theta = 0.50$ | **80.87%** | 0.8434 | 0.8257 | 92.68% | 7.32% |
| $\theta = 0.60$ | **77.36%** | 0.8670 | 0.8177 | 94.21% | 5.79% |
| $\theta = 0.70$ | **73.97%** | 0.8855 | 0.8061 | 95.33% | 4.67% |
| $\theta = 0.80$ | **72.28%** | 0.9059 | 0.8040 | 96.34% | 3.66% |

---

## PHONE — `1D-CNN (Champion)` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **90.22%** | 0.7850 | 0.8395 | 73.07% | 26.93% |
| $\theta = 0.40$ | **85.53%** | 0.8374 | 0.8463 | 81.90% | 18.10% |
| $\theta = 0.50$ | **80.58%** | 0.8730 | 0.8380 | 87.23% | 12.77% |
| $\theta = 0.60$ | **75.15%** | 0.9005 | 0.8193 | 90.95% | 9.05% |
| $\theta = 0.70$ | **69.99%** | 0.9191 | 0.7947 | 93.28% | 6.72% |
| $\theta = 0.80$ | **66.18%** | 0.9410 | 0.7770 | 95.47% | 4.53% |

---

## PHONE — `Gradient Boosting` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **93.30%** | 0.7558 | 0.8351 | 67.15% | 32.85% |
| $\theta = 0.40$ | **91.16%** | 0.7872 | 0.8448 | 73.14% | 26.86% |
| $\theta = 0.50$ | **88.48%** | 0.8195 | 0.8509 | 78.76% | 21.24% |
| $\theta = 0.60$ | **85.13%** | 0.8445 | 0.8479 | 82.92% | 17.08% |
| $\theta = 0.70$ | **81.71%** | 0.8733 | 0.8443 | 87.08% | 12.92% |
| $\theta = 0.80$ | **76.76%** | 0.9205 | 0.8371 | 92.77% | 7.23% |

---

## PHONE — `HistGradientBoosting` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **91.69%** | 0.7868 | 0.8469 | 72.92% | 27.08% |
| $\theta = 0.40$ | **90.15%** | 0.8021 | 0.8489 | 75.77% | 24.23% |
| $\theta = 0.50$ | **88.55%** | 0.8166 | 0.8496 | 78.32% | 21.68% |
| $\theta = 0.60$ | **86.27%** | 0.8267 | 0.8443 | 80.29% | 19.71% |
| $\theta = 0.70$ | **84.59%** | 0.8409 | 0.8434 | 82.55% | 17.45% |
| $\theta = 0.80$ | **82.72%** | 0.8576 | 0.8421 | 85.04% | 14.96% |

---

## PHONE — `CNN-BiLSTM Hybrid` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **86.34%** | 0.7793 | 0.8192 | 73.36% | 26.64% |
| $\theta = 0.40$ | **84.46%** | 0.8130 | 0.8285 | 78.83% | 21.17% |
| $\theta = 0.50$ | **82.59%** | 0.8393 | 0.8325 | 82.77% | 17.23% |
| $\theta = 0.60$ | **80.31%** | 0.8583 | 0.8298 | 85.55% | 14.45% |
| $\theta = 0.70$ | **76.89%** | 0.8783 | 0.8200 | 88.39% | 11.61% |
| $\theta = 0.80$ | **72.20%** | 0.9423 | 0.8176 | 95.18% | 4.82% |

---

## PHONE — `Bi-LSTM` Threshold Response

| Threshold ($\theta$) | Fall Recall (Sensitivity) | Fall Precision | Binary Fall F1 | Specificity | False Positive Rate (FPR) |
|---|---|---|---|---|---|
| $\theta = 0.30$ | **79.50%** | 0.7945 | 0.7948 | 77.59% | 22.41% |
| $\theta = 0.40$ | **77.90%** | 0.8207 | 0.7993 | 81.46% | 18.54% |
| $\theta = 0.50$ | **76.36%** | 0.8382 | 0.7992 | 83.94% | 16.06% |
| $\theta = 0.60$ | **74.82%** | 0.8553 | 0.7981 | 86.20% | 13.80% |
| $\theta = 0.70$ | **70.60%** | 0.8675 | 0.7784 | 88.25% | 11.75% |
| $\theta = 0.80$ | **68.12%** | 0.8836 | 0.7693 | 90.22% | 9.78% |

---
