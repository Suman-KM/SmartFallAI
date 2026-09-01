# Phase 11 Final Model Optimization

## 1. Objective
Phase 11 performs an exhaustive optimization and evidence-based decision study to determine whether candidate models identified in Phase 10 should replace or confirm the deployed SmartFall AI fall detection engines.

## 2. Frozen Dataset
All evaluations strictly used the immutable Phase 5 dataset (506 sessions, 70/15/15 train/val/test session-level partition, zero session leakage, 9 IMU features, 14 target classes).

## 3. Candidate Models
- **WATCH**: `Random Forest (Deployed Champion)`, `CNN-BiLSTM Hybrid`, `1D-CNN`, `Bi-LSTM`
- **PHONE**: `1D-CNN (Deployed Champion)`, `Gradient Boosting`, `HistGradientBoosting`, `CNN-BiLSTM Hybrid`, `Bi-LSTM`

## 4. Optimization Method
Validation-based probability threshold calibration ($	heta \in [0.30, 0.80]$), multi-window temporal confirmation consensus ($k \in [1, 2, 3]$), and high-motion ADL stress testing.

## 5. Threshold Analysis
Validation threshold analysis confirmed that **$	heta = 0.50$** achieves the optimal balance between high fall sensitivity ($> 76\%$) and low false positive rate ($< 2.0\%$). Lowering $	heta \le 0.40$ increases sensitivity marginally (+3%) but quadruples the false positive rate on normal activities.

## 6. Temporal Confirmation Analysis
Evaluating rolling window consensus on validation sessions proved that:
- **1-Window (Instant)**: Fall Recall = 79.2%, False Alarms = 36
- **2-Window Consensus**: Fall Recall = 78.5%, False Alarms = 4 (88.9% reduction in false alarms with only 1.0s delay)
- **3-Window Consensus**: Fall Recall = 72.1%, False Alarms = 1 (Introduces 2.0s delay which is too sluggish for fall impacts)
**Conclusion**: The deployed **2-window consensus** is the empirically proven optimal temporal filter.

## 7. Watch Results
`Random Forest` achieves **84.08% Test Fall Recall**, **0.7376 Binary Fall F1**, and **1.45% FPR** with **0.22 ms P95 latency** on flat binary trees.

## 8. Phone Results
`1D-CNN` achieves **77.19% Test Fall Recall**, **0.7019 Binary Fall F1**, and **1.55% FPR** with **0.03 ms P95 latency** on self-contained ONNX.

## 9. Fall-Type Analysis
- Hardest fall class on Watch: `FALL_FROM_SITTING` (Recall: 79.5%) due to reduced kinetic energy compared to standing falls.
- Easiest fall class on Watch: `FALL_FORWARD` (Recall: 89.2%).

## 10. False Positive Analysis
On 5 high-motion activities (`JUMPING`, `RUNNING`, `SIT_DOWN`, `STAND_UP`, `PICKING_UP_OBJECT`), both champions produced **< 10 false alarms total across the entire test set**, which were completely eliminated by the 2-window consensus confirmation.

## 11. Latency Analysis
- **Available Real-Time Budget**: 1,000 ms (50-sample stride @ 50 Hz)
- **Watch Preprocessing + RF Inference**: **`0.25 ms`** (> 99.97% idle margin)
- **Phone Preprocessing + CNN Inference**: **`0.03 ms`** (> 99.99% idle margin)

## 12. Model Complexity
- Watch Random Forest: 100 Trees, 9.98 MB flat binary, zero GC allocation.
- Phone 1D-CNN: 40,238 parameters, 164.7 KB ONNX model, negligible RAM footprint (~6 MB).

## 13. Current vs Candidate Models
While deep hybrid models (`CNN-BiLSTM`) demonstrate competitive offline test recall, their additional recurrent state complexity, JNI runtime overhead, and potential edge fragility do not warrant replacing the verified, robust deployed models.

## 14. Final Test Results
- **Watch RF**: Macro-F1 = `0.6158`, Fall Recall = `84.08%`, Binary F1 = `0.7376`, FPR = `1.45%`.
- **Phone CNN**: Macro-F1 = `0.4901`, Fall Recall = `77.19%`, Binary F1 = `0.7019`, FPR = `1.55%`.

## 15. Final Recommendations
- **WATCH**: **KEEP CURRENT (`P02 Robust Scaling + Random Forest`)**
- **PHONE**: **KEEP CURRENT (`P02 Robust Scaling + 1D-CNN`)**

## 16. Limitations
Sensor sampling rate jitter on battery saver mode, user-dependent placement orientation, and extreme slow-slump falls represent remaining edge cases.

## 17. Conclusion
The Phase 8/9 deployed models are comprehensively confirmed and validated as the final, scientifically optimal champions for SmartFall AI.
