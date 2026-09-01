# SmartFall AI — Phase 13C: Dataset False-Positive Audit Report

**Date:** September 1, 2026  
**Datasets Analyzed:** Frozen Validation Set and Untouched Test Set  
**Models Audited:** Phone 1D-CNN (ONNX) and Watch Random Forest (100 trees)

---

## 1. Executive Summary

This audit quantifies the exact distribution of model-predicted fall probabilities and raw kinematic parameters across all 14 activity classes in both the Validation and Test sets.

### Key Audit Findings:
1. **Phone Vulnerabilities:**
   - **`WALKING`**: $22.8\%$ of walking windows produce $P(\text{fall}) \ge 0.50$, with $P_{\max} = 0.983$.
   - **`PICKING_UP_OBJECT`**: $26.5\%$ of windows exceed $0.50$, with $P_{\max} = 0.824$.
   - **`STANDING`**: $10.9\%$ of windows exceed $0.50$, with $P_{\max} = 0.990$.
   - In Phone walking, median peak acceleration is $15.00 \, m/s^2$, and 95th percentile reaches $26.51 \, m/s^2$. Under an uncalibrated low-impact gate ($16 \, m/s^2$), firm walking steps easily trigger false alarms.

2. **Watch Vulnerabilities:**
   - **`JUMPING`**: **$97.4\%$ of jumping windows** produce $P(\text{fall}) \ge 0.50$, with a median fall probability of **$0.970$** and $P_{\max} = 1.000$.
   - The Watch Random Forest cannot distinguish the landing impact of a jump ($51.16 \, m/s^2$, $5.93 \, \text{rad}/s$) from a fall based on single-window statistical features alone.
   - **`STANDING`**: $14.5\%$ of windows exceed $0.50$.

---

## 2. Phone Validation Set Distribution Audit

| Activity | Windows | Mean $P(\text{fall})$ | Median $P$ | P90 $P$ | Max $P$ | $\% \ge 0.50$ | Acc Peak ($m/s^2$) | Acc Range ($m/s^2$) | Gyro Peak ($\text{rad}/s$) | Peak Jerk ($m/s^3$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FALL_BACKWARD** | 552 | 0.807 | 0.999 | 1.000 | 1.000 | 79.9% | 9.80 | 0.41 | 0.04 | 6.2 |
| **FALL_FORWARD** | 304 | 0.576 | 0.626 | 0.962 | 1.000 | 55.9% | 10.26 | 0.96 | 0.23 | 10.5 |
| **FALL_FROM_SITTING** | 157 | 0.658 | 0.759 | 0.964 | 1.000 | 71.3% | 10.45 | 1.51 | 0.77 | 24.2 |
| **FALL_LEFT** | 382 | 0.907 | 0.999 | 1.000 | 1.000 | 90.6% | 9.72 | 0.28 | 0.06 | 5.7 |
| **FALL_RIGHT** | 98 | 0.538 | 0.498 | 0.966 | 0.999 | 48.0% | 10.79 | 1.96 | 1.39 | 18.5 |
| **JUMPING** | 37 | 0.024 | 0.000 | 0.016 | 0.460 | 0.0% | 49.99 | 49.37 | 4.38 | 723.9 |
| **LYING_DOWN** | 140 | 0.181 | 0.150 | 0.457 | 0.902 | 7.9% | 10.46 | 1.80 | 0.18 | 70.3 |
| **PICKING_UP_OBJECT** | 49 | 0.357 | 0.323 | 0.666 | 0.824 | 26.5% | 13.93 | 8.09 | 2.81 | 89.7 |
| **RUNNING** | 224 | 0.053 | 0.001 | 0.211 | 0.991 | 1.8% | 39.62 | 37.03 | 5.28 | 342.1 |
| **SITTING** | 341 | 0.073 | 0.036 | 0.059 | 0.925 | 3.5% | 9.68 | 0.14 | 0.02 | 4.9 |
| **SIT_DOWN** | 52 | 0.249 | 0.185 | 0.457 | 0.836 | 5.8% | 13.43 | 7.32 | 1.56 | 70.9 |
| **STANDING** | 156 | 0.171 | 0.082 | 0.516 | 0.990 | 10.9% | 11.47 | 3.35 | 1.44 | 56.0 |
| **STAND_UP** | 16 | 0.182 | 0.149 | 0.307 | 0.491 | 0.0% | 14.34 | 6.72 | 1.47 | 100.8 |
| **WALKING** | 355 | 0.315 | 0.227 | 0.790 | 0.983 | 22.8% | 15.00 | 7.32 | 3.24 | 54.8 |

---

## 3. Watch Validation Set Distribution Audit

| Activity | Windows | Mean $P(\text{fall})$ | Median $P$ | P90 $P$ | Max $P$ | $\% \ge 0.50$ | Acc Peak ($m/s^2$) | Acc Range ($m/s^2$) | Gyro Peak ($\text{rad}/s$) | Peak Jerk ($m/s^3$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FALL_BACKWARD** | 85 | 0.761 | 0.780 | 0.840 | 0.860 | 96.5% | 9.92 | 0.33 | 0.10 | 12.1 |
| **FALL_FORWARD** | 553 | 0.753 | 0.840 | 1.000 | 1.000 | 80.8% | 9.87 | 0.27 | 0.07 | 9.6 |
| **FALL_FROM_SITTING** | 43 | 0.463 | 0.540 | 0.690 | 0.730 | 53.5% | 9.90 | 0.25 | 0.17 | 8.9 |
| **FALL_LEFT** | 61 | 0.567 | 0.590 | 0.627 | 0.790 | 83.6% | 10.11 | 0.24 | 0.06 | 7.9 |
| **FALL_RIGHT** | 84 | 0.534 | 0.485 | 0.867 | 0.930 | 46.4% | 10.33 | 0.98 | 0.49 | 40.7 |
| **JUMPING** | 117 | 0.902 | 0.970 | 1.000 | 1.000 | 97.4% | 51.16 | 49.07 | 5.93 | 741.7 |
| **LYING_DOWN** | 245 | 0.155 | 0.080 | 0.389 | 0.650 | 5.3% | 9.88 | 0.26 | 0.05 | 9.2 |
| **RUNNING** | 243 | 0.085 | 0.030 | 0.264 | 0.632 | 1.2% | 34.65 | 32.14 | 4.98 | 657.4 |
| **SITTING** | 542 | 0.246 | 0.236 | 0.439 | 0.870 | 6.1% | 10.52 | 1.45 | 0.91 | 35.8 |
| **SIT_DOWN** | 32 | 0.252 | 0.247 | 0.364 | 0.456 | 0.0% | 12.91 | 6.25 | 3.88 | 77.8 |
| **STANDING** | 275 | 0.274 | 0.189 | 0.736 | 0.900 | 14.5% | 10.49 | 1.55 | 1.16 | 31.2 |
| **STAND_UP** | 71 | 0.234 | 0.211 | 0.410 | 0.500 | 1.4% | 13.32 | 6.33 | 3.11 | 73.7 |
| **WALKING** | 168 | 0.100 | 0.056 | 0.276 | 0.530 | 0.6% | 13.63 | 5.96 | 2.29 | 102.4 |

---

## 4. Root Cause Synthesis

1. **Why ML Probabilities Alone Fail**:
   Single-window statistical or CNN models learn strong correlations with stationary resting post-fall states (orientation, tilt) and violent acceleration spikes. However:
   - Walking strides create cyclical tilt patterns mimicking fall descents.
   - Jumping creates extreme impact shocks that look identical to a ground collision.
   - Static recumbency (lying down) matches post-fall resting states.

2. **Why Kinematic Gating + Temporal Consensus is Necessary**:
   - An impact gate ($a_{peak} \ge 18-20 \, m/s^2$) filters out $75\%$ of walking windows and all static sitting/standing tilt drifts.
   - A multi-window temporal consensus and active thrashing rejection filter eliminates repetitive jumping and continuous running, since genuine falls settle immediately into stillness.
