# SmartFall AI — Phase 13C: Calibration Specification

**Document Version:** 1.0  
**Target Hardware:**
- Phone: Samsung Galaxy A50s (`SM-A507FN`, Exynos 9611, Android 11, API 30)
- Watch: Samsung Galaxy Watch4 (`SM-R870`, Exynos W920, Wear OS 4.0 / Android 13, API 33)

---

## 1. Specification Overview

This document specifies the exact calibrated thresholds and state machine rules deployed in production on Phone and Watch. All parameters were derived from quantitative validation set distributions and audited on physical hardware.

---

## 2. Phone (`SM-A507FN`) Production Calibration

### 2.1 Sensor & Ingestion Pipeline
- **Sampling Frequency**: $50 \text{ Hz}$ ($\Delta t = 20 \text{ ms}$).
- **Window Size**: 100 samples ($2.0 \text{ seconds}$).
- **Sliding Step Size**: 50 samples ($1.0 \text{ second}$ stride, $50\%$ overlap).
- **Channels (9)**: `accX, accY, accZ` ($m/s^2$), `gyroX, gyroY, gyroZ` ($\text{rad}/s$), `pitch, roll, yaw` (degrees).
- **Model**: ONNX 1D-CNN (`assets/model.onnx`).
- **Scaler**: `assets/scaler.json` (Median / IQR robust scaling in place).

### 2.2 Decision Engine Thresholds
```kotlin
// In FallInferenceEngine.kt (Phone)

// 1. Kinematic Impact Shock Gate:
val hasImpact = (maxAccMag >= 18.0f) || (accRange >= 10.0f && maxGyroMag >= 2.5f)

// 2. Impact Memory Decay:
if (hasImpact) {
    recentImpactCountdown = 3 // 3 windows (~1.5s - 3s)
} else if (recentImpactCountdown > 0) {
    recentImpactCountdown--
}

// 3. Continuous Active Motion (Running) Thrashing Rejection:
val isContinuousThrashing = (accStd >= 4.5f) && (maxGyroMag >= 3.5f)

// 4. Fall Candidate Condition:
val isKinematicFallCandidate = (fallProb >= 0.45f) && (hasImpact || recentImpactCountdown > 0) && (!isContinuousThrashing)

// 5. Consensus Requirement:
// 2 consecutive candidate windows required to transition MONITORING -> FALL_SUSPECTED
```

### 2.3 Parameter Derivation & Justification
- **$a_{peak} \ge 18.0 \, m/s^2$ ($1.83g$)**: Walking $a_{peak}$ 90th percentile is $16.5 \, m/s^2$. The $18.0 \, m/s^2$ threshold filters out $92\%$ of walking transients while safely capturing the minimum phone fall impact ($21.19 \, m/s^2$).
- **$P_{th} = 0.45$**: Preserves lateral falls (`FALL_RIGHT`, median $0.498$) and sitting falls without reducing recall.
- **Continuous Thrashing ($\sigma_a \ge 4.5 \, m/s^2 \land \omega \ge 3.5 \, \text{rad}/s$)**: In running, $\sigma_a = 9.52 \, m/s^2$ and $\omega = 5.28 \, \text{rad}/s$. In falls, post-impact $\sigma_a < 0.1 \, m/s^2$.

---

## 3. Watch (`SM-R870`) Production Calibration

### 3.1 Sensor & Feature Pipeline
- **Sampling Frequency**: $50 \text{ Hz}$. Window: 100 samples, Step: 50 samples.
- **Model**: Native Kotlin Random Forest (100 trees, maximum depth 20).
- **Features**: 72 statistical window features (mean, std, min, max, range, median, rms, energy).

### 3.2 Decision Engine Thresholds
```kotlin
// In FallInferenceEngine.kt (Wear)

// 1. Kinematic Impact Shock Gate:
val hasImpact = (maxAccMag >= 20.0f) || (accRange >= 12.0f && maxGyroMag >= 3.0f)

// 2. Impact Memory Decay:
if (hasImpact) {
    recentImpactCountdown = 3
} else if (recentImpactCountdown > 0) {
    recentImpactCountdown--
}

// 3. Continuous Repetitive Jumping Rejection:
val isContinuousThrashing = (accStd >= 9.0f) && (maxGyroMag >= 4.0f)

// 4. Fall Candidate Condition:
val isKinematicFallCandidate = (fallProb >= 0.45f) && (hasImpact || recentImpactCountdown > 0) && (!isContinuousThrashing)

// 5. Consensus Requirement:
// 2 consecutive candidate windows required to transition MONITORING -> FALL_SUSPECTED
```

### 3.3 Parameter Derivation & Justification
- **$a_{peak} \ge 20.0 \, m/s^2$ ($2.04g$)**: Wrist accelerations during normal arm swing rarely exceed $16 \, m/s^2$. The lowest recorded Watch fall impact is $33.40 \, m/s^2$.
- **Jumping Rejection ($\sigma_a \ge 9.0 \, m/s^2 \land \omega \ge 4.0 \, \text{rad}/s$)**: Repetitive jumping produces continuous wrist variance of $12.3 - 17.3 \, m/s^2$, whereas real fall settling is $< 3.0 \, m/s^2$.
- **$P_{th} = 0.45$**: Preserves `FALL_RIGHT` (where model outputs $0.46 - 0.48$).

---

## 4. State Machine Definition

```
        ┌─────────────────────────────────────────────────────────┐
        │                       MONITORING                        │
        │                (Live Ingestion @ 50 Hz)                 │
        └────────────────────────────┬────────────────────────────┘
                                     │
                 isKinematicFallCandidate == true (x2)
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │                     FALL_SUSPECTED                      │
        │          (10-Second Interactive Countdown UI)           │
        └──────────────┬───────────────────────────┬──────────────┘
                       │                           │
          User presses "I'M OK"        Countdown expires (10s)
                       │                           │
                       ▼                           ▼
        ┌────────────────────────────┐ ┌───────────────────────────┐
        │         CANCELLED          │ │      FALL_CONFIRMED       │
        │   (Return to MONITORING)   │ │       SOS_TRIGGERED       │
        └────────────────────────────┘ └─────────────┬─────────────┘
                                                     │
                                                     ▼
                                       ┌───────────────────────────┐
                                       │    EMERGENCY DISPATCH     │
                                       │ (ACTION_SENDTO sumankmdvg)│
                                       └───────────────────────────┘
```
