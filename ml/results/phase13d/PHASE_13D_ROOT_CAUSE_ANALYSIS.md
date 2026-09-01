# SmartFall AI — Phase 13D: Forensic Root Cause Analysis

**Date:** September 1, 2026  
**Focus:** Mechanical and Signal Processing Etiology of Real-World False Positives During Ordinary Human Movement.

---

## 1. Executive Summary

Phase 13B and 13C physical device trials revealed a persistent operational failure: normal human movements—including walking, running, sitting down firmly, and simple device handling—repeatedly triggered the fall detector countdown.

This forensic analysis identifies the precise mathematical and biomechanical root causes of this failure across both the Phone (`SM-A507FN`) and Watch (`SM-R870`).

---

## 2. The Four Root Causes of False Positives

### Root Cause 1: Static Euler-Angle Bias in Single-Window ML Models
- **Mechanism**:
  The Phone 1D-CNN and Watch Random Forest were trained on sliding windows where post-fall states are characterized by horizontal recumbency.
  Consequently, the models heavily weight static orientation Euler angles (`pitch` and `roll`).
- **Physical Failure**:
  - When the user places the phone flat on a desk, or sets it on their lap while sitting, `pitch \approx 0^\circ, roll \approx 0^\circ`.
  - The uncalibrated 1D-CNN outputs $P(\text{fall}) \ge 0.80$ while the phone is completely motionless on the desk!
  - Similarly, when the Galaxy Watch4 is taken off or resting on a charging puck, the Watch Random Forest outputs $P(\text{fall}) = 0.58 - 0.59$ on static stillness.

### Root Cause 2: Low-Threshold Acceleration Range & Arming Memory
- **Mechanism**:
  In Phase 13B/13C, an impact arming condition was introduced:
  `val hasImpact = (maxAccMag >= 18.0f) || (accRange >= 10.0f && maxGyroMag >= 2.5f)`
  Once tripped, an impact memory buffer held the detector in an armed state for 3 sliding windows ($1.5 - 3.0 \text{ seconds}$).
- **Physical Failure**:
  - In normal walking, forward leg swing naturally rotates the thigh, producing $\Delta a = 11.0 \, m/s^2$ and angular velocity $\omega = 2.8 \, \text{rad}/s$.
  - This innocuously armed the impact memory on every stride.
  - While armed, any subsequent window with model probability $P(\text{fall}) \ge 0.45$ (which occurs in $22.8\%$ of walking windows) triggered the 2-window consensus state machine!
  - Normal walking became an automatic trigger loop.

### Root Cause 3: Disregard of Impact Jerk ($|da/dt|$)
- **Mechanism**:
  Previous architectures only examined raw acceleration magnitude ($\|a\|_{peak}$).
- **Physical Failure**:
  - A running foot strike produces $\|a\|_{peak} = 35 - 45 \, m/s^2$.
  - A jump landing produces $\|a\|_{peak} = 50 - 65 \, m/s^2$.
  - However, in controlled human locomotion, muscular elasticity and joint compliance decelerate the body smoothly over $200 - 400 \text{ ms}$ (jerk: $50 - 200 \, m/s^3$).
  - In contrast, an uncontrolled ground collision with a floor has deceleration times $< 20 \text{ ms}$, producing jerk spikes of **$1,500 - 4,000 \, m/s^3$** (a 15x to 30x difference!). Ignoring jerk allowed smooth high-motion steps to be classified as hard impacts.

### Root Cause 4: Absence of Post-Impact Locomotion Rejection
- **Mechanism**:
  Previous filters evaluated windows independently or required immediate stillness in the very next window ($t+1$).
- **Physical Failure**:
  - In genuine falls, the body collides in window $t$, experiences secondary rebound/sliding in window $t+1$, and settles into recumbent immobility in windows $t+2$ and $t+3$.
  - In running or jumping, locomotion cadence continues unbroken in window $t+1, t+2, t+3$ with continuous dynamic variance ($\sigma_a > 5 \, m/s^2$).
  - Because previous pipelines did not distinguish active ongoing locomotion from post-impact immobility, runners taking firm strides triggered fall detection.

---

## 3. Summary of Root Cause Diagnoses

| Movement | Why It Triggered in Phase 13C | Physical Discriminator Discovered in Phase 13D |
| :--- | :--- | :--- |
| **Walking** | Thigh swing exceeded $\Delta a \ge 10, \omega \ge 2.5$, arming memory while 1D-CNN output $P \ge 0.45$. | Jerk is low ($< 150 \, m/s^3$); ground impact shock ($20 \, m/s^2$) is never reached. |
| **Running** | Foot strikes exceeded $18 \, m/s^2$, arm swing had low gyro in tight pocket. | Active locomotion continues in subsequent windows ($\sigma_a > 3.2 \, m/s^2$); motion never collapses to stillness. |
| **Jumping** | Landing impact on Watch was $50 \, m/s^2$, RF model predicted $P = 0.97$. | Knee recovery and post-landing body stabilization prevent recumbent immobility. |
| **Desk Placement** | Tapping desk produced transient shock while flat orientation had $P(\text{fall}) \ge 0.80$. | No pre-impact free-fall descent drop; jerk is localized without whole-body tumble. |
