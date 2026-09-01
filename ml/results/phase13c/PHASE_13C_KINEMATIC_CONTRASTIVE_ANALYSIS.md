# SmartFall AI — Phase 13C: Contrastive Kinematic Analysis

**Scope:** Direct pair-wise kinematic contrast between genuine falls and high-motion ADL activities.  
**Hardware Evaluated:** Phone (`SM-A507FN`) and Watch (`SM-R870`).

---

## 1. Executive Summary

This report establishes the physical and signal processing boundaries separating true falls from everyday high-motion activities that commonly trigger false fall detections.

```
                  ┌────────────────────────────────────────┐
                  │          Real Fall Signature           │
                  │   Pre-Drop → Impact → Post-Stillness   │
                  └──────────────────┬─────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│   vs. WALKING    │       │   vs. RUNNING    │       │   vs. JUMPING    │
│ No free-fall drop│       │ Continuous cyclic│       │ High shock but   │
│ Peak acc < 17m/s²│       │ variance > 6m/s² │       │ rebound & motion │
│ No post-stillness│       │ No post-stillness│       │ continues        │
└──────────────────┘       └──────────────────┘       └──────────────────┘
```

---

## 2. Pairwise Contrastive Investigations

### 2.1 FALL vs WALKING
- **Distinguishing Features**:
  - **Peak Acceleration ($a_{peak}$)**: Real falls exhibit impact peaks of $21.19 - 116.8 \, m/s^2$ (median: $79.7 \, m/s^2$). Walking produces heel-strike peaks of $13.6 - 16.5 \, m/s^2$ (median: $15.0 \, m/s^2$).
  - **Dynamic Range ($\Delta a$)**: Fall impact range is $> 15.7 \, m/s^2$ (median: $77.3 \, m/s^2$). Walking range is $5.9 - 8.5 \, m/s^2$.
  - **Pre-impact Descent ($a_{\min}$)**: Falls experience weightlessness drop to $0.3 - 6.2 \, m/s^2$. In walking, $a_{\min}$ remains supported at $7.5 - 7.9 \, m/s^2$.
- **Overlapping Features**:
  - Torso and thigh inclination during walking strides produce orientation pitch angles overlapping with tilted post-fall recumbency.
- **Failure Scenario**: Firm brisk walking or stomping can produce localized transients of $17 - 19 \, m/s^2$, causing single-window ML models to trigger.
- **Boundary Condition**: $a_{peak} \ge 18.0 \, m/s^2$ on Phone and $20.0 \, m/s^2$ on Watch, coupled with 2-window consensus.

---

### 2.2 FALL vs RUNNING
- **Distinguishing Features**:
  - **Post-Impact Tail Variance ($\sigma_{tail}$)**: In running, $\sigma_{tail} = 7.07 \, m/s^2$ (Phone) and $6.78 \, m/s^2$ (Watch) because the runner continues taking strides. In falls, $\sigma_{tail} = 0.08 \, m/s^2$ (Phone) and $0.06 \, m/s^2$ (Watch) due to post-impact stillness.
  - **Mean Angular Rate ($\bar{\omega}$)**: Running arm/pocket swing produces $\bar{\omega} = 2.12 - 2.36 \, \text{rad}/s$. In post-impact falls, $\bar{\omega} = 0.04 \, \text{rad}/s$ (30 to 50 times lower).
- **Overlapping Features**:
  - Peak foot strike in running reaches $35 - 45 \, m/s^2$, exceeding basic fall impact thresholds.
- **Failure Scenario**: A runner taking a hard stride produces an impact transient and strong rotational swing, causing uncalibrated models to suspect a fall.
- **Boundary Condition**: Active motion thrashing rejection: $\sigma_a \ge 4.5 \, m/s^2$ and $\|\omega\| \ge 3.5 \, \text{rad}/s$ rejects continuous running.

---

### 2.3 FALL vs JUMPING
- **Distinguishing Features**:
  - **Post-Impact Inactivity**: In jumping, the landing impact is followed by knee flexion, body rebound, or further steps ($\sigma_a > 12.0 \, m/s^2$, $\bar{\omega} > 2.0 \, \text{rad}/s$). In a fall, the subject remains recumbent on the ground ($\sigma_a < 1.0 \, m/s^2$, $\bar{\omega} < 0.5 \, \text{rad}/s$).
  - **Free-fall Phase Duration**: Vertical jumping has an airborne ballistic arc where $a \approx 0$, followed by bilateral leg landing. Falls have an involuntary asymmetric tumbling trajectory.
- **Overlapping Features**:
  - Jumping landing impact is violent: $51.16 \, m/s^2$ on Watch, $49.99 \, m/s^2$ on Phone. The Watch Random Forest outputs $P(\text{fall}) \ge 0.97$ on $97.4\%$ of jumping windows.
- **Failure Scenario**: Single-window impact gates cannot differentiate a jump landing from a fall collision.
- **Boundary Condition**: Jumping is rejected by requiring multi-window temporal confirmation where the subject does not continue continuous jumping cadence ($\sigma_a < 9.0 \, m/s^2$ in confirmation window).

---

### 2.4 FALL vs SIT_DOWN & STAND_UP
- **Distinguishing Features**:
  - **Impact Shock Magnitude**: Sitting down produces a gentle deceleration when the pelvis meets the chair ($a_{peak} = 12.9 - 13.4 \, m/s^2$, peak jerk: $70 - 78 \, m/s^3$). Genuine falls produce hard ground deceleration ($a_{peak} > 21 - 100 \, m/s^2$, jerk: $200 - 1000 \, m/s^3$).
  - **Descent Velocity**: Sitting down is controlled by eccentric quadriceps contraction; acceleration magnitude rarely drops below $6.8 \, m/s^2$.
- **Overlapping Features**:
  - Orientation change: Going from standing to sitting rotates thigh orientation by $\approx 90^\circ$, mimicking a forward or backward pitch change.
- **Failure Scenario**: Dropping heavily into a rigid wooden chair can create a localized spike of $15 - 17 \, m/s^2$.
- **Boundary Condition**: Impact shock threshold $a_{peak} \ge 18.0 \, m/s^2$ completely rejects normal and firm sitting.

---

### 2.5 FALL vs PICKING_UP_OBJECT
- **Distinguishing Features**:
  - **Absence of Ground Collision**: Picking up an object has zero collision shock ($a_{peak} = 13.9 \, m/s^2$, $\Delta a = 8.1 \, m/s^2$).
  - **Motion Smoothness**: Jerk is low ($< 90 \, m/s^3$), whereas falls produce jerk $> 300 \, m/s^3$.
- **Overlapping Features**:
  - Pitch angle tilts from $0^\circ$ down to $-80^\circ$ as the torso bends over, which strongly excites the orientation channels of the 1D-CNN.
- **Failure Scenario**: Bending quickly and bumping a table or knee can trigger orientation-biased models.
- **Boundary Condition**: Requiring both physical deceleration shock ($a_{peak} \ge 18.0 \, m/s^2$) and 2-window consensus eliminates picking up objects.

---

## 3. Summary Contrast Matrix

| Activity | Typical $a_{peak}$ ($m/s^2$) | Free-fall $a_{\min}$ ($m/s^2$) | Post-Impact Stillness | Continues Cyclic Motion? |
| :--- | :---: | :---: | :---: | :---: |
| **REAL FALL** | **$25 - 116$** | **$< 6.0$** | **YES ($\sigma < 0.1$)** | **NO (Stops)** |
| **Walking** | $12 - 16$ | $> 7.5$ | NO | YES |
| **Running** | $30 - 45$ | $2.5 - 3.5$ | NO | YES |
| **Jumping** | $45 - 65$ | $1.0 - 2.5$ | NO | YES |
| **Sit Down** | $12 - 14$ | $> 6.8$ | YES (Sitting) | NO |
| **Stand Up** | $13 - 15$ | $> 7.3$ | YES (Standing) | NO |
| **Pick Up Object**| $12 - 14$ | $> 6.5$ | NO | NO |
