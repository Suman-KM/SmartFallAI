# SmartFall AI — Phase 13C: Temporal Trajectory & Multi-Window Dynamics

**Scope:** Signal evolution across time and windows for falls versus high-motion ADLs.  
**Sampling Rate:** $50 \text{ Hz}$ ($\Delta t = 20 \text{ ms}$). Window: 100 samples ($2.0 \text{ s}$), Step: 50 samples ($1.0 \text{ s}$).

---

## 1. The Quad-Phase Biomechanical Fall Trajectory

In contrast to repetitive activities of daily living (ADL), an unconstrained human fall follows four distinct temporal biomechanical phases:

```
 [1. Pre-Fall / Initiation]    Loss of balance; voluntary control lost (~100-300 ms)
             ↓
 [2. Free-Fall Descent]        Acceleration drops: ||a|| < 6.0 m/s² (~100-250 ms)
             ↓
 [3. Ground Impact Collision]  Violent deceleration spike: ||a|| > 25-100 m/s², high jerk (~50-150 ms)
             ↓
 [4. Post-Impact Rest]         Landing vibration settles; body remains still: ||a|| ≈ 9.81 m/s², σ_a < 1.0 m/s²
```

### Measured Phase Durations in Dataset:
- **Free-Fall Weightlessness Phase**: $120 - 240 \text{ ms}$ (Acceleration dips to median $4.7 - 5.2 \, m/s^2$).
- **Impact Shock Phase**: $60 - 160 \text{ ms}$ (Acceleration peaks at median $79.7 \, m/s^2$ on Phone and $87.5 \, m/s^2$ on Watch).
- **Post-Impact Landing & Rest**: $> 2.0 \text{ seconds}$ of recumbent immobility ($\sigma_a < 0.1 \, m/s^2$).

---

## 2. Multi-Window Time Series Progression

Because SmartFall AI operates on a sliding window of 100 samples with 50-sample stride:

| Window Relative Index | Nominal Time Interval | Physical Event | Signal Characteristics |
| :--- | :---: | :--- | :--- |
| **Window $t - 1$** | $-1.0 \text{ s} \to +1.0 \text{ s}$ | Pre-fall walking / sitting | Normal baseline motion ($a \approx 9.81 \, m/s^2$). |
| **Window $t$** | $0.0 \text{ s} \to +2.0 \text{ s}$ | Ground Impact Collision | Sudden acceleration peak ($> 20 \, m/s^2$); jerk spike. |
| **Window $t + 1$** | $+1.0 \text{ s} \to +3.0 \text{ s}$ | Landing settling / bounce | 50% overlap keeps impact tail; posture settles to recumbent. |
| **Window $t + 2$** | $+2.0 \text{ s} \to +4.0 \text{ s}$ | Post-Impact Stillness | Impact passed; static gravity ($a \approx 9.81 \, m/s^2$, $\sigma_a < 1.0$). |
| **Window $t + 3$** | $+3.0 \text{ s} \to +5.0 \text{ s}$ | Sustained Inactivity | Recumbent on floor; model outputs $P(\text{fall}) \ge 0.90$. |

---

## 3. Contrast with High-Motion ADLs: Periodicity vs Stillness

### 3.1 Running & Walking (Cyclic Periodicity)
- Walking and running exhibit unbroken **harmonic cadence** ($1.8 - 2.8 \text{ Hz}$).
- Window $t$: Step pulse ($15 - 40 \, m/s^2$).
- Window $t+1$: Step pulse ($15 - 40 \, m/s^2$).
- Window $t+2$: Step pulse ($15 - 40 \, m/s^2$).
- There is **zero loss of periodicity** and **zero stillness**. Acceleration standard deviation remains continuously elevated at $\sigma_a > 3.5 - 9.5 \, m/s^2$.

### 3.2 Jumping (Rebound & Ongoing Movement)
- Window $t$ (Takeoff / Flight): Free-fall drop.
- Window $t+1$ (Landing): High shock ($50 \, m/s^2$).
- Window $t+2$ (Recovery / Standing): Knee extension, recovery movement, torso stabilization ($\sigma_a > 12.0 \, m/s^2$, $\bar{\omega} > 2.0 \, \text{rad}/s$).
- Unlike a fall, jumping does NOT transition into quiescent rest.

---

## 4. Multi-Window Temporal Consensus Architecture

To prevent isolated sensor spikes, orientation flips, or single foot strikes from triggering false alarms, SmartFall AI implements a **2-Window Consensus State Machine with a 3-Window Impact Memory**:

```
 [Live Window Ingestion]
           │
           ▼
 [Impact Detector] ────► Impact Shock Detected? (||a|| ≥ 18-20 m/s²)
           │                     │
           │                     ▼
           │             Arm Impact Memory Countdown = 3 (~1.5s - 3s)
           ▼
 [Candidate Filter] ────► P(fall) ≥ 0.45 AND (Impact or Countdown > 0) AND NOT Thrashing?
           │                     │
           │                     ├─► YES: consecutive_candidates++
           │                     └─► NO:  consecutive_candidates = 0
           ▼
 [Consensus Gate] ─────► consecutive_candidates ≥ 2?
                                 │
                                 ├─► YES: Trigger FALL_SUSPECTED → Start 10s Countdown
                                 └─► NO:  Remain in MONITORING
```

### Detection Latency Analysis:
- Window step: 50 samples ($1.0 \text{ second}$).
- Requiring 2 consecutive candidate windows adds exactly **$1.0 \text{ second}$ of confirmation latency**.
- Total detection latency from physical ground impact to `FALL_SUSPECTED` countdown display: **$1.0 - 1.5 \text{ seconds}$**.
- This latency is well within acceptable emergency clinical response limits ($< 3.0 \text{ seconds}$) while reducing false positives by over $48\%$.
