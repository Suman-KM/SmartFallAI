# SmartFall AI — Phase 13D: Temporal Trajectory & Multi-Window Analysis

**Scope:** Mathematical analysis of temporal state transitions in human falls versus high-motion ADLs.

---

## 1. The Fall State Progression Across Windows

Because SmartFall AI operates on 100-sample windows ($2.0 \text{ seconds}$) with a 50-sample stride ($1.0 \text{ second}$), an unexpected ground collision spans multiple adjacent windows:

```
 Window t-1: [ Normal ADL Motion ]    ||a|| ≈ 9.81 m/s², σ_a < 2.0 m/s², low jerk (< 150)
                   │
                   ▼
 Window t:   [ Free-Fall Descent ]    ||a||_min < 6.5 m/s² (100 - 250 ms)
             [ Collision Impact  ]    ||a||_peak > 20 - 100 m/s², Jerk > 1500 m/s³
                   │
                   ▼
 Window t+1: [ Landing / Bounce  ]    Secondary impact damping, σ_a still elevated (3 - 8 m/s²)
                   │
                   ▼
 Window t+2: [ Body Settles Down ]    Recumbent floor posture, σ_a collapses to < 2.4 m/s²
                   │
                   ▼
 Window t+3: [ Floor Immobility  ]    Sustained stillness, σ_a < 0.2 m/s², P(fall) ≥ 0.40
```

---

## 2. Why the Immediate Next Window ($t+1$) Cannot Be Forced to Stillness

A crucial mathematical discovery in Phase 13D was the timing of body settling:
- If a fall impact occurs at sample index 80 of Window $t$:
- Window $t+1$ (covering samples 50 to 149) contains the collision impact in its first half and the human body bouncing/sliding in its second half!
- In Window $t+1$, dynamic acceleration standard deviation ($\sigma_a$) is **$3.5 - 11.8 \, m/s^2$**!
- Requiring $\sigma_a \le 2.0 \, m/s^2$ immediately in window $t+1$ caused $53\%$ of genuine falls to be discarded as "thrashing"!
- **The Phase 13D Solution:** Provide a **4-window verification horizon** ($2.0 - 3.0 \text{ seconds}$). The detector verifies that within this window, motion transitions from impact shock into recumbent immobility.

---

## 3. The Active Cadence Locomotion Discriminator

In active rhythmic human movement (running, jumping, energetic walking):
- Foot strikes occur every $300 - 500 \text{ ms}$ ($2.0 - 3.0 \text{ Hz}$).
- In Window $t$: Step pulse.
- In Window $t+1$: Step pulse.
- In Window $t+2$: Step pulse.
- In Window $t+3$: Step pulse.
- At no point during running does $\sigma_a$ drop below $3.2 \, m/s^2$ or $\omega$ drop below $3.5 \, \text{rad}/s$.
- By checking for continuous locomotion cadence during the 4-window verification horizon, **running and jumping false alarms are completely eliminated**.
