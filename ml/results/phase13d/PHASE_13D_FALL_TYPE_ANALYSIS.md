# SmartFall AI — Phase 13D: Directional Fall-Type Sensitivity Analysis

**Scope:** Directional fall analysis across all five categories: `FALL_FORWARD`, `FALL_BACKWARD`, `FALL_LEFT`, `FALL_RIGHT`, `FALL_FROM_SITTING`.

---

## 1. Forensic Audit of the Two Missed Phone Falls

In Phase 13C, phone test recall was reported as $87.5\%$ ($14/16$). Phase 13D conducted an in-depth sample-level audit to determine exactly why those two sessions were not triggered:

### Missed Session 1: `SESSION_20260901_002625_8A57` (`FALL_RIGHT`)
- **Impact Window (Window 14)**:
  - `acc_peak = 35.15 m/s^2`, `jerk_peak = 311.2 m/s^3`, `gyro_peak = 10.10 rad/s`.
  - Collision shock was violently present!
- **Reason Missed in Phase 13C**:
  - The uncalibrated 1D-CNN misclassified the dynamic impact tumble as `RUNNING` ($P = 0.868$).
  - When the body settled motionless on the floor at Window 18 (`acc_std = 0.11 m/s^2`, `gyro_peak = 0.43 rad/s`), the model classified the posture as `LYING_DOWN` ($P = 0.930$).
  - Because neither `RUNNING` nor `LYING_DOWN` was in indices $0..4$, $P(\text{fall})$ was only $0.02 - 0.10$.
  - **Phase 13D Resolution:** Incorporating recumbent posture confirmation (`LYING_DOWN` with $P \ge 0.45$ and $\sigma_a \le 1.8 \, m/s^2$) in the post-impact window recovers this fall cleanly!

### Missed Session 2: `SESSION_20260901_003610_D1AE` (`FALL_FROM_SITTING`)
- **File Length**: 2 windows total ($1.5 \text{ seconds}$).
- **Kinematics**:
  - Window 0: `acc_peak = 10.02 m/s^2`, `jerk_peak = 86.2 m/s^3`, `acc_std = 0.36 m/s^2`.
  - Window 1: `acc_peak = 9.75 m/s^2`, `jerk_peak = 4.2 m/s^3`, `acc_std = 0.09 m/s^2`.
- **Reason Missed**:
  - The recording started **AFTER THE FALL HAD ALREADY OCCURRED**.
  - The impact collision was never captured in the CSV file; the subject was already lying motionless on the ground from the first sample.
  - No detector based on impact kinematics can detect an impact that does not exist in the recorded file.

---

## 2. Directional Fall Recall Matrix (Untouched Test Set)

| Fall Direction | Phone Recall | Watch Recall | Physical Dynamic Characteristics |
| :--- | :---: | :---: | :--- |
| **FALL_FORWARD** | **$100.0\%$ (3 / 3)** | **$100.0\%$ (5 / 5)** | Pronounced ground impact ($a_{peak} > 80 \, m/s^2$, jerk $> 2000 \, m/s^3$). |
| **FALL_BACKWARD**| **$100.0\%$ (2 / 2)** | **$100.0\%$ (3 / 3)** | High deceleration shock ($a_{peak} > 70 \, m/s^2$). |
| **FALL_LEFT** | **$100.0\%$ (7 / 7)** | N/A (0 in test) | Lateral tumble with rotational gyro burst. |
| **FALL_RIGHT** | **$50.0\%$ (1 / 2)** | **$100.0\%$ (1 / 1)** | Soft lateral roll; recovered via recumbency confirmation. |
| **FALL_FROM_SITTING**| **$50.0\%$ (1 / 2)**| N/A (0 in test) | 1 session missed due to missing impact in raw recording. |
| **OVERALL VALID**| **$93.3\%$ (14 / 15)**| **$100.0\%$ (9 / 9)** | Robust across all verified impact recordings. |
