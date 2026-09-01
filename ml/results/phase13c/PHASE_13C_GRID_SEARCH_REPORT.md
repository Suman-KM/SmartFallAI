# SmartFall AI — Phase 13C: Validation Grid Search & Optimization Report

**Date:** September 1, 2026  
**Scope:** Hyperparameter grid search across kinematic gates, probability thresholds, temporal consensus, and stillness parameters.  
**Dataset:** Frozen Validation Sets (Phone: 15 Fall, 26 ADL sessions; Watch: 10 Fall, 25 ADL sessions).

---

## 1. Grid Search Design Space

A multi-dimensional grid search was executed over the following parameter space:
1. **Model Fall Probability Threshold ($P_{th}$)**: $0.35, 0.40, 0.45, 0.50, 0.55, 0.60$
2. **Acceleration Peak Shock ($a_{peak}$)**:
   - Phone: $16.0, 18.0, 20.0, 22.0, 24.0, 26.0 \, m/s^2$
   - Watch: $18.0, 20.0, 22.0, 25.0, 28.0, 30.0 \, m/s^2$
3. **Pre-Impact Descent Drop ($a_{\min}$)**: None vs. $\le 8.0 \, m/s^2$
4. **Temporal Consensus Windows ($N_{consec}$)**: $1, 2, 3$ consecutive candidate windows
5. **Impact Memory Decay Buffer ($N_{mem}$)**: $2, 3, 4$ sliding windows ($1.0 - 2.0 \text{ s}$)

---

## 2. Top Performing Configurations (Phone SM-A507FN)

| Rank | $a_{peak}$ ($m/s^2$) | $P_{th}$ | Consensus | Memory | Recall | Precision | Binary F1 | False Alarm Rate | False Alarm Sessions |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Selected)** | **$18.0$** | **$0.45$** | **2** | **3** | **$100.0\%$** | **$75.0\%$** | **$85.7\%$** | **$15.4\%$** | **4 / 26** |
| 2 | $20.0$ | $0.45$ | 2 | 2 | $93.3\%$ | $82.4\%$ | $87.5\%$ | $11.5\%$ | 3 / 26 |
| 3 | $24.0$ | $0.60$ | 2 | 2 | $93.3\%$ | $82.4\%$ | $87.5\%$ | $11.5\%$ | 3 / 26 |
| 4 | $18.0$ | $0.40$ | 2 | 3 | $100.0\%$ | $68.2\%$ | $81.1\%$ | $26.9\%$ | 7 / 26 |
| 5 | $20.0$ | $0.50$ | 2 | 3 | $86.7\%$ | $81.2\%$ | $83.9\%$ | $11.5\%$ | 3 / 26 |

### Phone Optimization Rationale:
Configuration Rank 1 achieves the clinical objective: **$100\%$ Fall Recall** (15/15 falls detected, preserving lateral and sitting falls) while reducing false alarm sessions by $76.5\%$ compared to baseline ML-only inference. Raising $a_{peak}$ above $20.0 \, m/s^2$ caused subtle lateral falls (`FALL_RIGHT`) to be missed, violating the safety-critical priority.

---

## 3. Top Performing Configurations (Watch SM-R870)

| Rank | $a_{peak}$ ($m/s^2$) | $P_{th}$ | Consensus | Memory | Recall | Precision | Binary F1 | False Alarm Rate | False Alarm Sessions |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Selected)** | **$20.0$** | **$0.45$** | **2** | **3** | **$100.0\%$** | **$58.8\%$** | **$74.1\%$** | **$24.0\%$** | **6 / 25** |
| 2 | $18.0$ | $0.45$ | 2 | 3 | $100.0\%$ | $58.8\%$ | $74.1\%$ | $24.0\%$ | 6 / 25 |
| 3 | $20.0$ | $0.50$ | 2 | 3 | $70.0\%$ | $70.0\%$ | $70.0\%$ | $12.0\%$ | 3 / 25 |
| 4 | $22.0$ | $0.50$ | 2 | 3 | $70.0\%$ | $70.0\%$ | $70.0\%$ | $12.0\%$ | 3 / 25 |
| 5 | $25.0$ | $0.55$ | 1 | 3 | $100.0\%$ | $62.5\%$ | $76.9\%$ | $24.0\%$ | 6 / 25 |

### Watch Optimization Rationale:
On the wrist, setting $P_{th} = 0.45$ and $a_{peak} = 20.0 \, m/s^2$ with 2-window consensus ensures $100\%$ detection of all 10 fall validation sessions (including soft lateral falls). Raising $P_{th}$ to $0.50$ missed `FALL_FROM_SITTING` and `FALL_RIGHT` (dropping recall to $70\%$).

---

## 4. Final Operating Point Selection

| Parameter | Phone Operating Value | Watch Operating Value | Physical Justification |
| :--- | :---: | :---: | :--- |
| **Model Fall Threshold ($P_{th}$)** | **$0.45$** | **$0.45$** | Preserves sensitivity to lateral and sitting falls. |
| **Impact Shock Peak ($a_{peak}$)** | **$18.0 \, m/s^2$** | **$20.0 \, m/s^2$** | Exceeds $95\%$ of walking strides; wrist is higher energy. |
| **Impact Range ($\Delta a$)** | **$10.0 \, m/s^2$** | **$12.0 \, m/s^2$** | Identifies rotational tumbles. |
| **Impact Gyro Peak ($\omega_{peak}$)** | **$2.5 \, \text{rad}/s$** | **$3.0 \, \text{rad}/s$** | Validates angular velocity during loss of balance. |
| **Active Thrashing Rejection ($\sigma_a$)** | **$\ge 4.5 \, m/s^2$** | **$\ge 9.0 \, m/s^2$** | Rejects continuous running (Phone) and jumping (Watch). |
| **Impact Memory ($N_{mem}$)** | **3 windows** | **3 windows** | Bridges impact shock and post-impact posture classification. |
| **Temporal Consensus ($N_{consec}$)**| **2 windows** | **2 windows** | Adds $1.0 \text{ s}$ confirmation; eliminates transient sensor spikes. |
