# SmartFall AI — Phase 13C: Decision Architecture Experiments

**Scope:** Rigorous evaluation of five candidate decision architectures (A through E) on frozen validation data.  
**Hardware Evaluated:** Phone (`SM-A507FN`, 15 Fall, 26 ADL sessions) and Watch (`SM-R870`, 10 Fall, 25 ADL sessions).

---

## 1. Candidate Architectures Description

- **Candidate A (ML Probability Only)**: Trigger `FALL_SUSPECTED` whenever $P(\text{fall}) \ge 0.50$ in a single window.
- **Candidate B (ML + Impact Gate)**: Trigger when $P(\text{fall}) \ge 0.50$ AND physical impact shock is detected ($\|a\|_{peak} \ge 16.0 \, m/s^2$ or $\Delta a \ge 10 \, m/s^2 \land \|\omega\| \ge 2.5 \, \text{rad}/s$).
- **Candidate C (ML + Impact Gate + Temporal Consensus — Phase 13B Baseline)**: Trigger when two consecutive windows satisfy $P(\text{fall}) \ge 0.50$ and impact was registered within the preceding 3 windows.
- **Candidate D (ML + Calibrated Impact Gate + Temporal Consensus + Thrashing/Stillness Filter — Selected Phase 13C)**:
  Calibrated impact ($a_{peak} \ge 18-20 \, m/s^2$), $P(\text{fall}) \ge 0.45$, 2-window consensus, plus continuous high-motion thrashing rejection.
- **Candidate E (Candidate D + Mandatory Large Posture/Tilt Change $\ge 15^\circ$)**:
  Candidate D with an additional hard requirement that pitch or roll must change by $\ge 15^\circ$ within the impact window.

---

## 2. Quantitative Experimental Results (Validation Set)

### 2.1 Phone (`SM-A507FN`) Evaluation

| Architecture | Recall | Precision | Binary F1 | False Alarm Rate | False Alarm Sessions | High-Motion FAs | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate A (ML Only)** | $100.0\%$ | $46.9\%$ | $63.8\%$ | $65.4\%$ | 17 / 26 | 5 (Walk, Run) | $0.0 \text{ s}$ |
| **Candidate B (ML + Impact)** | $100.0\%$ | $55.6\%$ | $71.4\%$ | $46.2\%$ | 12 / 26 | 5 (Walk, Run) | $0.0 \text{ s}$ |
| **Candidate C (Phase 13B Baseline)** | $93.3\%$ | $63.6\%$ | $75.7\%$ | $30.8\%$ | 8 / 26 | 5 (Walk, Run) | $1.0 \text{ s}$ |
| **Candidate D (Phase 13C Calibrated)** | **$100.0\%$** | **$75.0\%$** | **$85.7\%$** | **$15.4\%$** | **4 / 26** | **2 (Walk only; 0 Run)** | **$1.0 \text{ s}$** |
| **Candidate E (Strict Posture)** | $6.7\%$ | $100.0\%$ | $12.5\%$ | $0.0\%$ | 0 / 26 | 0 | Fail |

### 2.2 Watch (`SM-R870`) Evaluation

| Architecture | Recall | Precision | Binary F1 | False Alarm Rate | False Alarm Sessions | High-Motion FAs | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate A (ML Only)** | $100.0\%$ | $38.5\%$ | $55.6\%$ | $64.0\%$ | 16 / 25 | 6 (Jump, Run, Walk) | $0.0 \text{ s}$ |
| **Candidate B (ML + Impact)** | $100.0\%$ | $47.6\%$ | $64.5\%$ | $44.0\%$ | 11 / 25 | 5 (Jump, Run) | $0.0 \text{ s}$ |
| **Candidate C (Phase 13B Baseline)** | $90.0\%$ | $60.0\%$ | $72.0\%$ | $24.0\%$ | 6 / 25 | 3 (Jump only) | $1.0 \text{ s}$ |
| **Candidate D (Phase 13C Calibrated)** | **$100.0\%$** | **$58.8\%$** | **$74.1\%$** | **$24.0\%$** | **6 / 25** | **3 (Jump only; 0 Walk/Run)**| **$1.0 \text{ s}$** |
| **Candidate E (Strict Posture)** | $10.0\%$ | $50.0\%$ | $16.7\%$ | $4.0\%$ | 1 / 25 | 1 | Fail |

---

## 3. Comparative Architecture Analysis & Failure Modes

### 3.1 Candidate A: ML Probability Only
- **Pros**: Zero added detection latency ($0 \text{ s}$).
- **Cons**: Catastrophic false alarm rate ($> 64\%$). Static sitting, lying down, rotating the device, and brisk walking all trigger false emergency countdowns.
- **Verdict**: Unsuitable for real-world deployment.

### 3.2 Candidate B: ML + Single-Window Impact Gate
- **Pros**: Eliminates purely static false positives (e.g. slowly rotating phone or resting on desk).
- **Cons**: Any energetic walking stride or firm chair sitting exceeds the $16 \, m/s^2$ threshold, triggering single-window false alarms ($44 - 46\%$ FPR).
- **Verdict**: Insufficient false-positive protection.

### 3.3 Candidate C: Phase 13B Baseline (Consensus + Memory)
- **Pros**: Reduced false alarms to $24 - 30\%$; successfully proved physical feasibility on hardware.
- **Cons**: Still vulnerable to rhythmic activities like walking on phone (heel strikes repeatedly trigger impact memory) and jumping on watch.
- **Verdict**: Functional but needs kinematic shock calibration.

### 3.4 Candidate D: Phase 13C Calibrated (WINNER)
- **Pros**:
  - Highest F1 score on both Phone ($85.7\%$) and Watch ($74.1\%$).
  - Preserves $100\%$ validation fall recall across all 5 fall directions.
  - Reduces Phone false alarms from $17$ down to $4$ sessions (a $76.5\%$ reduction in false alarm events).
  - Completely eliminates running false alarms on both devices.
- **Verdict**: **SELECTED AS PRODUCTION ARCHITECTURE**.

### 3.5 Candidate E: Strict Posture Change Filter (The Trap)
- **Pros**: Zero false positives on Phone ($0\%$ FPR).
- **Cons**: Catastrophic collapse in fall sensitivity: Phone recall crashes to $6.7\%$ (14 of 15 falls missed); Watch recall crashes to $10.0\%$.
- **Physical Reason**: Falls from sitting, rolling falls, and pocket shifts do not consistently register large orientation changes in sensor Euler angles. It represents the quintessential **"No false positives = No falls detected" trap**.
- **Verdict**: **REJECTED**.
