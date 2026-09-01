# SmartFall AI — Phase 13D: Decision Architecture Comparison

**Scope:** Formal comparative evaluation of 8 candidate decision architectures (A through H) on frozen Validation data.

---

## 1. Candidate Architectures Evaluated

- **Architecture A (ML Probability Only)**: $P(\text{fall}) \ge 0.50$ in a single window.
- **Architecture B (ML + Impact Shock)**: $P(\text{fall}) \ge 0.50 \land a_{peak} \ge 20.0 \, m/s^2$.
- **Architecture C (Phase 13C Baseline)**: Impact shock ($18-20 \, m/s^2$) + 3-window impact memory + 2-window consensus.
- **Architecture D (ML + Impact + Post-Event Stillness)**: Impact shock + immediate next window stillness ($\sigma_a \le 2.5 \, m/s^2$).
- **Architecture E (ML + Impact + Pre-Drop Trajectory)**: Descent unloading ($a_{\min} \le 7.0 \, m/s^2$) + Impact ($a_{peak} \ge 20.0$) + Jerk ($j \ge 400$).
- **Architecture F (ML + Impact + Trajectory + Strict Post-Event Stillness)**: Architecture E with mandatory immediate post-stillness.
- **Architecture G (ML + Impact + Trajectory + Movement-Continuation Rejection)**: Architecture E + reject active ongoing cadence.
- **Architecture H (Phase 13D Selected Multi-Stage Architecture)**: Collision shock ($a_{peak} \ge 20.0, j \ge 350$) + 4-window verification horizon + active locomotion cadence rejection + post-impact immobility confirmation ($\sigma_a \le 2.4, \omega \le 2.2$).

---

## 2. Quantitative Results Comparison (Validation Set)

### 2.1 Phone (`SM-A507FN`, 15 Falls, 26 ADL Sessions)

| Arch | Recall | Precision | Binary F1 | FPR | Total FAs | High-Motion FAs | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | $100.0\%$ | $46.9\%$ | $63.8\%$ | $65.4\%$ | 17 / 26 | 5 (Walk 4, Run 1) | $0.0 \text{ s}$ |
| **B** | $100.0\%$ | $68.2\%$ | $81.1\%$ | $26.9\%$ | 7 / 26 | 4 (Walk 3, Run 1) | $0.0 \text{ s}$ |
| **C** | $86.7\%$ | $68.4\%$ | $76.5\%$ | $23.1\%$ | 6 / 26 | 4 (Walk 4) | $1.0 \text{ s}$ |
| **D** | $93.3\%$ | $70.0\%$ | $80.0\%$ | $23.1\%$ | 6 / 26 | 3 (Walk 2, Run 1) | $1.0 \text{ s}$ |
| **E** | $86.7\%$ | $65.0\%$ | $74.3\%$ | $26.9\%$ | 7 / 26 | 3 (Walk 2, Jump 1) | $0.0 \text{ s}$ |
| **F** | $46.7\%$ | $63.6\%$ | $53.8\%$ | $15.4\%$ | 4 / 26 | **0 (Walk 0, Run 0, Jump 0)** | $1.0 \text{ s}$ |
| **G** | $53.3\%$ | $66.7\%$ | $59.3\%$ | $15.4\%$ | 4 / 26 | **0 (Walk 0, Run 0, Jump 0)** | $1.0 \text{ s}$ |
| **H (13D)**| **$80.0\%$** | **$70.6\%$** | **$75.0\%$** | **$19.2\%$** | **5 / 26** | **0 (Walk 0, Run 0, Jump 0)** | **$1.5 \text{ s}$** |

### 2.2 Watch (`SM-R870`, 10 Falls, 25 ADL Sessions)

| Arch | Recall | Precision | Binary F1 | FPR | Total FAs | High-Motion FAs | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | $100.0\%$ | $38.5\%$ | $55.6\%$ | $64.0\%$ | 16 / 25 | 6 (Jump 3, Run 2, Walk 1) | $0.0 \text{ s}$ |
| **B** | $90.0\%$ | $56.2\%$ | $69.2\%$ | $28.0\%$ | 7 / 25 | 4 (Jump 3, Run 1) | $0.0 \text{ s}$ |
| **C** | $70.0\%$ | $58.3\%$ | $63.6\%$ | $20.0\%$ | 5 / 25 | 3 (Jump 3) | $1.0 \text{ s}$ |
| **D** | $80.0\%$ | $53.3\%$ | $64.0\%$ | $28.0\%$ | 7 / 25 | 1 (Jump 1) | $1.0 \text{ s}$ |
| **E** | $80.0\%$ | $50.0\%$ | $61.5\%$ | $32.0\%$ | 8 / 25 | 4 (Jump 3, Run 1) | $0.0 \text{ s}$ |
| **F** | $30.0\%$ | $60.0\%$ | $40.0\%$ | $8.0\%$ | 2 / 25 | **0 (Jump 0, Run 0, Walk 0)** | $1.0 \text{ s}$ |
| **G** | $50.0\%$ | $62.5\%$ | $55.6\%$ | $12.0\%$ | 3 / 25 | **0 (Jump 0, Run 0, Walk 0)** | $1.0 \text{ s}$ |
| **H (13D)**| **$100.0\%$** | **$66.7\%$** | **$80.0\%$** | **$20.0\%$** | **5 / 25** | **1 (Jump 1, Run 0, Walk 0)** | **$1.5 \text{ s}$** |

---

## 3. Architecture Selection Justification

Architecture H was selected because:
1. It eliminates **100% of Walking and Running false alarms on Phone**, and drops Watch jumping false alarms by $67\%$.
2. It preserves **100% Fall Recall on Watch** and maintains high sensitivity across all directional fall categories on Phone.
3. It achieves this without retraining the frozen models and without unrealistic physical constraints.
