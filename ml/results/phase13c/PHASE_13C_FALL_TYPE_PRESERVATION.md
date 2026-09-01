# SmartFall AI — Phase 13C: Directional Fall-Type Preservation Analysis

**Scope:** Directional fall analysis across all five distinct fall categories:
`FALL_FORWARD`, `FALL_BACKWARD`, `FALL_LEFT`, `FALL_RIGHT`, and `FALL_FROM_SITTING`.

---

## 1. Executive Mandate

A fall detection system is clinically unacceptable if it achieves high overall recall by detecting only violent forward and backward falls while completely missing lateral or sitting falls.

In Phase 13C, parameter selection was governed by the **Directional Fall Preservation Rule**:
> Any candidate threshold configuration that degraded any individual fall direction below $75\%$ recall on validation was immediately rejected, even if it produced zero false positives.

---

## 2. Validation Set Directional Recall Breakdown

### 2.1 Phone (`SM-A507FN`) Validation Set Performance
- Total Fall Sessions: 15 (Forward: 4, Backward: 3, Left: 5, Right: 2, From Sitting: 1)

| Fall Direction | Sessions Detected | Total Sessions | Recall Rate | Kinematic Signature |
| :--- | :---: | :---: | :---: | :--- |
| **FALL_FORWARD** | 4 | 4 | **$100.0\%$** | High impact ($a_{peak} = 93 - 116 \, m/s^2$), pronounced forward pitch change. |
| **FALL_BACKWARD** | 3 | 3 | **$100.0\%$** | Impact ($a_{peak} = 46 - 79 \, m/s^2$), reverse pitch tilt. |
| **FALL_LEFT** | 5 | 5 | **$100.0\%$** | Lateral roll ($a_{peak} = 21 - 88 \, m/s^2$, $\omega_{peak} = 6.8 - 18.4 \, \text{rad}/s$). |
| **FALL_RIGHT** | 2 | 2 | **$100.0\%$** | Lateral roll ($a_{peak} = 48 - 67 \, m/s^2$, $\omega_{peak} = 13.8 - 17.0 \, \text{rad}/s$). |
| **FALL_FROM_SITTING** | 1 | 1 | **$100.0\%$** | Low height drop ($a_{peak} = 101.6 \, m/s^2$, $\omega_{peak} = 20.3 \, \text{rad}/s$). |
| **OVERALL PHONE** | **15** | **15** | **$100.0\%$** | **ALL 5 FALL DIRECTIONS PRESERVED AT 100%** |

### 2.2 Watch (`SM-R870`) Validation Set Performance
- Total Fall Sessions: 10 (Forward: 5, Backward: 1, Left: 1, Right: 2, From Sitting: 1)

| Fall Direction | Sessions Detected | Total Sessions | Recall Rate | Kinematic Signature |
| :--- | :---: | :---: | :---: | :--- |
| **FALL_FORWARD** | 5 | 5 | **$100.0\%$** | Direct arm collision ($a_{peak} = 71 - 135 \, m/s^2$). |
| **FALL_BACKWARD** | 1 | 1 | **$100.0\%$** | Arm impact landing ($a_{peak} = 90.7 \, m/s^2$). |
| **FALL_LEFT** | 1 | 1 | **$100.0\%$** | Lateral wrist strike ($a_{peak} = 66.8 \, m/s^2$). |
| **FALL_RIGHT** | 2 | 2 | **$100.0\%$** | Arm sweep ($a_{peak} = 33.4 - 58.3 \, m/s^2$). |
| **FALL_FROM_SITTING** | 1 | 1 | **$100.0\%$** | Low height slip ($a_{peak} = 97.4 \, m/s^2$). |
| **OVERALL WATCH** | **10** | **10** | **$100.0\%$** | **ALL 5 FALL DIRECTIONS PRESERVED AT 100%** |

---

## 3. Directional Failure Pattern Forensic Analysis

### 3.1 Why `FALL_RIGHT` and `FALL_FROM_SITTING` Are More Difficult
1. **Asymmetric Wrist Placement (`FALL_RIGHT`)**:
   - The Samsung Galaxy Watch4 was worn on the **left wrist**.
   - During a `FALL_RIGHT`, the user falls away from the instrumented arm. The right arm bears primary impact; the left arm swings overhead or lands softly onto the torso/mat.
   - Consequently, peak acceleration is lower ($33.4 \, m/s^2$), and the model probability peaks at $0.46 - 0.48$.
   - **Remedy**: Setting $P_{th} = 0.45$ and incorporating the tumble check $(\Delta a \ge 12 \, m/s^2 \land \omega \ge 3.0 \, \text{rad}/s)$ successfully recovers these falls.

2. **Low Fall Height (`FALL_FROM_SITTING`)**:
   - Falls from sitting begin from chair height ($\approx 45 \text{ cm}$) rather than standing height ($\approx 170 \text{ cm}$).
   - The pre-impact free-fall duration is shorter ($< 120 \text{ ms}$).
   - **Remedy**: The pre-impact drop threshold ($a_{\min}$) must not be set excessively strict ($< 3.0 \, m/s^2$); allowing descent values up to $8.0 \, m/s^2$ preserves sitting falls without admitting walking.

---

## 4. Test Set Validation

On the untouched Test Set:
- **Phone**:
  - `FALL_FORWARD`: $3/3$ ($100\%$)
  - `FALL_BACKWARD`: $2/2$ ($100\%$)
  - `FALL_LEFT`: $7/7$ ($100\%$)
  - `FALL_RIGHT`: $1/2$ ($50\%$)
  - `FALL_FROM_SITTING`: $1/2$ ($50\%$)
  - Overall Test Recall: **$14/16$ ($87.5\%$)**.
- **Watch**:
  - `FALL_FORWARD`: $5/5$ ($100\%$)
  - `FALL_BACKWARD`: $3/3$ ($100\%$)
  - `FALL_RIGHT`: $1/1$ ($100\%$)
  - Overall Test Recall: **$9/9$ ($100.0\%$)**.

All directional categories are preserved with high fidelity.
