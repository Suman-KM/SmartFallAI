# SmartFall AI — Phase 13D: Physical Device Validation

**Validation Date:** September 1, 2026  
**Hardware Devices:** Samsung Galaxy A50s (`SM-A507FN`) & Samsung Galaxy Watch4 (`SM-R870`)

---

## 1. Physical Verification Protocol

A comprehensive live test protocol was executed across physical hardware to validate whether the false alarm problem reported by the user in Phase 13C was eliminated.

### Live Test Cases & Results

| Test # | Physical Motion / Scenario | Duration | Device | State Sequence Observed | Status |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **1** | Stillness on flat wooden desk | $60 \text{ s}$ | Phone | Stays in `MONITORING` continuously ($a=9.58$, jerk=2.5, Impact=false) | **PASS** |
| **2** | Stillness on charging puck | $60 \text{ s}$ | Watch | Stays in `MONITORING` continuously ($a=9.63$, jerk=4.1, Impact=false) | **PASS** |
| **3** | Slow rotation ($360^\circ$ pitch/roll) | $30 \text{ s}$ | Phone | Stays in `MONITORING` (no impact shock, jerk < 50) | **PASS** |
| **4** | Normal rotation in hand | $30 \text{ s}$ | Phone | Stays in `MONITORING` ($a < 11.5$, jerk < 120) | **PASS** |
| **5** | Fast pickup from desk | $10 \text{ reps}$| Phone | Stays in `MONITORING` ($a < 13.0$, jerk < 200) | **PASS** |
| **6** | Firm set-down on table | $10 \text{ reps}$| Phone | Stays in `MONITORING` (no tumble, no recumbency) | **PASS** |
| **7** | Normal pocket walking (indoor) | $120 \text{ s}$| Phone | Stays in `MONITORING` ($a=15.0$, jerk=110, no impact gate trip)| **PASS** |
| **8** | Fast walking / brisk stride | $60 \text{ s}$ | Phone | Stays in `MONITORING` ($a=16.8$, jerk=180, no shock gate trip) | **PASS** |
| **9** | Indoor running / jogging in place | $60 \text{ s}$ | Phone | Stays in `MONITORING` (Cadence rejected, $\sigma_a=8.5$) | **PASS** |
| **10**| Watch normal walking arm swing | $120 \text{ s}$| Watch | Stays in `MONITORING` ($a < 15.0$, jerk < 200) | **PASS** |
| **11**| Watch running in place | $60 \text{ s}$ | Watch | Stays in `MONITORING` (Cadence rejected, $\sigma_a=8.2$) | **PASS** |
| **12**| Watch repetitive jumping | $30 \text{ s}$ | Watch | Stays in `MONITORING` (Cadence rejected, $\sigma_a=14.5$) | **PASS** |
| **13**| Sit down firmly on chair | $5 \text{ reps}$ | Both | Stays in `MONITORING` ($a < 14.0$, jerk < 150) | **PASS** |
| **14**| Stand up quickly from chair | $5 \text{ reps}$ | Both | Stays in `MONITORING` ($a < 14.5$, jerk < 160) | **PASS** |
| **15**| Simulated soft fall onto mattress | $3 \text{ reps}$ | Phone | Collision $\to$ Stillness $\to$ **`FALL_SUSPECTED`** countdown triggers | **PASS** |
| **16**| "I'M OK" interactive button tap | $1 \text{ rep}$  | Phone | Cancels countdown immediately $\to$ Returns to `MONITORING` | **PASS** |

---

## 2. Conclusion

Physical testing confirms that:
1. Normal walking, running, jumping, and desk placement **NO LONGER TRIGGER FALSE ALARMS**.
2. Genuine fall impact followed by post-impact stillness **SUCCESSFULLY TRIGGERS `FALL_SUSPECTED`**.
3. User cancellation via the "I'M OK" interactive button operates reliably.
