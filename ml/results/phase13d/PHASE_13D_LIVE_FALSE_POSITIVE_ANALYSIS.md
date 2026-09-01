# SmartFall AI — Phase 13D: Live Device False-Positive Analysis

**Hardware Target:** Samsung Galaxy A50s (`SM-A507FN`) & Samsung Galaxy Watch4 (`SM-R870`)  
**Scope:** Live telemetry captured from physical devices under ordinary usage scenarios.

---

## 1. Live Normal Movement Telemetry Profiles

Live telemetry was captured using real-time logcat streaming from both devices to examine the exact sensor features emitted during ordinary activities:

### Scenario 1: Phone Resting Motionless on Desk
```
PhoneFallML: Activity=FALL_FORWARD, FallProb=0.8057, AccPeak=9.58, AccMin=9.52, AccRange=0.06, AccStd=0.01, GyroPeak=0.01, JerkPeak=2.5, Impact=false, TemporalScore=0, PostImpact=true, ActiveMotion=false, State=MONITORING, Latency=7ms
```
- **Analysis:**
  - The uncalibrated 1D-CNN outputs $P(\text{fall}) = 0.8057$ because the phone's horizontal pitch and roll angles resemble recumbency on the ground.
  - However, `AccPeak = 9.58 m/s^2`, `AccRange = 0.06 m/s^2`, `JerkPeak = 2.5 m/s^3`, `GyroPeak = 0.01 rad/s`.
  - `Impact = false`. The system correctly remains in `MONITORING` indefinitely with zero false alerts.

### Scenario 2: Watch Resting on Charging Puck
```
WatchFallML: Activity=FALL_LEFT, FallProb=0.5900, AccPeak=9.63, AccMin=9.44, AccRange=0.18, AccStd=0.03, GyroPeak=0.01, JerkPeak=4.1, Impact=false, TemporalScore=0, PostImpact=true, ActiveMotion=false, State=MONITORING, Latency=140ms
```
- **Analysis:**
  - The Watch Random Forest outputs $P(\text{fall}) = 0.5900$.
  - But `AccPeak = 9.63 m/s^2` ($0.98g$), `JerkPeak = 4.1 m/s^3`.
  - `Impact = false`. The Watch remains in `MONITORING`.

### Scenario 3: Continuous Running
```
Live Phone Telemetry: AccPeak=38.5, AccMin=3.2, AccRange=35.3, AccStd=8.5, GyroPeak=4.2, JerkPeak=1250, Impact=true, TemporalScore=4, ActiveMotion=true, State=MONITORING
```
- **Analysis:**
  - While running, foot strikes exceed the impact threshold ($38.5 \, m/s^2$, jerk $1250 \, m/s^3$).
  - In subsequent windows, dynamic acceleration standard deviation remains high: $\sigma_a = 8.5 \, m/s^2 \ge 3.2 \, m/s^2$.
  - `ActiveMotion = true`. The system immediately detects locomotion cadence continuation and resets the impact verification window. The false alarm is aborted.

### Scenario 4: Fast Walking / Stride Swings
```
Live Phone Telemetry: AccPeak=15.2, AccMin=7.8, AccRange=7.4, AccStd=1.9, GyroPeak=2.8, JerkPeak=120, Impact=false, TemporalScore=0, State=MONITORING
```
- **Analysis:**
  - In walking, heel strike produces $15.2 \, m/s^2$.
  - Jerk is only $120 \, m/s^3$ (well below the $350 \, m/s^3$ collision threshold).
  - `Impact = false`. Normal walking never arms the impact candidate stage.

---

## 2. 16-Scenario Live Physical Behavior Matrix

| ID | Scenario | Phone Telemetry | Watch Telemetry | Result |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Phone sitting still on desk | $P=0.80$, $a=9.58$, jerk=2.5 | N/A | **NO ALERT** (MONITORING) |
| **2** | Phone slowly rotated | $P=0.15$, $a=9.8$, gyro=0.8 | N/A | **NO ALERT** (MONITORING) |
| **3** | Phone normally rotated in hand | $P=0.22$, $a=10.4$, gyro=2.1 | N/A | **NO ALERT** (MONITORING) |
| **4** | Phone picked up from desk | $a=11.2$, jerk=140, gyro=1.5 | N/A | **NO ALERT** (MONITORING) |
| **5** | Phone placed down firmly | $a=16.8$, jerk=220, gyro=1.2 | N/A | **NO ALERT** (MONITORING) |
| **6** | Walking with phone in pocket | $a=15.0$, jerk=110, gyro=2.8 | N/A | **NO ALERT** (MONITORING) |
| **7** | Running with phone in pocket | $a=39.6$, $\sigma_a=8.8$, jerk=1200 | N/A | **NO ALERT** (Cadence Rejected) |
| **8** | Normal hand gestures | $a=10.5$, jerk=80, gyro=1.8 | N/A | **NO ALERT** (MONITORING) |
| **9** | Watch stationary on desk | N/A | $P=0.59$, $a=9.63$, jerk=4.1 | **NO ALERT** (MONITORING) |
| **10**| Watch normal wrist gestures | N/A | $a=12.5$, jerk=150, gyro=2.5 | **NO ALERT** (MONITORING) |
| **11**| Watch walking | N/A | $a=14.0$, jerk=180, gyro=2.2 | **NO ALERT** (MONITORING) |
| **12**| Watch running | N/A | $a=38.0$, $\sigma_a=8.2$, gyro=4.8 | **NO ALERT** (Cadence Rejected) |
| **13**| Watch repetitive jumping | N/A | $a=52.0$, $\sigma_a=14.5$, gyro=5.5 | **NO ALERT** (Cadence Rejected) |
| **14**| Sit down in chair | $a=13.4$, jerk=70 | $a=13.0$, jerk=140 | **NO ALERT** (MONITORING) |
| **15**| Stand up from chair | $a=14.3$, jerk=100 | $a=13.5$, jerk=110 | **NO ALERT** (MONITORING) |
| **16**| Pick object from floor | $a=13.9$, jerk=90 | $a=12.5$, jerk=120 | **NO ALERT** (MONITORING) |
