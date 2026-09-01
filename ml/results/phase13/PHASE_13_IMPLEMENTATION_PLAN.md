# SMARTFALL AI — PHASE 13 IMPLEMENTATION PLAN
## Real-World Manual Fall Detection + Countdown + Emergency Gmail Validation

---

## 1. Current Architecture & Fall Detection Flow

### Components:
- **Watch**: Samsung Galaxy Watch 4 (`SM-R870`), Wear OS 4.0.
  - Sensors: `WatchSensorManager` streams accelerometer, gyroscope, orientation at ~50 Hz.
  - Preprocessing: `WatchRobustScaler` (median/IQR from Train set) in-place scaling.
  - Feature Extraction: `WatchFeatureExtractor` generates 72 statistical summary features per 100-sample (2.0s) window.
  - ML Engine: `WatchRandomForestEngine` evaluates 100 decision trees natively via `trees.bin` in pure Kotlin flat primitive arrays.
  - Inference Cadence: 100 samples window, 50 samples stride (~1.0s).
- **Phone**: Samsung Galaxy A50s (`SM-A507FN`), Android 11 (API 30).
  - Sensors: `PhoneSensorManager` streams 9-DoF IMU.
  - Preprocessing: `PhoneRobustScaler` in-place scaling.
  - ML Engine: `PhoneOnnxEngine` executes self-contained 1D-CNN (`model.onnx`, 164.7 KB) via Microsoft ONNX Runtime Mobile.
  - Inference Cadence: 100 samples window, 50 samples stride (~1.0s).

### Current Fall State Progression:
1. IMU samples enter circular buffer (100 samples).
2. Model calculates 14-class probability vector.
3. Fall probability $P(\text{fall}) = \sum_{i=0}^4 P_i$.
4. If $P(\text{fall}) \ge 0.50$, window is flagged as an instantaneous fall candidate.
5. In current Phase 8/9 code:
   - Window 1: `MONITORING` $\rightarrow$ `FALL_SUSPECTED`
   - Window 2: `FALL_SUSPECTED` $\rightarrow$ `FALL_CONFIRMED`
   - Window 3: `FALL_CONFIRMED` $\rightarrow$ `SOS_TRIGGERED`

---

## 2. Current SOS & Countdown Behavior (Gap Analysis)

### Deficiencies in Current Implementation:
1. **No Interactive Countdown UI**:
   - Currently, when `FALL_SUSPECTED` is reached, the state transitions automatically into `FALL_CONFIRMED` and `SOS_TRIGGERED` on the next inference stride without presenting an interactive countdown UI with a visible timer.
   - The user has no way to press "I'M OK" to cancel a false alarm before emergency escalation.
2. **Missing Notification Delivery**:
   - `EmergencyManager.kt` on both Phone and Watch is an empty class stub (`class EmergencyManager { }`).
   - No email or emergency dispatch actually occurs when `SOS_TRIGGERED` is reached.
3. **Missing Internet & Notification Permissions**:
   - Neither `app/src/main/AndroidManifest.xml` nor `wear/src/main/AndroidManifest.xml` declares `android.permission.INTERNET`.
4. **No Watch-to-Phone Emergency Escalation**:
   - If the Watch detects a fall in standalone mode or connected mode, there is no message payload notifying the Phone to dispatch an emergency email to `sumankmdvg@gmail.com`.

---

## 3. Proposed Minimal Changes

### A. Fall State Machine & Countdown Timer (`FallInferenceEngine.kt` / `EmergencyManager.kt`):
- When a fall is detected:
  1. State becomes `FallState.FALL_SUSPECTED`.
  2. A 10-second countdown timer starts (`countdownRemaining = 10`).
  3. Haptic feedback (vibration) alerts the user.
  4. If the user taps `"I'M OK"`:
     - Countdown immediately stops and cancels.
     - State transitions to `FallState.CANCELLED` and returns to `FallState.MONITORING`.
     - No SOS is triggered, zero notification is sent.
     - Buffer resets cleanly so normal monitoring continues uninterrupted.
  5. If countdown reaches `0` without cancellation:
     - State transitions to `FallState.FALL_CONFIRMED`.
     - State transitions to `FallState.SOS_TRIGGERED`.
     - `EmergencyManager` triggers the emergency notification pipeline.

### B. Emergency Recipient & Notification Dispatch:
- Target Recipient: **`sumankmdvg@gmail.com`**
- **Security Guarantee**: ZERO passwords, app passwords, SMTP credentials, or API keys are stored in code or repository.
- **Dispatch Mechanism**:
  1. **Phone Transport**:
     - System `ACTION_SENDTO` Intent with `mailto:sumankmdvg@gmail.com`, subject `SMARTFALL AI — FALL DETECTED`, and detailed body identifying device, timestamp, and confirmed fall status.
     - Local emergency notification / high-priority channel alert.
  2. **Watch Transport**:
     - Local high-priority alert & persistent vibration.
     - If connected to Phone via `Wearable.getMessageClient`, transmits `/smartfall/sos_triggered` with timestamp and device ID so the companion Phone immediately launches the emergency dispatch.
     - If standalone, logs and shows local emergency screen with SOS alert dispatched flag.

### C. Interactive Countdown & Alert UI (`SmartFallScreen.kt`):
- **Phone UI**:
  - Displays full-screen or prominent overlay when state is `FALL_SUSPECTED`:
    - ⚠️ **POSSIBLE FALL DETECTED**
    - "Are you okay?"
    - Large visible countdown timer: `10, 9, 8...`
    - Prominent `[ I'M OK ]` button (Green / high contrast).
  - When `SOS_TRIGGERED`:
    - 🚨 **EMERGENCY SOS TRIGGERED**
    - "Alert dispatched to sumankmdvg@gmail.com"
- **Wear OS UI**:
  - Wear-optimized full-screen dialog / card:
    - ⚠️ **FALL DETECTED**
    - "Emergency in: **10s**"
    - `[ I'M OK ]` button sized for watch touch targets.

---

## 4. Test Procedure & Validation Matrix

| Test Suite | Conditions | Expected Outcome |
|---|---|---|
| **Manual Cancellation (x3)** | Safe motion $\rightarrow$ `FALL_SUSPECTED` $\rightarrow$ Tap "I'M OK" | Countdown halts, returns to `MONITORING`, zero SOS, zero email |
| **No-Response SOS (x3)** | Safe motion $\rightarrow$ `FALL_SUSPECTED` $\rightarrow$ Allow 10s to expire | `FALL_CONFIRMED` $\rightarrow$ `SOS_TRIGGERED` $\rightarrow$ Email sent to `sumankmdvg@gmail.com` |
| **Normal Activities (x9)** | `SITTING, STANDING, WALKING, RUNNING, LYING_DOWN, JUMPING, SIT_DOWN, STAND_UP, PICKING_UP` | Zero false SOS, zero unwanted emails |
| **Controlled Falls (x5)** | `FALL_FORWARD, FALL_BACKWARD, FALL_LEFT, FALL_RIGHT, FALL_FROM_SITTING` | Fall suspected $\rightarrow$ Countdown triggers $\rightarrow$ SOS |
| **Standalone Watch** | Phone disconnected/Bluetooth off | Local inference, countdown, and local SOS execute autonomously |
| **Standalone Phone** | Watch disconnected/Bluetooth off | Local inference, countdown, and email dispatch execute autonomously |
| **Screen Wake** | Device idle during active session | `FLAG_KEEP_SCREEN_ON` maintains screen wakefulness |
| **Sensor Interruption** | Sensor pauses or app backgrounded | Graceful recovery, zero crashes, zero spurious falls |

---

## 5. Failure Criteria & Thresholds
1. **Critical Failure (NO-GO)**:
   - App crashes during countdown or SOS dispatch.
   - User presses "I'M OK" but emergency email is sent anyway.
   - Countdown expires but emergency dispatch is not attempted.
   - Any credential or password committed to Git.
   - Screen sleeps/locks during active countdown preventing user response.
2. **Acceptable Behavior**:
   - High-motion activities (`JUMPING`) may occasionally trigger `FALL_SUSPECTED`, but 10-second countdown gives user ample time to tap "I'M OK".
