# SmartFall AI — Phase 13D: Calibration Specification

**Scope:** Formal engineering calibration parameters implemented in `FallInferenceEngine.kt` for Phone and Watch.

---

## 1. Phone Calibration Parameters (`SM-A507FN`)

```kotlin
// Raw Kinematic Collision Shock Gates:
val isCollisionShock = (maxAccMag >= 20.0f && maxJerk >= 350.0f) ||
                       (accRange >= 14.0f && maxJerk >= 250.0f && maxGyroMag >= 3.5f)

// Locomotion Cadence Rejection:
val isLocomotionCadence = (accStd >= 3.2f && maxGyroMag >= 3.5f) || (accStd >= 5.0f)

// Post-Impact Immobility / Recumbency Verification:
val isSettledImmobility = (accStd <= 2.4f) && (maxGyroMag <= 2.2f)
val hasFallPosture = (fallProb >= 0.40f) || (lyingDownProb >= 0.45f && accStd <= 1.8f)

// Verification Horizon:
val verificationHorizon = 4 // sliding windows (2.0 to 3.0 seconds post-impact)
```

---

## 2. Watch Calibration Parameters (`SM-R870`)

```kotlin
// Raw Kinematic Collision Shock Gates:
val isCollisionShock = (maxAccMag >= 24.0f && maxJerk >= 500.0f) ||
                       (accRange >= 16.0f && maxJerk >= 350.0f && maxGyroMag >= 4.0f)

// Locomotion Cadence Rejection:
val isLocomotionCadence = (accStd >= 5.5f && maxGyroMag >= 4.0f) || (accStd >= 8.0f)

// Post-Impact Immobility / Recumbency Verification:
val isSettledImmobility = (accStd <= 3.8f) && (maxGyroMag <= 3.2f)
val hasFallPosture = (fallProb >= 0.40f) || (lyingDownProb >= 0.45f && accStd <= 2.0f)

// Verification Horizon:
val verificationHorizon = 4 // sliding windows (2.0 to 3.0 seconds post-impact)
```

---

## 3. Parameter Derivation Justification

1. **Jerk Peak ($350 - 500 \, m/s^3$)**:
   Derived from empirical distributions where uncontrolled falls average $2,146 \, m/s^3$ while controlled walking heel strikes average $54.8 \, m/s^3$.
2. **Acceleration Peak ($20.0 - 24.0 \, m/s^2$)**:
   Set above walking peaks ($15 - 17 \, m/s^2$) while capturing all genuine falls ($33 - 135 \, m/s^2$).
3. **Locomotion Variance Gate ($\sigma_a \ge 3.2 - 5.5 \, m/s^2$)**:
   Derived from continuous foot-strike cycles where running maintains $\sigma_a > 8.0 \, m/s^2$.
4. **Verification Horizon ($4 \text{ windows}$)**:
   Accounts for human biomechanical rebound time ($1.0 - 2.0 \text{ seconds}$) before muscular collapse onto the floor.
