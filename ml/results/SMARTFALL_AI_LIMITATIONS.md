# SMARTFALL AI — RESEARCH LIMITATIONS & SAFETY STATEMENT

## 1. Research Prototype Disclosure
SmartFall AI is an academic research prototype for real-time edge fall detection. It has NOT undergone clinical trials, FDA/CE medical device certification, or production emergency compliance certification.

## 2. Technical & Experimental Limitations
1. **Controlled Fall Simulations**:
   - Fall data collection was conducted under laboratory conditions using protective crash mats. Real-world unscripted geriatric falls involve complex pre-fall slip/trip dynamics, muscle stiffness, and post-fall unconsciousness that may introduce domain shift.
2. **Device Placement Invariance**:
   - Watch models assume the smartwatch is snugly worn on the wrist.
   - Phone models assume the smartphone is carried in a front or rear trouser pocket. Irregular placements (e.g., loose inside a backpack or handbag) alter sensor kinematics.
3. **Slow Slump Falls**:
   - Falls with minimal vertical impact velocity (e.g., slowly sliding down a wall into sitting) produce lower peak acceleration than dynamic impact falls.
4. **Physical Trial Sample Size**:
   - Physical device validation (Phase 9) validated 5 controlled physical fall trials per device. While 100% detection was achieved with 0 false alarms, this is a small-sample physical verification and not statistically equivalent to the 2,629-window offline test set.
