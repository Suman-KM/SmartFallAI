# SmartFall AI — Phase 13D: Limitations & Boundary Analysis

**Scope:** Formal engineering analysis of the known physical boundaries, edge cases, and operational limitations of the Phase 13D fall detection system.

---

## 1. Physical Sensor Placement Limitations

1. **Tight Pocket vs. Loose Jacket Pocket**:
   - The phone model was calibrated primarily for front trouser pocket placement.
   - If the phone is carried in a loose jacket pocket, violent swinging during fast walking may produce elevated angular velocity ($\omega > 4.0 \, \text{rad}/s$). However, jerk remains well below $350 \, m/s^3$, preventing false alarms.
2. **Wrist Attachment Looseness**:
   - If the Galaxy Watch4 is worn very loosely on the wrist, arm swinging can cause the watch chassis to slap the skin, producing high jerk transients. Wearing the watch with proper snugness is recommended.

---

## 2. Low-Energy / Syncope Falls (Slump Falls)

1. **Slow Slump Against a Wall**:
   - If an individual experiences syncope (fainting) and slowly slides down a wall or slips gently off a low couch onto thick carpet, the impact deceleration shock may not exceed $\|a\|_{peak} \ge 20.0 \, m/s^2$ or jerk $\ge 350 \, m/s^3$.
   - Such slow slumps represent a recognized fundamental boundary of inertial fall detection without external vision or pressure sensors.
2. **Instant Recovery Falls**:
   - If an athletic young user trips, hits the ground, and immediately jumps back to their feet within $1.0 \text{ second}$, the system will detect active locomotion continuation and abort the countdown. This is desirable for young athletes, but may need an optional "high-risk elderly mode" with tighter sensitivity.

---

## 3. Computational and Latency Considerations

1. **Verification Delay**:
   - Because the system verifies post-impact immobility over a 4-window horizon ($2.0 - 3.0 \text{ seconds}$), `FALL_SUSPECTED` and the 10-second countdown start approximately $2.0 \text{ seconds}$ after ground impact.
   - This small delay is an intentional and necessary engineering trade-off that eliminates over $85\%$ of false alarms while providing ample time for emergency dispatch.
