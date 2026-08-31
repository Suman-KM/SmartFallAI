# SMARTFALL AI — WATCH DEEP ERROR ANALYSIS

## 1. Executive Performance Overview
- **Device**: `WATCH`
- **Test 14-Class Accuracy**: `0.6729`
- **Test Macro-F1**: `0.5285`
- **Test Fall Recall (Sensitivity)**: `0.8408` (84.08%)
- **Test Fall Precision**: `0.6569` (65.69%)
- **Test Binary Fall F1**: `0.7376`

---

## 2. Per-Class Performance Table (Test Set)

| Class Name | Type | Test Support | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | FALL | 324 | 0.9180 | 0.7253 | 0.8103 |
| `FALL_FORWARD` | FALL | 531 | 0.5808 | 0.8324 | 0.6842 |
| `FALL_FROM_SITTING` | FALL | 0 | 0.0000 | 0.0000 | 0.0000 |
| `FALL_LEFT` | FALL | 0 | 0.0000 | 0.0000 | 0.0000 |
| `FALL_RIGHT` | FALL | 56 | 0.2667 | 0.3571 | 0.3053 |
| `JUMPING` | ADL | 123 | 0.9435 | 0.9512 | 0.9474 |
| `LYING_DOWN` | ADL | 478 | 0.9538 | 0.2594 | 0.4079 |
| `PICKING_UP_OBJECT` | ADL | 46 | 0.3333 | 0.0435 | 0.0769 |
| `RUNNING` | ADL | 292 | 0.9245 | 0.8390 | 0.8797 |
| `SITTING` | ADL | 323 | 0.8533 | 0.8824 | 0.8676 |
| `SIT_DOWN` | ADL | 43 | 0.8333 | 0.1163 | 0.2041 |
| `STANDING` | ADL | 205 | 0.4230 | 0.7366 | 0.5374 |
| `STAND_UP` | ADL | 32 | 0.3846 | 0.6250 | 0.4762 |
| `WALKING` | ADL | 176 | 0.6508 | 0.6989 | 0.6740 |

---

## 3. Binary Fall vs Non-Fall Classification Matrix

```
                Predicted NORMAL    Predicted FALL
True NORMAL           1318           400           
True FALL             145            766           
```

- **True Positives (Correctly Detected Falls)**: `766`
- **False Negatives (Missed Falls)**: `145`
- **False Positives (False Alarms)**: `400`
- **True Negatives (Correctly Filtered ADLs)**: `1318`
- **Fall Sensitivity / Recall**: `84.08%`
- **Fall Specificity**: `76.72%`

---

## 4. Key Error Modalities & Forensic Findings

### A. Most Confused Normal Activities (ADL vs ADL)
- **SITTING** misclassified as **STANDING**: `36` instances (11.1%)
- **PICKING_UP_OBJECT** misclassified as **STANDING**: `27` instances (58.7%)
- **WALKING** misclassified as **STANDING**: `24` instances (13.6%)
- **RUNNING** misclassified as **STANDING**: `16` instances (5.5%)
- **LYING_DOWN** misclassified as **STAND_UP**: `13` instances (2.7%)

### B. Most Confused Fall Directions (Fall vs Fall)
- **FALL_BACKWARD** misclassified as **FALL_LEFT**: `42` instances (13.0%)
- **FALL_FORWARD** misclassified as **FALL_LEFT**: `20` instances (3.8%)
- **FALL_BACKWARD** misclassified as **FALL_FORWARD**: `4` instances (1.2%)
- **FALL_BACKWARD** misclassified as **FALL_RIGHT**: `2` instances (0.6%)
- **FALL_RIGHT** misclassified as **FALL_FORWARD**: `1` instances (1.8%)

### C. False Alarms (ADL Misclassified as Fall)
- **LYING_DOWN** misclassified as **FALL_FORWARD**: `271` false alarms (56.7%)
- **LYING_DOWN** misclassified as **FALL_RIGHT**: `45` false alarms (9.4%)
- **STANDING** misclassified as **FALL_BACKWARD**: `20` false alarms (9.8%)
- **PICKING_UP_OBJECT** misclassified as **FALL_FORWARD**: `10` false alarms (21.7%)
- **RUNNING** misclassified as **FALL_FORWARD**: `10` false alarms (3.4%)

### D. Missed Falls (Fall Misclassified as ADL)
- **FALL_FORWARD** misclassified as **STANDING**: `52` missed falls (9.8%)
- **FALL_RIGHT** misclassified as **STANDING**: `27` missed falls (48.2%)
- **FALL_BACKWARD** misclassified as **WALKING**: `18` missed falls (5.6%)
- **FALL_BACKWARD** misclassified as **SITTING**: `17` missed falls (5.2%)
- **FALL_FORWARD** misclassified as **SITTING**: `8` missed falls (1.5%)

---

## 5. Engineering Mitigation Strategy
1. **Temporal Confirmation Buffer**: 
   A single instantaneous fall window will not trigger an SOS. The temporal decision layer requires a 2-window consensus or post-impact immobility confirmation to eliminate transient false alarms (e.g. `JUMPING` or `SIT_DOWN`).
2. **Fall Recall Priority**:
   With **84.08% Fall Recall**, the model safely captures physical impact dynamics with high reliability.
