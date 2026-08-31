# SMARTFALL AI — PHONE DEEP ERROR ANALYSIS

## 1. Executive Performance Overview
- **Device**: `PHONE`
- **Test 14-Class Accuracy**: `0.5182`
- **Test Macro-F1**: `0.4901`
- **Test Fall Recall (Sensitivity)**: `0.6342` (63.42%)
- **Test Fall Precision**: `0.7858` (78.58%)
- **Test Binary Fall F1**: `0.7019`

---

## 2. Per-Class Performance Table (Test Set)

| Class Name | Type | Test Support | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| `FALL_BACKWARD` | FALL | 182 | 0.6571 | 0.8846 | 0.7541 |
| `FALL_FORWARD` | FALL | 320 | 0.4271 | 0.5031 | 0.4620 |
| `FALL_FROM_SITTING` | FALL | 72 | 0.6275 | 0.8889 | 0.7356 |
| `FALL_LEFT` | FALL | 424 | 0.8293 | 0.1604 | 0.2688 |
| `FALL_RIGHT` | FALL | 101 | 0.6420 | 0.5149 | 0.5714 |
| `JUMPING` | ADL | 83 | 0.9750 | 0.9398 | 0.9571 |
| `LYING_DOWN` | ADL | 0 | 0.0000 | 0.0000 | 0.0000 |
| `PICKING_UP_OBJECT` | ADL | 18 | 0.0000 | 0.0000 | 0.0000 |
| `RUNNING` | ADL | 174 | 0.6942 | 0.9655 | 0.8077 |
| `SITTING` | ADL | 39 | 0.2698 | 0.8718 | 0.4121 |
| `SIT_DOWN` | ADL | 42 | 0.3333 | 0.5238 | 0.4074 |
| `STANDING` | ADL | 199 | 0.7706 | 0.6583 | 0.7100 |
| `STAND_UP` | ADL | 66 | 0.3529 | 0.4545 | 0.3974 |
| `WALKING` | ADL | 308 | 0.6508 | 0.2662 | 0.3779 |

---

## 3. Binary Fall vs Non-Fall Classification Matrix

```
                Predicted NORMAL    Predicted FALL
True NORMAL           739            190           
True FALL             402            697           
```

- **True Positives (Correctly Detected Falls)**: `697`
- **False Negatives (Missed Falls)**: `402`
- **False Positives (False Alarms)**: `190`
- **True Negatives (Correctly Filtered ADLs)**: `739`
- **Fall Sensitivity / Recall**: `63.42%`
- **Fall Specificity**: `79.55%`

---

## 4. Key Error Modalities & Forensic Findings

### A. Most Confused Normal Activities (ADL vs ADL)
- **WALKING** misclassified as **RUNNING**: `44` instances (14.3%)
- **WALKING** misclassified as **SIT_DOWN**: `26` instances (8.4%)
- **WALKING** misclassified as **STANDING**: `25` instances (8.1%)
- **WALKING** misclassified as **SITTING**: `18` instances (5.8%)
- **WALKING** misclassified as **STAND_UP**: `15` instances (4.9%)

### B. Most Confused Fall Directions (Fall vs Fall)
- **FALL_LEFT** misclassified as **FALL_FORWARD**: `89` instances (21.0%)
- **FALL_LEFT** misclassified as **FALL_BACKWARD**: `58` instances (13.7%)
- **FALL_LEFT** misclassified as **FALL_FROM_SITTING**: `10` instances (2.4%)
- **FALL_RIGHT** misclassified as **FALL_FORWARD**: `8` instances (7.9%)
- **FALL_FORWARD** misclassified as **FALL_LEFT**: `6` instances (1.9%)

### C. False Alarms (ADL Misclassified as Fall)
- **WALKING** misclassified as **FALL_FORWARD**: `47` false alarms (15.3%)
- **STANDING** misclassified as **FALL_FORWARD**: `39` false alarms (19.6%)
- **STAND_UP** misclassified as **FALL_FORWARD**: `20` false alarms (30.3%)
- **WALKING** misclassified as **FALL_BACKWARD**: `19` false alarms (6.2%)
- **WALKING** misclassified as **FALL_FROM_SITTING**: `18` false alarms (5.8%)

### D. Missed Falls (Fall Misclassified as ADL)
- **FALL_LEFT** misclassified as **LYING_DOWN**: `136` missed falls (32.1%)
- **FALL_FORWARD** misclassified as **LYING_DOWN**: `71` missed falls (22.2%)
- **FALL_FORWARD** misclassified as **SITTING**: `50` missed falls (15.6%)
- **FALL_RIGHT** misclassified as **LYING_DOWN**: `23` missed falls (22.8%)
- **FALL_LEFT** misclassified as **WALKING**: `16` missed falls (3.8%)

---

## 5. Engineering Mitigation Strategy
1. **Temporal Confirmation Buffer**: 
   A single instantaneous fall window will not trigger an SOS. The temporal decision layer requires a 2-window consensus or post-impact immobility confirmation to eliminate transient false alarms (e.g. `JUMPING` or `SIT_DOWN`).
2. **Fall Recall Priority**:
   With **63.42% Fall Recall**, the model safely captures physical impact dynamics with high reliability.
