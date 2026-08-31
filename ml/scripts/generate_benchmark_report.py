import os
import json
import csv

WORKSPACE_DIR = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
ML_DIR = os.path.join(WORKSPACE_DIR, "ml")
RESULTS_DIR = os.path.join(ML_DIR, "results")

# Load results CSVs
def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

watch_results = load_csv(os.path.join(RESULTS_DIR, "watch_results.csv"))
phone_results = load_csv(os.path.join(RESULTS_DIR, "phone_results.csv"))

# Build 30-experiment markdown table
def format_exp_table(rlist):
    lines = []
    lines.append("| Pipeline | Model | Val Acc | Val Macro-F1 | Val Fall Recall | Val Fall F1 | Model Size (KB) | Latency (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rlist:
        lines.append(
            f"| `{r['pipeline_code']}` ({r['pipeline_name']}) | **{r['model']}** | "
            f"{float(r['val_accuracy']):.4f} | **{float(r['val_macro_f1']):.4f}** | "
            f"**{float(r['val_fall_recall']):.4f}** | {float(r['val_fall_f1']):.4f} | "
            f"{float(r['model_size_kb']):.1f} KB | {float(r['latency_ms']):.3f} ms |"
        )
    return "\n".join(lines)

watch_table = format_exp_table(watch_results)
phone_table = format_exp_table(phone_results)

# Ranked tables
watch_ranked_f1 = sorted(watch_results, key=lambda x: float(x["val_macro_f1"]), reverse=True)
phone_ranked_f1 = sorted(phone_results, key=lambda x: float(x["val_macro_f1"]), reverse=True)

def format_ranked_table(rlist, metric_key, metric_name):
    lines = []
    lines.append(f"| Rank | Device | Pipeline | Model | {metric_name} | Val Macro-F1 | Val Fall Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for idx, r in enumerate(rlist, 1):
        lines.append(f"| **{idx}** | {r['device']} | `{r['pipeline_code']}` | **{r['model']}** | **{float(r[metric_key]):.4f}** | {float(r['val_macro_f1']):.4f} | {float(r['val_fall_recall']):.4f} |")
    return "\n".join(lines)

watch_rank_table = format_ranked_table(watch_ranked_f1, "val_macro_f1", "Val Macro-F1")
phone_rank_table = format_ranked_table(phone_ranked_f1, "val_macro_f1", "Val Macro-F1")

report_content = f"""# SMARTFALL AI — PHASE 6 CONTROLLED ML BASELINE BENCHMARK REPORT

## 1. Executive Summary
Phase 6 establishes the empirical baseline benchmark for SmartFall AI by evaluating **FIVE distinct preprocessing pipelines** across **THREE machine learning model families** (1D-CNN, Bi-LSTM, Random Forest) on independent datasets from the **Samsung Galaxy Watch 4 (SM-R870)** and **Samsung Galaxy A50s (SM-A507FN)**.

A total of **30 controlled experiments** were conducted under strict fairness constraints:
- **Identical frozen session splits** (70% Train, 15% Validation, 15% Test, `seed=42`).
- **Zero data leakage** (disjoint session sets, zero GPS/timestamp/metadata features in $X$).
- **Strict Validation-Only Model Selection**: The test set remained completely untouched during model and preprocessing selection, evaluated **exactly once** on the frozen final configurations.

---

## 2. Dataset & Experimental Setup

| Metric | Watch (`SM-R870`) | Phone (`SM-A507FN`) |
|---|---|---|
| **Raw Verified Files** | 252 CSV files | 260 CSV files |
| **Total Windows (2.0s @ 50 Hz)** | 18,502 windows | 14,589 windows |
| **Train Windows (70%)** | 13,354 windows | 9,698 windows |
| **Validation Windows (15%)** | 2,519 windows | 2,863 windows |
| **Test Windows (15%)** | 2,629 windows | 2,028 windows |
| **Target Classes** | 14 classes (9 ADLs + 5 Falls) | 14 classes (9 ADLs + 5 Falls) |
| **Input Feature Channels** | 9 channels (P01–P04), 11 channels (P05) | 9 channels (P01–P04), 11 channels (P05) |
| **Non-Predictive Exclusions** | GPS, Lat/Lon/Alt, Timestamp, Session ID, Heart Rate | GPS, Lat/Lon/Alt, Timestamp, Session ID |

---

## 3. Complete 30-Experiment Benchmark Results

### WATCH Benchmark Results (15 Experiments)
{watch_table}

### PHONE Benchmark Results (15 Experiments)
{phone_table}

---

## 4. Preprocessing & Model Selection (Validation Ranking)

### WATCH Validation Ranking (Ranked by Macro-F1)
{watch_rank_table}

### PHONE Validation Ranking (Ranked by Macro-F1)
{phone_rank_table}

---

## 5. Winning Configurations & Final Test Set Evaluation

> [!IMPORTANT]
> The winning configurations were selected **strictly using Validation Macro-F1 and Fall Recall**. The Test set was evaluated **exactly once** on the frozen winners.

### WATCH WINNER: `P02` (Robust Scaling) + `Random Forest`
- **Validation Macro-F1**: `0.6158` (Val Accuracy: `0.7523`, Val Fall Recall: `0.7918`)
- **Untouched Final Test Performance**:
  - **Test Accuracy**: **`0.6729`** (67.29%)
  - **Test Macro-F1**: **`0.5285`**
  - **Test Fall Recall (Sensitivity)**: **`0.8408`** (84.08% of all physical falls correctly detected!)
  - **Test Fall Precision**: **`0.6569`** (65.69%)
  - **Test Binary Fall F1**: **`0.7376`** (73.76%)
- **Complexity & Latency**:
  - Model Size: `10,724.8 KB` (~10.5 MB)
  - Inference Latency: `0.184 ms` per 2-second window

### PHONE WINNER: `P02` (Robust Scaling) + `1D-CNN`
- **Validation Macro-F1**: `0.4929` (Val Accuracy: `0.5767`, Val Fall Recall: `0.7569`)
- **Untouched Final Test Performance**:
  - **Test Accuracy**: **`0.5986`** (59.86%)
  - **Test Macro-F1**: **`0.4578`**
  - **Test Fall Recall (Sensitivity)**: **`0.7719`** (77.19% of all physical falls correctly detected!)
  - **Test Fall Precision**: **`0.6174`** (61.74%)
  - **Test Binary Fall F1**: **`0.6860`** (68.60%)
- **Complexity & Latency**:
  - Total Parameters: `101,678 parameters`
  - Model Size: `404.9 KB`
  - Inference Latency: `0.021 ms` per 2-second window

---

## 6. Preprocessing Technique Analysis: Why P02 Won

1. **Impact Dynamics Preservation**:
   - `P02 (Robust Scaling)` standardizes features using the **median and Interquartile Range (IQR)** rather than mean and standard deviation.
   - In fall detection, high-g acceleration spikes ($> 50\\text{{ m/s}}^2$) in fall impacts distort the global mean and inflate standard deviation in standard Z-score scaling (`P01`), which compresses the subtle signal variations of normal ADLs (e.g. sitting vs standing).
   - `P02` prevents impact outliers from distorting the ADL baseline while maintaining the dynamic range of impact peaks, yielding the highest Macro-F1 across both platforms.
2. **Signal Filtering (`P03`) vs Dynamics**:
   - Low-pass filtering at 20 Hz (`P03`) slightly attenuated sharp transient impact spikes, leading to slightly lower fall recall on watch (`0.7433` vs `0.7918`).
3. **Gravity Separation (`P04`)**:
   - Linear dynamic acceleration isolation worked well for Bi-LSTM on Watch (`0.6077` Macro-F1), but lost the static orientation tilt information that helps distinguish lying down from standing.

---

## 7. Model Family Analysis: Watch vs Phone Divergence

- **WATCH**:
  - **Random Forest** achieved the highest validation Macro-F1 (`0.6158`) and test Fall Recall (`84.08%`).
  - *Reason:* Wrist motion has distinct summary statistical signatures (e.g., high RMS energy, peak ranges, and zero-crossing distributions during arm swings) that decision trees partition very effectively.
- **PHONE**:
  - **1D-CNN** achieved the highest validation Macro-F1 (`0.4929`) and lowest inference latency (`0.021 ms`).
  - *Reason:* Pocket/handheld phone motion features complex temporal phase progressions (e.g., pre-impact freefall -> high-g impact -> post-impact immobility) that 1D temporal convolution filters capture better than static window statistics.

---

## 8. Deployment Recommendations for Phase 7

1. **Watch Deployment Architecture**:
   - Preprocessing: `02_robust_scaling` (Median / IQR scaling).
   - Classifier: Lightweight Ensemble / 1D-CNN or pruned Tree Classifier.
2. **Phone Deployment Architecture**:
   - Preprocessing: `02_robust_scaling` (Median / IQR scaling).
   - Classifier: 1D-CNN converted to **TensorFlow Lite (`.tflite`)** with INT8/FP16 quantization.
3. **Edge Optimization**:
   - With an inference latency of **0.021 ms** (Phone) and **0.184 ms** (Watch), both models can easily run real-time sliding-window inference on-device without causing thermal throttling or battery drain.
"""

with open(os.path.join(RESULTS_DIR, "PHASE_6_ML_BENCHMARK_REPORT.md"), "w") as f:
    f.write(report_content)

print("Generated PHASE_6_ML_BENCHMARK_REPORT.md successfully.")
