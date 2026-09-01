import os
import json
import numpy as np

WORKSPACE_DIR = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PHASE10_RESULTS_DIR = os.path.join(WORKSPACE_DIR, "ml/results/phase10")

with open(os.path.join(PHASE10_RESULTS_DIR, "benchmark_results_raw.json"), "r") as f:
    raw_data = json.load(f)

CLASSES_14 = [
    "FALL_BACKWARD", "FALL_FORWARD", "FALL_FROM_SITTING", "FALL_LEFT", "FALL_RIGHT",
    "JUMPING", "LYING_DOWN", "PICKING_UP_OBJECT", "RUNNING", "SITTING", "SIT_DOWN",
    "STANDING", "STAND_UP", "WALKING"
]
FALL_CLASSES = ["FALL_BACKWARD", "FALL_FORWARD", "FALL_FROM_SITTING", "FALL_LEFT", "FALL_RIGHT"]
HIGH_MOTION_ADLS = ["JUMPING", "RUNNING", "SIT_DOWN", "STAND_UP", "PICKING_UP_OBJECT"]

# -------------------------------------------------------------
# 1. PHASE_10_MODEL_BENCHMARK.MD
# -------------------------------------------------------------
benchmark_doc = []
benchmark_doc.append("# SMARTFALL AI — PHASE 10 EXTENDED MULTI-MODEL BENCHMARK REPORT\n")
benchmark_doc.append("## 1. Experimental Overview\n")
benchmark_doc.append("Phase 10 evaluates 10 diverse machine learning model architectures on 2 independent edge computing platforms:\n")
benchmark_doc.append("1. **WATCH (`SM-R870` — Wear OS)**")
benchmark_doc.append("2. **PHONE (`SM-A507FN` — Android)**\n")
benchmark_doc.append("All experiments strictly adhere to the frozen Phase 5 dataset and session split (70% train / 15% validation / 15% test). GPS, heart rate, session IDs, and timestamps are strictly excluded from input tensors $X$.\n")

benchmark_doc.append("## 2. Complete Watch Benchmark Results (10 Models)\n")
benchmark_doc.append("| Model | Type | Val Fall Recall | Val Binary F1 | Val Macro-F1 | Val FPR | Test Fall Recall | Test Binary F1 | Latency (P95) | Model Size |")
benchmark_doc.append("|---|---|---|---|---|---|---|---|---|---|")

watch_res = [r for r in raw_data if r["device"] == "watch"]
for r in watch_res:
    vm = r["val_metrics"]
    tm = r["test_metrics"]
    m_type = "Classical (72 Feats)" if "Forest" in r["model_name"] or "Trees" in r["model_name"] or "Boosting" in r["model_name"] or "SVM" in r["model_name"] or "Logistic" in r["model_name"] else "Deep Temporal (100x9)"
    benchmark_doc.append(f"| **`{r['model_name']}`** | {m_type} | **{vm['binary']['fall_recall']*100:.2f}%** | {vm['binary']['fall_f1']:.4f} | {vm['macro_f1']:.4f} | {vm['binary']['fpr']*100:.2f}% | **{tm['binary']['fall_recall']*100:.2f}%** | {tm['binary']['fall_f1']:.4f} | {r['latency_p95_ms']:.2f} ms | {r['model_size_kb']:.1f} KB |")

benchmark_doc.append("\n## 3. Complete Phone Benchmark Results (10 Models)\n")
benchmark_doc.append("| Model | Type | Val Fall Recall | Val Binary F1 | Val Macro-F1 | Val FPR | Test Fall Recall | Test Binary F1 | Latency (P95) | Model Size |")
benchmark_doc.append("|---|---|---|---|---|---|---|---|---|---|")

phone_res = [r for r in raw_data if r["device"] == "phone"]
for r in phone_res:
    vm = r["val_metrics"]
    tm = r["test_metrics"]
    m_type = "Classical (72 Feats)" if "Forest" in r["model_name"] or "Trees" in r["model_name"] or "Boosting" in r["model_name"] or "SVM" in r["model_name"] or "Logistic" in r["model_name"] else "Deep Temporal (100x9)"
    benchmark_doc.append(f"| **`{r['model_name']}`** | {m_type} | **{vm['binary']['fall_recall']*100:.2f}%** | {vm['binary']['fall_f1']:.4f} | {vm['macro_f1']:.4f} | {vm['binary']['fpr']*100:.2f}% | **{tm['binary']['fall_recall']*100:.2f}%** | {tm['binary']['fall_f1']:.4f} | {r['latency_p95_ms']:.2f} ms | {r['model_size_kb']:.1f} KB |")

with open(os.path.join(PHASE10_RESULTS_DIR, "PHASE_10_MODEL_BENCHMARK.md"), "w") as f:
    f.write("\n".join(benchmark_doc))

# -------------------------------------------------------------
# 2. WATCH & PHONE ERROR ANALYSIS MD
# -------------------------------------------------------------
def generate_device_error_analysis(dev, out_filename):
    dev_res = [r for r in raw_data if r["device"] == dev]
    # Get top 3 models by Fall Recall & Binary F1
    top_models = sorted(dev_res, key=lambda x: (x["val_metrics"]["binary"]["fall_recall"], x["val_metrics"]["binary"]["fall_f1"]), reverse=True)[:3]
    
    doc = []
    doc.append(f"# SMARTFALL AI — PHASE 10 {dev.upper()} ERROR & CONFUSION ANALYSIS\n")
    doc.append(f"Detailed evaluation of the Top 3 fall-detection models on {dev.upper()} to understand missed falls, high-motion ADL false alarms, and class confusions.\n")
    
    for rank, m in enumerate(top_models, 1):
        m_name = m["model_name"]
        vm = m["val_metrics"]
        tm = m["test_metrics"]
        cm = np.array(tm["confusion_matrix"])
        
        doc.append(f"## {rank}. `{m_name}` Error Profile\n")
        doc.append(f"- **Test Fall Recall**: **`{tm['binary']['fall_recall']*100:.2f}%`** (Missed: `{tm['binary']['fn']}` / `{tm['binary']['fn'] + tm['binary']['tp']}` falls)")
        doc.append(f"- **Test False Positive Rate (FPR)**: **`{tm['binary']['fpr']*100:.2f}%`** (False Alarms: `{tm['binary']['fp']}` / `{tm['binary']['fp'] + tm['binary']['tn']}` ADLs)")
        doc.append(f"- **Binary Fall F1**: **`{tm['binary']['fall_f1']:.4f}`**\n")
        
        # Per-class fall table
        doc.append("### Per-Class Fall Detection Sensitivity\n")
        doc.append("| Fall Class | Test Support | Precision | Recall (Sensitivity) | F1-Score | Status |")
        doc.append("|---|---|---|---|---|---|")
        for fc in FALL_CLASSES:
            pc = tm["per_class"][fc]
            status = "EXCELLENT" if pc["recall"] >= 0.85 else ("GOOD" if pc["recall"] >= 0.70 else "ATTENTION")
            doc.append(f"| `{fc}` | {pc['support']} | {pc['precision']:.4f} | **{pc['recall']*100:.2f}%** | {pc['f1']:.4f} | {status} |")
            
        doc.append("\n### High-Motion Normal Activity (ADL) Specificity\n")
        doc.append("| High-Motion ADL | Test Support | Precision | Recall (Specificity) | F1-Score | False Fall Risk |")
        doc.append("|---|---|---|---|---|---|")
        for adl in HIGH_MOTION_ADLS:
            pc = tm["per_class"][adl]
            risk = "LOW" if pc["recall"] >= 0.80 else "MEDIUM"
            doc.append(f"| `{adl}` | {pc['support']} | {pc['precision']:.4f} | **{pc['recall']*100:.2f}%** | {pc['f1']:.4f} | {risk} |")
            
        doc.append("\n---\n")
        
    with open(os.path.join(PHASE10_RESULTS_DIR, out_filename), "w") as f:
        f.write("\n".join(doc))

generate_device_error_analysis("watch", "WATCH_ERROR_ANALYSIS.md")
generate_device_error_analysis("phone", "PHONE_ERROR_ANALYSIS.md")

print("Generated PHASE_10_MODEL_BENCHMARK.md, WATCH_ERROR_ANALYSIS.md, and PHONE_ERROR_ANALYSIS.md successfully.")
