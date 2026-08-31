import os
import sys
import json
import csv
import time
import joblib
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from train_cnn import Conv1DNet
from train_bilstm import BiLSTMNet
from train_random_forest import extract_window_features
from evaluate import compute_all_metrics

WORKSPACE_DIR = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROCESSING_DIR = os.path.join(WORKSPACE_DIR, "preprocessing")
ML_DIR = os.path.join(WORKSPACE_DIR, "ml")
RESULTS_DIR = os.path.join(ML_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CONFUSION_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)

with open(os.path.join(ML_DIR, "common/label_map.json"), "r") as f:
    label_map = json.load(f)

CLASSES_14 = label_map["classes_14"]
SEED = 42

PIPELINE_MAP = {
    "P01": "01_raw_standardized",
    "P02": "02_robust_scaling",
    "P03": "03_signal_filtering",
    "P04": "04_gravity_motion_separation",
    "P05": "05_motion_magnitude_features"
}

def plot_confusion_matrix(cm, class_names, title, out_png):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=13, pad=15, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)
    
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)
            
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_binary_confusion_matrix(bin_cm, title, out_png):
    plt.figure(figsize=(6, 5))
    plt.imshow(bin_cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(title, fontsize=12, pad=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['NORMAL', 'FALL'], fontsize=10, fontweight='bold')
    plt.yticks(tick_marks, ['NORMAL', 'FALL'], fontsize=10, fontweight='bold')
    
    thresh = bin_cm.max() / 2.0 if bin_cm.max() > 0 else 1.0
    for i in range(bin_cm.shape[0]):
        for j in range(bin_cm.shape[1]):
            plt.text(j, i, format(bin_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if bin_cm[i, j] > thresh else "black", fontsize=12, fontweight='bold')
            
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def load_all_existing_results():
    all_results = []
    for dev in ["watch", "phone"]:
        dev_upper = dev.upper()
        for p_code, p_dir_name in PIPELINE_MAP.items():
            for m_type in ["CNN", "BiLSTM", "RandomForest"]:
                res_file = os.path.join(ML_DIR, dev, p_code, m_type, "results.json")
                if not os.path.exists(res_file):
                    print(f"Error: Missing {res_file}")
                    continue
                with open(res_file, "r") as f:
                    r = json.load(f)
                vm = r["validation_metrics"]
                model_name = "1D_CNN" if m_type == "CNN" else m_type
                all_results.append({
                    "device": dev_upper,
                    "pipeline_code": p_code,
                    "pipeline_name": p_dir_name,
                    "model": model_name,
                    "val_accuracy": vm["accuracy"],
                    "val_macro_precision": vm["macro_precision"],
                    "val_macro_recall": vm["macro_recall"],
                    "val_macro_f1": vm["macro_f1"],
                    "val_weighted_f1": vm["weighted_f1"],
                    "val_fall_precision": vm["binary"]["fall_precision"],
                    "val_fall_recall": vm["binary"]["fall_recall"],
                    "val_fall_f1": vm["binary"]["fall_f1"],
                    "val_non_fall_f1": vm["binary"]["non_fall_f1"],
                    "params": r.get("total_parameters", "N/A"),
                    "model_size_kb": r.get("model_size_kb", 0.0),
                    "latency_ms": r.get("inference_latency_ms", 0.0),
                    "train_duration_sec": r.get("train_duration_sec", 0.0),
                    "training_history": r.get("training_history", [])
                })
    return all_results

def generate_comparative_plots(all_results):
    pipelines = ["P01", "P02", "P03", "P04", "P05"]
    models = ["1D_CNN", "BiLSTM", "RandomForest"]
    
    # 1. Validation Macro-F1 Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, dev in enumerate(["WATCH", "PHONE"]):
        ax = axes[idx]
        x = np.arange(len(pipelines))
        width = 0.25
        
        for m_idx, m_name in enumerate(models):
            vals = []
            for p in pipelines:
                row = next(r for r in all_results if r["device"] == dev and r["pipeline_code"] == p and r["model"] == m_name)
                vals.append(row["val_macro_f1"])
            ax.bar(x + m_idx * width - width, vals, width, label=m_name)
            
        ax.set_title(f"{dev} — Validation Macro-F1 Across Preprocessing Pipelines", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, fontsize=10)
        ax.set_xlabel("Preprocessing Pipeline", fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Validation Macro-F1", fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        ax.set_ylim([0.3, 0.7])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "validation_macro_f1_comparison.png"), dpi=180)
    plt.close()
    
    # 2. Validation Fall Recall Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, dev in enumerate(["WATCH", "PHONE"]):
        ax = axes[idx]
        x = np.arange(len(pipelines))
        width = 0.25
        
        for m_idx, m_name in enumerate(models):
            vals = []
            for p in pipelines:
                row = next(r for r in all_results if r["device"] == dev and r["pipeline_code"] == p and r["model"] == m_name)
                vals.append(row["val_fall_recall"])
            ax.bar(x + m_idx * width - width, vals, width, label=m_name)
            
        ax.set_title(f"{dev} — Validation Fall Recall (Sensitivity)", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, fontsize=10)
        ax.set_xlabel("Preprocessing Pipeline", fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Validation Fall Recall", fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        ax.set_ylim([0.65, 0.95])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "validation_fall_recall_comparison.png"), dpi=180)
    plt.close()

    # 3. Validation Accuracy Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, dev in enumerate(["WATCH", "PHONE"]):
        ax = axes[idx]
        x = np.arange(len(pipelines))
        width = 0.25
        
        for m_idx, m_name in enumerate(models):
            vals = []
            for p in pipelines:
                row = next(r for r in all_results if r["device"] == dev and r["pipeline_code"] == p and r["model"] == m_name)
                vals.append(row["val_accuracy"])
            ax.bar(x + m_idx * width - width, vals, width, label=m_name)
            
        ax.set_title(f"{dev} — Validation Accuracy", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, fontsize=10)
        ax.set_xlabel("Preprocessing Pipeline", fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Validation Accuracy", fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        ax.set_ylim([0.45, 0.82])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "validation_accuracy_comparison.png"), dpi=180)
    plt.close()

    # 4. Binary Fall F1 Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, dev in enumerate(["WATCH", "PHONE"]):
        ax = axes[idx]
        x = np.arange(len(pipelines))
        width = 0.25
        
        for m_idx, m_name in enumerate(models):
            vals = []
            for p in pipelines:
                row = next(r for r in all_results if r["device"] == dev and r["pipeline_code"] == p and r["model"] == m_name)
                vals.append(row["val_fall_f1"])
            ax.bar(x + m_idx * width - width, vals, width, label=m_name)
            
        ax.set_title(f"{dev} — Binary Fall F1-Score", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pipelines, fontsize=10)
        ax.set_xlabel("Preprocessing Pipeline", fontsize=10, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Binary Fall F1", fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        ax.set_ylim([0.65, 0.90])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "binary_fall_f1_comparison.png"), dpi=180)
    plt.close()

    # 7. Training curves for CNN (P02 Watch and Phone)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for idx, (dev, p_code) in enumerate([("WATCH", "P02"), ("PHONE", "P02")]):
        ax = axes[idx]
        r = next(x for x in all_results if x["device"] == dev and x["pipeline_code"] == p_code and x["model"] == "1D_CNN")
        hist = r["training_history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["train_loss"] for h in hist], 'b-o', label="Train Loss")
        ax.plot(epochs, [h["val_macro_f1"] for h in hist], 'g-s', label="Val Macro-F1")
        ax.plot(epochs, [h["val_fall_recall"] for h in hist], 'r-^', label="Val Fall Recall")
        ax.set_title(f"{dev} 1D-CNN ({p_code}) Training Curves", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Metric Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_training_curves.png"), dpi=180)
    plt.close()

    # 8. Training curves for BiLSTM (P01 Watch and Phone)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for idx, (dev, p_code) in enumerate([("WATCH", "P01"), ("PHONE", "P01")]):
        ax = axes[idx]
        r = next(x for x in all_results if x["device"] == dev and x["pipeline_code"] == p_code and x["model"] == "BiLSTM")
        hist = r["training_history"]
        epochs = [h["epoch"] for h in hist]
        ax.plot(epochs, [h["train_loss"] for h in hist], 'b-o', label="Train Loss")
        ax.plot(epochs, [h["val_macro_f1"] for h in hist], 'g-s', label="Val Macro-F1")
        ax.plot(epochs, [h["val_fall_recall"] for h in hist], 'r-^', label="Val Fall Recall")
        ax.set_title(f"{dev} Bi-LSTM ({p_code}) Training Curves", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Metric Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bilstm_training_curves.png"), dpi=180)
    plt.close()

def main():
    all_results = load_all_existing_results()
    assert len(all_results) == 30, f"Expected 30 results, found {len(all_results)}"
    
    # Save CSVs
    watch_res_list = [r for r in all_results if r["device"] == "WATCH"]
    phone_res_list = [r for r in all_results if r["device"] == "PHONE"]
    
    fields = [
        "device", "pipeline_code", "pipeline_name", "model",
        "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1",
        "val_weighted_f1", "val_fall_precision", "val_fall_recall", "val_fall_f1", "val_non_fall_f1",
        "params", "model_size_kb", "latency_ms", "train_duration_sec"
    ]
    
    def save_csv(rlist, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rlist:
                row_copy = {k: row[k] for k in fields}
                writer.writerow(row_copy)
                
    save_csv(watch_res_list, os.path.join(RESULTS_DIR, "watch_results.csv"))
    save_csv(phone_res_list, os.path.join(RESULTS_DIR, "phone_results.csv"))
    
    # Model Selection based on VALIDATION metrics
    watch_sorted = sorted(watch_res_list, key=lambda x: (x["val_macro_f1"], x["val_fall_recall"]), reverse=True)
    phone_sorted = sorted(phone_res_list, key=lambda x: (x["val_macro_f1"], x["val_fall_recall"]), reverse=True)
    
    best_watch = watch_sorted[0]
    best_phone = phone_sorted[0]
    
    # -------------------------------------------------------------
    # RUN TEST EVALUATION ON FROZEN WINNERS (EXACTLY ONCE)
    # -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # 1. Evaluate Best Watch: P02 + RandomForest
    w_pcode = best_watch["pipeline_code"]
    w_pname = best_watch["pipeline_name"]
    w_mtype = best_watch["model"]
    w_test_dir = os.path.join(PREPROCESSING_DIR, w_pname, "watch")
    X_test_w = np.load(os.path.join(w_test_dir, "test/X.npy"))
    y_test_w = np.load(os.path.join(w_test_dir, "test/y_14.npy"))
    
    w_model_path = os.path.join(ML_DIR, "watch", w_pcode, w_mtype if w_mtype != "1D_CNN" else "CNN", "model.joblib" if w_mtype == "RandomForest" else "model.pth")
    if w_mtype == "RandomForest":
        w_model = joblib.load(w_model_path)
        feats_test_w = extract_window_features(X_test_w)
        test_preds_w = w_model.predict(feats_test_w)
    elif w_mtype == "1D_CNN":
        w_model = Conv1DNet(in_channels=X_test_w.shape[2], num_classes=14).to(device)
        w_model.load_state_dict(torch.load(w_model_path, map_location=device))
        w_model.eval()
        with torch.no_grad():
            t = torch.tensor(X_test_w, dtype=torch.float32).to(device)
            test_preds_w = torch.argmax(w_model(t), dim=1).cpu().numpy()
    else:
        w_model = BiLSTMNet(in_channels=X_test_w.shape[2], hidden_size=64, num_layers=2, num_classes=14).to(device)
        w_model.load_state_dict(torch.load(w_model_path, map_location=device))
        w_model.eval()
        with torch.no_grad():
            t = torch.tensor(X_test_w, dtype=torch.float32).to(device)
            test_preds_w = torch.argmax(w_model(t), dim=1).cpu().numpy()
            
    final_test_watch = compute_all_metrics(y_test_w, test_preds_w, CLASSES_14)
    
    # 2. Evaluate Best Phone: P02 + 1D_CNN
    p_pcode = best_phone["pipeline_code"]
    p_pname = best_phone["pipeline_name"]
    p_mtype = best_phone["model"]
    p_test_dir = os.path.join(PREPROCESSING_DIR, p_pname, "phone")
    X_test_p = np.load(os.path.join(p_test_dir, "test/X.npy"))
    y_test_p = np.load(os.path.join(p_test_dir, "test/y_14.npy"))
    
    p_model_path = os.path.join(ML_DIR, "phone", p_pcode, p_mtype if p_mtype != "1D_CNN" else "CNN", "model.joblib" if p_mtype == "RandomForest" else "model.pth")
    if p_mtype == "RandomForest":
        p_model = joblib.load(p_model_path)
        feats_test_p = extract_window_features(X_test_p)
        test_preds_p = p_model.predict(feats_test_p)
    elif p_mtype == "1D_CNN":
        p_model = Conv1DNet(in_channels=X_test_p.shape[2], num_classes=14).to(device)
        p_model.load_state_dict(torch.load(p_model_path, map_location=device))
        p_model.eval()
        with torch.no_grad():
            t = torch.tensor(X_test_p, dtype=torch.float32).to(device)
            test_preds_p = torch.argmax(p_model(t), dim=1).cpu().numpy()
    else:
        p_model = BiLSTMNet(in_channels=X_test_p.shape[2], hidden_size=64, num_layers=2, num_classes=14).to(device)
        p_model.load_state_dict(torch.load(p_model_path, map_location=device))
        p_model.eval()
        with torch.no_grad():
            t = torch.tensor(X_test_p, dtype=torch.float32).to(device)
            test_preds_p = torch.argmax(p_model(t), dim=1).cpu().numpy()
            
    final_test_phone = compute_all_metrics(y_test_p, test_preds_p, CLASSES_14)
    
    # Save Confusion Matrix plots
    plot_confusion_matrix(
        np.array(final_test_watch["confusion_matrix_14"]), CLASSES_14,
        f"Watch Final Test 14-Class Confusion Matrix ({w_pcode} + {w_mtype})",
        os.path.join(CONFUSION_DIR, "watch_final_test_cm14.png")
    )
    plot_binary_confusion_matrix(
        np.array(final_test_watch["binary"]["confusion_matrix_binary"]),
        f"Watch Final Test Binary Confusion Matrix ({w_pcode} + {w_mtype})",
        os.path.join(CONFUSION_DIR, "watch_final_test_cm_binary.png")
    )
    plot_confusion_matrix(
        np.array(final_test_phone["confusion_matrix_14"]), CLASSES_14,
        f"Phone Final Test 14-Class Confusion Matrix ({p_pcode} + {p_mtype})",
        os.path.join(CONFUSION_DIR, "phone_final_test_cm14.png")
    )
    plot_binary_confusion_matrix(
        np.array(final_test_phone["binary"]["confusion_matrix_binary"]),
        f"Phone Final Test Binary Confusion Matrix ({p_pcode} + {p_mtype})",
        os.path.join(CONFUSION_DIR, "phone_final_test_cm_binary.png")
    )
    
    # Generate all comparative plots
    generate_comparative_plots(all_results)
    
    # Generate FINAL_MODEL_SELECTION.md
    final_selection_md = f"""# SMARTFALL AI — FINAL MODEL & PREPROCESSING SELECTION

## Selected Configurations

### 1. WATCH WINNER
- **Device**: Samsung Galaxy Watch 4 (`SM-R870`)
- **Selected Preprocessing**: **P02 — Robust Scaling (`02_robust_scaling`)**
- **Selected Model Architecture**: **Random Forest (`RandomForestClassifier`, 100 estimators, max depth 20)**
- **Selection Basis (Validation Set)**:
  - **Validation Macro-F1**: `{best_watch['val_macro_f1']:.4f}`
  - **Validation Accuracy**: `{best_watch['val_accuracy']:.4f}`
  - **Validation Fall Recall**: `{best_watch['val_fall_recall']:.4f}`
- **Untouched Final Test Set Performance**:
  - **Test Accuracy**: `{final_test_watch['accuracy']:.4f}`
  - **Test Macro-F1**: `{final_test_watch['macro_f1']:.4f}`
  - **Test Fall Recall (Sensitivity)**: `{final_test_watch['binary']['fall_recall']:.4f}`
  - **Test Fall Precision**: `{final_test_watch['binary']['fall_precision']:.4f}`
  - **Test Binary Fall F1**: `{final_test_watch['binary']['fall_f1']:.4f}`
- **Operational Metrics**:
  - **Model File Size**: `{best_watch['model_size_kb']:.1f} KB`
  - **Inference Latency**: `{best_watch['latency_ms']:.3f} ms / window`

---

### 2. PHONE WINNER
- **Device**: Samsung Galaxy A50s (`SM-A507FN`)
- **Selected Preprocessing**: **P02 — Robust Scaling (`02_robust_scaling`)**
- **Selected Model Architecture**: **1D-CNN (3-stage Temporal Convolution)**
- **Selection Basis (Validation Set)**:
  - **Validation Macro-F1**: `{best_phone['val_macro_f1']:.4f}`
  - **Validation Accuracy**: `{best_phone['val_accuracy']:.4f}`
  - **Validation Fall Recall**: `{best_phone['val_fall_recall']:.4f}`
- **Untouched Final Test Set Performance**:
  - **Test Accuracy**: `{final_test_phone['accuracy']:.4f}`
  - **Test Macro-F1**: `{final_test_phone['macro_f1']:.4f}`
  - **Test Fall Recall (Sensitivity)**: `{final_test_phone['binary']['fall_recall']:.4f}`
  - **Test Fall Precision**: `{final_test_phone['binary']['fall_precision']:.4f}`
  - **Test Binary Fall F1**: `{final_test_phone['binary']['fall_f1']:.4f}`
- **Operational Metrics**:
  - **Total Parameters**: `{best_phone['params']}`
  - **Model File Size**: `{best_phone['model_size_kb']:.1f} KB`
  - **Inference Latency**: `{best_phone['latency_ms']:.3f} ms / window`
"""
    with open(os.path.join(RESULTS_DIR, "FINAL_MODEL_SELECTION.md"), "w") as f:
        f.write(final_selection_md)
        
    print("Master benchmark complete, test evaluation executed, plots and reports saved.")

if __name__ == "__main__":
    main()
