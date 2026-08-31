import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

FALL_INDICES = {0, 1, 2, 3, 4}

def compute_all_metrics(y_true_14, y_pred_14, class_names):
    """
    Computes complete 14-class and binary fall detection metrics.
    """
    # 1. 14-Class Multi-class metrics
    acc = accuracy_score(y_true_14, y_pred_14)
    macro_p = precision_score(y_true_14, y_pred_14, average="macro", zero_division=0)
    macro_r = recall_score(y_true_14, y_pred_14, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true_14, y_pred_14, average="macro", zero_division=0)
    
    weighted_p = precision_score(y_true_14, y_pred_14, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true_14, y_pred_14, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true_14, y_pred_14, average="weighted", zero_division=0)
    
    per_class_p = precision_score(y_true_14, y_pred_14, average=None, zero_division=0, labels=list(range(len(class_names))))
    per_class_r = recall_score(y_true_14, y_pred_14, average=None, zero_division=0, labels=list(range(len(class_names))))
    per_class_f1 = f1_score(y_true_14, y_pred_14, average=None, zero_division=0, labels=list(range(len(class_names))))
    
    cm_14 = confusion_matrix(y_true_14, y_pred_14, labels=list(range(len(class_names))))
    
    per_class_dict = {}
    for idx, cname in enumerate(class_names):
        per_class_dict[cname] = {
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1": float(per_class_f1[idx]),
            "support": int(np.sum(y_true_14 == idx))
        }
        
    # 2. Binary Fall vs Non-Fall metrics
    y_true_bin = np.array([1 if y in FALL_INDICES else 0 for y in y_true_14])
    y_pred_bin = np.array([1 if y in FALL_INDICES else 0 for y in y_pred_14])
    
    bin_acc = accuracy_score(y_true_bin, y_pred_bin)
    
    # Class 1 = FALL
    fall_p = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    fall_r = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    fall_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    
    # Class 0 = NORMAL / NON-FALL
    non_fall_p = precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    non_fall_r = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    non_fall_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    
    bin_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    # bin_cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(bin_cm[0, 0]), int(bin_cm[0, 1]), int(bin_cm[1, 0]), int(bin_cm[1, 1])
    
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_dict,
        "confusion_matrix_14": cm_14.tolist(),
        "binary": {
            "accuracy": float(bin_acc),
            "fall_precision": float(fall_p),
            "fall_recall": float(fall_r),
            "fall_f1": float(fall_f1),
            "non_fall_precision": float(non_fall_p),
            "non_fall_recall": float(non_fall_r),
            "non_fall_f1": float(non_fall_f1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "confusion_matrix_binary": bin_cm.tolist()
        }
    }

def measure_inference_latency(predict_fn, sample_batch, num_runs=50):
    """
    Measures mean inference latency per window in milliseconds.
    """
    # Warmup
    for _ in range(5):
        _ = predict_fn(sample_batch)
        
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = predict_fn(sample_batch)
    elapsed = time.perf_counter() - start_time
    
    total_samples = len(sample_batch) * num_runs
    latency_ms_per_window = (elapsed / total_samples) * 1000.0
    return float(latency_ms_per_window)
