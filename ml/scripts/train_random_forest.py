import os
import sys
import time
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(__file__))
from evaluate import compute_all_metrics, measure_inference_latency

def extract_window_features(X_3d):
    """
    Extracts 8 statistical/time-domain features per channel for each window:
    [mean, std, min, max, range, median, rms, energy]
    Input: (N, 100, D)
    Output: (N, D * 8)
    """
    N, T, D = X_3d.shape
    
    means = np.mean(X_3d, axis=1)                          # (N, D)
    stds = np.std(X_3d, axis=1)                            # (N, D)
    mins = np.min(X_3d, axis=1)                            # (N, D)
    maxs = np.max(X_3d, axis=1)                            # (N, D)
    ranges = maxs - mins                                   # (N, D)
    medians = np.median(X_3d, axis=1)                      # (N, D)
    rms = np.sqrt(np.mean(X_3d ** 2, axis=1))              # (N, D)
    energy = np.mean(X_3d ** 2, axis=1)                    # (N, D)
    
    feats_2d = np.hstack([means, stds, mins, maxs, ranges, medians, rms, energy]) # (N, D * 8)
    return feats_2d.astype(np.float32)

def train_and_eval_rf(X_train, y_train, X_val, y_val, out_dir, class_names, device_name, pipeline_id, n_estimators=100, max_depth=20, seed=42):
    np.random.seed(seed)
    
    # Extract window-level statistical features
    X_train_feats = extract_window_features(X_train)
    X_val_feats = extract_window_features(X_val)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )
    
    start_train_time = time.time()
    rf.fit(X_train_feats, y_train)
    train_duration = time.time() - start_train_time
    
    # Validation evaluation
    val_preds = rf.predict(X_val_feats)
    final_val_metrics = compute_all_metrics(y_val, val_preds, class_names)
    
    # Measure Latency
    def predict_batch(x_arr):
        f = extract_window_features(x_arr)
        return rf.predict_proba(f)
        
    latency_ms = measure_inference_latency(predict_batch, X_val[:64])
    
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.joblib")
    joblib.dump(rf, model_path)
    model_size_kb = os.path.getsize(model_path) / 1024.0
    
    results = {
        "model_type": "RandomForest",
        "device": device_name,
        "pipeline_id": pipeline_id,
        "raw_channels": X_train.shape[2],
        "extracted_feature_count": X_train_feats.shape[1],
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "model_size_kb": model_size_kb,
        "train_duration_sec": train_duration,
        "inference_latency_ms": latency_ms,
        "validation_metrics": final_val_metrics
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    return results, rf
