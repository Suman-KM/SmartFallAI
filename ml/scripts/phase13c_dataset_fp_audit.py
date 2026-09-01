import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")
RESULTS_DIR = os.path.join(WORKSPACE, "ml/results/phase13c")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = [
    'FALL_BACKWARD', 'FALL_FORWARD', 'FALL_FROM_SITTING', 'FALL_LEFT', 'FALL_RIGHT',
    'JUMPING', 'LYING_DOWN', 'PICKING_UP_OBJECT', 'RUNNING', 'SITTING',
    'SIT_DOWN', 'STANDING', 'STAND_UP', 'WALKING'
]

# Load scalers
with open(os.path.join(PREPROC_DIR, "phone/scaler.json")) as f:
    phone_scaler = json.load(f)
phone_median = np.array(phone_scaler["median"], dtype=np.float32)
phone_iqr = np.array(phone_scaler["iqr"], dtype=np.float32)

with open(os.path.join(PREPROC_DIR, "watch/scaler.json")) as f:
    watch_scaler = json.load(f)
watch_median = np.array(watch_scaler["median"], dtype=np.float32)
watch_iqr = np.array(watch_scaler["iqr"], dtype=np.float32)

# Load Phone ONNX model
phone_onnx_path = os.path.join(WORKSPACE, "app/src/main/assets/model.onnx")
phone_ort = ort.InferenceSession(phone_onnx_path)

# Load Watch Random Forest model
watch_rf_path = os.path.join(WORKSPACE, "ml/models/watch/model.joblib")
watch_rf = joblib.load(watch_rf_path)

def extract_watch_features(X_scaled):
    means = np.mean(X_scaled, axis=1)
    stds = np.std(X_scaled, axis=1)
    mins = np.min(X_scaled, axis=1)
    maxs = np.max(X_scaled, axis=1)
    ranges = maxs - mins
    medians = np.median(X_scaled, axis=1)
    rms = np.sqrt(np.mean(X_scaled ** 2, axis=1))
    energy = np.mean(X_scaled ** 2, axis=1)
    return np.hstack([means, stds, mins, maxs, ranges, medians, rms, energy]).astype(np.float32)

def compute_kinematics(X_raw):
    # X_raw shape: (N, 100, 9)
    # channels: accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw
    acc = X_raw[:, :, :3]
    gyro = X_raw[:, :, 3:6]
    
    acc_mag = np.linalg.norm(acc, axis=2) # (N, 100)
    gyro_mag = np.linalg.norm(gyro, axis=2) # (N, 100)
    
    acc_peak = np.max(acc_mag, axis=1)
    acc_min = np.min(acc_mag, axis=1)
    acc_range = acc_peak - acc_min
    acc_std = np.std(acc_mag, axis=1)
    
    gyro_peak = np.max(gyro_mag, axis=1)
    gyro_min = np.min(gyro_mag, axis=1)
    gyro_range = gyro_peak - gyro_min
    gyro_std = np.std(gyro_mag, axis=1)
    
    # Jerk: difference between consecutive samples (50 Hz -> dt = 0.02s)
    # jerk = |da/dt|
    dt = 0.02
    acc_diff = np.diff(acc_mag, axis=1) / dt # (N, 99)
    jerk_peak = np.max(np.abs(acc_diff), axis=1)
    
    # Duration of elevated acceleration (> 15 m/s^2) in seconds
    elev_acc_dur = np.sum(acc_mag > 15.0, axis=1) * dt
    # Duration of elevated angular velocity (> 2.5 rad/s) in seconds
    elev_gyro_dur = np.sum(gyro_mag > 2.5, axis=1) * dt
    
    return {
        "acc_mag_mean": np.mean(acc_mag, axis=1),
        "acc_peak": acc_peak,
        "acc_min": acc_min,
        "acc_range": acc_range,
        "acc_std": acc_std,
        "gyro_mag_mean": np.mean(gyro_mag, axis=1),
        "gyro_peak": gyro_peak,
        "gyro_range": gyro_range,
        "gyro_std": gyro_std,
        "jerk_peak": jerk_peak,
        "elev_acc_dur": elev_acc_dur,
        "elev_gyro_dur": elev_gyro_dur
    }

def evaluate_device_split(device, split):
    print(f"Auditing {device} - {split} split...")
    split_dir = os.path.join(PREPROC_DIR, device, split)
    X_scaled = np.load(os.path.join(split_dir, "X.npy")) # (N, 100, 9)
    y_14 = np.load(os.path.join(split_dir, "y_14.npy"))
    meta_df = pd.read_csv(os.path.join(split_dir, "metadata.csv"))
    
    # Unscale to raw
    scaler_median = phone_median if device == "phone" else watch_median
    scaler_iqr = phone_iqr if device == "phone" else watch_iqr
    X_raw = (X_scaled * scaler_iqr) + scaler_median
    
    # Compute ML probabilities
    if device == "phone":
        ort_inputs = {phone_ort.get_inputs()[0].name: X_scaled.astype(np.float32)}
        logits = phone_ort.run(None, ort_inputs)[0]
        # Softmax
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    else:
        feats = extract_watch_features(X_scaled)
        probs = watch_rf.predict_proba(feats)
        
    fall_probs = np.sum(probs[:, :5], axis=1)
    
    # Compute kinematics
    kin = compute_kinematics(X_raw)
    
    # Organize per activity
    records = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = (y_14 == cls_idx)
        n_windows = int(np.sum(mask))
        if n_windows == 0:
            continue
            
        cls_sessions = meta_df[mask]['session_id'].nunique() if 'session_id' in meta_df.columns else 0
        cls_fps = fall_probs[mask]
        
        rec = {
            "device": device,
            "split": split,
            "class_idx": cls_idx,
            "activity": cls_name,
            "is_fall": cls_idx < 5,
            "sessions": cls_sessions,
            "windows": n_windows,
            "fp_mean": float(np.mean(cls_fps)),
            "fp_median": float(np.median(cls_fps)),
            "fp_p90": float(np.percentile(cls_fps, 90)),
            "fp_p95": float(np.percentile(cls_fps, 95)),
            "fp_max": float(np.max(cls_fps)),
            "pct_ge_030": float(np.mean(cls_fps >= 0.30) * 100),
            "pct_ge_040": float(np.mean(cls_fps >= 0.40) * 100),
            "pct_ge_050": float(np.mean(cls_fps >= 0.50) * 100),
            "pct_ge_060": float(np.mean(cls_fps >= 0.60) * 100),
            "pct_ge_070": float(np.mean(cls_fps >= 0.70) * 100),
            "pct_ge_080": float(np.mean(cls_fps >= 0.80) * 100),
            "pct_ge_090": float(np.mean(cls_fps >= 0.90) * 100),
            # Kinematics medians and p95
            "acc_peak_median": float(np.median(kin["acc_peak"][mask])),
            "acc_peak_p95": float(np.percentile(kin["acc_peak"][mask], 95)),
            "acc_min_median": float(np.median(kin["acc_min"][mask])),
            "acc_range_median": float(np.median(kin["acc_range"][mask])),
            "acc_std_median": float(np.median(kin["acc_std"][mask])),
            "gyro_peak_median": float(np.median(kin["gyro_peak"][mask])),
            "gyro_peak_p95": float(np.percentile(kin["gyro_peak"][mask], 95)),
            "gyro_range_median": float(np.median(kin["gyro_range"][mask])),
            "gyro_std_median": float(np.median(kin["gyro_std"][mask])),
            "jerk_peak_median": float(np.median(kin["jerk_peak"][mask])),
            "elev_acc_dur_median": float(np.median(kin["elev_acc_dur"][mask])),
            "elev_gyro_dur_median": float(np.median(kin["elev_gyro_dur"][mask])),
        }
        records.append(rec)
        
    return pd.DataFrame(records), {
        "fall_probs": fall_probs,
        "y_14": y_14,
        "meta_df": meta_df,
        "kin": kin
    }

# Run for both devices on validation and test sets
p_val_df, p_val_data = evaluate_device_split("phone", "validation")
p_test_df, p_test_data = evaluate_device_split("phone", "test")
w_val_df, w_val_data = evaluate_device_split("watch", "validation")
w_test_df, w_test_data = evaluate_device_split("watch", "test")

# Save summary tables
p_val_df.to_csv(os.path.join(RESULTS_DIR, "phone_validation_audit.csv"), index=False)
p_test_df.to_csv(os.path.join(RESULTS_DIR, "phone_test_audit.csv"), index=False)
w_val_df.to_csv(os.path.join(RESULTS_DIR, "watch_validation_audit.csv"), index=False)
w_test_df.to_csv(os.path.join(RESULTS_DIR, "watch_test_audit.csv"), index=False)

print("Saved audit CSVs to ml/results/phase13c/")
