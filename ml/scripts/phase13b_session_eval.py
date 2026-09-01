import os
import json
import numpy as np
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

# Load Phone ONNX & Watch RF
phone_onnx = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))
with open(os.path.join(WORKSPACE, "app/src/main/assets/scaler.json")) as f:
    phone_scaler = json.load(f)
p_med = np.array(phone_scaler["median"], dtype=np.float32)
p_iqr = np.array(phone_scaler["iqr"], dtype=np.float32)

watch_rf = joblib.load(os.path.join(WORKSPACE, "ml/models/watch/model.joblib"))
with open(os.path.join(WORKSPACE, "wear/src/main/assets/scaler.json")) as f:
    watch_scaler = json.load(f)
w_med = np.array(watch_scaler["median"], dtype=np.float32)
w_iqr = np.array(watch_scaler["iqr"], dtype=np.float32)

def extract_watch_features(X_3d):
    means = np.mean(X_3d, axis=1)
    stds = np.std(X_3d, axis=1)
    mins = np.min(X_3d, axis=1)
    maxs = np.max(X_3d, axis=1)
    ranges = maxs - mins
    medians = np.median(X_3d, axis=1)
    rms = np.sqrt(np.mean(X_3d ** 2, axis=1))
    energy = np.mean(X_3d ** 2, axis=1)
    return np.hstack([means, stds, mins, maxs, ranges, medians, rms, energy]).astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

import csv

for dev in ["phone", "watch"]:
    val_dir = f"preprocessing/02_robust_scaling/{dev}/validation"
    X_val_scaled = np.load(os.path.join(val_dir, "X.npy"))
    y_val = np.load(os.path.join(val_dir, "y_binary.npy"))
    
    # Read metadata.csv to group windows by session
    meta_rows = []
    with open(os.path.join(val_dir, "metadata.csv")) as f:
        reader = csv.DictReader(f)
        for r in reader:
            meta_rows.append(r)
            
    # Group by session_id
    sessions = {}
    for idx, r in enumerate(meta_rows):
        sid = r["source_session_id"]
        if sid not in sessions:
            sessions[sid] = {
                "act": r["target_activity"],
                "is_fall": (r["fall_binary"] == "FALL"),
                "window_indices": []
            }
        sessions[sid]["window_indices"].append(idx)
        
    med = p_med if dev == "phone" else w_med
    iqr = p_iqr if dev == "phone" else w_iqr
    X_val_phys = X_val_scaled * iqr + med
    
    # Compute ML probabilities
    if dev == "phone":
        out = phone_onnx.run(None, {phone_onnx.get_inputs()[0].name: X_val_scaled})[0]
        probs = softmax(out)
        fall_probs = np.sum(probs[:, :5], axis=1)
    else:
        feats = extract_watch_features(X_val_scaled)
        probs = watch_rf.predict_proba(feats)
        fall_probs = np.sum(probs[:, :5], axis=1)
        
    acc_mag = np.sqrt(X_val_phys[:, :, 0]**2 + X_val_phys[:, :, 1]**2 + X_val_phys[:, :, 2]**2)
    gyro_mag = np.sqrt(X_val_phys[:, :, 3]**2 + X_val_phys[:, :, 4]**2 + X_val_phys[:, :, 5]**2)
    acc_peak = np.max(acc_mag, axis=1)
    acc_range = np.max(acc_mag, axis=1) - np.min(acc_mag, axis=1)
    gyro_peak = np.max(gyro_mag, axis=1)
    
    print(f"\n=======================================================")
    print(f"SESSION-LEVEL FALL DETECTION EVALUATION: {dev.upper()}")
    print(f"Total Validation Sessions: {len(sessions)}")
    fall_sessions = [s for s in sessions.values() if s["is_fall"]]
    norm_sessions = [s for s in sessions.values() if not s["is_fall"]]
    print(f"Fall Sessions: {len(fall_sessions)}, Normal Sessions: {len(norm_sessions)}")
    
    # Baseline: Session is detected as fall if ANY 2 consecutive windows have fall_prob >= 0.50
    def eval_detection(require_motion=False, min_acc_range=3.0, min_gyro=0.5):
        detected_falls = 0
        fp_normals = 0
        for s in fall_sessions:
            indices = s["window_indices"]
            # Check temporal sequence
            detected = False
            consec = 0
            has_motion_history = False
            for idx in indices:
                fp = fall_probs[idx]
                motion = (acc_range[idx] >= min_acc_range) or (gyro_peak[idx] >= min_gyro)
                if motion:
                    has_motion_history = True
                
                cond = (fp >= 0.50)
                if require_motion:
                    # A fall requires that either this window has motion OR a recent motion occurred!
                    cond = cond and has_motion_history
                    
                if cond:
                    consec += 1
                    if consec >= 2:
                        detected = True
                        break
                else:
                    consec = 0
            if detected:
                detected_falls += 1
                
        for s in norm_sessions:
            indices = s["window_indices"]
            detected = False
            consec = 0
            has_motion_history = False
            for idx in indices:
                fp = fall_probs[idx]
                motion = (acc_range[idx] >= min_acc_range) or (gyro_peak[idx] >= min_gyro)
                if motion:
                    has_motion_history = True
                cond = (fp >= 0.50)
                if require_motion:
                    cond = cond and has_motion_history
                if cond:
                    consec += 1
                    if consec >= 2:
                        detected = True
                        break
                else:
                    consec = 0
            if detected:
                fp_normals += 1
                
        return detected_falls / len(fall_sessions), fp_normals / len(norm_sessions)

    r_base, fpr_base = eval_detection(require_motion=False)
    print(f"Baseline (2 consec windows ML>=0.50): Fall Recall = {r_base*100:.1f}%, Normal Session FPR = {fpr_base*100:.1f}%")
    
    for r_th in [2.5, 3.5, 5.0, 7.0, 10.0]:
        for g_th in [0.5, 1.0, 1.5, 2.0]:
            r_gated, fpr_gated = eval_detection(require_motion=True, min_acc_range=r_th, min_gyro=g_th)
            if r_th in [3.5, 5.0, 7.0] and g_th in [1.0, 1.5]:
                print(f"Motion Gate (AccRange>={r_th:4.1f} OR Gyro>={g_th:3.1f}): Recall = {r_gated*100:5.1f}%, Normal Session FPR = {fpr_gated*100:5.1f}%")
