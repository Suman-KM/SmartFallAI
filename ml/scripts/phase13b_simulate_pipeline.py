import os
import json
import csv
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

for dev in ["phone", "watch"]:
    val_dir = f"preprocessing/02_robust_scaling/{dev}/validation"
    X_val_scaled = np.load(os.path.join(val_dir, "X.npy"))
    y_val = np.load(os.path.join(val_dir, "y_binary.npy"))
    
    with open(os.path.join(val_dir, "metadata.csv")) as f:
        meta = list(csv.DictReader(f))
        
    sessions = {}
    for idx, r in enumerate(meta):
        sid = r["source_session_id"]
        if sid not in sessions:
            sessions[sid] = {
                "act": r["target_activity"],
                "is_fall": (r["fall_binary"] == "FALL"),
                "indices": []
            }
        sessions[sid]["indices"].append(idx)
        
    med = p_med if dev == "phone" else w_med
    iqr = p_iqr if dev == "phone" else w_iqr
    X_val_phys = X_val_scaled * iqr + med
    
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
    acc_peaks = np.max(acc_mag, axis=1)
    acc_ranges = np.max(acc_mag, axis=1) - np.min(acc_mag, axis=1)
    gyro_peaks = np.max(gyro_mag, axis=1)
    
    print(f"\n=======================================================")
    print(f"EVALUATING IMPACT-GATED STATE MACHINE: {dev.upper()}")
    print(f"=======================================================")
    
    # Simulate real-time stream per session
    def simulate(impact_acc_peak=15.0, impact_acc_range=8.0, impact_gyro_peak=2.5, memory_windows=3):
        fall_detections = 0
        normal_fps = 0
        total_falls = 0
        total_normals = 0
        
        for sid, s in sessions.items():
            is_fall = s["is_fall"]
            if is_fall: total_falls += 1
            else: total_normals += 1
            
            # State machine variables
            consec_falls = 0
            recent_impact_count = 0
            detected = False
            
            # Stream windows in chronological order
            for w_idx in s["indices"]:
                # Check kinematics
                is_impact = (acc_peaks[w_idx] >= impact_acc_peak) or \
                            (acc_ranges[w_idx] >= impact_acc_range) or \
                            (gyro_peaks[w_idx] >= impact_gyro_peak)
                            
                if is_impact:
                    recent_impact_count = memory_windows # remember impact for next few windows
                elif recent_impact_count > 0:
                    recent_impact_count -= 1
                    
                is_ml_fall = (fall_probs[w_idx] >= 0.50)
                
                # Rule: Candidate fall requires ML fall AND recent/current impact shock
                if is_ml_fall and (is_impact or recent_impact_count > 0):
                    consec_falls += 1
                    if consec_falls >= 2:
                        detected = True
                        break
                else:
                    consec_falls = 0
                    
            if detected:
                if is_fall: fall_detections += 1
                else: normal_fps += 1
                
        recall = fall_detections / total_falls
        fpr = normal_fps / total_normals
        return recall, fpr, normal_fps, total_normals
        
    r_raw, fpr_raw, fp_cnt_raw, tot_n = simulate(0, 0, 0, 0)
    print(f"Raw 2-Window ML (No Impact Gate) : Recall = {r_raw*100:5.1f}% | Normal Session FP = {fp_cnt_raw}/{tot_n} ({fpr_raw*100:4.1f}%)")
    
    for a_pk in [14.0, 15.0, 16.0, 18.0]:
        for a_rng in [6.0, 8.0, 10.0]:
            for g_pk in [2.0, 2.5, 3.0]:
                r, fpr, fp_cnt, _ = simulate(a_pk, a_rng, g_pk, memory_windows=3)
                if a_pk in [15.0, 16.0] and a_rng in [8.0] and g_pk in [2.5]:
                    print(f"Impact Gate (AccPeak>={a_pk}, AccRng>={a_rng}, Gyro>={g_pk}, Mem=3): Recall = {r*100:5.1f}% | Normal Session FP = {fp_cnt}/{tot_n} ({fpr*100:4.1f}%)")
