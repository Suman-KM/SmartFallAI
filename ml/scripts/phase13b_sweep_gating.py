import os
import json
import numpy as np
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

# Load Phone ONNX
phone_onnx = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))
with open(os.path.join(WORKSPACE, "app/src/main/assets/scaler.json")) as f:
    phone_scaler = json.load(f)
p_med = np.array(phone_scaler["median"], dtype=np.float32)
p_iqr = np.array(phone_scaler["iqr"], dtype=np.float32)

# Load Watch RF
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

# Let's test on the VALIDATION SET for both devices
for dev in ["phone", "watch"]:
    print(f"\n=======================================================")
    print(f"EVALUATING FALL DETECTOR ON VALIDATION SET: {dev.upper()}")
    print(f"=======================================================")
    
    val_dir = f"preprocessing/02_robust_scaling/{dev}/validation"
    X_val_scaled = np.load(os.path.join(val_dir, "X.npy"))
    y_val = np.load(os.path.join(val_dir, "y_binary.npy"))
    y_14 = np.load(os.path.join(val_dir, "y_14.npy"))
    
    med = p_med if dev == "phone" else w_med
    iqr = p_iqr if dev == "phone" else w_iqr
    X_val_phys = X_val_scaled * iqr + med
    
    # 1. Compute raw ML probabilities
    if dev == "phone":
        out = phone_onnx.run(None, {phone_onnx.get_inputs()[0].name: X_val_scaled})[0]
        probs = softmax(out)
        fall_probs = np.sum(probs[:, :5], axis=1)
    else:
        feats = extract_watch_features(X_val_scaled)
        probs = watch_rf.predict_proba(feats)
        fall_probs = np.sum(probs[:, :5], axis=1)
        
    # Compute Kinematics for each window
    # acc: cols 0, 1, 2. gyro: cols 3, 4, 5.
    acc_mag = np.sqrt(X_val_phys[:, :, 0]**2 + X_val_phys[:, :, 1]**2 + X_val_phys[:, :, 2]**2)
    gyro_mag = np.sqrt(X_val_phys[:, :, 3]**2 + X_val_phys[:, :, 4]**2 + X_val_phys[:, :, 5]**2)
    
    acc_peak = np.max(acc_mag, axis=1)
    acc_min = np.min(acc_mag, axis=1)
    acc_range = acc_peak - acc_min
    acc_std = np.std(acc_mag, axis=1)
    gyro_peak = np.max(gyro_mag, axis=1)
    
    # Baseline: ML only (fall_prob >= 0.50)
    pred_base = (fall_probs >= 0.50).astype(int)
    
    fall_mask = (y_val == 1)
    norm_mask = (y_val == 0)
    
    recall_base = np.sum(pred_base[fall_mask] == 1) / np.sum(fall_mask)
    fpr_base = np.sum(pred_base[norm_mask] == 1) / np.sum(norm_mask)
    
    # Static activities: Sitting (9), Standing (11), Lying (6)
    static_mask = np.isin(y_14, [6, 9, 11])
    static_fp_base = np.sum(pred_base[static_mask] == 1) / np.sum(static_mask)
    
    print(f"--- BASELINE (ML Only >= 0.50) ---")
    print(f"Fall Recall: {recall_base*100:.2f}% | Normal FPR: {fpr_base*100:.2f}% | Static Activity FPR: {static_fp_base*100:.2f}%")
    
    # Now let's test Kinematic Impact Gating:
    # A fall must have:
    # (acc_range >= R or acc_peak >= P or gyro_peak >= G)
    # Let's test a sweep of thresholds
    print(f"\n--- SWEEPING KINEMATIC GATING ON VALIDATION SET ---")
    for acc_range_th in [2.0, 3.0, 4.0, 5.0, 6.0]:
        for gyro_th in [0.5, 1.0, 1.5]:
            # Motion condition: device was actually moving / tumbling / experiencing dynamic acceleration
            motion_gate = (acc_range >= acc_range_th) | (gyro_peak >= gyro_th)
            pred_gated = ((fall_probs >= 0.50) & motion_gate).astype(int)
            
            recall_gated = np.sum(pred_gated[fall_mask] == 1) / np.sum(fall_mask)
            fpr_gated = np.sum(pred_gated[norm_mask] == 1) / np.sum(norm_mask)
            static_fp_gated = np.sum(pred_gated[static_mask] == 1) / np.sum(static_mask)
            
            if acc_range_th in [3.0, 4.0, 5.0] and gyro_th in [0.5, 1.0]:
                print(f"Thresholds: AccRange>={acc_range_th:3.1f} OR GyroPeak>={gyro_th:3.1f} | Recall: {recall_gated*100:5.2f}% | FPR: {fpr_gated*100:5.2f}% | Static FPR: {static_fp_gated*100:5.2f}%")
