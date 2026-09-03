import os
import json
import numpy as np
import pandas as pd
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

with open(os.path.join(WORKSPACE, "preprocessing/02_robust_scaling/watch/scaler.json")) as f:
    w_scaler = json.load(f)
w_med = np.array(w_scaler["median"], dtype=np.float32)
w_iqr = np.array(w_scaler["iqr"], dtype=np.float32)

rf_model = joblib.load(os.path.join(WORKSPACE, "ml/models/watch/model.joblib"))

split_df = pd.read_csv(os.path.join(WORKSPACE, "preprocessing/common_split/all_split_sessions.csv"))
w_sessions = split_df[split_df["device"] == "WATCH"]

def extract_watch_features(window):
    feats = []
    for ch in range(9):
        col = window[:, ch]
        mean = float(np.mean(col))
        std = float(np.std(col))
        cmin = float(np.min(col))
        cmax = float(np.max(col))
        q25 = float(np.percentile(col, 25))
        q50 = float(np.percentile(col, 50))
        q75 = float(np.percentile(col, 75))
        iqr = q75 - q25
        feats.extend([mean, std, cmin, cmax, q25, q50, q75, iqr])
    return np.array(feats, dtype=np.float32).reshape(1, -1)

print("="*70)
print("TESTING WATCH SESSIONS ACROSS ENTIRE DATASET")
print("="*70)

falls_total = 0
falls_detected = 0
adls_total = 0
adls_fa = 0
fa_by_act = {}
recall_by_act = {}

for _, row in w_sessions.iterrows():
    fpath = os.path.join(WORKSPACE, "raw_dataset/watch", row["filename"])
    if not os.path.exists(fpath): continue
    is_fall = row["fall_binary"] == "FALL"
    act = row["activity"]
    
    df = pd.read_csv(fpath)
    raw = df.iloc[:, 2:11].values.astype(np.float32)
    n = len(raw)
    if n < 100: continue
    
    state = "MONITORING"
    impactCountdown = 0
    triggered = False
    
    for start in range(0, n - 100 + 1, 50):
        window = raw[start : start + 100]
        
        acc_mag = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
        maxAccMag = float(np.max(acc_mag))
        minAccMag = float(np.min(acc_mag))
        accRange = maxAccMag - minAccMag
        accStd = float(np.std(acc_mag))
        
        jerk = np.abs(np.diff(acc_mag)) / 0.02
        maxJerk = float(np.max(jerk)) if len(jerk) > 0 else 0.0
        
        gyro_mag = np.sqrt(window[:, 3]**2 + window[:, 4]**2 + window[:, 5]**2)
        maxGyroMag = float(np.max(gyro_mag))
        
        isCollisionShock = (maxAccMag >= 24.0 and maxJerk >= 500.0) or \
                           (accRange >= 16.0 and maxJerk >= 350.0 and maxGyroMag >= 4.0)
        isLocomotionCadence = (accStd >= 5.5 and maxGyroMag >= 4.0) or (accStd >= 8.0)
        isSettledImmobility = (accStd <= 3.8) and (maxGyroMag <= 3.2)
        
        scaled = (window - w_med) / w_iqr
        feats = extract_watch_features(scaled)
        probs = rf_model.predict_proba(feats)[0]
        topIdx = int(np.argmax(probs))
        fallProb = float(np.sum(probs[0:5]))
        lyingDownProb = float(probs[6])
        
        # Consistent posture rule
        has_posture = (fallProb >= 0.40) or (lyingDownProb >= 0.45 and accStd <= 2.0)
        
        if state == "MONITORING":
            if isCollisionShock:
                impactCountdown = 4
            elif impactCountdown > 0:
                if isLocomotionCadence:
                    impactCountdown = 0
                elif isSettledImmobility and has_posture:
                    impactCountdown = 0
                    state = "FALL_SUSPECTED"
                    triggered = True
                    break
                else:
                    impactCountdown -= 1
                    
    if is_fall:
        falls_total += 1
        if act not in recall_by_act: recall_by_act[act] = [0, 0]
        recall_by_act[act][1] += 1
        if triggered:
            falls_detected += 1
            recall_by_act[act][0] += 1
    else:
        adls_total += 1
        if triggered:
            adls_fa += 1
            fa_by_act[act] = fa_by_act.get(act, 0) + 1

rec = (falls_detected / falls_total) * 100.0 if falls_total > 0 else 0.0
prec = (falls_detected / (falls_detected + adls_fa)) * 100.0 if (falls_detected + adls_fa) > 0 else 0.0
f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
fpr = (adls_fa / adls_total) * 100.0 if adls_total > 0 else 0.0

print(f"Fall Recall:      {rec:.2f}% ({falls_detected}/{falls_total})")
print(f"Fall Precision:   {prec:.2f}%")
print(f"Binary Fall F1:   {f1:.2f}%")
print(f"False Alarm Rate: {fpr:.2f}% ({adls_fa}/{adls_total})")
print(f"False Alarms by Activity: {fa_by_act}")
print("Recall by Fall Type:")
for k, v in recall_by_act.items():
    pct = (v[0]/v[1])*100.0 if v[1]>0 else 0.0
    print(f"  {k:<20}: {v[0]}/{v[1]} ({pct:.1f}%)")
