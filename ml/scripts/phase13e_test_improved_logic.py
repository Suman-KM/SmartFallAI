import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

with open(os.path.join(WORKSPACE, "preprocessing/02_robust_scaling/phone/scaler.json")) as f:
    p_scaler = json.load(f)
p_med = np.array(p_scaler["median"], dtype=np.float32)
p_iqr = np.array(p_scaler["iqr"], dtype=np.float32)

p_session = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))
p_inp_name = p_session.get_inputs()[0].name

split_df = pd.read_csv(os.path.join(WORKSPACE, "preprocessing/common_split/all_split_sessions.csv"))

# Test all phone sessions across all activities
print("="*70)
print("TESTING PRINCIPLED STAGE 4 LOGIC ON ALL PHONE SESSIONS")
print("="*70)

UPRIGHT_ADL_CLASSES = {5, 8, 9, 10, 11, 12, 13} # Jumping, Running, Sitting, SitDown, Standing, StandUp, Walking
FALL_AND_RECUMBENT_CLASSES = {0, 1, 2, 3, 4, 6} # Falls + LyingDown

def evaluate_phone_session(fpath, rule_mode="improved"):
    df = pd.read_csv(fpath)
    raw = df.iloc[:, 2:11].values.astype(np.float32)
    n = len(raw)
    if n < 100: return False, "Too short"
    
    state = "MONITORING"
    impactCountdown = 0
    
    for start in range(0, n - 100 + 1, 50):
        window = raw[start : start + 100]
        
        # Kinematics
        acc_mag = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
        maxAccMag = float(np.max(acc_mag))
        minAccMag = float(np.min(acc_mag))
        accRange = maxAccMag - minAccMag
        accStd = float(np.std(acc_mag))
        
        # Max jerk
        jerk = np.abs(np.diff(acc_mag)) / 0.02
        maxJerk = float(np.max(jerk)) if len(jerk) > 0 else 0.0
        
        # Gyro peak
        gyro_mag = np.sqrt(window[:, 3]**2 + window[:, 4]**2 + window[:, 5]**2)
        maxGyroMag = float(np.max(gyro_mag))
        
        isCollisionShock = (maxAccMag >= 20.0 and maxJerk >= 350.0) or \
                           (accRange >= 14.0 and maxJerk >= 250.0 and maxGyroMag >= 3.5)
                           
        isLocomotionCadence = (accStd >= 3.2 and maxGyroMag >= 3.5) or (accStd >= 5.0)
        isSettledImmobility = (accStd <= 2.4) and (maxGyroMag <= 2.2)
        
        # Model inference
        scaled = (window - p_med) / p_iqr
        inp = scaled.reshape(1, 100, 9)
        logits = p_session.run(None, {p_inp_name: inp})[0][0]
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        topIdx = int(np.argmax(probs))
        fallProb = float(np.sum(probs[0:5]))
        lyingDownProb = float(probs[6])
        walkingProb = float(probs[13])
        standingProb = float(probs[11])
        
        if rule_mode == "13d_baseline":
            has_fall_posture = (fallProb >= 0.40) or (lyingDownProb >= 0.45 and accStd <= 1.8)
        else: # improved principled rule
            # Rule 1: Model must NOT clearly identify the subject as actively walking or standing upright
            # Rule 2: Top class must be Fall/LyingDown OR fallProb must exceed walking/standing probabilities
            is_upright_adl = (topIdx in [11, 13]) and (probs[topIdx] > fallProb)
            has_fall_posture = (not is_upright_adl) and (
                (topIdx in FALL_AND_RECUMBENT_CLASSES and (fallProb >= 0.35 or lyingDownProb >= 0.40)) or
                (fallProb >= 0.50 and fallProb > (walkingProb + standingProb)) or
                (lyingDownProb >= 0.50 and accStd <= 1.5)
            )
            
        if state == "MONITORING":
            if isCollisionShock:
                impactCountdown = 4
            elif impactCountdown > 0:
                if isLocomotionCadence:
                    impactCountdown = 0
                elif isSettledImmobility and has_fall_posture:
                    impactCountdown = 0
                    state = "FALL_SUSPECTED"
                    return True, f"Triggered at win {start//50} (top={topIdx}, fallP={fallProb:.2f}, std={accStd:.2f})"
                else:
                    impactCountdown -= 1
                    
    return False, "Completed without trigger"

for mode in ["13d_baseline", "improved"]:
    print(f"\n--- EVALUATING MODE: {mode.upper()} ---")
    p_sessions = split_df[split_df["device"] == "PHONE"]
    
    falls_total = 0
    falls_detected = 0
    adls_total = 0
    adls_fa = 0
    fa_by_act = {}
    recall_by_act = {}
    
    for _, row in p_sessions.iterrows():
        fpath = os.path.join(WORKSPACE, "raw_dataset/phone", row["filename"])
        if not os.path.exists(fpath): continue
        is_fall = row["fall_binary"] == "FALL"
        act = row["activity"]
        
        trig, reason = evaluate_phone_session(fpath, rule_mode=mode)
        
        if is_fall:
            falls_total += 1
            if act not in recall_by_act: recall_by_act[act] = [0, 0]
            recall_by_act[act][1] += 1
            if trig:
                falls_detected += 1
                recall_by_act[act][0] += 1
        else:
            adls_total += 1
            if trig:
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
