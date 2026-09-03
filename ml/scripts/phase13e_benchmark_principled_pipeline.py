import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

# Load Phone Scaler & Model
with open(os.path.join(WORKSPACE, "preprocessing/02_robust_scaling/phone/scaler.json")) as f:
    p_scaler = json.load(f)
p_med = np.array(p_scaler["median"], dtype=np.float32)
p_iqr = np.array(p_scaler["iqr"], dtype=np.float32)
p_session = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))
p_inp_name = p_session.get_inputs()[0].name

# Load Watch Scaler & Model
with open(os.path.join(WORKSPACE, "preprocessing/02_robust_scaling/watch/scaler.json")) as f:
    w_scaler = json.load(f)
w_med = np.array(w_scaler["median"], dtype=np.float32)
w_iqr = np.array(w_scaler["iqr"], dtype=np.float32)
rf_model = joblib.load(os.path.join(WORKSPACE, "ml/models/watch/model.joblib"))

split_df = pd.read_csv(os.path.join(WORKSPACE, "preprocessing/common_split/all_split_sessions.csv"))

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

class PrincipledFallDetector:
    def __init__(self, device="phone"):
        self.device = device.lower()
        if self.device == "phone":
            self.imp_peak_th = 22.0
            self.imp_jerk_th = 400.0
            self.imp_range_th = 15.0
            self.imp_jerk_alt_th = 300.0
            self.imp_gyro_th = 3.5
            self.imp_unloading_th = 7.5
            
            # Locomotion cadence: walking or running produces continuous dynamic motion
            self.loco_std_th = 1.8
            self.loco_gyro_th = 1.8
            self.loco_std_high = 3.2
            self.loco_gyro_high = 3.0
            
            # Stillness: recumbent floor rest is completely motionless
            self.still_std_th = 1.2
            self.still_gyro_th = 1.2
            self.verification_horizon = 4
        else:
            self.imp_peak_th = 26.0
            self.imp_jerk_th = 600.0
            self.imp_range_th = 18.0
            self.imp_jerk_alt_th = 400.0
            self.imp_gyro_th = 4.0
            self.imp_unloading_th = 7.5
            
            self.loco_std_th = 2.5
            self.loco_gyro_th = 2.5
            self.loco_std_high = 5.0
            self.loco_gyro_high = 4.0
            
            self.still_std_th = 1.8
            self.still_gyro_th = 1.5
            self.verification_horizon = 4

    def evaluate_session(self, raw_feats):
        n = len(raw_feats)
        if n < 100: return False, "Too short"
        
        state = "MONITORING"
        impactCountdown = 0
        
        for start in range(0, n - 100 + 1, 50):
            window = raw_feats[start : start + 100]
            
            acc_mag = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
            maxAccMag = float(np.max(acc_mag))
            minAccMag = float(np.min(acc_mag))
            accRange = maxAccMag - minAccMag
            accStd = float(np.std(acc_mag))
            
            jerk = np.abs(np.diff(acc_mag)) / 0.02
            maxJerk = float(np.max(jerk)) if len(jerk) > 0 else 0.0
            
            gyro_mag = np.sqrt(window[:, 3]**2 + window[:, 4]**2 + window[:, 5]**2)
            maxGyroMag = float(np.max(gyro_mag))
            
            # 1. Collision Shock with Unloading
            isCollisionShock = (maxAccMag >= self.imp_peak_th and maxJerk >= self.imp_jerk_th and minAccMag <= self.imp_unloading_th) or \
                               (accRange >= self.imp_range_th and maxJerk >= self.imp_jerk_alt_th and maxGyroMag >= self.imp_gyro_th and minAccMag <= self.imp_unloading_th)
                               
            # 2. Locomotion Cadence
            isLocomotionCadence = (accStd >= self.loco_std_th and maxGyroMag >= self.loco_gyro_th) or \
                                  (accStd >= self.loco_std_high) or (maxGyroMag >= self.loco_gyro_high)
                                  
            # 3. Post-Impact Immobility
            isSettledImmobility = (accStd <= self.still_std_th) and (maxGyroMag <= self.still_gyro_th)
            
            # 4. Model Inference
            if self.device == "phone":
                scaled = (window - p_med) / p_iqr
                inp = scaled.reshape(1, 100, 9)
                logits = p_session.run(None, {p_inp_name: inp})[0][0]
                e = np.exp(logits - np.max(logits))
                probs = e / e.sum()
            else:
                scaled = (window - w_med) / w_iqr
                feats = extract_watch_features(scaled)
                probs = rf_model.predict_proba(feats)[0]
                
            topIdx = int(np.argmax(probs))
            topConf = float(probs[topIdx])
            fallProb = float(np.sum(probs[0:5]))
            lyingDownProb = float(probs[6]) if len(probs) > 6 else 0.0
            walkingProb = float(probs[13]) if len(probs) > 13 else 0.0
            standingProb = float(probs[11]) if len(probs) > 11 else 0.0
            
            # Posture Consistency Check:
            # The victim on the floor must NOT be classified as actively walking or standing upright!
            is_upright_adl = (topIdx in [11, 13]) and (topConf > fallProb)
            has_fall_posture = (not is_upright_adl) and (
                (topIdx in [0, 1, 2, 3, 4, 6] and (fallProb >= 0.35 or lyingDownProb >= 0.40)) or
                (fallProb >= 0.50 and fallProb > (walkingProb + standingProb)) or
                (lyingDownProb >= 0.50 and accStd <= 0.8)
            )
            
            if state == "MONITORING":
                if isCollisionShock:
                    impactCountdown = self.verification_horizon
                elif impactCountdown > 0:
                    if isLocomotionCadence:
                        impactCountdown = 0
                    elif isSettledImmobility and has_fall_posture:
                        impactCountdown = 0
                        state = "FALL_SUSPECTED"
                        return True, f"Win {start//50}: top={topIdx}, fallP={fallProb:.2f}, std={accStd:.2f}"
                    else:
                        impactCountdown -= 1
                        
        return False, "Completed without trigger"

def run_evaluation(device_name):
    detector = PrincipledFallDetector(device_name)
    sub = split_df[split_df["device"] == device_name.upper()]
    
    falls_tot = 0
    falls_det = 0
    adls_tot = 0
    adls_fa = 0
    fa_by_act = {}
    rec_by_act = {}
    
    for _, row in sub.iterrows():
        fpath = os.path.join(WORKSPACE, f"raw_dataset/{device_name.lower()}", row["filename"])
        if not os.path.exists(fpath): continue
        is_fall = row["fall_binary"] == "FALL"
        act = row["activity"]
        
        df = pd.read_csv(fpath)
        raw = df.iloc[:, 2:11].values.astype(np.float32)
        trig, reason = detector.evaluate_session(raw)
        
        if is_fall:
            falls_tot += 1
            if act not in rec_by_act: rec_by_act[act] = [0, 0]
            rec_by_act[act][1] += 1
            if trig:
                falls_det += 1
                rec_by_act[act][0] += 1
        else:
            adls_tot += 1
            if trig:
                adls_fa += 1
                fa_by_act[act] = fa_by_act.get(act, 0) + 1
                
    rec = (falls_det / falls_tot) * 100.0 if falls_tot > 0 else 0.0
    prec = (falls_det / (falls_det + adls_fa)) * 100.0 if (falls_det + adls_fa) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = (adls_fa / adls_tot) * 100.0 if adls_tot > 0 else 0.0
    
    print(f"==================================================")
    print(f"DEVICE: {device_name.upper()} (ALL SESSIONS N={len(sub)})")
    print(f"==================================================")
    print(f"Fall Recall:      {rec:6.2f}% ({falls_det}/{falls_tot})")
    print(f"Fall Precision:   {prec:6.2f}%")
    print(f"Binary Fall F1:   {f1:6.2f}%")
    print(f"False Alarm Rate: {fpr:6.2f}% ({adls_fa}/{adls_tot})")
    print(f"False Alarms by Activity: {fa_by_act}")
    print("Recall by Fall Type:")
    for k, v in rec_by_act.items():
        pct = (v[0]/v[1])*100.0 if v[1]>0 else 0.0
        print(f"  {k:<20}: {v[0]}/{v[1]} ({pct:.1f}%)")
    print()

run_evaluation("phone")
run_evaluation("watch")
