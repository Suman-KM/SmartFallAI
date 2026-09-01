import os, sys, json
import numpy as np
import pandas as pd
import onnxruntime as ort

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

# Load Scalers
with open(os.path.join(PREPROC_DIR, "phone/scaler.json")) as f:
    ps = json.load(f)
p_med = np.array(ps["median"], dtype=np.float32)
p_iqr = np.array(ps["iqr"], dtype=np.float32)

with open(os.path.join(PREPROC_DIR, "watch/scaler.json")) as f:
    ws = json.load(f)
w_med = np.array(ws["median"], dtype=np.float32)
w_iqr = np.array(ws["iqr"], dtype=np.float32)

# Load Models
phone_onnx = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))

with open(os.path.join(WORKSPACE, "app/src/main/assets/label_map.json")) as f:
    lm = json.load(f)
classes = lm['classes_14']

import joblib

watch_rf = joblib.load(os.path.join(WORKSPACE, "ml/models/watch/model.joblib"))

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

def extract_watch_features_from_window(w_scaled):
    # w_scaled: (100, 9)
    feats = []
    for col in range(9):
        c = w_scaled[:, col]
        feats.append(float(np.mean(c)))
        feats.append(float(np.std(c)))
        feats.append(float(np.min(c)))
        feats.append(float(np.max(c)))
        feats.append(float(np.max(c) - np.min(c)))
        feats.append(float(np.median(c)))
        feats.append(float(np.sqrt(np.mean(c**2))))
        feats.append(float(np.sum(c**2)))
    return np.array(feats, dtype=np.float32)

def prepare_dataset(device, split):
    split_dir = os.path.join(PREPROC_DIR, device, split)
    X_scaled = np.load(os.path.join(split_dir, "X.npy"))
    y_14 = np.load(os.path.join(split_dir, "y_14.npy"))
    meta_df = pd.read_csv(os.path.join(split_dir, "metadata.csv"))
    
    med = p_med if device == "phone" else w_med
    iqr = p_iqr if device == "phone" else w_iqr
    X_raw = (X_scaled * iqr) + med
    
    acc = X_raw[:, :, :3]
    gyro = X_raw[:, :, 3:6]
    acc_mag = np.linalg.norm(acc, axis=2) # (N, 100)
    gyro_mag = np.linalg.norm(gyro, axis=2) # (N, 100)
    
    dt = 0.02
    jerk = np.abs(np.diff(acc_mag, axis=1)) / dt # (N, 99)
    
    meta_df['acc_peak'] = np.max(acc_mag, axis=1)
    meta_df['acc_min'] = np.min(acc_mag, axis=1)
    meta_df['acc_range'] = meta_df['acc_peak'] - meta_df['acc_min']
    meta_df['acc_std'] = np.std(acc_mag, axis=1)
    meta_df['jerk_peak'] = np.max(jerk, axis=1)
    meta_df['gyro_peak'] = np.max(gyro_mag, axis=1)
    meta_df['gyro_mean'] = np.mean(gyro_mag, axis=1)
    meta_df['y_14'] = y_14
    meta_df['is_fall'] = y_14 < 5
    
    if device == "phone":
        logits = phone_onnx.run(None, {phone_onnx.get_inputs()[0].name: X_scaled.astype(np.float32)})[0]
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e / np.sum(e, axis=1, keepdims=True)
    else:
        # Watch
        feats = extract_watch_features(X_scaled)
        probs = watch_rf.predict_proba(feats)
        
    fall_probs = np.sum(probs[:, :5], axis=1)
    meta_df['fall_prob'] = fall_probs
    meta_df['top_class_idx'] = np.argmax(probs, axis=1)
    meta_df['top_class_prob'] = np.max(probs, axis=1)
    meta_df['lying_down_prob'] = probs[:, 6] # Index 6 is LYING_DOWN
    
    return meta_df

print("Loading validation datasets...")
p_val_df = prepare_dataset("phone", "validation")
w_val_df = prepare_dataset("watch", "validation")
print(f"Phone Val Windows: {len(p_val_df)}, Watch Val Windows: {len(w_val_df)}")

def evaluate_architecture(df, arch_name, device="phone"):
    unique_sids = df['source_session_id'].unique()
    
    # Parameters per device
    imp_peak_th = 20.0 if device == "phone" else 24.0
    imp_jerk_th = 400.0 if device == "phone" else 600.0
    unloading_th = 7.0 if device == "phone" else 6.5
    
    detected_falls = 0
    total_falls = 0
    false_alarms = 0
    total_adls = 0
    
    fa_by_activity = {}
    recall_by_fall = {}
    detection_latencies = []
    
    for sid in unique_sids:
        s_df = df[df['source_session_id'] == sid].sort_values('window_index').reset_index(drop=True)
        is_fall = s_df['is_fall'].iloc[0]
        act = s_df['target_activity'].iloc[0]
        n_wins = len(s_df)
        
        triggered = False
        trig_win = -1
        
        # State machine simulation
        if arch_name == 'A': # ML Only
            for i, r in s_df.iterrows():
                if r['fall_prob'] >= 0.50:
                    triggered = True
                    trig_win = i
                    break
                    
        elif arch_name == 'B': # ML + Impact
            for i, r in s_df.iterrows():
                has_imp = (r['acc_peak'] >= imp_peak_th)
                if r['fall_prob'] >= 0.50 and has_imp:
                    triggered = True
                    trig_win = i
                    break
                    
        elif arch_name == 'C': # ML + Impact + Temporal Consensus (Phase 13C Baseline)
            mem = 0
            consec = 0
            for i, r in s_df.iterrows():
                has_imp = (r['acc_peak'] >= (18.0 if device=="phone" else 20.0)) or (r['acc_range'] >= (10.0 if device=="phone" else 12.0) and r['gyro_peak'] >= (2.5 if device=="phone" else 3.0))
                if has_imp:
                    mem = 3
                elif mem > 0:
                    mem -= 1
                is_thrash = (r['acc_std'] >= (4.5 if device=="phone" else 9.0) and r['gyro_peak'] >= (3.5 if device=="phone" else 4.0))
                is_cand = (r['fall_prob'] >= 0.45) and (has_imp or mem > 0) and (not is_thrash)
                if is_cand:
                    consec += 1
                    if consec >= 2:
                        triggered = True
                        trig_win = i
                        break
                else:
                    consec = 0
                    
        elif arch_name == 'D': # ML + Impact + Post-Event Stillness
            for i in range(n_wins):
                r = s_df.iloc[i]
                has_imp = (r['acc_peak'] >= imp_peak_th)
                if has_imp and r['fall_prob'] >= 0.40:
                    # Require post-event stillness in next window (if exists)
                    if i + 1 < n_wins:
                        r_next = s_df.iloc[i+1]
                        if r_next['acc_std'] <= (2.5 if device=="phone" else 3.5):
                            triggered = True
                            trig_win = i + 1
                            break
                    else:
                        triggered = True
                        trig_win = i
                        break
                        
        elif arch_name == 'E': # ML + Impact + Temporal Trajectory (Pre-drop + Jerk)
            for i in range(n_wins):
                r = s_df.iloc[i]
                has_trajectory_impact = (r['acc_peak'] >= imp_peak_th) and (r['jerk_peak'] >= imp_jerk_th) and (r['acc_min'] <= unloading_th)
                if has_trajectory_impact and (r['fall_prob'] >= 0.40 or r['lying_down_prob'] >= 0.40):
                    triggered = True
                    trig_win = i
                    break
                    
        elif arch_name == 'F': # ML + Impact + Trajectory + Post-Event Stillness
            for i in range(n_wins):
                r = s_df.iloc[i]
                has_trajectory_impact = (r['acc_peak'] >= imp_peak_th) and (r['jerk_peak'] >= imp_jerk_th)
                if has_trajectory_impact:
                    if i + 1 < n_wins:
                        r_next = s_df.iloc[i+1]
                        is_still = (r_next['acc_std'] <= (2.5 if device=="phone" else 4.0)) and (r_next['gyro_peak'] <= 2.5)
                        if is_still and (r_next['fall_prob'] >= 0.35 or r_next['lying_down_prob'] >= 0.40 or r['fall_prob'] >= 0.40):
                            triggered = True
                            trig_win = i + 1
                            break
                    else:
                        if r['fall_prob'] >= 0.40:
                            triggered = True
                            trig_win = i
                            break
                            
        elif arch_name == 'G': # ML + Impact + Trajectory + Movement-Continuation Rejection
            mem = 0
            for i in range(n_wins):
                r = s_df.iloc[i]
                has_trajectory_impact = (r['acc_peak'] >= imp_peak_th) and (r['jerk_peak'] >= imp_jerk_th)
                if has_trajectory_impact:
                    mem = 2
                elif mem > 0:
                    mem -= 1
                    
                is_active_continuation = (r['acc_std'] >= (3.5 if device=="phone" else 6.0)) or (r['gyro_peak'] >= (3.5 if device=="phone" else 4.5))
                if mem > 0 and (not is_active_continuation) and (r['fall_prob'] >= 0.40 or r['lying_down_prob'] >= 0.50):
                    triggered = True
                    trig_win = i
                    break
                    
        elif arch_name == 'H': # Best Multi-Stage Unified Architecture
            # Stage 1: Candidate event: Impact Shock (High Acc + High Jerk OR Tumble)
            # Stage 2: Temporal trajectory: Descent/Unloading or Fast Jerk Shock
            # Stage 3: Movement Continuation Filter: Reject ongoing cadence
            # Stage 4: Post-Impact Posture/Stillness confirmation in window t or t+1
            impact_history = 0
            for i in range(n_wins):
                r = s_df.iloc[i]
                
                # Check for collision impact
                is_hard_impact = (r['acc_peak'] >= imp_peak_th) and (r['jerk_peak'] >= imp_jerk_th)
                is_rotational_fall = (r['acc_range'] >= 14.0) and (r['jerk_peak'] >= 300.0) and (r['gyro_peak'] >= 4.0)
                
                if is_hard_impact or is_rotational_fall:
                    impact_history = 2 # Valid for 2 steps (current and next)
                elif impact_history > 0:
                    impact_history -= 1
                    
                # Active motion check in this window
                is_continuous_locomotion = (r['acc_std'] >= (3.8 if device=="phone" else 6.5)) and (r['gyro_peak'] >= (3.5 if device=="phone" else 4.0))
                
                # Post-impact rest or fall classification
                has_fall_signature = (r['fall_prob'] >= 0.40) or (r['lying_down_prob'] >= 0.45 and r['acc_std'] <= 2.2)
                
                if impact_history > 0 and (not is_continuous_locomotion) and has_fall_signature:
                    # Require 1 confirmation of non-locomotion
                    triggered = True
                    trig_win = i
                    break
                    
        # Tally metrics
        if is_fall:
            total_falls += 1
            if act not in recall_by_fall:
                recall_by_fall[act] = [0, 0]
            recall_by_fall[act][1] += 1
            if triggered:
                detected_falls += 1
                recall_by_fall[act][0] += 1
                detection_latencies.append(trig_win * 1.0)
        else:
            total_adls += 1
            if triggered:
                false_alarms += 1
                fa_by_activity[act] = fa_by_activity.get(act, 0) + 1
                
    rec = (detected_falls / total_falls) * 100.0 if total_falls > 0 else 0.0
    prec = (detected_falls / (detected_falls + false_alarms)) * 100.0 if (detected_falls + false_alarms) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    spec = ((total_adls - false_alarms) / total_adls) * 100.0 if total_adls > 0 else 0.0
    fpr = (false_alarms / total_adls) * 100.0 if total_adls > 0 else 0.0
    med_lat = np.median(detection_latencies) if detection_latencies else 0.0
    
    return {
        "arch": arch_name,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "specificity": spec,
        "fpr": fpr,
        "fa_count": false_alarms,
        "total_adls": total_adls,
        "fa_activities": fa_by_activity,
        "recall_by_fall": recall_by_fall,
        "median_latency": med_lat
    }

archs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
print("\n--- PHONE VALIDATION ARCHITECTURE COMPARISON ---")
p_results = [evaluate_architecture(p_val_df, a, "phone") for a in archs]
print(f"{'Arch':<5} | {'Recall':<8} | {'Precision':<9} | {'F1':<6} | {'FPR':<7} | {'FAs':<8} | {'High-Motion FAs'}")
print("-" * 75)
for r in p_results:
    hm_fas = {k: v for k, v in r['fa_activities'].items() if k in ['WALKING', 'RUNNING', 'JUMPING']}
    print(f"{r['arch']:<5} | {r['recall']:>6.1f}% | {r['precision']:>8.1f}% | {r['f1']:>5.1f}% | {r['fpr']:>6.1f}% | {r['fa_count']}/{r['total_adls']} | {hm_fas}")

print("\n--- WATCH VALIDATION ARCHITECTURE COMPARISON ---")
w_results = [evaluate_architecture(w_val_df, a, "watch") for a in archs]
print(f"{'Arch':<5} | {'Recall':<8} | {'Precision':<9} | {'F1':<6} | {'FPR':<7} | {'FAs':<8} | {'High-Motion FAs'}")
print("-" * 75)
for r in w_results:
    hm_fas = {k: v for k, v in r['fa_activities'].items() if k in ['WALKING', 'RUNNING', 'JUMPING']}
    print(f"{r['arch']:<5} | {r['recall']:>6.1f}% | {r['precision']:>8.1f}% | {r['f1']:>5.1f}% | {r['fpr']:>6.1f}% | {r['fa_count']}/{r['total_adls']} | {hm_fas}")
