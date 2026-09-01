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

# Load Phone ONNX
phone_onnx = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))
# Load Watch RF
watch_rf = joblib.load(os.path.join(WORKSPACE, "ml/models/watch/model.joblib"))

# Load scalers
with open(os.path.join(PREPROC_DIR, "phone/scaler.json")) as f:
    ps = json.load(f)
p_med, p_iqr = np.array(ps["median"], dtype=np.float32), np.array(ps["iqr"], dtype=np.float32)

with open(os.path.join(PREPROC_DIR, "watch/scaler.json")) as f:
    ws = json.load(f)
w_med, w_iqr = np.array(ws["median"], dtype=np.float32), np.array(ws["iqr"], dtype=np.float32)

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

def prepare_dataset(device, split="validation"):
    split_dir = os.path.join(PREPROC_DIR, device, split)
    X_scaled = np.load(os.path.join(split_dir, "X.npy"))
    y_14 = np.load(os.path.join(split_dir, "y_14.npy"))
    meta_df = pd.read_csv(os.path.join(split_dir, "metadata.csv"))
    
    med = p_med if device == "phone" else w_med
    iqr = p_iqr if device == "phone" else w_iqr
    X_raw = (X_scaled * iqr) + med
    
    # Compute Model Probabilities
    if device == "phone":
        logits = phone_onnx.run(None, {phone_onnx.get_inputs()[0].name: X_scaled.astype(np.float32)})[0]
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e / np.sum(e, axis=1, keepdims=True)
    else:
        feats = extract_watch_features(X_scaled)
        probs = watch_rf.predict_proba(feats)
        
    fall_probs = np.sum(probs[:, :5], axis=1)
    
    acc = X_raw[:, :, :3]
    gyro = X_raw[:, :, 3:6]
    acc_mag = np.linalg.norm(acc, axis=2)
    gyro_mag = np.linalg.norm(gyro, axis=2)
    
    acc_peak = np.max(acc_mag, axis=1)
    acc_min = np.min(acc_mag, axis=1)
    acc_range = acc_peak - acc_min
    acc_std = np.std(acc_mag, axis=1)
    gyro_peak = np.max(gyro_mag, axis=1)
    gyro_std = np.std(gyro_mag, axis=1)
    
    # Stillness / Inactivity in tail (last 40 samples, 0.8 sec)
    tail_acc_std = np.std(acc_mag[:, 60:], axis=1)
    tail_gyro_mean = np.mean(gyro_mag[:, 60:], axis=1)
    
    # Posture change: absolute change in pitch and roll between first 20 and last 20 samples
    pitch = X_raw[:, :, 6]
    roll = X_raw[:, :, 7]
    pitch_change = np.abs(np.mean(pitch[:, 80:], axis=1) - np.mean(pitch[:, :20], axis=1))
    roll_change = np.abs(np.mean(roll[:, 80:], axis=1) - np.mean(roll[:, :20], axis=1))
    posture_change = np.maximum(pitch_change, roll_change)
    
    meta_df['fall_prob'] = fall_probs
    meta_df['acc_peak'] = acc_peak
    meta_df['acc_min'] = acc_min
    meta_df['acc_range'] = acc_range
    meta_df['acc_std'] = acc_std
    meta_df['gyro_peak'] = gyro_peak
    meta_df['gyro_std'] = gyro_std
    meta_df['tail_acc_std'] = tail_acc_std
    meta_df['tail_gyro_mean'] = tail_gyro_mean
    meta_df['posture_change'] = posture_change
    meta_df['y_14'] = y_14
    meta_df['is_fall'] = y_14 < 5
    
    return meta_df

def simulate_pipeline(df, p_thresh=0.50, acc_peak_thresh=18.0, acc_rng_thresh=10.0,
                      gyro_peak_thresh=2.5, min_acc_drop=8.0, require_stillness=False,
                      max_tail_gyro=1.5, max_tail_acc_std=3.0,
                      require_posture_change=False, min_posture_change=15.0,
                      consecutive_required=2, memory_len=3):
    """
    Simulates the state machine at the session level.
    Returns:
      fall_detected_sessions, false_alarm_sessions, metrics
    """
    total_fall_sessions = df[df['is_fall']]['source_session_id'].nunique()
    total_adl_sessions = df[~df['is_fall']]['source_session_id'].nunique()
    
    detected_falls = 0
    detected_by_type = {
        'FALL_BACKWARD': 0, 'FALL_FORWARD': 0, 'FALL_FROM_SITTING': 0,
        'FALL_LEFT': 0, 'FALL_RIGHT': 0
    }
    total_by_type = {
        'FALL_BACKWARD': 0, 'FALL_FORWARD': 0, 'FALL_FROM_SITTING': 0,
        'FALL_LEFT': 0, 'FALL_RIGHT': 0
    }
    for k in total_by_type:
        total_by_type[k] = df[df['target_activity'] == k]['source_session_id'].nunique()
        
    false_alarms_by_act = {}
    total_false_alarm_sessions = 0
    
    for sid, s_df in df.groupby('source_session_id'):
        s_df = s_df.sort_values('window_index')
        is_fall_session = s_df['is_fall'].iloc[0]
        act_name = s_df['target_activity'].iloc[0]
        
        consecutive = 0
        impact_memory = 0
        triggered = False
        trigger_delay = -1
        
        for w_idx, row in s_df.iterrows():
            fp = row['fall_prob']
            ap = row['acc_peak']
            ar = row['acc_range']
            gp = row['gyro_peak']
            amin = row['acc_min']
            t_gmean = row['tail_gyro_mean']
            t_astd = row['tail_acc_std']
            pchg = row['posture_change']
            
            # Impact condition: must have sudden deceleration and free-fall drop
            has_impact = (ap >= acc_peak_thresh) or (ar >= acc_rng_thresh and gp >= gyro_peak_thresh)
            if min_acc_drop is not None and has_impact:
                has_impact = has_impact and (amin <= min_acc_drop)
                
            if has_impact:
                impact_memory = memory_len
            elif impact_memory > 0:
                impact_memory -= 1
                
            # Candidate condition
            is_cand = (fp >= p_thresh) and (has_impact or impact_memory > 0)
            
            if require_stillness and is_cand:
                # Stillness requirement: tail must not be wildly thrashing
                is_cand = is_cand and (t_gmean <= max_tail_gyro and t_astd <= max_tail_acc_std)
                
            if require_posture_change and is_cand:
                is_cand = is_cand and (pchg >= min_posture_change)
                
            if is_cand:
                consecutive += 1
                if consecutive >= consecutive_required:
                    triggered = True
                    trigger_delay = int(row['window_index'])
                    break
            else:
                consecutive = 0
                
        if is_fall_session:
            if triggered:
                detected_falls += 1
                if act_name in detected_by_type:
                    detected_by_type[act_name] += 1
        else:
            if triggered:
                total_false_alarm_sessions += 1
                false_alarms_by_act[act_name] = false_alarms_by_act.get(act_name, 0) + 1
                
    recall = (detected_falls / total_fall_sessions) if total_fall_sessions > 0 else 0.0
    prec = (detected_falls / (detected_falls + total_false_alarm_sessions)) if (detected_falls + total_false_alarm_sessions) > 0 else 0.0
    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
    fpr = (total_false_alarm_sessions / total_adl_sessions) if total_adl_sessions > 0 else 0.0
    specificity = 1.0 - fpr
    
    return {
        "recall": recall,
        "precision": prec,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "detected_falls": detected_falls,
        "total_falls": total_fall_sessions,
        "false_alarm_sessions": total_false_alarm_sessions,
        "total_adl_sessions": total_adl_sessions,
        "detected_by_type": detected_by_type,
        "total_by_type": total_by_type,
        "false_alarms_by_act": false_alarms_by_act
    }

# Run evaluation of Candidates A, B, C, D, E on Phone Validation
p_val_df = prepare_dataset("phone", "validation")
w_val_df = prepare_dataset("watch", "validation")

print("\n=======================================================")
print("EVALUATING CANDIDATE ARCHITECTURES ON PHONE VALIDATION")
print("=======================================================")
# Candidate A: ML only (P >= 0.50, no impact, consec=1)
candA_p = simulate_pipeline(p_val_df, p_thresh=0.50, acc_peak_thresh=0, acc_rng_thresh=0, gyro_peak_thresh=0, min_acc_drop=None, consecutive_required=1, memory_len=0)
# Candidate B: ML + Impact Gate (P >= 0.50, ap >= 16, consec=1)
candB_p = simulate_pipeline(p_val_df, p_thresh=0.50, acc_peak_thresh=16.0, acc_rng_thresh=10.0, gyro_peak_thresh=2.5, min_acc_drop=None, consecutive_required=1, memory_len=3)
# Candidate C: ML + Impact Gate + Consensus (Phase 13B: ap >= 16, consec=2)
candC_p = simulate_pipeline(p_val_df, p_thresh=0.50, acc_peak_thresh=16.0, acc_rng_thresh=10.0, gyro_peak_thresh=2.5, min_acc_drop=None, consecutive_required=2, memory_len=3)
# Candidate D: ML + Impact Gate (ap >= 20.0, drop <= 8.0) + Consensus + Stillness
candD_p = simulate_pipeline(p_val_df, p_thresh=0.50, acc_peak_thresh=20.0, acc_rng_thresh=12.0, gyro_peak_thresh=2.5, min_acc_drop=8.0, require_stillness=True, max_tail_gyro=1.5, max_tail_acc_std=2.5, consecutive_required=2, memory_len=3)
# Candidate E: Candidate D + Posture Change
candE_p = simulate_pipeline(p_val_df, p_thresh=0.50, acc_peak_thresh=20.0, acc_rng_thresh=12.0, gyro_peak_thresh=2.5, min_acc_drop=8.0, require_stillness=True, max_tail_gyro=1.5, max_tail_acc_std=2.5, require_posture_change=True, min_posture_change=15.0, consecutive_required=2, memory_len=3)

for name, res in [("Cand A (ML Only)", candA_p), ("Cand B (ML+Impact)", candB_p), ("Cand C (ML+Impact+Consensus)", candC_p), ("Cand D (ML+Impact+Consensus+Stillness)", candD_p), ("Cand E (ML+Impact+Consensus+Stillness+Posture)", candE_p)]:
    print(f"{name:<45}: Recall={res['recall']*100:5.1f}%, Prec={res['precision']*100:5.1f}%, F1={res['f1']*100:5.1f}%, FPR={res['fpr']*100:5.1f}% (FA Sessions={res['false_alarm_sessions']}/{res['total_adl_sessions']}) | FAs: {res['false_alarms_by_act']}")

print("\n=======================================================")
print("EVALUATING CANDIDATE ARCHITECTURES ON WATCH VALIDATION")
print("=======================================================")
candA_w = simulate_pipeline(w_val_df, p_thresh=0.50, acc_peak_thresh=0, acc_rng_thresh=0, gyro_peak_thresh=0, min_acc_drop=None, consecutive_required=1, memory_len=0)
candB_w = simulate_pipeline(w_val_df, p_thresh=0.50, acc_peak_thresh=16.0, acc_rng_thresh=10.0, gyro_peak_thresh=2.5, min_acc_drop=None, consecutive_required=1, memory_len=3)
candC_w = simulate_pipeline(w_val_df, p_thresh=0.50, acc_peak_thresh=16.0, acc_rng_thresh=10.0, gyro_peak_thresh=2.5, min_acc_drop=None, consecutive_required=2, memory_len=3)
candD_w = simulate_pipeline(w_val_df, p_thresh=0.50, acc_peak_thresh=22.0, acc_rng_thresh=15.0, gyro_peak_thresh=2.5, min_acc_drop=8.0, require_stillness=True, max_tail_gyro=1.8, max_tail_acc_std=3.0, consecutive_required=2, memory_len=3)
candE_w = simulate_pipeline(w_val_df, p_thresh=0.50, acc_peak_thresh=22.0, acc_rng_thresh=15.0, gyro_peak_thresh=2.5, min_acc_drop=8.0, require_stillness=True, max_tail_gyro=1.8, max_tail_acc_std=3.0, require_posture_change=True, min_posture_change=15.0, consecutive_required=2, memory_len=3)

for name, res in [("Cand A (ML Only)", candA_w), ("Cand B (ML+Impact)", candB_w), ("Cand C (ML+Impact+Consensus)", candC_w), ("Cand D (ML+Impact+Consensus+Stillness)", candD_w), ("Cand E (ML+Impact+Consensus+Stillness+Posture)", candE_w)]:
    print(f"{name:<45}: Recall={res['recall']*100:5.1f}%, Prec={res['precision']*100:5.1f}%, F1={res['f1']*100:5.1f}%, FPR={res['fpr']*100:5.1f}% (FA Sessions={res['false_alarm_sessions']}/{res['total_adl_sessions']}) | FAs: {res['false_alarms_by_act']}")
