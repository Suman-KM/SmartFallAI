import os, sys, json
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")
RESULTS_DIR = os.path.join(WORKSPACE, "ml/results/phase13c")

phone_onnx = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))
watch_rf = joblib.load(os.path.join(WORKSPACE, "ml/models/watch/model.joblib"))

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

def load_split_data(device, split):
    split_dir = os.path.join(PREPROC_DIR, device, split)
    X_scaled = np.load(os.path.join(split_dir, "X.npy"))
    y_14 = np.load(os.path.join(split_dir, "y_14.npy"))
    meta_df = pd.read_csv(os.path.join(split_dir, "metadata.csv"))
    
    med = p_med if device == "phone" else w_med
    iqr = p_iqr if device == "phone" else w_iqr
    X_raw = (X_scaled * iqr) + med
    
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
    
    meta_df['fall_prob'] = fall_probs
    meta_df['acc_peak'] = np.max(acc_mag, axis=1)
    meta_df['acc_min'] = np.min(acc_mag, axis=1)
    meta_df['acc_range'] = meta_df['acc_peak'] - meta_df['acc_min']
    meta_df['acc_std'] = np.std(acc_mag, axis=1)
    meta_df['gyro_peak'] = np.max(gyro_mag, axis=1)
    meta_df['gyro_std'] = np.std(gyro_mag, axis=1)
    meta_df['y_14'] = y_14
    meta_df['is_fall'] = y_14 < 5
    return meta_df

def run_evaluation(df, is_calibrated, device):
    total_fall_sessions = df[df['is_fall']]['source_session_id'].nunique()
    total_adl_sessions = df[~df['is_fall']]['source_session_id'].nunique()
    
    # Baseline: ML only (P >= 0.50, no impact gate, no consensus)
    # Calibrated: Device-specific threshold, impact shock, temporal consensus, continuous thrash filter
    if not is_calibrated:
        p_th = 0.50
        ap_th = 0.0
        ar_th = 0.0
        gp_th = 0.0
        consec_req = 1
        mem_len = 0
        use_thrash_filter = False
    else:
        if device == "phone":
            p_th = 0.45
            ap_th = 18.0
            ar_th = 10.0
            gp_th = 2.5
            consec_req = 2
            mem_len = 3
            use_thrash_filter = True
            max_astd = 4.5
            max_gp = 3.5
        else:
            p_th = 0.45
            ap_th = 20.0
            ar_th = 12.0
            gp_th = 3.0
            consec_req = 2
            mem_len = 3
            use_thrash_filter = True
            max_astd = 9.0
            max_gp = 4.0
            
    det_falls = 0
    det_by_type = {'FALL_BACKWARD': 0, 'FALL_FORWARD': 0, 'FALL_FROM_SITTING': 0, 'FALL_LEFT': 0, 'FALL_RIGHT': 0}
    total_by_type = {'FALL_BACKWARD': 0, 'FALL_FORWARD': 0, 'FALL_FROM_SITTING': 0, 'FALL_LEFT': 0, 'FALL_RIGHT': 0}
    for k in total_by_type:
        total_by_type[k] = df[df['target_activity'] == k]['source_session_id'].nunique()
        
    fa_sessions = 0
    fa_by_act = {}
    
    for sid, s_df in df.groupby('source_session_id'):
        s_df = s_df.sort_values('window_index')
        is_fall = s_df['is_fall'].iloc[0]
        act = s_df['target_activity'].iloc[0]
        
        recent_mem = 0
        consec = 0
        triggered = False
        
        for _, row in s_df.iterrows():
            fp = row['fall_prob']
            ap = row['acc_peak']
            ar = row['acc_range']
            gp = row['gyro_peak']
            astd = row['acc_std']
            
            if is_calibrated:
                has_imp = (ap >= ap_th) or (ar >= ar_th and gp >= gp_th)
                if has_imp:
                    recent_mem = mem_len
                elif recent_mem > 0:
                    recent_mem -= 1
                    
                is_thrash = use_thrash_filter and (astd >= max_astd and gp >= max_gp)
                is_cand = (fp >= p_th) and (has_imp or recent_mem > 0) and (not is_thrash)
            else:
                is_cand = (fp >= p_th)
                
            if is_cand:
                consec += 1
                if consec >= consec_req:
                    triggered = True
                    break
            else:
                consec = 0
                
        if is_fall:
            if triggered:
                det_falls += 1
                if act in det_by_type:
                    det_by_type[act] += 1
        else:
            if triggered:
                fa_sessions += 1
                fa_by_act[act] = fa_by_act.get(act, 0) + 1
                
    rec = det_falls / total_fall_sessions if total_fall_sessions > 0 else 0.0
    prec = det_falls / (det_falls + fa_sessions) if (det_falls + fa_sessions) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fa_sessions / total_adl_sessions if total_adl_sessions > 0 else 0.0
    specificity = 1.0 - fpr
    
    # High motion FAs
    hm_fas = fa_by_act.get('WALKING', 0) + fa_by_act.get('RUNNING', 0) + fa_by_act.get('JUMPING', 0)
    
    return {
        "recall": rec, "precision": prec, "f1": f1, "specificity": specificity,
        "fpr": fpr, "det_falls": det_falls, "total_falls": total_fall_sessions,
        "fa_sessions": fa_sessions, "total_adl": total_adl_sessions,
        "hm_fas": hm_fas, "fa_by_act": fa_by_act, "det_by_type": det_by_type,
        "total_by_type": total_by_type
    }

print("=================================================================")
print("FINAL TEST SET EVALUATION: BASELINE VS CALIBRATED")
print("=================================================================")

p_test_df = load_split_data("phone", "test")
w_test_df = load_split_data("watch", "test")

# Phone Test
p_base = run_evaluation(p_test_df, False, "phone")
p_cal = run_evaluation(p_test_df, True, "phone")

# Watch Test
w_base = run_evaluation(w_test_df, False, "watch")
w_cal = run_evaluation(w_test_df, True, "watch")

def print_comparison(dev, base, cal):
    print(f"\n--- {dev.upper()} TEST RESULTS ---")
    print(f"{'Metric':<30} | {'Baseline':<15} | {'Calibrated':<15} | {'Delta'}")
    print("-" * 75)
    print(f"{'Fall Recall':<30} | {base['recall']*100:6.2f}% ({base['det_falls']}/{base['total_falls']}) | {cal['recall']*100:6.2f}% ({cal['det_falls']}/{cal['total_falls']}) | {(cal['recall']-base['recall'])*100:+6.2f}%")
    print(f"{'Fall Precision':<30} | {base['precision']*100:6.2f}%         | {cal['precision']*100:6.2f}%         | {(cal['precision']-base['precision'])*100:+6.2f}%")
    print(f"{'Binary F1 Score':<30} | {base['f1']*100:6.2f}%         | {cal['f1']*100:6.2f}%         | {(cal['f1']-base['f1'])*100:+6.2f}%")
    print(f"{'Specificity':<30} | {base['specificity']*100:6.2f}%         | {cal['specificity']*100:6.2f}%         | {(cal['specificity']-base['specificity'])*100:+6.2f}%")
    print(f"{'False Alarm Rate (FPR)':<30} | {base['fpr']*100:6.2f}% ({base['fa_sessions']}/{base['total_adl']}) | {cal['fpr']*100:6.2f}% ({cal['fa_sessions']}/{cal['total_adl']}) | {(cal['fpr']-base['fpr'])*100:+6.2f}%")
    print(f"{'High-Motion False Alarms':<30} | {base['hm_fas']} sessions        | {cal['hm_fas']} sessions        | {cal['hm_fas']-base['hm_fas']:+d} sessions")
    print(f"\nBaseline False Alarms:   {base['fa_by_act']}")
    print(f"Calibrated False Alarms: {cal['fa_by_act']}")
    print(f"\nCalibrated Recall by Fall Type:")
    for ft, cnt in cal['det_by_type'].items():
        tot = cal['total_by_type'].get(ft, 0)
        pct = (cnt / tot * 100) if tot > 0 else 0.0
        print(f"  {ft:<20}: {cnt}/{tot} ({pct:5.1f}%)")

print_comparison("Phone", p_base, p_cal)
print_comparison("Watch", w_base, w_cal)

# Save comparison results
summary_records = [
    {"device": "phone", "mode": "Baseline", **{k: v for k, v in p_base.items() if not isinstance(v, dict)}},
    {"device": "phone", "mode": "Calibrated", **{k: v for k, v in p_cal.items() if not isinstance(v, dict)}},
    {"device": "watch", "mode": "Baseline", **{k: v for k, v in w_base.items() if not isinstance(v, dict)}},
    {"device": "watch", "mode": "Calibrated", **{k: v for k, v in w_cal.items() if not isinstance(v, dict)}},
]
pd.DataFrame(summary_records).to_csv(os.path.join(RESULTS_DIR, "test_baseline_vs_calibrated.csv"), index=False)
