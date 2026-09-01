import os
import json
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

def load_data(device, split="validation"):
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

def run_grid_search(device):
    print(f"=== RUNNING VALIDATION GRID SEARCH FOR {device.upper()} ===")
    df = load_data(device, "validation")
    
    # Candidate parameter ranges
    p_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    acc_peak_thresholds = [16.0, 18.0, 20.0, 22.0, 24.0, 26.0] if device == "phone" else [18.0, 20.0, 22.0, 25.0, 28.0, 30.0]
    acc_min_thresholds = [None, 7.5, 8.0, 8.5] # pre-impact weightlessness / drop
    consec_options = [1, 2, 3]
    memory_options = [2, 3, 4]
    
    total_fall_sessions = df[df['is_fall']]['source_session_id'].nunique()
    total_adl_sessions = df[~df['is_fall']]['source_session_id'].nunique()
    
    grid_results = []
    
    # Efficient search
    for ap_th in acc_peak_thresholds:
        for p_th in p_thresholds:
            for amin_th in [None, 8.0]:
                for consec in [1, 2]:
                    for mem in [2, 3]:
                        # Evaluate across all sessions
                        det_falls = 0
                        det_falls_by_type = {'FALL_BACKWARD':0, 'FALL_FORWARD':0, 'FALL_FROM_SITTING':0, 'FALL_LEFT':0, 'FALL_RIGHT':0}
                        fa_sessions = 0
                        fa_by_act = {}
                        
                        for sid, s_df in df.groupby('source_session_id'):
                            s_df = s_df.sort_values('window_index')
                            is_fall = s_df['is_fall'].iloc[0]
                            act = s_df['target_activity'].iloc[0]
                            
                            consec_count = 0
                            impact_mem = 0
                            triggered = False
                            
                            for _, row in s_df.iterrows():
                                fp = row['fall_prob']
                                ap = row['acc_peak']
                                ar = row['acc_range']
                                gp = row['gyro_peak']
                                amin = row['acc_min']
                                
                                # Impact condition: acceleration peak shock
                                has_impact = (ap >= ap_th) or (ar >= (ap_th * 0.6) and gp >= 3.0)
                                if amin_th is not None and has_impact:
                                    has_impact = has_impact and (amin <= amin_th)
                                    
                                if has_impact:
                                    impact_mem = mem
                                elif impact_mem > 0:
                                    impact_mem -= 1
                                    
                                is_cand = (fp >= p_th) and (has_impact or impact_mem > 0)
                                
                                if is_cand:
                                    consec_count += 1
                                    if consec_count >= consec:
                                        triggered = True
                                        break
                                else:
                                    consec_count = 0
                                    
                            if is_fall:
                                if triggered:
                                    det_falls += 1
                                    if act in det_falls_by_type:
                                        det_falls_by_type[act] += 1
                            else:
                                if triggered:
                                    fa_sessions += 1
                                    fa_by_act[act] = fa_by_act.get(act, 0) + 1
                                    
                        rec = det_falls / total_fall_sessions
                        prec = det_falls / (det_falls + fa_sessions) if (det_falls + fa_sessions) > 0 else 0.0
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                        fpr = fa_sessions / total_adl_sessions
                        
                        # High motion FA sessions: WALKING, RUNNING, JUMPING
                        high_motion_fas = fa_by_act.get('WALKING', 0) + fa_by_act.get('RUNNING', 0) + fa_by_act.get('JUMPING', 0)
                        
                        grid_results.append({
                            "device": device,
                            "acc_peak_th": ap_th,
                            "p_th": p_th,
                            "acc_min_th": amin_th if amin_th is not None else "None",
                            "consec": consec,
                            "mem": mem,
                            "recall": rec,
                            "precision": prec,
                            "f1": f1,
                            "fpr": fpr,
                            "fa_sessions": fa_sessions,
                            "high_motion_fas": high_motion_fas,
                            "fa_details": fa_by_act,
                            "falls_by_type": det_falls_by_type
                        })
                        
    res_df = pd.DataFrame(grid_results)
    # Sort by F1 descending, then recall descending, then fa_sessions ascending
    res_df = res_df.sort_values(by=["f1", "recall", "high_motion_fas"], ascending=[False, False, True])
    print(f"Top 10 configurations for {device}:")
    top_cols = ["acc_peak_th", "p_th", "acc_min_th", "consec", "mem", "recall", "precision", "f1", "fpr", "fa_sessions", "high_motion_fas"]
    print(res_df[top_cols].head(10).to_string(index=False))
    
    res_df.to_csv(os.path.join(RESULTS_DIR, f"{device}_grid_search_results.csv"), index=False)
    return res_df

p_grid = run_grid_search("phone")
print("\n" + "="*80 + "\n")
w_grid = run_grid_search("watch")
