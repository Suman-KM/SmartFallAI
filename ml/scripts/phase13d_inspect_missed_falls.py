import os, sys, json
import numpy as np
import pandas as pd
import onnxruntime as ort

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

# Load Phone ONNX
phone_onnx = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))

with open(os.path.join(PREPROC_DIR, "phone/scaler.json")) as f:
    ps = json.load(f)
p_med = np.array(ps["median"], dtype=np.float32)
p_iqr = np.array(ps["iqr"], dtype=np.float32)

split_dir = os.path.join(PREPROC_DIR, "phone/test")
X_scaled = np.load(os.path.join(split_dir, "X.npy"))
y_14 = np.load(os.path.join(split_dir, "y_14.npy"))
meta_df = pd.read_csv(os.path.join(split_dir, "metadata.csv"))

X_raw = (X_scaled * p_iqr) + p_med

logits = phone_onnx.run(None, {phone_onnx.get_inputs()[0].name: X_scaled.astype(np.float32)})[0]
e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
probs = e / np.sum(e, axis=1, keepdims=True)
fall_probs = np.sum(probs[:, :5], axis=1)

acc = X_raw[:, :, :3]
gyro = X_raw[:, :, 3:6]
acc_mag = np.linalg.norm(acc, axis=2)
gyro_mag = np.linalg.norm(gyro, axis=2)

dt = 0.02
jerk = np.abs(np.diff(acc_mag, axis=1)) / dt

meta_df['fall_prob'] = fall_probs
meta_df['acc_peak'] = np.max(acc_mag, axis=1)
meta_df['acc_min'] = np.min(acc_mag, axis=1)
meta_df['acc_range'] = meta_df['acc_peak'] - meta_df['acc_min']
meta_df['acc_std'] = np.std(acc_mag, axis=1)
meta_df['jerk_peak'] = np.max(jerk, axis=1)
meta_df['gyro_peak'] = np.max(gyro_mag, axis=1)
meta_df['y_14'] = y_14
meta_df['is_fall'] = y_14 < 5

# Run Phase 13C calibrated pipeline and identify which 2 sessions were not detected
ap_th = 18.0
ar_th = 10.0
gp_th = 2.5
p_th = 0.45
max_astd = 4.5
max_gp = 3.5
consec_req = 2
mem_len = 3

fall_meta = meta_df[meta_df['is_fall']]
fall_sessions = fall_meta['source_session_id'].unique()
print(f"Total Phone Test Fall Sessions: {len(fall_sessions)}")

missed_sessions = []
detected_sessions = []

for sid in fall_sessions:
    s_df = fall_meta[fall_meta['source_session_id'] == sid].sort_values('window_index')
    act = s_df['target_activity'].iloc[0]
    
    recent_mem = 0
    consec = 0
    triggered = False
    trigger_win = -1
    
    for _, row in s_df.iterrows():
        fp = row['fall_prob']
        ap = row['acc_peak']
        ar = row['acc_range']
        gp = row['gyro_peak']
        astd = row['acc_std']
        
        has_imp = (ap >= ap_th) or (ar >= ar_th and gp >= gp_th)
        if has_imp:
            recent_mem = mem_len
        elif recent_mem > 0:
            recent_mem -= 1
            
        is_thrash = (astd >= max_astd and gp >= max_gp)
        is_cand = (fp >= p_th) and (has_imp or recent_mem > 0) and (not is_thrash)
        
        if is_cand:
            consec += 1
            if consec >= consec_req:
                triggered = True
                trigger_win = int(row['window_index'])
                break
        else:
            consec = 0
            
    if triggered:
        detected_sessions.append(sid)
    else:
        missed_sessions.append((sid, act))

print(f"\nDetected: {len(detected_sessions)} sessions")
print(f"Missed:   {len(missed_sessions)} sessions")
for sid, act in missed_sessions:
    print(f"\n=======================================================")
    print(f"MISSED SESSION: {sid} ({act})")
    print(f"=======================================================")
    s_df = fall_meta[fall_meta['source_session_id'] == sid].sort_values('window_index')
    cols = ['window_index', 'fall_prob', 'acc_peak', 'acc_min', 'acc_range', 'acc_std', 'gyro_peak', 'jerk_peak']
    print(s_df[cols].to_string(index=False))
