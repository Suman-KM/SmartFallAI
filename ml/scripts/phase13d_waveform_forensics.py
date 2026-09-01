import os
import json
import numpy as np
import pandas as pd

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

# Load scalers
with open(os.path.join(PREPROC_DIR, "phone/scaler.json")) as f:
    ps = json.load(f)
p_med, p_iqr = np.array(ps["median"], dtype=np.float32), np.array(ps["iqr"], dtype=np.float32)

with open(os.path.join(PREPROC_DIR, "watch/scaler.json")) as f:
    ws = json.load(f)
w_med, w_iqr = np.array(ws["median"], dtype=np.float32), np.array(ws["iqr"], dtype=np.float32)

def analyze_sample_level_dynamics(device, split="validation"):
    print(f"==================================================")
    print(f"SAMPLE-LEVEL WAVEFORM ANALYSIS: {device.upper()} ({split.upper()})")
    print(f"==================================================")
    
    split_dir = os.path.join(PREPROC_DIR, device, split)
    X_scaled = np.load(os.path.join(split_dir, "X.npy"))
    y_14 = np.load(os.path.join(split_dir, "y_14.npy"))
    meta_df = pd.read_csv(os.path.join(split_dir, "metadata.csv"))
    
    med = p_med if device == "phone" else w_med
    iqr = p_iqr if device == "phone" else w_iqr
    X_raw = (X_scaled * iqr) + med
    
    # acc shape: (N, 100, 3)
    acc = X_raw[:, :, :3]
    gyro = X_raw[:, :, 3:6]
    
    acc_mag = np.linalg.norm(acc, axis=2) # (N, 100)
    gyro_mag = np.linalg.norm(gyro, axis=2) # (N, 100)
    
    dt = 0.02 # 50 Hz
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
    
    # Analyze timing inside each window:
    # Index of peak acceleration
    idx_peak = np.argmax(acc_mag, axis=1)
    idx_min = np.argmin(acc_mag, axis=1)
    
    # Pre-impact drop check: minimum acceleration within 20 samples (0.4s) BEFORE peak
    # Stillness check: standard deviation of acceleration in the 30 samples AFTER peak
    has_pre_drop = []
    pre_drop_val = []
    post_impact_std = []
    post_impact_gyro = []
    
    for i in range(len(acc_mag)):
        p_idx = idx_peak[i]
        # Look at window before peak: max(0, p_idx - 25) to p_idx
        start_pre = max(0, p_idx - 25)
        if p_idx > start_pre:
            min_pre = np.min(acc_mag[i, start_pre:p_idx])
            has_drop = (min_pre <= 6.5) and (acc_mag[i, p_idx] >= 18.0)
        else:
            min_pre = acc_mag[i, 0]
            has_drop = False
            
        has_pre_drop.append(has_drop)
        pre_drop_val.append(min_pre)
        
        # Look at tail after peak: p_idx + 10 to 100
        start_post = min(99, p_idx + 10)
        if start_post < 95:
            post_std = np.std(acc_mag[i, start_post:])
            post_g = np.mean(gyro_mag[i, start_post:])
        else:
            post_std = np.nan
            post_g = np.nan
            
        post_impact_std.append(post_std)
        post_impact_gyro.append(post_g)
        
    meta_df['has_pre_drop'] = has_pre_drop
    meta_df['pre_drop_val'] = pre_drop_val
    meta_df['post_impact_std'] = post_impact_std
    meta_df['post_impact_gyro'] = post_impact_gyro
    
    # Look at sessions:
    # For fall sessions, find the actual impact window (max acc_peak in session)
    print("\n--- Fall Sessions Impact Signature ---")
    fall_sessions = meta_df[meta_df['is_fall']]['source_session_id'].unique()
    fall_impact_stats = []
    for sid in fall_sessions:
        s_df = meta_df[meta_df['source_session_id'] == sid]
        imp_row = s_df.loc[s_df['acc_peak'].idxmax()]
        imp_idx = imp_row['window_index']
        # Also check next window (settling)
        next_rows = s_df[s_df['window_index'] == imp_idx + 1]
        next_std = next_rows['acc_std'].values[0] if len(next_rows) > 0 else np.nan
        next_gp = next_rows['gyro_peak'].values[0] if len(next_rows) > 0 else np.nan
        
        fall_impact_stats.append({
            "session": sid,
            "activity": imp_row['target_activity'],
            "imp_peak": imp_row['acc_peak'],
            "imp_min": imp_row['acc_min'],
            "imp_range": imp_row['acc_range'],
            "imp_jerk": imp_row['jerk_peak'],
            "imp_gyro": imp_row['gyro_peak'],
            "has_pre_drop": imp_row['has_pre_drop'],
            "pre_drop_val": imp_row['pre_drop_val'],
            "next_window_acc_std": next_std,
            "next_window_gyro_peak": next_gp
        })
    df_fall_imp = pd.DataFrame(fall_impact_stats)
    print(df_fall_imp[['activity', 'imp_peak', 'imp_min', 'imp_jerk', 'pre_drop_val', 'next_window_acc_std', 'next_window_gyro_peak']].to_string(index=False))
    
    print("\n--- High Motion ADL Comparison (Max across session) ---")
    adl_stats = []
    adl_sessions = meta_df[~meta_df['is_fall']]['source_session_id'].unique()
    for sid in adl_sessions:
        s_df = meta_df[meta_df['source_session_id'] == sid]
        max_row = s_df.loc[s_df['acc_peak'].idxmax()]
        act = max_row['target_activity']
        if act in ['WALKING', 'RUNNING', 'JUMPING', 'SIT_DOWN', 'STAND_UP', 'PICKING_UP_OBJECT']:
            adl_stats.append({
                "session": sid,
                "activity": act,
                "max_peak": s_df['acc_peak'].max(),
                "min_acc": s_df['acc_min'].min(),
                "max_jerk": s_df['jerk_peak'].max(),
                "mean_acc_std": s_df['acc_std'].mean(),
                "mean_gyro_peak": s_df['gyro_peak'].mean()
            })
    df_adl = pd.DataFrame(adl_stats)
    print(df_adl.groupby('activity')[['max_peak', 'min_acc', 'max_jerk', 'mean_acc_std', 'mean_gyro_peak']].median())
    
    return meta_df

phone_dyn = analyze_sample_level_dynamics("phone")
print("\n" + "="*80 + "\n")
watch_dyn = analyze_sample_level_dynamics("watch")
