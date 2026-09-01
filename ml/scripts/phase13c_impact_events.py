import os
import json
import numpy as np
import pandas as pd

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

def analyze_impact_events(device):
    print(f"=== ANALYZING IMPACT WINDOWS FOR {device.upper()} ===")
    split_dir = os.path.join(PREPROC_DIR, device, "validation")
    X_scaled = np.load(os.path.join(split_dir, "X.npy"))
    y_14 = np.load(os.path.join(split_dir, "y_14.npy"))
    meta_df = pd.read_csv(os.path.join(split_dir, "metadata.csv"))
    
    with open(os.path.join(PREPROC_DIR, device, "scaler.json")) as f:
        scaler = json.load(f)
    med = np.array(scaler["median"], dtype=np.float32)
    iqr = np.array(scaler["iqr"], dtype=np.float32)
    X_raw = (X_scaled * iqr) + med
    
    acc = X_raw[:, :, :3]
    gyro = X_raw[:, :, 3:6]
    acc_mag = np.linalg.norm(acc, axis=2)
    gyro_mag = np.linalg.norm(gyro, axis=2)
    
    meta_df['acc_peak'] = np.max(acc_mag, axis=1)
    meta_df['acc_min'] = np.min(acc_mag, axis=1)
    meta_df['acc_range'] = meta_df['acc_peak'] - meta_df['acc_min']
    meta_df['gyro_peak'] = np.max(gyro_mag, axis=1)
    meta_df['tail_acc_std'] = np.std(acc_mag[:, 70:], axis=1)
    meta_df['tail_gyro_mean'] = np.mean(gyro_mag[:, 70:], axis=1)
    meta_df['y_14'] = y_14
    meta_df['is_fall'] = y_14 < 5
    
    # Isolate fall sessions
    fall_meta = meta_df[meta_df['is_fall']]
    fall_sessions = fall_meta['source_session_id'].unique()
    print(f"Total Fall Validation Sessions: {len(fall_sessions)}")
    
    session_records = []
    for sid in fall_sessions:
        s_df = fall_meta[fall_meta['source_session_id'] == sid].sort_values('window_index')
        # Find window with max acc_peak in session (the impact window)
        max_row = s_df.loc[s_df['acc_peak'].idxmax()]
        imp_idx = max_row['window_index']
        
        # Look at the window immediately after impact (imp_idx + 1) if it exists
        post_row = s_df[s_df['window_index'] == imp_idx + 1]
        post_acc_std = post_row['tail_acc_std'].values[0] if len(post_row) > 0 else np.nan
        post_gyro_mean = post_row['tail_gyro_mean'].values[0] if len(post_row) > 0 else np.nan
        post_acc_peak = post_row['acc_peak'].values[0] if len(post_row) > 0 else np.nan
        
        session_records.append({
            "session_id": sid,
            "activity": max_row['target_activity'],
            "impact_window": imp_idx,
            "imp_acc_peak": max_row['acc_peak'],
            "imp_acc_min": max_row['acc_min'],
            "imp_acc_range": max_row['acc_range'],
            "imp_gyro_peak": max_row['gyro_peak'],
            "post_acc_peak": post_acc_peak,
            "post_acc_std": post_acc_std,
            "post_gyro_mean": post_gyro_mean
        })
        
    df_sess = pd.DataFrame(session_records)
    print(df_sess[['activity', 'imp_acc_peak', 'imp_acc_min', 'imp_acc_range', 'imp_gyro_peak', 'post_acc_peak', 'post_gyro_mean']].to_string(index=False))
    print("\nSummary of Fall Impact Windows:")
    print(f"  Impact Acc Peak: Median = {df_sess['imp_acc_peak'].median():.2f}, Min = {df_sess['imp_acc_peak'].min():.2f}")
    print(f"  Impact Acc Min:  Median = {df_sess['imp_acc_min'].median():.2f}, Max = {df_sess['imp_acc_min'].max():.2f}")
    print(f"  Impact Acc Range: Median = {df_sess['imp_acc_range'].median():.2f}, Min = {df_sess['imp_acc_range'].min():.2f}")
    print(f"  Impact Gyro Peak: Median = {df_sess['imp_gyro_peak'].median():.2f}, Min = {df_sess['imp_gyro_peak'].min():.2f}")
    print(f"  Post-Impact Acc Peak: Median = {df_sess['post_acc_peak'].median():.2f}, P90 = {df_sess['post_acc_peak'].quantile(0.90):.2f}")
    print(f"  Post-Impact Gyro Mean: Median = {df_sess['post_gyro_mean'].median():.2f}, P90 = {df_sess['post_gyro_mean'].quantile(0.90):.2f}")
    return df_sess

p_sess = analyze_impact_events("phone")
print("\n" + "="*80 + "\n")
w_sess = analyze_impact_events("watch")
