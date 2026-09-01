import os
import json
import numpy as np
import pandas as pd

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

# Load Phone and Watch validation datasets
def analyze_temporal_structure(device):
    print(f"Analyzing temporal structure for {device}...")
    split_dir = os.path.join(PREPROC_DIR, device, "validation")
    X_scaled = np.load(os.path.join(split_dir, "X.npy"))
    y_14 = np.load(os.path.join(split_dir, "y_14.npy"))
    meta_df = pd.read_csv(os.path.join(split_dir, "metadata.csv"))
    
    with open(os.path.join(PREPROC_DIR, device, "scaler.json")) as f:
        scaler = json.load(f)
    med = np.array(scaler["median"], dtype=np.float32)
    iqr = np.array(scaler["iqr"], dtype=np.float32)
    X_raw = (X_scaled * iqr) + med
    
    # Calculate window kinematics
    acc = X_raw[:, :, :3]
    gyro = X_raw[:, :, 3:6]
    acc_mag = np.linalg.norm(acc, axis=2) # (N, 100)
    gyro_mag = np.linalg.norm(gyro, axis=2) # (N, 100)
    
    meta_df['acc_peak'] = np.max(acc_mag, axis=1)
    meta_df['acc_min'] = np.min(acc_mag, axis=1)
    meta_df['acc_range'] = meta_df['acc_peak'] - meta_df['acc_min']
    meta_df['acc_std'] = np.std(acc_mag, axis=1)
    meta_df['gyro_peak'] = np.max(gyro_mag, axis=1)
    meta_df['gyro_std'] = np.std(gyro_mag, axis=1)
    meta_df['y_14'] = y_14
    meta_df['is_fall'] = y_14 < 5
    
    # In each window: split into First Half (samples 0..49, ~1 sec) and Second Half (samples 50..99, ~1 sec)
    acc_mag_h1 = acc_mag[:, :50]
    acc_mag_h2 = acc_mag[:, 50:]
    gyro_mag_h1 = gyro_mag[:, :50]
    gyro_mag_h2 = gyro_mag[:, 50:]
    
    meta_df['acc_peak_h1'] = np.max(acc_mag_h1, axis=1)
    meta_df['acc_std_h1'] = np.std(acc_mag_h1, axis=1)
    meta_df['acc_peak_h2'] = np.max(acc_mag_h2, axis=1)
    meta_df['acc_std_h2'] = np.std(acc_mag_h2, axis=1)
    meta_df['gyro_std_h2'] = np.std(gyro_mag_h2, axis=1)
    
    # Let's inspect sessions: For each session, compute the sequence of windows
    grouped = meta_df.groupby('source_session_id')
    print(f"Total sessions: {len(grouped)}")
    
    # Analyze by activity class
    stats = []
    for cls_idx in range(14):
        sub = meta_df[meta_df['y_14'] == cls_idx]
        if len(sub) == 0:
            continue
        cls_name = sub['target_activity'].iloc[0] if 'target_activity' in sub.columns else str(cls_idx)
        stats.append({
            "activity": cls_name,
            "windows": len(sub),
            "acc_peak_med": sub['acc_peak'].median(),
            "acc_std_med": sub['acc_std'].median(),
            "gyro_peak_med": sub['gyro_peak'].median(),
            "gyro_std_med": sub['gyro_std'].median(),
            # Post-half metrics (samples 50..99)
            "h2_acc_std_med": sub['acc_std_h2'].median(),
            "h2_gyro_std_med": sub['gyro_std_h2'].median(),
        })
    df_res = pd.DataFrame(stats)
    print(df_res.to_string(index=False))
    return meta_df

p_meta = analyze_temporal_structure("phone")
print("\n" + "="*80 + "\n")
w_meta = analyze_temporal_structure("watch")
