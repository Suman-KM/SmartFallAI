import os
import json
import numpy as np
import pandas as pd

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

def contrastive_analysis(device):
    print(f"==================================================")
    print(f"CONTRASTIVE ANALYSIS FOR {device.upper()}")
    print(f"==================================================")
    
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
    acc_mag = np.linalg.norm(acc, axis=2) # (N, 100)
    gyro_mag = np.linalg.norm(gyro, axis=2) # (N, 100)
    
    # Pre-impact descent / drop: minimum acceleration in window
    acc_min = np.min(acc_mag, axis=1)
    acc_peak = np.max(acc_mag, axis=1)
    acc_range = acc_peak - acc_min
    acc_std = np.std(acc_mag, axis=1)
    
    # Jerk (da/dt with dt=0.02s)
    dt = 0.02
    acc_diff = np.diff(acc_mag, axis=1) / dt
    jerk_peak = np.max(np.abs(acc_diff), axis=1)
    
    # Gyro dynamics
    gyro_peak = np.max(gyro_mag, axis=1)
    gyro_std = np.std(gyro_mag, axis=1)
    
    # Post-impact stillness metrics:
    # Look at the last 30 samples (~0.6 sec) of the window
    tail_acc_std = np.std(acc_mag[:, 70:], axis=1)
    tail_gyro_std = np.std(gyro_mag[:, 70:], axis=1)
    tail_gyro_mean = np.mean(gyro_mag[:, 70:], axis=1)
    
    meta_df['y_14'] = y_14
    meta_df['is_fall'] = y_14 < 5
    meta_df['acc_min'] = acc_min
    meta_df['acc_peak'] = acc_peak
    meta_df['acc_range'] = acc_range
    meta_df['acc_std'] = acc_std
    meta_df['jerk_peak'] = jerk_peak
    meta_df['gyro_peak'] = gyro_peak
    meta_df['gyro_std'] = gyro_std
    meta_df['tail_acc_std'] = tail_acc_std
    meta_df['tail_gyro_std'] = tail_gyro_std
    meta_df['tail_gyro_mean'] = tail_gyro_mean
    
    activities = ['FALL', 'WALKING', 'RUNNING', 'JUMPING', 'SIT_DOWN', 'STAND_UP', 'PICKING_UP_OBJECT']
    
    results = []
    for act in activities:
        if act == 'FALL':
            sub = meta_df[meta_df['is_fall']]
        else:
            sub = meta_df[meta_df['target_activity'] == act]
            
        if len(sub) == 0:
            continue
            
        results.append({
            "Activity": act,
            "Windows": len(sub),
            "Acc_Peak_Med": f"{sub['acc_peak'].median():.2f}",
            "Acc_Min_Med": f"{sub['acc_min'].median():.2f}",
            "Acc_Min_P10": f"{sub['acc_min'].quantile(0.10):.2f}",
            "Acc_Range_Med": f"{sub['acc_range'].median():.2f}",
            "Jerk_Peak_Med": f"{sub['jerk_peak'].median():.2f}",
            "Gyro_Peak_Med": f"{sub['gyro_peak'].median():.2f}",
            "Acc_Std_Med": f"{sub['acc_std'].median():.2f}",
            "Tail_Acc_Std_Med": f"{sub['tail_acc_std'].median():.2f}",
            "Tail_Gyro_Std_Med": f"{sub['tail_gyro_std'].median():.2f}",
            "Tail_Gyro_Mean_Med": f"{sub['tail_gyro_mean'].median():.2f}",
        })
        
    df_out = pd.DataFrame(results)
    print(df_out.to_string(index=False))
    return df_out

phone_df = contrastive_analysis("phone")
print("\n")
watch_df = contrastive_analysis("watch")
