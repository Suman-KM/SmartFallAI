import numpy as np
import json

# Analyze Phone and Watch train/val/test datasets
for dev in ["phone", "watch"]:
    print(f"\n=======================================================")
    print(f"ANALYSIS OF KINEMATIC IMPACT CHARACTERISTICS: {dev.upper()}")
    print(f"=======================================================")
    
    with open(f"preprocessing/02_robust_scaling/{dev}/scaler.json") as f:
        scaler = json.load(f)
    med = np.array(scaler["median"])
    iqr = np.array(scaler["iqr"])
    
    # Load Validation set (never touch test set for tuning!)
    X_val = np.load(f"preprocessing/02_robust_scaling/{dev}/validation/X.npy")
    y_val = np.load(f"preprocessing/02_robust_scaling/{dev}/validation/y_binary.npy")
    y_14 = np.load(f"preprocessing/02_robust_scaling/{dev}/validation/y_14.npy")
    
    # Unscale X_val to physical units: acc in m/s^2, gyro in rad/s
    X_phys = X_val * iqr + med
    
    # Compute per-window kinematics:
    # acc: cols 0, 1, 2. gyro: cols 3, 4, 5.
    acc_mag = np.sqrt(X_phys[:, :, 0]**2 + X_phys[:, :, 1]**2 + X_phys[:, :, 2]**2)
    gyro_mag = np.sqrt(X_phys[:, :, 3]**2 + X_phys[:, :, 4]**2 + X_phys[:, :, 5]**2)
    
    acc_peak = np.max(acc_mag, axis=1) # max acc magnitude in 2s window
    acc_min = np.min(acc_mag, axis=1)
    acc_range = acc_peak - acc_min
    acc_std = np.std(acc_mag, axis=1)
    gyro_peak = np.max(gyro_mag, axis=1)
    
    fall_mask = (y_val == 1)
    norm_mask = (y_val == 0)
    
    print(f"Total Validation Windows: {len(y_val)} (Falls: {np.sum(fall_mask)}, Normals: {np.sum(norm_mask)})")
    
    print("\n--- Fall Windows Kinematics ---")
    print(f"Acc Peak  : Median={np.median(acc_peak[fall_mask]):.2f} m/s^2, 10th%={np.percentile(acc_peak[fall_mask], 10):.2f}, 90th%={np.percentile(acc_peak[fall_mask], 90):.2f}")
    print(f"Acc Range : Median={np.median(acc_range[fall_mask]):.2f} m/s^2, 10th%={np.percentile(acc_range[fall_mask], 10):.2f}, 90th%={np.percentile(acc_range[fall_mask], 90):.2f}")
    print(f"Gyro Peak : Median={np.median(gyro_peak[fall_mask]):.2f} rad/s, 10th%={np.percentile(gyro_peak[fall_mask], 10):.2f}, 90th%={np.percentile(gyro_peak[fall_mask], 90):.2f}")
    
    print("\n--- Normal Windows Kinematics ---")
    print(f"Acc Peak  : Median={np.median(acc_peak[norm_mask]):.2f} m/s^2, 10th%={np.percentile(acc_peak[norm_mask], 10):.2f}, 90th%={np.percentile(acc_peak[norm_mask], 90):.2f}")
    print(f"Acc Range : Median={np.median(acc_range[norm_mask]):.2f} m/s^2, 10th%={np.percentile(acc_range[norm_mask], 10):.2f}, 90th%={np.percentile(acc_range[norm_mask], 90):.2f}")
    print(f"Gyro Peak : Median={np.median(gyro_peak[norm_mask]):.2f} rad/s, 10th%={np.percentile(gyro_peak[norm_mask], 10):.2f}, 90th%={np.percentile(gyro_peak[norm_mask], 90):.2f}")
    
    # Specific Static Activities: SITTING (9), STANDING (11), LYING_DOWN (6)
    static_mask = np.isin(y_14, [6, 9, 11])
    print(f"\n--- Static Normal Activities (Sitting, Standing, Lying) [N={np.sum(static_mask)}] ---")
    print(f"Acc Peak  : Max={np.max(acc_peak[static_mask]):.2f} m/s^2, 99th%={np.percentile(acc_peak[static_mask], 99):.2f}")
    print(f"Acc Range : Max={np.max(acc_range[static_mask]):.2f} m/s^2, 99th%={np.percentile(acc_range[static_mask], 99):.2f}")
    print(f"Gyro Peak : Max={np.max(gyro_peak[static_mask]):.2f} rad/s, 99th%={np.percentile(gyro_peak[static_mask], 99):.2f}")
