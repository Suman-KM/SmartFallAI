import os
import csv
import numpy as np

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

for dev in ["phone", "watch"]:
    manifest = os.path.join(WORKSPACE, "preprocessing/common_split/all_split_sessions.csv")
    with open(manifest) as f:
        reader = csv.DictReader(f)
        falls = [r for r in reader if r["device"] == dev.upper() and "FALL" in r["activity"]]
        
    print(f"\n=======================================================")
    print(f"IMPACT CHARACTERISTICS OF ALL {len(falls)} {dev.upper()} FALL SESSIONS")
    print(f"=======================================================")
    
    peak_accs = []
    ranges_acc = []
    peak_gyros = []
    
    for r in falls:
        fpath = os.path.join(WORKSPACE, f"raw_dataset/{dev}", r["filename"])
        with open(fpath) as fp:
            cr = csv.reader(fp)
            header = next(cr)
            rows = [[float(x) for x in line[2:8]] for line in cr if len(line) >= 8]
        if not rows:
            continue
        arr = np.array(rows)
        acc_mag = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
        gyro_mag = np.sqrt(arr[:, 3]**2 + arr[:, 4]**2 + arr[:, 5]**2)
        
        # Max acc in the session
        peak_accs.append(np.max(acc_mag))
        # Max range in any 2s window (100 samples)
        max_w_range = 0.0
        max_w_gyro = 0.0
        for i in range(0, len(arr) - 100, 50):
            w_acc = acc_mag[i:i+100]
            w_gyro = gyro_mag[i:i+100]
            rng = np.max(w_acc) - np.min(w_acc)
            if rng > max_w_range:
                max_w_range = rng
            g_max = np.max(w_gyro)
            if g_max > max_w_gyro:
                max_w_gyro = g_max
        ranges_acc.append(max_w_range)
        peak_gyros.append(max_w_gyro)
        
    peak_accs = np.array(peak_accs)
    ranges_acc = np.array(ranges_acc)
    peak_gyros = np.array(peak_gyros)
    
    print(f"Session Peak Acc Mag   : Min={np.min(peak_accs):.2f}, 10th%={np.percentile(peak_accs, 10):.2f}, Median={np.median(peak_accs):.2f}, Max={np.max(peak_accs):.2f}")
    print(f"Session Max Acc Range  : Min={np.min(ranges_acc):.2f}, 10th%={np.percentile(ranges_acc, 10):.2f}, Median={np.median(ranges_acc):.2f}, Max={np.max(ranges_acc):.2f}")
    print(f"Session Peak Gyro Mag  : Min={np.min(peak_gyros):.2f}, 10th%={np.percentile(peak_gyros, 10):.2f}, Median={np.median(peak_gyros):.2f}, Max={np.max(peak_gyros):.2f}")
