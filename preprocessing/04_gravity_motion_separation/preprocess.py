import os
import csv
import json
import numpy as np
from scipy.signal import butter, filtfilt

WORKSPACE_DIR = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
RAW_WATCH = os.path.join(WORKSPACE_DIR, "raw_dataset/watch")
RAW_PHONE = os.path.join(WORKSPACE_DIR, "raw_dataset/phone")
SPLIT_MANIFEST = os.path.join(WORKSPACE_DIR, "preprocessing/common_split/all_split_sessions.csv")
OUT_DIR = os.path.join(WORKSPACE_DIR, "preprocessing/04_gravity_motion_separation")

WINDOW_SIZE = 100  # 2.0 seconds @ 50 Hz
STEP_SIZE = 50     # 50% overlap

# Gravity estimation filter parameters (0.5 Hz low-pass)
FS = 50.0
GRAVITY_CUTOFF = 0.5
ORDER = 2

FEATURE_NAMES = ["dyn_accX", "dyn_accY", "dyn_accZ", "gyroX", "gyroY", "gyroZ", "pitch", "roll", "yaw"]

CLASS_MAPPING_14 = {
    "FALL_BACKWARD": 0, "FALL_FORWARD": 1, "FALL_FROM_SITTING": 2,
    "FALL_LEFT": 3, "FALL_RIGHT": 4, "JUMPING": 5, "LYING_DOWN": 6,
    "PICKING_UP_OBJECT": 7, "RUNNING": 8, "SITTING": 9, "SIT_DOWN": 10,
    "STANDING": 11, "STAND_UP": 12, "WALKING": 13
}

FALL_CLASSES = {"FALL_FORWARD", "FALL_BACKWARD", "FALL_LEFT", "FALL_RIGHT", "FALL_FROM_SITTING"}

# Design Butterworth low-pass filter for gravity estimation
nyquist = 0.5 * FS
normal_cutoff = GRAVITY_CUTOFF / nyquist
b_g, a_g = butter(ORDER, normal_cutoff, btype='low', analog=False)

def separate_gravity_dynamic(feats):
    # feats shape: (N, 9)
    # feats[:, 0:3] -> accX, accY, accZ
    if len(feats) < 18:
        return feats
    try:
        acc = feats[:, 0:3]
        # Low-pass filter to extract gravity vector
        gravity = filtfilt(b_g, a_g, acc, axis=0)
        dynamic_acc = acc - gravity
        
        # Construct new feature matrix
        out_feats = np.zeros_like(feats)
        out_feats[:, 0:3] = dynamic_acc
        out_feats[:, 3:9] = feats[:, 3:9]
        return out_feats.astype(np.float32)
    except Exception:
        return feats

def load_session_raw(file_path, is_watch):
    expected_cols = 20 if is_watch else 17
    rows = []
    timestamps = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None, None
            
        for r in reader:
            if not r or len(r) != expected_cols:
                continue
            try:
                ts = int(r[1])
                feats = [float(r[i]) for i in range(2, 11)]
                rows.append(feats)
                timestamps.append(ts)
            except Exception:
                continue
                
    if not rows:
        return None, None
    raw_feats = np.array(rows, dtype=np.float32)
    # Apply Gravity-Dynamic separation strictly within this session
    processed_feats = separate_gravity_dynamic(raw_feats)
    return processed_feats, np.array(timestamps, dtype=np.int64)

def extract_windows(feats, timestamps, sid, fname, device, act):
    windows = []
    meta = []
    n = len(feats)
    if n < WINDOW_SIZE:
        return windows, meta
        
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        w = feats[start:end]
        ts_start = int(timestamps[start])
        ts_end = int(timestamps[end - 1])
        
        y_14 = CLASS_MAPPING_14[act]
        y_bin = 1 if act in FALL_CLASSES else 0
        
        windows.append(w)
        meta.append({
            "source_session_id": sid,
            "source_file": fname,
            "source_device": device,
            "target_activity": act,
            "y_14": y_14,
            "fall_binary": "FALL" if y_bin == 1 else "NORMAL",
            "y_binary": y_bin,
            "start_timestamp": ts_start,
            "end_timestamp": ts_end,
            "window_index": len(windows) - 1
        })
        
    return windows, meta

def process_device(device_name, raw_dir, is_watch, split_dict):
    dev_out = os.path.join(OUT_DIR, device_name.lower())
    os.makedirs(dev_out, exist_ok=True)
    
    split_windows = {"TRAIN": [], "VALIDATION": [], "TEST": []}
    split_meta = {"TRAIN": [], "VALIDATION": [], "TEST": []}
    
    sessions_for_dev = [s for s in split_dict.values() if s["device"] == device_name]
    
    for sinfo in sessions_for_dev:
        fpath = os.path.join(raw_dir, sinfo["filename"])
        feats, ts = load_session_raw(fpath, is_watch)
        if feats is None:
            continue
            
        w_list, m_list = extract_windows(feats, ts, sinfo["session_id"], sinfo["filename"], device_name, sinfo["activity"])
        split = sinfo["split"]
        
        for w, m in zip(w_list, m_list):
            m["split"] = split
            split_windows[split].append(w)
            split_meta[split].append(m)
            
    X_train = np.array(split_windows["TRAIN"], dtype=np.float32)
    X_val = np.array(split_windows["VALIDATION"], dtype=np.float32)
    X_test = np.array(split_windows["TEST"], dtype=np.float32)
    
    train_2d = X_train.reshape(-1, X_train.shape[-1])
    mean = np.mean(train_2d, axis=0)
    std = np.std(train_2d, axis=0)
    std[std == 0.0] = 1.0
    
    def apply_standardization(X):
        return (X - mean) / std
        
    X_train_norm = apply_standardization(X_train)
    X_val_norm = apply_standardization(X_val)
    X_test_norm = apply_standardization(X_test)
    
    scaler_info = {
        "method": "Gravity_Separation_StandardScaler",
        "gravity_filter": {
            "type": "Butterworth Low-pass",
            "order": ORDER,
            "cutoff_hz": GRAVITY_CUTOFF,
            "fs_hz": FS,
            "equation": "dynamic_acc = acc - gravity"
        },
        "fitted_on": "TRAIN",
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    with open(os.path.join(dev_out, "scaler.json"), "w") as f:
        json.dump(scaler_info, f, indent=2)
        
    for split_name, X_data, m_data in [("train", X_train_norm, split_meta["TRAIN"]),
                                       ("validation", X_val_norm, split_meta["VALIDATION"]),
                                       ("test", X_test_norm, split_meta["TEST"])]:
        s_dir = os.path.join(dev_out, split_name)
        os.makedirs(s_dir, exist_ok=True)
        
        y_14 = np.array([m["y_14"] for m in m_data], dtype=np.int64)
        y_bin = np.array([m["y_binary"] for m in m_data], dtype=np.int64)
        
        np.save(os.path.join(s_dir, "X.npy"), X_data)
        np.save(os.path.join(s_dir, "y_14.npy"), y_14)
        np.save(os.path.join(s_dir, "y_binary.npy"), y_bin)
        
        if m_data:
            m_fields = list(m_data[0].keys())
            with open(os.path.join(s_dir, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=m_fields)
                writer.writeheader()
                for row in m_data:
                    writer.writerow(row)
                    
    print(f"[{device_name}] Pipeline 04 Complete:")
    print(f"  Train: X={X_train_norm.shape}, y_14={len(split_meta['TRAIN'])}")
    print(f"  Val:   X={X_val_norm.shape}, y_14={len(split_meta['VALIDATION'])}")
    print(f"  Test:  X={X_test_norm.shape}, y_14={len(split_meta['TEST'])}")
    
    return {
        "train_shape": list(X_train_norm.shape),
        "val_shape": list(X_val_norm.shape),
        "test_shape": list(X_test_norm.shape),
        "total_windows": len(split_meta["TRAIN"]) + len(split_meta["VALIDATION"]) + len(split_meta["TEST"])
    }

def main():
    split_dict = {}
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            split_dict[r["filename"]] = r
            
    watch_summary = process_device("WATCH", RAW_WATCH, True, split_dict)
    phone_summary = process_device("PHONE", RAW_PHONE, False, split_dict)
    
    config = {
        "pipeline_id": "04_gravity_motion_separation",
        "description": "Gravity-motion separation: 0.5 Hz low-pass gravity extraction isolating dynamic acceleration (dyn_acc = acc - gravity) with Train-only standardization",
        "window_size_samples": WINDOW_SIZE,
        "window_duration_seconds": 2.0,
        "step_size_samples": STEP_SIZE,
        "overlap_percentage": 50.0,
        "gravity_filter": {
            "type": "Butterworth Low-pass",
            "order": ORDER,
            "cutoff_hz": GRAVITY_CUTOFF,
            "fs_hz": FS
        },
        "feature_names": FEATURE_NAMES,
        "feature_count": 9,
        "normalization": "StandardScaler",
        "fit_split": "TRAIN",
        "watch_summary": watch_summary,
        "phone_summary": phone_summary
    }
    
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    report_md = f"""# Pipeline 04: Gravity-Motion Separation Report

## Pipeline Overview
- **Identifier**: `04_gravity_motion_separation`
- **Methodology**: 0.5 Hz 2nd-order Butterworth low-pass filter to isolate the static gravity vector $g$.
- **Dynamic Acceleration**: $a_{{dyn}} = a_{{raw}} - g$.
- **Session Isolation**: Gravity separation is performed per-session before window generation.
- **Normalization**: Z-score Standardization $(x - \\mu_{{train}}) / \\sigma_{{train}}$ computed strictly on TRAIN partitions only.
- **Features (9)**: `dyn_accX, dyn_accY, dyn_accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw`
- **GPS & Metadata Policy**: Completely excluded from input tensors $X$.

## Output Dataset Statistics

### WATCH
- **Train Tensor**: `{watch_summary['train_shape']}`
- **Validation Tensor**: `{watch_summary['val_shape']}`
- **Test Tensor**: `{watch_summary['test_shape']}`
- **Total Windows**: `{watch_summary['total_windows']}`

### PHONE
- **Train Tensor**: `{phone_summary['train_shape']}`
- **Validation Tensor**: `{phone_summary['val_shape']}`
- **Test Tensor**: `{phone_summary['test_shape']}`
- **Total Windows**: `{phone_summary['total_windows']}`
"""
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report_md)
        
    print("Pipeline 04 finished successfully.")

if __name__ == "__main__":
    main()
