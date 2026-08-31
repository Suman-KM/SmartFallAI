import os
import csv
import json
import random
import glob
from collections import defaultdict

WORKSPACE_DIR = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
RAW_WATCH = os.path.join(WORKSPACE_DIR, "raw_dataset/watch")
RAW_PHONE = os.path.join(WORKSPACE_DIR, "raw_dataset/phone")

COMMON_SPLIT_DIR = os.path.join(WORKSPACE_DIR, "preprocessing/common_split")
os.makedirs(COMMON_SPLIT_DIR, exist_ok=True)

# 14 Valid ML classes
VALID_CLASSES = [
    # Normal / ADL (9)
    "STANDING", "SITTING", "WALKING", "RUNNING", "LYING_DOWN",
    "JUMPING", "SIT_DOWN", "STAND_UP", "PICKING_UP_OBJECT",
    # Fall (5)
    "FALL_FORWARD", "FALL_BACKWARD", "FALL_LEFT", "FALL_RIGHT", "FALL_FROM_SITTING"
]

EXCLUDED_CLASSES = [
    "SUDDEN_SIT", "FALL_FORWARD_HANDS", "FALL_FORWARD_KNEES",
    "GOING_UPSTAIRS", "GOING_DOWNSTAIRS"
]

FALL_CLASSES = set(["FALL_FORWARD", "FALL_BACKWARD", "FALL_LEFT", "FALL_RIGHT", "FALL_FROM_SITTING"])

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def scan_and_audit(device_dir, is_watch):
    files = sorted(glob.glob(os.path.join(device_dir, "*.csv")))
    sessions = []
    excluded_files = []
    expected_cols = 20 if is_watch else 17
    
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            
            row_count = 0
            act = None
            sid = None
            first_ts = None
            last_ts = None
            
            for r in reader:
                if not r or len(r) != expected_cols:
                    continue
                row_count += 1
                if sid is None:
                    sid = r[0]
                    first_ts = int(r[1])
                last_ts = int(r[1])
                
                if is_watch:
                    act = r[19]
                else:
                    act = r[16]
                    
            if row_count == 0 or act is None:
                continue
                
            dur = (last_ts - first_ts) / 1000.0 if (first_ts and last_ts and last_ts >= first_ts) else 0.0
            
            if act in EXCLUDED_CLASSES or act not in VALID_CLASSES:
                excluded_files.append({
                    "filename": fname,
                    "session_id": sid,
                    "activity": act,
                    "rows": row_count,
                    "duration": dur,
                    "reason": f"Activity {act} is excluded from ML taxonomy"
                })
            else:
                sessions.append({
                    "device": "WATCH" if is_watch else "PHONE",
                    "filename": fname,
                    "session_id": sid,
                    "activity": act,
                    "fall_binary": "FALL" if act in FALL_CLASSES else "NORMAL",
                    "rows": row_count,
                    "duration": dur,
                    "first_timestamp": first_ts,
                    "last_timestamp": last_ts
                })
                
    return sessions, excluded_files

watch_sessions, watch_excl = scan_and_audit(RAW_WATCH, True)
phone_sessions, phone_excl = scan_and_audit(RAW_PHONE, False)

print(f"Watch: {len(watch_sessions)} valid sessions, {len(watch_excl)} excluded sessions")
print(f"Phone: {len(phone_sessions)} valid sessions, {len(phone_excl)} excluded sessions")

# Build unified unique session IDs mapping
all_sessions = watch_sessions + phone_sessions
unique_sessions = {}
for s in all_sessions:
    sid = s["session_id"]
    if sid not in unique_sessions:
        unique_sessions[sid] = s["activity"]

# Group unique session IDs by activity
by_act = defaultdict(list)
for sid, act in unique_sessions.items():
    by_act[act].append(sid)

train_sids = set()
val_sids = set()
test_sids = set()

train_ratio = 0.70
val_ratio = 0.15

for act in sorted(by_act.keys()):
    sids = by_act[act]
    sids.sort()
    rng = random.Random(RANDOM_SEED + hash(act) % 10000)
    rng.shuffle(sids)
    
    n = len(sids)
    if n == 1:
        train_sids.add(sids[0])
    elif n == 2:
        train_sids.add(sids[0])
        test_sids.add(sids[1])
    elif n == 3:
        train_sids.add(sids[0])
        val_sids.add(sids[1])
        test_sids.add(sids[2])
    else:
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
        for sid in sids[:n_train]:
            train_sids.add(sid)
        for sid in sids[n_train:n_train + n_val]:
            val_sids.add(sid)
        for sid in sids[n_train + n_val:]:
            test_sids.add(sid)

# Assert strict zero session ID overlap
assert len(train_sids.intersection(val_sids)) == 0, "Train & Val session overlap!"
assert len(train_sids.intersection(test_sids)) == 0, "Train & Test session overlap!"
assert len(val_sids.intersection(test_sids)) == 0, "Val & Test session overlap!"

# Assign split to all session records
for s in all_sessions:
    sid = s["session_id"]
    if sid in train_sids:
        s["split"] = "TRAIN"
    elif sid in val_sids:
        s["split"] = "VALIDATION"
    elif sid in test_sids:
        s["split"] = "TEST"
    else:
        raise ValueError(f"Session {sid} not assigned to any split!")

watch_train = [s for s in watch_sessions if s["split"] == "TRAIN"]
watch_val = [s for s in watch_sessions if s["split"] == "VALIDATION"]
watch_test = [s for s in watch_sessions if s["split"] == "TEST"]

phone_train = [s for s in phone_sessions if s["split"] == "TRAIN"]
phone_val = [s for s in phone_sessions if s["split"] == "VALIDATION"]
phone_test = [s for s in phone_sessions if s["split"] == "TEST"]

print(f"Watch Split: Train={len(watch_train)}, Val={len(watch_val)}, Test={len(watch_test)}")
print(f"Phone Split: Train={len(phone_train)}, Val={len(phone_val)}, Test={len(phone_test)}")

def export_csv(sessions, out_file):
    fields = ["device", "filename", "session_id", "activity", "fall_binary", "split", "rows", "duration", "first_timestamp", "last_timestamp"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in sessions:
            writer.writerow(s)

export_csv([s for s in all_sessions if s["split"] == "TRAIN"], os.path.join(COMMON_SPLIT_DIR, "train_sessions.csv"))
export_csv([s for s in all_sessions if s["split"] == "VALIDATION"], os.path.join(COMMON_SPLIT_DIR, "validation_sessions.csv"))
export_csv([s for s in all_sessions if s["split"] == "TEST"], os.path.join(COMMON_SPLIT_DIR, "test_sessions.csv"))
export_csv(all_sessions, os.path.join(COMMON_SPLIT_DIR, "all_split_sessions.csv"))

split_report = {
    "random_seed": RANDOM_SEED,
    "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "unique_sessions_total": len(unique_sessions),
    "unique_sessions_train": len(train_sids),
    "unique_sessions_val": len(val_sids),
    "unique_sessions_test": len(test_sids),
    "watch": {
        "total_sessions": len(watch_sessions),
        "train_sessions": len(watch_train),
        "val_sessions": len(watch_val),
        "test_sessions": len(watch_test),
        "train_rows": sum(x["rows"] for x in watch_train),
        "val_rows": sum(x["rows"] for x in watch_val),
        "test_rows": sum(x["rows"] for x in watch_test),
        "by_activity": {
            act: {
                "train": sum(1 for x in watch_train if x["activity"] == act),
                "val": sum(1 for x in watch_val if x["activity"] == act),
                "test": sum(1 for x in watch_test if x["activity"] == act)
            } for act in VALID_CLASSES
        }
    },
    "phone": {
        "total_sessions": len(phone_sessions),
        "train_sessions": len(phone_train),
        "val_sessions": len(phone_val),
        "test_sessions": len(phone_test),
        "train_rows": sum(x["rows"] for x in phone_train),
        "val_rows": sum(x["rows"] for x in phone_val),
        "test_rows": sum(x["rows"] for x in phone_test),
        "by_activity": {
            act: {
                "train": sum(1 for x in phone_train if x["activity"] == act),
                "val": sum(1 for x in phone_val if x["activity"] == act),
                "test": sum(1 for x in phone_test if x["activity"] == act)
            } for act in VALID_CLASSES
        }
    }
}

with open(os.path.join(COMMON_SPLIT_DIR, "split_report.json"), "w") as f:
    json.dump(split_report, f, indent=2)

print("Saved unified split manifests to preprocessing/common_split/")
