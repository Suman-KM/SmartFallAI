import os
import json
import numpy as np
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

# 1. Load Phone ONNX model
phone_onnx_path = os.path.join(WORKSPACE, "app/src/main/assets/model.onnx")
phone_scaler_path = os.path.join(WORKSPACE, "app/src/main/assets/scaler.json")
with open(phone_scaler_path) as f:
    phone_scaler = json.load(f)

phone_median = np.array(phone_scaler["median"], dtype=np.float32)
phone_iqr = np.array(phone_scaler["iqr"], dtype=np.float32)

ort_session = ort.InferenceSession(phone_onnx_path)

# 2. Load Watch RF model
watch_rf_path = os.path.join(WORKSPACE, "ml/models/watch/model.joblib")
watch_scaler_path = os.path.join(WORKSPACE, "wear/src/main/assets/scaler.json")
with open(watch_scaler_path) as f:
    watch_scaler = json.load(f)

watch_median = np.array(watch_scaler["median"], dtype=np.float32)
watch_iqr = np.array(watch_scaler["iqr"], dtype=np.float32)

watch_rf = joblib.load(watch_rf_path)

CLASS_NAMES = [
    "FALL_BACKWARD", "FALL_FORWARD", "FALL_FROM_SITTING", "FALL_LEFT", "FALL_RIGHT",
    "JUMPING", "LYING_DOWN", "PICKING_UP_OBJECT", "RUNNING", "SITTING", "SIT_DOWN",
    "STANDING", "STAND_UP", "WALKING"
]

print("Loaded Phone ONNX & Watch RF models successfully.")
print(f"Phone Median: {phone_median}")
print(f"Phone IQR:    {phone_iqr}")
print(f"Watch Median: {watch_median}")
print(f"Watch IQR:    {watch_iqr}")

# Feature names: accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw

def test_phone_window(raw_window, desc=""):
    scaled = (raw_window - phone_median) / phone_iqr
    # shape: (1, 100, 9)
    inp = scaled[np.newaxis, :, :]
    out = ort_session.run(None, {ort_session.get_inputs()[0].name: inp})[0]
    probs = out[0]
    fall_prob = np.sum(probs[:5])
    top_idx = np.argmax(probs)
    print(f"[{desc}] Top: {CLASS_NAMES[top_idx]:<18} ({probs[top_idx]:.3f}) | FallProb: {fall_prob:.3f} | {'FALL!' if fall_prob >= 0.5 else 'Normal'}")
    return fall_prob, top_idx

def extract_watch_features(scaled_window):
    # Watch feature extractor: 72 features (mean, std, min, max, median, iqr, range, rms) for 9 channels
    feats = []
    for ch in range(9):
        col = scaled_window[:, ch]
        mean = np.mean(col)
        std = np.std(col)
        min_v = np.min(col)
        max_v = np.max(col)
        median_v = np.median(col)
        iqr_v = np.percentile(col, 75) - np.percentile(col, 25)
        range_v = max_v - min_v
        rms_v = np.sqrt(np.mean(col**2))
        feats.extend([mean, std, min_v, max_v, median_v, iqr_v, range_v, rms_v])
    return np.array(feats, dtype=np.float32)

def test_watch_window(raw_window, desc=""):
    scaled = (raw_window - watch_median) / watch_iqr
    feats = extract_watch_features(scaled)[np.newaxis, :]
    probs = watch_rf.predict_proba(feats)[0]
    fall_prob = np.sum(probs[:5])
    top_idx = np.argmax(probs)
    print(f"[{desc}] Top: {CLASS_NAMES[top_idx]:<18} ({probs[top_idx]:.3f}) | FallProb: {fall_prob:.3f} | {'FALL!' if fall_prob >= 0.5 else 'Normal'}")
    return fall_prob, top_idx

print("\n--- TEST 1: STATIC POSITIONS ON PHONE ---")
# Phone flat on desk, screen up: accX=0, accY=0, accZ=9.81, gyro=0, pitch=0, roll=0, yaw=0
w_flat = np.zeros((100, 9), dtype=np.float32)
w_flat[:, 2] = 9.81  # accZ = 9.81
test_phone_window(w_flat, "Flat Screen-Up (0, 0, 9.81, pitch=0, roll=0)")

# Phone flat on desk, but with roll=90 (rotated on side)
w_side = np.zeros((100, 9), dtype=np.float32)
w_side[:, 0] = 9.81  # accX = 9.81
w_side[:, 7] = 90.0  # roll = 90
test_phone_window(w_side, "On Side (accX=9.81, roll=90)")

# Phone upright portrait (in pocket / hand): accY = 9.81
w_upright = np.zeros((100, 9), dtype=np.float32)
w_upright[:, 1] = 9.81 # accY = 9.81
w_upright[:, 6] = 90.0 # pitch = 90
test_phone_window(w_upright, "Upright Portrait (accY=9.81, pitch=90)")

# Phone lying down screen down: accZ = -9.81, roll=180
w_down = np.zeros((100, 9), dtype=np.float32)
w_down[:, 2] = -9.81
w_down[:, 7] = 180.0
test_phone_window(w_down, "Screen Down (accZ=-9.81, roll=180)")

print("\n--- TEST 2: STATIC POSITIONS ON WATCH ---")
# Watch flat on charger/table: accZ = 9.81
w_watch_flat = np.zeros((100, 9), dtype=np.float32)
w_watch_flat[:, 2] = 9.81
test_watch_window(w_watch_flat, "Watch Flat on Table (accZ=9.81)")

# Watch on wrist facing user: accX=0, accY=-9.81, accZ=0 (gravity along Y)
w_watch_wrist = np.zeros((100, 9), dtype=np.float32)
w_watch_wrist[:, 1] = -9.81
test_watch_window(w_watch_wrist, "Watch On Wrist Normal (accY=-9.81)")

# Watch tilted 90 degrees
w_watch_tilt = np.zeros((100, 9), dtype=np.float32)
w_watch_tilt[:, 0] = 9.81
test_watch_window(w_watch_tilt, "Watch Tilted 90 deg (accX=9.81)")
