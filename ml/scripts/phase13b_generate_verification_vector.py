import os
import json
import numpy as np
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

# Generate deterministic test window: 100 samples of 9 features
np.random.seed(42)
test_raw = np.random.uniform(low=-15.0, high=15.0, size=(100, 9)).astype(np.float32)

# Load scalers
with open(os.path.join(WORKSPACE, "app/src/main/assets/scaler.json")) as f:
    phone_scaler = json.load(f)
p_med = np.array(phone_scaler["median"], dtype=np.float32)
p_iqr = np.array(phone_scaler["iqr"], dtype=np.float32)

with open(os.path.join(WORKSPACE, "wear/src/main/assets/scaler.json")) as f:
    watch_scaler = json.load(f)
w_med = np.array(watch_scaler["median"], dtype=np.float32)
w_iqr = np.array(watch_scaler["iqr"], dtype=np.float32)

phone_scaled = (test_raw - p_med) / p_iqr
watch_scaled = (test_raw - w_med) / w_iqr

# Run Phone ONNX
phone_onnx = ort.InferenceSession(os.path.join(WORKSPACE, "app/src/main/assets/model.onnx"))
phone_logits = phone_onnx.run(None, {phone_onnx.get_inputs()[0].name: phone_scaled[np.newaxis, :, :]})[0][0]
e = np.exp(phone_logits - np.max(phone_logits))
phone_probs = e / np.sum(e)

# Run Watch RF
def extract_watch_features(X_3d):
    means = np.mean(X_3d, axis=1)
    stds = np.std(X_3d, axis=1)
    mins = np.min(X_3d, axis=1)
    maxs = np.max(X_3d, axis=1)
    ranges = maxs - mins
    medians = np.median(X_3d, axis=1)
    rms = np.sqrt(np.mean(X_3d ** 2, axis=1))
    energy = np.mean(X_3d ** 2, axis=1)
    return np.hstack([means, stds, mins, maxs, ranges, medians, rms, energy]).astype(np.float32)

watch_rf = joblib.load(os.path.join(WORKSPACE, "ml/models/watch/model.joblib"))
watch_feats = extract_watch_features(watch_scaled[np.newaxis, :, :])
watch_probs = watch_rf.predict_proba(watch_feats)[0]

out_data = {
    "test_raw_sample_0": test_raw[0].tolist(),
    "test_raw_sample_99": test_raw[99].tolist(),
    "phone_scaled_sample_0": phone_scaled[0].tolist(),
    "phone_logits": phone_logits.tolist(),
    "phone_probs": phone_probs.tolist(),
    "phone_top_class": int(np.argmax(phone_probs)),
    "phone_fall_prob": float(np.sum(phone_probs[:5])),
    "watch_scaled_sample_0": watch_scaled[0].tolist(),
    "watch_probs": watch_probs.tolist(),
    "watch_top_class": int(np.argmax(watch_probs)),
    "watch_fall_prob": float(np.sum(watch_probs[:5]))
}

out_path = os.path.join(WORKSPACE, "ml/results/phase13/verification_vector.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"Generated verification vector. Saved to {out_path}")
print(f"Phone Top: {out_data['phone_top_class']}, FallProb: {out_data['phone_fall_prob']:.4f}")
print(f"Watch Top: {out_data['watch_top_class']}, FallProb: {out_data['watch_fall_prob']:.4f}")
