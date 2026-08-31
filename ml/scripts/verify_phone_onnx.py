import os
import sys
import json
import torch
import numpy as np
import onnxruntime as ort

WORKSPACE_DIR = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
ML_DIR = os.path.join(WORKSPACE_DIR, "ml")
PHONE_MODELS_DIR = os.path.join(ML_DIR, "models/phone")
PREPROCESSING_DIR = os.path.join(WORKSPACE_DIR, "preprocessing")

sys.path.append(os.path.join(ML_DIR, "scripts"))
from train_cnn import Conv1DNet

def verify_phone_onnx():
    print("=" * 75)
    print("SMARTFALL AI — VERIFYING PHONE ONNX PREDICTION EQUIVALENCE")
    print("=" * 75)
    
    device = torch.device("cpu") # Test on CPU matching mobile
    
    # Load PyTorch Model
    pth_path = os.path.join(PHONE_MODELS_DIR, "model.pth")
    py_model = Conv1DNet(in_channels=9, num_classes=14).to(device)
    py_model.load_state_dict(torch.load(pth_path, map_location=device))
    py_model.eval()
    
    # Load ONNX Model
    onnx_path = os.path.join(PHONE_MODELS_DIR, "model.onnx")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Load 100 validation windows
    phone_val_dir = os.path.join(PREPROCESSING_DIR, "02_robust_scaling/phone/validation")
    X_val = np.load(os.path.join(phone_val_dir, "X.npy"))[:100] # (100, 100, 9)
    y_val = np.load(os.path.join(phone_val_dir, "y_14.npy"))[:100]
    
    # 1. PyTorch Predictions
    with torch.no_grad():
        t = torch.tensor(X_val, dtype=torch.float32).to(device)
        py_logits = py_model(t).cpu().numpy()
        py_preds = np.argmax(py_logits, axis=1)
        
    # 2. ONNX Predictions
    ort_inputs = {ort_session.get_inputs()[0].name: X_val.astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_logits = ort_outputs[0]
    onnx_preds = np.argmax(onnx_logits, axis=1)
    
    # Compare
    max_logit_diff = float(np.max(np.abs(py_logits - onnx_logits)))
    matches = int(np.sum(py_preds == onnx_preds))
    agreement_pct = (matches / len(py_preds)) * 100.0
    
    verif_data = {
        "device": "PHONE",
        "model": "1D_CNN",
        "sample_windows_evaluated": len(py_preds),
        "matching_predictions": matches,
        "differing_predictions": len(py_preds) - matches,
        "max_absolute_logit_difference": max_logit_diff,
        "agreement_percentage": agreement_pct,
        "status": "VERIFIED_100_PERCENT_EQUIVALENT" if agreement_pct == 100.0 else "DISCREPANCY_DETECTED"
    }
    
    with open(os.path.join(PHONE_MODELS_DIR, "deployment_verification.json"), "w") as f:
        json.dump(verif_data, f, indent=2)
        
    print(f"Phone ONNX Verification: {matches}/{len(py_preds)} identical ({agreement_pct:.2f}% agreement, max diff={max_logit_diff:.6e}).")
    print("Deployment verification saved to ml/models/phone/deployment_verification.json")
    
    # Copy assets to Android app assets directory
    app_assets_dir = os.path.join(WORKSPACE_DIR, "app/src/main/assets")
    os.makedirs(app_assets_dir, exist_ok=True)
    
    import shutil
    shutil.copy(onnx_path, os.path.join(app_assets_dir, "model.onnx"))
    # Also copy onnx data if exists
    if os.path.exists(onnx_path + ".data"):
        shutil.copy(onnx_path + ".data", os.path.join(app_assets_dir, "model.onnx.data"))
        
    with open(os.path.join(PHONE_MODELS_DIR, "scaler.json"), "r") as f:
        p_scaler = json.load(f)
    with open(os.path.join(app_assets_dir, "scaler.json"), "w") as f:
        json.dump(p_scaler, f, indent=2)
        
    with open(os.path.join(ML_DIR, "models/label_map.json"), "r") as f:
        l_map = json.load(f)
    with open(os.path.join(app_assets_dir, "label_map.json"), "w") as f:
        json.dump(l_map, f, indent=2)
        
    print(f"Copied ONNX model and scaler to {app_assets_dir}")

if __name__ == "__main__":
    verify_phone_onnx()
