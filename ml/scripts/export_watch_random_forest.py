import os
import sys
import json
import joblib
import numpy as np

WORKSPACE_DIR = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
ML_DIR = os.path.join(WORKSPACE_DIR, "ml")
WATCH_MODELS_DIR = os.path.join(ML_DIR, "models/watch")
PREPROCESSING_DIR = os.path.join(WORKSPACE_DIR, "preprocessing")

sys.path.append(os.path.join(ML_DIR, "scripts"))
from train_random_forest import extract_window_features

def export_trees_to_json():
    print("=" * 75)
    print("SMARTFALL AI — EXPORTING WATCH RANDOM FOREST TREES FOR WEAR OS")
    print("=" * 75)
    
    rf_model_path = os.path.join(WATCH_MODELS_DIR, "model.joblib")
    rf = joblib.load(rf_model_path)
    class_indices = rf.classes_.tolist()
    
    trees_data = []
    for tree_idx, estimator in enumerate(rf.estimators_):
        t = estimator.tree_
        # Normalize leaf values to probability distributions across all 14 classes
        values_13 = t.value.squeeze(axis=1) # (nodes, 13)
        sums = values_13.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        norm_values_13 = values_13 / sums
        
        # Map 13 classes to full 14-class vector
        norm_values_14 = np.zeros((t.node_count, 14), dtype=np.float32)
        for k_idx, c_id in enumerate(class_indices):
            norm_values_14[:, c_id] = norm_values_13[:, k_idx]
            
        trees_data.append({
            "tree_id": tree_idx,
            "node_count": int(t.node_count),
            "children_left": t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": [round(float(x), 5) for x in t.threshold],
            "values": [[round(float(v), 5) for v in row] for row in norm_values_14]
        })
        
    forest_payload = {
        "n_estimators": len(trees_data),
        "n_classes": 14,
        "n_features": 72,
        "class_indices": class_indices,
        "trees": trees_data
    }
    
    trees_json_path = os.path.join(WATCH_MODELS_DIR, "trees.json")
    with open(trees_json_path, "w") as f:
        json.dump(forest_payload, f)
        
    print(f"Exported {len(trees_data)} trees to {trees_json_path} ({os.path.getsize(trees_json_path)/1024:.1f} KB)")
    
    # Also copy to Wear assets directory
    wear_assets_dir = os.path.join(WORKSPACE_DIR, "wear/src/main/assets")
    os.makedirs(wear_assets_dir, exist_ok=True)
    with open(os.path.join(wear_assets_dir, "trees.json"), "w") as f:
        json.dump(forest_payload, f)
        
    with open(os.path.join(WATCH_MODELS_DIR, "scaler.json"), "r") as f:
        w_scaler = json.load(f)
    with open(os.path.join(wear_assets_dir, "scaler.json"), "w") as f:
        json.dump(w_scaler, f, indent=2)
        
    with open(os.path.join(ML_DIR, "models/label_map.json"), "r") as f:
        l_map = json.load(f)
    with open(os.path.join(wear_assets_dir, "label_map.json"), "w") as f:
        json.dump(l_map, f, indent=2)

    # -------------------------------------------------------------
    # PREDICTION EQUIVALENCE VERIFICATION (PYTHON JOBLIB vs JSON TREES)
    # -------------------------------------------------------------
    def eval_json_trees(feats_2d, trees_list):
        N = feats_2d.shape[0]
        n_classes = 14
        all_probs = np.zeros((N, n_classes), dtype=np.float32)
        
        for tree in trees_list:
            cl = tree["children_left"]
            cr = tree["children_right"]
            feat = tree["feature"]
            th = tree["threshold"]
            vals = np.array(tree["values"], dtype=np.float32)
            
            for i in range(N):
                node = 0
                while cl[node] != -1: # Not a leaf
                    f_idx = feat[node]
                    if feats_2d[i, f_idx] <= th[node]:
                        node = cl[node]
                    else:
                        node = cr[node]
                all_probs[i] += vals[node]
                
        all_probs /= len(trees_list)
        return np.argmax(all_probs, axis=1), all_probs

    # Load 100 validation windows
    watch_val_dir = os.path.join(PREPROCESSING_DIR, "02_robust_scaling/watch/validation")
    X_val = np.load(os.path.join(watch_val_dir, "X.npy"))[:100]
    y_val = np.load(os.path.join(watch_val_dir, "y_14.npy"))[:100]
    
    feats_100 = extract_window_features(X_val)
    
    py_preds = rf.predict(feats_100)
    json_preds, json_probs = eval_json_trees(feats_100, trees_data)
    
    matches = int(np.sum(py_preds == json_preds))
    agreement_pct = (matches / len(py_preds)) * 100.0
    
    verif_data = {
        "device": "WATCH",
        "model": "RandomForest",
        "sample_windows_evaluated": len(py_preds),
        "matching_predictions": matches,
        "differing_predictions": len(py_preds) - matches,
        "agreement_percentage": agreement_pct,
        "status": "VERIFIED_100_PERCENT_EQUIVALENT" if agreement_pct == 100.0 else "DISCREPANCY_DETECTED"
    }
    
    with open(os.path.join(WATCH_MODELS_DIR, "deployment_verification.json"), "w") as f:
        json.dump(verif_data, f, indent=2)
        
    print(f"\nWatch Model Verification: {matches}/{len(py_preds)} identical ({agreement_pct:.2f}% agreement).")
    print("Deployment verification saved to ml/models/watch/deployment_verification.json")

if __name__ == "__main__":
    export_trees_to_json()
