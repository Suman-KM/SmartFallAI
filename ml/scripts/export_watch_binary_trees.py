import os
import sys
import struct
import joblib
import numpy as np
import json

WORKSPACE_DIR = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
ML_DIR = os.path.join(WORKSPACE_DIR, "ml")
WATCH_MODELS_DIR = os.path.join(ML_DIR, "models/watch")
WEAR_ASSETS_DIR = os.path.join(WORKSPACE_DIR, "wear/src/main/assets")

sys.path.append(os.path.join(ML_DIR, "scripts"))
from train_random_forest import extract_window_features

def export_binary_trees():
    print("=" * 75)
    print("SMARTFALL AI — EXPORTING WATCH RANDOM FOREST TO BINARY FORMAT")
    print("=" * 75)
    
    rf_model_path = os.path.join(WATCH_MODELS_DIR, "model.joblib")
    rf = joblib.load(rf_model_path)
    class_indices = rf.classes_.tolist()
    
    bin_path = os.path.join(WATCH_MODELS_DIR, "trees.bin")
    wear_bin_path = os.path.join(WEAR_ASSETS_DIR, "trees.bin")
    
    with open(bin_path, "wb") as f:
        # Header: Magic (4 bytes 'SFRF'), n_trees (int32), n_classes (int32), n_features (int32)
        f.write(b"SFRF")
        f.write(struct.pack(">iii", len(rf.estimators_), 14, 72))
        
        for tree_idx, estimator in enumerate(rf.estimators_):
            t = estimator.tree_
            node_count = int(t.node_count)
            f.write(struct.pack(">i", node_count))
            
            # Values normalization
            values_13 = t.value.squeeze(axis=1) # (nodes, 13)
            sums = values_13.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            norm_values_13 = values_13 / sums
            norm_values_14 = np.zeros((node_count, 14), dtype=np.float32)
            for k_idx, c_id in enumerate(class_indices):
                norm_values_14[:, c_id] = norm_values_13[:, k_idx]
                
            for n in range(node_count):
                left = int(t.children_left[n])
                right = int(t.children_right[n])
                feat = int(t.feature[n]) if left != -1 else 0
                th = float(t.threshold[n]) if left != -1 else 0.0
                is_leaf = 1 if left == -1 else 0
                
                # Format: left (short), right (short), feat (int16), th (float32), is_leaf (int8)
                f.write(struct.pack(">hhfhb", left, right, th, feat, is_leaf))
                if is_leaf:
                    # 14 float32 probabilities
                    for c in range(14):
                        f.write(struct.pack(">f", float(norm_values_14[n, c])))
                        
    file_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"Exported trees.bin ({file_size_mb:.2f} MB)")
    
    # Copy to wear assets
    with open(bin_path, "rb") as src, open(wear_bin_path, "wb") as dst:
        dst.write(src.read())
        
    print(f"Copied to {wear_bin_path}")
    
    # Verify binary evaluator in Python
    with open(bin_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"SFRF"
        n_trees, n_classes, n_feats = struct.unpack(">iii", f.read(12))
        
        parsed_trees = []
        for _ in range(n_trees):
            (n_nodes,) = struct.unpack(">i", f.read(4))
            nodes = []
            for _ in range(n_nodes):
                left, right, th, feat, is_leaf = struct.unpack(">hhfhb", f.read(11))
                if is_leaf:
                    probs = list(struct.unpack(">14f", f.read(56)))
                else:
                    probs = None
                nodes.append((left, right, feat, th, is_leaf, probs))
            parsed_trees.append(nodes)
            
    # Test on test split
    X_test_raw = np.load(os.path.join(WORKSPACE_DIR, "preprocessing/02_robust_scaling/watch/test/X.npy"))
    y_test = np.load(os.path.join(WORKSPACE_DIR, "preprocessing/02_robust_scaling/watch/test/y_14.npy"))
    X_feats = extract_window_features(X_test_raw[:100])
    
    # Evaluate with binary trees
    bin_preds = []
    for f_idx in range(len(X_feats)):
        feat = X_feats[f_idx]
        total_p = np.zeros(14, dtype=np.float32)
        for t_nodes in parsed_trees:
            curr = 0
            while t_nodes[curr][4] == 0: # not leaf
                c_left, c_right, f_i, th, _, _ = t_nodes[curr]
                if feat[f_i] <= th:
                    curr = c_left
                else:
                    curr = c_right
            total_p += np.array(t_nodes[curr][5])
        total_p /= len(parsed_trees)
        bin_preds.append(np.argmax(total_p))
        
    py_preds = rf.predict(X_feats)
    agreement = (np.array(bin_preds) == py_preds).mean() * 100
    print(f"Binary Tree Equivalence: {agreement:.2f}% (100/100 matching)")
    assert agreement == 100.0

if __name__ == "__main__":
    export_binary_trees()
