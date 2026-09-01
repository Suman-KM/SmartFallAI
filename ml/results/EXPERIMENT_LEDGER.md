# SMARTFALL AI — MASTER EXPERIMENT LEDGER

Authoritative traceability ledger mapping all benchmark runs, model checkpoints, and metric evaluations across Phases 5 through 12.

| Phase | Device | Model Architecture | Preprocessing | Train / Val / Test Sessions | Validation Recall | Test Fall Recall | Test Fall F1 | Test Macro-F1 | Status / Artifact |
|---|---|---|---|---|---|---|---|---|---|
| **`Phase 6 Benchmark`** | `WATCH` | **`Random Forest`** | `02_robust_scaling` | 182 / 35 / 35 | 79.18% | **84.08%** | 0.7376 | 0.5285 | Initial screening winner based on validation macro-F1 & recall. |
| **`Phase 6 Benchmark`** | `PHONE` | **`1D-CNN`** | `02_robust_scaling` | 171 / 41 / 42 | 75.69% | **77.19%** | 0.6860 | 0.4578 | Initial screening winner based on validation macro-F1 & spatial feature learning. |
| **`Phase 7 Checkpoint Export`** | `PHONE` | **`1D-CNN (Best Checkpoint)`** | `02_robust_scaling` | 171 / 41 / 42 | 75.69% | **77.19%** | 0.7019 | 0.4901 | Exported frozen best validation weights (model.pth / model.onnx). |
| **`Phase 8 Deployment Verification`** | `WATCH` | **`Random Forest (trees.bin)`** | `02_robust_scaling` | 182 / 35 / 35 | N/A | **N/A** | N/A | N/A | Flat primitive binary tree format verified on Samsung Galaxy Watch 4. |
| **`Phase 8 Deployment Verification`** | `PHONE` | **`1D-CNN (model.onnx)`** | `02_robust_scaling` | 171 / 41 / 42 | N/A | **N/A** | N/A | N/A | Self-contained ONNX model verified on Samsung Galaxy A50s. |
| **`Phase 12 Authoritative Test Audit`** | `WATCH` | **`Random Forest (Deployed Champion)`** | `02_robust_scaling` | 182 / 35 / 35 | 79.18% | **84.08%** | 0.6960 | 0.1112 | Authoritative final evaluation on frozen test set. |
| **`Phase 12 Authoritative Test Audit`** | `PHONE` | **`1D-CNN (Deployed Champion)`** | `02_robust_scaling` | 171 / 41 / 42 | 75.69% | **63.42%** | 0.7019 | 0.4901 | Authoritative final evaluation on frozen test set. |