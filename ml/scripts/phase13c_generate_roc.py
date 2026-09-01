import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
RESULTS_DIR = os.path.join(WORKSPACE, "ml/results/phase13c")

sys.path.append(os.path.join(WORKSPACE, "ml/scripts"))
from phase13c_test_set_evaluation import load_split_data

p_test_df = load_split_data("phone", "test")
w_test_df = load_split_data("watch", "test")

# Phone ROC & PR
fpr_p, tpr_p, _ = roc_curve(p_test_df['is_fall'], p_test_df['fall_prob'])
roc_auc_p = auc(fpr_p, tpr_p)
prec_p, rec_p, _ = precision_recall_curve(p_test_df['is_fall'], p_test_df['fall_prob'])
pr_auc_p = auc(rec_p, prec_p)

# Watch ROC & PR
fpr_w, tpr_w, _ = roc_curve(w_test_df['is_fall'], w_test_df['fall_prob'])
roc_auc_w = auc(fpr_w, tpr_w)
prec_w, rec_w, _ = precision_recall_curve(w_test_df['is_fall'], w_test_df['fall_prob'])
pr_auc_w = auc(rec_w, prec_w)

fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# ROC
ax[0].plot(fpr_p, tpr_p, color='#1f77b4', linewidth=2.2, label=f'Phone 1D-CNN (AUC = {roc_auc_p:.3f})')
ax[0].plot(fpr_w, tpr_w, color='#ff7f0e', linewidth=2.2, label=f'Watch Random Forest (AUC = {roc_auc_w:.3f})')
ax[0].plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5)
ax[0].set_xlabel('False Positive Rate', fontsize=11)
ax[0].set_ylabel('True Positive Rate (Recall)', fontsize=11)
ax[0].set_title('ROC Curves on Untouched Test Set', fontsize=12, fontweight='bold')
ax[0].legend(fontsize=9, loc='lower right')
ax[0].grid(True, linestyle=':', alpha=0.6)

# PR
ax[1].plot(rec_p, prec_p, color='#1f77b4', linewidth=2.2, label=f'Phone 1D-CNN (AUC = {pr_auc_p:.3f})')
ax[1].plot(rec_w, prec_w, color='#ff7f0e', linewidth=2.2, label=f'Watch Random Forest (AUC = {pr_auc_w:.3f})')
ax[1].set_xlabel('Recall', fontsize=11)
ax[1].set_ylabel('Precision', fontsize=11)
ax[1].set_title('Precision-Recall Curves on Untouched Test Set', fontsize=12, fontweight='bold')
ax[1].legend(fontsize=9, loc='lower left')
ax[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_pr_curves.png"))
plt.close()
print("Saved roc_pr_curves.png")
