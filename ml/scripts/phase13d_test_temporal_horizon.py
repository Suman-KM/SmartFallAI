import os, sys, json
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

sys.path.append(os.path.join(WORKSPACE, "ml/scripts"))
from phase13d_architecture_experiments import prepare_dataset

p_val_df = prepare_dataset("phone", "validation")
w_val_df = prepare_dataset("watch", "validation")

def test_temporal_horizon(df, device="phone"):
    unique_sids = df['source_session_id'].unique()
    
    # Grid search over impact thresholds and stillness horizons
    # For Phone:
    # Impact shock: acc_peak >= 20.0, jerk_peak >= 350 (or acc_range >= 12 && jerk >= 250 && gyro >= 3.0)
    # Confirmation horizon: within windows t+1, t+2, t+3
    # Active locomotion rejection: if subject keeps walking/running (continuous acc_std >= 3.0 or periodic step peaks)
    
    imp_p_th = 20.0 if device == "phone" else 22.0
    jerk_th = 350.0 if device == "phone" else 500.0
    
    detected_falls = 0
    total_falls = 0
    false_alarms = 0
    total_adls = 0
    fa_acts = {}
    recall_by_act = {}
    
    for sid in unique_sids:
        s_df = df[df['source_session_id'] == sid].sort_values('window_index').reset_index(drop=True)
        is_fall = s_df['is_fall'].iloc[0]
        act = s_df['target_activity'].iloc[0]
        n_wins = len(s_df)
        
        triggered = False
        
        for i in range(n_wins):
            r = s_df.iloc[i]
            
            # 1. Collision Impact Shock Gate
            is_collision = (r['acc_peak'] >= imp_p_th and r['jerk_peak'] >= jerk_th) or \
                           (r['acc_range'] >= 14.0 and r['jerk_peak'] >= 250.0 and r['gyro_peak'] >= 3.5)
                           
            if is_collision:
                # Look ahead into horizon: windows i+1 to i+3
                # Check if movement settles to stillness or if locomotion continues
                horizon_wins = s_df.iloc[i+1 : min(n_wins, i+4)]
                
                if len(horizon_wins) == 0:
                    # Near end of session, check current window
                    if r['fall_prob'] >= 0.45 or r['lying_down_prob'] >= 0.50:
                        triggered = True
                        break
                else:
                    # Check if continuous locomotion continues in every horizon window
                    # Running/walking continuously has acc_std >= 2.8 or gyro >= 3.0 in ALL subsequent windows
                    all_locomotion = True
                    has_settled_window = False
                    
                    for _, hw in horizon_wins.iterrows():
                        is_loco = (hw['acc_std'] >= (3.2 if device=="phone" else 5.5)) or (hw['gyro_peak'] >= (3.5 if device=="phone" else 4.0))
                        if not is_loco:
                            all_locomotion = False
                            
                        # Is this window settled / recumbent?
                        is_settled = (hw['acc_std'] <= (2.5 if device=="phone" else 3.8)) and (hw['gyro_peak'] <= (2.5 if device=="phone" else 3.5))
                        has_fall_posture = (hw['fall_prob'] >= 0.40) or (r['fall_prob'] >= 0.40)
                        
                        if is_settled and has_fall_posture:
                            has_settled_window = True
                            
                    if (not all_locomotion) and has_settled_window:
                        triggered = True
                        break
                        
        if is_fall:
            total_falls += 1
            if act not in recall_by_act: recall_by_act[act] = [0, 0]
            recall_by_act[act][1] += 1
            if triggered:
                detected_falls += 1
                recall_by_act[act][0] += 1
        else:
            total_adls += 1
            if triggered:
                false_alarms += 1
                fa_acts[act] = fa_acts.get(act, 0) + 1
                
    rec = (detected_falls / total_falls) * 100.0 if total_falls > 0 else 0.0
    prec = (detected_falls / (detected_falls + false_alarms)) * 100.0 if (detected_falls + false_alarms) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = (false_alarms / total_adls) * 100.0 if total_adls > 0 else 0.0
    
    print(f"Device: {device.upper()}")
    print(f"  Recall:    {rec:.1f}% ({detected_falls}/{total_falls})")
    print(f"  Precision: {prec:.1f}%")
    print(f"  F1 Score:  {f1:.1f}%")
    print(f"  FPR:       {fpr:.1f}% ({false_alarms}/{total_adls} sessions)")
    print(f"  False Alarms by Activity: {fa_acts}")
    print(f"  Recall by Fall Type:      {recall_by_act}")

print("--- TESTING TEMPORAL HORIZON ARCHITECTURE ---")
test_temporal_horizon(p_val_df, "phone")
print()
test_temporal_horizon(w_val_df, "watch")
