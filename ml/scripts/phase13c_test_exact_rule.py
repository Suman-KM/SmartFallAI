import os, sys, json
import numpy as np
import pandas as pd

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

sys.path.append(os.path.join(WORKSPACE, "ml/scripts"))
from phase13c_grid_search import prepare_dataset

def evaluate_calibrated_state_machine(device, split="validation"):
    df = prepare_dataset(device, split)
    
    total_fall_sessions = df[df['is_fall']]['source_session_id'].nunique()
    total_adl_sessions = df[~df['is_fall']]['source_session_id'].nunique()
    
    # Device-specific calibrated thresholds
    if device == "phone":
        # Phone parameters (SM-A507FN)
        # Impact: sudden deceleration peak or sharp angular tumble
        ap_th = 20.0       # m/s^2 (2.04 g)
        ar_th = 12.0       # m/s^2
        gp_th = 3.0        # rad/s
        p_th = 0.45        # Model fall probability threshold
        max_running_astd = 5.0  # reject continuous running / thrashing
        max_running_gp = 4.0    # rad/s
        mem_len = 3        # memory windows
        consec_req = 2     # consecutive windows
    else:
        # Watch parameters (SM-R870)
        # Impact: arm impact deceleration peak
        ap_th = 22.0       # m/s^2 (2.24 g)
        ar_th = 15.0       # m/s^2
        gp_th = 3.5        # rad/s
        p_th = 0.50        # Model fall probability threshold
        max_running_astd = 8.0  # reject continuous jumping / running thrashing (jumping is > 12 m/s^2)
        max_running_gp = 6.0    # rad/s
        mem_len = 3        # memory windows
        consec_req = 2     # consecutive windows

    det_falls = 0
    det_by_type = {'FALL_BACKWARD': 0, 'FALL_FORWARD': 0, 'FALL_FROM_SITTING': 0, 'FALL_LEFT': 0, 'FALL_RIGHT': 0}
    total_by_type = {'FALL_BACKWARD': 0, 'FALL_FORWARD': 0, 'FALL_FROM_SITTING': 0, 'FALL_LEFT': 0, 'FALL_RIGHT': 0}
    for k in total_by_type:
        total_by_type[k] = df[df['target_activity'] == k]['source_session_id'].nunique()
        
    fa_sessions = 0
    fa_by_act = {}
    
    for sid, s_df in df.groupby('source_session_id'):
        s_df = s_df.sort_values('window_index')
        is_fall = s_df['is_fall'].iloc[0]
        act = s_df['target_activity'].iloc[0]
        
        recent_impact_countdown = 0
        consecutive_candidates = 0
        triggered = False
        
        for _, row in s_df.iterrows():
            fp = row['fall_prob']
            ap = row['acc_peak']
            ar = row['acc_range']
            gp = row['gyro_peak']
            astd = row['acc_std']
            
            # Step 1: Detect dynamic impact shock
            has_impact = (ap >= ap_th) or (ar >= ar_th and gp >= gp_th)
            if has_impact:
                recent_impact_countdown = mem_len
            elif recent_impact_countdown > 0:
                recent_impact_countdown -= 1
                
            # Step 2: Continuous high-motion active thrash filter (Running / Jumping)
            is_active_thrash = (astd >= max_running_astd) and (gp >= max_running_gp)
            
            # Step 3: Fall candidate window
            is_candidate = (fp >= p_th) and (has_impact or recent_impact_countdown > 0) and (not is_active_thrash)
            
            if is_candidate:
                consecutive_candidates += 1
                if consecutive_candidates >= consec_req:
                    triggered = True
                    break
            else:
                consecutive_candidates = 0
                
        if is_fall:
            if triggered:
                det_falls += 1
                if act in det_by_type:
                    det_by_type[act] += 1
        else:
            if triggered:
                fa_sessions += 1
                fa_by_act[act] = fa_by_act.get(act, 0) + 1
                
    rec = (det_falls / total_fall_sessions) if total_fall_sessions > 0 else 0.0
    prec = (det_falls / (det_falls + fa_sessions)) if (det_falls + fa_sessions) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = (fa_sessions / total_adl_sessions) if total_adl_sessions > 0 else 0.0
    specificity = 1.0 - fpr
    
    print(f"=== {device.upper()} ({split.upper()}) RESULTS ===")
    print(f"  Fall Recall:      {rec*100:6.2f}% ({det_falls}/{total_fall_sessions} sessions detected)")
    print(f"  Fall Precision:   {prec*100:6.2f}%")
    print(f"  Binary F1 Score:  {f1*100:6.2f}%")
    print(f"  Specificity:      {specificity*100:6.2f}%")
    print(f"  False Alarm Rate: {fpr*100:6.2f}% ({fa_sessions}/{total_adl_sessions} ADL sessions)")
    print(f"  False Alarms by Activity: {fa_by_act}")
    print(f"  Recall by Fall Type:")
    for ft, cnt in det_by_type.items():
        tot = total_by_type.get(ft, 0)
        pct = (cnt / tot * 100) if tot > 0 else 0.0
        print(f"    {ft:<20}: {cnt}/{tot} ({pct:5.1f}%)")
        
    return {
        "device": device, "split": split, "recall": rec, "precision": prec,
        "f1": f1, "specificity": specificity, "fpr": fpr, "det_falls": det_falls,
        "total_falls": total_fall_sessions, "fa_sessions": fa_sessions,
        "total_adl": total_adl_sessions, "det_by_type": det_by_type,
        "total_by_type": total_by_type, "fa_by_act": fa_by_act
    }

print("RUNNING VALIDATION EVALUATION:")
p_val = evaluate_calibrated_state_machine("phone", "validation")
print()
w_val = evaluate_calibrated_state_machine("watch", "validation")
