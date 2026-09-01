import os, sys, json
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
PREPROC_DIR = os.path.join(WORKSPACE, "preprocessing/02_robust_scaling")

sys.path.append(os.path.join(WORKSPACE, "ml/scripts"))
from phase13d_architecture_experiments import prepare_dataset

class Phase13DFallDetector:
    def __init__(self, device="phone"):
        self.device = device
        if device == "phone":
            self.imp_peak_th = 20.0
            self.imp_jerk_th = 350.0
            self.imp_range_th = 14.0
            self.imp_gyro_th = 3.5
            self.imp_jerk_alt_th = 250.0
            self.thrash_std_th = 3.2
            self.thrash_gyro_th = 3.5
            self.still_std_th = 2.4
            self.still_gyro_th = 2.2
            self.fall_prob_th = 0.40
            self.lying_prob_th = 0.45
        else: # watch
            self.imp_peak_th = 24.0
            self.imp_jerk_th = 500.0
            self.imp_range_th = 16.0
            self.imp_gyro_th = 4.0
            self.imp_jerk_alt_th = 350.0
            self.thrash_std_th = 5.5
            self.thrash_gyro_th = 4.0
            self.still_std_th = 3.8
            self.still_gyro_th = 3.2
            self.fall_prob_th = 0.40
            self.lying_prob_th = 0.45

    def evaluate_session(self, s_df):
        # Returns (triggered, trigger_window, reason)
        n_wins = len(s_df)
        
        # State machine variables
        # State: 0 = MONITORING, 1 = POTENTIAL_IMPACT, 2 = FALL_SUSPECTED
        state = 0
        impact_window_idx = -1
        impact_windows_countdown = 0
        
        for i in range(n_wins):
            r = s_df.iloc[i]
            
            # Check for sudden collision impact shock
            is_collision_shock = (r['acc_peak'] >= self.imp_peak_th and r['jerk_peak'] >= self.imp_jerk_th)
            is_rotational_tumble = (r['acc_range'] >= self.imp_range_th and r['jerk_peak'] >= self.imp_jerk_alt_th and r['gyro_peak'] >= self.imp_gyro_th)
            
            # Active continuous locomotion in this window?
            is_locomotion_cadence = (r['acc_std'] >= self.thrash_std_th) and (r['gyro_peak'] >= self.thrash_gyro_th)
            
            if is_collision_shock or is_rotational_tumble:
                # Stage 2: Abnormal impact collision detected
                state = 1
                impact_window_idx = i
                impact_windows_countdown = 5 # Verify within next 5 windows (~2.5 to 3.0 seconds)
                continue
                
            if state == 1:
                # Stage 3: Verification of trajectory
                if is_locomotion_cadence:
                    # Subject is actively running, jumping, or briskly walking!
                    # Locomotion cadence resumed -> DISCARD IMPACT CANDIDATE!
                    state = 0
                    impact_windows_countdown = 0
                    continue
                    
                # Stage 4: Check if settled into stillness / recumbency
                is_settled_immobility = (r['acc_std'] <= self.still_std_th) and (r['gyro_peak'] <= self.still_gyro_th)
                has_posture = (r['fall_prob'] >= self.fall_prob_th) or (r['lying_down_prob'] >= self.lying_prob_th and r['acc_std'] <= 1.8)
                
                # Check impact window posture as well
                imp_r = s_df.iloc[impact_window_idx]
                if imp_r['fall_prob'] >= self.fall_prob_th:
                    has_posture = True
                    
                if is_settled_immobility and has_posture:
                    # Stage 5: Confirmed fall trajectory (Collision -> Immobility)
                    return True, i, "Collision shock followed by post-impact immobility"
                    
                impact_windows_countdown -= 1
                if impact_windows_countdown <= 0:
                    # Timeout without immobility confirmation
                    state = 0
                    
        return False, -1, "No confirmed fall trajectory"

def run_evaluation(device, split):
    df = prepare_dataset(device, split)
    detector = Phase13DFallDetector(device)
    
    unique_sids = df['source_session_id'].unique()
    total_falls = 0
    detected_falls = 0
    total_adls = 0
    false_alarms = 0
    fa_by_act = {}
    recall_by_act = {}
    
    for sid in unique_sids:
        s_df = df[df['source_session_id'] == sid].sort_values('window_index').reset_index(drop=True)
        is_fall = s_df['is_fall'].iloc[0]
        act = s_df['target_activity'].iloc[0]
        
        trig, twin, reason = detector.evaluate_session(s_df)
        
        if is_fall:
            total_falls += 1
            if act not in recall_by_act: recall_by_act[act] = [0, 0]
            recall_by_act[act][1] += 1
            if trig:
                detected_falls += 1
                recall_by_act[act][0] += 1
        else:
            total_adls += 1
            if trig:
                false_alarms += 1
                fa_by_act[act] = fa_by_act.get(act, 0) + 1
                
    rec = (detected_falls / total_falls) * 100.0 if total_falls > 0 else 0.0
    prec = (detected_falls / (detected_falls + false_alarms)) * 100.0 if (detected_falls + false_alarms) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    spec = ((total_adls - false_alarms) / total_adls) * 100.0 if total_adls > 0 else 0.0
    fpr = (false_alarms / total_adls) * 100.0 if total_adls > 0 else 0.0
    
    print(f"==================================================")
    print(f"{device.upper()} — {split.upper()} SET RESULTS")
    print(f"==================================================")
    print(f"Fall Recall:     {rec:6.2f}% ({detected_falls}/{total_falls})")
    print(f"Fall Precision:  {prec:6.2f}%")
    print(f"Binary Fall F1:  {f1:6.2f}%")
    print(f"Specificity:     {spec:6.2f}%")
    print(f"False Alarm Rate:{fpr:6.2f}% ({false_alarms}/{total_adls})")
    print(f"False Alarms by Activity: {fa_by_act}")
    print(f"Recall by Fall Type:")
    for k, v in recall_by_act.items():
        pct = (v[0]/v[1])*100.0 if v[1]>0 else 0.0
        print(f"  {k:<20}: {v[0]}/{v[1]} ({pct:.1f}%)")
    print()
    return rec, prec, f1, spec, fpr, fa_by_act, recall_by_act

print("\n>>> VALIDATION SET EVALUATION <<<")
p_val_res = run_evaluation("phone", "validation")
w_val_res = run_evaluation("watch", "validation")

print("\n>>> UNTOUCHED TEST SET EVALUATION <<<")
p_test_res = run_evaluation("phone", "test")
w_test_res = run_evaluation("watch", "test")
