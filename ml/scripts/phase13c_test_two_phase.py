import os, sys, json
import numpy as np
import pandas as pd

sys.path.append('ml/scripts')
from phase13c_grid_search import prepare_dataset

def eval_two_phase_state_machine(device, ap_thresh, p_thresh, max_post_astd=3.0, max_post_gpeak=2.5):
    df = prepare_dataset(device, "validation")
    
    total_fall_sessions = df[df['is_fall']]['source_session_id'].nunique()
    total_adl_sessions = df[~df['is_fall']]['source_session_id'].nunique()
    
    det_falls = 0
    det_by_type = {'FALL_BACKWARD': 0, 'FALL_FORWARD': 0, 'FALL_FROM_SITTING': 0, 'FALL_LEFT': 0, 'FALL_RIGHT': 0}
    fa_sessions = 0
    fa_by_act = {}
    
    for sid, s_df in df.groupby('source_session_id'):
        s_df = s_df.sort_values('window_index')
        is_fall = s_df['is_fall'].iloc[0]
        act = s_df['target_activity'].iloc[0]
        
        # State machine states:
        # 0: MONITORING
        # 1: IMPACT_ARMED (waiting for landing/stillness confirmation within next 1-2 windows)
        state = 0
        impact_window_idx = -1
        triggered = False
        
        for _, row in s_df.iterrows():
            w_idx = int(row['window_index'])
            fp = row['fall_prob']
            ap = row['acc_peak']
            ar = row['acc_range']
            gp = row['gyro_peak']
            astd = row['acc_std']
            
            if state == 0:
                # Stage 1: Fall Impact Shock Event
                # Requires severe deceleration shock AND elevated fall probability
                is_impact = (ap >= ap_thresh) or (ar >= (ap_thresh * 0.7) and gp >= 3.0)
                if is_impact and fp >= p_thresh:
                    state = 1
                    impact_window_idx = w_idx
            elif state == 1:
                # Stage 2: Post-Impact Landing Confirmation (within 1-2 windows after impact)
                # If the person continues running/jumping, dynamic thrashing continues.
                # If the person fell, they have landed and motion settles, or ML confirms fall posture.
                time_since_impact = w_idx - impact_window_idx
                
                if time_since_impact in [1, 2]:
                    # Check that the person is NOT actively running / jumping
                    # In running, astd > 4.0 and gp > 3.0 continuously
                    is_active_running_jumping = (astd > max_post_astd and gp > max_post_gpeak)
                    
                    if not is_active_running_jumping and fp >= p_thresh:
                        triggered = True
                        break
                elif time_since_impact > 2:
                    # Timeout: impact was not confirmed by landing/fall posture
                    state = 0
                    
        if is_fall:
            if triggered:
                det_falls += 1
                if act in det_by_type:
                    det_by_type[act] += 1
        else:
            if triggered:
                fa_sessions += 1
                fa_by_act[act] = fa_by_act.get(act, 0) + 1
                
    rec = det_falls / total_fall_sessions
    prec = det_falls / (det_falls + fa_sessions) if (det_falls + fa_sessions) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fa_sessions / total_adl_sessions
    
    return {
        "device": device,
        "ap_thresh": ap_thresh,
        "p_thresh": p_thresh,
        "max_post_astd": max_post_astd,
        "max_post_gpeak": max_post_gpeak,
        "recall": rec,
        "precision": prec,
        "f1": f1,
        "fpr": fpr,
        "det_falls": det_falls,
        "total_falls": total_fall_sessions,
        "fa_sessions": fa_sessions,
        "total_adl": total_adl_sessions,
        "det_by_type": det_by_type,
        "fa_by_act": fa_by_act
    }

print("=== TESTING TWO-PHASE CONFIRMATION ON PHONE ===")
for ap in [18.0, 20.0, 22.0, 24.0]:
    for pth in [0.40, 0.45, 0.50]:
        for astd in [3.0, 3.5, 4.0]:
            r = eval_two_phase_state_machine("phone", ap, pth, max_post_astd=astd, max_post_gpeak=2.5)
            if r['recall'] >= 0.90:
                print(f"ap={ap}, p={pth}, astd={astd} -> Rec={r['recall']*100:5.1f}%, Prec={r['precision']*100:5.1f}%, F1={r['f1']*100:5.1f}%, FA={r['fa_sessions']}/{r['total_adl']} | FAs={r['fa_by_act']}")

print("\n=== TESTING TWO-PHASE CONFIRMATION ON WATCH ===")
for ap in [18.0, 20.0, 22.0, 25.0]:
    for pth in [0.45, 0.50, 0.55]:
        for astd in [3.0, 3.5, 4.0]:
            r = eval_two_phase_state_machine("watch", ap, pth, max_post_astd=astd, max_post_gpeak=3.0)
            if r['recall'] >= 0.90:
                print(f"ap={ap}, p={pth}, astd={astd} -> Rec={r['recall']*100:5.1f}%, Prec={r['precision']*100:5.1f}%, F1={r['f1']*100:5.1f}%, FA={r['fa_sessions']}/{r['total_adl']} | FAs={r['fa_by_act']}")
