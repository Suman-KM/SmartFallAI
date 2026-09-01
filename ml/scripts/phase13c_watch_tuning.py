import os, sys, json
import numpy as np
import pandas as pd

sys.path.append('ml/scripts')
from phase13c_test_exact_rule import prepare_dataset

df = prepare_dataset("watch", "validation")
total_fall_sessions = df[df['is_fall']]['source_session_id'].nunique()
total_adl_sessions = df[~df['is_fall']]['source_session_id'].nunique()

print(f"Watch Validation: {total_fall_sessions} falls, {total_adl_sessions} ADL sessions")

for pth in [0.35, 0.38, 0.40, 0.42, 0.45, 0.50]:
    for ap in [18.0, 20.0, 22.0]:
        for mem in [3, 4]:
            det_falls = 0
            det_by_type = {'FALL_BACKWARD': 0, 'FALL_FORWARD': 0, 'FALL_FROM_SITTING': 0, 'FALL_LEFT': 0, 'FALL_RIGHT': 0}
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
                    ap_val = row['acc_peak']
                    ar_val = row['acc_range']
                    gp_val = row['gyro_peak']
                    astd = row['acc_std']
                    
                    # Impact detection
                    has_impact = (ap_val >= ap) or (ar_val >= 14.0 and gp_val >= 3.0)
                    if has_impact:
                        recent_impact_countdown = mem
                    elif recent_impact_countdown > 0:
                        recent_impact_countdown -= 1
                        
                    # Rejection of continuous jumping (jumping has continuous astd > 12.0 and gp > 4.0)
                    is_active_jumping = (astd >= 11.0) and (gp_val >= 4.0)
                    
                    is_cand = (fp >= pth) and (has_impact or recent_impact_countdown > 0) and (not is_active_jumping)
                    
                    if is_cand:
                        consecutive_candidates += 1
                        if consecutive_candidates >= 2:
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
                        
            rec = det_falls / total_fall_sessions
            prec = det_falls / (det_falls + fa_sessions) if (det_falls + fa_sessions) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if rec >= 0.80 and fa_sessions <= 5:
                print(f"p={pth:4.2f}, ap={ap:4.1f}, mem={mem} -> Rec={rec*100:5.1f}% ({det_falls}/10), Prec={prec*100:5.1f}%, F1={f1*100:5.1f}%, FA={fa_sessions}/25 | FAs={fa_by_act}, ByType={det_by_type}")
