import os, sys, json
import numpy as np
import pandas as pd

sys.path.append('ml/scripts')
from phase13c_grid_search import prepare_dataset

def test_settling_rule(device, ap_thresh, p_thresh, max_continuous_astd=3.0, max_continuous_gpeak=3.0):
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
        
        # State machine tracking:
        recent_impact_countdown = 0
        consecutive_candidates = 0
        triggered = False
        
        for _, row in s_df.iterrows():
            fp = row['fall_prob']
            ap = row['acc_peak']
            ar = row['acc_range']
            gp = row['gyro_peak']
            astd = row['acc_std']
            
            # 1. Detect dynamic impact
            # Real fall impacts have sharp deceleration
            has_impact = (ap >= ap_thresh) or (ar >= (ap_thresh * 0.6) and gp >= 3.0)
            if has_impact:
                recent_impact_countdown = 3 # remember impact for up to 3 sliding windows (~3s)
            elif recent_impact_countdown > 0:
                recent_impact_countdown -= 1
                
            # 2. Check active non-fall motion (continuous running/jumping thrashing)
            # In running/jumping, astd > 3.0 AND gp > 3.0 continuously
            is_active_thrashing = (astd >= max_continuous_astd) and (gp >= max_continuous_gpeak)
            
            # 3. Fall candidate condition:
            # - Model indicates fall posture (fp >= p_thresh)
            # - Recent physical impact took place (has_impact or recent_impact_countdown > 0)
            # - NOT actively running/jumping (not is_active_thrashing)
            is_candidate = (fp >= p_thresh) and (has_impact or recent_impact_countdown > 0) and (not is_active_thrashing)
            
            if is_candidate:
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
    fpr = fa_sessions / total_adl_sessions
    
    return {
        "device": device,
        "ap_thresh": ap_thresh,
        "p_thresh": p_thresh,
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

print("=== SWEEPING CONTINUOUS THRASHING FILTER ON PHONE ===")
for ap in [18.0, 20.0, 22.0, 24.0]:
    for pth in [0.40, 0.45, 0.50]:
        for astd in [2.5, 3.0, 3.5]:
            for gpk in [2.5, 3.0, 3.5]:
                r = test_settling_rule("phone", ap, pth, max_continuous_astd=astd, max_continuous_gpeak=gpk)
                if r['recall'] >= 0.93 and r['fa_sessions'] <= 3:
                    print(f"PHONE ap={ap}, p={pth}, astd={astd}, gpk={gpk} -> Rec={r['recall']*100:5.1f}%, Prec={r['precision']*100:5.1f}%, F1={r['f1']*100:5.1f}%, FA={r['fa_sessions']}/{r['total_adl']} | FAs={r['fa_by_act']}")

print("\n=== SWEEPING CONTINUOUS THRASHING FILTER ON WATCH ===")
for ap in [18.0, 20.0, 22.0, 25.0]:
    for pth in [0.45, 0.50, 0.55]:
        for astd in [2.5, 3.0, 3.5]:
            for gpk in [2.5, 3.0, 3.5]:
                r = test_settling_rule("watch", ap, pth, max_continuous_astd=astd, max_continuous_gpeak=gpk)
                if r['recall'] >= 0.90 and r['fa_sessions'] <= 3:
                    print(f"WATCH ap={ap}, p={pth}, astd={astd}, gpk={gpk} -> Rec={r['recall']*100:5.1f}%, Prec={r['precision']*100:5.1f}%, F1={r['f1']*100:5.1f}%, FA={r['fa_sessions']}/{r['total_adl']} | FAs={r['fa_by_act']}")
