import os, sys, json
import numpy as np
import pandas as pd

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
sys.path.append(os.path.join(WORKSPACE, "ml/scripts"))
from phase13c_test_set_evaluation import load_split_data

w_test_df = load_split_data("watch", "test")
w_adl = w_test_df[~w_test_df['is_fall']]

# Calibrated Phase 13C rule
w_ap_th = 20.0
w_ar_th = 12.0
w_gp_th = 3.0
w_p_th = 0.45
w_astd_th = 9.0
w_gthrash_th = 4.0

triggered_sessions = []
for sid in w_adl['source_session_id'].unique():
    s_df = w_adl[w_adl['source_session_id'] == sid].sort_values('window_index')
    act = s_df['target_activity'].iloc[0]
    
    recent_mem = 0
    consec = 0
    trig = False
    trig_win = -1
    
    for _, row in s_df.iterrows():
        fp = row['fall_prob']
        ap = row['acc_peak']
        ar = row['acc_range']
        gp = row['gyro_peak']
        astd = row['acc_std']
        
        has_imp = (ap >= w_ap_th) or (ar >= w_ar_th and gp >= w_gp_th)
        if has_imp:
            recent_mem = 3
        elif recent_mem > 0:
            recent_mem -= 1
            
        is_thrash = (astd >= w_astd_th and gp >= w_gthrash_th)
        is_cand = (fp >= w_p_th) and (has_imp or recent_mem > 0) and (not is_thrash)
        
        if is_cand:
            consec += 1
            if consec >= 2:
                trig = True
                trig_win = int(row['window_index'])
                break
        else:
            consec = 0
            
    if trig:
        triggered_sessions.append((sid, act, trig_win))

print(f"Total Watch ADL Test Sessions: {len(w_adl['source_session_id'].unique())}")
print(f"False Alarm Sessions ({len(triggered_sessions)}):")
for sid, act, twin in triggered_sessions:
    print(f"  {sid} | {act} | triggered at window {twin}")
    s_df = w_adl[w_adl['source_session_id'] == sid].sort_values('window_index')
    rows = s_df[s_df['window_index'].isin(range(max(0, twin - 2), min(len(s_df), twin + 3)))]
    cols = ['window_index', 'fall_prob', 'acc_peak', 'acc_min', 'acc_range', 'acc_std', 'gyro_peak']
    print(rows[cols].to_string(index=False))
    print()
