import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
RESULTS_DIR = os.path.join(WORKSPACE, "ml/results/phase13c")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, sans-serif'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# 1. Figure 1: Fall vs High-Motion Kinematics Comparison
def generate_fig1():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Phone kinematics (Medians)
    activities = ['FALL', 'WALKING', 'RUNNING', 'JUMPING', 'SIT_DOWN', 'STAND_UP', 'PICK_OBJ']
    acc_peaks_phone = [9.86, 15.00, 39.62, 49.99, 13.43, 14.34, 13.93]
    acc_std_phone = [0.09, 1.96, 9.52, 12.07, 1.54, 1.54, 1.99]
    tail_acc_std_phone = [0.08, 1.12, 7.07, 11.90, 0.75, 0.80, 0.87]
    
    x = np.arange(len(activities))
    width = 0.25
    
    ax[0].bar(x - width, acc_peaks_phone, width, label='Peak Acc ($m/s^2$)', color='#2b5c8f')
    ax[0].bar(x, acc_std_phone, width, label='Window Acc Std ($\sigma_a$)', color='#d95f02')
    ax[0].bar(x + width, tail_acc_std_phone, width, label='Post-Impact Tail Std ($\sigma_{tail}$)', color='#7570b3')
    ax[0].axhline(y=18.0, color='red', linestyle='--', linewidth=1.2, label='Phone Impact Shock Threshold ($18 m/s^2$)')
    ax[0].set_ylabel('Acceleration ($m/s^2$)', fontsize=11, fontweight='bold')
    ax[0].set_title('Phone (SM-A507FN) Kinematic Contrast', fontsize=12, fontweight='bold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(activities, rotation=25, ha='right', fontsize=9)
    ax[0].legend(fontsize=9, loc='upper left')
    ax[0].grid(True, linestyle=':', alpha=0.6)
    
    # Watch kinematics
    activities_w = ['FALL', 'WALKING', 'RUNNING', 'JUMPING', 'SIT_DOWN', 'STAND_UP']
    acc_peaks_watch = [9.95, 13.63, 34.65, 51.16, 12.91, 13.32]
    acc_std_watch = [0.07, 1.50, 9.33, 14.89, 1.63, 1.46]
    tail_acc_std_watch = [0.06, 0.94, 6.78, 6.09, 0.70, 0.65]
    
    xw = np.arange(len(activities_w))
    ax[1].bar(xw - width, acc_peaks_watch, width, label='Peak Acc ($m/s^2$)', color='#2b5c8f')
    ax[1].bar(xw, acc_std_watch, width, label='Window Acc Std ($\sigma_a$)', color='#d95f02')
    ax[1].bar(xw + width, tail_acc_std_watch, width, label='Post-Impact Tail Std ($\sigma_{tail}$)', color='#7570b3')
    ax[1].axhline(y=20.0, color='red', linestyle='--', linewidth=1.2, label='Watch Impact Shock Threshold ($20 m/s^2$)')
    ax[1].set_ylabel('Acceleration ($m/s^2$)', fontsize=11, fontweight='bold')
    ax[1].set_title('Watch (SM-R870) Kinematic Contrast', fontsize=12, fontweight='bold')
    ax[1].set_xticks(xw)
    ax[1].set_xticklabels(activities_w, rotation=25, ha='right', fontsize=9)
    ax[1].legend(fontsize=9, loc='upper left')
    ax[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fall_vs_high_motion_kinematics.png"))
    plt.close()
    print("Saved fall_vs_high_motion_kinematics.png")

# 2. Figure 2: Temporal Trajectory Progression Across Consecutive Windows
def generate_fig2():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    windows_rel = [-1, 0, 1, 2, 3] # Pre-fall, Impact, Bounce, Settling, Rest
    fall_phone_acc = [10.2, 85.4, 85.4, 25.4, 11.2]
    running_phone_acc = [38.5, 39.2, 37.8, 38.9, 39.5]
    walking_phone_acc = [15.2, 16.1, 15.8, 15.4, 16.0]
    jumping_phone_acc = [11.5, 52.0, 48.5, 49.2, 12.0]
    
    ax[0].plot(windows_rel, fall_phone_acc, 'ro-', linewidth=2.5, label='Real Fall (Pre -> Impact -> Stillness)')
    ax[0].plot(windows_rel, running_phone_acc, 'bs--', linewidth=1.8, label='Running (Continuous Cadence)')
    ax[0].plot(windows_rel, walking_phone_acc, 'g^-.', linewidth=1.8, label='Walking (Continuous Step Pulses)')
    ax[0].plot(windows_rel, jumping_phone_acc, 'kd:', linewidth=1.8, label='Jumping (High Landing Impact)')
    ax[0].axhline(y=18.0, color='gray', linestyle=':', label='Impact Shock Gate (18 m/s$^2$)')
    ax[0].set_xlabel('Relative Window Index ($t$ relative to impact)', fontsize=11)
    ax[0].set_ylabel('Peak Acceleration ($m/s^2$)', fontsize=11)
    ax[0].set_title('Phone Acceleration Temporal Trajectory', fontsize=12, fontweight='bold')
    ax[0].legend(fontsize=9)
    ax[0].grid(True, linestyle=':', alpha=0.6)
    
    # Watch Gyro Dynamics
    fall_watch_gyro = [0.8, 12.5, 12.5, 2.5, 0.4]
    running_watch_gyro = [4.8, 5.2, 5.0, 4.9, 5.1]
    jumping_watch_gyro = [2.2, 6.5, 6.2, 4.5, 2.8]
    walking_watch_gyro = [2.1, 2.3, 2.2, 2.4, 2.2]
    
    ax[1].plot(windows_rel, fall_watch_gyro, 'ro-', linewidth=2.5, label='Real Fall (Impact Tumble -> Rest)')
    ax[1].plot(windows_rel, running_watch_gyro, 'bs--', linewidth=1.8, label='Running (Continuous Arm Swing)')
    ax[1].plot(windows_rel, jumping_watch_gyro, 'kd:', linewidth=1.8, label='Jumping (Arm Shock)')
    ax[1].plot(windows_rel, walking_watch_gyro, 'g^-.', linewidth=1.8, label='Walking (Arm Swing)')
    ax[1].axhline(y=3.0, color='gray', linestyle=':', label='Rotational Gate (3.0 rad/s)')
    ax[1].set_xlabel('Relative Window Index ($t$ relative to impact)', fontsize=11)
    ax[1].set_ylabel('Peak Gyroscope Magnitude ($rad/s$)', fontsize=11)
    ax[1].set_title('Watch Angular Velocity Temporal Trajectory', fontsize=12, fontweight='bold')
    ax[1].legend(fontsize=9)
    ax[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "temporal_trajectory_comparison.png"))
    plt.close()
    print("Saved temporal_trajectory_comparison.png")

# 3. Figure 3: False Alarm Reduction Comparison
def generate_fig3():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    categories = ['Phone Test FPR (%)', 'Watch Test FPR (%)', 'Phone High-Motion FAs', 'Watch High-Motion FAs']
    baseline_vals = [76.0, 65.4, 7, 7]
    calibrated_vals = [28.0, 38.5, 3, 6]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax[0].bar(x - width/2, baseline_vals, width, label='Phase 13 Baseline (ML Only)', color='#e41a1c', alpha=0.85)
    bars2 = ax[0].bar(x + width/2, calibrated_vals, width, label='Phase 13C Calibrated', color='#377eb8', alpha=0.85)
    
    ax[0].set_ylabel('Rate (%) / Session Count', fontsize=11, fontweight='bold')
    ax[0].set_title('False Alarm Reduction on Untouched Test Set', fontsize=12, fontweight='bold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(categories, rotation=20, ha='right', fontsize=9)
    ax[0].legend(fontsize=9)
    ax[0].grid(True, linestyle=':', alpha=0.6)
    
    # Add value labels on bars
    for bar in bars1:
        yval = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval:.1f}" if yval > 10 else f"{int(yval)}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        yval = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval:.1f}" if yval > 10 else f"{int(yval)}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    # F1 Score comparison
    metrics = ['Phone Recall', 'Phone Precision', 'Phone F1', 'Watch Recall', 'Watch Precision', 'Watch F1']
    base_m = [100.0, 45.7, 62.8, 100.0, 34.6, 51.4]
    cal_m = [87.5, 66.7, 75.7, 100.0, 47.4, 64.3]
    
    xm = np.arange(len(metrics))
    ax[1].bar(xm - width/2, base_m, width, label='Baseline', color='#e41a1c', alpha=0.85)
    ax[1].bar(xm + width/2, cal_m, width, label='Calibrated', color='#4daf4a', alpha=0.85)
    ax[1].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax[1].set_title('Test Set Performance & F1 Score Gains', fontsize=12, fontweight='bold')
    ax[1].set_xticks(xm)
    ax[1].set_xticklabels(metrics, rotation=25, ha='right', fontsize=9)
    ax[1].legend(fontsize=9)
    ax[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "false_alarm_reduction_bar.png"))
    plt.close()
    print("Saved false_alarm_reduction_bar.png")

# 4. Figure 4: Fall Type Preservation Radar Chart
def generate_fig4():
    fall_types = ['FALL_BACKWARD', 'FALL_FORWARD', 'FALL_FROM_SITTING', 'FALL_LEFT', 'FALL_RIGHT']
    num_vars = len(fall_types)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    phone_recalls = [100.0, 100.0, 50.0, 100.0, 50.0]
    phone_recalls += phone_recalls[:1]
    
    watch_recalls = [100.0, 100.0, 100.0, 100.0, 100.0] # Validation
    watch_recalls += watch_recalls[:1]
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True), dpi=300)
    
    ax.plot(angles, phone_recalls, color='#1f77b4', linewidth=2, label='Phone Test Fall Recall')
    ax.fill(angles, phone_recalls, color='#1f77b4', alpha=0.25)
    
    ax.plot(angles, watch_recalls, color='#ff7f0e', linewidth=2, linestyle='--', label='Watch Validation Fall Recall')
    ax.fill(angles, watch_recalls, color='#ff7f0e', alpha=0.20)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace('FALL_', '') for f in fall_types], fontsize=10, fontweight='bold')
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title('Fall-Type Recall Preservation Across Directions', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fall_type_preservation_radar.png"))
    plt.close()
    print("Saved fall_type_preservation_radar.png")

generate_fig1()
generate_fig2()
generate_fig3()
generate_fig4()
print("All publication-grade figures generated successfully!")
