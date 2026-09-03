#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import json
import glob
import re

PHONE_DEVICE = "adb-RZ8N11FMBKB-ubDAVA._adb-tls-connect._tcp"
WATCH_DEVICE = "adb-RFAW3061E6M-V3FTAH._adb-tls-connect._tcp"

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"
BASE_OUT = os.path.join(WORKSPACE, "ml/results/phase13e")

TEST_SUITE = [
    {
        "id": "TEST_A",
        "name": "STATIONARY / RESTING",
        "dir_name": "stationary",
        "target_activity": "STATIONARY",
        "is_fall": False,
        "duration": 20,
        "instruction": "Place the Phone and Watch flat on a desk/table. Keep them completely stationary.",
    },
    {
        "id": "TEST_B",
        "name": "NORMAL WALKING",
        "dir_name": "walking",
        "target_activity": "WALKING",
        "is_fall": False,
        "duration": 25,
        "instruction": "Place the Phone in your front pocket, wear the Watch on your wrist, and walk at your normal comfortable pace.",
    },
    {
        "id": "TEST_C",
        "name": "BRISK WALKING",
        "dir_name": "brisk_walking",
        "target_activity": "BRISK_WALKING",
        "is_fall": False,
        "duration": 20,
        "instruction": "Walk briskly/fast across the room with natural arm swing.",
    },
    {
        "id": "TEST_D",
        "name": "RUNNING",
        "dir_name": "running",
        "target_activity": "RUNNING",
        "is_fall": False,
        "duration": 20,
        "instruction": "Jog or run in place at a vigorous, steady cadence.",
    },
    {
        "id": "TEST_E",
        "name": "JUMPING",
        "dir_name": "jumping",
        "target_activity": "JUMPING",
        "is_fall": False,
        "duration": 15,
        "instruction": "Perform 5 to 8 vertical jumps, landing firmly on both feet.",
    },
    {
        "id": "TEST_F",
        "name": "SIT DOWN",
        "dir_name": "sit_down",
        "target_activity": "SIT_DOWN",
        "is_fall": False,
        "duration": 15,
        "instruction": "From standing, sit down firmly into a chair 3 to 4 times.",
    },
    {
        "id": "TEST_G",
        "name": "STAND UP",
        "dir_name": "stand_up",
        "target_activity": "STAND_UP",
        "is_fall": False,
        "duration": 15,
        "instruction": "From sitting, stand up smoothly 3 to 4 times.",
    },
    {
        "id": "TEST_H",
        "name": "PICKING UP OBJECT",
        "dir_name": "picking_object",
        "target_activity": "PICKING_OBJECT",
        "is_fall": False,
        "duration": 15,
        "instruction": "Bend down from standing, pick up an object from the floor, and stand back up.",
    },
    {
        "id": "TEST_I",
        "name": "NORMAL PHONE HANDLING / REORIENTATION",
        "dir_name": "handling",
        "target_activity": "HANDLING",
        "is_fall": False,
        "duration": 20,
        "instruction": "Pick up the phone, rotate it 360 degrees, tilt portrait to landscape, and place it down on the desk.",
    },
    {
        "id": "TEST_J",
        "name": "CONTROLLED FALL SIMULATION",
        "dir_name": "controlled_fall",
        "target_activity": "CONTROLLED_FALL",
        "is_fall": True,
        "duration": 20,
        "instruction": "SAFETY WARNING: Use a soft bed, mattress, or thick mat. Perform a controlled, safe simulated fall onto the mattress, and remain completely still on the mattress for 5 seconds.",
    }
]

def adb_cmd(device, cmd):
    full_cmd = ["adb", "-s", device] + cmd
    res = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.stdout.strip()

def tap_phone_start_stop():
    # Tap "START/STOP RECORDING" button on Phone (bounds [363,1016][718,1075])
    adb_cmd(PHONE_DEVICE, ["shell", "input", "tap", "540", "1045"])

def tap_watch_start_stop():
    # Tap "START/STOP" button on Watch (bounds [42,61][408,172])
    adb_cmd(WATCH_DEVICE, ["shell", "input", "tap", "225", "116"])

def run_test(test_idx, test_info):
    test_id = test_info["id"]
    name = test_info["name"]
    duration = test_info["duration"]
    instruction = test_info["instruction"]
    is_fall = test_info["is_fall"]
    dir_name = test_info["dir_name"]
    
    print("\n" + "="*70)
    print(f"TEST {test_idx + 1}/{len(TEST_SUITE)} — {name}")
    print("="*70)
    print(f"Instruction:\n\"{instruction}\"\n")
    print(f"Duration: {duration} seconds")
    print(f"Device Targets: Phone ({PHONE_DEVICE}) & Watch ({WATCH_DEVICE})")
    print("\nPress ENTER when you are ready to begin.")
    
    # MANUAL ENTER START GATE: MUST WAIT FOR USER INPUT!
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("\nTest execution interrupted.")
        return None
        
    print(f"\n>>> STARTING TEST: {name} <<<")
    
    # 1. Clear logcats
    adb_cmd(PHONE_DEVICE, ["logcat", "-c"])
    adb_cmd(WATCH_DEVICE, ["logcat", "-c"])
    
    # 2. Trigger recording on both devices
    tap_phone_start_stop()
    tap_watch_start_stop()
    print("Recording started on Phone & Watch. Executing movement...")
    
    # 3. Wait for duration while showing countdown
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = int(time.time() - start_time)
        rem = duration - elapsed
        sys.stdout.write(f"\rTime Remaining: {rem:02d}s | Elapsed: {elapsed:02d}s ... ")
        sys.stdout.flush()
        time.sleep(1.0)
    print("\nTest duration reached. Stopping recording...")
    
    # 4. Stop recording on both devices
    tap_phone_start_stop()
    tap_watch_start_stop()
    time.sleep(2.0) # wait for flush
    
    # 5. Capture logcat output
    phone_logs = adb_cmd(PHONE_DEVICE, ["logcat", "-d", "-s", "PhoneFallML"])
    watch_logs = adb_cmd(WATCH_DEVICE, ["logcat", "-d", "-s", "WatchFallML"])
    
    # 6. Parse diagnostics
    def parse_log(logs, dev_tag):
        lines = logs.split("\n")
        parsed = []
        max_prob = 0.0
        max_acc = 0.0
        max_jerk = 0.0
        countdown_triggered = False
        states_seen = set()
        
        for l in lines:
            if "Activity=" in l:
                # Activity=..., FallProb=..., AccPeak=..., AccMin=..., AccRange=..., AccStd=..., GyroPeak=..., JerkPeak=..., Impact=..., TemporalScore=..., PostImpact=..., ActiveMotion=..., State=..., Latency=...
                m_act = re.search(r"Activity=([A-Z_]+)", l)
                m_prob = re.search(r"FallProb=([0-9\.]+)", l)
                m_acc = re.search(r"AccPeak=([0-9\.]+)", l)
                m_jerk = re.search(r"JerkPeak=([0-9\.]+)", l)
                m_state = re.search(r"State=([A-Z_]+)", l)
                
                act = m_act.group(1) if m_act else "UNKNOWN"
                prob = float(m_prob.group(1)) if m_prob else 0.0
                acc = float(m_acc.group(1)) if m_acc else 0.0
                jerk = float(m_jerk.group(1)) if m_jerk else 0.0
                st = m_state.group(1) if m_state else "UNKNOWN"
                
                if prob > max_prob: max_prob = prob
                if acc > max_acc: max_acc = acc
                if jerk > max_jerk: max_jerk = jerk
                states_seen.add(st)
                
                if st in ["FALL_SUSPECTED", "FALL_CONFIRMED", "SOS_TRIGGERED"]:
                    countdown_triggered = True
                    
                parsed.append({
                    "raw": l,
                    "activity": act,
                    "prob": prob,
                    "acc": acc,
                    "jerk": jerk,
                    "state": st
                })
                
            if "Starting 10-second emergency countdown" in l:
                countdown_triggered = True
                
        return {
            "parsed_lines": parsed,
            "max_prob": max_prob,
            "max_acc": max_acc,
            "max_jerk": max_jerk,
            "countdown_triggered": countdown_triggered,
            "states_seen": list(states_seen),
            "raw_log": logs
        }
        
    phone_res = parse_log(phone_logs, "PhoneFallML")
    watch_res = parse_log(watch_logs, "WatchFallML")
    
    # 7. Save outputs
    phone_dir = os.path.join(BASE_OUT, "physical/phone", dir_name)
    watch_dir = os.path.join(BASE_OUT, "physical/watch", dir_name)
    os.makedirs(phone_dir, exist_ok=True)
    os.makedirs(watch_dir, exist_ok=True)
    
    with open(os.path.join(phone_dir, "trace.json"), "w") as f:
        json.dump(phone_res, f, indent=2)
    with open(os.path.join(phone_dir, "logcat.txt"), "w") as f:
        f.write(phone_logs)
        
    with open(os.path.join(watch_dir, "trace.json"), "w") as f:
        json.dump(watch_res, f, indent=2)
    with open(os.path.join(watch_dir, "logcat.txt"), "w") as f:
        f.write(watch_logs)
        
    # Check result
    if is_fall:
        phone_pass = phone_res["countdown_triggered"]
        watch_pass = watch_res["countdown_triggered"]
    else:
        phone_pass = not phone_res["countdown_triggered"]
        watch_pass = not watch_res["countdown_triggered"]
        
    print("\n" + "-"*50)
    print("TEST COMPLETE — RESULTS SUMMARY")
    print("-"*50)
    print(f"PHONE ({PHONE_DEVICE}):")
    print(f"  Countdown Triggered: {'YES' if phone_res['countdown_triggered'] else 'NO'}")
    print(f"  False Alarm:         {'YES' if (not is_fall and phone_res['countdown_triggered']) else 'NO'}")
    print(f"  Fall Detected:       {'YES' if (is_fall and phone_res['countdown_triggered']) else 'NO'}")
    print(f"  Max Fall Probability:{phone_res['max_prob']:.4f}")
    print(f"  Max Acc Peak:        {phone_res['max_acc']:.2f} m/s^2")
    print(f"  Max Jerk Peak:       {phone_res['max_jerk']:.1f} m/s^3")
    print(f"  States Visited:      {phone_res['states_seen']}")
    print(f"  Status:              {'PASS' if phone_pass else 'FAIL'}")
    print()
    print(f"WATCH ({WATCH_DEVICE}):")
    print(f"  Countdown Triggered: {'YES' if watch_res['countdown_triggered'] else 'NO'}")
    print(f"  False Alarm:         {'YES' if (not is_fall and watch_res['countdown_triggered']) else 'NO'}")
    print(f"  Fall Detected:       {'YES' if (is_fall and watch_res['countdown_triggered']) else 'NO'}")
    print(f"  Max Fall Probability:{watch_res['max_prob']:.4f}")
    print(f"  Max Acc Peak:        {watch_res['max_acc']:.2f} m/s^2")
    print(f"  Max Jerk Peak:       {watch_res['max_jerk']:.1f} m/s^3")
    print(f"  States Visited:      {watch_res['states_seen']}")
    print(f"  Status:              {'PASS' if watch_pass else 'FAIL'}")
    print("-" * 50)
    print(f"Saved traces to:\n  {phone_dir}/\n  {watch_dir}/")
    print("\nPress ENTER to continue to next test.")
    
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
        
    return {
        "test_id": test_id,
        "name": name,
        "phone_pass": phone_pass,
        "watch_pass": watch_pass,
        "phone_res": phone_res,
        "watch_res": watch_res
    }

def main():
    print("="*70)
    print("SMARTFALL AI — PHASE 13E PHYSICAL TEST CONTROLLER")
    print("="*70)
    print(f"Target Devices:\n  Phone: {PHONE_DEVICE}\n  Watch: {WATCH_DEVICE}")
    print(f"Suite: {len(TEST_SUITE)} Tests with Manual ENTER Gate")
    
    results = []
    for idx, test_info in enumerate(TEST_SUITE):
        res = run_test(idx, test_info)
        if res is None:
            break
        results.append(res)
        
    summary_path = os.path.join(BASE_OUT, "physical_suite_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE! SUMMARY TABLE:")
    print("="*70)
    print(f"{'Test Name':<35} | {'Phone':<10} | {'Watch':<10}")
    print("-" * 60)
    for r in results:
        p_str = "PASS" if r["phone_pass"] else "FAIL"
        w_str = "PASS" if r["watch_pass"] else "FAIL"
        print(f"{r['name']:<35} | {p_str:<10} | {w_str:<10}")
    print("="*70)

if __name__ == "__main__":
    main()
