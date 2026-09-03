import os
import sys
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib

WORKSPACE = "/Users/suman/AndroidStudioProjects/MobileSensorLogger backup"

class Phase13EReplayEngine:
    def __init__(self, device="phone"):
        self.device = device.lower()
        
        # Load scalers
        scaler_path = os.path.join(WORKSPACE, f"preprocessing/02_robust_scaling/{self.device}/scaler.json")
        with open(scaler_path) as f:
            s_data = json.load(f)
        self.medians = np.array(s_data["median"], dtype=np.float32)
        self.iqrs = np.array(s_data["iqr"], dtype=np.float32)
        
        if self.device == "phone":
            model_path = os.path.join(WORKSPACE, "app/src/main/assets/model.onnx")
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            
            # Calibration parameters (Phone)
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
            self.verification_horizon = 4
        else:
            model_path = os.path.join(WORKSPACE, "ml/models/watch_rf.joblib")
            if os.path.exists(model_path):
                self.rf_model = joblib.load(model_path)
            else:
                self.rf_model = None
                
            # Calibration parameters (Watch)
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
            self.verification_horizon = 4

    def extract_watch_features(self, window):
        # 72 statistical features: mean, std, min, max, q25, q50, q75, iqr for 9 channels
        feats = []
        for ch in range(9):
            col = window[:, ch]
            mean = np.mean(col)
            std = np.std(col)
            cmin = np.min(col)
            cmax = np.max(col)
            q25 = np.percentile(col, 25)
            q50 = np.percentile(col, 50)
            q75 = np.percentile(col, 75)
            iqr = q75 - q25
            feats.extend([mean, std, cmin, cmax, q25, q50, q75, iqr])
        return np.array(feats, dtype=np.float32)

    def replay_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        # raw features: accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw
        # Find column indices
        col_names = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "pitch", "roll", "yaw"]
        if all(c in df.columns for c in col_names):
            raw_feats = df[col_names].values.astype(np.float32)
        else:
            raw_feats = df.iloc[:, 2:11].values.astype(np.float32)
            
        timestamps = df["timestamp"].values if "timestamp" in df.columns else df.iloc[:, 1].values
        
        n_samples = len(raw_feats)
        window_size = 100
        step_size = 50
        
        state = "MONITORING"
        recentImpactCountdown = 0
        
        results = []
        
        for start in range(0, n_samples - window_size + 1, step_size):
            end = start + window_size
            window = raw_feats[start:end].copy()
            w_ts_start = int(timestamps[start])
            w_ts_end = int(timestamps[end - 1])
            
            # Kinematics
            maxAccMag = 0.0
            minAccMag = float('inf')
            maxGyroMag = 0.0
            maxJerk = 0.0
            prevAccMag = -1.0
            sumAccMag = 0.0
            sumAccMagSq = 0.0
            validAccCount = 0
            
            for s in window:
                ax, ay, az = s[0], s[1], s[2]
                accMag = np.sqrt(ax*ax + ay*ay + az*az)
                if accMag > 1.0:
                    if accMag > maxAccMag: maxAccMag = accMag
                    if accMag < minAccMag: minAccMag = accMag
                    sumAccMag += accMag
                    sumAccMagSq += accMag * accMag
                    validAccCount += 1
                    
                    if prevAccMag >= 0.0:
                        j = abs(accMag - prevAccMag) / 0.02
                        if j > maxJerk: maxJerk = j
                    prevAccMag = accMag
                    
                gx, gy, gz = s[3], s[4], s[5]
                gyroMag = np.sqrt(gx*gx + gy*gy + gz*gz)
                if gyroMag > maxGyroMag: maxGyroMag = gyroMag
                
            accRange = (maxAccMag - minAccMag) if minAccMag < float('inf') else 0.0
            accMean = (sumAccMag / validAccCount) if validAccCount > 0 else 9.81
            accVariance = max(0.0, (sumAccMagSq / validAccCount) - (accMean * accMean)) if validAccCount > 0 else 0.0
            accStd = np.sqrt(accVariance)
            
            isCollisionShock = (maxAccMag >= self.imp_peak_th and maxJerk >= self.imp_jerk_th) or \
                               (accRange >= self.imp_range_th and maxJerk >= self.imp_jerk_alt_th and maxGyroMag >= self.imp_gyro_th)
            isLocomotionCadence = (accStd >= self.thrash_std_th and maxGyroMag >= self.thrash_gyro_th) or (accStd >= 5.0)
            isSettledImmobility = (accStd <= self.still_std_th) and (maxGyroMag <= self.still_gyro_th)
            
            # Robust scaling
            scaled_window = (window - self.medians) / self.iqrs
            
            # Model inference
            if self.device == "phone":
                inp = scaled_window.reshape(1, 100, 9)
                logits = self.session.run(None, {self.input_name: inp})[0][0]
                exp_vals = np.exp(logits - np.max(logits))
                probs = exp_vals / exp_vals.sum()
            else:
                if self.rf_model is not None:
                    feats = self.extract_watch_features(scaled_window).reshape(1, -1)
                    probs = self.rf_model.predict_proba(feats)[0]
                else:
                    probs = np.zeros(14)
                    
            topIdx = int(np.argmax(probs))
            fallProb = float(np.sum(probs[0:5]))
            lyingDownProb = float(probs[6]) if len(probs) > 6 else 0.0
            
            # State machine
            state_before = state
            transition_reason = None
            
            if state == "MONITORING":
                if isCollisionShock:
                    recentImpactCountdown = self.verification_horizon
                    transition_reason = f"Collision shock (AccPeak={maxAccMag:.1f}, Jerk={maxJerk:.1f})"
                elif recentImpactCountdown > 0:
                    if isLocomotionCadence:
                        recentImpactCountdown = 0
                        transition_reason = f"Locomotion cadence resumed (AccStd={accStd:.2f}, Gyro={maxGyroMag:.2f})"
                    elif isSettledImmobility and (fallProb >= self.fall_prob_th or (lyingDownProb >= self.lying_prob_th and accStd <= 1.8)):
                        recentImpactCountdown = 0
                        state = "FALL_SUSPECTED"
                        transition_reason = f"Post-impact immobility confirmed (FallProb={fallProb:.3f}, AccStd={accStd:.2f})"
                    else:
                        recentImpactCountdown -= 1
            elif state == "FALL_SUSPECTED":
                pass
                
            results.append({
                "window_index": len(results),
                "ts_start": w_ts_start,
                "ts_end": w_ts_end,
                "acc_peak": float(maxAccMag),
                "acc_min": float(minAccMag),
                "acc_range": float(accRange),
                "acc_std": float(accStd),
                "gyro_peak": float(maxGyroMag),
                "jerk_peak": float(maxJerk),
                "is_shock": bool(isCollisionShock),
                "is_loco": bool(isLocomotionCadence),
                "is_settled": bool(isSettledImmobility),
                "top_class": topIdx,
                "fall_prob": fallProb,
                "lying_prob": lyingDownProb,
                "state_before": state_before,
                "state_after": state,
                "reason": transition_reason
            })
            
        return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase13e_replay_engine.py <csv_path> [device: phone|watch]")
        sys.exit(1)
    csv_file = sys.argv[1]
    dev = sys.argv[2] if len(sys.argv) > 2 else "phone"
    engine = Phase13EReplayEngine(dev)
    res = engine.replay_csv(csv_file)
    print(f"Replayed {len(res)} windows from {csv_file}")
    for r in res:
        if r["state_after"] != "MONITORING" or r["is_shock"] or r["reason"]:
            print(f"Win {r['window_index']:02d}: Peak={r['acc_peak']:.1f}, Jerk={r['jerk_peak']:.1f}, FallProb={r['fall_prob']:.3f}, State={r['state_after']} ({r['reason']})")
