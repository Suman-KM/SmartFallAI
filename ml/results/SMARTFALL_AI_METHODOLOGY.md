# SMARTFALL AI — COMPREHENSIVE RESEARCH METHODOLOGY

## 1. End-to-End System Architecture
SmartFall AI implements an autonomous, privacy-preserving, edge-computed fall detection framework for wearable smartwatches and mobile smartphones.

```
+-----------------------------------------------------------------------------------+
|                           SMARTFALL AI SYSTEM PIPELINE                             |
+-----------------------------------------------------------------------------------+
  [ Sensor Acquisition ] : 9-DoF IMU @ 50 Hz (accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw)
            |
  [ Circular Ring Buffer ] : 100-Sample Sliding Window (2.0s duration, 50% overlap / 50-sample stride)
            |
  [ Normalization ] : Frozen Train-Set RobustScaler (x - median_train) / IQR_train
            |
  +-----------------------------------+-----------------------------------+
  |           WATCH PATHWAY           |           PHONE PATHWAY           |
  | (Wear OS — Samsung Galaxy Watch 4)| (Android — Samsung Galaxy A50s)   |
  +-----------------------------------+-----------------------------------+
  | 72 Statistical Feature Extraction | Raw Temporal Window (100 x 9)     |
  | Native Kotlin Random Forest       | Microsoft ONNX Runtime 1D-CNN     |
  | Evaluates 100 Binary Trees        | 3-Stage Temporal Convolution      |
  +-----------------------------------+-----------------------------------+
            |                                           |
  [ Probabilistic Aggregation ] : P(fall) = Sum(P_i, i in [0..4]) >= 0.50
            |
  [ State Machine ] : MONITORING -> FALL_SUSPECTED -> FALL_CONFIRMED -> SOS_TRIGGERED
            |
  [ Temporal Consensus ] : Requires 2 consecutive positive windows (1.0s confirmation)
            |
  [ Local Emergency Dispatch ] : Autonomous on-device SOS via Wi-Fi / LTE (No Bluetooth bridge needed)
```

## 2. Scientific Rigor & Data Integrity Principles
1. **Zero Session Overlap (Session-Level Stratified Partition)**:
   - 506 total raw recording sessions split into 70% Train (353), 15% Validation (76), and 15% Test (77).
   - Zero sample from any subject session in the training set appears in validation or testing.
2. **Strict Feature Policy**:
   - GPS coordinates, timestamps, session IDs, and heart rate are strictly quarantined and never entered into ML tensors.
3. **Validation-Driven Selection**:
   - All hyperparameter tuning and model architecture selections occurred on Validation data only. The Test set was evaluated strictly for final reporting.
