# SMARTFALL AI — REAL-TIME INFERENCE ARCHITECTURE SPECIFICATION

## 1. High-Level Streaming Architecture

```
[ PHYSICAL SENSORS @ ~50 Hz ]
- 3D Accelerometer (accX, accY, accZ)
- 3D Gyroscope (gyroX, gyroY, gyroZ)
- Rotation Vector -> (pitch, roll, yaw)
             │
             ▼
[ CIRCULAR RING BUFFER (Capacity: 100 samples / 2.0s) ]
- Ingestion rate: 1 sample every ~20ms
- Lock-free ring buffer
             │  (Every 50 new samples = 1.0s stride)
             ▼
[ PREPROCESSING ENGINES (P02 RobustScaler) ]
- Apply frozen training parameters: x_norm = (x - median) / IQR
- Extract 8 statistical window features (WATCH) or keep 3D tensor (PHONE)
             │
             ▼
[ EMBEDDED INFERENCE ENGINE ]
- WATCH: Native Decision Ensemble (0.184 ms latency)
- PHONE: 1D-CNN ONNX / TFLite (0.021 ms latency)
             │
             ▼
[ POST-PROCESSING & TEMPORAL CONFIRMATION ]
- Evaluate P(Fall) = Sum(P(Fall_Classes))
- Threshold: theta = 0.50 (Sensitivity = 84.08% on Watch, 77.19% on Phone)
- Post-Impact Immobility Check (Hysteresis / 2-window consensus)
             │
             ▼
[ EMERGENCY TRIGGER DISPATCH ]
- Fall Verified -> Dispatch UI Alert -> 30s Countdown -> SOS + GPS Dispatch
```

## 2. Real-Time Latency Budget
- Sampling frequency: **50 Hz** (20.0 ms per sample).
- Window update stride: **50 samples** (1.0 second between inference evaluations).
- Inference latency:
  - **Phone 1D-CNN**: **0.021 ms** (< 0.003% of the 1.0s window stride).
  - **Watch Random Forest**: **0.184 ms** (< 0.02% of the 1.0s window stride).
- Conclusion: Zero risk of CPU starvation or frame drops on either platform.
