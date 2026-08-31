# Pipeline 02: Robust Scaling Report

## Pipeline Overview
- **Identifier**: `02_robust_scaling`
- **Methodology**: 9-DoF IMU signals windowed into 2.0-second sliding windows (100 samples @ 50 Hz, 50% overlap).
- **Normalization**: Robust scaling $(x - \text{median}_{train}) / \text{IQR}_{train}$ strictly computed on TRAIN partitions only.
- **Features (9)**: `accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw`
- **GPS & Metadata Policy**: Completely excluded from input tensors $X$.

## Output Dataset Statistics

### WATCH
- **Train Tensor**: `[13354, 100, 9]`
- **Validation Tensor**: `[2519, 100, 9]`
- **Test Tensor**: `[2629, 100, 9]`
- **Total Windows**: `18502`

### PHONE
- **Train Tensor**: `[9698, 100, 9]`
- **Validation Tensor**: `[2863, 100, 9]`
- **Test Tensor**: `[2028, 100, 9]`
- **Total Windows**: `14589`
