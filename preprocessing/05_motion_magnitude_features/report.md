# Pipeline 05: Motion Magnitude Features Report

## Pipeline Overview
- **Identifier**: `05_motion_magnitude_features`
- **Feature Augmentation**: 9-DoF raw IMU signals augmented with vector magnitudes:
  - $accMagnitude = \sqrt{accX^2 + accY^2 + accZ^2}$
  - $gyroMagnitude = \sqrt{gyroX^2 + gyroY^2 + gyroZ^2}$
- **Feature Vector (11 channels)**: `accX, accY, accZ, gyroX, gyroY, gyroZ, accMagnitude, gyroMagnitude, pitch, roll, yaw`
- **Normalization**: Z-score Standardization $(x - \mu_{train}) / \sigma_{train}$ computed strictly on TRAIN partitions only.
- **GPS & Metadata Policy**: Completely excluded from input tensors $X$.

## Output Dataset Statistics

### WATCH
- **Train Tensor**: `[13354, 100, 11]`
- **Validation Tensor**: `[2519, 100, 11]`
- **Test Tensor**: `[2629, 100, 11]`
- **Total Windows**: `18502`

### PHONE
- **Train Tensor**: `[9698, 100, 11]`
- **Validation Tensor**: `[2863, 100, 11]`
- **Test Tensor**: `[2028, 100, 11]`
- **Total Windows**: `14589`
