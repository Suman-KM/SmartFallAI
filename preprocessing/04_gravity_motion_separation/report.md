# Pipeline 04: Gravity-Motion Separation Report

## Pipeline Overview
- **Identifier**: `04_gravity_motion_separation`
- **Methodology**: 0.5 Hz 2nd-order Butterworth low-pass filter to isolate the static gravity vector $g$.
- **Dynamic Acceleration**: $a_{dyn} = a_{raw} - g$.
- **Session Isolation**: Gravity separation is performed per-session before window generation.
- **Normalization**: Z-score Standardization $(x - \mu_{train}) / \sigma_{train}$ computed strictly on TRAIN partitions only.
- **Features (9)**: `dyn_accX, dyn_accY, dyn_accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw`
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
