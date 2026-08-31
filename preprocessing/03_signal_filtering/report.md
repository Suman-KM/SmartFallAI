# Pipeline 03: Signal Filtering Report

## Pipeline Overview
- **Identifier**: `03_signal_filtering`
- **Filtering Methodology**: Zero-phase 4th-order Butterworth low-pass filter ($f_c = 20.0\text{ Hz}$, $f_s = 50.0\text{ Hz}$ via `scipy.signal.filtfilt`).
- **Session Isolation**: Filtering is strictly applied per-session file (no cross-session boundary leakage).
- **Normalization**: Z-score Standardization $(x - \mu_{train}) / \sigma_{train}$ computed strictly on TRAIN partitions only.
- **Features (9)**: Filtered `accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw`
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
