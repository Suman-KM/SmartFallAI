# SMARTFALL AI — EMBEDDED MODEL DEPLOYMENT CONTRACT

## 1. Input Tensor Specification
- **Window Dimension**: `(Batch=1, TimeSteps=100, Channels=9)`
- **Sampling Frequency**: $f_s = 50.0\text{ Hz}$ (2.0 seconds duration).
- **Required Channel Order**:
  1. `accX` (m/s²)
  2. `accY` (m/s²)
  3. `accZ` (m/s²)
  4. `gyroX` (rad/s)
  5. `gyroY` (rad/s)
  6. `gyroZ` (rad/s)
  7. `pitch` (degrees / radians normalized)
  8. `roll` (degrees / radians normalized)
  9. `yaw` (degrees / radians normalized)

## 2. Normalization Scheme (P02 RobustScaler)
$$x_{norm} = \frac{x - \text{median}_{train}}{\text{IQR}_{train}}$$
- Normalization parameters must be loaded directly from `scaler.json` and must **never** be re-computed on device.

## 3. Strict Non-Predictive Feature Exclusions
The following fields **must never enter** the input tensor:
- `latitude`, `longitude`, `altitude`, `speed`, `accuracy` (GPS)
- `heart_rate`, `SpO2` (Biometrics)
- `timestamp`, `session_id`, `filename` (Metadata)

## 4. Output Specification
- **14-Class Logits / Probabilities**: `[P_0, P_1, ..., P_13]`
- **Binary Fall Rule**:
  $$\text{Event} = \begin{cases} \text{FALL} & \text{if } \sum_{i=0}^4 P_i \ge 0.50 \\ \text{NORMAL} & \text{otherwise} \end{cases}$$
