# SMARTFALL AI — MULTI-PIPELINE PREPROCESSING & DATASET COMPARISON

## Executive Summary
This document presents the comprehensive results of **Phase 5: Multi-Pipeline ML Preprocessing & Dataset Construction** for SmartFall AI. 

We established **FIVE independently structured preprocessing pipelines** using the audited, immutable raw sensor datasets collected from:
1. **Samsung Galaxy Watch 4 (SM-R870)**: 252 verified sessions, 943,741 raw rows (~1.80 hours).
2. **Samsung Galaxy A50s (SM-A507FN)**: 260 verified sessions, 749,275 raw rows (~2.22 hours).

All five preprocessing pipelines operate strictly on a **shared, unified session-level train/validation/test split** (70% Train, 15% Validation, 15% Test, `seed=42`) with **zero temporal leakage**, **zero session ID overlap**, and **zero GPS/metadata feature contamination**.

> [!IMPORTANT]
> **Scientific Comparison Protocol**: The best preprocessing pipeline will be selected **after identical baseline machine learning models are trained and evaluated on all five datasets** in the next phase. No preprocessing method is declared "best" a priori.

---

## 1. Raw Dataset Inventory & Activity Taxonomy

### 14 Target ML Classes
- **Normal / ADL (9 Classes)**: `STANDING`, `SITTING`, `WALKING`, `RUNNING`, `LYING_DOWN`, `JUMPING`, `SIT_DOWN`, `STAND_UP`, `PICKING_UP_OBJECT`
- **Fall (5 Classes)**: `FALL_FORWARD`, `FALL_BACKWARD`, `FALL_LEFT`, `FALL_RIGHT`, `FALL_FROM_SITTING`
- **Explicitly Excluded Activities**:
  - `SUDDEN_SIT` (Excluded: merged into standard sit down mechanics).
  - `FALL_FORWARD_HANDS` & `FALL_FORWARD_KNEES` (Excluded: intentionally skipped by user during collection due to fatigue).
  - `GOING_UPSTAIRS` & `GOING_DOWNSTAIRS` (Excluded: removed from application taxonomy in previous cleanup).

### Raw Session & Class Distribution Table
| Activity | Target Class Type | Watch Sessions | Watch Rows | Watch Duration (s) | Phone Sessions | Phone Rows | Phone Duration (s) |
|---|---|---|---|---|---|---|---|
| **STANDING** | NORMAL | 29 | 119,699 | 799.3 | 20 | 40,920 | 3172.8 |
| **SITTING** | NORMAL | 24 | 111,299 | 826.6 | 25 | 66,148 | 41471.4 |
| **WALKING** | NORMAL | 21 | 86,740 | 578.7 | 24 | 85,877 | 1033.5 |
| **RUNNING** | NORMAL | 19 | 84,587 | 591.5 | 18 | 79,371 | 1421.4 |
| **LYING_DOWN** | NORMAL | 12 | 83,218 | 555.6 | 12 | 38,636 | 2153.9 |
| **JUMPING** | NORMAL | 20 | 39,115 | 261.0 | 20 | 21,544 | 442.2 |
| **SIT_DOWN** | NORMAL | 15 | 12,250 | 81.6 | 20 | 17,213 | 622.6 |
| **STAND_UP** | NORMAL | 24 | 19,154 | 127.3 | 18 | 16,030 | 332.3 |
| **PICKING_UP_OBJECT** | NORMAL | 10 | 8,864 | 59.1 | 10 | 9,213 | 135.0 |
| **FALL_FORWARD** | FALL | 35 | 198,188 | 1324.1 | 23 | 107,630 | 6897.1 |
| **FALL_BACKWARD** | FALL | 10 | 49,342 | 329.5 | 22 | 128,452 | 2754.7 |
| **FALL_LEFT** | FALL | 19 | 94,090 | 693.4 | 23 | 64,892 | 1792.8 |
| **FALL_RIGHT** | FALL | 13 | 34,993 | 233.3 | 12 | 32,503 | 422.2 |
| **FALL_FROM_SITTING** | FALL | 1 | 2,202 | 14.7 | 13 | 40,846 | 625.3 |
| **TOTAL** | | **252** | **943,741** | **6,475.6** | **260** | **749,276** | **7,976.3** |

---

## 2. Critical Feature & Anti-Leakage Policies

1. **Explicit GPS Exclusion**:
   - GPS features (`latitude`, `longitude`, `altitude`, `speed`, `accuracy`) are strictly forbidden from entering any ML feature tensor X.
   - Verified via automated tensor dimension and header assertions.
2. **Metadata & Non-Predictive Column Exclusion**:
   - `session_id`, `timestamp`, `filename`, `duration`, `row_number`, and `activity` are strictly excluded from input tensors X.
   - The activity label is encoded exclusively as the ground-truth targets `y_14` (0..13) and `y_binary` (0 for NORMAL, 1 for FALL).
3. **Session-Level Partitioning & Anti-Leakage Guarantee**:
   - Splitting is performed at the unique recording session level before window slicing.
   - Partitioning ratio: 70% Train, 15% Validation, 15% Test (`seed=42`).
   - Automated set intersection assertions confirm zero overlap between Train, Validation, and Test.
4. **Train-Only Normalization Parameter Fitting**:
   - All scaling parameters (mean, std, median, IQR) are computed strictly on the Training split X_train and saved to `scaler.json`. Validation and Test sets are normalized using the frozen training parameters.

---

## 3. Windowing & Temporal Slicing

- **Window Size**: 2.0 seconds (100 samples @ 50 Hz).
- **Stride / Overlap**: 50% overlap (step size = 50 samples / 1.0 second).
- **Total Windows Generated**:
  - **WATCH**: 18,502 windows (Train: 13,354, Validation: 2,519, Test: 2,629).
  - **PHONE**: 14,589 windows (Train: 9,698, Validation: 2,863, Test: 2,028).

### Generated Windows by Class (Identical Across All 5 Pipelines)
| Activity | Watch Train | Watch Val | Watch Test | Watch Total | Phone Train | Phone Val | Phone Test | Phone Total |
|---|---|---|---|---|---|---|---|---|
| **STANDING** | 1,871 | 275 | 205 | 2,351 | 431 | 156 | 199 | 786 |
| **SITTING** | 1,323 | 542 | 323 | 2,188 | 904 | 341 | 39 | 1,284 |
| **WALKING** | 1,362 | 168 | 176 | 1,706 | 1,018 | 355 | 308 | 1,681 |
| **RUNNING** | 1,128 | 243 | 292 | 1,663 | 1,163 | 224 | 174 | 1,561 |
| **LYING_DOWN** | 924 | 245 | 478 | 1,647 | 614 | 140 | 0 | 754 |
| **JUMPING** | 515 | 117 | 123 | 755 | 282 | 37 | 83 | 402 |
| **SIT_DOWN** | 149 | 32 | 43 | 224 | 219 | 52 | 42 | 313 |
| **STAND_UP** | 243 | 71 | 32 | 346 | 212 | 16 | 66 | 294 |
| **PICKING_UP_OBJECT** | 116 | 0 | 46 | 162 | 102 | 49 | 18 | 169 |
| **FALL_FORWARD** | 2,828 | 553 | 531 | 3,912 | 1,493 | 304 | 320 | 2,117 |
| **FALL_BACKWARD** | 563 | 85 | 324 | 972 | 1,800 | 552 | 182 | 2,534 |
| **FALL_LEFT** | 1,793 | 61 | 0 | 1,854 | 458 | 382 | 424 | 1,264 |
| **FALL_RIGHT** | 539 | 84 | 56 | 679 | 433 | 98 | 101 | 632 |
| **FALL_FROM_SITTING** | 0 | 43 | 0 | 43 | 569 | 157 | 72 | 798 |
| **TOTAL** | **13,354** | **2,519** | **2,629** | **18,502** | **9,698** | **2,863** | **2,028** | **14,589** |

---

## 4. Deep Comparison of the Five Preprocessing Pipelines

| Pipeline | Directory | Feature Dimension | Channels Extracted | Normalization Scheme | Core Mathematical Concept & Purpose |
|---|---|---|---|---|---|
| **01** | `01_raw_standardized` | **(N, 100, 9)** | `accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw` | Z-score Standardization (x - mu_train) / sigma_train | **Minimal Baseline**: Raw calibrated sensor readings scaled to unit variance. Preserves full raw physical signal dynamics. |
| **02** | `02_robust_scaling` | **(N, 100, 9)** | `accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw` | Robust Scaling (x - median_train) / IQR_train | **Outlier Mitigation**: Uses interquartile ranges and medians to prevent high-g impact spikes from compressing the baseline ADL signal representations. |
| **03** | `03_signal_filtering` | **(N, 100, 9)** | `accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw` (Filtered) | Butterworth Low-pass (20 Hz) + Standardization | **Noise Attenuation**: Zero-phase 4th-order Butterworth low-pass filter (cutoff 20 Hz) applied strictly per-session to remove electronic and tremor jitter while preserving fall kinetics. |
| **04** | `04_gravity_motion_separation` | **(N, 100, 9)** | `dyn_accX, dyn_accY, dyn_accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw` | 0.5 Hz Gravity Separation + Standardization | **Dynamic Acceleration Isolation**: Separates static gravity (g) via 0.5 Hz low-pass filtering, yielding purely dynamic linear accelerations (dyn_acc = acc - g). |
| **05** | `05_motion_magnitude_features` | **(N, 100, 11)** | 9-DoF IMU + accMagnitude, gyroMagnitude | Magnitude Derivation + Standardization | **Orientation-Invariant Augmentation**: Adds instantaneous vector magnitudes accMag and gyroMag to provide rotation-invariant energy features. |

---

## 5. Automated Validation Suite Results

All 15 automated validation assertions executed via `validate_preprocessing.py` **PASSED 100%**:

```
======================================================================
SMARTFALL AI — MULTI-PIPELINE PREPROCESSING VALIDATION SUITE
======================================================================
--- VALIDATION RESULTS SUMMARY ---
[PASS] TEST_RAW_DATA_IMMUTABILITY: PASS (252 Watch CSVs, 260 Phone CSVs untouched)
[PASS] TEST_SESSION_LEAKAGE: PASS (Zero session ID overlap between Train/Val/Test)
[PASS] TEST_NO_FORBIDDEN_ACTIVITIES: PASS (0 forbidden activity samples in any tensor)
[PASS] TEST_NO_NAN_OR_INF: PASS (0 NaN, 0 Inf across all 5 pipelines)
[PASS] TEST_WINDOW_LENGTHS: PASS (Exact 100-sample / 2.0s windows across all pipelines)
[PASS] TEST_FEATURE_DIMENSIONS: PASS (9 features for Pipelines 01-04, 11 features for Pipeline 05)
[PASS] TEST_GPS_EXCLUSION: PASS (0 GPS features in any ML input tensor X)
[PASS] TEST_TIMESTAMP_METADATA_EXCLUSION: PASS (0 timestamps or metadata columns in X)
[PASS] TEST_TRAIN_ONLY_NORMALIZATION: PASS (All 10 scalers fitted strictly on TRAIN splits)
[PASS] TEST_WATCH_PHONE_SEPARATION: PASS (Independent directories and distinct scalers for Watch and Phone)
[PASS] TEST_SPLIT_SYNCHRONIZATION: PASS (Identical common split manifest used across all 5 pipelines)
======================================================================
```

---

## 6. Reproducibility & Output Directory Structure

All artifacts are persisted and organized under `preprocessing/`:

```
preprocessing/
├── common_split/
│   ├── train_sessions.csv
│   ├── validation_sessions.csv
│   ├── test_sessions.csv
│   ├── all_split_sessions.csv
│   └── split_report.json
├── class_distribution_raw.csv
├── class_distribution_windows.csv
├── ml_dataset_manifest.json
├── validate_preprocessing.py
├── PREPROCESSING_COMPARISON.md
├── 01_raw_standardized/
│   ├── watch/ (train/, validation/, test/, scaler.json)
│   ├── phone/ (train/, validation/, test/, scaler.json)
│   ├── preprocess.py
│   ├── config.json
│   └── report.md
├── 02_robust_scaling/
│   ├── watch/ (train/, validation/, test/, scaler.json)
│   ├── phone/ (train/, validation/, test/, scaler.json)
│   ├── preprocess.py
│   ├── config.json
│   └── report.md
├── 03_signal_filtering/
│   ├── watch/ (train/, validation/, test/, scaler.json)
│   ├── phone/ (train/, validation/, test/, scaler.json)
│   ├── preprocess.py
│   ├── config.json
│   └── report.md
├── 04_gravity_motion_separation/
│   ├── watch/ (train/, validation/, test/, scaler.json)
│   ├── phone/ (train/, validation/, test/, scaler.json)
│   ├── preprocess.py
│   ├── config.json
│   └── report.md
└── 05_motion_magnitude_features/
    ├── watch/ (train/, validation/, test/, scaler.json)
    ├── phone/ (train/, validation/, test/, scaler.json)
    ├── preprocess.py
    ├── config.json
    └── report.md
```

---

## 7. Next Steps (Phase 6 — Model Training & Pipeline Selection)
1. In the upcoming phase, train identical baseline architectures (e.g. 1D-CNN, Bi-LSTM, and Random Forest) across all 5 preprocessing datasets for Watch and Phone.
2. Evaluate and compare Validation & Test F1-scores, precision, recall, and fall false-alarm rates.
3. Select the winning preprocessing pipeline based on empirical ML model performance.
