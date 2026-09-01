package com.suman.smartfallai.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class FallInferenceEngine(context: Context) {

    private val scaler = PhoneRobustScaler(context)
    private val onnxEngine = PhoneOnnxEngine(context)

    private val windowSize = 100 // 2.0s @ 50 Hz
    private val stepSize = 50     // 1.0s stride (50% overlap)

    private val sampleBuffer = ArrayList<FloatArray>(100)
    private var samplesSinceLastInference = 0

    private val _currentState = MutableStateFlow(FallState.MONITORING)
    val currentState: StateFlow<FallState> = _currentState.asStateFlow()

    private val _lastResult = MutableStateFlow<FallPredictionResult?>(null)
    val lastResult: StateFlow<FallPredictionResult?> = _lastResult.asStateFlow()

    private val scope = CoroutineScope(Dispatchers.Default)

    // Class mapping
    private val classNames = arrayOf(
        "FALL_BACKWARD", "FALL_FORWARD", "FALL_FROM_SITTING", "FALL_LEFT", "FALL_RIGHT",
        "JUMPING", "LYING_DOWN", "PICKING_UP_OBJECT", "RUNNING", "SITTING", "SIT_DOWN",
        "STANDING", "STAND_UP", "WALKING"
    )

    private val emergencyManager = com.suman.smartfallai.emergency.EmergencyManager(context)

    private val _countdownRemaining = MutableStateFlow(0)
    val countdownRemaining: StateFlow<Int> = _countdownRemaining.asStateFlow()

    private var countdownJob: Job? = null
    private var suspectedConsecutiveWindows = 0
    private var recentImpactCountdown = 0

    fun addSample(accX: Float, accY: Float, accZ: Float,
                  gyroX: Float, gyroY: Float, gyroZ: Float,
                  pitch: Float, roll: Float, yaw: Float) {
        // Discard uninitialized sensor samples before accelerometer is active
        if (accX == 0f && accY == 0f && accZ == 0f) return

        val sample = floatArrayOf(accX, accY, accZ, gyroX, gyroY, gyroZ, pitch, roll, yaw)

        synchronized(sampleBuffer) {
            if (sampleBuffer.size >= windowSize) {
                sampleBuffer.removeAt(0)
            }
            sampleBuffer.add(sample)
            samplesSinceLastInference++

            if (sampleBuffer.size == windowSize && samplesSinceLastInference >= stepSize) {
                samplesSinceLastInference = 0
                val windowCopy = Array(windowSize) { i -> sampleBuffer[i].clone() }
                scope.launch {
                    processWindow(windowCopy)
                }
            }
        }
    }

    private fun processWindow(window: Array<FloatArray>) {
        val startTime = System.currentTimeMillis()

        // 0. Compute raw kinematic dynamics before scaling (physical units: m/s^2, rad/s)
        var maxAccMag = 0.0f
        var minAccMag = Float.MAX_VALUE
        var maxGyroMag = 0.0f
        var sumAccMag = 0.0f
        var sumAccMagSq = 0.0f
        var validAccCount = 0

        for (sample in window) {
            val ax = sample[0]
            val ay = sample[1]
            val az = sample[2]
            val accMag = kotlin.math.sqrt(ax * ax + ay * ay + az * az)
            if (accMag > 1.0f) {
                if (accMag > maxAccMag) maxAccMag = accMag
                if (accMag < minAccMag) minAccMag = accMag
                sumAccMag += accMag
                sumAccMagSq += accMag * accMag
                validAccCount++
            }

            val gx = sample[3]
            val gy = sample[4]
            val gz = sample[5]
            val gyroMag = kotlin.math.sqrt(gx * gx + gy * gy + gz * gz)
            if (gyroMag > maxGyroMag) maxGyroMag = gyroMag
        }
        val accRange = if (minAccMag < Float.MAX_VALUE) (maxAccMag - minAccMag) else 0.0f
        val accMean = if (validAccCount > 0) sumAccMag / validAccCount else 9.81f
        val accVariance = if (validAccCount > 0) kotlin.math.max(0.0f, (sumAccMagSq / validAccCount) - (accMean * accMean)) else 0.0f
        val accStd = kotlin.math.sqrt(accVariance)

        // Phase 13C Calibrated Kinematic Impact Shock Gate (Phone SM-A507FN):
        // Real falls produce sharp ground impact (>= 18.0 m/s^2 = 1.83g) or intense rotational tumble (accRange >= 10 m/s^2 && gyro >= 2.5 rad/s)
        val hasImpact = (maxAccMag >= 18.0f) || (accRange >= 10.0f && maxGyroMag >= 2.5f)
        if (hasImpact) {
            recentImpactCountdown = 3 // remember impact across sliding windows (~3 seconds)
        } else if (recentImpactCountdown > 0) {
            recentImpactCountdown--
        }

        // Active non-fall thrashing rejection:
        // Continuous energetic activities (e.g. running or jumping) display continuous oscillating variance (accStd >= 4.5 m/s^2 && maxGyroMag >= 3.5 rad/s)
        // In contrast, post-impact fall windows settle to stillness or low dynamic variance (< 2.0 m/s^2)
        val isContinuousThrashing = (accStd >= 4.5f) && (maxGyroMag >= 3.5f)

        // 1. Preprocess with frozen Train RobustScaler
        scaler.transformInPlace(window)

        // 2. ONNX 1D-CNN Model Inference
        val probs = onnxEngine.predictProba(window)

        // 3. Calculate top prediction and fall probability
        var topIdx = 0
        var topConf = 0.0f
        var fallProb = 0.0f

        for (i in probs.indices) {
            if (probs[i] > topConf) {
                topConf = probs[i]
                topIdx = i
            }
            // Fall classes: 0..4
            if (i in 0..4) {
                fallProb += probs[i]
            }
        }

        val latency = System.currentTimeMillis() - startTime
        // Phase 13C Calibrated Fall Candidate:
        // 1. Model fall probability >= 0.45 (verified across all 5 fall types)
        // 2. Physical impact shock event verified (current or remembered within 3 windows)
        // 3. Reject active running/jumping thrashing
        val isKinematicFallCandidate = (fallProb >= 0.45f) && (hasImpact || recentImpactCountdown > 0) && (!isContinuousThrashing)

        // 4. Fall State Machine with 2-Window Consensus & Interactive Countdown
        synchronized(this) {
            when (_currentState.value) {
                FallState.MONITORING -> {
                    if (isKinematicFallCandidate) {
                        suspectedConsecutiveWindows++
                        if (suspectedConsecutiveWindows >= 2) {
                            Log.w("PhoneFallML", "2-Window consensus confirmed fall with impact (AccPeak=$maxAccMag, AccRng=$accRange, GyroPeak=$maxGyroMag)! Triggering FALL_SUSPECTED and countdown.")
                            _currentState.value = FallState.FALL_SUSPECTED
                            startCountdown()
                        }
                    } else {
                        suspectedConsecutiveWindows = 0
                    }
                }
                FallState.FALL_SUSPECTED -> {
                    // Countdown is running. Keep processing background inference but preserve countdown state
                }
                FallState.FALL_CONFIRMED -> {
                    // Escalated to SOS
                }
                FallState.SOS_TRIGGERED -> {
                    // Emergency dispatched
                }
                FallState.CANCELLED -> {
                    _currentState.value = FallState.MONITORING
                    suspectedConsecutiveWindows = 0
                    recentImpactCountdown = 0
                }
            }
        }

        _lastResult.value = FallPredictionResult(
            topActivityIndex = topIdx,
            topActivityName = classNames.getOrElse(topIdx) { "UNKNOWN" },
            topConfidence = topConf,
            fallProbability = fallProb,
            isFallDetected = (_currentState.value == FallState.FALL_CONFIRMED || _currentState.value == FallState.SOS_TRIGGERED),
            fallState = _currentState.value,
            inferenceLatencyMs = latency
        )

        Log.d("PhoneFallML", "Inference: Activity=${classNames.getOrElse(topIdx) { "UNKNOWN" }} ($topConf), FallProb=$fallProb, AccPeak=${"%.2f".format(maxAccMag)}, AccRng=${"%.2f".format(accRange)}, GyroPeak=${"%.2f".format(maxGyroMag)}, Impact=$hasImpact, State=${_currentState.value}, Latency=${latency}ms")
    }

    private fun startCountdown() {
        countdownJob?.cancel()
        countdownJob = CoroutineScope(Dispatchers.Main).launch {
            Log.i("PhoneFallML", "Starting 10-second emergency countdown...")
            emergencyManager.triggerVibration(isEmergency = false)
            for (sec in 10 downTo 1) {
                _countdownRemaining.value = sec
                kotlinx.coroutines.delay(1000)
                emergencyManager.triggerVibration(isEmergency = false)
            }
            _countdownRemaining.value = 0

            // Countdown expired without response!
            Log.w("PhoneFallML", "Countdown expired! Escalating to FALL_CONFIRMED -> SOS_TRIGGERED")
            _currentState.value = FallState.FALL_CONFIRMED
            _currentState.value = FallState.SOS_TRIGGERED
            emergencyManager.sendEmergencyAlert("Samsung Galaxy A50s (Phone)", System.currentTimeMillis())
        }
    }

    fun cancelCountdown() {
        Log.i("PhoneFallML", "User pressed I'M OK — Cancelling countdown and returning to MONITORING")
        countdownJob?.cancel()
        countdownJob = null
        _countdownRemaining.value = 0
        suspectedConsecutiveWindows = 0
        recentImpactCountdown = 0
        _currentState.value = FallState.CANCELLED
        _currentState.value = FallState.MONITORING
    }

    fun dismissAlert() {
        Log.i("PhoneFallML", "Dismissing emergency alert and returning to MONITORING")
        countdownJob?.cancel()
        countdownJob = null
        _countdownRemaining.value = 0
        suspectedConsecutiveWindows = 0
        recentImpactCountdown = 0
        _currentState.value = FallState.MONITORING
    }

    fun reset() {
        synchronized(sampleBuffer) {
            countdownJob?.cancel()
            countdownJob = null
            sampleBuffer.clear()
            samplesSinceLastInference = 0
            suspectedConsecutiveWindows = 0
            _countdownRemaining.value = 0
            _currentState.value = FallState.MONITORING
            _lastResult.value = null
        }
    }

    fun close() {
        countdownJob?.cancel()
        onnxEngine.close()
    }
}