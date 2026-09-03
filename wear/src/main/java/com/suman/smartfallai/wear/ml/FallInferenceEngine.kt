package com.suman.smartfallai.wear.ml

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

    private val scaler = WatchRobustScaler(context)
    private val rfEngine = WatchRandomForestEngine(context)

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

    private val emergencyManager = com.suman.smartfallai.wear.emergency.EmergencyManager(context)

    private val _countdownRemaining = MutableStateFlow(0)
    val countdownRemaining: StateFlow<Int> = _countdownRemaining.asStateFlow()

    private var countdownJob: Job? = null
    private var suspectedConsecutiveWindows = 0
    private var recentImpactCountdown = 0
    private var activeMotionRecoveryWindows = 0
    private var consecutiveCadenceWindows = 0

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

        // 0. Compute raw kinematic dynamics before scaling (physical units: m/s^2, rad/s, m/s^3)
        var maxAccMag = 0.0f
        var minAccMag = Float.MAX_VALUE
        var maxGyroMag = 0.0f
        var maxJerk = 0.0f
        var prevAccMag = -1.0f
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

                if (prevAccMag >= 0.0f) {
                    val j = kotlin.math.abs(accMag - prevAccMag) / 0.02f
                    if (j > maxJerk) maxJerk = j
                }
                prevAccMag = accMag
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

        // Stage 2: Abnormal Impact Collision Shock Gate (Watch SM-R870):
        // Real wrist collisions produce hard shock (>= 26.0 m/s^2) with high jerk (>= 550 m/s^3)
        // or dynamic arm tumbling (accRange >= 16 m/s^2, jerk >= 400 m/s^3, gyro >= 4.0 rad/s)
        val isCollisionShock = (maxAccMag >= 26.0f && maxJerk >= 550.0f) ||
                               (accRange >= 16.0f && maxJerk >= 400.0f && maxGyroMag >= 4.0f)

        // Stage 3: Continuous Locomotion Cadence Check:
        // Active arm swing in brisk walking, running, and jumping produces ongoing wrist dynamics
        val isLocomotionCadence = (accStd >= 2.5f && maxGyroMag >= 2.5f) || (accStd >= 5.0f)

        // Stage 4: Post-Impact Stillness & Immobility Check:
        // In wrist falls, arm motion settles into low dynamic variance (accStd <= 3.8, gyro <= 3.2)
        val isSettledImmobility = (accStd <= 3.8f) && (maxGyroMag <= 3.2f)

        // 1. Preprocess with frozen Train RobustScaler
        scaler.transformInPlace(window)

        // 2. Extract 72 window-level statistical features
        val features = WatchFeatureExtractor.extractFeatures(window)

        // 3. Random Forest Inference
        val probs = rfEngine.predictProba(features)

        // 4. Calculate top prediction, fall probability, and lying down probability
        var topIdx = 0
        var topConf = 0.0f
        var fallProb = 0.0f
        val lyingDownProb = probs.getOrElse(6) { 0.0f } // Class index 6: LYING_DOWN

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

        // Phase 13E Posture Consistency Check:
        // An active upright activity (STANDING or WALKING) cannot be a fallen victim on the floor
        val isUprightAdl = (topIdx == 11 || topIdx == 13) && (topConf >= fallProb)
        val hasFallPosture = (!isUprightAdl) && ((fallProb >= 0.40f) || (lyingDownProb >= 0.45f && accStd <= 2.0f))

        val latency = System.currentTimeMillis() - startTime

        // Multi-Stage Temporal Fall Verification:
        var triggeredFallSuspected = false

        synchronized(this) {
            when (_currentState.value) {
                FallState.MONITORING -> {
                    if (isCollisionShock) {
                        // Stage 2: Collision shock registered -> Arm 4-window verification horizon (~2 to 3s)
                        recentImpactCountdown = 4
                        consecutiveCadenceWindows = 0
                        Log.i("WatchFallML", "Potential impact collision detected! AccPeak=$maxAccMag, JerkPeak=$maxJerk. Awaiting post-impact stillness confirmation.")
                    } else if (recentImpactCountdown > 0) {
                        if (isLocomotionCadence) {
                            // Stage 3: Require 2 consecutive windows of active locomotion cadence to discard shock
                            consecutiveCadenceWindows++
                            if (consecutiveCadenceWindows >= 2) {
                                recentImpactCountdown = 0
                                consecutiveCadenceWindows = 0
                                Log.d("WatchFallML", "Consecutive locomotion cadence confirmed after shock. Aborting false alarm.")
                            }
                        } else {
                            consecutiveCadenceWindows = 0
                        }

                        if (recentImpactCountdown > 0 && isSettledImmobility && hasFallPosture) {
                            // Stage 4: Post-impact immobility confirmed with fall/recumbent posture!
                            recentImpactCountdown = 0
                            consecutiveCadenceWindows = 0
                            triggeredFallSuspected = true
                            _currentState.value = FallState.FALL_SUSPECTED
                            startCountdown()
                        } else if (recentImpactCountdown > 0) {
                            recentImpactCountdown--
                        }
                    }
                }
                FallState.FALL_SUSPECTED -> {
                    // Active Motion Recovery Cancellation on Wear OS:
                    // If the user resumes active locomotion (walking/running/jumping) while countdown is running, automatically abort
                    if (isLocomotionCadence) {
                        activeMotionRecoveryWindows++
                        if (activeMotionRecoveryWindows >= 2) {
                            Log.i("WatchFallML", "Active locomotion detected during countdown on Wear OS. Automatically cancelling false alarm.")
                            cancelCountdown()
                            activeMotionRecoveryWindows = 0
                        }
                    } else {
                        activeMotionRecoveryWindows = 0
                    }
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
                    activeMotionRecoveryWindows = 0
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

        // Structured Debug Logging for Phase 13D
        Log.d("WatchFallML", "Activity=${classNames.getOrElse(topIdx) { "UNKNOWN" }}, FallProb=${"%.4f".format(fallProb)}, AccPeak=${"%.2f".format(maxAccMag)}, AccMin=${"%.2f".format(minAccMag)}, AccRange=${"%.2f".format(accRange)}, AccStd=${"%.2f".format(accStd)}, GyroPeak=${"%.2f".format(maxGyroMag)}, JerkPeak=${"%.1f".format(maxJerk)}, Impact=$isCollisionShock, TemporalScore=$recentImpactCountdown, PostImpact=$isSettledImmobility, ActiveMotion=$isLocomotionCadence, State=${_currentState.value}, Latency=${latency}ms")
    }

    private fun startCountdown() {
        countdownJob?.cancel()
        countdownJob = CoroutineScope(Dispatchers.Main).launch {
            Log.i("WatchFallML", "Starting 10-second emergency countdown on Wear OS...")
            emergencyManager.triggerVibration(isEmergency = false)
            for (sec in 10 downTo 1) {
                _countdownRemaining.value = sec
                kotlinx.coroutines.delay(1000)
                emergencyManager.triggerVibration(isEmergency = false)
            }
            _countdownRemaining.value = 0

            // Countdown expired without response!
            Log.w("WatchFallML", "Countdown expired! Escalating to FALL_CONFIRMED -> SOS_TRIGGERED")
            _currentState.value = FallState.FALL_CONFIRMED
            _currentState.value = FallState.SOS_TRIGGERED
            emergencyManager.sendEmergencyAlert(System.currentTimeMillis())
        }
    }

    fun cancelCountdown() {
        Log.i("WatchFallML", "User pressed I'M OK — Cancelling countdown and returning to MONITORING")
        countdownJob?.cancel()
        countdownJob = null
        _countdownRemaining.value = 0
        suspectedConsecutiveWindows = 0
        recentImpactCountdown = 0
        _currentState.value = FallState.CANCELLED
        _currentState.value = FallState.MONITORING
    }

    fun dismissAlert() {
        Log.i("WatchFallML", "Dismissing emergency alert and returning to MONITORING")
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
            recentImpactCountdown = 0
            _countdownRemaining.value = 0
            _currentState.value = FallState.MONITORING
            _lastResult.value = null
        }
    }
}