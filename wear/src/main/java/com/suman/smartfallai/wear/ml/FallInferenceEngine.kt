package com.suman.smartfallai.wear.ml

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
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

    private var suspectedConsecutiveWindows = 0

    fun addSample(accX: Float, accY: Float, accZ: Float,
                  gyroX: Float, gyroY: Float, gyroZ: Float,
                  pitch: Float, roll: Float, yaw: Float) {
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

        // 1. Preprocess with frozen Train RobustScaler
        scaler.transformInPlace(window)

        // 2. Extract 72 window-level statistical features
        val features = WatchFeatureExtractor.extractFeatures(window)

        // 3. Random Forest Inference
        val probs = rfEngine.predictProba(features)

        // 4. Calculate top prediction and fall probability
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
        val isInstantFall = fallProb >= 0.50f

        // 5. Temporal Fall State Machine
        val newState = when (_currentState.value) {
            FallState.MONITORING -> {
                if (isInstantFall) {
                    suspectedConsecutiveWindows = 1
                    FallState.FALL_SUSPECTED
                } else {
                    suspectedConsecutiveWindows = 0
                    FallState.MONITORING
                }
            }
            FallState.FALL_SUSPECTED -> {
                if (isInstantFall) {
                    suspectedConsecutiveWindows++
                    if (suspectedConsecutiveWindows >= 2) {
                        FallState.FALL_CONFIRMED
                    } else {
                        FallState.FALL_SUSPECTED
                    }
                } else {
                    suspectedConsecutiveWindows = 0
                    FallState.MONITORING
                }
            }
            FallState.FALL_CONFIRMED -> {
                FallState.SOS_TRIGGERED
            }
            FallState.SOS_TRIGGERED -> FallState.SOS_TRIGGERED
            FallState.CANCELLED -> FallState.MONITORING
        }

        _currentState.value = newState
        _lastResult.value = FallPredictionResult(
            topActivityIndex = topIdx,
            topActivityName = classNames.getOrElse(topIdx) { "UNKNOWN" },
            topConfidence = topConf,
            fallProbability = fallProb,
            isFallDetected = (newState == FallState.FALL_CONFIRMED || newState == FallState.SOS_TRIGGERED),
            fallState = newState,
            inferenceLatencyMs = latency
        )

        Log.d("WatchFallML", "Inference: Activity=${classNames.getOrElse(topIdx) { "UNKNOWN" }} ($topConf), FallProb=$fallProb, State=$newState, Latency=${latency}ms")
    }

    fun reset() {
        synchronized(sampleBuffer) {
            sampleBuffer.clear()
            samplesSinceLastInference = 0
            suspectedConsecutiveWindows = 0
            _currentState.value = FallState.MONITORING
            _lastResult.value = null
        }
    }
}