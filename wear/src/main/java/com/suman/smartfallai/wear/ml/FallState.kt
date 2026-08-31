package com.suman.smartfallai.wear.ml

enum class FallState {
    MONITORING,
    FALL_SUSPECTED,
    FALL_CONFIRMED,
    SOS_TRIGGERED,
    CANCELLED
}

data class FallPredictionResult(
    val topActivityIndex: Int,
    val topActivityName: String,
    val topConfidence: Float,
    val fallProbability: Float,
    val isFallDetected: Boolean,
    val fallState: FallState,
    val inferenceLatencyMs: Long
)
