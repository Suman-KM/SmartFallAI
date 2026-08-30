package com.suman.smartfallai.wear.model

data class SensorData(

    val sequence: Long = 0L,

    val timestamp: Long = System.currentTimeMillis(),

    val accX: Float = 0f,
    val accY: Float = 0f,
    val accZ: Float = 0f,

    val gyroX: Float = 0f,
    val gyroY: Float = 0f,
    val gyroZ: Float = 0f,

    val pitch: Float = 0f,
    val roll: Float = 0f,
    val yaw: Float = 0f,

    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val altitude: Double = 0.0,
    val speed: Float = 0f,
    val accuracy: Float = 0f,

    val heartRate: Float = 0f,
    val spo2: Float = 0f,

    val pressure: Float = 0f,

    val activity: String = "Unknown",

    // Prevents the initial empty StateFlow value
    // from being written to the dataset.
    val isValid: Boolean = false
)