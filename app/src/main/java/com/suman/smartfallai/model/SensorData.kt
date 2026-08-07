package com.suman.smartfallai.model

data class SensorData(

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

    val activity: String = ""

)