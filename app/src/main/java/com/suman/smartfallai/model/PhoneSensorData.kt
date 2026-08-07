package com.suman.smartfallai.model

data class PhoneSensorData(

    val timestamp: Long,

    val accX: Float,
    val accY: Float,
    val accZ: Float,

    val gyroX: Float,
    val gyroY: Float,
    val gyroZ: Float,

    val pitch: Float,
    val roll: Float,
    val yaw: Float

)