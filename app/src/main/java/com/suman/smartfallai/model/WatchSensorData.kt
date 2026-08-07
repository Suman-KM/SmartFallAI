package com.suman.smartfallai.model

data class WatchSensorData(

    val timestamp: Long,

    val accX: Float,
    val accY: Float,
    val accZ: Float,

    val gyroX: Float,
    val gyroY: Float,
    val gyroZ: Float,

    val heartRate: Int,

    val spo2: Int

)