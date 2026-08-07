package com.suman.smartfallai.gps

data class GpsData(

    val latitude: Double = 0.0,

    val longitude: Double = 0.0,

    val altitude: Double = 0.0,

    val speed: Float = 0f,

    val accuracy: Float = 0f,

    val timestamp: Long = System.currentTimeMillis()

)