package com.suman.smartfallai.model

data class CombinedSensorData(

    val phone: PhoneSensorData? = null,

    val watch: WatchSensorData? = null,

    val latitude: Double = 0.0,

    val longitude: Double = 0.0,

    val altitude: Double = 0.0,

    val speed: Float = 0f,

    val activity: String = ""

)