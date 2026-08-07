package com.suman.smartfallai.controller

data class RecordingState(

    val isRecording: Boolean = false,

    val activity: String = "Walking",

    val sampleCount: Int = 0,

    val elapsedTime: Long = 0L,

    val currentFile: String = "",

    val status: String = "Ready"

)
