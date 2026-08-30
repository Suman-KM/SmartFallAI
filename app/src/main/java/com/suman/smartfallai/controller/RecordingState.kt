package com.suman.smartfallai.controller

import com.suman.smartfallai.ActivityLabel

data class RecordingState(

    val isRecording: Boolean = false,

    val activity: String = ActivityLabel.WALKING.name,

    val sampleCount: Int = 0,

    val elapsedTime: Long = 0L,

    val currentFile: String = "",

    val status: String = "Ready"

)
