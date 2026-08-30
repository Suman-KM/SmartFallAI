package com.suman.smartfallai.wear.controller

data class RecordingState(

    val isRecording: Boolean = false,

    val activity: String = com.suman.smartfallai.wear.ActivityLabel.WALKING.name,

    val sampleCount: Long = 0L,

    val elapsedTime: Long = 0L,

    val currentFile: String = "",

    val status: String = "Waiting for Phone..."

)