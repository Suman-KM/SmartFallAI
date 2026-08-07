package com.suman.smartfallai.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.suman.smartfallai.controller.RecordingController
import com.suman.smartfallai.controller.RecordingState
import com.suman.smartfallai.gps.GpsData
import com.suman.smartfallai.gps.GpsManager
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class SmartFallViewModel(
    application: Application
) : AndroidViewModel(application) {

    private val recordingController =
        RecordingController(application)

    private val gpsManager =
        GpsManager(application)

    val recordingState: StateFlow<RecordingState> =
        recordingController.state

    val gpsData: StateFlow<GpsData> =
        gpsManager.gpsData

    fun startRecording(activity: String) {

        viewModelScope.launch {
            recordingController.startRecording(activity)
        }

    }

    fun stopRecording() {

        viewModelScope.launch {
            recordingController.stopRecording()
        }

    }

    fun startGps() {

        gpsManager.start()

    }

    fun stopGps() {

        gpsManager.stop()

    }

}