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

    val fallState: StateFlow<com.suman.smartfallai.ml.FallState> =
        recordingController.fallInferenceEngine.currentState

    val countdownRemaining: StateFlow<Int> =
        recordingController.fallInferenceEngine.countdownRemaining

    val emailDeliveryStatus: StateFlow<com.suman.smartfallai.emergency.EmailDeliveryStatus> =
        com.suman.smartfallai.emergency.EmergencyManager.globalDeliveryStatus

    fun cancelFallAlert() {
        recordingController.fallInferenceEngine.cancelCountdown()
    }

    fun dismissAlert() {
        recordingController.fallInferenceEngine.dismissAlert()
    }

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