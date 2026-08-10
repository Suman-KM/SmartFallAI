package com.suman.smartfallai.wear.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.suman.smartfallai.wear.controller.RecordingController
import com.suman.smartfallai.wear.controller.RecordingState
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class SmartFallViewModel(
    application: Application
) : AndroidViewModel(application) {

    private val recordingController =
        RecordingController(application)

    val recordingState: StateFlow<RecordingState> =
        recordingController.state

    fun startRecording(activity: String) {

        viewModelScope.launch {

            recordingController.startRecording(
                activity
            )
        }
    }

    fun stopRecording() {

        viewModelScope.launch {

            recordingController.stopRecording()
        }
    }

    override fun onCleared() {

        recordingController.release()

        super.onCleared()
    }
}