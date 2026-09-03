package com.suman.smartfallai.wear.ui


import android.app.Application

import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope

import com.suman.smartfallai.wear.controller.RecordingController
import com.suman.smartfallai.wear.gps.GpsManager
import com.suman.smartfallai.wear.health.HealthManager


import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch



class SmartFallViewModel(
    application: Application
) : AndroidViewModel(application) {



    private val context =
        getApplication<Application>()

    private val healthManager =
        HealthManager(context)

    private val recordingController =
        RecordingController(context, healthManager)


    val recordingState =
        recordingController.state

    val fallState =
        recordingController.fallInferenceEngine.currentState

    val countdownRemaining =
        recordingController.fallInferenceEngine.countdownRemaining

    fun cancelFallAlert() {
        recordingController.fallInferenceEngine.cancelCountdown()
    }

    fun dismissAlert() {
        recordingController.fallInferenceEngine.dismissAlert()
    }

    val sensorData =
        recordingController.sensorData

    val gpsData =
        recordingController.gpsData

    private val _heartRate =
        MutableStateFlow(-1)

    val heartRate =
        _heartRate.asStateFlow()






    fun startHeartRate() {
        viewModelScope.launch {
            healthManager.start()
        }
    }

    init {


        viewModelScope.launch {


            healthManager.start()


        }



        viewModelScope.launch {


            healthManager.heartRate.collect { bpm ->


                _heartRate.value = bpm


            }


        }


    }













    fun startRecording(activity: String) {
        val dateFormat = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.getDefault())
        val shortUuid = java.util.UUID.randomUUID().toString().substring(0, 4).uppercase(java.util.Locale.getDefault())
        recordingController.startRecording(activity, sessionId = "SESSION_${dateFormat.format(java.util.Date())}_$shortUuid")
    }

    fun stopRecording(){


        recordingController.stopRecording()


    }




    override fun onCleared(){


        recordingController.release()

        healthManager.stop()


        super.onCleared()


    }


}