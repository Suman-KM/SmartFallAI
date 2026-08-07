package com.suman.smartfallai.controller

import android.content.Context
import com.suman.smartfallai.gps.GpsManager
import com.suman.smartfallai.sensors.PhoneSensorManager
import com.suman.smartfallai.storage.CsvLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class RecordingController(
    context: Context
) {

    private val phoneSensorManager =
        PhoneSensorManager(context)

    private val gpsManager =
        GpsManager(context)

    private val csvLogger =
        CsvLogger(context)

    private val scope =
        CoroutineScope(Dispatchers.IO)

    private var recordingJob: Job? = null

    private val _state =
        MutableStateFlow(RecordingState())

    val state: StateFlow<RecordingState> =
        _state.asStateFlow()

    private var currentActivity = "Walking"

    fun startRecording(activity: String) {

        if (_state.value.isRecording) return

        currentActivity = activity

        val fileName =
            csvLogger.startLogging(activity)

        _state.value = _state.value.copy(
            isRecording = true,
            activity = activity,
            sampleCount = 0,
            elapsedTime = 0,
            currentFile = fileName,
            status = "Recording"
        )

        gpsManager.start()

        phoneSensorManager.start()

        recordingJob = scope.launch {

            phoneSensorManager.sensorData.collectLatest { sensor ->

                val gps = gpsManager.currentLocation

                csvLogger.log(

                    sensor.copy(

                        latitude = gps.latitude,

                        longitude = gps.longitude,

                        altitude = gps.altitude,

                        speed = gps.speed,

                        accuracy = gps.accuracy,

                        activity = currentActivity

                    )

                )

                _state.value = _state.value.copy(

                    sampleCount = _state.value.sampleCount + 1

                )

            }

        }

    }

    fun stopRecording() {

        if (!_state.value.isRecording) return

        recordingJob?.cancel()

        phoneSensorManager.stop()

        gpsManager.stop()

        csvLogger.stopLogging()

        _state.value = _state.value.copy(

            isRecording = false,

            status = "Saved"

        )

    }

}