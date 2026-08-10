package com.suman.smartfallai.wear.controller

import android.content.Context
import com.suman.smartfallai.wear.gps.GpsManager
import com.suman.smartfallai.wear.sensors.WatchSensorManager
import com.suman.smartfallai.wear.storage.CsvLogger
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

    private val sensorManager =
        WatchSensorManager(context)

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

    private var currentActivity = "Unknown"

    fun startRecording(activity: String) {

        if (_state.value.isRecording) {
            return
        }

        currentActivity = activity

        val fileName =
            csvLogger.startLogging(activity)

        _state.value =
            _state.value.copy(
                isRecording = true,
                activity = activity,
                sampleCount = 0L,
                elapsedTime = 0L,
                currentFile = fileName,
                status = "Recording"
            )

        gpsManager.start()

        sensorManager.start()

        recordingJob = scope.launch {

            sensorManager.sensorData.collectLatest { sensor ->

                // Ignore the initial dummy StateFlow value.
                if (!sensor.isValid) {
                    return@collectLatest
                }

                val gps =
                    gpsManager.gpsData.value

                val combinedSensor =
                    sensor.copy(

                        latitude =
                            gps?.latitude ?: 0.0,

                        longitude =
                            gps?.longitude ?: 0.0,

                        altitude =
                            gps?.altitude ?: 0.0,

                        speed =
                            gps?.speed ?: 0f,

                        accuracy =
                            gps?.accuracy ?: 0f,

                        activity =
                            currentActivity
                    )

                csvLogger.log(
                    combinedSensor
                )

                _state.value =
                    _state.value.copy(
                        sampleCount =
                            _state.value.sampleCount + 1
                    )
            }
        }
    }

    fun stopRecording() {

        if (!_state.value.isRecording) {
            return
        }

        recordingJob?.cancel()
        recordingJob = null

        sensorManager.stop()

        gpsManager.stop()

        csvLogger.stopLogging()

        _state.value =
            _state.value.copy(
                isRecording = false,
                status = "Saved"
            )
    }

    fun release() {

        recordingJob?.cancel()
        recordingJob = null

        sensorManager.stop()

        gpsManager.stop()

        csvLogger.stopLogging()
    }
}