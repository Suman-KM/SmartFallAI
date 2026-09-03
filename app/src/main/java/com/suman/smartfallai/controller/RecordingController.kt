package com.suman.smartfallai.controller

import com.suman.smartfallai.ActivityLabel
import android.content.Context
import com.suman.smartfallai.communication.WatchManager
import com.suman.smartfallai.gps.GpsManager
import com.suman.smartfallai.sensors.PhoneSensorManager
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

    private val watchManager =
        WatchManager(context)

    val fallInferenceEngine =
        com.suman.smartfallai.ml.FallInferenceEngine(context)

    private val scope =
        CoroutineScope(Dispatchers.IO)

    private var recordingJob: Job? = null

    private val _state =
        MutableStateFlow(RecordingState())

    val state: StateFlow<RecordingState> =
        _state.asStateFlow()

    private var currentActivity = ActivityLabel.WALKING.name

    fun startRecording(activity: String) {

        if (_state.value.isRecording) return

        currentActivity = activity

        val dateFormat = java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.getDefault())
        val shortUuid = java.util.UUID.randomUUID().toString().substring(0, 4).uppercase(java.util.Locale.getDefault())
        val sessionId = "SESSION_${dateFormat.format(java.util.Date())}_$shortUuid"

        val fileName = "${sessionId}_IN_MEMORY"

        watchManager.startRecordingSession(activity, sessionId)

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

                // Feed 9-DoF IMU directly to Real-time ONNX Fall Inference Engine (in-memory)
                fallInferenceEngine.addSample(
                    accX = sensor.accX,
                    accY = sensor.accY,
                    accZ = sensor.accZ,
                    gyroX = sensor.gyroX,
                    gyroY = sensor.gyroY,
                    gyroZ = sensor.gyroZ,
                    pitch = sensor.pitch,
                    roll = sensor.roll,
                    yaw = sensor.yaw
                )

                _state.value = _state.value.copy(

                    sampleCount = _state.value.sampleCount + 1

                )

            }

        }

    }

    fun stopRecording() {

        if (!_state.value.isRecording) return

        val stopTimestamp = System.currentTimeMillis()

        recordingJob?.cancel()

        phoneSensorManager.stop()

        gpsManager.stop()

        watchManager.stopRecordingSession(stopTimestamp)

        fallInferenceEngine.reset()

        _state.value = _state.value.copy(

            isRecording = false,

            status = "Saved"

        )

    }

}
