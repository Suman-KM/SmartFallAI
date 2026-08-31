package com.suman.smartfallai.wear.controller

import android.content.Context
import com.suman.smartfallai.wear.health.HealthManager
import com.suman.smartfallai.wear.communication.PhoneSyncManager
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
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable

class RecordingController(
    private val context: Context,
    private val healthManager: HealthManager
) : MessageClient.OnMessageReceivedListener {

    private val sensorManager =
        WatchSensorManager(context)

    private val gpsManager =
        GpsManager(context)

    private val csvLogger =
        CsvLogger(context)

    private val phoneSyncManager =
        PhoneSyncManager(context)

    val fallInferenceEngine =
        com.suman.smartfallai.wear.ml.FallInferenceEngine(context)

    private val scope =
        CoroutineScope(Dispatchers.IO)

    private var recordingJob: Job? = null

    private val _state =
        MutableStateFlow(RecordingState())

    val state: StateFlow<RecordingState> =
        _state.asStateFlow()

    val sensorData = sensorManager.sensorData

    val gpsData = gpsManager.gpsData

    private var currentActivity = com.suman.smartfallai.wear.ActivityLabel.WALKING.name

    private var currentSessionId = ""
    private var stopTimestampBoundary: Long = 0L

    init {
        Wearable.getMessageClient(context).addListener(this)
    }

    fun startRecording(activity: String, sessionId: String) {
        require(sessionId.isNotBlank()) { "Session ID must be provided by Phone" }

        if (_state.value.isRecording || _state.value.status == "Stopping" || _state.value.status == "Starting") {
            return
        }

        currentActivity = activity
        currentSessionId = sessionId

        val fileName =
            csvLogger.startLogging(activity, currentSessionId)

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
            try {
                for (sensor in sensorManager.sensorChannel) {

                    if (!sensor.isValid) {
                        continue
                    }

                    if (stopTimestampBoundary > 0L && sensor.timestamp > stopTimestampBoundary) {
                        continue
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

                            heartRate =
                                healthManager.heartRate.value.toFloat(),

                            activity =
                                currentActivity
                        )

                    csvLogger.log(
                        combinedSensor
                    )

                    // Feed 9-DoF IMU to Real-time Watch Random Forest Fall Inference Engine
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

                    phoneSyncManager.sendWatchSample(
                        sessionId = currentSessionId,
                        data = combinedSensor
                    )

                    _state.value =
                        _state.value.copy(
                            sampleCount =
                                _state.value.sampleCount + 1
                        )
                }
            } finally {
                csvLogger.stopLogging()
                fallInferenceEngine.reset()
                android.util.Log.d("RecordingController", "Recording stopped. " +
                    "Received: ${sensorManager.sensorEventsReceived}, " +
                    "Queued: ${sensorManager.sensorEventsQueued}, " +
                    "Dropped: ${sensorManager.sensorEventsDropped}")

                currentSessionId = ""
                stopTimestampBoundary = 0L

                _state.value =
                    _state.value.copy(
                        isRecording = false,
                        status = "Saved"
                    )
            }
        }
    }

    fun stopRecording(stopTimestamp: Long = 0L) {

        if (_state.value.status != "Recording") {
            return
        }

        stopTimestampBoundary = stopTimestamp

        _state.value =
            _state.value.copy(
                status = "Stopping"
            )

        sensorManager.stop()

        gpsManager.stop()
    }

    fun release() {

        Wearable.getMessageClient(context).removeListener(this)

        recordingJob?.cancel()
        recordingJob = null

        sensorManager.stop()

        gpsManager.stop()

        csvLogger.stopLogging()

        currentSessionId = ""
    }



    override fun onMessageReceived(messageEvent: MessageEvent) {
        val path = messageEvent.path
        if (path == "/smartfallai/start_recording") {
            val payload = String(messageEvent.data, Charsets.UTF_8)
            val parts = payload.split(",")
            if (parts.size >= 2) {
                val sessionId = parts[0]
                val activity = parts[1]
                
                val isValidActivity = com.suman.smartfallai.wear.ActivityLabel.entries.any { it.name == activity }
                if (isValidActivity) {
                    startRecording(activity, sessionId)
                }
            }
        } else if (path == "/smartfallai/stop_recording") {
            val payload = String(messageEvent.data, Charsets.UTF_8)
            val parts = payload.split(",")
            if (parts.size >= 2) {
                val sessionId = parts[0]
                val stopTimestamp = parts[1].toLongOrNull() ?: 0L
                if (sessionId == currentSessionId) {
                    stopRecording(stopTimestamp)
                }
            } else {
                stopRecording()
            }
        }
    }
}
