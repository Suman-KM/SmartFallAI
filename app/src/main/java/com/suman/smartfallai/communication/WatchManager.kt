package com.suman.smartfallai.communication

import android.content.Context
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable

class WatchManager(
    context: Context
) : MessageClient.OnMessageReceivedListener {

    private val messageClient =
        Wearable.getMessageClient(context)

    private var isReceiving = false
    private val nodeClient = Wearable.getNodeClient(context)
    private var currentSessionId = ""

    private val emergencyManager =
        com.suman.smartfallai.emergency.EmergencyManager(context)

    init {
        messageClient.addListener(this)
    }

    fun startRecordingSession(activity: String, sessionId: String): String {
        currentSessionId = sessionId
        isReceiving = true

        val payload = "$sessionId,$activity".toByteArray(Charsets.UTF_8)
        nodeClient.connectedNodes.addOnSuccessListener { nodes ->
            nodes.forEach { node ->
                messageClient.sendMessage(node.id, START_RECORDING_PATH, payload)
            }
        }

        return "${sessionId}_WATCH"
    }

    fun stopRecordingSession(stopTimestamp: Long) {
        if (!isReceiving) return

        isReceiving = false

        val payload = "$currentSessionId,$stopTimestamp".toByteArray(Charsets.UTF_8)
        currentSessionId = ""

        nodeClient.connectedNodes.addOnSuccessListener { nodes ->
            nodes.forEach { node ->
                messageClient.sendMessage(node.id, STOP_RECORDING_PATH, payload)
            }
        }
    }

    override fun onMessageReceived(
        messageEvent: MessageEvent
    ) {
        when (messageEvent.path) {
            WATCH_SAMPLE_PATH -> {
                // In-memory runtime: sample streaming to disk eliminated
            }
            SOS_TRIGGERED_PATH -> {
                val dataStr = String(messageEvent.data, Charsets.UTF_8)
                android.util.Log.i("WatchManager", "Received SOS trigger from Watch: $dataStr")
                val parts = dataStr.split(",")
                val timestamp = parts.getOrNull(2)?.toLongOrNull() ?: System.currentTimeMillis()
                val heartRate = parts.getOrNull(4)?.toIntOrNull()
                val latitude = parts.getOrNull(5)?.toDoubleOrNull()
                val longitude = parts.getOrNull(6)?.toDoubleOrNull()
                val accuracy = parts.getOrNull(7)?.toFloatOrNull()
                emergencyManager.sendEmergencyAlert(
                    deviceSource = "Samsung Galaxy Watch 4 (SM-R870)",
                    fallTimeMs = timestamp,
                    heartRate = heartRate,
                    latitude = latitude,
                    longitude = longitude,
                    accuracy = accuracy
                )
            }
        }
    }

    companion object {
        const val WATCH_SAMPLE_PATH = "/smartfallai/watch_sample"
        const val START_RECORDING_PATH = "/smartfallai/start_recording"
        const val STOP_RECORDING_PATH = "/smartfallai/stop_recording"
        const val SOS_TRIGGERED_PATH = "/smartfallai/sos_triggered"
    }
}
