package com.suman.smartfallai.communication

import android.content.Context
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable
import com.suman.smartfallai.storage.WatchCsvLogger

class WatchManager(
    context: Context
) : MessageClient.OnMessageReceivedListener {

    private val messageClient =
        Wearable.getMessageClient(context)

    private val watchCsvLogger =
        WatchCsvLogger(context)

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
        val fileName = watchCsvLogger.startLogging(activity, sessionId)
        isReceiving = true

        val payload = "$sessionId,$activity".toByteArray(Charsets.UTF_8)
        nodeClient.connectedNodes.addOnSuccessListener { nodes ->
            nodes.forEach { node ->
                messageClient.sendMessage(node.id, START_RECORDING_PATH, payload)
            }
        }

        return fileName
    }

    fun stopRecordingSession(stopTimestamp: Long) {
        if (!isReceiving) return

        watchCsvLogger.stopLogging()
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
                if (isReceiving) {
                    watchCsvLogger.log(
                        String(messageEvent.data, Charsets.UTF_8)
                    )
                }
            }
            SOS_TRIGGERED_PATH -> {
                val dataStr = String(messageEvent.data, Charsets.UTF_8)
                android.util.Log.i("WatchManager", "Received SOS trigger from Watch: $dataStr")
                val parts = dataStr.split(",")
                val timestamp = parts.getOrNull(2)?.toLongOrNull() ?: System.currentTimeMillis()
                emergencyManager.sendEmergencyAlert("Samsung Galaxy Watch 4 (SM-R870)", timestamp)
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
