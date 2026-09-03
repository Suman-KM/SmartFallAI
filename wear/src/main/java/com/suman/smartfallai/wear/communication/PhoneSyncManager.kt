package com.suman.smartfallai.wear.communication

import android.content.Context
import com.google.android.gms.wearable.Wearable
import com.suman.smartfallai.wear.model.SensorData

class PhoneSyncManager(
    context: Context
) {

    private val messageClient =
        Wearable.getMessageClient(context)

    private val nodeClient =
        Wearable.getNodeClient(context)

    fun sendWatchSample(
        sessionId: String,
        data: SensorData
    ) {

        val payload =
            buildPayload(
                sessionId = sessionId,
                data = data
            ).toByteArray(Charsets.UTF_8)

        nodeClient.connectedNodes
            .addOnSuccessListener { nodes ->

                nodes.forEach { node ->

                    messageClient.sendMessage(
                        node.id,
                        WATCH_SAMPLE_PATH,
                        payload
                    )
                }
            }
    }

    private fun buildPayload(
        sessionId: String,
        data: SensorData
    ): String {

        return listOf(
            sessionId,
            data.timestamp,
            data.accX,
            data.accY,
            data.accZ,
            data.gyroX,
            data.gyroY,
            data.gyroZ,
            data.pitch,
            data.roll,
            data.yaw,
            data.latitude,
            data.longitude,
            data.altitude,
            data.speed,
            data.accuracy,
            data.heartRate,
            data.activity
        ).joinToString(",")
    }

    companion object {

        const val WATCH_SAMPLE_PATH =
            "/smartfallai/watch_sample"
    }
}
