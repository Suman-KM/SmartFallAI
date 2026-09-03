package com.suman.smartfallai.wear.emergency

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.android.gms.wearable.Wearable
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

class EmergencyManager(private val context: Context) {

    companion object {
        const val EMERGENCY_RECIPIENT = "sumankmdvg@gmail.com"
        const val CHANNEL_ID = "wear_emergency_channel"
        const val NOTIFICATION_ID = 912
        const val SOS_TRIGGERED_PATH = "/smartfallai/sos_triggered"
        private const val TAG = "WearEmergencyManager"

        val CANDIDATE_ENDPOINTS = listOf(
            "http://192.168.1.11:8000/api/v1/emergency",
            "http://127.0.0.1:8000/api/v1/emergency"
        )
    }

    private val messageClient = Wearable.getMessageClient(context)
    private val nodeClient = Wearable.getNodeClient(context)
    private val scope = CoroutineScope(Dispatchers.IO)

    init {
        createNotificationChannel()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "SmartFall Wear Emergency Alerts",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Wear OS Fall Emergency Alerts"
                enableVibration(true)
                vibrationPattern = longArrayOf(0, 500, 200, 500, 200, 500)
            }
            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannel(channel)
        }
    }

    fun triggerVibration(isEmergency: Boolean = false) {
        try {
            val pattern = if (isEmergency) {
                longArrayOf(0, 600, 200, 600, 200, 600)
            } else {
                longArrayOf(0, 200, 100, 200)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vibratorManager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
                vibratorManager?.defaultVibrator?.vibrate(VibrationEffect.createWaveform(pattern, -1))
            } else {
                @Suppress("DEPRECATION")
                val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
                @Suppress("DEPRECATION")
                vibrator?.vibrate(pattern, -1)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Watch vibration failed: ${e.message}")
        }
    }

    fun sendEmergencyAlert(
        fallTimeMs: Long = System.currentTimeMillis(),
        heartRate: Int? = null,
        latitude: Double? = null,
        longitude: Double? = null,
        accuracy: Float? = null
    ) {
        Log.i(TAG, "Wear OS Fall Confirmed — Triggering Emergency Escalation")
        triggerVibration(isEmergency = true)

        // 1. Post local notification on watch
        try {
            val notification = NotificationCompat.Builder(context, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentTitle("⚠️ FALL CONFIRMED")
                .setContentText("Emergency SOS Triggered")
                .setStyle(NotificationCompat.BigTextStyle().bigText("Emergency alert dispatched. Contacting $EMERGENCY_RECIPIENT."))
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .setAutoCancel(true)
                .build()

            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to post wear notification: ${e.message}")
        }

        // 2. Transmit to connected Phone via Wearable Data Layer (with telemetry)
        try {
            val hrVal = if (heartRate != null && heartRate > 0) heartRate else (if (com.suman.smartfallai.wear.health.HealthManager.latestBpm > 0) com.suman.smartfallai.wear.health.HealthManager.latestBpm else -1)
            val latVal = latitude ?: 0.0
            val lonVal = longitude ?: 0.0
            val accVal = accuracy ?: 0f
            val payload = "FALL_CONFIRMED,WATCH_SM_R870,$fallTimeMs,$EMERGENCY_RECIPIENT,$hrVal,$latVal,$lonVal,$accVal".toByteArray(Charsets.UTF_8)
            
            nodeClient.connectedNodes.addOnSuccessListener { nodes ->
                if (nodes.isEmpty()) {
                    Log.w(TAG, "No connected phone nodes found. Standalone emergency alert active.")
                    // Fallback to standalone direct HTTP dispatch over Watch Wi-Fi
                    dispatchStandaloneHttpAlert(fallTimeMs, heartRate, latitude, longitude, accuracy)
                } else {
                    nodes.forEach { node ->
                        messageClient.sendMessage(node.id, SOS_TRIGGERED_PATH, payload)
                            .addOnSuccessListener {
                                Log.i(TAG, "Successfully sent SOS signal to phone node ${node.displayName} (${node.id})")
                            }
                            .addOnFailureListener { err ->
                                Log.e(TAG, "Failed to send SOS signal to phone: ${err.message}. Attempting standalone fallback.")
                                dispatchStandaloneHttpAlert(fallTimeMs, heartRate, latitude, longitude, accuracy)
                            }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Wearable message dispatch error: ${e.message}")
            dispatchStandaloneHttpAlert(fallTimeMs, heartRate, latitude, longitude, accuracy)
        }
    }

    private fun dispatchStandaloneHttpAlert(
        fallTimeMs: Long,
        heartRate: Int?,
        latitude: Double?,
        longitude: Double?,
        accuracy: Float?
    ) {
        scope.launch {
            val eventId = UUID.randomUUID().toString()
            val timeFormat = SimpleDateFormat("dd MMMM yyyy, HH:mm:ss", Locale.ENGLISH)
            val formattedTime = timeFormat.format(Date(fallTimeMs))

            val payload = JSONObject().apply {
                put("event", "FALL_CONFIRMED")
                put("deviceSource", "Samsung Galaxy Watch 4 (SM-R870) [Standalone]")
                put("timestamp", fallTimeMs)
                put("timeString", formattedTime)
                put("eventId", eventId)
                if (heartRate != null && heartRate > 0) put("heartRate", heartRate) else put("heartRate", JSONObject.NULL)
                if (latitude != null && longitude != null && (Math.abs(latitude) > 0.0001)) {
                    put("latitude", latitude)
                    put("longitude", longitude)
                    if (accuracy != null) put("accuracy", accuracy)
                } else {
                    put("latitude", JSONObject.NULL)
                    put("longitude", JSONObject.NULL)
                }
                val recArray = JSONArray()
                recArray.put(EMERGENCY_RECIPIENT)
                put("recipients", recArray)
            }

            for (endpoint in CANDIDATE_ENDPOINTS) {
                try {
                    val url = URL(endpoint)
                    val conn = (url.openConnection() as HttpURLConnection).apply {
                        requestMethod = "POST"
                        connectTimeout = 4000
                        readTimeout = 8000
                        doOutput = true
                        setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                    }
                    OutputStreamWriter(conn.outputStream, "UTF-8").use { writer ->
                        writer.write(payload.toString())
                        writer.flush()
                    }
                    if (conn.responseCode in 200..299) {
                        Log.i(TAG, "Standalone emergency email sent successfully via $endpoint")
                        break
                    }
                } catch (e: Exception) {
                    Log.d(TAG, "Standalone dispatch to $endpoint failed: ${e.message}")
                }
            }
        }
    }
}