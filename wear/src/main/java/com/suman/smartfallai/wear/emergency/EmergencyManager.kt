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

class EmergencyManager(private val context: Context) {

    companion object {
        const val EMERGENCY_RECIPIENT = "sumankmdvg@gmail.com"
        const val CHANNEL_ID = "wear_emergency_channel"
        const val NOTIFICATION_ID = 912
        const val SOS_TRIGGERED_PATH = "/smartfallai/sos_triggered"
        private const val TAG = "WearEmergencyManager"
    }

    private val messageClient = Wearable.getMessageClient(context)
    private val nodeClient = Wearable.getNodeClient(context)

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

    fun sendEmergencyAlert(fallTimeMs: Long = System.currentTimeMillis()) {
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

        // 2. Transmit to connected Phone via Wearable Data Layer
        try {
            val payload = "FALL_CONFIRMED,WATCH_SM_R870,$fallTimeMs,$EMERGENCY_RECIPIENT".toByteArray(Charsets.UTF_8)
            nodeClient.connectedNodes.addOnSuccessListener { nodes ->
                if (nodes.isEmpty()) {
                    Log.w(TAG, "No connected phone nodes found. Standalone emergency alert active.")
                } else {
                    nodes.forEach { node ->
                        messageClient.sendMessage(node.id, SOS_TRIGGERED_PATH, payload)
                            .addOnSuccessListener {
                                Log.i(TAG, "Successfully sent SOS signal to phone node ${node.displayName} (${node.id})")
                            }
                            .addOnFailureListener { err ->
                                Log.e(TAG, "Failed to send SOS signal to phone: ${err.message}")
                            }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Wearable message dispatch error: ${e.message}")
        }
    }
}