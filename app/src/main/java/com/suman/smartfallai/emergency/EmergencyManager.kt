package com.suman.smartfallai.emergency

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import androidx.core.app.NotificationCompat

class EmergencyManager(private val context: Context) {

    companion object {
        const val EMERGENCY_RECIPIENT = "sumankmdvg@gmail.com"
        const val CHANNEL_ID = "smartfall_emergency_channel"
        const val NOTIFICATION_ID = 911
        private const val TAG = "EmergencyManager"
    }

    init {
        createNotificationChannel()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "SmartFall Emergency Alerts",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Urgent notifications dispatched upon confirmed fall detection"
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
                longArrayOf(0, 250, 150, 250)
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
            Log.w(TAG, "Vibration failed: ${e.message}")
        }
    }

    fun sendEmergencyAlert(deviceSource: String, fallTimeMs: Long = System.currentTimeMillis()) {
        Log.i(TAG, "Initiating Emergency Alert Dispatch for device: $deviceSource")

        triggerVibration(isEmergency = true)

        val subject = "SMARTFALL AI — FALL DETECTED"
        val body = """
SmartFall AI detected a possible fall and the emergency countdown expired without user cancellation.

Device:
$deviceSource

Detection:
FALL_CONFIRMED

Timestamp:
${java.util.Date(fallTimeMs)}

This is an automated research prototype notification.
        """.trimIndent()

        // 1. Post High-Priority Notification
        try {
            val emailIntent = Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("mailto:$EMERGENCY_RECIPIENT")
                putExtra(Intent.EXTRA_SUBJECT, subject)
                putExtra(Intent.EXTRA_TEXT, body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }

            val pendingIntent = PendingIntent.getActivity(
                context,
                0,
                emailIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            val notification = NotificationCompat.Builder(context, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentTitle("⚠️ FALL CONFIRMED — SOS TRIGGERED")
                .setContentText("Emergency alert dispatched to $EMERGENCY_RECIPIENT")
                .setStyle(NotificationCompat.BigTextStyle().bigText("Fall confirmed on $deviceSource. Emergency email ready for $EMERGENCY_RECIPIENT."))
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .setCategory(NotificationCompat.CATEGORY_ALARM)
                .setContentIntent(pendingIntent)
                .setAutoCancel(true)
                .build()

            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(NOTIFICATION_ID, notification)
            Log.i(TAG, "High-priority emergency notification posted.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to post emergency notification: ${e.message}", e)
        }

        // 2. Direct Activity Launch for Email Intent
        try {
            val directEmailIntent = Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("mailto:$EMERGENCY_RECIPIENT")
                putExtra(Intent.EXTRA_SUBJECT, subject)
                putExtra(Intent.EXTRA_TEXT, body)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            context.startActivity(directEmailIntent)
            Log.i(TAG, "Email dispatch intent successfully launched for $EMERGENCY_RECIPIENT")
        } catch (e: Exception) {
            Log.w(TAG, "Direct email intent launch fallback: ${e.message}")
        }
    }
}