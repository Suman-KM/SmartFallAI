package com.suman.smartfallai.emergency

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import androidx.core.app.NotificationCompat
import com.suman.smartfallai.gps.GpsManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
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

/**
 * Emergency Email Delivery States adhering to Phase 14 specifications.
 */
enum class EmailDeliveryStatus {
    IDLE,
    SENDING,
    SENT,
    FAILED
}

class EmergencyManager(private val context: Context) {

    companion object {
        // Internal configuration for emergency recipients (Phase 14: 1 recipient, expandable to 3)
        val emergencyRecipients = listOf(
            "sumankmdvg@gmail.com"
        )
        const val EMERGENCY_RECIPIENT = "sumankmdvg@gmail.com"
        const val CHANNEL_ID = "smartfall_emergency_channel"
        const val NOTIFICATION_ID = 911
        private const val TAG = "EmergencyManager"

        private val _globalDeliveryStatus = MutableStateFlow(EmailDeliveryStatus.IDLE)
        val globalDeliveryStatus: StateFlow<EmailDeliveryStatus> = _globalDeliveryStatus.asStateFlow()

        fun setGlobalStatus(status: EmailDeliveryStatus) {
            _globalDeliveryStatus.value = status
        }

        // Candidate backend endpoints for local development & physical device testing
        // Tries adb-reverse localhost first, then local Wi-Fi IP, then Android emulator host
        val CANDIDATE_ENDPOINTS = listOf(
            "http://127.0.0.1:8000/api/v1/emergency",
            "http://192.168.1.11:8000/api/v1/emergency",
            "http://10.0.2.2:8000/api/v1/emergency"
        )
    }

    private val gpsManager = GpsManager(context)
    private val scope = CoroutineScope(Dispatchers.IO)

    private val _deliveryStatus = MutableStateFlow(EmailDeliveryStatus.IDLE)
    val deliveryStatus: StateFlow<EmailDeliveryStatus> = _deliveryStatus.asStateFlow()

    // Duplicate protection: ensure one confirmed fall event produces exactly one email
    private var lastDispatchedEventId: String? = null
    private var lastDispatchedTimestamp: Long = 0L

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

    /**
     * Confirmed Fall Dispatch Entry Point.
     * Automatically formats telemetry and transmits email to emergency recipients without user interaction.
     */
    fun sendEmergencyAlert(
        deviceSource: String,
        fallTimeMs: Long = System.currentTimeMillis(),
        heartRate: Int? = null,
        latitude: Double? = null,
        longitude: Double? = null,
        accuracy: Float? = null
    ) {
        // 1. Duplicate protection check (within 30 seconds for same device/fall)
        val now = System.currentTimeMillis()
        if (now - lastDispatchedTimestamp < 30000L && _deliveryStatus.value == EmailDeliveryStatus.SENDING) {
            Log.w(TAG, "Duplicate emergency alert suppressed (Dispatched ${now - lastDispatchedTimestamp}ms ago)")
            return
        }
        lastDispatchedTimestamp = now
        val eventId = UUID.randomUUID().toString()
        lastDispatchedEventId = eventId

        Log.i(TAG, "Initiating Automatic Emergency Email Dispatch for device: $deviceSource (Event: $eventId)")
        triggerVibration(isEmergency = true)

        // 2. Resolve GPS Location (prefer passed coordinates if provided by Watch, else Phone's live GPS)
        val gps = gpsManager.currentLocation
        val finalLat = if (latitude != null && (Math.abs(latitude) > 0.0001)) latitude else (if (Math.abs(gps.latitude) > 0.0001) gps.latitude else null)
        val finalLon = if (longitude != null && (Math.abs(longitude) > 0.0001)) longitude else (if (Math.abs(gps.longitude) > 0.0001) gps.longitude else null)
        val finalAcc = if (accuracy != null && accuracy > 0f) accuracy else (if (gps.accuracy > 0f) gps.accuracy else null)

        // 3. Resolve Heart Rate
        val validHr = if (heartRate != null && heartRate > 0) heartRate else null

        // 4. Format Timestamp
        val timeFormat = SimpleDateFormat("dd MMMM yyyy, HH:mm:ss", Locale.ENGLISH)
        val formattedTime = timeFormat.format(Date(fallTimeMs))

        // 5. Build Payload via pure helper
        val payloadMap = buildEmergencyPayload(
            deviceSource = deviceSource,
            fallTimeMs = fallTimeMs,
            heartRate = validHr,
            latitude = finalLat,
            longitude = finalLon,
            accuracy = finalAcc
        ).toMutableMap()
        payloadMap["eventId"] = eventId

        val payload = JSONObject(payloadMap)

        // 6. Post initial local emergency notification
        postLocalNotification(
            title = "⚠️ FALL CONFIRMED — SOS TRIGGERED",
            text = "Transmitting emergency email to ${emergencyRecipients.joinToString()}..."
        )

        // 7. Dispatch Automatic Network Email in Background Coroutine (NO Intent composer, NO manual send)
        _deliveryStatus.value = EmailDeliveryStatus.SENDING
        setGlobalStatus(EmailDeliveryStatus.SENDING)
        scope.launch {
            dispatchAutomaticEmail(payload, eventId)
        }
    }

    fun buildEmergencyPayload(
        deviceSource: String,
        fallTimeMs: Long,
        heartRate: Int?,
        latitude: Double?,
        longitude: Double?,
        accuracy: Float?
    ): Map<String, Any?> {
        val timeFormat = SimpleDateFormat("dd MMMM yyyy, HH:mm:ss", Locale.ENGLISH)
        val formattedTime = timeFormat.format(Date(fallTimeMs))

        val validHr = if (heartRate != null && heartRate > 0) heartRate else null
        val validLat = if (latitude != null && Math.abs(latitude) > 0.0001) latitude else null
        val validLon = if (longitude != null && Math.abs(longitude) > 0.0001) longitude else null
        val validAcc = if (accuracy != null && accuracy > 0f) accuracy else null

        return mapOf(
            "event" to "FALL_CONFIRMED",
            "deviceSource" to deviceSource,
            "timestamp" to fallTimeMs,
            "timeString" to formattedTime,
            "heartRate" to validHr,
            "latitude" to validLat,
            "longitude" to validLon,
            "accuracy" to validAcc,
            "recipients" to emergencyRecipients
        )
    }

    /**
     * Executes the HTTP request against available endpoints with safe error handling and fallback.
     */
    private fun dispatchAutomaticEmail(payload: JSONObject, eventId: String) {
        var success = false
        var lastErrorMessage = "Unknown network error"

        for (endpoint in CANDIDATE_ENDPOINTS) {
            try {
                Log.d(TAG, "Attempting emergency email transmission via: $endpoint")
                val url = URL(endpoint)
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 4000
                    readTimeout = 8000
                    doOutput = true
                    doInput = true
                    setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                    setRequestProperty("Accept", "application/json")
                }

                OutputStreamWriter(conn.outputStream, "UTF-8").use { writer ->
                    writer.write(payload.toString())
                    writer.flush()
                }

                val responseCode = conn.responseCode
                Log.i(TAG, "Emergency dispatch response from $endpoint: HTTP $responseCode")

                if (responseCode in 200..299) {
                    val responseBody = conn.inputStream.bufferedReader().use { it.readText() }
                    Log.i(TAG, "Emergency email transmitted successfully. Response: $responseBody")
                    success = true
                    break
                } else {
                    val errBody = conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "No error body"
                    lastErrorMessage = "HTTP $responseCode: $errBody"
                    Log.w(TAG, "Endpoint $endpoint rejected dispatch: $lastErrorMessage")
                }
            } catch (e: Exception) {
                lastErrorMessage = "${e.javaClass.simpleName}: ${e.message}"
                Log.d(TAG, "Connection attempt to $endpoint failed: $lastErrorMessage")
            }
        }

        if (success) {
            _deliveryStatus.value = EmailDeliveryStatus.SENT
            setGlobalStatus(EmailDeliveryStatus.SENT)
            Log.i(TAG, "Emergency email successfully sent to ${emergencyRecipients.joinToString()}")
            postLocalNotification(
                title = "🚨 EMERGENCY EMAIL SENT",
                text = "Emergency alert delivered to ${emergencyRecipients.joinToString()}"
            )
        } else {
            _deliveryStatus.value = EmailDeliveryStatus.FAILED
            setGlobalStatus(EmailDeliveryStatus.FAILED)
            Log.e(TAG, "Emergency email failed: $lastErrorMessage")
            postLocalNotification(
                title = "⚠️ EMERGENCY EMAIL FAILED",
                text = "Failed to deliver emergency alert. Please call emergency services."
            )
        }
    }

    private fun postLocalNotification(title: String, text: String) {
        try {
            val notification = NotificationCompat.Builder(context, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentTitle(title)
                .setContentText(text)
                .setStyle(NotificationCompat.BigTextStyle().bigText(text))
                .setPriority(NotificationCompat.PRIORITY_MAX)
                .setCategory(NotificationCompat.CATEGORY_ALARM)
                .setAutoCancel(true)
                .build()

            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to post emergency notification: ${e.message}")
        }
    }

    /**
     * Controlled Test Mode for Development Verification (Section 21).
     * Dispatches an emergency test email without requiring an uncontrolled physical fall.
     */
    fun sendTestEmergencyAlert() {
        Log.i(TAG, "Triggering Controlled TEST Emergency Alert")
        sendEmergencyAlert(
            deviceSource = "SmartFall AI Test Mode (Phone)",
            fallTimeMs = System.currentTimeMillis(),
            heartRate = 72,
            latitude = 14.4594,
            longitude = 75.9240,
            accuracy = 10.0f
        )
    }
}