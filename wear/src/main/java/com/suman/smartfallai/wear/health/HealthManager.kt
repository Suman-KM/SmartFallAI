package com.suman.smartfallai.wear.health

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.health.services.client.HealthServices
import androidx.health.services.client.MeasureCallback
import androidx.health.services.client.MeasureClient
import androidx.health.services.client.data.Availability
import androidx.health.services.client.data.DataPointContainer
import androidx.health.services.client.data.DataType
import androidx.health.services.client.data.DataTypeAvailability
import androidx.health.services.client.data.DeltaDataType
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class HealthManager(
    private val context: Context
) {
    companion object {
        private const val TAG = "HealthManager"
        @Volatile
        var latestBpm: Int = -1
    }

    private val measureClient: MeasureClient =
        HealthServices.getClient(context).measureClient

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager

    private val hardwareHrSensor: Sensor? =
        sensorManager?.getDefaultSensor(Sensor.TYPE_HEART_RATE)

    private val _heartRate = MutableStateFlow(-1)
    val heartRate = _heartRate.asStateFlow()

    private var isHealthServicesRegistered = false
    private var isHardwareListenerRegistered = false

    // 1. AndroidX Health Services MeasureCallback
    private val healthServicesCallback = object : MeasureCallback {
        override fun onDataReceived(data: DataPointContainer) {
            val values = data.getData(DataType.HEART_RATE_BPM)
            if (values.isNotEmpty()) {
                val bpm = values.last().value.toInt()
                if (bpm > 0) {
                    Log.i(TAG, "HeartRate: Health Services BPM=$bpm")
                    _heartRate.value = bpm
                    latestBpm = bpm
                }
            }
        }

        override fun onAvailabilityChanged(
            dataType: DeltaDataType<*, *>,
            availability: Availability
        ) {
            val availStr = when (availability) {
                DataTypeAvailability.AVAILABLE -> "AVAILABLE"
                DataTypeAvailability.ACQUIRING -> "ACQUIRING"
                DataTypeAvailability.UNAVAILABLE -> "UNAVAILABLE"
                DataTypeAvailability.UNKNOWN -> "UNKNOWN"
                else -> availability.toString()
            }
            Log.i(TAG, "HeartRate: Health Services availability=$availStr")
        }
    }

    // 2. Hardware PPG Sensor Fallback Listener
    private val hardwareSensorListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent?) {
            if (event?.sensor?.type == Sensor.TYPE_HEART_RATE && event.values.isNotEmpty()) {
                val bpm = event.values[0].toInt()
                if (bpm > 0) {
                    Log.i(TAG, "HeartRate: Hardware PPG Sensor BPM=$bpm (accuracy=${event.accuracy})")
                    _heartRate.value = bpm
                    latestBpm = bpm
                }
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
            Log.d(TAG, "HeartRate: Hardware PPG sensor accuracy=$accuracy")
        }
    }

    fun hasSensorPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.BODY_SENSORS
        ) == PackageManager.PERMISSION_GRANTED
    }

    suspend fun start() {
        if (!hasSensorPermission()) {
            Log.w(TAG, "HeartRate: BODY_SENSORS permission not granted")
            return
        }

        Log.i(TAG, "HeartRate: Starting dual-engine Heart Rate monitoring")

        // Start Primary Engine: AndroidX Health Services
        try {
            if (!isHealthServicesRegistered) {
                measureClient.registerMeasureCallback(
                    DataType.HEART_RATE_BPM,
                    healthServicesCallback
                )
                isHealthServicesRegistered = true
                Log.i(TAG, "HeartRate: Health Services MeasureClient registered successfully")
            }
        } catch (e: Exception) {
            Log.w(TAG, "HeartRate: Health Services registration notice: ${e.message}")
        }

        // Start Fallback Engine: Android Hardware Sensor TYPE_HEART_RATE
        try {
            if (hardwareHrSensor != null && !isHardwareListenerRegistered) {
                val registered = sensorManager?.registerListener(
                    hardwareSensorListener,
                    hardwareHrSensor,
                    SensorManager.SENSOR_DELAY_NORMAL
                ) ?: false
                isHardwareListenerRegistered = registered
                Log.i(TAG, "HeartRate: Hardware PPG sensor listener registered: $registered (${hardwareHrSensor.name})")
            } else if (hardwareHrSensor == null) {
                Log.w(TAG, "HeartRate: No hardware TYPE_HEART_RATE sensor found on device")
            }
        } catch (e: Exception) {
            Log.e(TAG, "HeartRate: Hardware sensor registration failed: ${e.message}")
        }
    }

    fun stop() {
        Log.i(TAG, "HeartRate: Stopping Heart Rate monitoring")
        if (isHealthServicesRegistered) {
            try {
                measureClient.unregisterMeasureCallbackAsync(
                    DataType.HEART_RATE_BPM,
                    healthServicesCallback
                )
                isHealthServicesRegistered = false
                Log.i(TAG, "HeartRate: Health Services MeasureClient unregistered")
            } catch (e: Exception) {
                Log.w(TAG, "HeartRate: Health Services unregister notice: ${e.message}")
            }
        }

        if (isHardwareListenerRegistered) {
            try {
                sensorManager?.unregisterListener(hardwareSensorListener)
                isHardwareListenerRegistered = false
                Log.i(TAG, "HeartRate: Hardware sensor listener unregistered")
            } catch (e: Exception) {
                Log.w(TAG, "HeartRate: Hardware sensor unregister notice: ${e.message}")
            }
        }
    }
}