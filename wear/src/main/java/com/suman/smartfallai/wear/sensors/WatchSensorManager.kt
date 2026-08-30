package com.suman.smartfallai.wear.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import com.suman.smartfallai.wear.model.SensorData
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ReceiveChannel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class WatchSensorManager(
    context: Context
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accelerometer =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private val gyroscope =
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    private val rotationVector =
        sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    private var accX = 0f
    private var accY = 0f
    private var accZ = 0f

    private var gyroX = 0f
    private var gyroY = 0f
    private var gyroZ = 0f

    private var pitch = 0f
    private var roll = 0f
    private var yaw = 0f

    private val _sensorData =
        MutableStateFlow(SensorData())

    val sensorData: StateFlow<SensorData> =
        _sensorData.asStateFlow()

    private var _sensorChannel = Channel<SensorData>(
        capacity = 500,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val sensorChannel: ReceiveChannel<SensorData>
        get() = _sensorChannel

    var sensorEventsReceived: Long = 0L
        private set
    var sensorEventsQueued: Long = 0L
        private set
    var sensorEventsDropped: Long = 0L
        private set

    private var bootOffsetMillis: Long = 0L
    private var currentSequence: Long = 0L

    fun start() {
        bootOffsetMillis = System.currentTimeMillis() - (SystemClock.elapsedRealtimeNanos() / 1_000_000L)
        currentSequence = 0L
        sensorEventsReceived = 0L
        sensorEventsQueued = 0L
        sensorEventsDropped = 0L

        _sensorChannel.close()
        _sensorChannel = Channel<SensorData>(
            capacity = 500,
            onBufferOverflow = BufferOverflow.DROP_OLDEST
        )

        accelerometer?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_GAME
            )
        }

        gyroscope?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_GAME
            )
        }

        rotationVector?.let {
            sensorManager.registerListener(
                this,
                it,
                SensorManager.SENSOR_DELAY_GAME
            )
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        _sensorChannel.close()
    }

    override fun onSensorChanged(event: SensorEvent) {
        sensorEventsReceived++

        when (event.sensor.type) {

            Sensor.TYPE_ACCELEROMETER -> {
                accX = event.values[0]
                accY = event.values[1]
                accZ = event.values[2]
            }

            Sensor.TYPE_GYROSCOPE -> {
                gyroX = event.values[0]
                gyroY = event.values[1]
                gyroZ = event.values[2]
            }

            Sensor.TYPE_ROTATION_VECTOR -> {
                updateOrientation(event)
            }
        }

        publishSensorData(event.timestamp)
    }

    private fun updateOrientation(event: SensorEvent) {

        val rotationMatrix = FloatArray(9)

        SensorManager.getRotationMatrixFromVector(
            rotationMatrix,
            event.values
        )

        val orientation = FloatArray(3)

        SensorManager.getOrientation(
            rotationMatrix,
            orientation
        )

        yaw = Math.toDegrees(
            orientation[0].toDouble()
        ).toFloat()

        pitch = Math.toDegrees(
            orientation[1].toDouble()
        ).toFloat()

        roll = Math.toDegrees(
            orientation[2].toDouble()
        ).toFloat()
    }

    private fun publishSensorData(eventTimestampNanos: Long) {
        val epochMillis = bootOffsetMillis + (eventTimestampNanos / 1_000_000L)
        val data = SensorData(
            sequence = currentSequence++,
            timestamp = epochMillis,
            accX = accX,
            accY = accY,
            accZ = accZ,
            gyroX = gyroX,
            gyroY = gyroY,
            gyroZ = gyroZ,
            pitch = pitch,
            roll = roll,
            yaw = yaw,
            isValid = true
        )
        _sensorData.value = data
        val result = _sensorChannel.trySend(data)
        if (result.isSuccess) {
            sensorEventsQueued++
        } else {
            sensorEventsDropped++
        }
    }

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int
    ) {
    }
}