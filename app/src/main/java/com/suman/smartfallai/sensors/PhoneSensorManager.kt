package com.suman.smartfallai.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.suman.smartfallai.model.SensorData
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class PhoneSensorManager(
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

    private val _sensorData = MutableStateFlow(

        SensorData(
            timestamp = System.currentTimeMillis(),
            accX = 0f,
            accY = 0f,
            accZ = 0f,
            gyroX = 0f,
            gyroY = 0f,
            gyroZ = 0f
        )

    )

    val sensorData: StateFlow<SensorData> =
        _sensorData.asStateFlow()

    fun start() {

        accelerometer?.also {

            sensorManager.registerListener(
                this,
                it,
                20000
            )

        }

        gyroscope?.also {

            sensorManager.registerListener(
                this,
                it,
                20000
            )

        }

        rotationVector?.also {

            sensorManager.registerListener(
                this,
                it,
                20000
            )

        }

    }

    fun stop() {

        sensorManager.unregisterListener(this)

    }

    override fun onSensorChanged(event: SensorEvent) {

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

        }

        _sensorData.value = SensorData(

            timestamp = System.currentTimeMillis(),

            accX = accX,
            accY = accY,
            accZ = accZ,

            gyroX = gyroX,
            gyroY = gyroY,
            gyroZ = gyroZ,

            pitch = pitch,
            roll = roll,
            yaw = yaw

        )

    }

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int
    ) {

    }

}