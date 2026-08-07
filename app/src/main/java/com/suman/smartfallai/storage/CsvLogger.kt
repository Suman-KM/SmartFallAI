package com.suman.smartfallai.storage

import android.content.Context
import com.suman.smartfallai.model.SensorData
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class CsvLogger(
    private val context: Context
) {

    private var writer: BufferedWriter? = null

    private val dateFormat =
        SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())

    fun startLogging(activity: String): String {

        val folder = File(
            context.getExternalFilesDir(null),
            "Datasets"
        )

        if (!folder.exists()) {
            folder.mkdirs()
        }

        val fileName =
            "${activity}_${dateFormat.format(Date())}.csv"

        val file = File(folder, fileName)

        writer = BufferedWriter(FileWriter(file))

        writer?.write(
            "timestamp," +
                    "accX,accY,accZ," +
                    "gyroX,gyroY,gyroZ," +
                    "pitch,roll,yaw," +
                    "latitude,longitude,altitude," +
                    "speed,accuracy," +
                    "activity"
        )

        writer?.newLine()

        return fileName
    }

    fun log(data: SensorData) {

        writer?.apply {

            write(
                "${data.timestamp}," +
                        "${data.accX}," +
                        "${data.accY}," +
                        "${data.accZ}," +
                        "${data.gyroX}," +
                        "${data.gyroY}," +
                        "${data.gyroZ}," +
                        "${data.pitch}," +
                        "${data.roll}," +
                        "${data.yaw}," +
                        "${data.latitude}," +
                        "${data.longitude}," +
                        "${data.altitude}," +
                        "${data.speed}," +
                        "${data.accuracy}," +
                        data.activity
            )

            newLine()

        }

    }

    fun stopLogging() {

        writer?.flush()
        writer?.close()
        writer = null

    }

}