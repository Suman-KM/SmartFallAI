package com.suman.smartfallai.wear.storage

import android.content.Context
import com.suman.smartfallai.wear.model.SensorData
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

    private var currentFile: File? = null

    private val dateFormat =
        SimpleDateFormat(
            "yyyyMMdd_HHmmss",
            Locale.getDefault()
        )

    private var currentSessionId = ""

    fun startLogging(activity: String, sessionId: String): String {

        currentSessionId = sessionId

        val folder = File(
            context.getExternalFilesDir(null),
            "Datasets"
        )

        if (!folder.exists()) {
            folder.mkdirs()
        }

        val fileName =
            "${sessionId}_WATCH.csv"

        val file = File(
            folder,
            fileName
        )

        currentFile = file

        writer = BufferedWriter(
            FileWriter(file)
        )

        writer?.write(
            "session_id," +
                    "timestamp," +
                    "accX,accY,accZ," +
                    "gyroX,gyroY,gyroZ," +
                    "pitch,roll,yaw," +
                    "latitude,longitude,altitude," +
                    "speed,accuracy," +
                    "heartRate,spo2," +
                    "pressure," +
                    "activity"
        )

        writer?.newLine()

        writer?.flush()

        return fileName
    }

    fun log(data: SensorData) {

        writer?.apply {

            write(
                "${currentSessionId}," +
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

                        "${data.heartRate}," +
                        "${data.spo2}," +

                        "${data.pressure}," +

                        data.activity
            )

            newLine()

            /*
             * Flush periodically so that if the watch
             * loses power, we don't lose the entire file.
             */
            flush()
        }
    }

    fun stopLogging() {

        writer?.flush()

        writer?.close()

        writer = null

        currentFile = null
    }

    fun getCurrentFile(): File? {

        return currentFile
    }
}