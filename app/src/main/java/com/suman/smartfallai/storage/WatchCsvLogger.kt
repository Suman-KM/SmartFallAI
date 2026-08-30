package com.suman.smartfallai.storage

import android.content.Context
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class WatchCsvLogger(
    private val context: Context
) {

    private var writer: BufferedWriter? = null

    private val dateFormat =
        SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())

    fun startLogging(activity: String, sessionId: String): String {

        val folder = File(
            context.getExternalFilesDir(null),
            "Datasets"
        )

        if (!folder.exists()) {
            folder.mkdirs()
        }

        val fileName =
            "${sessionId}_WATCH.csv"

        val file = File(folder, fileName)

        writer = BufferedWriter(FileWriter(file))

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

    fun log(payload: String) {

        writer?.apply {

            write(payload)
            newLine()
            flush()
        }
    }

    fun stopLogging() {

        writer?.flush()
        writer?.close()
        writer = null
    }
}
