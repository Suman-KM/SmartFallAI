package com.suman.smartfallai.wear.ml

import android.content.Context
import org.json.JSONObject
import java.io.InputStream

class WatchRobustScaler(context: Context) {

    private val medians = FloatArray(9)
    private val iqrs = FloatArray(9)

    init {
        try {
            val jsonString = context.assets.open("scaler.json").bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)
            val mediansArray = jsonObject.getJSONArray("median")
            val iqrArray = jsonObject.getJSONArray("iqr")

            for (i in 0 until 9) {
                medians[i] = mediansArray.getDouble(i).toFloat()
                var iqrVal = iqrArray.getDouble(i).toFloat()
                if (iqrVal == 0.0f) iqrVal = 1.0f
                iqrs[i] = iqrVal
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Applies Robust Scaling: (x - median) / IQR
     */
    fun transformInPlace(window: Array<FloatArray>) {
        for (sample in window) {
            for (i in 0 until 9) {
                sample[i] = (sample[i] - medians[i]) / iqrs[i]
            }
        }
    }
}
