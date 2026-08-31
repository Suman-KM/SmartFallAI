package com.suman.smartfallai.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp

class PhoneOnnxEngine(context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    init {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open("model.onnx").use { it.readBytes() }
            ortSession = ortEnv?.createSession(modelBytes)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Executes 1D-CNN ONNX model on a (100, 9) normalized sensor window.
     * Returns a 14-class probability distribution.
     */
    fun predictProba(window: Array<FloatArray>): FloatArray {
        val env = ortEnv ?: return FloatArray(14)
        val session = ortSession ?: return FloatArray(14)

        val flatBuffer = FloatArray(100 * 9)
        var idx = 0
        for (i in 0 until 100) {
            for (j in 0 until 9) {
                flatBuffer[idx++] = window[i][j]
            }
        }

        val shape = longArrayOf(1, 100, 9)
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatBuffer), shape)

        try {
            val inputName = session.inputNames.iterator().next()
            val outputs = session.run(Collections.singletonMap(inputName, tensor))
            val rawOutput = outputs[0].value

            val logits: FloatArray = if (rawOutput is Array<*> && rawOutput.isNotEmpty() && rawOutput[0] is FloatArray) {
                @Suppress("UNCHECKED_CAST")
                (rawOutput as Array<FloatArray>)[0]
            } else {
                FloatArray(14)
            }

            outputs.close()
            tensor.close()

            // Apply Softmax to obtain probabilities
            var maxLogit = Float.NEGATIVE_INFINITY
            for (l in logits) {
                if (l > maxLogit) maxLogit = l
            }

            var sumExp = 0.0f
            val expVals = FloatArray(logits.size)
            for (i in logits.indices) {
                val e = exp(logits[i] - maxLogit)
                expVals[i] = e
                sumExp += e
            }

            if (sumExp > 0.0f) {
                for (i in expVals.indices) {
                    expVals[i] /= sumExp
                }
            }
            return expVals
        } catch (e: Exception) {
            e.printStackTrace()
            tensor.close()
            return FloatArray(14)
        }
    }

    fun close() {
        try {
            ortSession?.close()
            ortEnv?.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
