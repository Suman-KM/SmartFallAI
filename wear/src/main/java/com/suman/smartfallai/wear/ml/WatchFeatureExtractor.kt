package com.suman.smartfallai.wear.ml

import kotlin.math.sqrt

object WatchFeatureExtractor {

    /**
     * Extracts 8 statistical features per channel across 9 IMU channels:
     * [mean, std, min, max, range, median, rms, energy]
     * Total = 9 * 8 = 72 features.
     */
    fun extractFeatures(window: Array<FloatArray>): FloatArray {
        val numSamples = window.size // 100
        val numChannels = 9
        val features = FloatArray(numChannels * 8)

        val means = FloatArray(numChannels)
        val stds = FloatArray(numChannels)
        val mins = FloatArray(numChannels) { Float.MAX_VALUE }
        val maxs = FloatArray(numChannels) { -Float.MAX_VALUE }
        val ranges = FloatArray(numChannels)
        val medians = FloatArray(numChannels)
        val rmss = FloatArray(numChannels)
        val energies = FloatArray(numChannels)

        for (c in 0 until numChannels) {
            val channelValues = FloatArray(numSamples)
            var sum = 0.0
            var sumSq = 0.0

            for (s in 0 until numSamples) {
                val v = window[s][c]
                channelValues[s] = v
                sum += v
                sumSq += (v * v)
                if (v < mins[c]) mins[c] = v
                if (v > maxs[c]) maxs[c] = v
            }

            val mean = (sum / numSamples).toFloat()
            means[c] = mean
            ranges[c] = maxs[c] - mins[c]

            var varSum = 0.0
            for (s in 0 until numSamples) {
                val diff = channelValues[s] - mean
                varSum += (diff * diff)
            }
            stds[c] = sqrt(varSum / numSamples).toFloat()

            // Median
            channelValues.sort()
            medians[c] = if (numSamples % 2 == 0) {
                (channelValues[numSamples / 2 - 1] + channelValues[numSamples / 2]) / 2.0f
            } else {
                channelValues[numSamples / 2]
            }

            val meanSq = (sumSq / numSamples).toFloat()
            energies[c] = meanSq
            rmss[c] = sqrt(meanSq)
        }

        // Layout matching Python np.hstack: [means, stds, mins, maxs, ranges, medians, rms, energy]
        var offset = 0
        System.arraycopy(means, 0, features, offset, numChannels); offset += numChannels
        System.arraycopy(stds, 0, features, offset, numChannels); offset += numChannels
        System.arraycopy(mins, 0, features, offset, numChannels); offset += numChannels
        System.arraycopy(maxs, 0, features, offset, numChannels); offset += numChannels
        System.arraycopy(ranges, 0, features, offset, numChannels); offset += numChannels
        System.arraycopy(medians, 0, features, offset, numChannels); offset += numChannels
        System.arraycopy(rmss, 0, features, offset, numChannels); offset += numChannels
        System.arraycopy(energies, 0, features, offset, numChannels)

        return features
    }
}
