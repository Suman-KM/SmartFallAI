package com.suman.smartfallai.wear.ml

import android.content.Context
import java.io.BufferedInputStream
import java.io.DataInputStream

class WatchRandomForestEngine(context: Context) {

    private class FastTree(
        val nodeCount: Int,
        val left: ShortArray,
        val right: ShortArray,
        val threshold: FloatArray,
        val feature: ShortArray,
        val isLeaf: ByteArray,
        val leafValues: FloatArray // size: nodeCount * 14
    )

    private val trees = ArrayList<FastTree>()
    private var numClasses = 14

    init {
        try {
            val inputStream = context.assets.open("trees.bin")
            DataInputStream(BufferedInputStream(inputStream, 65536)).use { dis ->
                val magic = ByteArray(4)
                dis.readFully(magic)
                val nTrees = dis.readInt()
                numClasses = dis.readInt()
                val nFeats = dis.readInt()

                for (t in 0 until nTrees) {
                    val nodeCount = dis.readInt()
                    val left = ShortArray(nodeCount)
                    val right = ShortArray(nodeCount)
                    val threshold = FloatArray(nodeCount)
                    val feature = ShortArray(nodeCount)
                    val isLeaf = ByteArray(nodeCount)
                    val leafValues = FloatArray(nodeCount * numClasses)

                    for (n in 0 until nodeCount) {
                        left[n] = dis.readShort()
                        right[n] = dis.readShort()
                        threshold[n] = dis.readFloat()
                        feature[n] = dis.readShort()
                        val leafFlag = dis.readByte()
                        isLeaf[n] = leafFlag

                        if (leafFlag.toInt() == 1) {
                            val baseOffset = n * numClasses
                            for (c in 0 until numClasses) {
                                leafValues[baseOffset + c] = dis.readFloat()
                            }
                        }
                    }

                    trees.add(FastTree(nodeCount, left, right, threshold, feature, isLeaf, leafValues))
                }
            }
            android.util.Log.d("WatchFallML", "Successfully loaded ${trees.size} fast binary trees ($numClasses classes).")
        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("WatchFallML", "Failed to load trees.bin: ${e.message}")
        }
    }

    /**
     * Evaluates 72 extracted features across all 100 trees in flat primitive arrays.
     * Returns a 14-class probability distribution.
     */
    fun predictProba(features: FloatArray): FloatArray {
        val totalProbs = FloatArray(numClasses)
        val numTrees = trees.size
        if (numTrees == 0) return totalProbs

        for (tIdx in 0 until numTrees) {
            val tree = trees[tIdx]
            var node = 0

            while (tree.isLeaf[node].toInt() == 0) {
                val fIdx = tree.feature[node].toInt()
                val fVal = features[fIdx]
                node = if (fVal <= tree.threshold[node]) {
                    tree.left[node].toInt()
                } else {
                    tree.right[node].toInt()
                }
            }

            val baseOffset = node * numClasses
            for (c in 0 until numClasses) {
                totalProbs[c] += tree.leafValues[baseOffset + c]
            }
        }

        for (c in 0 until numClasses) {
            totalProbs[c] /= numTrees
        }
        return totalProbs
    }
}
