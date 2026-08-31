package com.suman.smartfallai.wear.ml

import android.content.Context
import org.json.JSONObject

class WatchRandomForestEngine(context: Context) {

    private class DecisionTreeNode(
        val left: Int,
        val right: Int,
        val feature: Int,
        val threshold: Float,
        val values: FloatArray
    )

    private class DecisionTree(
        val nodes: Array<DecisionTreeNode>
    )

    private val trees = ArrayList<DecisionTree>()
    private var numClasses = 14

    init {
        try {
            val jsonString = context.assets.open("trees.json").bufferedReader().use { it.readText() }
            val root = JSONObject(jsonString)
            numClasses = root.optInt("n_classes", 14)
            val treesArray = root.getJSONArray("trees")

            for (t in 0 until treesArray.length()) {
                val tObj = treesArray.getJSONObject(t)
                val nodeCount = tObj.getInt("node_count")
                val cl = tObj.getJSONArray("children_left")
                val cr = tObj.getJSONArray("children_right")
                val feat = tObj.getJSONArray("feature")
                val th = tObj.getJSONArray("threshold")
                val vals = tObj.getJSONArray("values")

                val nodes = Array(nodeCount) { i ->
                    val vArr = vals.getJSONArray(i)
                    val vFloat = FloatArray(vArr.length()) { k -> vArr.getDouble(k).toFloat() }
                    DecisionTreeNode(
                        left = cl.getInt(i),
                        right = cr.getInt(i),
                        feature = feat.getInt(i),
                        threshold = th.getDouble(i).toFloat(),
                        values = vFloat
                    )
                }
                trees.add(DecisionTree(nodes))
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Evaluates 72 extracted features across all 100 trees.
     * Returns a 14-class probability distribution.
     */
    fun predictProba(features: FloatArray): FloatArray {
        val totalProbs = FloatArray(numClasses)
        val numTrees = trees.size
        if (numTrees == 0) return totalProbs

        for (tree in trees) {
            var nodeIdx = 0
            val nodes = tree.nodes

            while (nodes[nodeIdx].left != -1) {
                val node = nodes[nodeIdx]
                val fVal = features[node.feature]
                nodeIdx = if (fVal <= node.threshold) {
                    node.left
                } else {
                    node.right
                }
            }

            val leafValues = nodes[nodeIdx].values
            for (c in 0 until numClasses) {
                if (c < leafValues.size) {
                    totalProbs[c] += leafValues[c]
                }
            }
        }

        for (c in 0 until numClasses) {
            totalProbs[c] /= numTrees
        }
        return totalProbs
    }
}
