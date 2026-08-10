package com.suman.smartfallai.wear.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.wear.compose.material3.Button
import androidx.wear.compose.material3.Text

@Composable
fun SmartFallScreen(
    isRecording: Boolean,
    activity: String,
    sampleCount: Long,
    status: String,
    onStart: () -> Unit,
    onStop: () -> Unit
) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {

        Text(
            text = "SmartFall AI"
        )

        Text(
            text = status
        )

        Text(
            text = "Activity: $activity"
        )

        Text(
            text = "Samples: $sampleCount"
        )

        if (isRecording) {

            Button(
                onClick = onStop
            ) {

                Text(
                    text = "STOP"
                )
            }

        } else {

            Button(
                onClick = onStart
            ) {

                Text(
                    text = "START"
                )
            }
        }
    }
}