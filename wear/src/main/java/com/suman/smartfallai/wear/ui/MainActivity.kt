package com.suman.smartfallai.wear.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.suman.smartfallai.wear.ui.theme.MobileSensorLoggerTheme

class MainActivity : ComponentActivity() {

    private val viewModel: SmartFallViewModel by viewModels()

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(savedInstanceState)

        setContent {

            val state =
                viewModel.recordingState
                    .collectAsStateWithLifecycle()
                    .value

            MobileSensorLoggerTheme {

                SmartFallScreen(

                    isRecording =
                        state.isRecording,

                    activity =
                        state.activity,

                    sampleCount =
                        state.sampleCount,

                    status =
                        state.status,

                    onStart = {

                        viewModel.startRecording(
                            "Walking"
                        )
                    },

                    onStop = {

                        viewModel.stopRecording()
                    }
                )
            }
        }
    }
}