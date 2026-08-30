package com.suman.smartfallai.wear.ui


import android.Manifest
import android.os.Bundle

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels

import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember

import androidx.lifecycle.compose.collectAsStateWithLifecycle

import com.suman.smartfallai.wear.ui.theme.MobileSensorLoggerTheme



class MainActivity : ComponentActivity() {



    private val viewModel: SmartFallViewModel by viewModels()





    private val requestPermissionsLauncher =

        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) {

        }

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {


        super.onCreate(savedInstanceState)





        requestPermissionsLauncher.launch(

            arrayOf(

                Manifest.permission.ACCESS_FINE_LOCATION,

                Manifest.permission.ACCESS_COARSE_LOCATION,

                Manifest.permission.BODY_SENSORS

            )

        )







        setContent {



            val state =

                viewModel.recordingState
                    .collectAsStateWithLifecycle()
                    .value





            val gpsData =

                viewModel.gpsData
                    .collectAsStateWithLifecycle()
                    .value





            val heartRate =

                viewModel.heartRate
                    .collectAsStateWithLifecycle()
                    .value






            androidx.compose.runtime.DisposableEffect(state.isRecording) {
                val view = window.decorView
                if (state.isRecording) {
                    window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    view.keepScreenOn = true
                } else {
                    window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    view.keepScreenOn = false
                }
                onDispose {
                    window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    view.keepScreenOn = false
                }
            }

            MobileSensorLoggerTheme {



                SmartFallScreen(



                    isRecording = state.isRecording,



                    activity = state.activity,



                    sampleCount = state.sampleCount.toInt(),



                    gpsData = gpsData,



                    heartRate = heartRate,



                    spo2 = 0,
                    pressure = 0f,

                    onStart = { activity ->
                        viewModel.startRecording(activity)
                    },

                    onStop = {
                        viewModel.stopRecording()
                    }
                )



            }




        }


    }



}