package com.suman.smartfallai.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle

class MainActivity : ComponentActivity() {

    private val viewModel: SmartFallViewModel by viewModels()

    private val locationPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            if (granted) {
                viewModel.startGps()
            }

        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        checkLocationPermission()

        setContent {

            val recordingState by viewModel
                .recordingState
                .collectAsStateWithLifecycle()

            SmartFallScreen(
                state = recordingState,
                onStart = { activity ->
                    viewModel.startRecording(activity)
                },
                onStop = {
                    viewModel.stopRecording()
                }
            )
        }
    }

    private fun checkLocationPermission() {

        when {

            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED -> {

                viewModel.startGps()

            }

            else -> {

                locationPermissionLauncher.launch(
                    Manifest.permission.ACCESS_FINE_LOCATION
                )

            }

        }

    }

    override fun onDestroy() {
        super.onDestroy()
        viewModel.stopGps()
    }
}