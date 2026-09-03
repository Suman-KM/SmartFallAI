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

    private val testEmergencyReceiver = object : android.content.BroadcastReceiver() {
        override fun onReceive(context: android.content.Context?, intent: android.content.Intent?) {
            android.util.Log.i("MainActivity", "Received TEST_EMERGENCY_ALERT broadcast")
            com.suman.smartfallai.emergency.EmergencyManager(this@MainActivity).sendEmergencyAlert(
                deviceSource = "Samsung Galaxy A50s (Controlled Test)",
                heartRate = 78
            )
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val filter = android.content.IntentFilter("com.suman.smartfallai.TEST_EMERGENCY_ALERT")
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(testEmergencyReceiver, filter, android.content.Context.RECEIVER_EXPORTED)
        } else {
            registerReceiver(testEmergencyReceiver, filter)
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                android.view.WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                android.view.WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
            )
        }
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        checkLocationPermission()

        setContent {

            val recordingState by viewModel
                .recordingState
                .collectAsStateWithLifecycle()

            val fallState by viewModel
                .fallState
                .collectAsStateWithLifecycle()

            val countdownRemaining by viewModel
                .countdownRemaining
                .collectAsStateWithLifecycle()

            val emailDeliveryStatus by viewModel
                .emailDeliveryStatus
                .collectAsStateWithLifecycle()

            androidx.compose.runtime.DisposableEffect(recordingState.isRecording) {
                val view = window.decorView
                if (recordingState.isRecording) {
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

            SmartFallScreen(
                state = recordingState,
                fallState = fallState,
                countdownRemaining = countdownRemaining,
                emailDeliveryStatus = emailDeliveryStatus,
                onCancelFallAlert = { viewModel.cancelFallAlert() },
                onDismissAlert = { viewModel.dismissAlert() },
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
        try {
            unregisterReceiver(testEmergencyReceiver)
        } catch (e: Exception) {}
        viewModel.stopGps()
    }
}