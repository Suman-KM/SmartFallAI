package com.suman.smartfallai.gps

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import androidx.core.content.ContextCompat
import com.google.android.gms.location.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class GpsManager(
    private val context: Context
) {

    private val fusedLocationClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val _gpsData = MutableStateFlow(GpsData())

    val gpsData: StateFlow<GpsData> =
        _gpsData.asStateFlow()

    val currentLocation: GpsData
        get() = _gpsData.value

    private val locationRequest =
        LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            1000L
        )
            .setMinUpdateIntervalMillis(500L)
            .build()

    private val locationCallback = object : LocationCallback() {

        override fun onLocationResult(result: LocationResult) {

            result.lastLocation?.let {
                updateLocation(it)
            }

        }
    }

    @SuppressLint("MissingPermission")
    fun start() {

        if (
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }

        // 1. Get last known location immediately
        fusedLocationClient.lastLocation
            .addOnSuccessListener { location ->

                if (location != null) {
                    updateLocation(location)
                }

            }

        // 2. Continue listening for live updates
        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            context.mainLooper
        )

    }

    fun stop() {

        fusedLocationClient.removeLocationUpdates(locationCallback)

    }

    private fun updateLocation(location: Location) {

        _gpsData.value = GpsData(

            latitude = location.latitude,

            longitude = location.longitude,

            altitude = location.altitude,

            speed = location.speed,

            accuracy = location.accuracy,

            timestamp = System.currentTimeMillis()

        )

    }

}