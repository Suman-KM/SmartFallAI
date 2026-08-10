package com.suman.smartfallai.wear.gps

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class GpsManager(
    private val context: Context
) {

    private val fusedLocationClient:
            FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private val _gpsData =
        MutableStateFlow<GpsData?>(null)

    val gpsData: StateFlow<GpsData?> =
        _gpsData.asStateFlow()

    private val locationRequest =
        LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            2000L
        )
            .setMinUpdateIntervalMillis(1000L)
            .setMinUpdateDistanceMeters(0f)
            .setWaitForAccurateLocation(true)
            .build()

    private val locationCallback =
        object : LocationCallback() {

            override fun onLocationResult(
                result: LocationResult
            ) {

                for (location in result.locations) {

                    updateLocation(location)
                }
            }
        }

    @SuppressLint("MissingPermission")
    fun start() {

        if (!hasLocationPermission()) {
            return
        }

        // First try to obtain the latest cached location.
        fusedLocationClient
            .lastLocation
            .addOnSuccessListener { location ->

                if (location != null) {

                    updateLocation(location)
                }
            }

        // Then start continuous high-accuracy updates.
        fusedLocationClient
            .requestLocationUpdates(
                locationRequest,
                locationCallback,
                context.mainLooper
            )
    }

    fun stop() {

        fusedLocationClient
            .removeLocationUpdates(locationCallback)
    }

    private fun hasLocationPermission(): Boolean {

        val fine =
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        val coarse =
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED

        return fine || coarse
    }

    private fun updateLocation(location: Location) {

        _gpsData.value =
            GpsData(

                latitude =
                    location.latitude,

                longitude =
                    location.longitude,

                altitude =
                    location.altitude,

                speed =
                    if (location.hasSpeed()) {
                        location.speed
                    } else {
                        0f
                    },

                accuracy =
                    if (location.hasAccuracy()) {
                        location.accuracy
                    } else {
                        0f
                    },

                timestamp =
                    location.time
            )
    }
}