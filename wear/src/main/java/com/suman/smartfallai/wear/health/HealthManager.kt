package com.suman.smartfallai.wear.health


import android.content.Context

import androidx.health.services.client.HealthServices
import androidx.health.services.client.MeasureCallback
import androidx.health.services.client.MeasureClient

import androidx.health.services.client.data.Availability
import androidx.health.services.client.data.DataPointContainer
import androidx.health.services.client.data.DataType
import androidx.health.services.client.data.DeltaDataType

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow



class HealthManager(
    context: Context
) {


    private val measureClient: MeasureClient =
        HealthServices
            .getClient(context)
            .measureClient



    private val _heartRate =
        MutableStateFlow(-1)


    val heartRate =
        _heartRate.asStateFlow()





    private val callback =
        object : MeasureCallback {



            override fun onDataReceived(
                data: DataPointContainer
            ) {


                val values =
                    data.getData(
                        DataType.HEART_RATE_BPM
                    )



                if(values.isNotEmpty()){


                    val bpm =
                        values
                            .last()
                            .value
                            .toInt()



                    _heartRate.value = bpm


                }


            }




            override fun onAvailabilityChanged(
                dataType: DeltaDataType<*, *>,
                availability: Availability
            ) {


            }



        }







    suspend fun start(){


        measureClient.registerMeasureCallback(

            DataType.HEART_RATE_BPM,

            callback

        )


    }

    fun stop() {
        measureClient.unregisterMeasureCallbackAsync(
            DataType.HEART_RATE_BPM,
            callback
        )
    }



}