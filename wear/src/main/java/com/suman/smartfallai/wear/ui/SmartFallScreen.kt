package com.suman.smartfallai.wear.ui


import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp


import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.material3.*


import com.suman.smartfallai.wear.gps.GpsData



@Composable
fun SmartFallScreen(
    isRecording: Boolean,
    activity: String,
    sampleCount: Int,
    gpsData: GpsData?,
    heartRate: Int = 0,
    spo2: Int = 0,
    pressure: Float = 0f,
    onStart: (String) -> Unit,
    onStop: () -> Unit
){
    val activities = com.suman.smartfallai.wear.ActivityLabel.entries
    val selectedIndex = remember { mutableStateOf(0) }
    val currentLabel = activities[selectedIndex.value]

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ){
        ScalingLazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ){
            // TITLE
            item {
                Text(
                    text = "SmartFall AI",
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            item {
                Spacer(
                    modifier = Modifier.height(3.dp)
                )
                Text(
                    text =
                        if(isRecording)
                            "RECORDING"
                        else
                            "WAITING",
                    color =
                        if(isRecording)
                            Color.Green
                        else
                            Color.White,
                    fontSize = 11.sp
                )
            }

            item {
                Spacer(
                    modifier = Modifier.height(5.dp)
                )
                Text(
                    text = "Samples : $sampleCount",
                    color = Color.White,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            item {
                Spacer(
                    modifier = Modifier.height(8.dp)
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ){
                    if (!isRecording) {
                        Button(
                            onClick = {
                                selectedIndex.value = (selectedIndex.value + 1) % activities.size
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF202020)),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = "Activity: ${currentLabel.displayName}",
                                color = Color.White,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Medium
                            )
                        }
                    } else {
                        Text(
                            text = "Activity: ${com.suman.smartfallai.wear.ActivityLabel.entries.find { it.name == activity }?.displayName ?: activity}",
                            color = Color.White,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            item {
                Spacer(modifier = Modifier.height(6.dp))
                if (!isRecording) {
                    Button(
                        onClick = { onStart(currentLabel.name) },
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF00AA00)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(text = "START", color = Color.White, fontWeight = FontWeight.Bold)
                    }
                } else {
                    Button(
                        onClick = onStop,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFDD0000)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(text = "STOP", color = Color.White, fontWeight = FontWeight.Bold)
                    }
                }
            }






            // SENSOR PANEL


            // SENSOR PANEL

            item {

                Spacer(
                    modifier = Modifier.height(8.dp)
                )


                Card(

                    modifier = Modifier
                        .fillMaxWidth(),

                    shape = RoundedCornerShape(14.dp),


                    colors = CardDefaults.cardColors(

                        containerColor = Color(0xFF202020)

                    ),


                    onClick = {}

                ){


                    Column(

                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(
                                horizontal = 12.dp,
                                vertical = 8.dp
                            )


                    ){






                        Spacer(
                            modifier = Modifier.height(2.dp)
                        )



                        Text(

                            text =

                                "GPS  %.4f  %.4f  %.1fm".format(

                                    gpsData?.latitude ?: 0.0,

                                    gpsData?.longitude ?: 0.0,

                                    gpsData?.altitude ?: 0.0

                                ),


                            color = Color.White,

                            fontSize = 10.sp

                        )



                        Spacer(
                            modifier = Modifier.height(3.dp)
                        )



                        Text(

                            text =

                                "${if (heartRate <= 0) "N/A" else "$heartRate BPM"}    SpO2 ${if (spo2 <= 0) "N/A" else "$spo2"}    P ${if (pressure <= 0f) "N/A" else "$pressure"}",


                            color = Color.White,

                            fontSize = 10.sp

                        )



                        Spacer(
                            modifier = Modifier.height(2.dp)
                        )








                    }


                }


            }

            }




        }


    }



