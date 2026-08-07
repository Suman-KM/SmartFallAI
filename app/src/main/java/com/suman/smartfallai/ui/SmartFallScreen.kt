package com.suman.smartfallai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.suman.smartfallai.controller.RecordingState

@Composable
fun SmartFallScreen(

    state: RecordingState,

    onStart: (String) -> Unit,

    onStop: () -> Unit

) {

    val activities = listOf(

        "Standing",
        "Walking",
        "Running",
        "Sitting",
        "Lying Down",
        "Standing Up",
        "Sitting Down",
        "Upstairs",
        "Downstairs",
        "Forward Fall",
        "Backward Fall",
        "Left Side Fall",
        "Right Side Fall",
        "Fall While Sitting",
        "Fall While Walking",
        "Slip and Fall",
        "Trip and Fall"

    )

    var expanded by remember {

        mutableStateOf(false)

    }

    var selectedActivity by remember {

        mutableStateOf("Walking")

    }

    Column(

        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp)
            .verticalScroll(rememberScrollState()),

        horizontalAlignment = Alignment.CenterHorizontally,

        verticalArrangement = Arrangement.spacedBy(16.dp)

    ) {

        Text(

            text = "SmartFall AI",

            style = MaterialTheme.typography.headlineMedium

        )

        OutlinedButton(

            onClick = {

                expanded = true

            }

        ) {

            Text(selectedActivity)

        }

        DropdownMenu(

            expanded = expanded,

            onDismissRequest = {

                expanded = false

            }

        ) {

            activities.forEach {

                DropdownMenuItem(

                    text = {

                        Text(it)

                    },

                    onClick = {

                        selectedActivity = it

                        expanded = false

                    }

                )

            }

        }

        HorizontalDivider()

        Text("Status : ${state.status}")

        Text("Samples : ${state.sampleCount}")

        Text("Current File")

        Text(state.currentFile)

        HorizontalDivider()

        if (!state.isRecording) {

            Button(

                onClick = {

                    onStart(

                        selectedActivity

                    )

                }

            ) {

                Text("START RECORDING")

            }

        } else {

            Button(

                onClick = onStop

            ) {

                Text("STOP RECORDING")

            }

        }

    }

}