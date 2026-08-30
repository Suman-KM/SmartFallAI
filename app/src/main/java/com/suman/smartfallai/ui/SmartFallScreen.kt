package com.suman.smartfallai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.suman.smartfallai.ActivityLabel
import com.suman.smartfallai.controller.RecordingState

@Composable
fun SmartFallScreen(

    state: RecordingState,

    onStart: (String) -> Unit,

    onStop: () -> Unit

) {

    val activities = ActivityLabel.entries.map { it.displayName }

    var expanded by remember {

        mutableStateOf(false)

    }

    var selectedActivityLabel by remember {
        mutableStateOf(ActivityLabel.WALKING)
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
            Text(selectedActivityLabel.displayName)
        }

        DropdownMenu(
            expanded = expanded,
            onDismissRequest = {
                expanded = false
            }
        ) {
            ActivityLabel.entries.forEach { label ->
                DropdownMenuItem(
                    text = {
                        Text(label.displayName)
                    },
                    onClick = {
                        selectedActivityLabel = label
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
                    onStart(selectedActivityLabel.name)
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