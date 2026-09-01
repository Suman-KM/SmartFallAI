package com.suman.smartfallai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.suman.smartfallai.ActivityLabel
import com.suman.smartfallai.controller.RecordingState
import com.suman.smartfallai.ml.FallState

@Composable
fun SmartFallScreen(
    state: RecordingState,
    fallState: FallState = FallState.MONITORING,
    countdownRemaining: Int = 0,
    onCancelFallAlert: () -> Unit = {},
    onDismissAlert: () -> Unit = {},
    onStart: (String) -> Unit,
    onStop: () -> Unit
) {
    val activities = ActivityLabel.entries.map { it.displayName }
    var expanded by remember { mutableStateOf(false) }
    var selectedActivityLabel by remember { mutableStateOf(ActivityLabel.WALKING) }

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

        // --- INTERACTIVE COUNTDOWN & EMERGENCY OVERLAYS ---
        if (fallState == FallState.FALL_SUSPECTED) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp).fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "⚠️ POSSIBLE FALL DETECTED",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Are you okay?",
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Emergency notification in:",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "$countdownRemaining",
                        fontSize = 48.sp,
                        fontWeight = FontWeight.Black,
                        color = MaterialTheme.colorScheme.error
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Button(
                        onClick = onCancelFallAlert,
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                        modifier = Modifier.fillMaxWidth(0.7f).height(48.dp)
                    ) {
                        Text("I'M OK", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    }
                }
            }
        } else if (fallState == FallState.SOS_TRIGGERED || fallState == FallState.FALL_CONFIRMED) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.error)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp).fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "🚨 EMERGENCY SOS TRIGGERED",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onError
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Emergency alert dispatched to:",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onError
                    )
                    Text(
                        text = "sumankmdvg@gmail.com",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onError
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Button(
                        onClick = onDismissAlert,
                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.surface),
                        modifier = Modifier.fillMaxWidth(0.7f).height(48.dp)
                    ) {
                        Text("DISMISS ALERT", color = MaterialTheme.colorScheme.onSurface, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }

        OutlinedButton(
            onClick = { expanded = true }
        ) {
            Text(selectedActivityLabel.displayName)
        }

        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            ActivityLabel.entries.forEach { label ->
                DropdownMenuItem(
                    text = { Text(label.displayName) },
                    onClick = {
                        selectedActivityLabel = label
                        expanded = false
                    }
                )
            }
        }

        HorizontalDivider()
        Text("Status : ${state.status}")
        Text("Fall Monitor : ${fallState.name}")
        Text("Samples : ${state.sampleCount}")
        Text("Current File")
        Text(state.currentFile)
        HorizontalDivider()

        if (!state.isRecording) {
            Button(
                onClick = { onStart(selectedActivityLabel.name) }
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