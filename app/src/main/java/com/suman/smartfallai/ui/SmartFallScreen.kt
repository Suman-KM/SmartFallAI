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
    emailDeliveryStatus: com.suman.smartfallai.emergency.EmailDeliveryStatus = com.suman.smartfallai.emergency.EmailDeliveryStatus.IDLE,
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
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = when (emailDeliveryStatus) {
                            com.suman.smartfallai.emergency.EmailDeliveryStatus.IDLE -> "Emergency Email: Ready"
                            com.suman.smartfallai.emergency.EmailDeliveryStatus.SENDING -> "Emergency Email: Sending..."
                            com.suman.smartfallai.emergency.EmailDeliveryStatus.SENT -> "Emergency Email: Sent"
                            com.suman.smartfallai.emergency.EmailDeliveryStatus.FAILED -> "Emergency Email: Failed"
                        },
                        style = MaterialTheme.typography.bodyMedium,
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
        Text(
            text = "System Status : ${if (state.isRecording) "Active Fall Monitoring" else "Standby"}",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )
        Text("Fall State : ${fallState.name}")
        Text("Pipeline : Real-Time In-Memory (Zero CSV Overhead)")
        HorizontalDivider()

        if (!state.isRecording) {
            Button(
                onClick = { onStart(selectedActivityLabel.name) },
                modifier = Modifier.fillMaxWidth(0.8f).height(48.dp)
            ) {
                Text("START MONITORING", fontWeight = FontWeight.Bold)
            }
        } else {
            Button(
                onClick = onStop,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary),
                modifier = Modifier.fillMaxWidth(0.8f).height(48.dp)
            ) {
                Text("STOP MONITORING", fontWeight = FontWeight.Bold)
            }
        }
    }
}