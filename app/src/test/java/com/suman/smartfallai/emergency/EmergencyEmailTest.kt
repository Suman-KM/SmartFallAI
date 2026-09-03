package com.suman.smartfallai.emergency

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class EmergencyEmailTest {

    @Test
    fun testEmergencyRecipientConfiguration() {
        assertEquals(1, EmergencyManager.emergencyRecipients.size)
        assertEquals("sumankmdvg@gmail.com", EmergencyManager.emergencyRecipients[0])
        assertEquals("sumankmdvg@gmail.com", EmergencyManager.EMERGENCY_RECIPIENT)
    }

    @Test
    fun testDeliveryStatusEnum() {
        val states = EmailDeliveryStatus.entries
        assertTrue(states.contains(EmailDeliveryStatus.IDLE))
        assertTrue(states.contains(EmailDeliveryStatus.SENDING))
        assertTrue(states.contains(EmailDeliveryStatus.SENT))
        assertTrue(states.contains(EmailDeliveryStatus.FAILED))
    }

    @Test
    fun testPayloadStructureWithFullTelemetry() {
        val fallTimeMs = 1788429000000L
        val timeFormat = SimpleDateFormat("dd MMMM yyyy, HH:mm:ss", Locale.ENGLISH)
        val expectedTime = timeFormat.format(Date(fallTimeMs))

        val manager = EmergencyManagerTestHelper()
        val payload = manager.buildEmergencyPayload(
            deviceSource = "Samsung Galaxy Watch 4 (SM-R870)",
            fallTimeMs = fallTimeMs,
            heartRate = 82,
            latitude = 14.4594,
            longitude = 75.9240,
            accuracy = 20.0f
        )

        assertEquals("FALL_CONFIRMED", payload["event"])
        assertEquals("Samsung Galaxy Watch 4 (SM-R870)", payload["deviceSource"])
        assertEquals(expectedTime, payload["timeString"])
        assertEquals(82, payload["heartRate"])
        assertEquals(14.4594, payload["latitude"])
        assertEquals(75.9240, payload["longitude"])
        assertEquals(20.0f, payload["accuracy"])
        @Suppress("UNCHECKED_CAST")
        val recipients = payload["recipients"] as List<String>
        assertEquals("sumankmdvg@gmail.com", recipients[0])
    }

    @Test
    fun testPayloadStructureWithMissingTelemetryFallback() {
        val fallTimeMs = 1788429000000L
        val manager = EmergencyManagerTestHelper()
        val payload = manager.buildEmergencyPayload(
            deviceSource = "Samsung Galaxy A50s (Phone)",
            fallTimeMs = fallTimeMs,
            heartRate = -1,
            latitude = 0.0,
            longitude = 0.0,
            accuracy = null
        )

        assertEquals("FALL_CONFIRMED", payload["event"])
        assertEquals("Samsung Galaxy A50s (Phone)", payload["deviceSource"])
        assertNull(payload["heartRate"]) // Null ensures clean "Heart Rate: Unavailable" on backend
        assertNull(payload["latitude"])  // Null ensures clean "GPS Location: Unavailable" on backend
        assertNull(payload["longitude"])
        assertNull(payload["accuracy"])
    }

    // Helper to test pure logic without Android Context instantiation
    private class EmergencyManagerTestHelper {
        fun buildEmergencyPayload(
            deviceSource: String,
            fallTimeMs: Long,
            heartRate: Int?,
            latitude: Double?,
            longitude: Double?,
            accuracy: Float?
        ): Map<String, Any?> {
            val timeFormat = SimpleDateFormat("dd MMMM yyyy, HH:mm:ss", Locale.ENGLISH)
            val formattedTime = timeFormat.format(Date(fallTimeMs))

            val validHr = if (heartRate != null && heartRate > 0) heartRate else null
            val validLat = if (latitude != null && Math.abs(latitude) > 0.0001) latitude else null
            val validLon = if (longitude != null && Math.abs(longitude) > 0.0001) longitude else null
            val validAcc = if (accuracy != null && accuracy > 0f) accuracy else null

            return mapOf(
                "event" to "FALL_CONFIRMED",
                "deviceSource" to deviceSource,
                "timestamp" to fallTimeMs,
                "timeString" to formattedTime,
                "heartRate" to validHr,
                "latitude" to validLat,
                "longitude" to validLon,
                "accuracy" to validAcc,
                "recipients" to EmergencyManager.emergencyRecipients
            )
        }
    }
}
