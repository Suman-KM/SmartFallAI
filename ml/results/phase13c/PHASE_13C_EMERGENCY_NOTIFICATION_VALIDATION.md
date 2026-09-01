# SmartFall AI — Phase 13C: Emergency Notification Pathway Validation

**Recipient:** `sumankmdvg@gmail.com`  
**Component Audited:** `app/src/main/java/com/suman/smartfallai/emergency/EmergencyManager.kt`  
**Platform:** Android 11 (API 30) on Samsung Galaxy A50s (`SM-A507FN`)

---

## 1. Executive Determination: Notification Mechanism Classification

> [!IMPORTANT]
> **Audit Finding:** The SmartFall AI emergency email system uses **Android `Intent.ACTION_SENDTO` with a `mailto:` URI scheme**.  
> It does **NOT** perform automatic background SMTP or REST API transmission without user interaction.

### Classification:
- **Category B: Prepares and launches an Email Intent requiring user interaction.**
- The application constructs a pre-populated email containing subject, recipient, body, device source, and timestamp, launches the default email application (e.g. Gmail), and posts a high-priority system notification.
- The user (or bystander) must physically tap the "Send" button in the email client to transmit the email across the network.

---

## 2. Code-Level Implementation Forensic

In `EmergencyManager.kt`:
```kotlin
fun sendEmergencyAlert(deviceSource: String, fallTimeMs: Long = System.currentTimeMillis()) {
    triggerVibration(isEmergency = true)

    val subject = "SMARTFALL AI — FALL DETECTED"
    val body = """
SmartFall AI detected a possible fall and the emergency countdown expired without user cancellation.

Device:
$deviceSource

Detection:
FALL_CONFIRMED

Timestamp:
${java.util.Date(fallTimeMs)}

This is an automated research prototype notification.
    """.trimIndent()

    // 1. Post High-Priority Notification with PendingIntent
    val emailIntent = Intent(Intent.ACTION_SENDTO).apply {
        data = Uri.parse("mailto:$EMERGENCY_RECIPIENT")
        putExtra(Intent.EXTRA_SUBJECT, subject)
        putExtra(Intent.EXTRA_TEXT, body)
        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }
    val pendingIntent = PendingIntent.getActivity(context, 0, emailIntent, ...)
    ...
    notificationManager.notify(NOTIFICATION_ID, notification)

    // 2. Direct Activity Launch for Immediate User Prompt
    try {
        val directEmailIntent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:$EMERGENCY_RECIPIENT")
            putExtra(Intent.EXTRA_SUBJECT, subject)
            putExtra(Intent.EXTRA_TEXT, body)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        context.startActivity(directEmailIntent)
    } catch (e: Exception) {
        Log.w(TAG, "Direct email intent launch fallback: ${e.message}")
    }
}
```

---

## 3. Detailed Pathway Specification

| Parameter | Specification |
| :--- | :--- |
| **Recipient Address** | `sumankmdvg@gmail.com` |
| **Email Subject** | `SMARTFALL AI — FALL DETECTED` |
| **Intent Action** | `android.intent.action.SENDTO` |
| **URI Scheme** | `mailto:sumankmdvg@gmail.com` |
| **User Interaction Required?** | **YES**. User or bystander must press "Send" in the launched mail app. |
| **Haptic Feedback** | Emergency SOS vibration pattern (`[0, 800, 200, 800, 200, 800] ms`). |
| **System Notification** | `PRIORITY_MAX`, `CATEGORY_ALARM`, Heads-up banner with `ic_dialog_alert`. |
| **Delivery Verification** | Pre-population and intent dispatch are verified. Actual network delivery requires email app completion. |

---

## 4. Architectural Analysis: Intent vs. Background SMTP

### 4.1 Advantages of Intent-Based Approach (Current)
1. **Privacy & Security**: Does not require hardcoding or storing Gmail API OAuth tokens, app passwords, or SMTP credentials inside the Android APK.
2. **Device Compatibility**: Operates across all Android versions without requiring specialized network background execution permissions (`POST_NOTIFICATIONS` only).
3. **No Central Backend Dependency**: Operates independently without requiring an active intermediary server infrastructure.

### 4.2 Limitations of Intent-Based Approach
1. If the fall victim is completely unconscious and alone, they cannot tap the "Send" button in the email app.
2. The phone screen must unlock to display the email client UI.

### 4.3 Requirements for True Automated Background Delivery (Future Production)
To achieve autonomous background delivery without user intervention:
1. **Backend Relay Webhook**: The Android app dispatches an encrypted HTTPS `POST` request to a secure cloud webhook (e.g. AWS Lambda / Google Cloud Function / Firebase Cloud Messaging).
2. **Cloud Mail Dispatcher**: The cloud service sends the email via SendGrid, Amazon SES, or Gmail REST API with proper DKIM/SPF authentication.
3. **SMS Fallback (Telephony)**: Utilize `SmsManager.sendTextMessage()` with `SEND_SMS` permission for instant direct carrier dispatch to emergency contacts.
