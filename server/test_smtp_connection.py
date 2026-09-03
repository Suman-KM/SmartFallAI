"""
SmartFall AI — Safe SMTP Connection & Delivery Verification
Tests authenticated SMTP submission against smtp.gmail.com:587.
Does NOT print secrets, passwords, or tokens.
"""

import os
import sys

# Ensure server root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emergency_service import get_smtp_config, format_emergency_email, send_smtp_email

def test_smtp():
    config = get_smtp_config()
    print("=== SmartFall AI SMTP Diagnostic ===")
    print(f"SMTP Host:       {config['host']}")
    print(f"SMTP Port:       {config['port']}")
    print(f"SMTP User:       {config['user'] if config['user'] else '(NOT CONFIGURED)'}")
    print(f"SMTP Password:   {'[CONFIGURED, len=' + str(len(config['password'])) + ']' if config['password'] else '(NOT CONFIGURED)'}")
    print(f"SMTP From:       {config['from']}")
    print(f"SMTP TLS:        {config['use_tls']}")
    print(f"SMTP SSL:        {config['use_ssl']}")
    print(f"Recipient:       {config['recipient']}")
    print(f"Configured:      {config['configured']}")
    print("=====================================")

    if not config["configured"]:
        print("RESULT: FAILED — Missing SMTP_USER or SMTP_PASSWORD in server/.env")
        return False

    test_payload = {
        "event": "FALL_CONFIRMED",
        "deviceSource": "SmartFall AI Diagnostic Tester",
        "timeString": "03 September 2026, 17:30:00",
        "heartRate": 82,
        "latitude": 14.4594,
        "longitude": 75.9240,
        "accuracy": 12.0,
        "eventId": "test-diag-event"
    }
    subject, body = format_emergency_email(test_payload)
    print(f"Attempting REAL SMTP delivery to: {config['recipient']}...")
    result = send_smtp_email([config["recipient"]], subject, body)
    print(f"Delivery Result: {result}")
    return result.get("success", False)

if __name__ == "__main__":
    success = test_smtp()
    sys.exit(0 if success else 1)
