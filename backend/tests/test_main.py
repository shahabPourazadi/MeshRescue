import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_analyze_only():
    """
    Submitting a valid payload to /api/analyze_only successfully returns a structured JSON triage report.
    This tests the public behavior without triggering WebSockets or LoRa.
    """
    payload = {
        "packet_id": "test_123",
        "drone_id": "TEST-DRONE-01",
        "location": "34.05, -118.24",
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        "pose_data": {"landmarks": [{"x": 0.5, "y": 0.5, "z": 0.1}], "posture": "SUPINE"}
    }
    
    response = client.post("/api/analyze_only", json=payload)
    
    # Assert
    assert response.status_code == 200
    json_data = response.json()
    
    assert json_data["status"] == "analyzed"
    assert "data" in json_data
    
    # Check that the data conforms to our schema
    triage_data = json_data["data"]
    assert "casualty_status" in triage_data
    assert "environmental_hazards" in triage_data
    assert "priority_level" in triage_data
    assert "reasoning_summary" in triage_data
    
    # Check for the added telemetry fields
    assert "gps_coordinates" in triage_data
    assert "drone_id" in triage_data
    assert "packet_id" in triage_data

def test_invalid_payload_rejection():
    """
    If the client sends an invalid payload (e.g., missing the image), 
    the API should reject it with a 422 Validation Error.
    """
    invalid_payload = {
        "packet_id": "test_123",
        "pose_data": {"landmarks": [{"x": 0.5, "y": 0.5, "z": 0.1}]}
        # 'image' is missing
    }
    
    response = client.post("/api/analyze_only", json=invalid_payload)
    
    assert response.status_code == 422
    assert "detail" in response.json()

def test_telemetry_endpoint():
    """
    Test that the simple telemetry endpoint accepts posture data and returns success.
    """
    payload = {
        "posture": "UPRIGHT / MOVING",
        "status": "safe",
        "location": "0,0",
        "drone_id": "TEST",
        "raw_data": {}
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_lora_broadcast_endpoint():
    """
    Test the manual LoRa broadcast route.
    """
    payload = {
        "data": {
            "packet_id": "test_123",
            "drone_id": "TEST-DRONE-01",
            "priority_level": 1,
            "casualty_status": "Severe trauma"
        }
    }
    response = client.post("/api/lora_broadcast", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "transmitted_lora"
