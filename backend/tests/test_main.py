import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_triage_processing_tracer_bullet():
    """
    Tracer Bullet: Submitting a valid payload to /api/triage successfully returns a structured JSON triage report.
    This tests the public behavior without mocking the internal `analyze_with_gemma` function.
    Because we don't have a real Ollama instance guaranteed during testing, it should gracefully fall back to the mock.
    """
    payload = {
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        "pose_data": {"landmarks": [{"x": 0.5, "y": 0.5, "z": 0.1}]}
    }
    
    response = client.post("/api/triage", json=payload)
    
    # Assert
    assert response.status_code == 200
    json_data = response.json()
    
    assert json_data["status"] == "transmitted"
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

def test_invalid_payload_rejection():
    """
    If the client sends an invalid payload (e.g., missing the image), 
    the API should reject it with a 422 Validation Error.
    """
    invalid_payload = {
        "pose_data": {"landmarks": [{"x": 0.5, "y": 0.5, "z": 0.1}]}
        # 'image' is missing
    }
    
    response = client.post("/api/triage", json=invalid_payload)
    
    assert response.status_code == 422
    assert "detail" in response.json()

def test_telemetry_endpoint():
    """
    Test that the simple telemetry endpoint accepts posture data and returns success.
    """
    payload = {
        "posture": "UPRIGHT / MOVING",
        "status": "safe"
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "transmitted"
