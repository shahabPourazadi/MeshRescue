import asyncio
import json
import base64
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import sys
import datetime

# Use Homebrew-installed Cactus (with NPU/ANE kernels) if available, fallback to source build
CACTUS_BREW_LIB = "/opt/homebrew/opt/cactus/lib/libcactus.dylib"
CACTUS_BREW_WEIGHTS = "/opt/homebrew/Cellar/cactus/1.14_1/libexec/weights"
if os.path.exists(CACTUS_BREW_LIB):
    os.environ["CACTUS_LIB_PATH"] = CACTUS_BREW_LIB
    sys.path.append(os.path.join(os.path.dirname(__file__), "cactus", "python"))
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), "cactus", "python"))
try:
    import meshtastic
    import meshtastic.serial_interface
    MESHTASTIC_AVAILABLE = True
except ImportError:
    MESHTASTIC_AVAILABLE = False

local_mesh = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend directory over HTTP to fix browser camera/mic permissions
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")

CACTUS_MODEL = None

@app.on_event("startup")
async def startup_event():
    global local_mesh
    global CACTUS_MODEL
    
    print("\n" + "="*50)
    print("🧠 PRELOADING CACTUS AI ENGINE INTO MEMORY...")
    print("   (This eliminates the 5-10 second cold start delay)")
    try:
        from src.downloads import ensure_model
        from src.cactus import cactus_init
        
        brew_weights = os.path.join(CACTUS_BREW_WEIGHTS, "gemma-4-e2b-it")
        if os.path.exists(CACTUS_BREW_LIB) and os.path.isdir(brew_weights):
            print("⚡ Using Homebrew Cactus with NPU/ANE acceleration!")
            weights = brew_weights
        else:
            print("🔧 Using source-built Cactus (CPU mode)")
            weights = str(ensure_model("google/gemma-4-E2B-it"))
        
        # Load the model into RAM right now
        CACTUS_MODEL = cactus_init(str(weights), None, False)
        print("✅ CACTUS ENGINE READY - Inference will be instantaneous!")
    except Exception as e:
        print(f"⚠️ Failed to preload Cactus: {e}")
    print("="*50 + "\n")
    
    if MESHTASTIC_AVAILABLE:
        try:
            print("📻 Attempting to connect to physical LoRa Radio via Serial...")
            local_mesh = meshtastic.serial_interface.SerialInterface()
            print("✅ Meshtastic LoRa Radio Connected! Broadcasting live to field nodes.")
        except Exception as e:
            print(f"⚠️ No physical LoRa radio detected on USB (or no permission): {e}. Falling back to WebSocket simulation.")
    else:
        print("⚠️ 'meshtastic' package not installed. Skipping physical radio detection.")

@app.on_event("shutdown")
async def shutdown_event():
    global CACTUS_MODEL
    if CACTUS_MODEL is not None:
        from src.cactus import cactus_destroy
        try:
            cactus_destroy(CACTUS_MODEL)
            print("✅ Cactus Engine safely shut down.")
        except Exception:
            pass

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        # 1. Send to local WebSocket Commander (Software Simulation)
        for connection in self.active_connections:
            await connection.send_text(message)
            
        # 2. Transmit to physical radio if plugged in
        if local_mesh:
            try:
                # We slice to ensure it doesn't exceed 200 bytes max payload for LoRa
                compressed_msg = message[:200]
                local_mesh.sendText(compressed_msg)
                print(f"📡 Broadcasted to physical LoRa network: {compressed_msg}")
            except Exception as e:
                # Suppress the 'currentPacketId' spam when a non-radio USB device is connected
                pass

manager = ConnectionManager()

class FramePayload(BaseModel):
    image: str
    pose_data: Dict[str, Any]
    drone_id: str = "UNKNOWN-NODE"
    audio_clip: str = None
    packet_id: str = None
    location: Optional[str] = None  # top-level GPS from drone, overrides pose_data.location

# Fallback mock response
MOCK_GEMMA_RESPONSE = {
    "casualty_status": "conscious_injured",
    "environmental_hazards": ["unstable_rubble"],
    "priority_level": 1,
    "reasoning_summary": "Supine pose detected near debris. High priority."
}

def decode_audio_to_pcm(audio_b64: str) -> bytes | None:
    """
    Decodes a base64-encoded WebM/OGG audio blob from the browser into
    raw 16-bit signed little-endian PCM bytes at 16kHz mono.

    Uses ffmpeg to pipe directly to stdout — no temp WAV file, no disk I/O.
    Cactus Gemma 4 accepts this as pcm_data and handles resampling internally,
    so we output at ANY sample rate and let Cactus resample (simpler, portable).
    Returns raw PCM bytes or None on failure.
    """
    import tempfile, subprocess
    try:
        if ',' in audio_b64:
            audio_b64 = audio_b64.split(',')[1]
        raw_audio = base64.b64decode(audio_b64)

        # Write browser audio to a temp input file (ffmpeg needs seekable input for WebM)
        in_fd, in_path = tempfile.mkstemp(suffix=".webm")
        try:
            with os.fdopen(in_fd, 'wb') as f:
                f.write(raw_audio)

            # Pipe raw s16le PCM directly to stdout — no output file needed
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", in_path,
                    "-ar", "16000",      # 16kHz (Gemma 4 native rate)
                    "-ac", "1",          # Mono
                    "-f", "s16le",       # Raw 16-bit signed LE PCM, no WAV header
                    "pipe:1"             # Write to stdout
                ],
                capture_output=True, timeout=30
            )
        finally:
            try: os.remove(in_path)
            except Exception: pass

        if result.returncode != 0:
            print(f"⚠️ ffmpeg audio conversion failed: {result.stderr.decode(errors='ignore')[:300]}")
            return None

        pcm_bytes = bytearray(result.stdout)
        if len(pcm_bytes) < 1024:
            print("⚠️ Audio clip too short or empty after conversion")
            return None

        print(f"🔊 Audio decoded: {len(pcm_bytes)//2:,} samples @ 16kHz = {len(pcm_bytes)/32000:.1f}s")
        return pcm_bytes
    except Exception as e:
        print(f"⚠️ Audio decode error: {e}")
        return None


def analyze_with_gemma(payload: FramePayload) -> dict:
    """
    Calls a local Ollama instance running Gemma 4 E2B Multimodal.
    """
    has_audio = bool(payload.audio_clip)
    yamnet_label = payload.pose_data.get('audio', 'Unknown')
    if has_audio:
        audio_instruction = (
            f"An AUDIO CLIP is attached. The edge sensor (YAMNet) pre-classified the sound as: '{yamnet_label}'. "
            "Listen carefully to confirm this. Pay extreme attention to ANY distress sounds including human "
            "(coughs, groans, grunts, wheezing, screams, crying) or animal (barks, meows, whimpers) and treat them as an active emergency."
        )
    else:
        audio_instruction = (
            f"No live audio clip attached. The edge sensor (YAMNet) confidently detected: '{yamnet_label}'. "
            "You MUST treat this label as confirmed audio evidence of the scene."
        )

    prompt = f"""
    You are an emergency rescue drone AI analyzing a live scene.
    You have been provided with:
    1. A CAMERA IMAGE of the scene
    {"2. A LIVE AUDIO CLIP recorded at the scene (transcribe it accurately)" if has_audio else ""}

    Real-time sensor telemetry:
    - Posture detector: {payload.pose_data.get('posture', 'Unknown')}
    - Hand gesture: {payload.pose_data.get('hand_gesture', 'No Hands')}
    - Environment class: {payload.pose_data.get('environment', 'Unknown')}
    - Audio note: {audio_instruction}

    INSTRUCTIONS: Think through what you see and hear, then output ONLY a JSON object in a markdown block:
    ```json
    {{
      "audio_transcription": "Exact words spoken in the audio clip. If no speech, describe the dominant sound (e.g. 'crying', 'silence', 'vehicle noise'). Write 'None detected' if no audio provided.",
      "casualty_status": "Highly specific 3-4 word status (e.g. 'conscious and mobile', 'unconscious supine', 'injured calling for help')",
      "environmental_hazards": ["List specific hazards visible in image or audible, e.g. 'smoke', 'rubble', 'screaming nearby'"],
      "priority_level": 1,
      "reasoning_summary": "2-3 sentence explanation of your visual AND audio reasoning that led to this priority level"
    }}
    ```
    CRITICAL RULE: priority_level is 1 (Critical life threat) to 5 (Safe/Normal).
    If the audio clip OR the sensor telemetry indicates screams, cries for help, groans, heavy coughing, or animal distress — ALWAYS set priority 1 or 2.
    If person is SUPINE/unconscious — ALWAYS set priority 1 or 2.
    """
    
    try:
        from src.cactus import cactus_complete
        
        global CACTUS_MODEL
        if CACTUS_MODEL is None:
            # Fallback if startup failed
            print("⚠️ Model not preloaded, starting CPU mock fallback")
            return MOCK_GEMMA_RESPONSE
            
        import base64
        import tempfile
        import os
        
        base64_image = payload.image.split(',')[1] if ',' in payload.image else payload.image
        
        # Cactus requires physical file paths for images, not Base64 data URIs
        temp_img_fd, temp_img_path = tempfile.mkstemp(suffix=".jpg")
        with os.fdopen(temp_img_fd, 'wb') as f:
            f.write(base64.b64decode(base64_image))
        
        print("\n" + "="*50)
        print(f"🚀 FIRING TIER 2 REASONING (Cactus Gemma 4 E2B)")
        print("="*50)
        
        messages = [
            {
                "role": "system",
                "content": "You are an emergency triage AI on a rescue drone."
            },
            {
                "role": "user",
                "content": prompt,
                "images": [temp_img_path]
            }
        ]
        
        # --- Audio Processing ---
        pcm_bytes = None
        if payload.audio_clip:
            print(f"🔊 Processing audio clip → 16kHz mono PCM for Cactus audio encoder...")
            pcm_bytes = decode_audio_to_pcm(payload.audio_clip)
            if pcm_bytes:
                # PCM bytes passed directly to cactus_complete (Priority 1 in Cactus engine).
                # No need to also set 'audio' path in message — PCM takes precedence.
                print(f"✅ PCM audio ready: {len(pcm_bytes):,} bytes")
            else:
                print("⚠️ Audio decode failed — proceeding with image-only analysis")
        else:
            print("🔇 No audio clip received — image-only analysis")

        # --- Native Function Calling (Gemma 4 via Cactus) ---
        # Define dispatch_rescue_team tool — Gemma 4 will call this autonomously
        # when it determines a situation is critical and requires immediate response.
        dispatch_tool = [
            {
                "type": "function",
                "function": {
                    "name": "dispatch_rescue_team",
                    "description": (
                        "Dispatch an emergency rescue team to a casualty location. "
                        "Call this ONLY when priority_level is 1 or 2 (critical or high risk to life). "
                        "Do NOT call for priority 3-5."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "GPS coordinates or best description of casualty location"
                            },
                            "priority": {
                                "type": "integer",
                                "description": "Priority level (1=Critical, 2=High)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief reason for dispatch (what the model observed)"
                            }
                        },
                        "required": ["location", "priority", "reason"]
                    }
                }
            }
        ]
        tools_json_str = json.dumps(dispatch_tool)

        # cactus_complete(model, messages_json, options_json, tools_json, callback, pcm_data)
        options = {"auto_handoff": False, "max_tokens": 700}
        
        def on_token(text, token_id):
            print(text, end="", flush=True)
        
        print("📝 Generating response (with function calling enabled)...")
        response_text = cactus_complete(
            CACTUS_MODEL,
            json.dumps(messages),
            json.dumps(options),
            tools_json_str,  # Native Gemma 4 function calling
            on_token,        # streaming token callback
            pcm_bytes        # raw PCM bytes for audio encoder
        )
        
        print(f"\n\nRaw Gemma 4 Response:\n{response_text}")
        
        # cactus_complete returns JSON: {"success":true,"response":"...","tool_calls":[...],"confidence":...}
        import re
        cactus_result = json.loads(response_text)

        # --- Handle Tool Calls (Gemma 4 native function calling) ---
        dispatch_event = None
        tool_calls = cactus_result.get("tool_calls", [])
        if tool_calls:
            for call in tool_calls:
                fn = call.get("function", {})
                if fn.get("name") == "dispatch_rescue_team":
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception: args = {}
                    print(f"\n🚨 GEMMA 4 DISPATCHING RESCUE TEAM → {args}")

                    # Try Meshtastic LoRa radio dispatch if device is configured
                    lora_device = os.getenv("LORA_DEVICE", "")  # e.g. /dev/ttyUSB0 or COM3
                    lora_method = "queued_wifi"
                    if lora_device:
                        try:
                            import meshtastic
                            import meshtastic.serial_interface
                            iface = meshtastic.serial_interface.SerialInterface(lora_device)
                            dispatch_msg = (
                                f"DISPATCH P{args.get('priority',1)}: "
                                f"{args.get('reason','')} @ {args.get('location','?')}"
                            )
                            iface.sendText(dispatch_msg)
                            iface.close()
                            lora_method = "lora_transmitted"
                            print(f"📡 Dispatch transmitted over LoRa ({lora_device}): {dispatch_msg}")
                        except Exception as e:
                            print(f"⚠️ LoRa dispatch failed ({lora_device}): {e}")
                    else:
                        print("📡 [SIMULATED] No LORA_DEVICE configured — dispatch will sync via WiFi/WebSocket")

                    dispatch_event = {
                        "dispatched": True,
                        "location": args.get("location", "Unknown"),
                        "priority": args.get("priority", 1),
                        "reason": args.get("reason", ""),
                        "method": lora_method,
                        "timestamp": datetime.datetime.now().isoformat()
                    }

        if cactus_result.get("success"):
            llm_text = cactus_result.get("response", "")
        else:
            llm_text = response_text
            
        # Cleanup temp image only (no WAV temp file with PCM pipe approach)
        if temp_img_path:
            try:
                os.remove(temp_img_path)
            except Exception:
                pass
        
        # Extract the triage JSON from the LLM's text response
        json_match = re.search(r'```json\s*(.*?)\s*```', llm_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
            json_str = json_match.group(0) if json_match else "{}"
            
        result = json.loads(json_str)
        # Attach dispatch event if Gemma 4 invoked the function tool
        if dispatch_event:
            result["dispatch"] = dispatch_event
            print(f"✅ Dispatch event attached to triage result: {dispatch_event['method']}")
        return result
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Cactus local call failed or timed out: {e}. Using mock response for demo.")
        
        # Cleanup on error
        try:
            os.remove(temp_img_path)
        except Exception:
            pass
    
    return MOCK_GEMMA_RESPONSE

@app.post("/api/analyze_only")
async def process_frame_only(payload: FramePayload):
    print("Received frame from drone tier 1. Waking Gemma...")
    triage_result = analyze_with_gemma(payload)
    
    triage_result["type"] = "triage"
    triage_result["timestamp"] = datetime.datetime.now().isoformat()
    # Resolve GPS: prefer top-level location field, then pose_data.location,
    # then .env EDGE_LOCATION. Reject non-parseable values (e.g. 'Locating GPS...').
    def _valid_coords(s):
        if not s or ',' not in s: return False
        try:
            parts = s.split(',')
            float(parts[0]); float(parts[1])
            return True
        except (ValueError, IndexError):
            return False

    raw_location = payload.location or payload.pose_data.get("location", "")
    if _valid_coords(raw_location):
        gps = raw_location.strip()
    else:
        gps = os.getenv("EDGE_LOCATION", "Unknown")
    triage_result["gps_coordinates"] = gps
    triage_result["drone_id"] = payload.drone_id
    
    # Pass along the unique ID from the edge node if present
    if hasattr(payload, 'packet_id') and payload.packet_id:
        triage_result["packet_id"] = payload.packet_id
        
    triage_result["image"] = payload.image
    if payload.audio_clip:
        triage_result["audio_clip"] = payload.audio_clip
        
    return {"status": "analyzed", "data": triage_result}

class LoraPayload(BaseModel):
    data: Dict[str, Any]

@app.post("/api/lora_broadcast")
async def lora_broadcast(payload: LoraPayload):
    print("📡 TRANSMITTING LIGHTWEIGHT LORA PAYLOAD (NO MEDIA)")
    triage_result = payload.data
    # Ensure no image or audio is transmitted over LoRa
    if "image" in triage_result:
        del triage_result["image"]
    if "audio_clip" in triage_result:
        del triage_result["audio_clip"]
        
    await manager.broadcast(json.dumps(triage_result))
    return {"status": "transmitted_lora"}

@app.post("/api/wifi_broadcast")
async def wifi_broadcast(payload: LoraPayload):
    print("📶 TRANSMITTING FULL WIFI PAYLOAD (WITH MEDIA)")
    triage_result = payload.data
    await manager.broadcast(json.dumps(triage_result))
    return {"status": "transmitted_wifi"}

import os

# Load .env variables manually to avoid extra dependencies
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

@app.get("/api/config")
async def get_config():
    return {
        "auto_trigger_period_sec": int(os.getenv("AUTO_TRIGGER_PERIOD_SEC", "10")),
        "llm_model": os.getenv("LLM_MODEL", "gemma4:e2b"),
        "enable_pose": os.getenv("ENABLE_POSE", "true").lower() == "true",
        "enable_hand": os.getenv("ENABLE_HAND", "true").lower() == "true",
        "enable_audio": os.getenv("ENABLE_AUDIO", "true").lower() == "true",
        "enable_env": os.getenv("ENABLE_ENV", "true").lower() == "true",
        "edge_location": os.getenv("EDGE_LOCATION", "Unknown Location"),
        "commander_hub_url": os.getenv("COMMANDER_HUB_URL", "http://localhost:8000")
    }

class TelemetryData(BaseModel):
    posture: str
    location: str = "Unknown Location"
    status: str = "normal"
    drone_id: str = "UNKNOWN-NODE"
    raw_data: Dict[str, Any] = {}

@app.post("/api/telemetry")
async def receive_telemetry(data: TelemetryData):
    # Broadcast telemetry to all connected Commanders
    payload = {
        "type": "telemetry",
        "posture": data.posture,
        "location": data.location,
        "status": data.status,
        "drone_id": data.drone_id,
        "raw_data": data.raw_data
    }
    await manager.broadcast(json.dumps(payload))
    return {"status": "ok"}

@app.websocket("/ws/commander")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
