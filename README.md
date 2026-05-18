# MeshRescue: Edge AI Triage Network

MeshRescue is an offline-capable, decentralized Edge AI triage system designed for remote rescue operations. It utilizes the [Cactus Engine](https://github.com/cactus-compute/cactus) to run Google's **Gemma 4 E2B Multimodal** locally on field devices — no internet connection required.

By simultaneously analyzing real-time images and raw audio waveforms, the system autonomously categorizes casualties, invokes **native Gemma 4 function calling** to dispatch rescue teams, and synchronizes tactical alerts over low-bandwidth LoRa/Meshtastic radios.

## Key Features
*   **Fully Offline:** No cloud APIs. The entire intelligence stack runs on-device using the Cactus C++ Inference Engine with ANE/NPU acceleration on Apple Silicon.
*   **Native Multimodal Inference:** Raw PCM audio bytes are fed directly into Gemma 4's Conformer audio encoder. The model *listens* and *looks* simultaneously — not speech-to-text, actual audio reasoning.
*   **Gemma 4 Native Function Calling:** When a critical situation is detected, Gemma 4 autonomously invokes the `dispatch_rescue_team` function, transmitting a structured dispatch order over LoRa or logging it for queue sync.
*   **Two-Phase LoRa + WiFi Sync:** Critical text alerts travel over LoRa mesh radio instantly. Full image and audio evidence uploads asynchronously over WiFi, dynamically filling the Commander's incident card.

## Architecture
1.  **Field Node (`drone.html`):** Runs in any browser on any device — phone, tablet, laptop, or Raspberry Pi with Chromium. Captures images and 8-second audio clips. Uses an ultra-low-power "Instinct Gate" via **MediaPipe Vision (Pose/Hand Gestures)** and **MediaPipe YAMNet (Audio Classification)** to detect distress sounds (coughs, groans) and human bodies before waking up the LLM.
2.  **Compute Node (`main.py`):** A headless Python backend running FastAPI + Cactus Python SDK. Runs on the device with the most compute (Mac, Linux server, or Raspberry Pi 5). Hosts the model and all inference.
3.  **Commander Dashboard (`commander.html`):** Runs in any browser on any device on the same network. Receives real-time WebSocket triage data and renders prioritized incident cards on a live Leaflet map.

---

## 🖥️ Platform Coverage

| Role | Supported Platforms | Notes |
|---|---|---|
| **Compute Node** (Cactus + Python) | macOS 13+ (Apple Silicon) · Linux ARM64 · Linux x86_64 | Raspberry Pi 5 (ARM64) supported — see Linux setup below |
| **Field Node** (drone.html) | Any modern browser — iOS, Android, macOS, Windows, ChromeOS | Camera/mic need HTTPS on iOS — use ngrok or LAN IP on Android |
| **Commander** (commander.html) | Any modern browser on any device | Navigate to `http://[COMPUTE_IP]:8000/app/commander.html` |

---

## 🚀 Getting Started

### 1. Prerequisites (Compute Node only)
*   macOS 13+ (Apple Silicon recommended for ANE acceleration) **or** Linux ARM64/x86_64 (including Raspberry Pi 5)
*   Python 3.10+
*   Git
*   `ffmpeg` (for audio transcoding): `brew install ffmpeg` on macOS · `sudo apt install ffmpeg` on Linux

> **Field Node & Commander** require only a modern browser — no installation needed.

### 2. Setup & Installation

MeshRescue includes a bootstrapping script that automatically configures your Python environment, installs dependencies, builds the Cactus Engine, and downloads the Gemma 4 E2B weights (~4.6 GB).

**macOS / Linux:**
```bash
git clone https://github.com/shahabPourazadi/MeshRescue
cd MeshRescue
cp .env.example .env        # Add your HuggingFace token
chmod +x start_demo.sh
./start_demo.sh
```

### What `start_demo.sh` does automatically:
1.  Creates a Python virtual environment in `backend/`.
2.  Installs Python dependencies (`fastapi`, `uvicorn`, `meshtastic`, etc.).
3.  Clones and builds the Cactus Compute Engine from source.
4.  Downloads `google/gemma-4-E2B-it` weights from HuggingFace (~4.6 GB, one-time).
5.  Starts the FastAPI backend on port 8000.
6.  Opens `commander.html` and `drone.html` in your browser.

### 3. Running the Live Demo (Local Simulation)
1.  Arrange the **Commander** and **Drone** windows side-by-side (or open on separate devices).
2.  On the **Drone** screen, wait for MediaPipe to detect your posture.
3.  Say something out loud or cough (e.g., *"Help! I need a medic!"*) — the microphone records 8 seconds.
4.  The browser's "Instinct Gate" automatically wakes up the local Gemma 4 backend when a distress sound or person is detected, or you can manually click **"Capture & Analyze"**.
5.  Gemma 4 analyzes both the image and audio entirely offline. Watch the backend terminal for the real-time token stream.
6.  If priority is Critical or High, Gemma 4 will autonomously call `dispatch_rescue_team` — a 🚨 banner appears on the Commander card.
7.  The LoRa stub card appears on Commander instantly via a simulated airgap websocket. The full image/audio uploads automatically if WiFi is available.

### 4. Connecting from a Phone or Tablet
1.  Ensure your phone and Compute Node are on the same Wi-Fi network.
2.  Find your computer's local IP: `ipconfig getifaddr en0` (Mac) or `hostname -I` (Linux).
3.  Open your phone's browser and go to: `http://[COMPUTER_IP]:8000/app/drone.html`

> **iOS note:** Safari restricts camera/microphone on plain HTTP. For full functionality, tunnel with `ngrok http 8000` and use the `https://` URL it provides.

### 5. True Edge Hardware Deployment (Raspberry Pi / Drone)
For a true off-grid "Compute Backpack" deployment:
1. Flash a Raspberry Pi 5 (8GB) or Nvidia Jetson with Ubuntu.
2. Run `sudo apt update && sudo apt install -y python3 python3-pip git ffmpeg cmake build-essential`
3. Plug a Meshtastic LoRa Radio (e.g., LILYGO T-Echo) into the Pi's USB port.
4. In the `.env` file, set `LORA_DEVICE=/dev/ttyACM0` (or your specific USB port).
5. Run `uvicorn backend.main:app --host 0.0.0.0 --port 8000` on the Pi.
6. The Pi broadcasts a local offline Wi-Fi hotspot. A paramedic connects their smartphone to this Wi-Fi, navigates to `http://<PI_IP>:8000/app/drone.html`, and acts as the mobile sensor. The Pi securely handles the heavy Gemma 4 inference and physical LoRa radio transmission from inside the backpack.

## Acknowledgements
*   [Cactus Compute](https://github.com/cactus-compute/cactus) — on-device multimodal inference engine with Gemma 4 support.
*   [Google Gemma](https://huggingface.co/google/gemma-4-e2b-it) — highly-optimized Edge AI model with audio + vision + text reasoning.
*   [MediaPipe](https://developers.google.com/mediapipe) — real-time pose and hand gesture detection in the browser.
