#!/bin/bash
echo "🚀 Starting MeshRescue Demo Environment..."
if [ -f ".env" ]; then
    echo "🔐 Loading configuration from .env..."
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
        export "$line"
    done < .env
fi

# Setup Python Backend
echo "📦 Setting up Python backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Increase macOS open-file limit to prevent hf-xet from hanging during chunked downloads
ulimit -n 65536

# Install Cactus Engine if not present
if [ ! -d "cactus" ]; then
    echo "🌵 Cactus Compute Engine not found. Installing and Building from Source..."
    git clone https://github.com/cactus-compute/cactus
    cd cactus
    source ./setup
    cactus build --python
    cd ..
    echo "✅ Cactus Engine Installed."
else
    echo "✅ Cactus Engine is already installed."
fi

# Pre-fetch Gemma 4 E2B weights using the python sdk downloads tool
echo "🧠 Pre-fetching google/gemma-4-E2B-it weights (this may take a moment on first run)..."
python3 download_model.py

echo "-----------------------------------------------------"

# Start FastAPI server in the background
echo "🟢 Starting FastAPI Backend Node (Cactus Multimodal)..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "✅ Backend running on http://localhost:8000"

# Wait until the backend has finished preloading the model and is accepting requests
echo "⏳ Waiting for backend to finish preloading model (this takes ~10 seconds)..."
while ! curl -s http://localhost:8000/api/config > /dev/null; do
    sleep 1
done

echo "🌐 Opening Drone and Commander nodes in your browser..."

# For Mac
open http://localhost:8000/app/commander.html
open http://localhost:8000/app/drone.html

echo "====================================================="
echo "🎥 DEMO READY FOR RECORDING!"
echo "1. Position the Commander window on one side of your screen."
echo "2. Position the Drone window on the other side."
echo "3. Step into the webcam frame and lie down to trigger the gate."
echo "4. Press Ctrl+C in this terminal to kill the backend when done."
echo "====================================================="

# Wait for user to kill script
wait $BACKEND_PID
