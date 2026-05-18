import sys
import os

# Add Cactus Python SDK to path
sys.path.append(os.path.join(os.path.dirname(__file__), "cactus", "python"))

try:
    from src.downloads import ensure_model
    print("🧠 Starting Gemma 4 E2B weights download (4.68GB)...")
    print("⏳ Please do not close the terminal. The progress bar will update shortly...")
    
    # Clear stale file locks caused by Ctrl+C interruptions
    lock_dir = os.path.expanduser("~/.cache/huggingface/hub/.locks/models--Cactus-Compute--gemma-4-E2B-it")
    if os.path.exists(lock_dir):
        import shutil
        shutil.rmtree(lock_dir, ignore_errors=True)
        
    # ensure_model will show the tqdm progress bar
    weights_path = ensure_model('google/gemma-4-E2B-it')
    
    print(f"✅ Model successfully downloaded and extracted to: {weights_path}")
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    sys.exit(1)
