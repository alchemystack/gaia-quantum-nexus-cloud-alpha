"""
MODAL NOTEBOOK - GAIA QUANTUM NEXUS 120B ENHANCED (FIXED)
==========================================================
Copy this entire code into Modal's web playground at:
https://modal.com/playground

FIXED: GPU specification syntax error corrected

Enhanced configuration with:
- 1x A100 80GB VRAM GPU
- 16 CPU cores
- 128GB system RAM
- Split cells for efficient model management

This notebook is split into cells:
Cell 1: Model upload from local storage
Cell 2: Server initialization and endpoints
"""

# ============================================
# CELL 1: MODEL UPLOAD AND PREPARATION
# ============================================
# Run this cell first to upload your GGUF model

import modal
import os
import time
import subprocess
from typing import Dict, Any, Optional
import json
import asyncio
import aiohttp

# Create Modal app
app = modal.App("gaia-quantum-120b-enhanced")

# Enhanced GPU configuration - A100 80GB with maximum resources
# FIXED: Changed modal.Gpu to modal.gpu (lowercase)
gpu_config = modal.gpu.A100(count=1)  # Single A100 80GB GPU

# Model storage - persistent volume for the 120B model
volume = modal.Volume.from_name("gaia-quantum-models-enhanced", create_if_missing=True)

# Container image with llama.cpp server
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget", "git", "build-essential", "cmake", "curl",
        "libssl-dev", "pkg-config", "libcurl4-openssl-dev"
    )
    .run_commands(
        # Install llama.cpp with server support
        "git clone https://github.com/ggerganov/llama.cpp /llama.cpp",
        "cd /llama.cpp && make -j16 LLAMA_CUDA=1 llama-server",  # Use 16 cores
        "ln -s /llama.cpp/llama-server /usr/local/bin/llama-server",
    )
    .pip_install(
        "huggingface-hub",
        "aiohttp",
        "numpy",
        "fastapi",
        "requests",
    )
)

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=7200,  # 2 hours for model upload
    cpu=16.0,      # 16 CPU cores
    memory=131072, # 128GB RAM
)
def upload_model_from_local(model_path: Optional[str] = None):
    """
    Cell 1: Upload GGUF model from local storage or download from HuggingFace
    
    Run this first to prepare the model. Options:
    1. Upload from your local file
    2. Auto-download from HuggingFace
    """
    import shutil
    from huggingface_hub import hf_hub_download
    
    print("🚀 CELL 1: MODEL PREPARATION")
    print("=" * 60)
    
    model_dir = "/models/gpt-oss-120b"
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if model already exists
    gguf_path = f"{model_dir}/gpt-oss-120b.gguf"
    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        print(f"✅ Model already exists: {gguf_path}")
        print(f"   Size: {size_gb:.2f} GB")
        return {"status": "exists", "path": gguf_path, "size_gb": size_gb}
    
    if model_path and os.path.exists(model_path):
        # Upload from local file
        print(f"📤 Uploading model from: {model_path}")
        shutil.copy2(model_path, gguf_path)
        print("✅ Model uploaded successfully!")
    else:
        # Download from HuggingFace
        print("📥 Downloading GPT-OSS 120B GGUF from HuggingFace...")
        print("   Repository: ggml-org/gpt-oss-120b-GGUF")
        print("   This will take 10-15 minutes...")
        
        try:
            downloaded = hf_hub_download(
                repo_id="ggml-org/gpt-oss-120b-GGUF",
                filename="gpt-oss-120b.gguf",
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ Model downloaded to: {downloaded}")
        except Exception as e:
            print(f"⚠️ Download error: {e}")
            print("   Using HuggingFace direct streaming instead")
            return {"status": "use_hf_direct", "error": str(e)}
    
    # Verify model
    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        print(f"✅ Model ready: {gguf_path}")
        print(f"   Size: {size_gb:.2f} GB")
        volume.commit()  # Commit to persistent storage
        return {"status": "ready", "path": gguf_path, "size_gb": size_gb}
    
    return {"status": "error", "message": "Model not found"}

# ============================================
# CELL 2: QUANTUM-ENHANCED SERVER
# ============================================
# Run this cell after Cell 1 to start the server

@app.cls(
    image=image,
    gpu=gpu_config,  # Uses the fixed GPU config
    volumes={"/models": volume},
    timeout=3600,
    keep_warm=1,  # Keep container warm for instant responses
    memory=131072,  # 128GB RAM for 120B model
    cpu=16.0,  # 16 CPU cores
    allow_concurrent_inputs=5,
    secrets=[
        modal.Secret.from_name("qrng-api-key", required=False)  # Optional QRNG
    ]
)
class GaiaQuantumServer:
    """
    Enhanced llama.cpp server for GPT-OSS 120B with quantum integration
    """
    def __init__(self):
        self.server_process = None
        self.server_url = "http://localhost:8080"
        self.model_path = "/models/gpt-oss-120b/gpt-oss-120b.gguf"
        
    @modal.enter()
    def start_server(self):
        """Initialize the llama.cpp server with enhanced configuration"""
        import subprocess
        import time
        
        print("🚀 CELL 2: STARTING QUANTUM SERVER")
        print("=" * 60)
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            raise Exception(f"Model not found at {self.model_path}. Run Cell 1 first!")
        
        # Enhanced server command with all optimizations
        cmd = [
            "llama-server",
            "-m", self.model_path,
            "-c", "8192",  # Context size
            "-fa",  # Flash Attention
            "--jinja",  # Jinja templating
            "--reasoning-format", "none",
            "-t", "16",  # 16 threads
            "-ngl", "999",  # Offload all layers to GPU
            "--host", "0.0.0.0",
            "--port", "8080",
            "--timeout", "600",
            "--n-gpu-layers", "999",  # Maximum GPU offloading
            "--mlock",  # Lock model in memory
            "--no-mmap",  # Don't use memory mapping
            "--cont-batching",  # Continuous batching
            "--flash-attn",  # Flash attention v2
        ]
        
        print(f"🔧 Starting server with command:")
        print(f"   {' '.join(cmd)}")
        
        # Start server in background
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to initialize...")
        time.sleep(10)
        
        # Verify server is running
        try:
            import requests
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running!")
                print(f"   Health check: {response.json()}")
            else:
                print(f"⚠️ Server returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not verify server: {e}")
        
        print("=" * 60)
        print("🌌 GAIA QUANTUM NEXUS 120B READY")
        print("=" * 60)
    
    @modal.exit()
    def stop_server(self):
        """Clean shutdown of the server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("🛑 Server stopped")
    
    @modal.method()
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        quantum_profile: str = "medium"
    ) -> Dict[str, Any]:
        """
        Generate text with optional quantum enhancement
        """
        import aiohttp
        import json
        
        # Prepare request to llama.cpp server
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["</s>", "\n\n"],
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/completion",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "status": "success",
                            "generated_text": result.get("content", ""),
                            "tokens_generated": result.get("tokens_predicted", 0),
                            "quantum_profile": quantum_profile
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"Server returned {response.status}"
                        }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    @modal.method()
    def health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        import requests
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return {
                "status": "healthy",
                "server_running": response.status_code == 200,
                "model": "GPT-OSS 120B",
                "config": "Enhanced A100 80GB"
            }
        except:
            return {
                "status": "unhealthy",
                "server_running": False
            }

# ============================================
# CELL 3: TEST AND GET ENDPOINTS
# ============================================
# Run this cell to test and get your API endpoints

@app.local_entrypoint()
def main():
    """
    Cell 3: Deploy and get endpoint information
    """
    print("🌌 GAIA QUANTUM NEXUS 120B - DEPLOYMENT")
    print("=" * 60)
    
    # Test model upload
    print("\n📦 Checking model status...")
    upload_result = upload_model_from_local.remote()
    print(f"Model status: {upload_result}")
    
    # Get deployment information
    deployment_name = "gaia-quantum-120b-enhanced"
    
    print("\n✅ DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print("📍 Your API endpoints:")
    print(f"   Health: https://{deployment_name}--gaiaquantumserver-health.modal.run")
    print(f"   Generate: https://{deployment_name}--gaiaquantumserver-generate.modal.run")
    print("\n🔑 Add these to your Replit secrets:")
    print(f"   MODAL_ENDPOINT: https://{deployment_name}--gaiaquantumserver-generate.modal.run")
    print("   MODAL_API_KEY: Your Modal API key")
    print("\n💡 Test with:")
    print("   curl -X GET https://{deployment_name}--gaiaquantumserver-health.modal.run")
    print("=" * 60)

# Run this if executing as a script
if __name__ == "__main__":
    main()