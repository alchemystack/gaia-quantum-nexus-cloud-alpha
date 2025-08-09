#!/usr/bin/env python3
"""
🌌 OPENAI GPT-OSS 120B - MODAL DEPLOYMENT (CLEAN VERSION)
Copy and paste this into your Modal notebook cells
"""

# ============================================
# CELL 1: SETUP AND IMPORTS
# ============================================
import modal
import os
import time
from typing import Dict, Any, Optional, List

# Create Modal app
app = modal.App("gaia-quantum-120b-enhanced")

# GPU configuration - A100 80GB (Modal auto-selects 80GB variant)
gpu_config = modal.gpu.A100(count=1)

# Model storage volume
volume = modal.Volume.from_name("gaia-quantum-models-enhanced", create_if_missing=True)

# Create container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "build-essential",
        "cmake",
        "git",
        "wget",
        "curl",
        "libssl-dev",
        "libcurl4-openssl-dev"
    ])
    .run_commands(
        # Install llama.cpp
        "cd /tmp && git clone https://github.com/ggerganov/llama.cpp",
        "cd /tmp/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j 8",
        "cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/"
    )
    .pip_install(
        "fastapi",
        "uvicorn",
        "httpx",
        "numpy",
        "huggingface-hub",
        "requests",
    )
)

# ============================================
# CELL 2: MODEL UPLOAD FUNCTION
# ============================================
@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=7200,
    cpu=16.0,
    memory=131072,
)
def upload_model_from_local():
    """
    Upload OpenAI GPT-OSS 120B model
    This will download from HuggingFace since local paths won't work in Modal
    """
    from huggingface_hub import hf_hub_download
    
    print("🚀 OPENAI GPT-OSS 120B MODEL PREPARATION")
    print("=" * 60)
    
    model_dir = "/models/openai-gpt-oss-120b"
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if model exists
    part1_path = f"{model_dir}/gpt-oss-120b-MXFP4-00001-of-00002.gguf"
    part2_path = f"{model_dir}/gpt-oss-120b-MXFP4-00002-of-00002.gguf"
    
    if os.path.exists(part1_path) and os.path.exists(part2_path):
        print("✅ Model already exists in volume")
        return {"status": "exists"}
    
    print("📥 Downloading from HuggingFace...")
    print("   This will take 10-15 minutes...")
    
    try:
        # Download both parts
        hf_hub_download(
            repo_id="lmstudio-community/gpt-oss-120b-GGUF",
            filename="gpt-oss-120b-MXFP4-00001-of-00002.gguf",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        hf_hub_download(
            repo_id="lmstudio-community/gpt-oss-120b-GGUF",
            filename="gpt-oss-120b-MXFP4-00002-of-00002.gguf",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        print("✅ Model downloaded successfully!")
        return {"status": "ready"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}

# ============================================
# CELL 3: SERVER CLASS
# ============================================
@app.cls(
    image=image,
    gpu=gpu_config,
    volumes={"/models": volume},
    timeout=3600,
    keep_warm=1,
    memory=131072,
    cpu=16.0,
    allow_concurrent_inputs=10,
)
class QuantumGPT120B:
    """OpenAI GPT-OSS 120B Server"""
    
    def __init__(self):
        self.server_process = None
        self.model_loaded = False
        self.server_url = "http://localhost:8080"
        
    @modal.enter()
    def load_model(self):
        """Initialize the server"""
        import subprocess
        import requests
        
        print("🚀 STARTING OPENAI GPT-OSS 120B SERVER")
        print("=" * 60)
        
        model_dir = "/models/openai-gpt-oss-120b"
        part1_path = f"{model_dir}/gpt-oss-120b-MXFP4-00001-of-00002.gguf"
        
        if os.path.exists(part1_path):
            print(f"✅ Using local model: {part1_path}")
            model_arg = ["-m", part1_path]
        else:
            print("📡 Using HuggingFace streaming")
            model_arg = ["-hf", "lmstudio-community/gpt-oss-120b-GGUF"]
        
        # Start server
        cmd = [
            "llama-server",
            *model_arg,
            "-c", "8192",
            "-fa",
            "--jinja",
            "--reasoning-format", "none",
            "--host", "0.0.0.0",
            "--port", "8080",
            "-ngl", "999",
            "-t", "16",
            "--mlock",
            "--verbose"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server
        print("\n⏳ Loading model (1-2 minutes)...")
        time.sleep(30)
        
        for i in range(10):
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ Server is running!")
                    self.model_loaded = True
                    break
            except:
                pass
            
            if i < 9:
                print(f"   Retry {i+1}/10...")
                time.sleep(10)
        
        if self.model_loaded:
            print("\n✅ MODEL READY")
    
    @modal.method()
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text"""
        import requests
        
        start_time = time.time()
        
        request = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/completion",
                json=request,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                elapsed = (time.time() - start_time) * 1000
                
                return {
                    "status": "success",
                    "generated_text": result.get("content", ""),
                    "latency_ms": int(elapsed),
                    "model": "OpenAI GPT-OSS 120B"
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @modal.web_endpoint(method="POST")
    async def generate_endpoint(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Web endpoint for generation"""
        return await self.generate(
            prompt=request.get("prompt", ""),
            max_tokens=request.get("max_tokens", 128),
            temperature=request.get("temperature", 0.7)
        )
    
    @modal.web_endpoint(method="GET")
    async def health(self) -> Dict[str, Any]:
        """Health check"""
        return {
            "status": "healthy",
            "model": "OpenAI GPT-OSS 120B",
            "loaded": self.model_loaded
        }

# ============================================
# CELL 4: TEST FUNCTION
# ============================================
@app.local_entrypoint()
def main():
    """Test the deployment"""
    print("🌌 OPENAI GPT-OSS 120B DEPLOYMENT")
    print("=" * 60)
    
    # First upload model
    print("\n📤 Uploading model...")
    result = upload_model_from_local.remote()
    print(f"Upload result: {result}")
    
    # Get endpoints
    print("\n🔗 Your endpoints:")
    print("   Generate: https://YOUR-MODAL-USERNAME--gaia-quantum-120b-enhanced-quantumgpt120b-generate-endpoint.modal.run")
    print("   Health: https://YOUR-MODAL-USERNAME--gaia-quantum-120b-enhanced-quantumgpt120b-health.modal.run")
    print("\nReplace YOUR-MODAL-USERNAME with your actual Modal username")
    print("\n✅ Add these to Replit Secrets as MODAL_ENDPOINT and MODAL_API_KEY")