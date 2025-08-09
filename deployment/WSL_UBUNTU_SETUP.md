# Running GPT-OSS 120B with QRNG on WSL Ubuntu

## Prerequisites

### 1. WSL2 with Ubuntu Installation
```powershell
# In Windows PowerShell (as Administrator)
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2

# Verify WSL2 is running
wsl -l -v
```

### 2. GPU Support in WSL2 (Required for 120B Model)
```powershell
# Install NVIDIA GPU drivers for WSL2
# Download from: https://developer.nvidia.com/cuda/wsl

# Verify GPU is available in WSL
wsl
nvidia-smi
```

**IMPORTANT GPU REQUIREMENTS:**
- Minimum: NVIDIA RTX 4090 (24GB VRAM) - Will run Q2_K quantization only
- Recommended: NVIDIA A100 80GB or H100 80GB
- The 120B model requires 60-80GB VRAM for Q4_K_M quantization

## Ubuntu Setup Instructions

### Step 1: Update Ubuntu and Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential cmake git wget curl python3-pip

# Install CUDA toolkit for Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# Verify CUDA installation
nvcc --version
```

### Step 2: Install Python and Node.js
```bash
# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installations
python3.11 --version
node --version
npm --version
```

### Step 3: Clone and Setup the Project
```bash
# Create workspace
mkdir -p ~/quantum-nexus
cd ~/quantum-nexus

# Clone the repository (or copy your files)
git clone https://github.com/your-repo/gaia-quantum-nexus.git
cd gaia-quantum-nexus

# Install project dependencies
npm install
```

### Step 4: Install llama.cpp for GGUF Model Support
```bash
# Clone llama.cpp
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CUDA support
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON
make -j$(nproc)

# Install Python bindings with CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Step 5: Download the GPT-OSS 120B Model
```bash
# Install huggingface-cli
pip install huggingface-hub

# Create models directory
mkdir -p ~/models/gpt-oss-120b

# Download the specific GGUF model (choose based on your GPU memory)
cd ~/models/gpt-oss-120b

# For RTX 4090 (24GB) - Use smallest quantization
huggingface-cli download bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental \
  openai_gpt-oss-120b-Q2_K.gguf \
  --local-dir .

# For A100 80GB - Use Q4_K_M for better quality
huggingface-cli download bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental \
  openai_gpt-oss-120b-Q4_K_M.gguf \
  --local-dir .

# List available files to choose from
huggingface-cli download bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental \
  --list-files
```

Available quantizations:
- `Q2_K.gguf` (~40GB) - Lowest quality, fits on RTX 4090
- `Q3_K_S.gguf` (~50GB) - Better quality
- `Q4_K_M.gguf` (~65GB) - Recommended for A100 80GB
- `Q5_K_M.gguf` (~80GB) - High quality
- `Q6_K.gguf` (~95GB) - Very high quality
- `Q8_0.gguf` (~120GB) - Near full precision

## Running the Local Server

### Step 1: Create Local Server Script
```bash
cd ~/quantum-nexus/gaia-quantum-nexus
```

Create `deployment/local-server.py`:
```python
#!/usr/bin/env python3
"""
Local server for GPT-OSS 120B with QRNG integration
Runs on WSL Ubuntu with GPU acceleration
"""

import os
import json
import asyncio
from typing import List, Dict
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from llama_cpp import Llama
import aiohttp

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/user/models/gpt-oss-120b/openai_gpt-oss-120b-Q4_K_M.gguf")
QRNG_API_KEY = os.environ.get("QRNG_API_KEY", "")
PORT = int(os.environ.get("PORT", "8000"))

# Verify GPU is available
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if result.returncode != 0:
    print("WARNING: No GPU detected. Model loading will be extremely slow or fail.")
else:
    print("GPU detected:")
    print(result.stdout.split('\n')[8:11])  # Show GPU info

app = FastAPI(title="GPT-OSS 120B Local Server")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    qrng_modifiers: List[float] = []
    max_tokens: int = 100
    temperature: float = 0.7
    use_qrng: bool = True

class QRNGService:
    """Fetch quantum random numbers from API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://qrng.qbck.io"
    
    async def get_random_floats(self, count: int, min_val: float = -2, max_val: float = 2) -> List[float]:
        if not self.api_key:
            raise ValueError("QRNG API key not configured")
        
        url = f"{self.base_url}/{self.api_key}/qbck/block/double"
        params = {"size": count, "min": min_val, "max": max_val}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, ssl=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="QRNG API error")
                data = await response.json()
                return data["data"]["result"]

# Initialize model and QRNG
print(f"Loading model from {MODEL_PATH}")
print("This may take several minutes...")

try:
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=4096,       # Context window
        n_batch=512,
        verbose=True,
        seed=-1,
        n_threads=os.cpu_count() or 4
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("Make sure you have enough GPU memory and the model file exists.")
    exit(1)

qrng = QRNGService(QRNG_API_KEY)

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "GPT-OSS 120B (GGUF)",
        "model_path": MODEL_PATH,
        "gpu_available": "nvidia-smi" in os.popen("which nvidia-smi").read(),
        "qrng_configured": bool(QRNG_API_KEY)
    }

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate text with optional QRNG modification"""
    
    # Get QRNG modifiers if requested
    if request.use_qrng and not request.qrng_modifiers:
        try:
            # Fetch fresh QRNG data
            vocab_size = 1000  # Use subset for efficiency
            request.qrng_modifiers = await qrng.get_random_floats(vocab_size)
        except Exception as e:
            return {"error": f"QRNG fetch failed: {str(e)}. Set QRNG_API_KEY environment variable."}
    
    # Apply QRNG to generation
    if request.qrng_modifiers:
        # Custom logit processor using QRNG
        def qrng_logit_processor(input_ids, scores):
            # Apply QRNG modifiers to logits
            for i in range(min(len(scores), len(request.qrng_modifiers))):
                scores[i] += request.qrng_modifiers[i]
            return scores
        
        # Generate with custom logits
        output = model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            logits_processor=qrng_logit_processor if request.qrng_modifiers else None,
            echo=False
        )
    else:
        # Standard generation
        output = model(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            echo=False
        )
    
    return {
        "text": output["choices"][0]["text"],
        "tokens_generated": output["usage"]["completion_tokens"],
        "qrng_used": bool(request.qrng_modifiers),
        "qrng_vector_sample": request.qrng_modifiers[:5] if request.qrng_modifiers else None
    }

@app.post("/stream")
async def stream_generation(request: GenerationRequest):
    """Stream tokens with QRNG modification"""
    
    # Fetch QRNG if needed
    if request.use_qrng and not request.qrng_modifiers:
        try:
            request.qrng_modifiers = await qrng.get_random_floats(1000)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
    
    # Create generator
    stream = model(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=True,
        echo=False
    )
    
    token_count = 0
    for output in stream:
        token = output["choices"][0]["text"]
        
        # Apply QRNG influence to streamed token
        if request.qrng_modifiers and token_count < len(request.qrng_modifiers):
            influence = request.qrng_modifiers[token_count]
        else:
            influence = 0
        
        yield f"data: {json.dumps({'token': token, 'qrng_influence': influence})}\n\n"
        token_count += 1

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 GPT-OSS 120B Local Server Starting")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Port: {PORT}")
    print(f"QRNG: {'✅ Configured' if QRNG_API_KEY else '❌ Not configured (set QRNG_API_KEY)'}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)
```

### Step 2: Install Python Dependencies
```bash
cd ~/quantum-nexus/gaia-quantum-nexus
pip install fastapi uvicorn aiohttp pydantic numpy
```

### Step 3: Set Environment Variables
```bash
# Add to ~/.bashrc or create .env file
export QRNG_API_KEY="your-qrng-api-key-here"
export MODEL_PATH="$HOME/models/gpt-oss-120b/openai_gpt-oss-120b-Q4_K_M.gguf"
export PORT=8000

# Reload environment
source ~/.bashrc
```

### Step 4: Run the Server
```bash
# Make script executable
chmod +x deployment/local-server.py

# Run the server
python3 deployment/local-server.py
```

The server will be available at `http://localhost:8000`

## Integrating with Your Frontend

### Update Your Frontend Configuration
```javascript
// In your frontend code, update the API endpoint
const API_ENDPOINT = process.env.NODE_ENV === 'production' 
  ? 'https://your-cloud-endpoint.modal.run'
  : 'http://localhost:8000';  // Local WSL server
```

### Test the Integration
```bash
# Test the local server
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum consciousness",
    "use_qrng": true,
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Performance Optimization

### 1. Memory Mapping (Faster Loading)
```bash
# Use mmap for faster model loading
export LLAMA_MMAP=1
export LLAMA_MLOCK=1  # Keep model in RAM
```

### 2. CPU Optimization
```bash
# Set thread count
export OMP_NUM_THREADS=8  # Adjust based on your CPU
```

### 3. GPU Memory Management
```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# If OOM errors occur, reduce batch size or use smaller quantization
```

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Use smaller quantization
- RTX 4090 (24GB): Use Q2_K
- RTX 3090 (24GB): Use Q2_K
- A100 40GB: Use Q3_K_S
- A100 80GB: Use Q4_K_M or Q5_K_M

### Issue: Slow Loading
**Solution**: 
1. Ensure model is on SSD, not HDD
2. Use memory mapping: `export LLAMA_MMAP=1`
3. Pre-load model in RAM if you have 128GB+ system memory

### Issue: CUDA Not Found
**Solution**:
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Reinstall CUDA toolkit if needed
sudo apt install --reinstall cuda-toolkit-12-3
```

### Issue: WSL2 GPU Not Detected
**Solution**:
1. Update Windows to latest version
2. Install latest NVIDIA GPU drivers for WSL
3. Ensure WSL2 (not WSL1): `wsl --set-version Ubuntu-22.04 2`

## Running as a Service

### Create systemd service (optional)
```bash
sudo nano /etc/systemd/system/gpt-oss-120b.service
```

Add:
```ini
[Unit]
Description=GPT-OSS 120B Local Server
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/quantum-nexus/gaia-quantum-nexus
Environment="QRNG_API_KEY=your-key"
Environment="MODEL_PATH=/home/your-username/models/gpt-oss-120b/openai_gpt-oss-120b-Q4_K_M.gguf"
ExecStart=/usr/bin/python3 deployment/local-server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable gpt-oss-120b
sudo systemctl start gpt-oss-120b
sudo systemctl status gpt-oss-120b
```

## Next Steps

1. **Verify GPU**: Run `nvidia-smi` in WSL to confirm GPU access
2. **Download Model**: Choose appropriate quantization for your GPU
3. **Set QRNG API Key**: Get from https://qrng.qbck.io
4. **Run Server**: Start the local server
5. **Test API**: Verify generation works with curl
6. **Integrate**: Update frontend to use local endpoint

## Important Notes

- **Model Size**: The 120B model is HUGE. Download will take hours on typical internet
- **GPU Memory**: You NEED a high-end GPU. Consumer GPUs may not have enough VRAM
- **System RAM**: Recommend 64GB+ system RAM for smooth operation
- **Storage**: Need 100GB+ free space for model files
- **Power**: Ensure adequate PSU (850W+ for high-end GPUs)

For cloud deployment instead (if local GPU insufficient), see `deployment/DEPLOYMENT_GUIDE.md`