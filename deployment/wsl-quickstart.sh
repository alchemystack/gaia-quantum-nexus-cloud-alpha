#!/bin/bash
# Quick Start Script for GPT-OSS 120B on WSL Ubuntu
# Run this script to set up everything automatically

set -e  # Exit on error

echo "================================================"
echo "GPT-OSS 120B with QRNG - WSL Ubuntu Quick Setup"
echo "================================================"
echo ""
echo "Model: bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental"
echo "Source: https://huggingface.co/bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo ""
else
    echo "❌ No GPU detected. The 120B model requires a GPU with 60-80GB VRAM."
    echo "Please ensure NVIDIA drivers for WSL2 are installed."
    exit 1
fi

# Determine appropriate quantization based on GPU memory
if [ "$GPU_MEMORY" -lt 30000 ]; then
    echo "⚠️  Your GPU has ${GPU_MEMORY}MB VRAM"
    echo "The 120B model requires minimum 40GB for smallest quantization (Q2_K)"
    echo "Consider using cloud deployment instead (see deployment/DEPLOYMENT_GUIDE.md)"
    MODEL_FILE="openai_gpt-oss-120b-Q2_K.gguf"
    echo "Attempting Q2_K quantization (40GB) - may fail with OOM"
elif [ "$GPU_MEMORY" -lt 50000 ]; then
    MODEL_FILE="openai_gpt-oss-120b-Q2_K.gguf"
    echo "Selected Q2_K quantization (40GB) for your GPU"
elif [ "$GPU_MEMORY" -lt 70000 ]; then
    MODEL_FILE="openai_gpt-oss-120b-Q3_K_S.gguf"
    echo "Selected Q3_K_S quantization (50GB) for your GPU"
elif [ "$GPU_MEMORY" -lt 85000 ]; then
    MODEL_FILE="openai_gpt-oss-120b-Q4_K_M.gguf"
    echo "Selected Q4_K_M quantization (65GB) for your GPU"
else
    MODEL_FILE="openai_gpt-oss-120b-Q5_K_M.gguf"
    echo "Selected Q5_K_M quantization (80GB) for your GPU"
fi

echo ""
echo "Step 1: Installing system dependencies..."
sudo apt update
sudo apt install -y build-essential cmake git wget curl python3-pip python3.11 python3.11-venv

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-3
    rm cuda-keyring_1.0-1_all.deb
fi

echo ""
echo "Step 2: Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate

echo ""
echo "Step 3: Installing Python packages..."
pip install --upgrade pip
pip install huggingface-hub fastapi uvicorn aiohttp pydantic numpy
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

echo ""
echo "Step 4: Creating models directory..."
mkdir -p ~/models/gpt-oss-120b

echo ""
echo "Step 5: Downloading model (this will take a while)..."
echo "Downloading ${MODEL_FILE} from HuggingFace..."

cd ~/models/gpt-oss-120b

# Download the model file
huggingface-cli download bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental \
    ${MODEL_FILE} \
    --local-dir . \
    --resume-download

echo ""
echo "Step 6: Creating environment configuration..."
cd -  # Return to project directory

cat > .env.local << EOF
# WSL Ubuntu Local Configuration
export QRNG_API_KEY="${QRNG_API_KEY:-your-qrng-api-key-here}"
export MODEL_PATH="$HOME/models/gpt-oss-120b/${MODEL_FILE}"
export PORT=8000
export LLAMA_MMAP=1
export LLAMA_MLOCK=1
export OMP_NUM_THREADS=8
EOF

echo "Created .env.local file"

echo ""
echo "Step 7: Creating run script..."
cat > run-local-server.sh << 'EOF'
#!/bin/bash
# Load environment variables
source .env.local
source venv/bin/activate

echo "Starting GPT-OSS 120B Local Server"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "QRNG: ${QRNG_API_KEY:+Configured}"

python3 deployment/local-server.py
EOF

chmod +x run-local-server.sh

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Set your QRNG API key:"
echo "   export QRNG_API_KEY='your-actual-key'"
echo ""
echo "2. Run the server:"
echo "   ./run-local-server.sh"
echo ""
echo "3. Test the API:"
echo "   curl -X POST http://localhost:8000/generate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"prompt\": \"Test quantum generation\", \"use_qrng\": true}'"
echo ""
echo "Model location: ~/models/gpt-oss-120b/${MODEL_FILE}"
echo "Server script: deployment/local-server.py"
echo ""
echo "⚠️  Important Notes:"
echo "- Model download may take 1-3 hours depending on internet speed"
echo "- Ensure you have at least 100GB free disk space"
echo "- The model requires significant GPU memory (60-80GB)"
echo "- If you encounter OOM errors, use cloud deployment instead"
echo ""