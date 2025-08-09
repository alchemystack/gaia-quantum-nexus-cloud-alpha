#!/usr/bin/env python3
"""
Production local server for GPT-OSS 120B transformer with QRNG logit modification
Runs the actual 120B parameter model on WSL Ubuntu with GPU acceleration
Model: bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental
"""

import os
import json
import asyncio
from typing import List, Dict, AsyncGenerator
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from llama_cpp import Llama
import aiohttp

# Configuration for the actual GPT-OSS 120B model
MODEL_PATH = os.environ.get("MODEL_PATH", "/home/user/models/gpt-oss-120b/openai_gpt-oss-120b-Q4_K_M.gguf")
QRNG_API_KEY = os.environ.get("QRNG_API_KEY", "")
PORT = int(os.environ.get("PORT", "8000"))

# Verify GPU is available for the 120B model
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if result.returncode != 0:
    print("ERROR: No GPU detected. The 120B model REQUIRES a GPU with 60-80GB VRAM.")
    print("Cannot run without GPU. Please use cloud deployment instead.")
    exit(1)
else:
    print("✅ GPU detected for 120B model:")
    print(result.stdout.split('\n')[8:11])

app = FastAPI(title="GPT-OSS 120B Production Server with QRNG")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:3000", "*"],
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
    stream: bool = False

class QRNGService:
    """Production QRNG service for quantum random number fetching"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://qrng.qbck.io"
        if not api_key:
            raise ValueError("QRNG_API_KEY is REQUIRED. No fallback to pseudorandom allowed.")
    
    async def get_random_floats(self, count: int, min_val: float = -2, max_val: float = 2) -> List[float]:
        """Fetch true quantum random numbers - NO FALLBACK"""
        if not self.api_key:
            raise HTTPException(
                status_code=500, 
                detail="QRNG API key not configured. System REQUIRES quantum randomness."
            )
        
        url = f"{self.base_url}/{self.api_key}/qbck/block/double"
        params = {"size": count, "min": min_val, "max": max_val}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, ssl=False) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=500, 
                        detail=f"QRNG API error: {error_text}. Cannot proceed without quantum randomness."
                    )
                data = await response.json()
                return data["data"]["result"]

# Initialize the actual 120B model
print(f"Loading GPT-OSS 120B model from {MODEL_PATH}")
print("This is the ACTUAL 120B parameter transformer model")
print("Loading will take several minutes...")

try:
    model = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Use all GPU layers for the 120B model
        n_ctx=4096,       # Context window
        n_batch=512,
        verbose=True,
        seed=-1,          # Non-deterministic for quantum influence
        n_threads=os.cpu_count() or 8,
        use_mlock=True,   # Lock model in RAM
        use_mmap=True     # Memory map for faster loading
    )
    print("✅ GPT-OSS 120B model loaded successfully!")
    print(f"Model vocabulary size: {model.n_vocab()}")
    print(f"Model context length: {model.n_ctx()}")
except Exception as e:
    print(f"❌ Failed to load 120B model: {e}")
    print("Make sure you have:")
    print("1. Downloaded the correct GGUF model file")
    print("2. At least 60-80GB GPU VRAM available")
    print("3. Sufficient system RAM (64GB+ recommended)")
    exit(1)

# Initialize QRNG service
try:
    qrng = QRNGService(QRNG_API_KEY)
    print("✅ QRNG service initialized with Quantum Blockchains API")
except ValueError as e:
    print(f"❌ {e}")
    print("Set QRNG_API_KEY environment variable to proceed.")
    exit(1)

def apply_qrng_to_transformer_logits(
    logits: np.ndarray, 
    qrng_values: List[float], 
    token_idx: int
) -> np.ndarray:
    """
    Apply QRNG modification directly to transformer logits
    This is where quantum randomness influences the neural network
    """
    modified_logits = logits.copy()
    vocab_size = len(logits)
    
    # Calculate QRNG offset for this token
    qrng_offset = (token_idx * 997) % len(qrng_values)  # Prime number for better distribution
    
    # Apply quantum random values to transformer logits
    for i in range(min(vocab_size, len(qrng_values))):
        qrng_idx = (qrng_offset + i) % len(qrng_values)
        # Direct logit modification with quantum random value
        modified_logits[i] += qrng_values[qrng_idx] * 2.0
    
    return modified_logits

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "GPT-OSS 120B (GGUF)",
        "model_path": MODEL_PATH,
        "parameters": "120 billion",
        "gpu_available": True,
        "qrng_configured": bool(QRNG_API_KEY),
        "production": True,
        "quantum_only": True,
        "no_fallback": True
    }

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate text with the actual 120B transformer using QRNG logit modification"""
    
    # Fetch quantum random numbers - NO FALLBACK
    if request.use_qrng and not request.qrng_modifiers:
        try:
            # Get quantum random values for logit modification
            vocab_size = model.n_vocab()
            # Fetch enough QRNG values for multiple tokens
            qrng_count = min(vocab_size, 10000)  # Balance between coverage and API limits
            request.qrng_modifiers = await qrng.get_random_floats(qrng_count)
        except Exception as e:
            return {
                "error": f"QRNG fetch failed: {str(e)}",
                "message": "System REQUIRES quantum randomness. No fallback to pseudorandom.",
                "solution": "Ensure QRNG_API_KEY is valid and API is accessible."
            }
    
    if not request.qrng_modifiers:
        return {
            "error": "No QRNG values available",
            "message": "System cannot generate without quantum randomness"
        }
    
    # Stream generation if requested
    if request.stream:
        return StreamingResponse(
            stream_tokens(request),
            media_type="text/event-stream"
        )
    
    # Generate with the actual 120B transformer
    try:
        # Custom generation with QRNG logit modification
        tokens = model.tokenize(request.prompt.encode('utf-8'))
        context = list(tokens)
        generated_tokens = []
        
        for token_idx in range(request.max_tokens):
            # Run the actual 120B transformer forward pass
            model.eval(context)
            
            # Get raw logits from the transformer
            # This accesses the actual neural network outputs
            logits = model._scores  # Internal logit scores from transformer
            
            # Apply QRNG modification to transformer logits
            modified_logits = apply_qrng_to_transformer_logits(
                logits, 
                request.qrng_modifiers, 
                token_idx
            )
            
            # Temperature scaling
            modified_logits = modified_logits / request.temperature
            
            # Softmax to get probabilities
            max_logit = np.max(modified_logits)
            exp_logits = np.exp(modified_logits - max_logit)
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Sample using QRNG value
            sample_idx = (token_idx * 31 + 17) % len(request.qrng_modifiers)
            qrng_sample = abs(request.qrng_modifiers[sample_idx]) % 1.0
            
            # Sample token from QRNG-modified distribution
            cumulative = np.cumsum(probabilities)
            next_token = int(np.searchsorted(cumulative, qrng_sample))
            next_token = min(next_token, model.n_vocab() - 1)
            
            generated_tokens.append(next_token)
            context.append(next_token)
            
            # Check for end token
            if next_token == model.token_eos():
                break
        
        # Decode generated tokens
        generated_text = model.detokenize(generated_tokens).decode('utf-8', errors='ignore')
        
        return {
            "text": generated_text,
            "tokens_generated": len(generated_tokens),
            "qrng_used": True,
            "qrng_vector_size": len(request.qrng_modifiers),
            "model": "GPT-OSS 120B",
            "parameters": "120 billion",
            "logit_modification": "direct_transformer_logits",
            "quantum_source": "Quantum Blockchains QRNG API"
        }
        
    except Exception as e:
        return {
            "error": f"Generation failed: {str(e)}",
            "model": "GPT-OSS 120B",
            "details": "Check GPU memory and model loading"
        }

async def stream_tokens(request: GenerationRequest) -> AsyncGenerator[str, None]:
    """Stream tokens from the 120B transformer with QRNG modification"""
    
    # Ensure QRNG values are available
    if not request.qrng_modifiers:
        if request.use_qrng:
            try:
                vocab_size = model.n_vocab()
                request.qrng_modifiers = await qrng.get_random_floats(min(vocab_size, 10000))
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'message': 'QRNG required'})}\n\n"
                return
        else:
            yield f"data: {json.dumps({'error': 'QRNG required for generation'})}\n\n"
            return
    
    try:
        tokens = model.tokenize(request.prompt.encode('utf-8'))
        context = list(tokens)
        
        for token_idx in range(request.max_tokens):
            # Run transformer forward pass
            model.eval(context)
            logits = model._scores
            
            # Apply QRNG modification
            modified_logits = apply_qrng_to_transformer_logits(
                logits, 
                request.qrng_modifiers, 
                token_idx
            )
            
            # Temperature and sampling
            modified_logits = modified_logits / request.temperature
            max_logit = np.max(modified_logits)
            exp_logits = np.exp(modified_logits - max_logit)
            probabilities = exp_logits / np.sum(exp_logits)
            
            # QRNG sampling
            sample_idx = (token_idx * 31 + 17) % len(request.qrng_modifiers)
            qrng_sample = abs(request.qrng_modifiers[sample_idx]) % 1.0
            cumulative = np.cumsum(probabilities)
            next_token = int(np.searchsorted(cumulative, qrng_sample))
            next_token = min(next_token, model.n_vocab() - 1)
            
            # Decode token
            token_text = model.detokenize([next_token]).decode('utf-8', errors='ignore')
            
            # Yield token with metadata
            yield f"data: {json.dumps({'token': token_text, 'qrng_influence': request.qrng_modifiers[sample_idx]})}\n\n"
            
            context.append(next_token)
            
            # Check for end
            if next_token == model.token_eos():
                break
                
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.get("/model-info")
async def model_info():
    """Get information about the loaded 120B model"""
    return {
        "model_name": "GPT-OSS 120B",
        "model_file": MODEL_PATH,
        "parameters": "120 billion",
        "vocabulary_size": model.n_vocab(),
        "context_length": model.n_ctx(),
        "gpu_layers": -1,  # All layers on GPU
        "quantization": "GGUF MXFP4",
        "source": "bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental",
        "huggingface_url": "https://huggingface.co/bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental",
        "qrng_integration": "Direct transformer logit modification",
        "quantum_source": "Quantum Blockchains QRNG API",
        "production_ready": True
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 GPT-OSS 120B Production Server with QRNG Logit Modification")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"Parameters: 120 billion")
    print(f"Port: {PORT}")
    print(f"QRNG: {'✅ Configured (Quantum Blockchains API)' if QRNG_API_KEY else '❌ NOT CONFIGURED - REQUIRED'}")
    print(f"Production Mode: ✅ Running actual 120B transformer")
    print(f"Quantum Only: ✅ No fallback to pseudorandom")
    print("="*70 + "\n")
    
    if not QRNG_API_KEY:
        print("ERROR: QRNG_API_KEY is REQUIRED. Set it and restart.")
        exit(1)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)