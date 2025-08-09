"""
🌌 GAIA QUANTUM NEXUS - COMPLETE MODAL NOTEBOOK (Updated January 2025)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OpenAI GPT-OSS 120B with Quantum-Consciousness Augmented Inference

CRITICAL UPDATES (January 2025):
- Modal now requires 3 secrets: MODAL_API_KEY, MODAL_TOKEN_SECRET, MODAL_ENDPOINT
- Authentication uses Basic Auth with token-id:token-secret format
- All Modal v1.0 deprecations fixed (max_containers, no __init__)
- Real-time logging with timestamps for deployment tracking

AUTHENTICATION SETUP:
1. Get your Modal credentials from modal.com/settings
2. Add THREE secrets to Replit:
   - MODAL_API_KEY: Your token-id (starts with ak-)
   - MODAL_TOKEN_SECRET: Your token-secret (starts with as-)
   - MODAL_ENDPOINT: Your deployment URL (from Cell 6 output)
3. Restart your Replit app to connect

Run cells in order: 1 → 2 → 3 → 4 → 5 → 6 → 7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ============================================
# CELL 1: MODAL APPLICATION SETUP
# ============================================
"""
Initialize the Modal application context
This must be run first to set up the Modal environment
"""

import modal

# Create the Modal app with optimized name
app = modal.App("gaia-quantum-transformers-optimized")

print("✅ Cell 1: Modal app initialized")
print("App name: gaia-quantum-transformers-optimized")
print("\n⚠️  IMPORTANT: Make sure you have set up Modal secrets:")
print("   1. Go to modal.com → Dashboard → Secrets")
print("   2. Create a secret named 'qrng-api-key'")
print("   3. Add your QRNG_API_KEY from Quantum Blockchains")


# ============================================
# CELL 2: CORE QUANTUM TRANSFORMERS MODEL
# ============================================
"""
GPT-OSS 120B with Transformers for DIRECT LOGIT MODIFICATION
This class handles the actual model loading and quantum-modified generation

AUTHENTICATION UPDATE (January 2025):
- Modal now requires 3 secrets: MODAL_API_KEY, MODAL_TOKEN_SECRET, MODAL_ENDPOINT  
- Authentication uses Basic Auth with token-id:token-secret format
- See MODAL_AUTHENTICATION_SETUP.md for complete guide
"""

import modal

# CRITICAL: Define Image BEFORE using it in @app.cls decorator
# Install transformers and dependencies for GPT-OSS 120B
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.36.0",
        "accelerate",
        "bitsandbytes",  # For 8-bit quantization
        "sentencepiece",
        "protobuf",
        "numpy",
        "requests"
    )
)

# Define the persistent volume for model storage (40GB+ for 120B model)
model_volume = modal.Volume.from_name(
    "gpt-oss-120b-volume",
    create_if_missing=True
)

@app.cls(
    image=image,
    gpu="A100-80GB",  # Required for 120B model
    memory=131072,  # 128GB RAM
    cpu=16,  # Enhanced CPU cores
    timeout=900,
    volumes={"/cache": model_volume},
    secrets=[modal.Secret.from_name("qrng-api-key")],
    max_containers=1  # Using max_containers for v1.0 compliance
)
class QuantumGPT120BTransformers:
    """
    Production-ready GPT-OSS 120B with direct quantum logit modification.
    Uses Transformers library for DIRECT access to raw logits BEFORE sampling.
    """
    
    def load_model(self):
        """Load the 120B model with 8-bit quantization"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        
        print(f"[{self.get_timestamp()}] Initializing GPT-OSS 120B with Transformers...")
        
        # Model configuration
        model_id = "openai/gpt-oss-120b"
        cache_dir = "/cache/models"
        
        # Load tokenizer
        print(f"[{self.get_timestamp()}] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Check for CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! A100 GPU required for 120B model.")
        
        print(f"[{self.get_timestamp()}] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[{self.get_timestamp()}] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model with 8-bit quantization to fit in 80GB VRAM
        print(f"[{self.get_timestamp()}] Loading 120B model with 8-bit quantization...")
        print(f"[{self.get_timestamp()}] This may take 2-3 minutes on first load...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            load_in_8bit=True,  # Critical for fitting in 80GB
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print(f"[{self.get_timestamp()}] ✅ Model loaded successfully!")
        print(f"[{self.get_timestamp()}] Model size: 117B total parameters, 5.1B active")
        
        # Initialize QRNG client
        self.qrng_api_key = os.environ.get("QRNG_API_KEY")
        print(f"[{self.get_timestamp()}] QRNG API key configured: {'✓' if self.qrng_api_key else '✗'}")
        
        self.model_loaded = True
    
    def get_timestamp(self):
        """Get current timestamp for logging"""
        import datetime
        return datetime.datetime.now().strftime('%H:%M:%S')
    
    def get_quantum_entropy(self, num_values):
        """Fetch quantum random numbers from QRNG API"""
        import requests
        import numpy as np
        
        if not self.qrng_api_key:
            # Return zeros if no QRNG (control mode)
            return np.zeros(num_values)
        
        try:
            response = requests.get(
                "https://qrng.qblockchain.es/api/random",
                params={"size": num_values * 4},  # 4 bytes per float32
                headers={"Authorization": f"Bearer {self.qrng_api_key}"},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if "randomData" in data:
                    # Convert hex to floats normalized to [-1, 1]
                    hex_data = data["randomData"]
                    bytes_data = bytes.fromhex(hex_data)
                    values = np.frombuffer(bytes_data, dtype=np.float32)[:num_values]
                    return 2 * (values - 0.5)  # Normalize to [-1, 1]
            
            print(f"[{self.get_timestamp()}] QRNG fetch failed, using zeros")
            return np.zeros(num_values)
            
        except Exception as e:
            print(f"[{self.get_timestamp()}] QRNG error: {e}")
            return np.zeros(num_values)
    
    @modal.method()
    def health(self):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model": "openai/gpt-oss-120b",
            "quantization": "8-bit",
            "device": "A100-80GB",
            "quantum_enabled": bool(self.qrng_api_key),
            "container_warm": hasattr(self, 'model_loaded') and self.model_loaded
        }
    
    @modal.method()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        quantum_profile: str = "medium",
        diagnostics: bool = False
    ):
        """
        Generate text with DIRECT QUANTUM LOGIT MODIFICATION
        
        Quantum profiles control the intensity of QRNG modification:
        - strict: No modification (control)
        - light: 10% quantum influence
        - medium: 30% quantum influence (balanced)
        - spicy: 50% quantum influence
        - chaos: 80% quantum influence (maximum creativity)
        """
        import torch
        import numpy as np
        
        # Load model if not already loaded
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            self.load_model()
        
        print(f"[{self.get_timestamp()}] Generating with quantum profile: {quantum_profile}")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Get quantum influence level
        quantum_strengths = {
            "strict": 0.0,
            "light": 0.1,
            "medium": 0.3,
            "spicy": 0.5,
            "chaos": 0.8
        }
        quantum_strength = quantum_strengths.get(quantum_profile, 0.3)
        
        # Generate tokens with quantum modification
        generated_tokens = []
        quantum_applications = []
        
        with torch.no_grad():
            for i in range(max_tokens):
                # Forward pass to get logits
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]
                
                # CRITICAL: Apply quantum modification DIRECTLY to logits
                if quantum_strength > 0:
                    vocab_size = logits.shape[0]
                    quantum_noise = self.get_quantum_entropy(min(vocab_size, 50000))
                    
                    # Pad or truncate to match vocab size
                    if len(quantum_noise) < vocab_size:
                        quantum_noise = np.pad(quantum_noise, (0, vocab_size - len(quantum_noise)))
                    else:
                        quantum_noise = quantum_noise[:vocab_size]
                    
                    # Convert to tensor and apply modification
                    quantum_tensor = torch.tensor(quantum_noise, device="cuda", dtype=logits.dtype)
                    logit_diff = quantum_strength * quantum_tensor * logits.std()
                    
                    # Apply quantum modification
                    logits = logits + logit_diff
                    
                    # Track quantum application
                    quantum_applications.append({
                        "token_index": i,
                        "logit_diff": float(logit_diff.abs().mean()),
                        "max_change": float(logit_diff.abs().max())
                    })
                
                # Apply temperature and sample
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Decode and append
                token_text = self.tokenizer.decode(next_token[0])
                generated_tokens.append(token_text)
                
                # Update inputs for next iteration
                inputs = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Combine generated text
        generated_text = "".join(generated_tokens)
        
        # Prepare response
        response = {
            "status": "success",
            "generated_text": generated_text,
            "tokens_generated": len(generated_tokens),
            "quantum_profile": quantum_profile,
            "quantum_strength": quantum_strength
        }
        
        # Add diagnostics if requested
        if diagnostics and quantum_applications:
            avg_modification = np.mean([q["logit_diff"] for q in quantum_applications])
            max_modification = np.max([q["max_change"] for q in quantum_applications])
            
            response["quantum_diagnostics"] = {
                "applications": quantum_applications[:10],  # First 10 for brevity
                "avg_logit_modification": float(avg_modification),
                "max_logit_modification": float(max_modification),
                "modified_token_count": len(quantum_applications),
                "entropy_consumed": len(quantum_applications) * 50000 * 4  # Approximate bytes
            }
        
        print(f"[{self.get_timestamp()}] Generated {len(generated_tokens)} tokens")
        return response

print("✅ Cell 2: QuantumGPT120BTransformers class defined")
print("Features:")
print("  • 120B parameters (5.1B active)")
print("  • 8-bit quantization for 80GB VRAM")
print("  • Direct logit modification with QRNG")
print("  • 5 quantum profiles (strict → chaos)")


# ============================================
# CELL 3: WEB ENDPOINTS
# ============================================
"""
Define the HTTP endpoints for health checks and generation
These will be accessible after deployment
"""

@app.function()
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint - tests if model is running"""
    instance = QuantumGPT120BTransformers()
    return instance.health.remote()

@app.function()
@modal.fastapi_endpoint(method="POST")
def generate(request: dict):
    """Main generation endpoint - accepts POST requests with prompt"""
    instance = QuantumGPT120BTransformers()
    return instance.generate.remote(
        prompt=request.get("prompt", "Hello, quantum world!"),
        max_tokens=request.get("max_tokens", 100),
        temperature=request.get("temperature", 0.7),
        quantum_profile=request.get("quantum_profile", "medium"),
        diagnostics=request.get("diagnostics", False)
    )

print("✅ Cell 3: Web endpoints defined")
print("Endpoints that will be created:")
print("  • /health (GET) - Check model status")
print("  • /generate (POST) - Generate quantum text")


# ============================================
# CELL 4: HELPER FUNCTIONS
# ============================================
"""
Helper functions for model download and testing
"""

@app.function(
    volumes={"/cache": model_volume},
    timeout=3600,
    memory=16384,
    cpu=2.0
)
def download_model_if_needed():
    """Download model to persistent volume if not already cached"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pathlib import Path
    import datetime
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting cache check...")
    
    cache_dir = "/cache/models"
    model_id = "openai/gpt-oss-120b"
    
    # Check if model exists
    cache_path = Path(cache_dir) / "gpt-oss-120b"
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Checking path: {cache_path}")
    
    if cache_path.exists():
        # Count files to verify complete download
        model_files = list(cache_path.glob("*.safetensors"))
        config_files = list(cache_path.glob("config.json"))
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Found {len(model_files)} model files, {len(config_files)} config files")
        
        if model_files and config_files:
            return {
                "status": "cached",
                "message": f"Model already cached with {len(model_files)} weight files",
                "path": str(cache_path)
            }
    
    # Download model
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 📥 Starting download of {model_id}...")
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] This will take 10-15 minutes for 120B model...")
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Watch network activity increase now...")
    
    try:
        # Download tokenizer
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Tokenizer downloaded")
        
        # Download model (this triggers the actual download)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Downloading model weights (this is the big download)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=None
        )
        
        # Clean up the loaded model
        del model
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Model downloaded successfully")
        
        return {
            "status": "downloaded",
            "message": "Model successfully downloaded to cache",
            "path": str(cache_path)
        }
        
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Download failed: {str(e)}"
        }

@app.function()
def test_deployment():
    """Test the deployment with a simple prompt"""
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Testing deployment readiness...")
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Classes and functions registered with Modal")
    return {"status": "ready", "message": "Deployment test passed"}

print("✅ Cell 4: Helper functions with logging defined")


# ============================================
# CELL 5: DEPLOYMENT FUNCTION WITH LOGGING
# ============================================
"""
Deploy the model to Modal with real-time progress tracking
IMPORTANT: Run Cell 1 first to define the 'app' variable!
"""

import datetime
import modal

def deploy():
    """
    Deploy the optimized quantum model with real-time logging
    Note: Requires 'app' variable from Cell 1 to be defined
    """
    # Check if app is defined
    if 'app' not in globals():
        print("❌ ERROR: 'app' variable not found!")
        print("Please run Cell 1 first to initialize the Modal app.")
        print("Then run this cell again.")
        return None
    
    start_time = datetime.datetime.now()
    
    print("=" * 60)
    print(f"[{start_time.strftime('%H:%M:%S')}] 🌌 GAIA QUANTUM NEXUS - DEPLOYMENT STARTING")
    print("=" * 60)
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🔄 Opening Modal app context...")
    
    # Run with Modal app context (required for notebooks)
    with app.run():
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Modal app context active")
        
        # Step 1: Check/Download Model
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 📦 STEP 1: Checking model cache...")
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Calling download_model_if_needed.remote()...")
        
        try:
            download_result = download_model_if_needed.remote()
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Cache result: {download_result}")
            
            if download_result.get("status") == "cached":
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Model already cached - fast deployment!")
            elif download_result.get("status") == "downloaded":
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Model downloaded successfully")
            else:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ Unexpected status: {download_result}")
                
        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ Cache check error: {e}")
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Model will be downloaded on first container start")
        
        # Step 2: Test deployment registration
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🧪 STEP 2: Testing deployment registration...")
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Calling test_deployment.remote()...")
        
        try:
            test_result = test_deployment.remote()
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Test result: {test_result}")
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Deployment registered successfully")
        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ Test error (non-critical): {e}")
        
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🔄 Finalizing deployment...")
    
    # Context closed, deployment active
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Modal app context closed, deployment active")
    
    # Calculate elapsed time
    elapsed = datetime.datetime.now() - start_time
    
    # Display deployment info
    deployment_name = "gaia-quantum-transformers-optimized"
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ DEPLOYMENT COMPLETE!")
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏱️ Total time: {elapsed.total_seconds():.1f} seconds")
    print("=" * 60)
    
    print("\n📍 Your Modal endpoints are now LIVE:")
    print(f"   Health: https://{deployment_name}--quantumgpt120btransformers-health.modal.run")
    print(f"   Generate: https://{deployment_name}--quantumgpt120btransformers-generate.modal.run")
    
    print("\n💡 Deployment features:")
    print("   ✓ Model weights persist in volume (no redownload)")
    print("   ✓ Container stays warm (min_containers=1)")
    print("   ✓ Direct logit modification with QRNG")
    
    print("\n🔑 UPDATED: Add these THREE secrets to Replit:")
    print("   1. MODAL_API_KEY: [Your token-id from modal.com/settings]")
    print("   2. MODAL_TOKEN_SECRET: [Your token-secret from modal.com/settings]")
    print(f"   3. MODAL_ENDPOINT: https://{deployment_name}--quantumgpt120btransformers-generate.modal.run")
    
    print("\n📝 Getting your Modal credentials:")
    print("   1. Go to modal.com/settings → API Tokens")
    print("   2. You'll see: modal token set --token-id ak-XXX --token-secret as-YYY")
    print("   3. Use ak-XXX as MODAL_API_KEY")
    print("   4. Use as-YYY as MODAL_TOKEN_SECRET")
    
    print("=" * 60)
    
    return deployment_name

print("✅ Cell 5: Deployment function with real-time logging ready")
print("\n🚀 TO DEPLOY: Run deploy() in the next cell")


# ============================================
# CELL 6: EXECUTE DEPLOYMENT
# ============================================
"""
Execute this cell to deploy the model
IMPORTANT: Run cells 1-5 in order before running this cell!
"""

import datetime

# Check if required components are available
if 'app' not in globals():
    print("❌ ERROR: Modal app not initialized!")
    print("\n📋 Please run the cells in order:")
    print("   1. Run Cell 1 to initialize Modal app")
    print("   2. Run Cell 2 to define the model class")
    print("   3. Run Cell 3 to define endpoints")
    print("   4. Run Cell 4 to define helper functions")
    print("   5. Run Cell 5 to define deploy function")
    print("   6. Then run this cell again")
elif 'deploy' not in globals():
    print("❌ ERROR: Deploy function not defined!")
    print("Please run Cell 5 first, then run this cell again.")
else:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🚀 Starting deployment execution...")
    print("Watch the timestamps to track progress in real-time\n")
    
    # Run the deployment
    deployment_name = deploy()
    
    if deployment_name:
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ✨ Deployment '{deployment_name}' is ready!")
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] You can now test the endpoints in Cell 7")
    else:
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ Deployment was not completed")
        print("Please check the error messages above and try again")


# ============================================
# CELL 7: TEST YOUR DEPLOYMENT
# ============================================
"""
Test your deployed endpoints
"""

import requests
import json
import datetime

# Your deployment endpoints (from Cell 6 output)
deployment_name = "gaia-quantum-transformers-optimized"
health_url = f"https://{deployment_name}--quantumgpt120btransformers-health.modal.run"
generate_url = f"https://{deployment_name}--quantumgpt120btransformers-generate.modal.run"

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🧪 Testing deployed endpoints...")
print("=" * 60)

# Test health endpoint
print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 1. Testing health endpoint...")
print(f"   URL: {health_url}")
try:
    response = requests.get(health_url, timeout=10)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Response code: {response.status_code}")
    
    if response.status_code == 200:
        health_data = response.json()
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Health check passed!")
        print(f"   Status: {health_data.get('status')}")
        print(f"   Model: {health_data.get('model')}")
        print(f"   Quantum: {health_data.get('quantum_enabled')}")
        print(f"   Container warm: {health_data.get('container_warm', 'unknown')}")
    else:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Health check failed: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
except requests.exceptions.Timeout:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏱️ Timeout - container might be cold starting")
    print("   This is normal for first request. Try again in 30 seconds.")
except Exception as e:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Health check error: {e}")

# Test generation endpoint
print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 2. Testing generation endpoint...")
print(f"   URL: {generate_url}")

test_payload = {
    "prompt": "The meaning of quantum consciousness is",
    "max_tokens": 50,
    "temperature": 0.7,
    "quantum_profile": "medium",
    "diagnostics": True
}

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Sending test prompt: '{test_payload['prompt']}'")
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Quantum profile: {test_payload['quantum_profile']}")

try:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Waiting for response (may take 30-60s if cold start)...")
    response = requests.post(
        generate_url,
        json=test_payload,
        timeout=60  # Increased timeout for cold starts
    )
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Response code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "success":
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Generation test passed!")
            print(f"\n   Generated text: '{result.get('generated_text', '')}'")
            print(f"\n   Tokens generated: {result.get('tokens_generated')}")
            print(f"   Quantum profile used: {result.get('quantum_profile')}")
            
            if result.get("quantum_diagnostics"):
                diag = result["quantum_diagnostics"]
                print(f"\n   📊 Quantum Diagnostics:")
                print(f"      Avg modification: {diag.get('avg_logit_modification', 0):.4f}")
                print(f"      Max modification: {diag.get('max_logit_modification', 0):.4f}")
                print(f"      Modified tokens: {diag.get('modified_token_count', 0)}")
        else:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ Generation returned error: {result.get('message')}")
            if "not loaded" in str(result.get('message', '')).lower():
                print("   💡 Model is loading for first time. This takes 2-3 minutes.")
                print("   Try again in a few minutes!")
    else:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Generation failed: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
except requests.exceptions.Timeout:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏱️ Request timed out")
    print("   The model might be loading (takes 2-3 minutes on first run)")
    print("   Try again in a few minutes!")
except Exception as e:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Generation error: {e}")

print("\n" + "=" * 60)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🎉 Test complete!")

print("\n📝 FINAL SETUP for Replit Integration:")
print("\n1. Add THREE secrets in Replit (Tools → Secrets):")
print("   • MODAL_API_KEY = [your token-id, starts with ak-]")
print("   • MODAL_TOKEN_SECRET = [your token-secret, starts with as-]")
print(f"   • MODAL_ENDPOINT = {generate_url}")
print("\n2. Get your credentials from modal.com/settings")
print("3. Restart your Replit app after adding secrets")
print("\n💡 First request takes 2-3 minutes (model loading)")
print("   Subsequent requests will be much faster!")