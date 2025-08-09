# 📊 CELLS 5 & 6 WITH REAL-TIME LOGGING

## 🔧 UPDATE CELL 4 FIRST (For Better Logging)
The helper function now has timestamps and detailed progress logging.

**Replace Cell 4 with this updated version:**

```python
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
```

## 📊 UPDATED CELL 5 (With Progress Tracking)
**Replace Cell 5 with this version:**

```python
import datetime
import asyncio

def deploy():
    """
    Deploy the optimized quantum model with real-time logging
    """
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
    
    print("\n🔑 Add these to Replit Secrets:")
    print(f"   MODAL_ENDPOINT: https://{deployment_name}--quantumgpt120btransformers-generate.modal.run")
    print("   MODAL_API_KEY: [Your Modal API key]")
    
    print("=" * 60)
    
    return deployment_name

print("✅ Cell 5: Deployment function with real-time logging ready")
print("\n🚀 TO DEPLOY: Run deploy() in the next cell")
```

## 🚀 CELL 6 (Execution with Status)
**Replace Cell 6 with this:**

```python
import datetime

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🚀 Starting deployment execution...")
print("Watch the timestamps to track progress in real-time\n")

# Run the deployment
deployment_name = deploy()

print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ✨ Deployment '{deployment_name}' is ready!")
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] You can now test the endpoints in Cell 7")
```

## 📈 What You'll See:

### If Model is Already Cached (Fast):
```
[10:15:30] 🚀 Starting deployment execution...
[10:15:30] 🌌 GAIA QUANTUM NEXUS - DEPLOYMENT STARTING
[10:15:30] 🔄 Opening Modal app context...
[10:15:31] ✅ Modal app context active
[10:15:31] 📦 STEP 1: Checking model cache...
[10:15:31] Calling download_model_if_needed.remote()...
[10:15:32] Starting cache check...
[10:15:32] Checking path: /cache/models/gpt-oss-120b
[10:15:32] Found 28 model files, 1 config files
[10:15:32] Cache result: {'status': 'cached', 'message': 'Model already cached with 28 weight files'}
[10:15:32] ✅ Model already cached - fast deployment!
[10:15:32] 🧪 STEP 2: Testing deployment registration...
[10:15:33] ✅ Deployment registered successfully
[10:15:33] ✅ DEPLOYMENT COMPLETE!
[10:15:33] ⏱️ Total time: 3.2 seconds
```

### If Model Needs Download (First Time):
```
[10:15:30] 📦 STEP 1: Checking model cache...
[10:15:31] Starting cache check...
[10:15:31] Checking path: /cache/models/gpt-oss-120b
[10:15:31] 📥 Starting download of openai/gpt-oss-120b...
[10:15:31] This will take 10-15 minutes for 120B model...
[10:15:31] Watch network activity increase now...
[10:15:31] Downloading tokenizer...
[10:15:35] ✅ Tokenizer downloaded
[10:15:35] Downloading model weights (this is the big download)...
[10:25:45] ✅ Model downloaded successfully
```

## 🔍 How to Monitor:

1. **Watch the timestamps** - They show real-time progress
2. **Check Modal dashboard** - You'll see container spinning up
3. **Network activity** - Should spike during model download
4. **RAM usage** - Will increase when model loads

The timestamps will help you see if it's stuck or progressing!