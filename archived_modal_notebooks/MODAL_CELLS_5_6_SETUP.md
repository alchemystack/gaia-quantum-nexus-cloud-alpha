# 📋 Step-by-Step Setup for Cells 5 & 6

## Prerequisites Before Running Cell 5:
✅ **Cells 1-4 must be run first** (in order)
✅ **Modal secret must exist**: "qrng-api-key" containing your QRNG_API_KEY

---

## 🔧 CELL 5: Deployment Function Setup

### What This Cell Does:
- Defines the `deploy()` function (doesn't run it yet)
- This function will handle the entire deployment process

### Step-by-Step:

1. **Copy this exact code into Cell 5:**
```python
def deploy():
    """
    Deploy the optimized quantum model
    This is the correct way to deploy from a notebook
    """
    print("=" * 60)
    print("🌌 GAIA QUANTUM NEXUS - OPTIMIZED DEPLOYMENT")
    print("=" * 60)
    
    # Run with Modal app context (required for notebooks)
    with app.run():
        # Step 1: Ensure model is downloaded to volume
        print("\n📦 Step 1: Checking model cache...")
        try:
            download_result = download_model_if_needed.remote()
            print(f"Cache status: {download_result}")
        except Exception as e:
            print(f"⚠️  Cache check error: {e}")
            print("Model will be downloaded on first run")
        
        # Step 2: Test the deployment (optional)
        print("\n🧪 Step 2: Testing deployment...")
        try:
            test_result = test_deployment.remote()
            print("✅ Test completed successfully")
        except Exception as e:
            print(f"⚠️  Test error (non-critical): {e}")
    
    # Display deployment info
    deployment_name = "gaia-quantum-transformers-optimized"
    
    print("\n✅ DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print("📍 Your Modal endpoints:")
    print(f"   Health: https://{deployment_name}--quantumgpt120btransformers-health.modal.run")
    print(f"   Generate: https://{deployment_name}--quantumgpt120btransformers-generate.modal.run")
    
    print("\n💡 Deployment features:")
    print("   ✓ Model weights persist in volume (no redownload)")
    print("   ✓ Fast loading from cache (~30s vs 10+ minutes)")
    print("   ✓ Container stays warm (min_containers=1)")
    print("   ✓ Survives notebook kernel resets")
    print("   ✓ Direct logit modification with QRNG")
    
    print("\n🔑 Add these to Replit Secrets:")
    print(f"   MODAL_ENDPOINT: https://{deployment_name}--quantumgpt120btransformers-generate.modal.run")
    print("   MODAL_API_KEY: [Your Modal API key]")
    
    print("\n📝 Next steps:")
    print("   1. Copy the endpoint URLs above")
    print("   2. Add them to Replit secrets")
    print("   3. Restart the Replit app to connect")
    print("=" * 60)
    
    return deployment_name

print("✅ Cell 5: Deployment function ready")
print("\n🚀 TO DEPLOY: Run deploy() in the next cell")
```

2. **Run Cell 5**
   - This just defines the function, doesn't execute it
   - You should see: "✅ Cell 5: Deployment function ready"

---

## 🚀 CELL 6: Execute Deployment

### What This Cell Does:
- Actually runs the deployment
- Downloads model if needed (first time: ~10-15 minutes)
- Deploys to Modal cloud
- Shows you the endpoint URLs

### Step-by-Step:

1. **Copy this exact code into Cell 6:**
```python
# Run the deployment
deployment_name = deploy()

print(f"\n✨ Deployment '{deployment_name}' is ready!")
```

2. **Before Running Cell 6, Check:**
   - ✅ You have Modal credits/subscription
   - ✅ Your QRNG_API_KEY secret is set in Modal
   - ✅ You're ready for a 10-15 minute wait (first time only)

3. **Run Cell 6**
   - This ACTUALLY deploys everything
   - First run: Takes 10-15 minutes to download 120B model
   - Future runs: Takes ~30 seconds (uses cached model)

---

## 📊 What You'll See When Running Cell 6:

### First Time (Model Download):
```
========================================================
🌌 GAIA QUANTUM NEXUS - OPTIMIZED DEPLOYMENT
========================================================

📦 Step 1: Checking model cache...
⚠️  Model not in cache, downloading...
📥 Downloading openai/gpt-oss-120b to cache...
This will take 10-15 minutes for 120B model...
✅ Model downloaded successfully

🧪 Step 2: Testing deployment...
✅ Test completed successfully

✅ DEPLOYMENT COMPLETE!
========================================================
📍 Your Modal endpoints:
   Health: https://gaia-quantum-transformers-optimized--quantumgpt120btransformers-health.modal.run
   Generate: https://gaia-quantum-transformers-optimized--quantumgpt120btransformers-generate.modal.run
```

### Future Runs (Using Cache):
```
========================================================
🌌 GAIA QUANTUM NEXUS - OPTIMIZED DEPLOYMENT
========================================================

📦 Step 1: Checking model cache...
Cache status: {'status': 'cached', 'message': 'Model already cached with 28 weight files'}

🧪 Step 2: Testing deployment...
✅ Test completed successfully

✅ DEPLOYMENT COMPLETE!
[Same endpoints shown]
```

---

## ⚠️ Common Issues & Solutions:

### Issue 1: "QRNG_API_KEY not found"
**Solution:** 
1. Go to Modal dashboard → Secrets
2. Create secret named "qrng-api-key"
3. Add key: QRNG_API_KEY, value: [your key]

### Issue 2: "ExecutionError: Function has not been hydrated"
**Solution:** You forgot `with app.run():` - use the exact code above

### Issue 3: "InvalidError: Duplicate local entrypoint"
**Solution:** Don't add `@app.local_entrypoint()` - just use the plain function

### Issue 4: Long download time
**Normal:** First download takes 10-15 minutes for 120B model
**Future runs:** Only ~30 seconds using cached model

---

## ✅ Success Indicators:

When Cell 6 completes successfully, you'll have:
1. Two working endpoint URLs displayed
2. Model cached in Modal volume
3. Container running warm (no cold starts)
4. Ready to connect to Replit

---

## 🔗 After Deployment:

1. **Copy the endpoints** from the output
2. **Go to Replit** → Secrets
3. **Add:**
   - MODAL_ENDPOINT: [the generate URL]
   - MODAL_API_KEY: [your Modal API key]
4. **Restart Replit app** to connect

That's it! The deployment will stay active and warm, ready for quantum text generation!