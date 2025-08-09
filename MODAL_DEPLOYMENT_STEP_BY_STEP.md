# 📘 Complete Modal Deployment Guide - Step by Step

## Prerequisites Checklist

✅ Modal account (free tier works)  
✅ Modal API key from dashboard  
✅ QRNG API key from Quantum Blockchains  
✅ 120B model files (or let Modal download them)  

## Step 1: Set Up Modal

### 1.1 Get Your Modal API Key
1. Go to [modal.com](https://modal.com)
2. Sign in/Sign up
3. Go to Settings → API Keys
4. Create new key and copy it

### 1.2 Install Modal CLI (Optional but Recommended)
```bash
pip install modal
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

## Step 2: Prepare the Notebook

### 2.1 Open Modal Playground
1. Go to [modal.com/playground](https://modal.com/playground)
2. Create a new notebook
3. Name it: "gaia-quantum-120b"

### 2.2 Copy the Fixed Code
Copy the entire contents of `MODAL_NOTEBOOK_FIXED.py` into the notebook.

The notebook has 3 cells:
- **Cell 1**: Model upload/preparation
- **Cell 2**: Server initialization
- **Cell 3**: Get endpoints

## Step 3: Add Secrets to Modal

### 3.1 Add QRNG API Key
1. Go to Modal Dashboard → Secrets
2. Click "Create Secret"
3. Name: `qrng-api-key`
4. Type: Custom
5. Add key-value pair:
   - Key: `QRNG_API_KEY`
   - Value: Your actual QRNG key

## Step 4: Deploy the Model

### 4.1 Run Cell 1 - Model Upload
```python
# This cell uploads or downloads the model
upload_model_from_local()
```

**Expected Output:**
```
🚀 CELL 1: MODEL PREPARATION
============================================================
📥 Downloading GPT-OSS 120B GGUF from HuggingFace...
   Repository: ggml-org/gpt-oss-120b-GGUF
   This will take 10-15 minutes...
✅ Model ready: /models/gpt-oss-120b/gpt-oss-120b.gguf
   Size: 68.50 GB
```

### 4.2 Run Cell 2 - Start Server
```python
# This creates the server class
# Just run it, no output expected
```

### 4.3 Run Cell 3 - Deploy and Get Endpoints
```python
# This deploys everything
main()
```

**Expected Output:**
```
🌌 GAIA QUANTUM NEXUS 120B - DEPLOYMENT
============================================================
✅ DEPLOYMENT COMPLETE!
============================================================
📍 Your API endpoints:
   Health: https://gaia-quantum-120b-enhanced--gaiaquantumserver-health.modal.run
   Generate: https://gaia-quantum-120b-enhanced--gaiaquantumserver-generate.modal.run
```

## Step 5: Test Your Deployment

### 5.1 Test Health Endpoint
```bash
curl https://YOUR-APP--gaiaquantumserver-health.modal.run
```

Expected response:
```json
{
  "status": "healthy",
  "server_running": true,
  "model": "GPT-OSS 120B",
  "config": "Enhanced A100 80GB"
}
```

### 5.2 Test Generation
```bash
curl -X POST https://YOUR-APP--gaiaquantumserver-generate.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The quantum nature of consciousness",
    "max_tokens": 50,
    "quantum_profile": "medium"
  }'
```

## Step 6: Connect to Replit

### 6.1 Add Secrets in Replit
Go to Replit Secrets and add:
- `MODAL_ENDPOINT`: Your generate endpoint URL
- `MODAL_API_KEY`: Your Modal API key (from step 1.1)

### 6.2 Verify Connection
The Replit app should now connect to your Modal deployment!

## Troubleshooting

### Issue: "GPU not available"
**Solution:** Make sure you have GPU access in Modal (may need to upgrade account)

### Issue: "Model not found"
**Solution:** Run Cell 1 again to download the model

### Issue: "QRNG_API_KEY not found"
**Solution:** Add the secret in Modal dashboard (Step 3)

### Issue: "Server timeout"
**Solution:** The model takes 30-60s to load initially. Wait and retry.

## Cost Optimization Tips

1. **Use keep_warm=1** - Keeps container running (included in fixed code)
2. **Use persistent volume** - Avoids re-downloading model
3. **Monitor usage** - Check Modal dashboard for GPU hours
4. **Stop when not needed** - Use Modal dashboard to stop deployment

## Alternative: Using the Optimized Version

If you want even better performance with cached models:
1. Use `MODAL_TRANSFORMERS_OPTIMIZED.py` instead
2. This version caches models persistently
3. Loads much faster after first deployment

## Success Checklist

✅ Health endpoint returns "healthy"  
✅ Generate endpoint produces text  
✅ Replit connects successfully  
✅ QRNG is active (check Replit UI)  

## Next Steps

1. Test different quantum profiles (strict, light, medium, spicy, chaos)
2. Adjust temperature for creativity
3. Monitor performance in Modal dashboard
4. Scale up concurrent requests if needed

---

**Need help?** The endpoints will be shown in Cell 3 output. Copy them to Replit secrets and you're ready to go!