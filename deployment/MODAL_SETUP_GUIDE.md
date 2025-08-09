# Complete Modal Setup Guide for Gaia Quantum Nexus

## Step 1: Create a Modal Account

1. **Sign up for Modal:**
   - Go to https://modal.com
   - Click "Sign up" and create your account
   - Verify your email address

2. **Get your Modal token:**
   - After signing in, go to https://modal.com/settings/tokens
   - Click "Create token"
   - Name it "gaia-quantum-nexus"
   - Copy the token value - you'll need this as `MODAL_API_KEY`

## Step 2: Install Modal CLI Locally

You'll need to run the deployment from your local machine (not from Replit).

### On Windows:
```powershell
# Install Python if you don't have it
# Download from https://www.python.org/downloads/

# Install Modal
pip install modal

# Authenticate Modal
modal setup
```

### On Mac/Linux:
```bash
# Install Modal
pip install modal

# Authenticate Modal
modal setup
```

When you run `modal setup`, it will:
1. Ask for your Modal token (paste the one you copied)
2. Create a config file at `~/.modal/config.toml`
3. Verify the connection

## Step 3: Prepare for Deployment

1. **Clone or download the deployment files:**
   
   Create a new folder on your computer and save these files:
   - `modal_deployment_notebook.py` (already created in your project)
   - Create a `.env` file with your QRNG API key:
   
   ```env
   QRNG_API_KEY=your_quantum_blockchains_api_key_here
   ```

2. **Set up Modal secret for QRNG:**
   
   Run this command to create a Modal secret:
   ```bash
   modal secret create qrng-api-key QRNG_API_KEY=your_actual_qrng_key_here
   ```

## Step 4: Deploy the Model

1. **Navigate to your deployment folder:**
   ```bash
   cd path/to/your/deployment/folder
   ```

2. **Run the deployment:**
   ```bash
   python modal_deployment_notebook.py
   ```

   This will:
   - Build the container image
   - Download the 120B model (first run only, ~60GB)
   - Deploy to Modal's cloud
   - Provide you with endpoint URLs

3. **Expected output:**
   ```
   🚀 Deploying Gaia Quantum GPT-OSS 120B to Modal...
   ============================================================
   
   📦 Building container image...
   🔄 Deploying to Modal cloud...
   
   ✅ Deployment complete!
   
   📌 Your endpoints:
     - Generation: https://YOUR-USERNAME--gaia-quantum-gpt-oss-120b-generate-endpoint.modal.run
     - Health: https://YOUR-USERNAME--gaia-quantum-gpt-oss-120b-health-check.modal.run
   ```

## Step 5: Test the Deployment

1. **Test the health endpoint:**
   
   Open your browser and visit:
   ```
   https://YOUR-USERNAME--gaia-quantum-gpt-oss-120b-health-check.modal.run
   ```
   
   You should see:
   ```json
   {
     "status": "healthy",
     "model": "GPT-OSS 120B",
     "quantum": "enabled",
     "gpu": "2x A100 80GB",
     "timestamp": 1754719123.456
   }
   ```

2. **Test generation with curl:**
   
   ```bash
   curl -X POST https://YOUR-USERNAME--gaia-quantum-gpt-oss-120b-generate-endpoint.modal.run \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The quantum nature of consciousness",
       "max_tokens": 50,
       "temperature": 0.7,
       "profile": "medium"
     }'
   ```

## Step 6: Connect to Replit

1. **Go to your Replit project**

2. **Open the Secrets tab** (lock icon in the left sidebar)

3. **Add these secrets:**
   
   - **MODAL_ENDPOINT**: 
     ```
     https://YOUR-USERNAME--gaia-quantum-gpt-oss-120b-generate-endpoint.modal.run
     ```
   
   - **MODAL_API_KEY**: 
     ```
     Your Modal token from Step 1
     ```

4. **Restart your Replit app**
   
   The app will automatically detect Modal is configured and use the 120B model!

## Step 7: Verify Integration

1. **Check the console output:**
   
   You should see:
   ```
   [Routes] Using Modal-hosted GPT-OSS 120B model with QRNG logit modification
   ```

2. **Check the UI:**
   
   In the header, you should see:
   - "Model: Modal GPT-OSS 120B" (purple indicator)
   - "QRNG: Active" (green indicator)

3. **Test generation:**
   - Enter a prompt in the UI
   - Select a quantum profile
   - Click Generate
   - Watch the quantum-augmented 120B model generate text!

## Troubleshooting

### "Modal not found" error
- Make sure you installed Modal: `pip install modal`
- Run `modal --version` to verify installation

### "Authentication failed"
- Run `modal setup` again
- Make sure you're using the correct token
- Check that the token hasn't expired

### "QRNG_API_KEY not found"
- Create the Modal secret: `modal secret create qrng-api-key QRNG_API_KEY=your_key`
- Verify with: `modal secret list`

### "Model download failed"
- The 120B model is ~60GB, ensure you have:
  - Stable internet connection
  - Enough disk space
  - Patience (first download takes 30-60 minutes)

### "Cold start taking too long"
- First request after deployment takes 10-30 seconds
- Subsequent requests are faster
- Consider upgrading to keep-warm instances for production

## Cost Breakdown

### Development/Testing (~$95/month)
- Cold starts: 10-30 seconds
- Pay per second of GPU time
- Auto-scales to zero when idle
- Perfect for demos and experiments

### Production (~$1,900/month)
- Keep 1 instance always warm
- Near-instant response times
- Handle multiple concurrent requests
- Suitable for production applications

### High-Traffic (~$24,000/month)
- Multiple warm instances
- Load balancing
- Zero cold starts
- Enterprise-grade performance

## Next Steps

1. **Monitor usage:**
   - Check Modal dashboard: https://modal.com/dashboard
   - View logs: `modal logs -f gaia-quantum-gpt-oss-120b`
   - Track costs in billing section

2. **Optimize performance:**
   - Adjust `keep_warm` parameter in deployment
   - Fine-tune batch sizes
   - Implement request caching

3. **Scale as needed:**
   - Start with development tier
   - Monitor actual usage patterns
   - Upgrade when traffic justifies cost

## Support Resources

- **Modal Documentation:** https://modal.com/docs
- **Modal Discord:** https://discord.gg/modal
- **Quantum Blockchains:** https://quantumblockchains.io/support
- **Replit Community:** https://ask.replit.com

---

**Remember:** The system requires REAL quantum randomness. Without QRNG_API_KEY, generation will fail. This is intentional to maintain quantum authenticity!