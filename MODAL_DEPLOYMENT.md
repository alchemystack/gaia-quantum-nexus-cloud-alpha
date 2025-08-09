# Modal Deployment Guide for Gaia Quantum Nexus Cloud

## Overview
This guide will help you deploy the Gaia Quantum Nexus Cloud system with the **bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental** model hosted on Modal. The system uses ONLY true quantum randomness from Quantum Blockchains QRNG API for non-deterministic AI generation.

## Prerequisites

### 1. Modal Account Setup
1. Sign up for Modal at https://modal.com
2. Install Modal CLI: `pip install modal`
3. Authenticate: `modal setup`

### 2. QRNG API Key
1. Get your Quantum Blockchains API key from https://quantumblockchains.io
2. This is REQUIRED - the system will NOT work without true quantum randomness

### 3. System Requirements
- Modal: ~$95/month for light usage, ~$1,900/month for 24/7 operation
- Cold start time: 10-30 seconds (model loading)
- Model size: 60-80GB VRAM required

## Step 1: Deploy the Model to Modal

1. Navigate to the deployment directory:
```bash
cd deployment
```

2. Deploy the GPT-OSS 120B model to Modal:
```bash
modal deploy modal-gpt-oss-120b.py
```

3. After deployment, Modal will provide your endpoint URL:
```
✓ Created web endpoint => https://YOUR-ORG--gaia-quantum-gpt-oss-120b-web.modal.run
```

Save this URL - you'll need it for configuration.

## Step 2: Configure Environment Variables

Set the following environment variables in your Replit project:

1. **MODAL_ENDPOINT**: Your Modal endpoint URL from Step 1
   ```
   https://YOUR-ORG--gaia-quantum-gpt-oss-120b-web.modal.run
   ```

2. **MODAL_API_KEY**: Your Modal API key
   - Get from: https://modal.com/settings/tokens
   - Create a new token and copy the secret

3. **QRNG_API_KEY**: Your Quantum Blockchains API key (REQUIRED)
   - Already configured in your project secrets

## Step 3: Verify Configuration

1. Start the application:
```bash
npm run dev
```

2. Check the console output. You should see:
```
[Routes] Using Modal-hosted GPT-OSS 120B model with QRNG logit modification
```

3. Visit the status endpoint:
```
http://localhost:5000/api/qrng-status
```

You should see:
```json
{
  "available": true,
  "provider": "Quantum Blockchains",
  "modelEngine": "Modal GPT-OSS 120B"
}
```

## Step 4: Test the System

1. Open the Quantum Interface at http://localhost:5000
2. Enter a prompt and select a quantum profile
3. Click "Generate" to start quantum-augmented generation
4. Observe the real-time token streaming with QRNG modifications

## Production Deployment on Replit

1. **Configure Secrets in Replit**:
   - Go to the Secrets tab in your Replit project
   - Add `MODAL_ENDPOINT` with your Modal URL
   - Add `MODAL_API_KEY` with your Modal API token
   - Ensure `QRNG_API_KEY` is already set

2. **Deploy to Replit**:
   - Click the "Deploy" button in Replit
   - Your app will be available at: `https://YOUR-PROJECT.replit.app`

## Cost Optimization

### Light Usage (~$95/month)
- Cold starts: 10-30 seconds
- Suitable for demos and experiments
- Auto-scales down when not in use

### Production Usage (~$1,900/month)
- Keep model warm with periodic health checks
- Set up Modal scheduled functions to prevent cold starts
- Consider batching requests for efficiency

### Alternative: Dedicated GPU (~$24,000/month)
- For 24/7 high-traffic production
- Zero cold start time
- Dedicated A100 80GB instance

## Monitoring & Debugging

### Check Modal Logs
```bash
modal logs -f gaia-quantum-gpt-oss-120b
```

### Monitor QRNG Usage
- Check entropy pool status at `/api/qrng-status`
- Monitor `entropyUsed` in performance metrics
- Set up alerts for QRNG API failures

### Performance Metrics
The system provides real-time metrics:
- **Latency**: Time to generate each token
- **Tokens/sec**: Generation throughput
- **Entropy Used**: Quantum bits consumed
- **Layer Analysis**: Attention and FFN modifications

## Troubleshooting

### "Modal not configured" Message
- Verify `MODAL_ENDPOINT` and `MODAL_API_KEY` are set
- Check Modal deployment status: `modal list`

### "QRNG unavailable" Error
- System REQUIRES true quantum randomness
- Verify `QRNG_API_KEY` is correct
- Check Quantum Blockchains API status

### Slow Generation
- Normal cold start: 10-30 seconds
- Subsequent requests should be faster
- Consider upgrading Modal tier for better performance

## Security Considerations

1. **Never expose API keys in code**
2. **Use environment variables for all secrets**
3. **Enable Modal authentication for production**
4. **Monitor QRNG usage to prevent abuse**
5. **Set up rate limiting for public deployments**

## Support

- Modal Documentation: https://modal.com/docs
- Quantum Blockchains Support: https://quantumblockchains.io/support
- Replit Community: https://ask.replit.com

## Next Steps

1. Test the system thoroughly with various prompts
2. Monitor performance metrics and costs
3. Fine-tune quantum influence profiles
4. Consider implementing user authentication
5. Set up automated monitoring and alerts

---

**Remember**: This system uses ONLY true quantum randomness. Generation will halt if QRNG becomes unavailable. This is by design to maintain quantum authenticity.