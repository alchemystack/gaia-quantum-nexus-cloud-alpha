# Complete Modal Deployment Guide for Quantum Model (qgpt)

## Overview
This guide provides the complete, unified process for deploying the Gaia Quantum Nexus Cloud system with OpenAI OSS 120B model on Modal, with full integration to the Replit frontend.

## Architecture
- **Modal**: Handles ALL LLM inference, transformers, and QRNG logit modification
- **Replit**: Provides web interface, frontend UI, and user interaction
- **App Name**: `qgpt` (short name for shorter URLs)

## Endpoint URLs (After Deployment)
Your Modal endpoints will be:
- **Health**: `https://qgpt--health.modal.run`
- **Generate**: `https://qgpt--generate.modal.run`

## Step-by-Step Deployment Process

### Step 1: Configure Modal Secrets
Before deploying, create two secrets in Modal:

1. **Secret Name**: `modal-auth`
   - `MODAL_API_KEY`: Will be generated (e.g., `ak-XXXXXXXXXXXXX`)
   - `MODAL_TOKEN_SECRET`: Will be generated (e.g., `as-XXXXXXXXXXXXXXXX`)

2. **Secret Name**: `qrng-api-key`
   - `QRNG_API_KEY`: Your Quantum Blockchains API key

### Step 2: Deploy to Modal
1. Open Modal notebook
2. Copy each cell from `MODAL_NOTEBOOK_UPDATED_2025.py`
3. Run cells 1-5 (defines the functions)
4. Run Cell 4 (uncomment `setup_modal_secrets()`) to generate auth tokens
5. Add the generated tokens to Modal secrets as shown
6. Run Cell 6 to deploy

### Step 3: Configure Replit
After successful deployment, add these secrets to Replit:

1. **MODAL_ENDPOINT**: `https://qgpt--generate.modal.run`
2. **MODAL_API_KEY**: Same as in Modal (e.g., `ak-XXXXXXXXXXXXX`)
3. **MODAL_TOKEN_SECRET**: Same as in Modal (e.g., `as-XXXXXXXXXXXXXXXX`)
4. **QRNG_API_KEY**: Your Quantum Blockchains API key

### Step 4: Test the Connection
Run the test script to verify everything works:
```bash
python test_modal_endpoint.py
```

## Authentication Flow
1. Replit sends request with Basic Auth header
2. Auth format: `base64(MODAL_API_KEY:MODAL_TOKEN_SECRET)`
3. Modal validates against its `modal-auth` secret
4. If valid, processes the generation request

## Request Format
```json
{
  "prompt": "Your text prompt",
  "max_tokens": 512,
  "temperature": 0.8,
  "quantum_profile": "medium"
}
```

## Quantum Profiles
- `strict`: No quantum modification (control)
- `light`: 10% quantum influence on logits
- `medium`: 30% quantum influence (balanced)
- `spicy`: 50% quantum influence (strong)
- `chaos`: 80% quantum influence (maximum)

## Response Format
```json
{
  "generated_text": "The generated response",
  "tokens_generated": 42,
  "quantum_profile": "medium",
  "quantum_diagnostics": {
    "total_entropy_consumed": 168000,
    "modifications_applied": 42,
    "average_logit_diff": 0.0234
  },
  "generation_time": 3.456,
  "model": "GPT-OSS 120B (8-bit)",
  "temperature": 0.8
}
```

## Troubleshooting

### Issue: Modal endpoints not resolving
**Solution**: Ensure Cell 6 deployment completed successfully. Check Modal dashboard for active deployment.

### Issue: Authentication errors
**Solution**: Verify MODAL_API_KEY and MODAL_TOKEN_SECRET match in both Modal and Replit secrets.

### Issue: QRNG not working
**Solution**: Check QRNG_API_KEY is set in both Modal and Replit. Verify API key is valid with Quantum Blockchains.

### Issue: Generation fails
**Solution**: Check Modal logs for GPU allocation. Ensure A100 80GB is available in your Modal account.

## Cost Optimization
- **keep_warm=1**: Keeps one instance always ready (~$95/month)
- **container_idle_timeout=300**: Shuts down after 5 minutes of inactivity
- **max_containers=1**: Prevents scaling to multiple expensive instances

## Quick Reference

### Environment Variables (Replit)
```
MODAL_ENDPOINT=https://qgpt--generate.modal.run
MODAL_API_KEY=ak-XXXXXXXXXXXXX
MODAL_TOKEN_SECRET=as-XXXXXXXXXXXXXXXX
QRNG_API_KEY=your-quantum-blockchains-key
```

### Modal App Configuration
```python
app = modal.App("qgpt")  # Short name for short URLs
gpu_config = modal.gpu.A100(count=1, size="80GB")
memory = 131072  # 128GB RAM
cpu = 16  # 16 cores
```

### Model Configuration
```python
model = "openai/gpt-oss-120b"
quantization = "8-bit"
framework = "transformers"
```

## Success Indicators
✅ Health endpoint returns: `{"status": "healthy", "service": "Quantum GPT API"}`
✅ Generate endpoint accepts POST requests with auth
✅ QRNG entropy pool fills successfully
✅ Tokens stream to frontend with quantum diagnostics
✅ Layer analysis shows quantum influence metrics

## Support
- **Modal Issues**: Check Modal dashboard and logs
- **Replit Issues**: Check workflow logs and console
- **QRNG Issues**: Verify API key with Quantum Blockchains
- **Model Issues**: Ensure GPU allocation and memory limits