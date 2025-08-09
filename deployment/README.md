# GPT-OSS 120B Deployment Options

## Model Information
- **Model**: bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental
- **HuggingFace URL**: https://huggingface.co/bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental
- **Size**: 40GB-120GB depending on quantization
- **Requirements**: 60-80GB VRAM for good quality

## Quick Start Options

### Option 1: Run Locally on WSL Ubuntu (Requires High-End GPU)

```bash
# Quick setup (run in WSL Ubuntu)
cd deployment
chmod +x wsl-quickstart.sh
./wsl-quickstart.sh
```

**Requirements**:
- Windows 11 with WSL2
- NVIDIA GPU with 60GB+ VRAM (A100, H100, or RTX 6000 Ada)
- 100GB+ free disk space
- 64GB+ system RAM

**Full Guide**: See `WSL_UBUNTU_SETUP.md`

### Option 2: Deploy to Cloud (Recommended for Most Users)

#### Modal (Best Value - $95/month)
```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Deploy
modal deploy deployment/modal-gpt-oss-120b.py
```

**Pros**: 
- Serverless (pay only when used)
- Auto-scaling
- No GPU required locally

**Full Guide**: See `DEPLOYMENT_GUIDE.md`

## GPU Memory Requirements

| Quantization | Size | Min VRAM | Quality | Use Case |
|-------------|------|----------|---------|----------|
| Q2_K | 40GB | 48GB | Low | Testing only |
| Q3_K_S | 50GB | 60GB | Medium | Budget deployments |
| Q4_K_M | 65GB | 80GB | Good | **Recommended** |
| Q5_K_M | 80GB | 96GB | High | Quality focus |
| Q6_K | 95GB | 112GB | Very High | Research |
| Q8_0 | 120GB | 140GB | Near Perfect | Maximum quality |

## QRNG Integration

All deployments integrate with Quantum Blockchains QRNG API:
1. Get API key from: https://qrng.qbck.io
2. Set environment variable: `QRNG_API_KEY=your-key`
3. System uses true quantum randomness for logit modification
4. NO fallback to pseudorandom - halts if QRNG unavailable

## File Structure

```
deployment/
├── README.md                    # This file
├── WSL_UBUNTU_SETUP.md         # Complete WSL setup guide
├── DEPLOYMENT_GUIDE.md         # Cloud deployment guide
├── wsl-quickstart.sh           # Quick setup script for WSL
├── local-server.py             # Local server implementation
├── modal-gpt-oss-120b.py       # Modal deployment script
├── integration-example.ts      # Integration examples
└── cloud-model-providers.ts    # Cloud provider implementations
```

## Cost Comparison

| Platform | Setup Cost | Per Request | Monthly (Light) | Monthly (24/7) |
|----------|------------|-------------|-----------------|----------------|
| Local WSL | GPU Cost | Electricity | ~$50 power | ~$200 power |
| Modal | $0.50 | $0.01-0.03 | ~$95 | ~$1,900 |
| RunPod | $0 | N/A | ~$500 | ~$2,190 |
| AWS | $0 | N/A | ~$1,000 | ~$24,000 |

## Common Issues

### "Out of Memory" Error
- **Solution**: Use smaller quantization or cloud deployment

### "CUDA not found"
- **Solution**: Install NVIDIA drivers for WSL2

### "Model download too slow"
- **Solution**: Model is 40-120GB, expect 1-3 hour download

### "QRNG not available"
- **Solution**: Set QRNG_API_KEY environment variable

## Support

- **Model Issues**: Check HuggingFace discussions
- **WSL Issues**: See WSL_UBUNTU_SETUP.md troubleshooting
- **Cloud Issues**: Check provider-specific logs
- **QRNG Issues**: Verify API key at https://qrng.qbck.io