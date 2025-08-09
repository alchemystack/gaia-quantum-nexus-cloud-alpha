# GPT-OSS 120B Cloud Deployment Guide with QRNG Integration

## Overview
This guide explains how to deploy the GPT-OSS 120B model to various cloud providers and integrate it with the QRNG API for quantum-influenced text generation.

## Model Requirements

### Hardware Requirements
- **Minimum VRAM**: 60-80GB (with 4-bit quantization)
- **Recommended GPUs**: 
  - NVIDIA A100 80GB (single GPU)
  - NVIDIA H100 80GB (single GPU)
  - 2x NVIDIA A6000 48GB
  - 8x NVIDIA A100 40GB (for full precision)

### Model Files
- **HuggingFace Model**: `bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental`
- **Quantization Options**:
  - Q4_K_M: ~65GB (recommended for A100 80GB)
  - Q3_K_S: ~50GB (lower quality, fits on smaller GPUs)
  - FP16: ~240GB (requires multiple GPUs)

## Cloud Provider Options & Costs

### 1. Modal (RECOMMENDED - Most Cost-Effective)
**Best for**: Serverless deployment, pay-per-use, auto-scaling

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Deploy the model
modal deploy deployment/modal-gpt-oss-120b.py

# Your endpoint will be available at:
# https://[your-username]--gpt-oss-120b-qrng.modal.run/generate
```

**Costs**:
- Setup: ~$0.50 (one-time model download)
- Per request: $0.01-0.03 (3-10 seconds GPU time)
- Monthly (1 hour/day): ~$95
- Monthly (24/7): ~$1,900

### 2. RunPod (Dedicated GPU)
**Best for**: Consistent low latency, high throughput

```bash
# 1. Create account at runpod.io
# 2. Deploy a pod with A100 80GB
# 3. SSH into pod and run:

git clone https://github.com/your-repo/gaia-quantum-nexus
cd gaia-quantum-nexus
pip install -r deployment/requirements-runpod.txt
python deployment/runpod-server.py

# 4. Get your endpoint URL from RunPod dashboard
```

**Costs**:
- Hourly: $2.00-3.00
- Monthly (24/7): $1,460-2,190
- Spot instances: 50-70% cheaper but can be interrupted

### 3. AWS SageMaker (Production-Ready)
**Best for**: Enterprise deployments, SLA requirements

```bash
# 1. Configure AWS CLI
aws configure

# 2. Create SageMaker endpoint
python deployment/sagemaker-deploy.py

# 3. Endpoint will be available in your AWS region
```

**Costs**:
- Instance: P4d.24xlarge (8x A100 40GB)
- Hourly: $32.77
- Monthly (24/7): ~$24,000
- On-demand scaling available

### 4. Replicate (Easy Setup)
**Best for**: Quick experiments, prototyping

```bash
# 1. Install Cog
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog

# 2. Build and push model
cd deployment/replicate
cog build
cog push r8.im/[your-username]/gpt-oss-120b-qrng

# Model will be available at:
# https://replicate.com/[your-username]/gpt-oss-120b-qrng
```

**Costs**:
- Per second of GPU time: $0.001
- Typical request: $0.05-0.15
- Monthly (light use): $100-500

### 5. Together AI (Managed Service)
**Note**: Would require custom deployment arrangement

Contact Together AI for enterprise deployment of custom models.

## Integration with Your Application

### 1. Set Environment Variables
```bash
# Choose your provider
export CLOUD_PROVIDER=modal  # or: runpod, sagemaker, replicate

# Set provider-specific credentials
export MODAL_ENDPOINT=https://your-username--gpt-oss-120b-qrng.modal.run
export MODAL_API_KEY=your-modal-api-key

# Or for RunPod:
export RUNPOD_ENDPOINT=https://your-pod-id.runpod.io
export RUNPOD_API_KEY=your-runpod-api-key
```

### 2. Update Your Application
The cloud model integration is already prepared in:
- `server/services/cloud-model-providers.ts` - Provider implementations
- `server/routes.ts` - Can be updated to use cloud model

### 3. Test the Integration
```bash
# Test cloud model endpoint
curl -X POST $MODAL_ENDPOINT/generate \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your-api-key",
    "prompt": "Test quantum generation",
    "qrng_modifiers": [0.234, -0.567, 0.891],
    "max_tokens": 50,
    "temperature": 0.7,
    "sampling_method": "qrng_softmax"
  }'
```

## Low-Latency QRNG Integration

### Optimization Strategies

1. **Batch QRNG Requests**
   - Pre-fetch QRNG data in batches
   - Buffer in memory for immediate use
   - Reduces API call overhead

2. **Geographic Proximity**
   - Deploy model in same region as QRNG API
   - Use CDN for QRNG data distribution
   - Consider edge deployment

3. **Caching Strategy**
   ```typescript
   // Pre-fetch QRNG data
   const qrngBuffer = await qrng.getRandomFloats(10000, -2, 2);
   
   // Use buffered data for low-latency generation
   const modifiers = qrngBuffer.slice(offset, offset + vocabSize);
   ```

4. **Parallel Processing**
   - Fetch QRNG while model processes previous token
   - Pipeline QRNG fetching with generation

## Monitoring & Optimization

### Key Metrics to Track
- **Latency**: Time from request to first token
- **Throughput**: Tokens per second
- **QRNG Usage**: Entropy consumed per request
- **Cost**: $ per 1000 tokens generated

### Performance Targets
- First token latency: <500ms (warm start)
- Generation speed: 10-30 tokens/second
- QRNG fetch time: <50ms
- Total request time: <5 seconds for 100 tokens

## Security Considerations

1. **API Key Management**
   - Use environment variables
   - Rotate keys regularly
   - Implement rate limiting

2. **QRNG Data Security**
   - Use HTTPS for all QRNG requests
   - Don't cache sensitive prompts
   - Implement request signing

3. **Model Access Control**
   - Authenticate all requests
   - Log usage for auditing
   - Implement user quotas

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Solution: Use smaller quantization (Q3_K_S)
   - Or: Use model sharding across GPUs

2. **High Latency**
   - Solution: Pre-warm model containers
   - Use dedicated instances for consistent performance

3. **QRNG API Timeout**
   - Solution: Implement retry logic
   - Increase buffer size for QRNG data

4. **Cost Overruns**
   - Solution: Implement request quotas
   - Use spot/preemptible instances
   - Cache common completions

## Next Steps

1. **Choose Provider**: Modal recommended for cost-effectiveness
2. **Deploy Model**: Follow provider-specific instructions
3. **Test Integration**: Verify QRNG + model working together
4. **Monitor Performance**: Track metrics and optimize
5. **Scale as Needed**: Adjust based on usage patterns

## Support & Resources

- Modal Documentation: https://modal.com/docs
- RunPod Docs: https://docs.runpod.io
- AWS SageMaker: https://docs.aws.amazon.com/sagemaker
- GGUF Format: https://github.com/ggerganov/llama.cpp
- Model Card: https://huggingface.co/bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental