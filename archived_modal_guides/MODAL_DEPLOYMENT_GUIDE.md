# 🌌 Modal Deployment Guide - Enhanced Configuration

## Overview
This guide helps you deploy the GPT-OSS 120B GGUF model with enhanced resources on Modal.

## Enhanced Specifications
- **GPU**: 1x A100 with 80GB VRAM (upgraded from 64GB)
- **RAM**: 128GB system memory (doubled from 64GB)
- **CPU**: 16 cores (doubled from 8 cores)
- **Context**: 8K tokens (expandable with 128GB RAM)

## Files Available

### 1. `MODAL_NOTEBOOK_ENHANCED.py` (Recommended)
- **Purpose**: Cell-based deployment with local model upload support
- **Features**: 
  - Cell 1: Upload GGUF model from local storage
  - Cell 2: Server initialization (model stays loaded)
  - Cell 3: Test and get endpoints
- **Benefits**: More control, better resource usage, persistent model

### 2. `MODAL_WEB_NOTEBOOK.py` (Original)
- **Purpose**: Single-script deployment
- **Features**: Auto-downloads from HuggingFace
- **Specs**: Lower resources (64GB RAM, 8 cores)

## Deployment Steps

### Step 1: Prepare Modal Notebook
1. Go to https://modal.com/playground
2. Copy the entire content of `MODAL_NOTEBOOK_ENHANCED.py`
3. Paste into Modal playground

### Step 2: Run Cell 1 - Model Upload
```python
# This cell uploads your GGUF model
# Options:
# 1. Upload from local file (if you have it)
# 2. Auto-download from HuggingFace
```

### Step 3: Run Cell 2 - Start Server
```python
# This initializes the llama.cpp server
# Model stays loaded in memory (keep_warm=1)
# Uses all 16 CPU cores for processing
```

### Step 4: Get Your Endpoints
After deployment completes, you'll see:
```
✅ Your endpoints are ready at:
https://YOUR-ID--gaia-quantum-120b-enhanced-generate-endpoint.modal.run
https://YOUR-ID--gaia-quantum-120b-enhanced-health.modal.run
```

### Step 5: Connect to Replit
1. Copy your endpoint URLs from Modal
2. In Replit, go to Tools → Secrets
3. Add these secrets:
   - `MODAL_ENDPOINT` = (your generation endpoint URL)
   - `MODAL_API_KEY` = `ak-4jAZeEPxVf7YMT0MYey2dw`
4. Restart the Replit application

## Cost Optimization
- **Enhanced Config**: ~$95-120/month (80GB GPU + 128GB RAM)
- **Original Config**: ~$60/month (64GB GPU + 64GB RAM)
- **Keep Warm**: Model stays loaded, no cold starts
- **Concurrent**: Handles 10 simultaneous requests

## Features
- ✅ True quantum randomness (NO pseudorandom fallback)
- ✅ Flash Attention for faster inference
- ✅ 8K context window (expandable)
- ✅ Persistent model storage
- ✅ WebSocket streaming support

## Troubleshooting

### Model Download Takes Too Long
- First download takes 10-15 minutes (120B model is huge)
- Model is cached in persistent volume after first download
- Consider uploading from local if you have the GGUF file

### Endpoint Not Found
- Wait for "✅ MODEL READY FOR INFERENCE" message
- Check Modal dashboard for deployment status
- Run `python3 CONNECT_MODAL_AUTO.py` to auto-detect

### Out of Memory
- The enhanced config (128GB RAM) should handle the model
- If issues persist, reduce context size in server command

## Testing Connection
Once deployed, test with:
```bash
python3 CONNECT_MODAL_AUTO.py
```

This will automatically detect and configure your endpoints!