# 🔐 Modal Authentication Setup Guide

## Overview
Modal uses a two-part authentication system with a token-id and token-secret. The Replit application now requires THREE secrets to connect to your Modal-hosted GPT-OSS 120B model.

## Required Secrets

You need to set up THREE secrets in Replit:

### 1. MODAL_API_KEY (Token ID)
- **What it is**: The first part of your Modal credentials (token-id)
- **Format**: Starts with `ak-` 
- **Example**: `ak-4SNpJXOvye2giKFCcbS0un`

### 2. MODAL_TOKEN_SECRET (Token Secret)
- **What it is**: The second part of your Modal credentials (token-secret)
- **Format**: Starts with `as-`
- **Example**: `as-udzBb9p95KHG2W1ts29l0p`

### 3. MODAL_ENDPOINT
- **What it is**: The URL of your deployed Modal function
- **Format**: `https://[deployment-name]--[class-method].modal.run`
- **Example**: `https://gaia-quantum-transformers-optimized--quantumgpt120btransformers-generate.modal.run`

## Step-by-Step Setup in Replit

### Step 1: Get Your Modal Credentials
1. Go to [modal.com/settings](https://modal.com/settings)
2. Click on "API Tokens"
3. Create a new token or use an existing one
4. You'll see something like:
   ```
   modal token set --token-id ak-4SNpJXOvye2giKFCcbS0un --token-secret as-udzBb9p95KHG2W1ts29l0p
   ```

### Step 2: Add Secrets to Replit

1. **Open Replit Secrets**:
   - In your Replit workspace, look for **Tools** in the left sidebar
   - Click on **Secrets** (or click the + button and type "Secrets")

2. **Add MODAL_API_KEY**:
   - Click **+ New Secret**
   - **Secret Key**: `MODAL_API_KEY`
   - **Value**: Your token-id (e.g., `ak-4SNpJXOvye2giKFCcbS0un`)
   - Click **Add Secret**

3. **Add MODAL_TOKEN_SECRET**:
   - Click **+ New Secret**
   - **Secret Key**: `MODAL_TOKEN_SECRET`
   - **Value**: Your token-secret (e.g., `as-udzBb9p95KHG2W1ts29l0p`)
   - Click **Add Secret**

4. **Add MODAL_ENDPOINT**:
   - Click **+ New Secret**
   - **Secret Key**: `MODAL_ENDPOINT`
   - **Value**: Your Modal endpoint URL from the deployment
   - Example: `https://gaia-quantum-transformers-optimized--quantumgpt120btransformers-generate.modal.run`
   - Click **Add Secret**

### Step 3: Verify Your Setup

Your Replit Secrets should now show:
```
MODAL_API_KEY = ak-4SNpJXOvye2giKFCcbS0un
MODAL_TOKEN_SECRET = as-udzBb9p95KHG2W1ts29l0p
MODAL_ENDPOINT = https://gaia-quantum-transformers-optimized--quantumgpt120btransformers-generate.modal.run
QRNG_API_KEY = [your existing QRNG key]
```

### Step 4: Restart Your Application

After adding all three secrets, restart your Replit application:
1. Click the **Stop** button in the console
2. Click **Run** again
3. You should see: `[Routes] Using Modal-hosted GPT-OSS 120B model with QRNG logit modification`

## Troubleshooting

### "Modal not configured" message still appears
- Make sure all THREE secrets are set (MODAL_API_KEY, MODAL_TOKEN_SECRET, MODAL_ENDPOINT)
- Check that there are no extra spaces in the secret values
- Restart the application after adding secrets

### Authentication errors
- Verify your token-id starts with `ak-`
- Verify your token-secret starts with `as-`
- Make sure you're using the correct endpoint URL from your deployment

### Connection timeouts
- The first request to Modal can take 30-60 seconds (cold start)
- Subsequent requests will be much faster
- The model stays warm for continued use

## How It Works

The application now uses Basic Authentication with Modal:
1. Combines `token-id:token-secret` into a single string
2. Encodes this with Base64
3. Sends as `Authorization: Basic [encoded-credentials]` header

This is the proper way to authenticate with Modal endpoints and ensures secure access to your deployed model.

## Testing Your Connection

Once configured, you can test by:
1. Opening your Replit app
2. Entering a prompt
3. Selecting any quantum profile
4. Clicking "Generate"

You should see quantum-enhanced text generation powered by the real GPT-OSS 120B model!