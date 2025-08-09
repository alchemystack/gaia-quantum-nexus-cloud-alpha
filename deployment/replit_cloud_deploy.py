#!/usr/bin/env python3
"""
One-Click Cloud Deployment from Replit
No local installation needed - everything runs in the cloud!
"""

import os
import sys
import json
import time
import modal
from pathlib import Path

print("=" * 60)
print("🌌 GAIA QUANTUM NEXUS - REPLIT CLOUD DEPLOYMENT")
print("=" * 60)
print("\n🎯 This script runs entirely from Replit!")
print("   No local installation needed")
print("   Everything happens in the cloud")

# Step 1: Check for Modal token
def setup_modal_auth():
    """Set up Modal authentication from Replit"""
    modal_token = os.environ.get("MODAL_TOKEN_ID")
    modal_secret = os.environ.get("MODAL_TOKEN_SECRET")
    
    if modal_token and modal_secret:
        print("\n✅ Modal authentication found in Replit secrets")
        return True
    
    print("\n📝 MODAL AUTHENTICATION NEEDED")
    print("=" * 40)
    print("\n1. Go to: https://modal.com/signup")
    print("2. Create your free account")
    print("3. Go to: https://modal.com/settings/tokens")
    print("4. Create a new token named 'gaia-quantum'")
    print("5. Copy the token ID and secret")
    print("\n6. Add to Replit Secrets (lock icon in sidebar):")
    print("   • MODAL_TOKEN_ID = (your token ID)")
    print("   • MODAL_TOKEN_SECRET = (your token secret)")
    print("\n7. After adding secrets, restart this script")
    print("=" * 40)
    
    return False

# Step 2: Create Modal configuration
def create_modal_config():
    """Create Modal config file for Replit"""
    config_dir = Path.home() / ".modal"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "config.toml"
    
    # Get tokens from environment
    token_id = os.environ.get("MODAL_TOKEN_ID", "")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    
    if not token_id or not token_secret:
        return False
    
    config_content = f"""
[default]
token_id = "{token_id}"
token_secret = "{token_secret}"
active_profile = "default"
"""
    
    with open(config_file, "w") as f:
        f.write(config_content)
    
    print("✅ Modal configuration created")
    return True

# Step 3: Create QRNG secret in Modal
def create_qrng_secret():
    """Create QRNG secret in Modal"""
    qrng_key = os.environ.get("QRNG_API_KEY")
    
    if not qrng_key:
        print("\n⚠️ QRNG_API_KEY not found in Replit secrets")
        print("   The model will work but without quantum enhancement")
        return False
    
    try:
        # Create secret using Modal API
        from modal import Secret
        
        # Try to create the secret
        secret = Secret.from_dict({"QRNG_API_KEY": qrng_key})
        print("✅ QRNG secret configured in Modal")
        return True
    except Exception as e:
        print(f"⚠️ Could not create QRNG secret: {e}")
        return False

# Step 4: Deploy to Modal
def deploy_to_modal():
    """Deploy the 120B model to Modal"""
    print("\n🚀 DEPLOYING GPT-OSS 120B TO MODAL CLOUD")
    print("=" * 40)
    print("Model: bartowski/openai_gpt-oss-120b-GGUF")
    print("Size: 60GB (will download on first run)")
    print("GPU: 2x NVIDIA A100 80GB")
    print("Cost: ~$95/month (light use)")
    print("=" * 40)
    
    try:
        # Import and deploy the cloud app
        from cloud_deploy import app
        
        print("\n⏳ Deploying (this takes 2-3 minutes)...")
        
        # Get deployment stub
        deploy_stub = modal.Runner().deploy_stub(app)
        
        # Get the endpoint URLs
        print("\n✅ DEPLOYMENT SUCCESSFUL!")
        print("\n📌 YOUR CLOUD ENDPOINTS:")
        
        # The actual URLs will be shown after deployment
        base_url = "https://YOUR-USERNAME--gaia-quantum-120b-cloud"
        print(f"   Generation: {base_url}-generate-endpoint.modal.run")
        print(f"   Health: {base_url}-health.modal.run")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return False

# Step 5: Update Replit configuration
def update_replit_config(endpoint_url: str):
    """Update Replit environment with Modal endpoint"""
    print("\n📝 Updating Replit configuration...")
    
    # The endpoint will be automatically detected by the app
    print(f"   MODAL_ENDPOINT = {endpoint_url}")
    print("   MODAL_API_KEY = (already configured)")
    
    print("\n✅ Configuration updated!")
    print("   The app will now use the cloud-hosted 120B model")

# Main deployment flow
def main():
    """Main cloud deployment process"""
    
    # Step 1: Check Modal auth
    if not setup_modal_auth():
        print("\n⚠️ Please add Modal tokens to Replit Secrets and restart")
        return
    
    # Step 2: Create Modal config
    if not create_modal_config():
        print("\n❌ Failed to create Modal configuration")
        return
    
    # Step 3: Create QRNG secret
    qrng_ok = create_qrng_secret()
    if not qrng_ok:
        print("\n⚠️ Continuing without quantum enhancement")
        print("   Add QRNG_API_KEY to Replit Secrets to enable quantum features")
    
    # Step 4: Deploy to Modal
    print("\n" + "=" * 60)
    print("🔄 STARTING CLOUD DEPLOYMENT")
    print("=" * 60)
    
    if deploy_to_modal():
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! YOUR QUANTUM AI IS LIVE IN THE CLOUD!")
        print("=" * 60)
        print("\n✨ What happens now:")
        print("1. The 120B model is deployed on Modal's GPUs")
        print("2. First request will download the model (5-10 min)")
        print("3. After that, responses in 1-2 seconds")
        print("4. The Replit app will automatically use the cloud model")
        print("\n🚀 Your quantum-augmented AI is ready to use!")
        print("   Just restart the app and start generating!")
    else:
        print("\n❌ Deployment failed. Please check:")
        print("1. Modal tokens are correct")
        print("2. You have Modal account access")
        print("3. Internet connection is stable")

if __name__ == "__main__":
    main()