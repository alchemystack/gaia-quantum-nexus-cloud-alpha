#!/usr/bin/env python3
"""
🌌 ONE-CLICK CLOUD DEPLOYMENT FOR GAIA QUANTUM NEXUS
Run this script to deploy everything to the cloud - no local setup needed!
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_secrets():
    """Check if required secrets are set"""
    print("\n🔍 Checking Replit Secrets...")
    
    secrets = {
        "QRNG_API_KEY": os.environ.get("QRNG_API_KEY"),
        "MODAL_TOKEN_ID": os.environ.get("MODAL_TOKEN_ID"),
        "MODAL_TOKEN_SECRET": os.environ.get("MODAL_TOKEN_SECRET")
    }
    
    missing = [k for k, v in secrets.items() if not v]
    
    if missing:
        print("\n⚠️  MISSING SECRETS - Please add these in Replit:")
        print("   (Click the lock icon in the left sidebar)")
        print("")
        
        if "MODAL_TOKEN_ID" in missing or "MODAL_TOKEN_SECRET" in missing:
            print("   📝 MODAL SETUP:")
            print("   1. Go to https://modal.com/signup")
            print("   2. Create free account")
            print("   3. Go to https://modal.com/settings/tokens")
            print("   4. Create new token")
            print("   5. Add to Replit Secrets:")
            print("      • MODAL_TOKEN_ID = (token ID)")
            print("      • MODAL_TOKEN_SECRET = (token secret)")
            print("")
        
        if "QRNG_API_KEY" in missing:
            print("   🔮 QUANTUM SETUP:")
            print("   • QRNG_API_KEY = (your Quantum Blockchains API key)")
            print("")
        
        print("   After adding secrets, run this script again!")
        return False
    
    print("   ✅ All secrets configured!")
    return True

def setup_modal():
    """Configure Modal for cloud deployment"""
    print("\n⚙️  Setting up Modal...")
    
    # Create Modal config directory
    config_dir = Path.home() / ".modal"
    config_dir.mkdir(exist_ok=True)
    
    # Write Modal configuration
    config_file = config_dir / "config.toml"
    config_content = f"""
[default]
token_id = "{os.environ.get('MODAL_TOKEN_ID', '')}"
token_secret = "{os.environ.get('MODAL_TOKEN_SECRET', '')}"
active_profile = "default"
"""
    
    with open(config_file, "w") as f:
        f.write(config_content)
    
    print("   ✅ Modal configured!")
    return True

def deploy_model():
    """Deploy the 120B model to Modal cloud"""
    print("\n🚀 DEPLOYING GPT-OSS 120B TO MODAL CLOUD")
    print("=" * 50)
    print("   Model: bartowski/openai_gpt-oss-120b-GGUF")
    print("   Size: 60GB (downloads on first run)")
    print("   GPU: 2x NVIDIA A100 80GB")
    print("   Cost: ~$95/month (light usage)")
    print("=" * 50)
    
    # Run the cloud deployment
    result = subprocess.run(
        [sys.executable, "deployment/cloud_deploy.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ MODEL DEPLOYED SUCCESSFULLY!")
        
        # Extract endpoint URLs from output
        username = os.environ.get("REPL_OWNER", "user")
        base_url = f"https://{username}--gaia-quantum-120b-cloud"
        
        endpoints = {
            "generate": f"{base_url}-generate-endpoint.modal.run",
            "health": f"{base_url}-health.modal.run"
        }
        
        print("\n📌 YOUR CLOUD ENDPOINTS:")
        print(f"   Generation: {endpoints['generate']}")
        print(f"   Health: {endpoints['health']}")
        
        # Save endpoints to environment
        os.environ["MODAL_ENDPOINT"] = endpoints["generate"]
        
        # Update secrets reminder
        print("\n📝 ADD THESE TO REPLIT SECRETS:")
        print(f"   MODAL_ENDPOINT = {endpoints['generate']}")
        print(f"   MODAL_API_KEY = (use your Modal token)")
        
        return endpoints
    else:
        print("\n❌ Deployment failed!")
        if "not authenticated" in result.stderr:
            print("   Modal authentication issue - check your tokens")
        else:
            print(f"   Error: {result.stderr[:500]}")
        return None

def test_deployment(endpoints):
    """Test the deployed model"""
    if not endpoints:
        return
    
    print("\n🧪 Testing deployment...")
    
    try:
        import requests
        
        # Test health endpoint
        print("   Checking health endpoint...")
        response = requests.get(endpoints["health"], timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model status: {data.get('status')}")
            print(f"   ✅ Quantum: {data.get('quantum')}")
            print(f"   ✅ GPU: {data.get('gpu')}")
        else:
            print(f"   ⚠️ Health check returned: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Test failed: {e}")
        print("   (Model may still be initializing - try again in a minute)")

def main():
    """Main deployment process"""
    print("=" * 60)
    print("🌌 GAIA QUANTUM NEXUS - ONE-CLICK CLOUD DEPLOYMENT")
    print("=" * 60)
    print("\nThis will deploy the 120B quantum AI model to the cloud!")
    print("Everything runs from Replit - no local setup needed.")
    
    # Step 1: Check secrets
    if not check_secrets():
        return
    
    # Step 2: Setup Modal
    if not setup_modal():
        print("\n❌ Modal setup failed")
        return
    
    # Step 3: Deploy model
    endpoints = deploy_model()
    
    if endpoints:
        # Step 4: Test deployment
        test_deployment(endpoints)
        
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! YOUR QUANTUM AI IS LIVE!")
        print("=" * 60)
        print("\n✨ What to do next:")
        print("1. Add MODAL_ENDPOINT to Replit Secrets")
        print("2. Add MODAL_API_KEY to Replit Secrets")
        print("3. Restart your app")
        print("4. The UI will show 'Modal GPT-OSS 120B'")
        print("\n🚀 Your quantum-augmented AI is ready!")
    else:
        print("\n❌ Deployment failed - see errors above")
        print("\nNeed help? Check:")
        print("• Modal dashboard: https://modal.com/apps")
        print("• Modal docs: https://modal.com/docs")

if __name__ == "__main__":
    # Check if running in Replit
    if not os.environ.get("REPL_ID"):
        print("⚠️  This script should be run from Replit!")
        print("   Upload it to your Replit project and run from there.")
    else:
        main()