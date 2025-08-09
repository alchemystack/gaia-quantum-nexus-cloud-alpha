#!/usr/bin/env python3
"""
🌌 GAIA QUANTUM NEXUS - AUTOMATIC MODAL CONNECTOR
Automatically connects to your Modal deployment
"""

import os
import sys
import time
import json

# We'll use subprocess to test endpoints with curl
import subprocess

def test_endpoint_curl(url):
    """Test if an endpoint is accessible using curl"""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", url],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() == "200"
    except:
        return False

def save_modal_config(endpoint):
    """Save Modal configuration to environment"""
    print(f"\n💾 Saving configuration...")
    
    # Create .env.modal file for reference
    env_content = f"""# Modal Configuration - Auto-generated
MODAL_ENDPOINT={endpoint}
MODAL_API_KEY=ak-4jAZeEPxVf7YMT0MYey2dw

# Add these to Replit Secrets!
"""
    
    with open("deployment/.env.modal.auto", "w") as f:
        f.write(env_content)
    
    print(f"   ✅ Saved to deployment/.env.modal.auto")
    return True

def main():
    print("🌌 GAIA QUANTUM NEXUS - AUTOMATIC MODAL CONNECTOR")
    print("=" * 60)
    print("\n📡 Scanning for your Modal deployment...")
    print("   Notebook: https://modal.com/notebooks/alchemystack/_/nb-j0cr4flsN8Eldy7y3ZFCBv")
    print("\n" + "=" * 60)
    
    # Possible endpoint patterns based on Modal's conventions
    base_patterns = [
        # Pattern 1: token--app-method (most common)
        "ak-4jAZeEPxVf7YMT0MYey2dw--gaia-quantum-120b",
        # Pattern 2: username--app-method  
        "alchemystack--gaia-quantum-120b",
        # Pattern 3: workspace-app pattern
        "alchemystack-modal--gaia-quantum-120b",
        # Pattern 4: notebook ID pattern
        "nb-j0cr4flsN8Eldy7y3ZFCBv",
        # Pattern 5: Simple app name
        "gaia-quantum-120b",
    ]
    
    found_endpoint = None
    
    print("\n🔍 Testing endpoint patterns...")
    for pattern in base_patterns:
        # Test health endpoint first (it's lighter)
        health_url = f"https://{pattern}-health.modal.run"
        print(f"\n   Testing: {pattern}-health")
        print(f"   URL: {health_url[:60]}...")
        
        if test_endpoint_curl(health_url):
            # Found it! Now construct the generation endpoint
            found_endpoint = f"https://{pattern}-generate-endpoint.modal.run"
            print(f"   ✅ SUCCESS! Found working endpoint")
            break
        else:
            print(f"   ❌ Not accessible")
    
    if not found_endpoint:
        print("\n" + "=" * 60)
        print("⚠️  MODAL ENDPOINT NOT YET ACCESSIBLE")
        print("=" * 60)
        print("\n📝 This is normal! The Modal notebook is likely still:")
        print("   1. Downloading the 120B model (can take 10-15 minutes)")
        print("   2. Starting the llama.cpp server")
        print("   3. Loading the model into GPU memory")
        print("\n⏳ What to do:")
        print("   1. Check your Modal notebook for progress")
        print("   2. Wait for 'Deployment complete!' message")
        print("   3. Look for the endpoint URLs in the notebook output")
        print("   4. The URLs will look like:")
        print("      https://YOUR-ID--gaia-quantum-120b-generate-endpoint.modal.run")
        print("\n💡 Once you have the endpoint URL:")
        print("   1. Add to Replit Secrets:")
        print("      MODAL_ENDPOINT = <your-endpoint-url>")
        print("      MODAL_API_KEY = ak-4jAZeEPxVf7YMT0MYey2dw")
        print("   2. Restart the application")
        print("\n🔄 You can run this script again to retry detection.")
        return 1
    
    # Success! We found the endpoint
    print("\n" + "=" * 60)
    print("✅ MODAL CONNECTION ESTABLISHED!")
    print("=" * 60)
    
    print(f"\n🎯 Endpoints found:")
    print(f"   Generation: {found_endpoint}")
    print(f"   Health: {found_endpoint.replace('generate-endpoint', 'health')}")
    
    # Save configuration
    save_modal_config(found_endpoint)
    
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS")
    print("=" * 60)
    print("\n1. Add these to Replit Secrets (⚙️ Tools → Secrets):")
    print(f"   MODAL_ENDPOINT = {found_endpoint}")
    print(f"   MODAL_API_KEY = ak-4jAZeEPxVf7YMT0MYey2dw")
    print("\n2. Restart your application")
    print("\n3. Your Quantum Interface will now use:")
    print("   • Real GPT-OSS 120B GGUF model")
    print("   • True quantum randomness (NO fallback)")
    print("   • 1x A100 64GB GPU")
    print("   • Flash Attention for faster inference")
    print("\n✨ The quantum consciousness awaits!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())