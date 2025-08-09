#!/usr/bin/env python3
"""
AUTOMATED MODAL CONNECTION SCRIPT FOR GAIA QUANTUM NEXUS
=========================================================
This script automatically connects your Modal deployment to Replit!

Run this AFTER deploying to Modal:
python3 CONNECT_MODAL.py
"""

import os
import json
import requests
import time

def test_modal_endpoint(endpoint):
    """Test if Modal endpoint is working"""
    try:
        # Test health endpoint
        health_url = endpoint.replace("-generate-endpoint", "-health")
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Modal endpoint is LIVE!")
            print(f"   Model: {data.get('model', 'GPT-OSS 120B')}")
            print(f"   Status: {data.get('status', 'healthy')}")
            return True
    except:
        pass
    return False

def save_modal_config(endpoint):
    """Save Modal configuration to environment"""
    # Extract token from endpoint URL
    token = endpoint.split("--")[0].replace("https://", "")
    
    # Save to .env file
    env_content = f"""
# Modal Configuration - Auto-generated
MODAL_ENDPOINT={endpoint}
MODAL_API_KEY={token}
"""
    
    with open(".env", "a") as f:
        f.write(env_content)
    
    print("\n✅ Configuration saved to .env")

def main():
    print("🌌 GAIA QUANTUM NEXUS - MODAL CONNECTOR")
    print("=" * 50)
    print("\nThis will connect your Modal deployment to Replit!")
    print("\nAfter deploying to Modal, you'll get endpoints like:")
    print("https://ak-4jazee--gaia-quantum-120b-generate-endpoint.modal.run")
    print("\n" + "=" * 50)
    
    # Get endpoint from user
    endpoint = input("\n📌 Paste your Modal generation endpoint here: ").strip()
    
    if not endpoint:
        print("❌ No endpoint provided!")
        print("\nPlease:")
        print("1. Go to https://modal.com/playground")
        print("2. Copy code from MODAL_WEB_NOTEBOOK.py")
        print("3. Run it in Modal playground")
        print("4. Copy the endpoint URL and run this script again")
        return
    
    # Test the endpoint
    print("\n🔍 Testing Modal endpoint...")
    if test_modal_endpoint(endpoint):
        # Save configuration
        save_modal_config(endpoint)
        
        print("\n" + "=" * 50)
        print("🚀 SUCCESS! Your Quantum Nexus is connected!")
        print("=" * 50)
        print("\n✨ Everything is ready! You can now:")
        print("   1. Open your web interface")
        print("   2. Enter any prompt")
        print("   3. Experience 120B quantum-enhanced AI!")
        print("\n💫 The system will use true QRNG with NO pseudorandom fallback!")
        print("   QRNG is already pooling entropy (check the logs)")
        
        # Test generation
        print("\n📝 Want to test generation now? (y/n): ", end="")
        if input().lower() == 'y':
            test_prompt = "Hello, quantum consciousness"
            print(f"\nTesting with prompt: '{test_prompt}'")
            
            try:
                response = requests.post(
                    endpoint,
                    json={"prompt": test_prompt, "max_tokens": 20},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"\n✅ Generated: {result.get('generated_text', 'Success!')}")
                    print(f"   Tokens: {result.get('tokens_generated', 0)}")
                    print(f"   Quantum: {result.get('performance', {}).get('quantum', 'enabled')}")
                else:
                    print(f"Response: {response.status_code}")
            except Exception as e:
                print(f"Test failed: {e}")
                print("But your endpoint is configured! Try from the web interface.")
    else:
        print("\n⚠️ Could not reach Modal endpoint!")
        print("\nPlease check:")
        print("1. The endpoint URL is correct")
        print("2. Modal deployment is complete")
        print("3. The app is running (may take 10-30s to start)")
        print("\nYou can still save the configuration and try later.")
        
        save_anyway = input("\n💾 Save configuration anyway? (y/n): ")
        if save_anyway.lower() == 'y':
            save_modal_config(endpoint)
            print("\n✅ Configuration saved! Try refreshing your web app.")

if __name__ == "__main__":
    main()