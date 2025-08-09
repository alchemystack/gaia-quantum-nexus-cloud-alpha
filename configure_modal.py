#!/usr/bin/env python3
"""
Modal Configuration Helper for Quantum GPT (qgpt)
This script helps configure and validate Modal endpoints for the Quantum GPT system.
"""

import os
import sys
import base64
import requests
from typing import Optional, Tuple

# QGPT endpoints (consistent across entire codebase)
HEALTH_ENDPOINT = "https://qgpt--health.modal.run"
GENERATE_ENDPOINT = "https://qgpt--generate.modal.run"

def check_environment() -> Tuple[bool, str]:
    """Check if all required environment variables are set"""
    required = ["MODAL_API_KEY", "MODAL_TOKEN_SECRET", "MODAL_ENDPOINT", "QRNG_API_KEY"]
    missing = []
    
    for var in required:
        if not os.environ.get(var):
            missing.append(var)
    
    if missing:
        return False, f"Missing environment variables: {', '.join(missing)}"
    
    # Verify endpoint format
    endpoint = os.environ.get("MODAL_ENDPOINT", "")
    if "qgpt" not in endpoint:
        return False, f"MODAL_ENDPOINT should be: {GENERATE_ENDPOINT}"
    
    return True, "All environment variables configured correctly"

def test_health_endpoint() -> bool:
    """Test the health endpoint (no auth required)"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {HEALTH_ENDPOINT}")
        print("   Please ensure Modal deployment is complete")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_generate_endpoint() -> bool:
    """Test the generate endpoint with authentication"""
    api_key = os.environ.get("MODAL_API_KEY", "")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    
    if not api_key or not token_secret:
        print("❌ Missing authentication credentials")
        return False
    
    # Create auth header
    auth_string = f"{api_key}:{token_secret}"
    auth_header = f"Basic {base64.b64encode(auth_string.encode()).decode()}"
    
    # Test payload
    test_payload = {
        "prompt": "Test quantum generation:",
        "max_tokens": 10,
        "temperature": 0.8,
        "quantum_profile": "strict"  # Use strict to avoid QRNG requirement
    }
    
    try:
        response = requests.post(
            GENERATE_ENDPOINT,
            json=test_payload,
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "error" not in data:
                print(f"✅ Generate endpoint working")
                print(f"   Generated {data.get('tokens_generated', 0)} tokens")
                return True
            else:
                print(f"❌ Generation error: {data['error']}")
                return False
        else:
            print(f"❌ Generate request failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {GENERATE_ENDPOINT}")
        print("   Please ensure Modal deployment is complete")
        return False
    except Exception as e:
        print(f"❌ Generate endpoint error: {e}")
        return False

def display_configuration():
    """Display the current configuration"""
    print("\n" + "="*60)
    print("QUANTUM GPT (qgpt) CONFIGURATION")
    print("="*60)
    
    print("\n📍 Expected Modal Endpoints:")
    print(f"   Health:   {HEALTH_ENDPOINT}")
    print(f"   Generate: {GENERATE_ENDPOINT}")
    
    print("\n🔑 Current Environment:")
    endpoint = os.environ.get("MODAL_ENDPOINT", "Not set")
    api_key = os.environ.get("MODAL_API_KEY", "Not set")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "Not set")
    qrng_key = os.environ.get("QRNG_API_KEY", "Not set")
    
    print(f"   MODAL_ENDPOINT:     {endpoint}")
    print(f"   MODAL_API_KEY:      {api_key[:10]}..." if api_key != "Not set" else "   MODAL_API_KEY:      Not set")
    print(f"   MODAL_TOKEN_SECRET: {token_secret[:10]}..." if token_secret != "Not set" else "   MODAL_TOKEN_SECRET: Not set")
    print(f"   QRNG_API_KEY:       {qrng_key[:10]}..." if qrng_key != "Not set" else "   QRNG_API_KEY:       Not set")
    
    print("\n" + "="*60)

def setup_instructions():
    """Display setup instructions"""
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1️⃣  Deploy to Modal:")
    print("   - Copy cells from MODAL_NOTEBOOK_UPDATED_2025.py")
    print("   - Run cells 1-6 in Modal notebook")
    print("   - Cell 6 will show your endpoints")
    
    print("\n2️⃣  Configure Replit Secrets:")
    print(f"   MODAL_ENDPOINT = {GENERATE_ENDPOINT}")
    print("   MODAL_API_KEY = [from Modal deployment]")
    print("   MODAL_TOKEN_SECRET = [from Modal deployment]")
    print("   QRNG_API_KEY = [your Quantum Blockchains key]")
    
    print("\n3️⃣  Test Connection:")
    print("   python configure_modal.py --test")
    
    print("\n" + "="*60)

def main():
    """Main configuration helper"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure and test Modal endpoints for Quantum GPT")
    parser.add_argument("--test", action="store_true", help="Test Modal endpoints")
    parser.add_argument("--check", action="store_true", help="Check environment configuration")
    parser.add_argument("--setup", action="store_true", help="Show setup instructions")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_instructions()
        return
    
    # Always display current configuration
    display_configuration()
    
    if args.check or args.test:
        print("\n🔍 Checking configuration...")
        valid, message = check_environment()
        print(f"   {message}")
        
        if not valid and not args.test:
            print("\n⚠️  Please configure missing environment variables")
            setup_instructions()
            return
    
    if args.test:
        print("\n🧪 Testing Modal endpoints...")
        
        # Test health endpoint (no auth)
        print("\n1. Testing health endpoint...")
        health_ok = test_health_endpoint()
        
        if not health_ok:
            print("\n⚠️  Modal deployment not reachable")
            print("   Please complete Modal deployment first")
            return
        
        # Test generate endpoint (with auth)
        print("\n2. Testing generate endpoint...")
        generate_ok = test_generate_endpoint()
        
        if generate_ok:
            print("\n✅ All tests passed! Quantum GPT is ready to use!")
        else:
            print("\n⚠️  Generation test failed")
            print("   Check authentication credentials and Modal logs")
    
    if not any([args.test, args.check, args.setup]):
        print("\nUse --help to see available options")

if __name__ == "__main__":
    main()