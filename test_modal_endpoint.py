#!/usr/bin/env python3
"""
Test script to verify Modal deployment is working.
Run this after deploying to Modal to test the endpoints.
"""

import requests
import base64
import json
import os
from typing import Dict, Any

def test_modal_endpoints():
    """Test both health and generate endpoints"""
    
    # Get credentials from environment
    api_key = os.environ.get("MODAL_API_KEY", "")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    endpoint_base = os.environ.get("MODAL_ENDPOINT", "").replace("/generate", "")
    
    if not all([api_key, token_secret, endpoint_base]):
        print("❌ Missing Modal credentials in environment")
        print("   Please set: MODAL_API_KEY, MODAL_TOKEN_SECRET, MODAL_ENDPOINT")
        return False
    
    # For qgpt app, the URLs are simple:
    # https://qgpt--health.modal.run
    # https://qgpt--generate.modal.run
    
    # Extract base URL and construct endpoints
    if "--generate" in endpoint_base:
        health_url = endpoint_base.replace("--generate", "--health")
        generate_url = endpoint_base
    elif "qgpt" in endpoint_base:
        # Handle if just the base URL is provided
        base = endpoint_base.rstrip("/")
        if not base.endswith(".modal.run"):
            base = "https://qgpt.modal.run"
        health_url = base.replace(".modal.run", "--health.modal.run")
        generate_url = base.replace(".modal.run", "--generate.modal.run")
    else:
        # Default to qgpt endpoints
        health_url = "https://qgpt--health.modal.run"
        generate_url = "https://qgpt--generate.modal.run"
    
    print("🔍 Testing Modal endpoints...")
    print(f"   Health: {health_url}")
    print(f"   Generate: {generate_url}")
    print()
    
    # Test 1: Health check (no auth required for health)
    print("1️⃣ Testing health endpoint...")
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            print(f"   ✅ Health check passed: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health endpoint unreachable: {e}")
        return False
    
    # Test 2: Generate endpoint with auth
    print("\n2️⃣ Testing generate endpoint with authentication...")
    
    # Create auth header
    auth_string = f"{api_key}:{token_secret}"
    auth_header = f"Basic {base64.b64encode(auth_string.encode()).decode()}"
    
    # Test payload
    test_payload = {
        "prompt": "Hello, quantum world! The universe is",
        "max_tokens": 20,
        "temperature": 0.8,
        "quantum_profile": "medium"
    }
    
    try:
        response = requests.post(
            generate_url,
            json=test_payload,
            headers={
                "Authorization": auth_header,
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                print(f"   ❌ Generation error: {result['error']}")
                return False
            else:
                print(f"   ✅ Generation successful!")
                print(f"      Generated: {result.get('generated_text', '')[:100]}...")
                print(f"      Tokens: {result.get('tokens_generated', 0)}")
                print(f"      Profile: {result.get('quantum_profile', 'unknown')}")
                return True
        else:
            print(f"   ❌ Generation failed: {response.status_code}")
            print(f"      Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Generate endpoint error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("MODAL ENDPOINT TESTER")
    print("="*60)
    print()
    
    success = test_modal_endpoints()
    
    print()
    print("="*60)
    if success:
        print("✅ ALL TESTS PASSED! Your Modal deployment is working!")
        print("   You can now use the Quantum GPT interface in Replit.")
    else:
        print("❌ TESTS FAILED. Please check your Modal deployment.")
        print("\nTroubleshooting:")
        print("1. Ensure Modal notebook cells 1-6 ran successfully")
        print("2. Check that Modal secrets are configured")
        print("3. Verify MODAL_ENDPOINT URL is correct")
        print("4. Ensure your Modal app is deployed and running")
    print("="*60)