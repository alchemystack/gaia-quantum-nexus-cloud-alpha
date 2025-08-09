#!/usr/bin/env python3
"""
Test script to verify Modal deployment and connection
Run this after deploying to Modal to test the endpoints
"""

import os
import sys
import json
import time
import requests
from typing import Dict, Any

def test_health_endpoint(endpoint_base: str) -> bool:
    """Test the health check endpoint"""
    health_url = endpoint_base.replace('generate-endpoint', 'health-check')
    
    print(f"\n🔍 Testing health endpoint: {health_url}")
    
    try:
        response = requests.get(health_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check successful!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model: {data.get('model')}")
            print(f"   Quantum: {data.get('quantum')}")
            print(f"   GPU: {data.get('gpu')}")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️  Health check timed out (cold start can take 10-30 seconds)")
        print("   Try again in a moment...")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_generation_endpoint(endpoint_url: str) -> bool:
    """Test the generation endpoint"""
    print(f"\n🔍 Testing generation endpoint: {endpoint_url}")
    
    # Test payload
    payload = {
        "prompt": "The quantum nature of consciousness reveals",
        "max_tokens": 30,
        "temperature": 0.7,
        "profile": "medium"
    }
    
    print(f"   Sending test prompt: '{payload['prompt']}'")
    print(f"   Profile: {payload['profile']}, Temperature: {payload['temperature']}")
    
    try:
        print("   ⏳ Generating (this may take 10-30 seconds on cold start)...")
        
        start_time = time.time()
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=120  # 2 minute timeout for cold starts
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            if 'error' in data:
                print(f"❌ Generation error: {data['error']}")
                if 'QRNG' in data['error']:
                    print("   ⚠️  QRNG API key not configured in Modal secret")
                    print("   Run: modal secret create qrng-api-key QRNG_API_KEY=your_key")
                return False
            
            print(f"✅ Generation successful! (took {elapsed:.1f} seconds)")
            print(f"   Generated text: {data.get('generated_text', '')[:100]}...")
            print(f"   Tokens generated: {data.get('tokens_generated', 0)}")
            print(f"   Entropy used: {data.get('entropy_used', 0)} bits")
            
            if 'performance' in data:
                perf = data['performance']
                print(f"   Performance: {perf.get('tokens_per_sec', 0):.1f} tokens/sec")
                print(f"   Model: {perf.get('model', 'Unknown')}")
            
            return True
        else:
            print(f"❌ Generation failed with status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️  Generation timed out")
        print("   This might be a cold start issue. Try again.")
        return False
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return False

def test_replit_connection() -> Dict[str, Any]:
    """Test the connection from Replit's perspective"""
    print("\n🔍 Testing Replit environment variables...")
    
    modal_endpoint = os.environ.get('MODAL_ENDPOINT')
    modal_api_key = os.environ.get('MODAL_API_KEY')
    qrng_api_key = os.environ.get('QRNG_API_KEY')
    
    status = {
        'modal_endpoint': '✅' if modal_endpoint else '❌',
        'modal_api_key': '✅' if modal_api_key else '❌',
        'qrng_api_key': '✅' if qrng_api_key else '❌'
    }
    
    print(f"   MODAL_ENDPOINT: {status['modal_endpoint']} {'(set)' if modal_endpoint else '(not set)'}")
    print(f"   MODAL_API_KEY: {status['modal_api_key']} {'(set)' if modal_api_key else '(not set)'}")
    print(f"   QRNG_API_KEY: {status['qrng_api_key']} {'(set)' if qrng_api_key else '(not set)'}")
    
    if modal_endpoint:
        return {
            'configured': True,
            'endpoint': modal_endpoint,
            'has_api_key': bool(modal_api_key),
            'has_qrng': bool(qrng_api_key)
        }
    else:
        return {'configured': False}

def main():
    """Main test routine"""
    print("=" * 60)
    print("🧪 Modal Connection Test for Gaia Quantum Nexus")
    print("=" * 60)
    
    # Check if running from command line with endpoint argument
    if len(sys.argv) > 1:
        endpoint = sys.argv[1]
        print(f"\n📍 Testing endpoint: {endpoint}")
    else:
        # Check Replit environment
        config = test_replit_connection()
        
        if config.get('configured'):
            endpoint = config['endpoint']
            print(f"\n📍 Using endpoint from environment: {endpoint}")
        else:
            print("\n📝 No endpoint configured. Please provide one:")
            print("   Usage: python test_modal_connection.py <endpoint_url>")
            print("   Example: python test_modal_connection.py https://username--gaia-quantum-gpt-oss-120b-generate-endpoint.modal.run")
            return
    
    # Test health endpoint
    health_ok = test_health_endpoint(endpoint)
    
    if not health_ok:
        print("\n⚠️  Health check failed. The model might be:")
        print("   1. Still deploying (wait a few minutes)")
        print("   2. Not deployed yet (run deploy_to_modal.py)")
        print("   3. Having issues (check Modal dashboard)")
        return
    
    # Test generation endpoint
    generation_ok = test_generation_endpoint(endpoint)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if health_ok and generation_ok:
        print("✅ All tests passed! Your Modal deployment is working.")
        print("\n🎯 Next steps:")
        print("1. Add MODAL_ENDPOINT to Replit Secrets")
        print("2. Add MODAL_API_KEY to Replit Secrets")
        print("3. Restart your Replit app")
        print("4. The UI will show 'Modal GPT-OSS 120B'")
    elif health_ok and not generation_ok:
        print("⚠️  Health check passed but generation failed.")
        print("\nPossible issues:")
        print("1. QRNG API key not configured in Modal")
        print("2. Model still loading (try again in a minute)")
        print("3. Check Modal logs: modal logs -f gaia-quantum-gpt-oss-120b")
    else:
        print("❌ Tests failed. Please check:")
        print("1. Deployment status in Modal dashboard")
        print("2. Endpoint URL is correct")
        print("3. Network connectivity")

if __name__ == "__main__":
    main()