# ✅ FIXED CELL 7 - Ready to Test Your Endpoints

## Copy this EXACT code into Cell 7:

```python
# ============================================
# CELL 7: TEST YOUR DEPLOYMENT
# ============================================

import requests
import json
import datetime

# Your deployment endpoints (from Cell 6 output)
deployment_name = "gaia-quantum-transformers-optimized"
health_url = f"https://{deployment_name}--quantumgpt120btransformers-health.modal.run"
generate_url = f"https://{deployment_name}--quantumgpt120btransformers-generate.modal.run"

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🧪 Testing deployed endpoints...")
print("=" * 60)

# Test health endpoint
print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 1. Testing health endpoint...")
print(f"   URL: {health_url}")
try:
    response = requests.get(health_url, timeout=10)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Response code: {response.status_code}")
    
    if response.status_code == 200:
        health_data = response.json()
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Health check passed!")
        print(f"   Status: {health_data.get('status')}")
        print(f"   Model: {health_data.get('model')}")
        print(f"   Quantum: {health_data.get('quantum_enabled')}")
        print(f"   Container warm: {health_data.get('container_warm', 'unknown')}")
    else:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Health check failed: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
except requests.exceptions.Timeout:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏱️ Timeout - container might be cold starting")
    print("   This is normal for first request. Try again in 30 seconds.")
except Exception as e:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Health check error: {e}")

# Test generation endpoint
print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 2. Testing generation endpoint...")
print(f"   URL: {generate_url}")

test_payload = {
    "prompt": "The meaning of quantum consciousness is",
    "max_tokens": 50,
    "temperature": 0.7,
    "quantum_profile": "medium",
    "diagnostics": True
}

print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Sending test prompt: '{test_payload['prompt']}'")
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Quantum profile: {test_payload['quantum_profile']}")

try:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Waiting for response (may take 30-60s if cold start)...")
    response = requests.post(
        generate_url,
        json=test_payload,
        timeout=60  # Increased timeout for cold starts
    )
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Response code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "success":
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ Generation test passed!")
            print(f"\n   Generated text: '{result.get('generated_text', '')}'")
            print(f"\n   Tokens generated: {result.get('tokens_generated')}")
            print(f"   Quantum profile used: {result.get('quantum_profile')}")
            
            if result.get("quantum_diagnostics"):
                diag = result["quantum_diagnostics"]
                print(f"\n   📊 Quantum Diagnostics:")
                print(f"      Avg modification: {diag.get('avg_logit_modification', 0):.4f}")
                print(f"      Max modification: {diag.get('max_logit_modification', 0):.4f}")
                print(f"      Modified tokens: {diag.get('modified_token_count', 0)}")
        else:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⚠️ Generation returned error: {result.get('message')}")
            if "not loaded" in str(result.get('message', '')).lower():
                print("   💡 Model is loading for first time. This takes 2-3 minutes.")
                print("   Try again in a few minutes!")
    else:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Generation failed: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
except requests.exceptions.Timeout:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏱️ Request timed out")
    print("   The model might be loading (takes 2-3 minutes on first run)")
    print("   Try again in a few minutes!")
except Exception as e:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ❌ Generation error: {e}")

print("\n" + "=" * 60)
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 🎉 Test complete!")

print("\n📝 Next steps:")
print("1. If tests failed with timeouts, wait 2-3 minutes for model to load")
print("2. Copy the endpoints to Replit secrets:")
print(f"   MODAL_ENDPOINT: {generate_url}")
print("   MODAL_API_KEY: [Your Modal API key from modal.com/settings]")
print("3. Restart your Replit app to connect to the quantum model!")
print("\n💡 The first request always takes longer (cold start).")
print("   Subsequent requests will be much faster!")
```

## What this does:

1. **Health Check** - Tests if the container is running
2. **Generation Test** - Tests actual text generation with quantum modification
3. **Real-time timestamps** - Shows exactly when each step happens
4. **Better error handling** - Explains timeouts and cold starts
5. **Diagnostic info** - Shows quantum modification statistics if available

## Expected output on first run:

```
[10:15:30] 🧪 Testing deployed endpoints...
[10:15:30] 1. Testing health endpoint...
[10:15:32] Response code: 200
[10:15:32] ✅ Health check passed!
   Status: healthy
   Model: openai/gpt-oss-120b
   Quantum: true

[10:15:32] 2. Testing generation endpoint...
[10:15:32] Waiting for response (may take 30-60s if cold start)...
[10:16:05] Response code: 200
[10:16:05] ✅ Generation test passed!

   Generated text: 'The meaning of quantum consciousness is deeply intertwined with the fundamental nature of reality itself, suggesting that awareness emerges from quantum mechanical processes...'

   Tokens generated: 50
   Quantum profile used: medium

   📊 Quantum Diagnostics:
      Avg modification: 0.3125
      Max modification: 0.8921
      Modified tokens: 45
```

## If you get timeouts:
- This is normal on first run!
- The model takes 2-3 minutes to load initially
- Just wait and run Cell 7 again in a few minutes
- Once warm, responses take only a few seconds