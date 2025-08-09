#!/usr/bin/env python3
"""
Modal Deployment Troubleshooting Script
This helps diagnose common Modal deployment issues
"""

import os
import sys
import subprocess
import json

def check_modal_cli():
    """Check Modal CLI installation and version"""
    print("\n1️⃣  Checking Modal CLI...")
    try:
        result = subprocess.run(['modal', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Modal CLI installed: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Modal CLI error")
            return False
    except FileNotFoundError:
        print("   ❌ Modal CLI not found")
        print("   Run: pip install modal")
        return False

def check_modal_auth():
    """Check Modal authentication"""
    print("\n2️⃣  Checking Modal authentication...")
    try:
        result = subprocess.run(['modal', 'profile', 'current'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Authenticated as: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Not authenticated")
            print("   Run: modal setup")
            return False
    except:
        print("   ❌ Authentication check failed")
        return False

def check_modal_secrets():
    """Check Modal secrets"""
    print("\n3️⃣  Checking Modal secrets...")
    try:
        result = subprocess.run(['modal', 'secret', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            if 'qrng-api-key' in result.stdout:
                print("   ✅ QRNG secret exists")
                return True
            else:
                print("   ⚠️  QRNG secret not found")
                print("   Run: modal secret create qrng-api-key QRNG_API_KEY=your_key")
                return False
        else:
            print("   ❌ Failed to list secrets")
            return False
    except:
        print("   ❌ Secret check failed")
        return False

def check_deployment_file():
    """Check if deployment file exists and is valid"""
    print("\n4️⃣  Checking deployment file...")
    
    deployment_file = 'modal_deployment_notebook.py'
    if not os.path.exists(deployment_file):
        print(f"   ❌ {deployment_file} not found")
        print("   Make sure you're in the deployment directory")
        return False
    
    # Check file syntax
    try:
        with open(deployment_file, 'r') as f:
            code = f.read()
            compile(code, deployment_file, 'exec')
        print(f"   ✅ {deployment_file} syntax is valid")
        return True
    except SyntaxError as e:
        print(f"   ❌ Syntax error in {deployment_file}:")
        print(f"      Line {e.lineno}: {e.msg}")
        return False

def check_modal_apps():
    """Check existing Modal apps"""
    print("\n5️⃣  Checking existing Modal apps...")
    try:
        result = subprocess.run(['modal', 'app', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            if 'gaia-quantum-gpt-oss-120b' in result.stdout:
                print("   ✅ App 'gaia-quantum-gpt-oss-120b' exists")
                print("   You can view it at: https://modal.com/apps")
                return True
            else:
                print("   ℹ️  App 'gaia-quantum-gpt-oss-120b' not deployed yet")
                return None
        else:
            print("   ❌ Failed to list apps")
            return False
    except:
        print("   ❌ App check failed")
        return False

def test_simple_deployment():
    """Test with a simple Modal function"""
    print("\n6️⃣  Testing simple Modal deployment...")
    
    test_code = '''
import modal

app = modal.App("test-deployment")

@app.function()
def hello():
    return "Modal is working!"

@app.local_entrypoint()
def main():
    result = hello.remote()
    print(f"Test result: {result}")
'''
    
    # Write test file
    with open('test_modal.py', 'w') as f:
        f.write(test_code)
    
    print("   Running test deployment...")
    result = subprocess.run(
        ['modal', 'run', 'test_modal.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Clean up
    os.remove('test_modal.py')
    
    if result.returncode == 0 and "Modal is working!" in result.stdout:
        print("   ✅ Simple deployment works!")
        return True
    else:
        print("   ❌ Simple deployment failed")
        if result.stderr:
            print(f"   Error: {result.stderr[:200]}")
        return False

def main():
    """Main troubleshooting routine"""
    print("=" * 60)
    print("🔧 Modal Deployment Troubleshooting")
    print("=" * 60)
    
    results = {
        'cli': check_modal_cli(),
        'auth': check_modal_auth(),
        'secrets': check_modal_secrets(),
        'file': check_deployment_file(),
        'apps': check_modal_apps(),
    }
    
    # Only test deployment if basics are working
    if results['cli'] and results['auth']:
        results['test'] = test_simple_deployment()
    else:
        results['test'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TROUBLESHOOTING SUMMARY")
    print("=" * 60)
    
    all_good = all(v for v in results.values() if v is not None)
    
    if all_good:
        print("\n✅ Everything looks good! You can deploy with:")
        print("   modal deploy modal_deployment_notebook.py")
    else:
        print("\n⚠️  Issues found:")
        
        if not results['cli']:
            print("\n1. Install Modal:")
            print("   pip install modal")
        
        if results['cli'] and not results['auth']:
            print("\n2. Authenticate Modal:")
            print("   modal setup")
        
        if results['auth'] and not results['secrets']:
            print("\n3. Create QRNG secret:")
            print("   modal secret create qrng-api-key QRNG_API_KEY=your_key_here")
        
        if not results['file']:
            print("\n4. Fix deployment file issues or ensure you're in the right directory")
        
        if results['test'] is False:
            print("\n5. Modal deployment test failed. Check:")
            print("   - Internet connection")
            print("   - Modal service status: https://status.modal.com")
            print("   - Your Modal account: https://modal.com/settings")
    
    print("\n" + "=" * 60)
    print("For more help, visit: https://modal.com/docs")
    print("=" * 60)

if __name__ == "__main__":
    main()