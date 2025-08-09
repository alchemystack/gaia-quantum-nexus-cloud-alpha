#!/usr/bin/env python3
"""
Simplified Modal Deployment Script for Gaia Quantum Nexus
Run this script to deploy the GPT-OSS 120B model to Modal
"""

import os
import sys
import subprocess

def check_modal_installed():
    """Check if Modal is installed"""
    try:
        result = subprocess.run(['modal', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Modal CLI is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Modal CLI not found. Installing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'modal'], check=True)
    return True

def check_modal_auth():
    """Check if Modal is authenticated"""
    try:
        result = subprocess.run(['modal', 'profile', 'current'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Modal is authenticated")
            return True
    except:
        pass
    
    print("⚠️  Modal not authenticated. Running setup...")
    subprocess.run(['modal', 'setup'], check=True)
    return True

def create_qrng_secret():
    """Create or update QRNG secret in Modal"""
    qrng_key = os.environ.get('QRNG_API_KEY')
    
    if not qrng_key:
        print("\n⚠️  QRNG_API_KEY not found in environment")
        print("Please enter your Quantum Blockchains API key:")
        qrng_key = input("QRNG_API_KEY: ").strip()
        
        if not qrng_key:
            print("❌ QRNG API key is required for quantum operations")
            return False
    
    print("📝 Creating Modal secret for QRNG...")
    result = subprocess.run(
        ['modal', 'secret', 'create', 'qrng-api-key', f'QRNG_API_KEY={qrng_key}'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 or 'already exists' in result.stderr:
        print("✅ QRNG secret configured")
        return True
    else:
        print(f"❌ Failed to create secret: {result.stderr}")
        return False

def deploy_model():
    """Deploy the model to Modal"""
    print("\n🚀 Deploying GPT-OSS 120B to Modal...")
    print("=" * 60)
    
    # Check if deployment file exists
    deployment_file = 'modal_deployment_notebook.py'
    if not os.path.exists(deployment_file):
        print(f"❌ {deployment_file} not found in current directory")
        print("Please run this script from the deployment directory")
        return False
    
    print("\n📊 Deployment Configuration:")
    print("  • Model: bartowski/openai_gpt-oss-120b-GGUF-MXFP4")
    print("  • GPU: 2x NVIDIA A100 80GB")
    print("  • Quantum: QRNG logit modification")
    print("  • Cost: ~$95/month (light use)")
    print("")
    
    response = input("Continue with deployment? (y/n): ")
    if response.lower() != 'y':
        print("Deployment cancelled")
        return False
    
    print("\n🔄 Running deployment...")
    result = subprocess.run(
        ['modal', 'deploy', deployment_file],
        text=True
    )
    
    if result.returncode == 0:
        print("\n✅ Deployment successful!")
        return True
    else:
        print("\n❌ Deployment failed")
        return False

def show_next_steps():
    """Show next steps after deployment"""
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)
    
    print("\n1️⃣  Get your endpoint URLs from Modal dashboard:")
    print("   https://modal.com/apps")
    print("   Look for 'gaia-quantum-gpt-oss-120b'")
    
    print("\n2️⃣  Your endpoints will be:")
    print("   • Generation: https://YOUR-USERNAME--gaia-quantum-gpt-oss-120b-generate-endpoint.modal.run")
    print("   • Health: https://YOUR-USERNAME--gaia-quantum-gpt-oss-120b-health-check.modal.run")
    
    print("\n3️⃣  Add to Replit Secrets:")
    print("   • MODAL_ENDPOINT = [your generation endpoint URL]")
    print("   • MODAL_API_KEY = [your Modal token from https://modal.com/settings/tokens]")
    
    print("\n4️⃣  Test the health endpoint:")
    print("   Open the health URL in your browser")
    print("   You should see: {\"status\": \"healthy\", \"model\": \"GPT-OSS 120B\"}")
    
    print("\n5️⃣  Restart your Replit app")
    print("   The UI will show 'Modal GPT-OSS 120B' when connected")
    
    print("\n" + "=" * 60)
    print("✨ Your quantum-augmented 120B model is ready!")
    print("=" * 60)

def main():
    """Main deployment process"""
    print("=" * 60)
    print("🌌 Gaia Quantum Nexus - Modal Deployment")
    print("=" * 60)
    
    # Step 1: Check Modal installation
    if not check_modal_installed():
        return
    
    # Step 2: Check Modal authentication
    if not check_modal_auth():
        return
    
    # Step 3: Create QRNG secret
    if not create_qrng_secret():
        print("\n⚠️  Warning: Proceeding without QRNG will cause generation to fail")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 4: Deploy the model
    if deploy_model():
        show_next_steps()
    else:
        print("\n❌ Deployment failed. Check the error messages above.")
        print("For help, visit: https://modal.com/docs")

if __name__ == "__main__":
    main()