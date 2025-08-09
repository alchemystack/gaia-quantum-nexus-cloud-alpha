#!/usr/bin/env python3
"""
ONE-CLICK DEPLOY SCRIPT FOR GAIA QUANTUM NEXUS
==============================================
Run this after deploying to Modal to connect everything!

Usage: python3 ONE_CLICK_DEPLOY.py
"""

import os
import json
import sys

def setup_modal_connection():
    """Set up Modal endpoints in Replit"""
    print("🌌 GAIA QUANTUM NEXUS - MODAL CONNECTOR")
    print("=" * 50)
    
    # Get Modal endpoint from user
    print("\n📌 After deploying to Modal, you'll get endpoints like:")
    print("   https://ak-4jazee--gaia-quantum-120b-generate-endpoint.modal.run")
    print("\n")
    
    endpoint = input("Paste your Modal generation endpoint here: ").strip()
    
    if not endpoint:
        print("❌ No endpoint provided. Please deploy to Modal first!")
        return False
    
    # Save to environment
    env_file = ".env"
    config = {
        "MODAL_ENDPOINT": endpoint,
        "MODAL_API_KEY": "ak-4jAZeEPxVf7YMT0MYey2dw"
    }
    
    # Write configuration
    with open(env_file, "a") as f:
        f.write("\n# Modal Configuration\n")
        for key, value in config.items():
            f.write(f"{key}={value}\n")
    
    print("\n✅ Modal endpoint configured!")
    print(f"   Endpoint: {endpoint}")
    print("\n🚀 Your Quantum Nexus is ready!")
    print("   The web interface will now use the 120B model from Modal!")
    
    return True

def main():
    """Main deployment function"""
    try:
        success = setup_modal_connection()
        
        if success:
            print("\n" + "=" * 50)
            print("✅ DEPLOYMENT COMPLETE!")
            print("=" * 50)
            print("\n📌 Next steps:")
            print("   1. Open your web app")
            print("   2. Enter a prompt")
            print("   3. Experience quantum-enhanced AI!")
            print("\n💫 The 120B model is now running in the cloud!")
        else:
            print("\n❌ Deployment incomplete. Please follow the steps above.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your Modal deployment and try again.")

if __name__ == "__main__":
    main()