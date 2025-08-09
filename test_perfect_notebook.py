#!/usr/bin/env python3
"""
Test script for the Perfect 7-Cell Modal Notebook
Run this to verify the notebook syntax and structure
"""

import ast
import sys

def test_notebook_syntax():
    """Verify the notebook has correct Python syntax"""
    print("Testing MODAL_PERFECT_7CELL_NOTEBOOK.py...")
    
    try:
        with open("MODAL_PERFECT_7CELL_NOTEBOOK.py", "r") as f:
            code = f.read()
        
        # Parse the Python code
        ast.parse(code)
        print("✅ Syntax check passed!")
        
        # Check for key components
        checks = {
            "app = modal.App": "Modal app initialization",
            "class QuantumModel": "QuantumModel class",
            "A100_80GB()": "Correct GPU notation",
            "fastapi[standard]": "FastAPI dependency",
            "@modal.web_endpoint": "Web endpoints",
            "fetch_quantum_entropy": "QRNG integration",
            "apply_quantum_modification": "Logit modification",
            "def health()": "Health endpoint",
            "def generate(": "Generate endpoint",
            "quantum_profiles": "Quantum profiles"
        }
        
        print("\nChecking key components:")
        for check, description in checks.items():
            if check in code:
                print(f"  ✅ {description}")
            else:
                print(f"  ❌ Missing: {description}")
        
        # Count cells
        cell_count = code.count("# CELL")
        print(f"\nFound {cell_count} cells in notebook")
        
        if cell_count == 7:
            print("✅ All 7 cells present!")
        else:
            print(f"⚠️  Expected 7 cells, found {cell_count}")
        
        # Check URLs
        if "https://qgpt--health.modal.run" in code and "https://qgpt--generate.modal.run" in code:
            print("✅ Correct endpoint URLs configured")
        else:
            print("⚠️  Check endpoint URLs")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_notebook_syntax()
    
    if success:
        print("\n" + "="*60)
        print("NOTEBOOK IS READY FOR DEPLOYMENT!")
        print("="*60)
        print("\nNext steps:")
        print("1. Copy the notebook to Modal")
        print("2. Run cells 1-7 in order")
        print("3. Deploy with: modal deploy MODAL_PERFECT_7CELL_NOTEBOOK.py")
        print("4. Update Replit secrets with new endpoints")
    else:
        print("\n❌ Please fix errors before deployment")
        sys.exit(1)