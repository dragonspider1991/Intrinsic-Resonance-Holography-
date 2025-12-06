#!/usr/bin/env python3
"""
Test script to verify backend can start
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python/src'))

print("Testing backend imports...")

try:
    print("1. Testing FastAPI import...")
    import fastapi
    print(f"   ✓ FastAPI {fastapi.__version__}")
    
    print("2. Testing Uvicorn import...")
    import uvicorn
    print(f"   ✓ Uvicorn {uvicorn.__version__}")
    
    print("3. Testing IRH imports...")
    from irh.graph_state import HyperGraph
    print("   ✓ IRH graph_state")
    
    from irh.spectral_dimension import SpectralDimension
    print("   ✓ IRH spectral_dimension")
    
    from irh.predictions.constants import predict_alpha_inverse
    print("   ✓ IRH predictions")
    
    print("4. Testing backend modules...")
    sys.path.insert(0, os.path.dirname(__file__))
    from visualization import serialize_network_3d
    print("   ✓ visualization module")
    
    from integration import IRHSimulation
    print("   ✓ integration module")
    
    print("\n" + "="*50)
    print("✓ All imports successful!")
    print("Backend is ready to run.")
    print("="*50)
    print("\nTo start the backend server, run:")
    print("  cd webapp/backend")
    print("  python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000")
    print("\nOr use the startup script:")
    print("  cd webapp")
    print("  ./start_backend.sh")
    
except ImportError as e:
    print(f"\n✗ Import failed: {e}")
    print("\nPlease install missing dependencies:")
    print("  pip3 install -e . (from repo root)")
    print("  pip3 install -r requirements.txt (from webapp/backend)")
    sys.exit(1)
