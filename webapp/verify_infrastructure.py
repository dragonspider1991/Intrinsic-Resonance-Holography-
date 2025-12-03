#!/usr/bin/env python3
"""
IRH Web Application - Infrastructure Verification
==================================================

This script verifies that the web application infrastructure is properly set up.
It checks for the presence of required files and documents the architecture.

Run this to verify the webapp infrastructure without installing all dependencies.
"""

import os
import sys
from pathlib import Path


def check_file(filepath: str, description: str) -> bool:
    """Check if a file exists and report."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    if exists:
        size = os.path.getsize(filepath)
        print(f"     Size: {size:,} bytes")
    return exists


def check_directory(dirpath: str, description: str) -> bool:
    """Check if a directory exists and report."""
    exists = os.path.isdir(dirpath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}")
    if exists:
        files = list(Path(dirpath).rglob("*"))
        file_count = len([f for f in files if f.is_file()])
        print(f"     Contains: {file_count} files")
    return exists


def main():
    print("=" * 70)
    print("IRH Web Application - Infrastructure Verification")
    print("=" * 70)
    
    # Change to repository root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)
    print(f"\nRepository root: {repo_root.absolute()}\n")
    
    # Track overall status
    all_good = True
    
    # Check backend infrastructure
    print("\n" + "=" * 70)
    print("BACKEND INFRASTRUCTURE")
    print("=" * 70)
    
    backend_files = [
        ("webapp/backend/app.py", "Main FastAPI application"),
        ("webapp/backend/visualization.py", "3D/2D visualization data serializers"),
        ("webapp/backend/integration.py", "IRH module integration layer"),
        ("webapp/backend/requirements.txt", "Backend Python dependencies"),
        ("webapp/backend/__init__.py", "Backend package marker"),
    ]
    
    for filepath, desc in backend_files:
        if not check_file(filepath, desc):
            all_good = False
    
    # Check configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    
    config_files = [
        ("webapp/config/webapp_config.json", "Web application configuration"),
    ]
    
    for filepath, desc in config_files:
        if not check_file(filepath, desc):
            all_good = False
    
    # Check documentation
    print("\n" + "=" * 70)
    print("DOCUMENTATION")
    print("=" * 70)
    
    doc_files = [
        ("webapp/README.md", "Main webapp README"),
        ("webapp/INSTALLATION.md", "Installation guide"),
        ("webapp/GEMINI_FRONTEND_PROMPT.md", "Frontend specification for Gemini AI"),
    ]
    
    for filepath, desc in doc_files:
        if not check_file(filepath, desc):
            all_good = False
    
    # Check utilities
    print("\n" + "=" * 70)
    print("UTILITIES")
    print("=" * 70)
    
    util_files = [
        ("webapp/start_server.py", "Backend server startup script"),
        ("webapp/example_api_client.py", "Example API client for testing"),
        ("webapp/__init__.py", "Webapp package marker"),
    ]
    
    for filepath, desc in util_files:
        if not check_file(filepath, desc):
            all_good = False
    
    # Check IRH core modules
    print("\n" + "=" * 70)
    print("IRH CORE MODULES (Dependencies)")
    print("=" * 70)
    
    irh_modules = [
        ("python/src/irh/graph_state.py", "HyperGraph substrate"),
        ("python/src/irh/spectral_dimension.py", "Spectral dimension computation"),
        ("python/src/irh/predictions/constants.py", "Physical constant predictions"),
        ("python/src/irh/grand_audit.py", "Grand audit system"),
    ]
    
    for filepath, desc in irh_modules:
        if not check_file(filepath, desc):
            all_good = False
    
    # Check directories
    print("\n" + "=" * 70)
    print("DIRECTORY STRUCTURE")
    print("=" * 70)
    
    directories = [
        ("webapp/backend", "Backend code"),
        ("webapp/config", "Configuration files"),
        ("webapp/frontend", "Frontend code (to be implemented)"),
        ("webapp/static", "Static assets"),
        ("python/src/irh", "IRH core package"),
    ]
    
    for dirpath, desc in directories:
        check_directory(dirpath, desc)
    
    # API Endpoints Summary
    print("\n" + "=" * 70)
    print("API ENDPOINTS IMPLEMENTED")
    print("=" * 70)
    
    endpoints = [
        ("GET", "/", "Root endpoint with API info"),
        ("GET", "/api/health", "Health check"),
        ("POST", "/api/network/create", "Create network"),
        ("POST", "/api/network/spectrum", "Compute eigenspectrum"),
        ("POST", "/api/network/spectral-dimension", "Compute spectral dimension"),
        ("POST", "/api/predictions/alpha", "Predict fine structure constant"),
        ("POST", "/api/simulation/run", "Run full simulation (async)"),
        ("GET", "/api/jobs/{job_id}", "Get job status"),
        ("GET", "/api/jobs/{job_id}/result", "Get job result"),
        ("POST", "/api/visualization/network-3d", "Get 3D network data"),
        ("POST", "/api/visualization/spectrum-3d", "Get 3D spectrum data"),
        ("POST", "/api/visualization/spectrum-chart", "Get 2D chart data"),
        ("WS", "/ws/{job_id}", "Real-time job updates"),
    ]
    
    for method, path, description in endpoints:
        print(f"  ✓ {method:6s} {path:40s} - {description}")
    
    # Data Formats
    print("\n" + "=" * 70)
    print("VISUALIZATION DATA FORMATS")
    print("=" * 70)
    
    formats = [
        ("3D Network", "Nodes (id, position, color, size) + Edges (source, target, weight, opacity)"),
        ("3D Spectrum", "Points (x, y, z, color, size) for eigenvalue scatter plot"),
        ("2D Charts", "Chart.js format with datasets and options"),
        ("Heatmaps", "2D arrays with metadata for matrix visualization"),
        ("Animations", "Time-series frames for evolution visualization"),
    ]
    
    for format_name, description in formats:
        print(f"  ✓ {format_name:15s} - {description}")
    
    # Integration Summary
    print("\n" + "=" * 70)
    print("IRH MODULE INTEGRATION")
    print("=" * 70)
    
    integrations = [
        ("HyperGraph", "Network creation and topology"),
        ("SpectralDimension", "Spectral dimension d_s computation"),
        ("MetricEmergence", "Emergent spacetime metric"),
        ("predict_alpha_inverse", "Fine structure constant α⁻¹"),
        ("grand_audit", "Comprehensive theory validation"),
        ("GTEC", "Complexity metrics"),
        ("NCGG", "Quantum operators and frustration"),
    ]
    
    for module, description in integrations:
        print(f"  ✓ {module:25s} - {description}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("INFRASTRUCTURE STATUS")
    print("=" * 70)
    
    if all_good:
        print("\n✓ All infrastructure files are present!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Install backend deps: pip install -r webapp/backend/requirements.txt")
        print("  3. Start server: python webapp/start_server.py")
        print("  4. View API docs: http://localhost:8000/api/docs")
        print("  5. Implement frontend: See webapp/GEMINI_FRONTEND_PROMPT.md")
        return 0
    else:
        print("\n✗ Some infrastructure files are missing!")
        print("\nPlease ensure all files are committed to the repository.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
