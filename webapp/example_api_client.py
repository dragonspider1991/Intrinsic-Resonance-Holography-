#!/usr/bin/env python3
"""
Example API Client for IRH Web Application
===========================================

Demonstrates how to interact with the IRH backend API using Python.
This script shows:
- Creating networks with different configurations
- Running simulations
- Fetching results
- Getting visualization data

Run the backend server first: python webapp/start_server.py
"""

import requests
import json
import time
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"


def create_network(N: int = 64, topology: str = "Random", seed: int = 42) -> Dict[str, Any]:
    """Create a network and get basic properties."""
    print(f"\n{'='*60}")
    print(f"Creating network: N={N}, topology={topology}, seed={seed}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/api/network/create",
        json={
            "N": N,
            "topology": topology,
            "seed": seed,
            "edge_probability": 0.3,
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Network created successfully!")
        print(f"  Nodes: {data['N']}")
        print(f"  Edges: {data['edge_count']}")
        print(f"  Spectrum: λ_min={data['spectrum']['min']:.4f}, λ_max={data['spectrum']['max']:.4f}")
        return data
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return {}


def compute_spectral_dimension(N: int = 64, topology: str = "Lattice") -> Dict[str, Any]:
    """Compute spectral dimension of a network."""
    print(f"\n{'='*60}")
    print(f"Computing spectral dimension: N={N}, topology={topology}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/api/network/spectral-dimension",
        json={
            "N": N,
            "topology": topology,
            "seed": 42,
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        ds = data.get('spectral_dimension')
        if ds is not None:
            print(f"✓ Spectral dimension: d_s = {ds:.4f}")
            print(f"  Target: d_s = 4.0")
            print(f"  Match: {'✓ Yes' if abs(ds - 4.0) < 0.5 else '✗ No'}")
        else:
            print("✗ Could not compute spectral dimension (network too small?)")
        return data
    else:
        print(f"✗ Error: {response.status_code}")
        return {}


def predict_alpha(N: int = 256, topology: str = "Lattice") -> Dict[str, Any]:
    """Predict fine structure constant."""
    print(f"\n{'='*60}")
    print(f"Predicting fine structure constant: N={N}, topology={topology}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/api/predictions/alpha",
        json={
            "N": N,
            "topology": topology,
            "seed": 42,
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        alpha_inv = data['alpha_inverse']
        codata = data['codata_value']
        diff = data['difference']
        
        print(f"✓ Fine structure constant prediction:")
        print(f"  α⁻¹ (predicted): {alpha_inv:.9f}")
        print(f"  α⁻¹ (CODATA):    {codata:.9f}")
        print(f"  Difference:      {diff:.9f}")
        print(f"  Relative error:  {abs(diff/codata)*100:.4f}%")
        
        if abs(diff) < 1.0:
            print(f"  Status: ✓ Excellent agreement!")
        elif abs(diff) < 10.0:
            print(f"  Status: ✓ Good agreement")
        else:
            print(f"  Status: ⚠ Needs larger network or optimization")
        
        return data
    else:
        print(f"✗ Error: {response.status_code}")
        return {}


def run_simulation(N: int = 128, topology: str = "Random") -> str:
    """Run a full simulation and return job_id."""
    print(f"\n{'='*60}")
    print(f"Running full simulation: N={N}, topology={topology}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/api/simulation/run",
        json={
            "network_config": {
                "N": N,
                "topology": topology,
                "seed": 42,
                "edge_probability": 0.3,
            },
            "compute_spectral_dimension": True,
            "compute_predictions": True,
            "run_grand_audit": False,
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        job_id = data['job_id']
        print(f"✓ Simulation started!")
        print(f"  Job ID: {job_id}")
        return job_id
    else:
        print(f"✗ Error: {response.status_code}")
        return ""


def poll_job_status(job_id: str, timeout: int = 60) -> Dict[str, Any]:
    """Poll job status until completion or timeout."""
    print(f"\n{'='*60}")
    print(f"Polling job status: {job_id}")
    print(f"{'='*60}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(f"{BASE_URL}/api/jobs/{job_id}")
        
        if response.status_code == 200:
            data = response.json()
            status = data['status']
            progress = data['progress']
            
            print(f"\r  Status: {status:10s} | Progress: {progress:5.1f}%", end='', flush=True)
            
            if status == 'completed':
                print("\n✓ Simulation completed!")
                return data
            elif status == 'failed':
                print(f"\n✗ Simulation failed: {data.get('error', 'Unknown error')}")
                return data
            
            time.sleep(1)
        else:
            print(f"\n✗ Error polling status: {response.status_code}")
            return {}
    
    print(f"\n⚠ Timeout after {timeout} seconds")
    return {}


def get_job_result(job_id: str) -> Dict[str, Any]:
    """Get full result of a completed job."""
    print(f"\n{'='*60}")
    print(f"Fetching job result: {job_id}")
    print(f"{'='*60}")
    
    response = requests.get(f"{BASE_URL}/api/jobs/{job_id}/result")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Results retrieved!")
        
        # Display key results
        if 'network' in data:
            net = data['network']
            print(f"\nNetwork:")
            print(f"  N = {net['N']}, edges = {net['edge_count']}")
        
        if 'spectral_dimension' in data:
            sd = data['spectral_dimension']
            if sd.get('value') is not None:
                print(f"\nSpectral Dimension:")
                print(f"  d_s = {sd['value']:.4f}")
        
        if 'predictions' in data:
            pred = data['predictions']
            if 'alpha_inverse' in pred:
                alpha_inv = pred['alpha_inverse']
                print(f"\nFine Structure Constant:")
                print(f"  α⁻¹ = {alpha_inv:.9f}")
        
        return data
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
        return {}


def get_3d_visualization(N: int = 64, topology: str = "Random") -> Dict[str, Any]:
    """Get 3D visualization data."""
    print(f"\n{'='*60}")
    print(f"Fetching 3D visualization: N={N}, topology={topology}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/api/visualization/network-3d",
        json={
            "N": N,
            "topology": topology,
            "seed": 42,
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ 3D data retrieved!")
        print(f"  Nodes: {len(data['nodes'])}")
        print(f"  Edges: {len(data['edges'])}")
        print(f"  Metadata: {data['metadata']}")
        
        # Show sample node and edge
        if data['nodes']:
            node = data['nodes'][0]
            print(f"\nSample node: {json.dumps(node, indent=2)}")
        
        if data['edges']:
            edge = data['edges'][0]
            print(f"\nSample edge: {json.dumps(edge, indent=2)}")
        
        return data
    else:
        print(f"✗ Error: {response.status_code}")
        return {}


def main():
    """Run example demonstrations."""
    print("\n" + "="*70)
    print("IRH Web API - Example Client")
    print("="*70)
    print("\nMake sure the backend server is running:")
    print("  python webapp/start_server.py")
    print("\nThen run this script to test the API.")
    print("="*70)
    
    # Test 1: Create simple network
    create_network(N=64, topology="Random", seed=42)
    
    # Test 2: Compute spectral dimension
    compute_spectral_dimension(N=64, topology="Lattice")
    
    # Test 3: Predict alpha
    predict_alpha(N=128, topology="Lattice")
    
    # Test 4: Run full simulation
    job_id = run_simulation(N=128, topology="Random")
    
    if job_id:
        # Poll for completion
        job_status = poll_job_status(job_id, timeout=60)
        
        if job_status.get('status') == 'completed':
            # Get full results
            get_job_result(job_id)
    
    # Test 5: Get 3D visualization data
    get_3d_visualization(N=32, topology="Cycle")
    
    print("\n" + "="*70)
    print("Example completed!")
    print("="*70)


if __name__ == "__main__":
    main()
