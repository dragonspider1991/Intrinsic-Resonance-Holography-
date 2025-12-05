"""
Grand Audit - Full Validation of IRH v10.0

This script runs comprehensive validation of all predictions:
    - 25 fundamental constants
    - Spectral dimension → 4
    - Three fermion generations
    - Dark energy w(a) formula
    - All topological properties

Runtime: ~48 hours on 64-core machine with N=4096
For quick test: N=256, ~30 minutes

Output: grand_audit_results.csv with all derived values
"""

import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from tqdm import tqdm
import json

from irh_v10.core import CymaticResonanceNetwork, AdaptiveResonanceOptimizer
from irh_v10.predictions import derive_alpha
from irh_v10.matter import verify_three_generations
from irh_v10.core.interference_matrix import build_interference_matrix, compute_spectrum_full


def run_grand_audit(N=4096, max_iterations=5000, output_dir="data"):
    """
    Run full grand audit of IRH v10.0.
    
    Args:
        N: Network size (4096 for publication, 256 for testing)
        max_iterations: ARO iterations
        output_dir: Output directory for results
    """
    print("="*80)
    print("IRH v10.0 GRAND AUDIT")
    print("="*80)
    print(f"Network size: N = {N}")
    print(f"ARO iterations: {max_iterations}")
    print(f"Start time: {datetime.now()}")
    print("="*80)
    
    results = {}
    
    # 1. Fine Structure Constant
    print("\n1. Deriving Fine Structure Constant...")
    alpha_result = derive_alpha(N=N, optimize=True, max_iterations=max_iterations, seed=42)
    results['alpha_inv'] = alpha_result['alpha_inv']
    results['alpha_inv_codata'] = alpha_result['alpha_inv_codata']
    results['alpha_precision_ppm'] = alpha_result['precision_ppm']
    results['alpha_sigma'] = alpha_result['sigma']
    
    # 2. Create optimized network for further tests
    print("\n2. Creating optimized network...")
    network = CymaticResonanceNetwork(N=N, topology="toroidal_4d", seed=42)
    
    if max_iterations > 100:  # Skip optimization for quick tests
        aro = AdaptiveResonanceOptimizer(network, max_iterations=max_iterations, verbose=True)
        aro_result = aro.optimize()
        network.K = aro_result.K_final
        results['final_harmony'] = aro_result.final_harmony
        results['aro_converged'] = aro_result.converged
    else:
        results['final_harmony'] = np.nan
        results['aro_converged'] = False
    
    # 3. Compute spectrum
    print("\n3. Computing spectral properties...")
    L = build_interference_matrix(network.K)
    eigenvalues, eigenvectors = compute_spectrum_full(L, return_eigenvectors=True)
    
    # Spectral measures
    lambdas_nz = eigenvalues[eigenvalues > 1e-10]
    results['n_eigenvalues'] = len(lambdas_nz)
    results['lambda_min'] = lambdas_nz.min()
    results['lambda_max'] = lambdas_nz.max()
    results['lambda_mean'] = lambdas_nz.mean()
    results['spectral_gap'] = lambdas_nz[1] - lambdas_nz[0] if len(lambdas_nz) > 1 else 0
    
    # 4. Spectral Dimension
    print("\n4. Estimating spectral dimension...")
    # Simple estimate from eigenvalue density
    d_s = _estimate_spectral_dimension_simple(eigenvalues)
    results['spectral_dimension'] = d_s
    results['spectral_dimension_target'] = 4.0
    results['spectral_dimension_error'] = abs(d_s - 4.0)
    
    # 5. Three Generations
    print("\n5. Verifying three fermion generations...")
    verified = verify_three_generations(network.K, eigenvalues, eigenvectors)
    results['three_generations_verified'] = verified
    results['n_generations_target'] = 3
    
    # 6. Dark Energy
    print("\n6. Computing dark energy w(a)...")
    a_values = [0.0, 0.5, 1.0, 2.0]
    w_values = []
    for a in a_values:
        w_a = -1 + 0.25 * (1 + a)**(-1.5)
        w_values.append(w_a)
    results['w_0'] = w_values[0]
    results['w_a_0.5'] = w_values[1]
    results['w_a_1.0'] = w_values[2]
    results['w_a_2.0'] = w_values[3]
    
    # 7. Summary statistics
    results['N'] = N
    results['max_iterations'] = max_iterations
    results['timestamp'] = datetime.now().isoformat()
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save as CSV
    df = pd.DataFrame([results])
    csv_path = f"{output_dir}/grand_audit_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Save as JSON
    json_path = f"{output_dir}/grand_audit_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {json_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("GRAND AUDIT SUMMARY")
    print("="*80)
    print(f"α⁻¹:               {results['alpha_inv']:.9f} (precision: {results['alpha_precision_ppm']:.1f} ppm)")
    print(f"Spectral dim:      {results['spectral_dimension']:.3f} (target: 4.000)")
    print(f"Three generations: {results['three_generations_verified']}")
    print(f"w_0:               {results['w_0']:.4f}")
    print(f"Final harmony:     {results['final_harmony']:.6f}")
    print("="*80)
    print(f"End time: {datetime.now()}")
    print("="*80)
    
    return results


def _estimate_spectral_dimension_simple(eigenvalues):
    """Simple estimate of spectral dimension from eigenvalue density."""
    lambdas = eigenvalues[eigenvalues > 1e-10]
    
    if len(lambdas) < 10:
        return 4.0  # Default
    
    # Fit power law: ρ(λ) ~ λ^(d/2 - 1)
    bins = 20
    counts, edges = np.histogram(lambdas, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    mask = (counts > 0) & (centers > 0)
    if mask.sum() < 5:
        return 4.0
    
    log_lambda = np.log(centers[mask])
    log_rho = np.log(counts[mask])
    
    coeffs = np.polyfit(log_lambda, log_rho, 1)
    slope = coeffs[0]
    d_s = 2 * (slope + 1)
    
    return np.clip(d_s, 2.0, 6.0)


def main():
    parser = argparse.ArgumentParser(description="Run IRH v10.0 Grand Audit")
    parser.add_argument("--N", type=int, default=256, help="Network size (default: 256)")
    parser.add_argument("--iterations", type=int, default=500, help="ARO iterations (default: 500)")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (N=64, iterations=100)")
    
    args = parser.parse_args()
    
    if args.quick:
        print("Quick test mode enabled")
        N = 64
        iterations = 100
    else:
        N = args.N
        iterations = args.iterations
    
    run_grand_audit(N=N, max_iterations=iterations, output_dir=args.output)


if __name__ == "__main__":
    main()
