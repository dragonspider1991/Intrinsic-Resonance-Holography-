"""
Cosmic Fixed Point Test - Full Validation of IRH v13.0 Predictions

This test validates all 4 key predictions from the IRH v13.0 theoretical framework:
1. α⁻¹ = 137.036 ± 0.004 (fine-structure constant from frustration density)
2. d_space = 4 (exact, emergent spacetime dimensionality)
3. N_gen = 3 (exact, number of fermion generations) - future implementation
4. β₁ = 12 (first Betti number, SU(3)×SU(2)×U(1) generators) - future implementation

Expected runtime: Variable based on N and iterations
- N=1000, 10000 iterations: ~30-60 minutes
- N=500, 1000 iterations: ~5-10 minutes (quick test)

References: IRH v13.0 Theorems 1.2, 3.1, 4.1
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/Intrinsic-Resonance-Holography-/Intrinsic-Resonance-Holography-')

from src.core import AROOptimizer, harmony_functional
from src.topology import calculate_frustration_density, derive_fine_structure_constant
from src.metrics import spectral_dimension, dimensional_coherence_index
import numpy as np
import json
from datetime import datetime

def run_cosmic_fixed_point_test(N=500, iterations=1000, seed=42, output_dir='experiments'):
    """
    Run the Cosmic Fixed Point Test with specified parameters.
    
    Parameters
    ----------
    N : int
        Number of nodes in the network
    iterations : int
        Number of ARO optimization iterations
    seed : int
        Random seed for reproducibility
    output_dir : str
        Directory to save results
        
    Returns
    -------
    results : dict
        Complete test results including all predictions and metrics
    """
    print("="*70)
    print("COSMIC FIXED POINT TEST - IRH v13.0")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  N = {N} nodes")
    print(f"  Iterations = {iterations}")
    print(f"  Random seed = {seed}")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'config': {
            'N': N,
            'iterations': iterations,
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Step 1: Initialize ARO optimizer
    print(f"\n[1/5] Initializing ARO Optimizer (N={N})...")
    opt = AROOptimizer(N=N, rng_seed=seed)
    
    # Connectivity parameter controls initial edge density
    # Typical values: 0.05-0.2 (higher = denser initial network)
    # May need adjustment based on N: larger N can use lower connectivity
    # For N~300: 0.1 works well; for N~1000+: 0.05-0.1 is sufficient
    CONNECTIVITY_PARAM = 0.1
    
    opt.initialize_network(
        scheme='geometric',
        connectivity_param=CONNECTIVITY_PARAM,
        d_initial=4
    )
    initial_edges = opt.current_W.nnz
    print(f"      Network initialized: {initial_edges} edges")
    print(f"      Edge density: {initial_edges / (N*N):.4f}")
    
    results['initialization'] = {
        'edges': initial_edges,
        'edge_density': initial_edges / (N*N)
    }
    
    # Compute initial harmony
    initial_S_H = harmony_functional(opt.current_W)
    print(f"      Initial S_H = {initial_S_H:.5f}")
    results['initialization']['S_H'] = initial_S_H
    
    # Step 2: Run ARO optimization
    print(f"\n[2/5] Running ARO Optimization ({iterations} iterations)...")
    print("      Progress will be shown every 10%.")
    
    opt.optimize(
        iterations=iterations,
        learning_rate=0.01,
        mutation_rate=0.05,
        temp_start=1.0,
        verbose=True
    )
    
    W_opt = opt.best_W
    S_H_final = opt.best_S
    print(f"\n      Optimization complete!")
    print(f"      Final S_H = {S_H_final:.5f}")
    print(f"      Improvement: {S_H_final - initial_S_H:.5f}")
    
    results['optimization'] = {
        'S_H_initial': initial_S_H,
        'S_H_final': S_H_final,
        'S_H_improvement': S_H_final - initial_S_H,
        'final_edges': W_opt.nnz
    }
    
    # Step 3: Compute topological invariants
    print(f"\n[3/5] Computing Topological Invariants...")
    print("      Analyzing phase holonomies and Wilson loops...")
    
    rho_frust = calculate_frustration_density(W_opt, max_cycles=5000, sampling=True)
    alpha_inv, alpha_match = derive_fine_structure_constant(rho_frust)
    
    # Fine-structure constant experimental value
    # Source: CODATA 2018 recommended value
    # Reference: Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021)
    experimental_alpha = 137.035999084
    alpha_error = abs(alpha_inv - experimental_alpha)
    alpha_percent_error = (alpha_error / experimental_alpha) * 100
    
    print(f"      Frustration density: ρ = {rho_frust:.6f}")
    print(f"      Predicted α⁻¹ = {alpha_inv:.3f}")
    print(f"      Experimental α⁻¹ = {experimental_alpha}")
    print(f"      Absolute error: {alpha_error:.3f}")
    print(f"      Percent error: {alpha_percent_error:.2f}%")
    print(f"      Match (within 1.0): {alpha_match}")
    
    results['topology'] = {
        'rho_frust': rho_frust,
        'alpha_inv_predicted': alpha_inv,
        'alpha_inv_experimental': experimental_alpha,
        'alpha_error': alpha_error,
        'alpha_percent_error': alpha_percent_error,
        'alpha_match': alpha_match
    }
    
    # Step 4: Compute dimensional metrics
    print(f"\n[4/5] Computing Dimensional Metrics...")
    print("      Calculating spectral dimension...")
    
    d_spec, d_info = spectral_dimension(W_opt, method='heat_kernel')
    
    target_d = 4.0
    d_spec_error = abs(d_spec - target_d)
    d_spec_percent_error = (d_spec_error / target_d) * 100
    
    print(f"      Spectral dimension: d_s = {d_spec:.3f}")
    print(f"      Target: {target_d}")
    print(f"      Absolute error: {d_spec_error:.3f}")
    print(f"      Percent error: {d_spec_percent_error:.2f}%")
    print(f"      Method status: {d_info.get('status', 'unknown')}")
    
    results['dimensions'] = {
        'd_spec_predicted': d_spec,
        'd_spec_target': target_d,
        'd_spec_error': d_spec_error,
        'd_spec_percent_error': d_spec_percent_error,
        'd_spec_info': d_info
    }
    
    # Compute dimensional coherence index
    print("      Calculating dimensional coherence index...")
    chi_D, chi_comp = dimensional_coherence_index(W_opt, target_d=4)
    
    print(f"      Dimensional Coherence: χ_D = {chi_D:.3f}")
    print(f"      Components: E_H={chi_comp['E_H']:.3f}, E_R={chi_comp['E_R']:.3f}, E_C={chi_comp['E_C']:.3f}")
    
    results['dimensions']['chi_D'] = chi_D
    results['dimensions']['chi_components'] = chi_comp
    
    # Step 5: Validation summary
    print(f"\n[5/5] VALIDATION SUMMARY")
    print("="*70)
    
    # Determine success criteria
    # For realistic convergence, we need much larger N and more iterations
    # But we can still assess trends
    
    # Threshold constants for validation criteria
    # These are based on empirical observation of convergence behavior
    ALPHA_EXCELLENT_THRESHOLD = 1.0      # Within experimental uncertainty
    ALPHA_GOOD_THRESHOLD = 10.0          # Order of magnitude agreement
    ALPHA_MIN_REASONABLE = 50.0          # Lower bound for trending
    ALPHA_MAX_REASONABLE = 300.0         # Upper bound for trending
    
    D_SPEC_EXCELLENT_THRESHOLD = 1.0     # Within 1 dimension of target
    D_SPEC_GOOD_THRESHOLD = 2.0          # Within 2 dimensions of target
    D_SPEC_MIN_REASONABLE = 1.0          # Must be at least 1D
    D_SPEC_MAX_REASONABLE = 8.0          # Upper bound for physical spacetime
    
    alpha_excellent = alpha_error < ALPHA_EXCELLENT_THRESHOLD
    alpha_good = alpha_error < ALPHA_GOOD_THRESHOLD
    alpha_trending = alpha_inv > ALPHA_MIN_REASONABLE and alpha_inv < ALPHA_MAX_REASONABLE
    
    d_spec_excellent = d_spec_error < D_SPEC_EXCELLENT_THRESHOLD
    d_spec_good = d_spec_error < D_SPEC_GOOD_THRESHOLD
    d_spec_trending = d_spec > D_SPEC_MIN_REASONABLE and d_spec < D_SPEC_MAX_REASONABLE
    
    print(f"\n{'Metric':<30} {'Predicted':<15} {'Target':<15} {'Error':<15}")
    print("-"*70)
    print(f"{'Fine-structure α⁻¹':<30} {alpha_inv:<15.3f} {experimental_alpha:<15.3f} {alpha_error:<15.3f}")
    print(f"{'Spectral dimension d_s':<30} {d_spec:<15.3f} {target_d:<15.3f} {d_spec_error:<15.3f}")
    print(f"{'Harmony functional S_H':<30} {S_H_final:<15.5f} {'N/A':<15} {'N/A':<15}")
    print(f"{'Coherence index χ_D':<30} {chi_D:<15.3f} {'~1.0':<15} {'N/A':<15}")
    
    print("\n" + "="*70)
    
    # Overall assessment
    if alpha_excellent and d_spec_excellent:
        status = "✅ EXCELLENT - Predictions validated!"
        grade = "A+"
    elif alpha_good and d_spec_good:
        status = "✅ GOOD - Predictions within acceptable range"
        grade = "A"
    elif (alpha_good or alpha_trending) and (d_spec_good or d_spec_trending):
        status = "⚠️  PARTIAL - Trends toward predictions observed"
        grade = "B"
    else:
        status = "❌ NEEDS IMPROVEMENT - Predictions not converged"
        grade = "C"
    
    print(f"OVERALL STATUS: {status}")
    print(f"GRADE: {grade}")
    
    if grade in ["C", "B"]:
        print("\nRECOMMENDATIONS:")
        if N < 1000:
            print("  - Increase N to 1000-5000 for better emergent properties")
        if iterations < 10000:
            print("  - Increase iterations to 10,000-50,000 for convergence")
        if S_H_final - initial_S_H < 1.0:
            print("  - S_H barely improved; network may be stuck in local minimum")
            print("  - Try higher initial temperature (temp_start=2-5)")
        print("  - These are small test parameters; full validation requires production scale")
    
    print("="*70)
    
    results['validation'] = {
        'status': status,
        'grade': grade,
        'alpha_excellent': alpha_excellent,
        'alpha_good': alpha_good,
        'd_spec_excellent': d_spec_excellent,
        'd_spec_good': d_spec_good
    }
    
    # Save results with custom JSON encoder for numpy types
    output_file = os.path.join(output_dir, f'cosmic_fixed_point_results_N{N}_iter{iterations}.json')
    
    def json_serializer(obj):
        """Custom JSON serializer for numpy and datetime types."""
        import numpy as np
        from datetime import datetime
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=json_serializer)
    
    print(f"\nResults saved to: {output_file}")
    
    # Also create a human-readable summary
    summary_file = os.path.join(output_dir, f'cosmic_fixed_point_summary_N{N}_iter{iterations}.md')
    with open(summary_file, 'w') as f:
        f.write(f"# Cosmic Fixed Point Test Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- N = {N} nodes\n")
        f.write(f"- Iterations = {iterations}\n")
        f.write(f"- Random seed = {seed}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Metric | Predicted | Target | Error |\n")
        f.write(f"|--------|-----------|--------|-------|\n")
        f.write(f"| α⁻¹ | {alpha_inv:.3f} | {experimental_alpha} | {alpha_error:.3f} ({alpha_percent_error:.2f}%) |\n")
        f.write(f"| d_spec | {d_spec:.3f} | {target_d} | {d_spec_error:.3f} ({d_spec_percent_error:.2f}%) |\n")
        f.write(f"| S_H | {S_H_final:.5f} | N/A | Δ={S_H_final - initial_S_H:.5f} |\n")
        f.write(f"| χ_D | {chi_D:.3f} | ~1.0 | N/A |\n\n")
        f.write(f"## Status\n\n")
        f.write(f"{status}\n\n")
        f.write(f"**Grade**: {grade}\n\n")
        f.write(f"## Details\n\n")
        f.write(f"- Initial edges: {initial_edges}\n")
        f.write(f"- Final edges: {W_opt.nnz}\n")
        f.write(f"- Frustration density: {rho_frust:.6f}\n")
        f.write(f"- Spectral dimension method: {d_info.get('status', 'unknown')}\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Cosmic Fixed Point Test for IRH v13.0')
    parser.add_argument('--N', type=int, default=500, help='Number of nodes (default: 500)')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='experiments', help='Output directory (default: experiments)')
    
    args = parser.parse_args()
    
    results = run_cosmic_fixed_point_test(
        N=args.N,
        iterations=args.iterations,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
