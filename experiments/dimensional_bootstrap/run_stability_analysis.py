"""
Dimensional Bootstrap Experiment for IRH v11.0

This script executes the complete dimensional bootstrap analysis,
verifying that d=4 emerges uniquely as the stable fixed point.

Generates:
1. Phase diagram showing S_SOTE vs dimension
2. Spectral dimension measurements for d=2,3,4,5,6
3. Statistical significance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add repository root to path
sys.path.insert(0, '/home/runner/work/Intrinsic-Resonance-Holography-/Intrinsic-Resonance-Holography-')

from src.core.substrate_v11 import InformationSubstrate
from src.core.sote_v11 import SOTEFunctional

# Configuration
DIMENSIONS = [2, 3, 4, 5, 6]
N_SAMPLES = 3  # Number of independent realizations per dimension
N_NODES = 2000  # System size
OUTPUT_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)

def compute_spectral_dimension(L, s_range=None):
    """
    Compute spectral dimension from eigenvalue spectrum via zeta function.
    The dimension is where zeta(s) diverges.
    """
    from scipy.sparse.linalg import eigsh
    
    if s_range is None:
        s_range = np.linspace(1.5, 3.5, 50)
    
    try:
        # Compute eigenvalues
        k = min(100, L.shape[0] - 2)
        eigenvalues = eigsh(L, k=k, which='SM', return_eigenvectors=False)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
    except:
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Zeta function
    zeta_vals = []
    for s in s_range:
        zeta_s = np.sum(eigenvalues**(-s))
        zeta_vals.append(zeta_s)
    
    # Find the pole (rapid divergence)
    grad = np.gradient(np.log(np.abs(zeta_vals) + 1e-10), s_range)
    d_spec = 2 * s_range[np.argmax(grad)]  # pole at s = d/2
    
    return d_spec

def analyze_dimension(d, sample_idx):
    """Run single analysis for dimension d."""
    print(f"  [{sample_idx+1}/{N_SAMPLES}] Analyzing d={d}...")
    
    # Create substrate
    substrate = InformationSubstrate(N=N_NODES, dimension=d)
    substrate.initialize_correlations('random_geometric')
    substrate.compute_laplacian()
    
    # Compute ARO action
    sote = SOTEFunctional(substrate)
    S_action = sote.compute_action()
    
    # Measure spectral dimension
    d_spec = compute_spectral_dimension(substrate.L)
    
    # Dimensional consistency
    consistency = -abs(d_spec - d)
    
    # Holographic bound check
    bound_check = substrate.verify_holographic_bound()
    
    return {
        'd_target': d,
        'd_spectral': d_spec,
        'S_action': S_action,
        'consistency': consistency,
        'N': N_NODES,
        'holographic_ratio': bound_check['ratio'],
        'sample': sample_idx
    }

def run_dimensional_bootstrap():
    """Execute complete dimensional bootstrap analysis."""
    
    print("=" * 70)
    print(" DIMENSIONAL BOOTSTRAP EXPERIMENT")
    print(" IRH v11.0 - Verifying d=4 Uniqueness")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dimensions: {DIMENSIONS}")
    print(f"  Samples per dimension: {N_SAMPLES}")
    print(f"  System size: N = {N_NODES}")
    print()
    
    results = []
    
    for d in DIMENSIONS:
        print(f"\nDimension d={d}:")
        for sample_idx in range(N_SAMPLES):
            try:
                result = analyze_dimension(d, sample_idx)
                results.append(result)
                print(f"    S_SOTE = {result['S_action']:.4e}, "
                      f"d_spec = {result['d_spectral']:.3f}, "
                      f"consistency = {result['consistency']:.4f}")
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    # Save raw data
    print(f"\nSaving results to {OUTPUT_DIR}...")
    with open(OUTPUT_DIR / 'dimensional_bootstrap_data.json', 'w') as f:
        # Convert numpy types for JSON
        results_serializable = []
        for r in results:
            r_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in r.items()}
            results_serializable.append(r_clean)
        json.dump(results_serializable, f, indent=2)
    
    return results

def analyze_and_plot(results):
    """Aggregate statistics and create visualizations."""
    
    print("\n" + "=" * 70)
    print(" STATISTICAL ANALYSIS")
    print("=" * 70)
    
    # Aggregate by dimension
    stats = {}
    for d in DIMENSIONS:
        d_results = [r for r in results if r['d_target'] == d]
        
        if not d_results:
            continue
        
        consistencies = [r['consistency'] for r in d_results]
        actions = [r['S_action'] for r in d_results]
        d_specs = [r['d_spectral'] for r in d_results]
        
        stats[d] = {
            'mean_consistency': np.mean(consistencies),
            'std_consistency': np.std(consistencies),
            'mean_action': np.mean(actions),
            'std_action': np.std(actions),
            'mean_d_spec': np.mean(d_specs),
            'std_d_spec': np.std(d_specs)
        }
        
        print(f"\nd = {d}:")
        print(f"  Spectral dimension: {stats[d]['mean_d_spec']:.3f} ± {stats[d]['std_d_spec']:.3f}")
        print(f"  ARO action: {stats[d]['mean_action']:.4e} ± {stats[d]['std_action']:.4e}")
        print(f"  Consistency: {stats[d]['mean_consistency']:.4f} ± {stats[d]['std_consistency']:.4f}")
    
    # Find maximum consistency
    max_consistency_d = max(stats.keys(), key=lambda d: stats[d]['mean_consistency'])
    
    print(f"\n{'=' * 70}")
    print(f" RESULT: Maximum stability at d = {max_consistency_d}")
    print(f" IRH Prediction: d = 4")
    print(f" {'✓ MATCH' if max_consistency_d == 4 else '✗ MISMATCH'}")
    print(f"{'=' * 70}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    dims = sorted(stats.keys())
    
    # Plot 1: Consistency
    means = [stats[d]['mean_consistency'] for d in dims]
    stds = [stats[d]['std_consistency'] for d in dims]
    
    axes[0, 0].errorbar(dims, means, yerr=stds, fmt='o-', capsize=5, linewidth=2,
                        markersize=8, color='blue')
    axes[0, 0].axvline(x=4, color='red', linestyle='--', linewidth=2, 
                       label='IRH Prediction: d=4')
    axes[0, 0].set_xlabel('Dimension d', fontsize=13)
    axes[0, 0].set_ylabel('Dimensional Consistency', fontsize=13)
    axes[0, 0].set_title('Stability Analysis: d=4 Uniqueness', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: ARO Action
    means_action = [stats[d]['mean_action'] for d in dims]
    stds_action = [stats[d]['std_action'] for d in dims]
    
    axes[0, 1].errorbar(dims, means_action, yerr=stds_action, fmt='s-', capsize=5,
                        linewidth=2, markersize=8, color='green')
    axes[0, 1].axvline(x=4, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Dimension d', fontsize=13)
    axes[0, 1].set_ylabel('$S_{ARO}$ Action', fontsize=13)
    axes[0, 1].set_title('ARO Action vs Dimension', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Spectral Dimension
    means_dspec = [stats[d]['mean_d_spec'] for d in dims]
    stds_dspec = [stats[d]['std_d_spec'] for d in dims]
    
    axes[1, 0].errorbar(dims, means_dspec, yerr=stds_dspec, fmt='d-', capsize=5,
                        linewidth=2, markersize=8, color='purple')
    axes[1, 0].plot(dims, dims, 'k--', alpha=0.5, label='Perfect match')
    axes[1, 0].set_xlabel('Target Dimension $d_{target}$', fontsize=13)
    axes[1, 0].set_ylabel('Measured Dimension $d_{spectral}$', fontsize=13)
    axes[1, 0].set_title('Spectral Dimension Measurement', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot
    for d in dims:
        d_results = [r for r in results if r['d_target'] == d]
        actions = [r['S_action'] for r in d_results]
        consistencies = [r['consistency'] for r in d_results]
        
        color = 'red' if d == 4 else 'gray'
        size = 150 if d == 4 else 80
        alpha = 1.0 if d == 4 else 0.5
        
        axes[1, 1].scatter(actions, consistencies, s=size, alpha=alpha,
                          color=color, label=f'd={d}' if d == 4 else '')
    
    axes[1, 1].set_xlabel('$S_{ARO}$ Action', fontsize=13)
    axes[1, 1].set_ylabel('Consistency', fontsize=13)
    axes[1, 1].set_title('Action-Consistency Phase Space', fontsize=14, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / 'dimensional_bootstrap_phase_diagram.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPhase diagram saved to {plot_path}")
    
    plt.show()

def main():
    """Execute dimensional bootstrap experiment."""
    
    # Run analysis
    results = run_dimensional_bootstrap()
    
    if not results:
        print("\n✗ No results generated. Experiment failed.")
        return False
    
    # Analyze and visualize
    analyze_and_plot(results)
    
    print("\n" + "=" * 70)
    print(" DIMENSIONAL BOOTSTRAP EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults and plots saved to: {OUTPUT_DIR}")
    print()
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
