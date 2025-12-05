#!/usr/bin/env python3
"""
Emergence Experiment

This script runs the main emergence simulation to demonstrate:
1. Spontaneous emergence of 4D geometry
2. Convergence to α ≈ 1/137
3. Development of 3 chiral zero modes

Results are saved to HDF5 for subsequent analysis.
"""

import argparse
import time
from pathlib import Path
import numpy as np
import h5py
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cncg import (
    FiniteSpectralTriple,
    riemannian_gradient_descent,
)
from cncg.analysis import (
    compute_spectral_dimension,
    compute_fine_structure_constant,
    analyze_zero_modes,
)
from cncg.vis import (
    plot_optimization_history,
    plot_spectral_density,
    plot_eigenvalue_flow,
)


def run_single_trial(
    N: int,
    seed: int,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    Lambda: float = 1.0,
    sparsity_weight: float = 0.001,
    log_interval: int = 10,
) -> Dict:
    """
    Run a single emergence trial.
    
    Parameters
    ----------
    N : int
        System size
    seed : int
        Random seed
    max_iterations : int
        Number of optimization iterations
    learning_rate : float
        Initial learning rate
    Lambda : float
        Energy scale
    sparsity_weight : float
        Sparsity penalty weight
    log_interval : int
        Logging interval
    
    Returns
    -------
    results : Dict
        Dictionary with final observables and history
    """
    print(f"\n{'='*60}")
    print(f"Trial: N={N}, seed={seed}")
    print(f"{'='*60}")
    
    # Initialize random spectral triple
    triple = FiniteSpectralTriple(N=N, seed=seed)
    
    # Track eigenvalue evolution
    eigenvalue_history = []
    
    def callback(iteration: int, S: float, triple: FiniteSpectralTriple) -> None:
        """Callback to track eigenvalues."""
        eigenvalue_history.append(triple.spectrum())
    
    # Run optimization
    print("\nRunning gradient descent...")
    start_time = time.time()
    
    history = riemannian_gradient_descent(
        triple=triple,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        Lambda=Lambda,
        cutoff="heat",
        cutoff_param=1.0,
        sparsity_weight=sparsity_weight,
        noise_strength=0.0,
        momentum=0.9,
        adaptive=True,
        log_interval=log_interval,
        callback=callback,
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
    
    # Compute final observables
    print("\nComputing physical observables...")
    
    d_s, d_s_error = compute_spectral_dimension(triple)
    print(f"  Spectral dimension: d_s = {d_s:.3f} ± {d_s_error:.3f}")
    
    alpha_inv, alpha_error = compute_fine_structure_constant(triple)
    print(f"  Fine-structure: α⁻¹ = {alpha_inv:.3f} ± {alpha_error:.3f}")
    
    zero_mode_info = analyze_zero_modes(triple)
    print(f"  Zero modes: {zero_mode_info['n_zero_modes']}")
    print(f"    (+) chirality: {zero_mode_info['n_plus']}")
    print(f"    (-) chirality: {zero_mode_info['n_minus']}")
    print(f"    Mass gap: {zero_mode_info['mass_gap']:.6e}")
    
    # Collect results
    results = {
        "N": N,
        "seed": seed,
        "elapsed_time": elapsed_time,
        "d_s": d_s,
        "d_s_error": d_s_error,
        "alpha_inv": alpha_inv,
        "alpha_error": alpha_error,
        "n_zero_modes": zero_mode_info["n_zero_modes"],
        "n_plus": zero_mode_info["n_plus"],
        "n_minus": zero_mode_info["n_minus"],
        "mass_gap": zero_mode_info["mass_gap"],
        "final_action": history["action"][-1] if history["action"] else 0.0,
        "final_spectrum": triple.spectrum(),
        "history": history,
        "eigenvalue_history": eigenvalue_history,
    }
    
    return results


def save_results_hdf5(results_list: List[Dict], output_path: Path) -> None:
    """
    Save results to HDF5 file.
    
    Parameters
    ----------
    results_list : List[Dict]
        List of trial results
    output_path : Path
        Output HDF5 file path
    """
    print(f"\nSaving results to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Create groups for each trial
        for i, results in enumerate(results_list):
            trial_group = f.create_group(f"trial_{i}")
            
            # Scalar attributes
            trial_group.attrs["N"] = results["N"]
            trial_group.attrs["seed"] = results["seed"]
            trial_group.attrs["elapsed_time"] = results["elapsed_time"]
            trial_group.attrs["d_s"] = results["d_s"]
            trial_group.attrs["d_s_error"] = results["d_s_error"]
            trial_group.attrs["alpha_inv"] = results["alpha_inv"]
            trial_group.attrs["alpha_error"] = results["alpha_error"]
            trial_group.attrs["n_zero_modes"] = results["n_zero_modes"]
            trial_group.attrs["n_plus"] = results["n_plus"]
            trial_group.attrs["n_minus"] = results["n_minus"]
            trial_group.attrs["mass_gap"] = results["mass_gap"]
            trial_group.attrs["final_action"] = results["final_action"]
            
            # Arrays
            trial_group.create_dataset("final_spectrum", data=results["final_spectrum"])
            
            # History
            hist_group = trial_group.create_group("history")
            for key, values in results["history"].items():
                hist_group.create_dataset(key, data=values)
            
            # Eigenvalue evolution
            if results["eigenvalue_history"]:
                eig_history = np.array(results["eigenvalue_history"])
                trial_group.create_dataset("eigenvalue_history", data=eig_history)
    
    print("Results saved successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Run spectral triple emergence experiment"
    )
    parser.add_argument("--N", type=int, default=100,
                       help="System size (default: 100)")
    parser.add_argument("--n-trials", type=int, default=1,
                       help="Number of trials with different seeds (default: 1)")
    parser.add_argument("--max-iterations", type=int, default=500,
                       help="Maximum optimization iterations (default: 500)")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Initial learning rate (default: 0.01)")
    parser.add_argument("--sparsity-weight", type=float, default=0.001,
                       help="Sparsity penalty weight (default: 0.001)")
    parser.add_argument("--output-dir", type=str, default="experiments/output",
                       help="Output directory (default: experiments/output)")
    parser.add_argument("--seed-start", type=int, default=42,
                       help="Starting seed for trials (default: 42)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SPECTRAL TRIPLE EMERGENCE EXPERIMENT")
    print(f"{'='*60}")
    print(f"System size: N = {args.N}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Sparsity weight: {args.sparsity_weight}")
    print(f"Output directory: {output_dir}")
    
    # Run trials
    results_list = []
    
    for trial in range(args.n_trials):
        seed = args.seed_start + trial
        
        results = run_single_trial(
            N=args.N,
            seed=seed,
            max_iterations=args.max_iterations,
            learning_rate=args.learning_rate,
            Lambda=1.0,
            sparsity_weight=args.sparsity_weight,
            log_interval=10,
        )
        
        results_list.append(results)
        
        # Save plot for this trial
        if results["history"]["iteration"]:
            plot_path = output_dir / f"optimization_N{args.N}_seed{seed}.png"
            plot_optimization_history(results["history"], save_path=str(plot_path))
            
            spectrum_path = output_dir / f"spectrum_N{args.N}_seed{seed}.png"
            # Need to reconstruct triple for plotting
            triple_final = FiniteSpectralTriple(N=args.N)
            triple_final.D = np.diag(results["final_spectrum"])
            plot_spectral_density(triple_final, save_path=str(spectrum_path))
    
    # Save all results to HDF5
    output_file = output_dir / f"emergence_N{args.N}_trials{args.n_trials}.h5"
    save_results_hdf5(results_list, output_file)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    d_s_values = [r["d_s"] for r in results_list]
    alpha_values = [r["alpha_inv"] for r in results_list]
    n_zero_values = [r["n_zero_modes"] for r in results_list]
    
    print(f"Spectral dimension: {np.mean(d_s_values):.3f} ± {np.std(d_s_values):.3f}")
    print(f"Fine-structure α⁻¹: {np.mean(alpha_values):.3f} ± {np.std(alpha_values):.3f}")
    print(f"Zero modes: {np.mean(n_zero_values):.1f} ± {np.std(n_zero_values):.1f}")
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
