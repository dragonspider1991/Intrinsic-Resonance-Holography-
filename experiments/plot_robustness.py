#!/usr/bin/env python3
"""
Robustness Plot Generator

This script loads HDF5 files from multiple emergence runs at different
system sizes and generates the robustness plot showing α^(-1) vs N.

This reproduces Figure 1 from the manuscript.
"""

import argparse
from pathlib import Path
import numpy as np
import h5py
from typing import List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cncg.vis import plot_robustness_analysis


def load_hdf5_results(file_path: Path) -> Tuple[List[float], List[int]]:
    """
    Load α^(-1) values and system sizes from HDF5 file.
    
    Parameters
    ----------
    file_path : Path
        Path to HDF5 file
    
    Returns
    -------
    alpha_values : List[float]
        List of α^(-1) values from all trials
    N_values : List[int]
        List of system sizes (should be constant per file)
    """
    alpha_values = []
    N_values = []
    
    with h5py.File(file_path, 'r') as f:
        # Iterate over all trial groups
        for trial_name in f.keys():
            trial_group = f[trial_name]
            alpha_values.append(trial_group.attrs["alpha_inv"])
            N_values.append(trial_group.attrs["N"])
    
    return alpha_values, N_values


def aggregate_results(
    data_dir: Path,
    pattern: str = "emergence_N*.h5",
) -> Tuple[List[int], List[float], List[float]]:
    """
    Aggregate results from multiple HDF5 files.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing HDF5 files
    pattern : str
        Glob pattern for HDF5 files
    
    Returns
    -------
    N_values : List[int]
        Unique system sizes (sorted)
    alpha_means : List[float]
        Mean α^(-1) for each N
    alpha_stds : List[float]
        Standard deviation of α^(-1) for each N
    """
    # Dictionary to collect results by N
    results_by_N = {}
    
    # Find all matching files
    hdf5_files = sorted(data_dir.glob(pattern))
    
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir} matching {pattern}")
    
    print(f"Found {len(hdf5_files)} HDF5 files:")
    for f in hdf5_files:
        print(f"  {f.name}")
    
    # Load each file
    for file_path in hdf5_files:
        alpha_values, N_values_file = load_hdf5_results(file_path)
        
        # Group by N
        for alpha, N in zip(alpha_values, N_values_file):
            if N not in results_by_N:
                results_by_N[N] = []
            results_by_N[N].append(alpha)
    
    # Compute statistics
    N_unique = sorted(results_by_N.keys())
    alpha_means = []
    alpha_stds = []
    
    for N in N_unique:
        values = np.array(results_by_N[N])
        alpha_means.append(np.mean(values))
        alpha_stds.append(np.std(values))
    
    return N_unique, alpha_means, alpha_stds


def main():
    parser = argparse.ArgumentParser(
        description="Generate robustness plot from HDF5 data"
    )
    parser.add_argument("--data-dir", type=str, default="experiments/output",
                       help="Directory containing HDF5 files (default: experiments/output)")
    parser.add_argument("--pattern", type=str, default="emergence_N*.h5",
                       help="Glob pattern for HDF5 files (default: emergence_N*.h5)")
    parser.add_argument("--output", type=str, default="experiments/output/robustness.png",
                       help="Output plot path (default: experiments/output/robustness.png)")
    parser.add_argument("--target-alpha", type=float, default=137.036,
                       help="Target α^(-1) value (default: 137.036)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    print(f"\n{'='*60}")
    print("ROBUSTNESS PLOT GENERATOR")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Output: {output_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Aggregate results
    print("\nAggregating results...")
    N_values, alpha_means, alpha_stds = aggregate_results(data_dir, args.pattern)
    
    # Display results
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"{'N':>6} {'α⁻¹ (mean)':>15} {'α⁻¹ (std)':>15}")
    print(f"{'-'*40}")
    
    for N, mean, std in zip(N_values, alpha_means, alpha_stds):
        print(f"{N:6d} {mean:15.3f} {std:15.3f}")
    
    # Generate plot
    print(f"\nGenerating robustness plot...")
    plot_robustness_analysis(
        N_values=N_values,
        alpha_inv_means=alpha_means,
        alpha_inv_stds=alpha_stds,
        target_alpha_inv=args.target_alpha,
        save_path=str(output_path),
    )
    
    print(f"Plot saved to {output_path}")
    
    # Statistical summary
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    
    overall_mean = np.mean(alpha_means)
    overall_std = np.std(alpha_means)
    
    print(f"Overall α⁻¹: {overall_mean:.3f} ± {overall_std:.3f}")
    print(f"Target α⁻¹: {args.target_alpha:.3f}")
    print(f"Deviation: {abs(overall_mean - args.target_alpha):.3f}")
    print(f"Relative error: {100 * abs(overall_mean - args.target_alpha) / args.target_alpha:.2f}%")
    
    print(f"\n{'='*60}")
    print("PLOT GENERATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
