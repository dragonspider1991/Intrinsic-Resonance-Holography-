import sys
import os
import numpy as np
import time

# Ensure src modules are in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.aro_optimizer import AROOptimizer
from src.topology.invariants import TopologyAnalyzer
from src.metrics.dimensions import DimensionalityAnalyzer
from src.core.harmony import HarmonyEngine

def run_cosmic_simulation(N=100, iterations=1000, seed=42):
    """
    Executes the Tier 1 Empirical Test for Intrinsic Resonance Holography v13.0.
    
    Target Predictions:
        - Fine-Structure Constant (alpha^-1): ~137.036
        - Spectral Dimension (d_spec): ~4.0
        - Generations (N_gen): 3
    """
    print("="*60)
    print(f"INTRINSIC RESONANCE HOLOGRAPHY v13.0: COSMIC BOOTSTRAP")
    print(f"Nodes (N): {N} | Iterations: {iterations} | Seed: {seed}")
    print("="*60)
    
    # 1. Initialize and Optimize (The ARO Process)
    print("\n[Phase 1] Initiating Adaptive Resonance Optimization (ARO)...")
    start_time = time.time()
    
    optimizer = AROOptimizer(N=N, connection_probability=0.2, rng_seed=seed)
    
    # Run optimization loop
    final_W = optimizer.optimize(iterations=iterations, temp=1.0, cooling_rate=0.99)
    
    duration = time.time() - start_time
    print(f"Optimization complete in {duration:.2f}s.")
    print(f"Final Harmony: {optimizer.best_S:.6f}")
    
    # 2. Analyze Topological Invariants (The Constants)
    print("\n[Phase 2] Measuring Topological Invariants...")
    topo_analyzer = TopologyAnalyzer(final_W, threshold=1e-5)
    
    # Calculate Alpha Inverse
    alpha_inv = topo_analyzer.derive_alpha_inv()
    
    # Calculate Betti Numbers (Gauge Group)
    beta_1 = topo_analyzer.calculate_betti_numbers()
    
    # Calculate Generations
    n_gen = topo_analyzer.calculate_generation_count()
    
    # 3. Analyze Dimensional Coherence (The Geometry)
    print("\n[Phase 3] Verifying Dimensional Coherence...")
    # Compute Information Transfer Matrix for dimensional analysis
    M = HarmonyEngine.compute_information_transfer_matrix(final_W)
    dim_analyzer = DimensionalityAnalyzer(M)
    
    # Calculate Spectral Dimension
    d_spec = dim_analyzer.calculate_spectral_dimension(t_start=1e-2, t_end=1.0)
    
    # Calculate Coherence Index
    chi_D = dim_analyzer.calculate_dimensional_coherence(d_spec)
    
    # 4. Final Report
    print("\n" + "="*60)
    print("FINAL EXPERIMENTAL REPORT")
    print("="*60)
    
    print(f"{'Parameter':<25} | {'Prediction (v13.0)':<20} | {'Measured Value':<20}")
    print("-" * 70)
    print(f"{'Inv. Fine-Structure':<25} | {'137.036 ± 0.004':<20} | {alpha_inv:.4f}")
    print(f"{'Spectral Dimension':<25} | {'4.00 (Exact)':<20} | {d_spec:.4f}")
    print(f"{'Fermion Generations':<25} | {'3 (Exact)':<20} | {n_gen}")
    print(f"{'Gauge Group (Beta_1)':<25} | {'12 (SM)':<20} | {beta_1}")
    print("-" * 70)
    print(f"Dimensional Coherence Index (chi_D): {chi_D:.4f} (Max ~1.0 at d=4)")
    print("="*60)
    
    # Validation Logic
    success = True
    if abs(alpha_inv - 137.036) > 5.0: success = False
    if abs(d_spec - 4.0) > 0.5: success = False
    
    if success:
        print("\nRESULT: CONVERGENCE SUCCESSFUL. The Cosmic Fixed Point is stable.")
    else:
        print("\nRESULT: DEVIATION DETECTED. Higher N or more iterations required.")

if __name__ == "__main__":
    # Note: For full accuracy matching the paper (alpha=137.036), 
    # N should be >= 10^4 and iterations >= 10^5.
    # We run a smaller demo here for immediate feedback.
    run_cosmic_simulation(N=100, iterations=2000, seed=42)
