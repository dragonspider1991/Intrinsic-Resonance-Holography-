"""
Minimal ARO Demo

Demonstrates Adaptive Resonance Optimization on a small network.
Shows how random networks evolve toward ordered 4D structure.
"""

from irh_v10.core import CymaticResonanceNetwork, AdaptiveResonanceOptimizer
import matplotlib.pyplot as plt


def main():
    print("="*60)
    print("ADAPTIVE RESONANCE OPTIMIZATION DEMO")
    print("="*60)
    
    # Create random network
    print("\n1. Creating random network (N=64)...")
    network = CymaticResonanceNetwork(
        N=64,
        topology="random",
        seed=42
    )
    
    # Initial state
    initial_spectrum = network.compute_spectrum()
    print(f"   Initial spectrum: {len(initial_spectrum)} eigenvalues")
    print(f"   λ_min = {initial_spectrum[1]:.6f}")
    print(f"   λ_max = {initial_spectrum[-1]:.6f}")
    
    # Run ARO
    print("\n2. Running Adaptive Resonance Optimization...")
    aro = AdaptiveResonanceOptimizer(
        network,
        max_iterations=500,
        T_initial=1.0,
        T_final=0.01,
        verbose=True
    )
    
    result = aro.optimize()
    
    # Final state
    print("\n3. Optimization complete!")
    print(f"   Initial harmony: {result.harmony_history[0]:.6f}")
    print(f"   Final harmony: {result.final_harmony:.6f}")
    print(f"   Improvement: {result.harmony_history[0] - result.final_harmony:.6f}")
    print(f"   Acceptance rate: {result.acceptance_rate:.1%}")
    print(f"   Converged: {result.converged}")
    
    # Plot harmony evolution
    print("\n4. Plotting harmony evolution...")
    plt.figure(figsize=(10, 6))
    plt.plot(result.harmony_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Harmony Functional ℋ', fontsize=12)
    plt.title('ARO: Evolution toward Minimal Harmony', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('aro_demo.png', dpi=150)
    print("   Saved: aro_demo.png")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
