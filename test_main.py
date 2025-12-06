"""
Quick test script to verify main.py functionality
"""
import sys
import os

# Ensure src modules are in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.aro_optimizer import AROOptimizer
from src.topology.invariants import TopologyAnalyzer
from src.metrics.dimensions import DimensionalityAnalyzer
from src.core.harmony import HarmonyEngine
import numpy as np

def test_harmony_engine():
    """Test HarmonyEngine static methods"""
    print("Testing HarmonyEngine...")
    W = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    M = HarmonyEngine.compute_information_transfer_matrix(W)
    assert M.shape == (10, 10), "M should have same shape as W"
    
    S_H = HarmonyEngine.calculate_harmony(W, 10)
    assert isinstance(S_H, (float, np.floating)), "Harmony should be a float"
    assert S_H > 0, "Harmony should be positive"
    print("  ✓ HarmonyEngine tests passed")

def test_aro_optimizer():
    """Test AROOptimizer initialization and optimization"""
    print("Testing AROOptimizer...")
    optimizer = AROOptimizer(N=20, connection_probability=0.2, rng_seed=42)
    assert optimizer.current_W is not None, "Network should be initialized"
    
    final_W = optimizer.optimize(iterations=10, temp=1.0, cooling_rate=0.9, verbose=False)
    assert final_W.shape == (20, 20), "Final W should have correct shape"
    assert isinstance(final_W, np.ndarray), "Final W should be numpy array"
    print("  ✓ AROOptimizer tests passed")

def test_topology_analyzer():
    """Test TopologyAnalyzer class"""
    print("Testing TopologyAnalyzer...")
    W = np.random.rand(15, 15) + 1j * np.random.rand(15, 15)
    analyzer = TopologyAnalyzer(W, threshold=1e-5)
    
    alpha_inv = analyzer.derive_alpha_inv()
    assert isinstance(alpha_inv, (float, np.floating)), "alpha_inv should be a float"
    
    beta_1 = analyzer.calculate_betti_numbers()
    assert isinstance(beta_1, (int, np.integer)), "beta_1 should be an integer"
    
    n_gen = analyzer.calculate_generation_count()
    assert isinstance(n_gen, (int, np.integer)), "n_gen should be an integer"
    print("  ✓ TopologyAnalyzer tests passed")

def test_dimensionality_analyzer():
    """Test DimensionalityAnalyzer class"""
    print("Testing DimensionalityAnalyzer...")
    W = np.random.rand(15, 15) + 1j * np.random.rand(15, 15)
    M = HarmonyEngine.compute_information_transfer_matrix(W)
    analyzer = DimensionalityAnalyzer(M)
    
    d_spec = analyzer.calculate_spectral_dimension()
    assert isinstance(d_spec, (float, np.floating)), "d_spec should be a float"
    assert 1.0 <= d_spec <= 10.0, "d_spec should be in reasonable range"
    
    chi_D = analyzer.calculate_dimensional_coherence(d_spec)
    assert isinstance(chi_D, (float, np.floating)), "chi_D should be a float"
    assert 0.0 <= chi_D <= 1.0, "chi_D should be in [0, 1]"
    print("  ✓ DimensionalityAnalyzer tests passed")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Running IRH v13.0 Component Tests")
    print("="*50 + "\n")
    
    test_harmony_engine()
    test_aro_optimizer()
    test_topology_analyzer()
    test_dimensionality_analyzer()
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50 + "\n")
