"""
Test suite for Harmony Functional

Tests the core optimization objective function.
"""

import pytest
import numpy as np
from irh_v10.core.substrate import CymaticResonanceNetwork
from irh_v10.core.harmony_functional import harmony_functional
from irh_v10.core.impedance_matching import impedance_coefficient


def test_harmony_functional_positive():
    """Harmony functional should be positive for physical networks."""
    network = CymaticResonanceNetwork(N=16, topology="grid_4d", seed=42)
    H = harmony_functional(network.K, network.N)
    assert H > 0, "Harmony functional should be positive"


def test_harmony_functional_decreases_with_order():
    """More ordered networks should have lower harmony."""
    # Random network (disordered)
    network_random = CymaticResonanceNetwork(N=64, topology="random", seed=42)
    H_random = harmony_functional(network_random.K, network_random.N)
    
    # Grid network (ordered)
    network_grid = CymaticResonanceNetwork(N=81, topology="toroidal_4d", seed=42)
    H_grid = harmony_functional(network_grid.K, network_grid.N)
    
    # Grid should have lower harmony (more harmonious)
    assert H_grid < H_random, "Ordered network should have lower harmony"


def test_impedance_coefficient_scaling():
    """Test ξ(N) = 1/(N ln N) scaling."""
    N_values = [10, 100, 1000]
    xi_values = [impedance_coefficient(N) for N in N_values]
    
    # Check scaling: N × ln(N) × ξ(N) ≈ 1
    for N, xi in zip(N_values, xi_values):
        scaled = N * np.log(N) * xi
        assert abs(scaled - 1.0) < 1e-10, f"Scaling violated for N={N}"


def test_harmony_functional_with_precomputed_eigenvalues():
    """Test that precomputed eigenvalues give same result."""
    network = CymaticResonanceNetwork(N=25, topology="toroidal_4d", seed=42)
    
    # Compute without eigenvalues
    H1 = harmony_functional(network.K, network.N)
    
    # Compute spectrum
    eigenvalues = network.compute_spectrum()
    
    # Compute with eigenvalues
    H2 = harmony_functional(network.K, network.N, eigenvalues=eigenvalues)
    
    assert abs(H1 - H2) < 1e-6, "Should get same harmony with/without precomputed eigenvalues"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
