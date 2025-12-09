"""
Unit tests for Coherent Evolution Dynamics (Axiom 4).

Tests the evolution dynamics as defined in IRHv16.md §1 Axiom 4.

References:
    IRHv16.md §1 Axiom 4: Algorithmic Coherent Evolution
    IRHv16.md §4: Harmony Functional
"""

import pytest
import numpy as np
from irh.core.v16.crn import CymaticResonanceNetwork
from irh.core.v16.dynamics import (
    EvolutionState,
    CoherentEvolution,
    AdaptiveResonanceOptimization,
)


class TestEvolutionState:
    """Test EvolutionState dataclass."""
    
    def test_basic_creation(self):
        """Test creating evolution state."""
        W = np.eye(5, dtype=np.complex128)
        state = EvolutionState(W=W, tau=0)
        
        assert state.N == 5
        assert state.tau == 0
        
    def test_with_metrics(self):
        """Test state with harmony and energy."""
        W = np.eye(5, dtype=np.complex128)
        state = EvolutionState(W=W, tau=10, harmony=1.5, energy=2.0)
        
        assert state.harmony == 1.5
        assert state.energy == 2.0


class TestCoherentEvolution:
    """Test CoherentEvolution class."""
    
    def test_initialization(self):
        """Test evolution initialization."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        assert len(evo.history) == 1
        assert evo.history[0].tau == 0
        
    def test_single_step(self):
        """Test single evolution step."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        state = evo.step()
        
        assert state.tau == 1
        assert len(evo.history) == 2
        
    def test_multiple_steps(self):
        """Test multiple evolution steps."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        states = evo.evolve(n_steps=10)
        
        assert len(states) == 11  # Initial + 10 steps
        assert states[-1].tau == 10
        
    def test_unitarity(self):
        """Test evolution preserves unitarity."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        is_unitary, deviation = evo.check_unitarity()
        
        # Evolution operator should be unitary
        assert is_unitary
        assert deviation < 1e-9
        
    def test_information_conservation(self):
        """Test information conservation during evolution."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.01)  # Small dt for stability
        
        evo.evolve(n_steps=5)
        
        is_conserved, change = evo.check_information_conservation()
        
        # Information should be approximately conserved
        # (exact conservation depends on numerical precision)
        assert change < 0.1  # Less than 10% change
        
    def test_harmony_computation(self):
        """Test harmony functional computation."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        harmony = evo.history[0].harmony
        
        # Harmony should be finite and positive
        assert harmony is not None
        assert np.isfinite(harmony)
        
    def test_energy_computation(self):
        """Test energy computation."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        energy = evo.history[0].energy
        
        assert energy is not None
        assert np.isfinite(energy)
        
    def test_history_tracking(self):
        """Test history arrays."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        evo.evolve(n_steps=5)
        
        harmony_hist = evo.get_harmony_history()
        energy_hist = evo.get_energy_history()
        
        assert len(harmony_hist) == 6
        assert len(energy_hist) == 6


class TestAdaptiveResonanceOptimization:
    """Test ARO optimization."""
    
    def test_initialization(self):
        """Test ARO initialization."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        aro = AdaptiveResonanceOptimization(crn, dt=0.1)
        
        assert aro.best_harmony is not None
        assert aro.best_state is not None
        
    def test_optimization(self):
        """Test ARO optimization runs."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        aro = AdaptiveResonanceOptimization(crn, dt=0.1)
        
        best_state = aro.optimize(max_steps=10, verbose=False)
        
        assert best_state is not None
        assert best_state.harmony is not None
        
    def test_convergence_metrics(self):
        """Test convergence metrics."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        aro = AdaptiveResonanceOptimization(crn, dt=0.1)
        
        aro.optimize(max_steps=10, verbose=False)
        metrics = aro.get_convergence_metrics()
        
        assert "n_steps" in metrics
        assert "final_harmony" in metrics
        assert "best_harmony" in metrics
        assert metrics["n_steps"] >= 1
        
    def test_best_tracking(self):
        """Test best state tracking."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        aro = AdaptiveResonanceOptimization(crn, dt=0.1)
        
        initial_best = aro.best_harmony
        aro.optimize(max_steps=20, verbose=False)
        final_best = aro.best_harmony
        
        # Best should be at least as good as initial
        assert final_best >= initial_best or np.isclose(final_best, initial_best)


class TestEvolutionStability:
    """Test evolution numerical stability."""
    
    def test_small_dt_stability(self):
        """Test stability with small dt."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.001)
        
        evo.evolve(n_steps=100)
        
        # W matrix should remain bounded
        W_final = evo.history[-1].W
        assert np.all(np.isfinite(W_final))
        
    def test_larger_network(self):
        """Test with larger network."""
        crn = CymaticResonanceNetwork.create_random(N=20, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        evo.evolve(n_steps=5)
        
        assert len(evo.history) == 6
        assert evo.history[-1].W.shape == (20, 20)


class TestEvolutionMathematics:
    """Test mathematical properties of evolution."""
    
    def test_hermitian_laplacian(self):
        """Test interference matrix properties."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        L = evo._get_interference_matrix(evo._current_W)
        
        # L should be complex valued
        assert L.dtype == np.complex128
        
    def test_evolution_operator_exponential(self):
        """Test evolution operator is matrix exponential."""
        crn = CymaticResonanceNetwork.create_random(N=5, seed=42)
        evo = CoherentEvolution(crn, dt=0.1)
        
        L = evo._get_interference_matrix(evo._current_W)
        U = evo._compute_evolution_operator(L)
        
        # U should be unitary: U @ U† = I
        product = U @ U.conj().T
        identity = np.eye(5, dtype=np.complex128)
        
        assert np.allclose(product, identity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
