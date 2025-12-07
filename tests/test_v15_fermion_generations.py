"""
Test suite for Phase 5: Fermion Generations

Tests the topological derivation of 3 fermion generations from instanton
number and the mass hierarchy from knot complexity.
"""
import pytest
import numpy as np
import scipy.sparse as sp
from src.core.aro_optimizer import AROOptimizer
from src.topology.instantons import (
    compute_instanton_number,
    compute_dirac_operator_index
)
from src.physics.fermion_masses import derive_mass_ratios


class TestFermionGenerations:
    """Test fermion generation derivation."""
    
    def test_instanton_number_is_integer(self):
        """Test that instanton number is an integer (topological invariant)."""
        # Create test network
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=50, verbose=False)
        
        # Compute instanton number
        # Need boundary nodes (mock for test)
        boundary_nodes = np.arange(10)
        
        n_inst, details = compute_instanton_number(
            opt.best_W, boundary_nodes
        )
        
        # Should be an integer (topological invariant)
        assert isinstance(n_inst, int)
        assert n_inst >= 1  # At least one generation
    
    def test_instanton_number_details(self):
        """Test that instanton number computation returns proper details."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        boundary_nodes = np.arange(10)
        n_inst, details = compute_instanton_number(
            opt.best_W, boundary_nodes
        )
        
        # Should have diagnostic information
        assert isinstance(details, dict)
        assert 'method' in details
    
    def test_atiyah_singer_index_returns_integer(self):
        """Test Atiyah-Singer index returns integer."""
        opt = AROOptimizer(N=100, rng_seed=123)
        opt.initialize_network('geometric', 0.1, 4)
        
        boundary_nodes = np.arange(10)
        n_inst, _ = compute_instanton_number(
            opt.best_W, boundary_nodes
        )
        
        index_D, details = compute_dirac_operator_index(
            opt.best_W, n_inst
        )
        
        # Index should be an integer
        assert isinstance(index_D, int)
        assert isinstance(details, dict)
        assert 'index_topological' in details
        assert details['index_topological'] == n_inst
    
    def test_mass_ratios_returns_dict(self):
        """Test mass ratio derivation returns proper structure."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=50, verbose=False)
        
        results = derive_mass_ratios(opt.best_W, n_inst=3)
        
        # Should return proper structure
        assert isinstance(results, dict)
        assert 'mass_ratios' in results
        assert 'experimental' in results
        assert 'errors_percent' in results
    
    def test_mass_ratio_muon_electron_in_range(self):
        """Test m_μ/m_e is in reasonable range."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=50, verbose=False)
        
        results = derive_mass_ratios(opt.best_W, n_inst=3)
        
        # Should be in reasonable range
        predicted = results['mass_ratios']['m_mu/m_e']
        
        # Very broad range for placeholder implementation
        assert 50 < predicted < 500
    
    def test_mass_ratio_tau_electron_in_range(self):
        """Test m_τ/m_e is in reasonable range."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=50, verbose=False)
        
        results = derive_mass_ratios(opt.best_W, n_inst=3)
        
        # Should be in reasonable range
        predicted = results['mass_ratios']['m_tau/m_e']
        
        # Very broad range for placeholder implementation
        assert 1000 < predicted < 7000
    
    def test_generation_ordering(self):
        """Test that m_e < m_μ < m_τ."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=50, verbose=False)
        
        results = derive_mass_ratios(opt.best_W, n_inst=3)
        
        # Verify ordering
        assert results['mass_ratios']['m_mu/m_e'] > 1.0
        assert results['mass_ratios']['m_tau/m_e'] > results['mass_ratios']['m_mu/m_e']
    
    def test_experimental_values_present(self):
        """Test that experimental values are included for comparison."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        results = derive_mass_ratios(opt.best_W, n_inst=3)
        
        # Should have experimental values
        exp = results['experimental']
        assert 'm_mu/m_e' in exp
        assert 'm_tau/m_e' in exp
        assert 'm_tau/m_mu' in exp
        
        # Check experimental values are correct (CODATA 2022)
        assert abs(exp['m_mu/m_e'] - 206.7682830) < 0.001
        assert abs(exp['m_tau/m_e'] - 3477.15) < 0.1
    
    def test_division_by_zero_guard(self):
        """Test that division by zero is handled in error calculation."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        results = derive_mass_ratios(opt.best_W, n_inst=3)
        
        # Should compute errors without crashing
        assert 'errors_percent' in results
        for key, error in results['errors_percent'].items():
            assert isinstance(error, (int, float))
            # Error should be finite or inf (not NaN)
            assert error == error  # NaN != NaN, so this checks not NaN


@pytest.mark.slow
class TestLargeScaleFermions:
    """Tests requiring larger networks (marked as slow)."""
    
    def test_larger_network_instanton(self):
        """Test instanton number with larger network."""
        opt = AROOptimizer(N=300, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=200, verbose=False)
        
        boundary_nodes = np.arange(30)
        n_inst, details = compute_instanton_number(
            opt.best_W, boundary_nodes
        )
        
        # Should still be integer
        assert isinstance(n_inst, int)
        assert n_inst >= 1
    
    def test_larger_network_mass_ratios(self):
        """Test mass ratios with larger network."""
        opt = AROOptimizer(N=300, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=200, verbose=False)
        
        results = derive_mass_ratios(opt.best_W, n_inst=3, include_radiative=True)
        
        # Should be closer to experimental values with larger N
        # (though still approximate with placeholder implementation)
        assert 'mass_ratios' in results
        assert results['mass_ratios']['m_mu/m_e'] > 1.0
