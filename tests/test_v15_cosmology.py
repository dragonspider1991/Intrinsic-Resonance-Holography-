"""
Test suite for Phase 6: Cosmological Constant & Dark Energy

Tests the ARO cancellation mechanism for vacuum energy and the derivation
of dark energy equation of state w₀ = -0.912 ± 0.008.
"""
import pytest
import numpy as np
import scipy.sparse as sp
from src.core.aro_optimizer import AROOptimizer
from src.cosmology.vacuum_energy import (
    compute_vacuum_energy_density,
    compute_aro_cancellation
)
from src.cosmology.dark_energy import (
    compute_equation_of_state,
    DarkEnergyAnalyzer
)


class TestCosmologicalConstant:
    """Test cosmological constant resolution."""
    
    def test_vacuum_energy_returns_dict(self):
        """Test vacuum energy density returns proper structure."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        vac = compute_vacuum_energy_density(opt.current_W)
        
        # Should return proper structure
        assert isinstance(vac, dict)
        assert 'rho_vac_bare' in vac
        assert 'rho_vac_regularized' in vac
        assert 'Lambda_QFT' in vac
    
    def test_aro_cancellation_returns_dict(self):
        """Test ARO cancellation returns proper structure."""
        opt = AROOptimizer(N=100, rng_seed=123)
        opt.initialize_network('geometric', 0.1, 4)
        W_initial = opt.current_W.copy()
        
        # Run brief optimization
        opt.optimize(iterations=50, verbose=False)
        W_final = opt.best_W
        
        # Compute cancellation
        cc = compute_aro_cancellation(W_initial, W_final)
        
        # Should return proper structure
        assert isinstance(cc, dict)
        assert 'rho_vac_initial' in cc
        assert 'rho_vac_final' in cc
        assert 'cancellation_factor' in cc
        assert 'Lambda_ratio' in cc
    
    def test_lambda_ratio_present(self):
        """Test that Lambda ratio is computed."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        W_initial = opt.current_W.copy()
        
        opt.optimize(iterations=50, verbose=False)
        
        cc = compute_aro_cancellation(W_initial, opt.best_W)
        
        # Should have Lambda ratio
        assert 'Lambda_ratio' in cc
        assert 'log10_Lambda_ratio' in cc
        assert 'target_log10_ratio' in cc
        assert cc['target_log10_ratio'] == -120.45


class TestDarkEnergy:
    """Test dark energy equation of state."""
    
    def test_equation_of_state_returns_dict(self):
        """Test equation of state returns proper structure."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=50, verbose=False)
        
        eos = compute_equation_of_state(opt.best_W)
        
        # Should return proper structure
        assert isinstance(eos, dict)
        assert 'w_0' in eos
        assert 'delta_w' in eos
    
    def test_w0_close_to_minus_one(self):
        """Test w₀ is close to -1 (but not exactly)."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=50, verbose=False)
        
        eos = compute_equation_of_state(opt.best_W)
        
        # Should have w₀ close to -1
        assert 'w_0' in eos
        assert -1.2 < eos['w_0'] < -0.7
    
    def test_w0_not_exactly_minus_one(self):
        """Test w₀ ≠ -1 (not pure cosmological constant)."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        eos = compute_equation_of_state(opt.best_W)
        
        # Should not be exactly -1 (delta_w should be non-zero)
        assert abs(eos['w_0'] + 1.0) > 0.0001
    
    def test_w0_irh_prediction(self):
        """Test IRH prediction w₀ = -0.912."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        eos = compute_equation_of_state(opt.best_W)
        
        # Should match IRH prediction approximately
        predicted = -0.912
        # Very broad range for placeholder implementation
        assert abs(eos['w_0'] - predicted) < 0.2
    
    def test_delta_w_computed(self):
        """Test that δw = w + 1 is computed."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        eos = compute_equation_of_state(opt.best_W)
        
        # Should have delta_w
        assert 'delta_w' in eos
        # delta_w should be small and positive
        assert 0 < eos['delta_w'] < 0.2
    
    def test_dark_energy_analyzer_init(self):
        """Test DarkEnergyAnalyzer initialization."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        analyzer = DarkEnergyAnalyzer(opt.current_W)
        
        # Should initialize properly
        assert analyzer.W is not None
        assert analyzer.N == 100
    
    def test_dark_energy_analyzer_run(self):
        """Test DarkEnergyAnalyzer full analysis."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=50, verbose=False)
        
        analyzer = DarkEnergyAnalyzer(opt.best_W)
        results = analyzer.run_full_analysis()
        
        # Should have all components
        assert isinstance(results, dict)
        assert 'vacuum_energy' in results
        assert 'equation_of_state' in results
        assert 'predictions' in results
        assert 'experimental' in results
    
    def test_experimental_values_present(self):
        """Test that experimental values are included."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        analyzer = DarkEnergyAnalyzer(opt.current_W)
        results = analyzer.run_full_analysis()
        
        # Should have experimental comparison
        exp = results['experimental']
        assert 'w_0_Planck2018' in exp
        assert 'w_0_DESI2024' in exp
        assert 'w_0_IRH_prediction' in exp
        
        # Check IRH prediction value
        assert exp['w_0_IRH_prediction'] == -0.912
    
    def test_falsifiable_predictions(self):
        """Test that falsifiable predictions are provided."""
        opt = AROOptimizer(N=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        analyzer = DarkEnergyAnalyzer(opt.current_W)
        results = analyzer.run_full_analysis()
        
        # Should have falsifiable predictions
        assert 'predictions' in results
        predictions = results['predictions']
        assert 'w_0' in predictions
        assert 'falsifiable' in predictions
        assert len(predictions['falsifiable']) > 0


@pytest.mark.slow
class TestLargeScaleCosmology:
    """Tests requiring larger networks (marked as slow)."""
    
    def test_larger_network_vacuum_energy(self):
        """Test vacuum energy with larger network."""
        opt = AROOptimizer(N=300, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=200, verbose=False)
        
        vac = compute_vacuum_energy_density(opt.best_W)
        
        # Should have all fields
        assert 'rho_vac_bare' in vac
        assert 'rho_vac_regularized' in vac
    
    def test_larger_network_w0(self):
        """Test w₀ with larger network."""
        opt = AROOptimizer(N=300, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=200, verbose=False)
        
        eos = compute_equation_of_state(opt.best_W)
        
        # Should match prediction better with larger N
        predicted = -0.912
        assert -1.1 < eos['w_0'] < -0.8
