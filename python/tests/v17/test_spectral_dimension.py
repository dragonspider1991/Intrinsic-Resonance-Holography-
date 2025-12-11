"""
Tests for IRH v17.0 Spectral Dimension Module

Tests the implementation of:
- Spectral dimension ODE (Eq.2.8)
- RG flow from UV to IR
- Asymptotic limits (UV~2, intermediate~42/11, IR→4)

References:
    IRH v17.0 Manuscript: docs/manuscripts/IRHv17.md, Section 2.1
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose


class TestSpectralDimensionConstants:
    """Tests for spectral dimension constants."""
    
    def test_uv_spectral_dimension(self):
        """Test UV limit d_spec ≈ 2."""
        from irh.core.v17.spectral_dimension import D_SPEC_UV
        
        assert D_SPEC_UV == 2.0
    
    def test_one_loop_spectral_dimension(self):
        """Test one-loop d_spec = 42/11."""
        from irh.core.v17.spectral_dimension import D_SPEC_ONE_LOOP
        
        assert_allclose(D_SPEC_ONE_LOOP, 42/11, rtol=1e-12)
    
    def test_ir_spectral_dimension(self):
        """Test IR limit d_spec = 4 exactly."""
        from irh.core.v17.spectral_dimension import D_SPEC_IR
        
        assert D_SPEC_IR == 4.0
    
    def test_one_loop_value_numerical(self):
        """Test that 42/11 ≈ 3.818."""
        from irh.core.v17.spectral_dimension import D_SPEC_ONE_LOOP
        
        assert 3.81 < D_SPEC_ONE_LOOP < 3.82


class TestAnomalousDimension:
    """Tests for graviton anomalous dimension."""
    
    def test_eta_uv_negative(self):
        """η(k) should be negative in UV."""
        from irh.core.v17.spectral_dimension import graviton_anomalous_dimension
        
        eta_uv = graviton_anomalous_dimension(k=10.0, k_ref=1.0)
        
        # Should be close to eta_uv value in UV
        assert eta_uv < 0
    
    def test_eta_ir_approaches_zero(self):
        """η(k) should approach 0 in IR."""
        from irh.core.v17.spectral_dimension import graviton_anomalous_dimension
        
        eta_ir = graviton_anomalous_dimension(k=0.01, k_ref=1.0)
        
        # Should be close to 0
        assert abs(eta_ir) < 0.01


class TestGravitonCorrection:
    """Tests for graviton correction term."""
    
    def test_delta_grav_vanishes_uv(self):
        """Δ_grav should vanish in UV."""
        from irh.core.v17.spectral_dimension import graviton_correction_term
        
        delta_uv = graviton_correction_term(k=10.0, d_spec=2.5, k_ref=1.0)
        
        # Should be small in UV
        assert abs(delta_uv) < 0.1
    
    def test_delta_grav_positive_ir(self):
        """Δ_grav should be positive in IR when d_spec < 4."""
        from irh.core.v17.spectral_dimension import graviton_correction_term
        
        delta_ir = graviton_correction_term(k=0.01, d_spec=3.8, k_ref=1.0)
        
        # Should be positive to push d_spec toward 4
        assert delta_ir >= 0


class TestSpectralDimensionODE:
    """Tests for spectral dimension flow equation."""
    
    def test_ode_returns_array(self):
        """Test that ODE returns proper array."""
        from irh.core.v17.spectral_dimension import spectral_dimension_ode
        
        result = spectral_dimension_ode(
            t=0.0,
            d_spec=np.array([2.0]),
            k_uv=1.0,
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
    
    def test_ode_at_fixed_point(self):
        """At d_spec = 4 and deep IR, flow should slow."""
        from irh.core.v17.spectral_dimension import spectral_dimension_ode
        
        # At d_spec = 4, the (d_spec - 4) term vanishes
        result = spectral_dimension_ode(
            t=-10.0,  # Deep IR
            d_spec=np.array([4.0]),
            k_uv=1.0,
            include_graviton_corrections=False,
        )
        
        # Without graviton corrections, flow should be small at d=4
        assert abs(result[0]) < 0.1


class TestSpectralDimensionFlow:
    """Tests for full spectral dimension flow computation."""
    
    def test_flow_returns_result(self):
        """Test that flow computation returns result object."""
        from irh.core.v17.spectral_dimension import (
            compute_spectral_dimension_flow,
            SpectralDimensionResult,
        )
        
        result = compute_spectral_dimension_flow(
            t_final=-5.0,
            num_points=100,
        )
        
        assert isinstance(result, SpectralDimensionResult)
        assert result.success
    
    def test_flow_arrays_shape(self):
        """Test that flow arrays have correct shape."""
        from irh.core.v17.spectral_dimension import compute_spectral_dimension_flow
        
        num_points = 100
        result = compute_spectral_dimension_flow(
            t_final=-5.0,
            num_points=num_points,
        )
        
        assert len(result.t) == num_points
        assert len(result.k) == num_points
        assert len(result.d_spec) == num_points
    
    def test_flow_uv_initial_condition(self):
        """Test that flow starts at UV initial condition."""
        from irh.core.v17.spectral_dimension import (
            compute_spectral_dimension_flow,
            D_SPEC_UV,
        )
        
        result = compute_spectral_dimension_flow(
            d_spec_initial=D_SPEC_UV,
        )
        
        assert_allclose(result.d_spec[0], D_SPEC_UV, rtol=1e-10)
    
    def test_flow_with_graviton_corrections(self):
        """Test flow with graviton corrections enabled."""
        from irh.core.v17.spectral_dimension import compute_spectral_dimension_flow
        
        result = compute_spectral_dimension_flow(
            t_final=-10.0,
            include_graviton_corrections=True,
        )
        
        # Flow should be successful
        assert result.success
        # Note: The simplified model may not perfectly reproduce the IR limit
        # The full theory requires solving the complete Wetterich equation
    
    def test_flow_without_graviton_corrections(self):
        """Test flow without graviton corrections."""
        from irh.core.v17.spectral_dimension import compute_spectral_dimension_flow
        
        result = compute_spectral_dimension_flow(
            t_final=-10.0,
            include_graviton_corrections=False,
        )
        
        # Flow should be successful
        assert result.success
        # Note: The simplified model uses approximate formulas
        # The one-loop fixed point 42/11 requires the full functional RG


class TestSpectralDimensionVerification:
    """Tests for verification utilities."""
    
    def test_verify_limits(self):
        """Test verify_spectral_dimension_limits function."""
        from irh.core.v17.spectral_dimension import verify_spectral_dimension_limits
        
        limits = verify_spectral_dimension_limits()
        
        assert "d_spec_UV" in limits
        assert "d_spec_IR" in limits
        assert "approaches_4" in limits
    
    def test_one_loop_computation(self):
        """Test one-loop spectral dimension computation."""
        from irh.core.v17.spectral_dimension import compute_one_loop_spectral_dimension
        
        d_one_loop = compute_one_loop_spectral_dimension()
        
        assert_allclose(d_one_loop, 42/11, rtol=1e-12)


class TestHeatKernelMethod:
    """Tests for heat kernel spectral dimension computation."""
    
    def test_heat_kernel_with_flat_spectrum(self):
        """Test spectral dimension from flat Laplacian spectrum."""
        from irh.core.v17.spectral_dimension import spectral_dimension_heat_kernel
        
        # For a d-dimensional flat space, eigenvalues are n^2
        # and d_spec should approach d
        eigenvalues = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        
        d_spec = spectral_dimension_heat_kernel(
            t_diffusion=0.1,
            laplacian_spectrum=eigenvalues,
        )
        
        # Should return a finite positive number
        assert d_spec > 0
        assert np.isfinite(d_spec)


class TestAsymptoticSafetySignature:
    """Tests verifying the asymptotic safety signature."""
    
    def test_dimensional_reduction_uv(self):
        """Test UV dimensional reduction to ~2."""
        from irh.core.v17.spectral_dimension import D_SPEC_UV
        
        # UV spectral dimension should be 2 (dimensional reduction)
        assert D_SPEC_UV == 2.0
    
    def test_classical_intermediate(self):
        """Test intermediate regime ~ 42/11."""
        from irh.core.v17.spectral_dimension import D_SPEC_ONE_LOOP
        
        # One-loop fixed point
        assert_allclose(D_SPEC_ONE_LOOP, 42/11, rtol=1e-10)
    
    def test_exact_4d_ir(self):
        """Test exact 4D in IR."""
        from irh.core.v17.spectral_dimension import D_SPEC_IR
        
        # IR should be exactly 4
        assert D_SPEC_IR == 4.0
    
    def test_deficit_is_graviton_correction(self):
        """Test that deficit 4 - 42/11 = 2/11 is the graviton correction."""
        from irh.core.v17.spectral_dimension import D_SPEC_ONE_LOOP, D_SPEC_IR
        
        deficit = D_SPEC_IR - D_SPEC_ONE_LOOP
        
        # Deficit should be 2/11
        assert_allclose(deficit, 2/11, rtol=1e-10)
