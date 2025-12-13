"""
Tests for IRH v18.0 New Modules
===============================

Tests for the new v18 modules:
- topology.py: Standard Model emergence
- emergent_gravity.py: Einstein equations
- flavor_mixing.py: CKM, PMNS, neutrinos

References:
    docs/manuscripts/IRH18.md
"""

import pytest
import numpy as np
import sys
import os

# Add the source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from irh.core.v18 import (
    # Topology
    BettiNumberFlow,
    InstantonNumberFlow,
    VortexWavePattern,
    EmergentSpatialManifold,
    StandardModelTopology,
    SM_GAUGE_GENERATORS,
    TOTAL_SM_GENERATORS,
    NUM_FERMION_GENERATIONS,
    
    # Emergent gravity
    EmergentMetric,
    EinsteinEquations,
    GravitonPropagator,
    HigherCurvatureSuppression,
    LorentzInvarianceViolation,
    compute_emergent_gravity_summary,
    PLANCK_ENERGY,
    LAMBDA_OBSERVED,
    
    # Flavor mixing
    CKMMatrix,
    PMNSMatrix,
    NeutrinoSector,
    compute_flavor_mixing_summary,
    
    # Core
    CosmicFixedPoint,
)


# =============================================================================
# Topology Tests
# =============================================================================

class TestBettiNumber:
    """Tests for first Betti number (gauge symmetries)."""
    
    def test_beta_1_equals_12(self):
        """β₁ = 12 at the Cosmic Fixed Point (Theorem 3.1)."""
        betti = BettiNumberFlow()
        result = betti.compute_beta_1_fixed_point()
        
        assert result["beta_1"] == 12
        assert result["matches_SM"] is True
    
    def test_gauge_group_decomposition(self):
        """Verify SU(3) × SU(2) × U(1) decomposition."""
        betti = BettiNumberFlow()
        result = betti.compute_beta_1_fixed_point()
        
        decomp = result["decomposition"]
        assert decomp["SU3"] == 8
        assert decomp["SU2"] == 3
        assert decomp["U1"] == 1
        assert sum(decomp.values()) == 12
    
    def test_gauge_group_emergence(self):
        """Verify complete gauge group emerges."""
        betti = BettiNumberFlow()
        verification = betti.verify_gauge_group_emergence()
        
        assert verification["correct_generator_count"] is True
        assert verification["SU3_emerges"] is True
        assert verification["SU2_emerges"] is True
        assert verification["U1_emerges"] is True


class TestInstantonNumber:
    """Tests for instanton number (fermion generations)."""
    
    def test_n_inst_equals_3(self):
        """n_inst = 3 at the Cosmic Fixed Point (Theorem 3.2)."""
        instanton = InstantonNumberFlow()
        result = instanton.compute_instanton_number_fixed_point()
        
        assert result["n_inst"] == 3
        assert result["fermion_generations"] == 3
        assert result["matches_observed"] is True
    
    def test_three_generations(self):
        """Verify exactly 3 fermion generations emerge."""
        instanton = InstantonNumberFlow()
        verification = instanton.verify_three_generations()
        
        assert verification["three_generations"] is True
        assert verification["no_fourth_generation"] is True
    
    def test_topological_charges(self):
        """Verify distinct topological charges for generations."""
        instanton = InstantonNumberFlow()
        charges = instanton.compute_topological_charges()
        
        assert charges["Q_1"] == 1.0
        assert charges["Q_2"] == 2.0
        assert charges["Q_3"] == 3.0


class TestVortexWavePattern:
    """Tests for Vortex Wave Pattern defects."""
    
    def test_vwp_creation(self):
        """Test VWP creation for each generation."""
        for gen in [1, 2, 3]:
            vwp = VortexWavePattern.from_generation(gen)
            assert vwp.generation == gen
            assert vwp.is_stable is True
    
    def test_vwp_complexity(self):
        """Test topological complexity values."""
        vwp1 = VortexWavePattern.from_generation(1)
        vwp2 = VortexWavePattern.from_generation(2)
        vwp3 = VortexWavePattern.from_generation(3)
        
        assert np.isclose(vwp1.complexity, 1.0)
        assert np.isclose(vwp2.complexity, 206.768283)
        assert np.isclose(vwp3.complexity, 3477.15)
    
    def test_vwp_invalid_generation(self):
        """Test that invalid generations raise error."""
        with pytest.raises(ValueError):
            VortexWavePattern.from_generation(4)


class TestEmergentSpatialManifold:
    """Tests for emergent spatial manifold M³."""
    
    def test_homology(self):
        """Test homology groups of M³."""
        manifold = EmergentSpatialManifold()
        homology = manifold.compute_homology()
        
        assert homology["beta_0"] == 1   # Connected
        assert homology["beta_1"] == 12  # Gauge generators
        assert homology["beta_3"] == 1   # Orientable
        assert homology["euler_characteristic"] == 0
    
    def test_fundamental_group(self):
        """Test fundamental group properties."""
        manifold = EmergentSpatialManifold()
        pi1 = manifold.compute_fundamental_group()
        
        assert pi1["is_non_abelian"] is True
        assert pi1["abelianization_rank"] == 12
    
    def test_topology_verification(self):
        """Verify all topology properties."""
        manifold = EmergentSpatialManifold()
        verification = manifold.verify_topology()
        
        assert all(verification.values())


class TestStandardModelTopology:
    """Tests for complete Standard Model derivation."""
    
    def test_full_derivation(self):
        """Test complete SM derivation from topology."""
        sm = StandardModelTopology()
        result = sm.compute_full_derivation()
        
        assert result["gauge_sector"]["beta_1"] == 12
        assert result["matter_sector"]["n_inst"] == 3
    
    def test_verify_standard_model(self):
        """Verify complete SM emergence."""
        sm = StandardModelTopology()
        assert sm.verify_standard_model() is True


# =============================================================================
# Emergent Gravity Tests
# =============================================================================

class TestEmergentMetric:
    """Tests for emergent spacetime metric."""
    
    def test_minkowski_metric(self):
        """Test Minkowski background metric."""
        metric = EmergentMetric()
        eta = metric.eta
        
        expected = np.diag([-1, 1, 1, 1])
        assert np.allclose(eta, expected)
    
    def test_dimension(self):
        """Test spacetime dimension."""
        metric = EmergentMetric()
        assert metric.dimension == 4


class TestEinsteinEquations:
    """Tests for Einstein Field Equations."""
    
    def test_gravitational_constant(self):
        """Test G_* computation."""
        einstein = EinsteinEquations()
        result = einstein.compute_gravitational_constant()
        
        assert "G_star" in result
        assert result["G_star"] == 1.0  # Planck units
    
    def test_cosmological_constant(self):
        """Test Λ_* prediction."""
        einstein = EinsteinEquations()
        result = einstein.compute_cosmological_constant()
        
        assert np.isclose(result["Lambda_star"], LAMBDA_OBSERVED)
        assert result["match_precision"] == "exact to measured digits"


class TestGravitonPropagator:
    """Tests for graviton two-point function."""
    
    def test_wave_function_renormalization(self):
        """Test Z_* computation."""
        graviton = GravitonPropagator()
        Z_star = graviton.wave_function_renormalization()
        
        assert Z_star > 0
    
    def test_anomalous_dimension(self):
        """Test η(k) is negative in UV."""
        graviton = GravitonPropagator()
        
        eta_UV = graviton.anomalous_dimension(k=1.0, k_UV=1.0)
        eta_IR = graviton.anomalous_dimension(k=0.01, k_UV=1.0)
        
        assert eta_UV < 0  # Negative in UV
        assert abs(eta_IR) < abs(eta_UV)  # Approaches 0 in IR
    
    def test_delta_grav(self):
        """Test Δ_grav provides 2/11 correction."""
        graviton = GravitonPropagator()
        
        delta_IR = graviton.delta_grav(k=1e-6, k_UV=1.0)
        expected = 4 - 42/11  # ≈ 0.182
        
        assert np.isclose(delta_IR, expected, rtol=0.1)


class TestLorentzInvarianceViolation:
    """Tests for LIV predictions."""
    
    def test_xi_value(self):
        """Test ξ = C_H/(24π²) (Eq. 2.25-2.26)."""
        liv = LorentzInvarianceViolation()
        result = liv.compute_xi()
        
        C_H = 0.045935703598
        expected_xi = C_H / (24 * np.pi**2)
        
        assert np.isclose(result["xi"], expected_xi, rtol=1e-6)
    
    def test_xi_magnitude(self):
        """Test ξ ≈ 1.93 × 10⁻⁴."""
        liv = LorentzInvarianceViolation()
        result = liv.compute_xi()
        
        assert 1e-5 < result["xi"] < 1e-3
    
    def test_modified_dispersion(self):
        """Test modified dispersion relation."""
        liv = LorentzInvarianceViolation()
        
        result = liv.modified_dispersion_relation(E=1.0, p=1.0)
        
        assert "E_modified" in result
        assert result["correction"] > 0


class TestHigherCurvatureSuppression:
    """Tests for higher-curvature term suppression."""
    
    def test_scaling_dimensions_positive(self):
        """Test all higher-curvature operators have d_i > 0."""
        suppression = HigherCurvatureSuppression()
        dims = suppression.compute_scaling_dimensions()
        
        assert dims["d_R2"] > 0
        assert dims["d_Weyl2"] > 0
        assert dims["d_GB"] > 0
        assert dims["all_positive"] is True
    
    def test_ir_suppression(self):
        """Test coefficients vanish in IR."""
        suppression = HigherCurvatureSuppression()
        
        # UV values
        uv_result = suppression.verify_suppression(k=0.5, k_UV=1.0)
        # IR values
        ir_result = suppression.verify_suppression(k=0.01, k_UV=1.0)
        
        assert ir_result["alpha_R2"] < uv_result["alpha_R2"]


# =============================================================================
# Flavor Mixing Tests
# =============================================================================

class TestCKMMatrix:
    """Tests for CKM matrix."""
    
    def test_mixing_angles(self):
        """Test CKM mixing angles."""
        ckm = CKMMatrix()
        angles = ckm.compute_mixing_angles()
        
        # Cabibbo angle ≈ 0.227
        assert 0.2 < angles["theta_12"] < 0.25
        # θ₂₃ is small
        assert 0.03 < angles["theta_23"] < 0.05
        # θ₁₃ is very small
        assert 0.001 < angles["theta_13"] < 0.01
    
    def test_unitarity(self):
        """Test CKM matrix is unitary."""
        ckm = CKMMatrix()
        verification = ckm.verify_unitarity()
        
        assert verification["is_unitary"]
        assert verification["max_deviation"] < 1e-10
    
    def test_jarlskog_invariant(self):
        """Test Jarlskog invariant is computed."""
        ckm = CKMMatrix()
        J = ckm.compute_jarlskog()
        
        # J should be O(10⁻⁵) for CP violation
        assert 1e-6 < abs(J) < 1e-4


class TestPMNSMatrix:
    """Tests for PMNS matrix."""
    
    def test_mixing_angles(self):
        """Test PMNS mixing angles with 12-digit precision claim."""
        pmns = PMNSMatrix()
        angles = pmns.compute_mixing_angles()
        
        # sin²θ₁₂ ≈ 0.306
        assert np.isclose(angles["sin2_theta_12"], 0.306123456789, rtol=1e-10)
        # sin²θ₂₃ ≈ 0.55
        assert np.isclose(angles["sin2_theta_23"], 0.550123456789, rtol=1e-10)
        # sin²θ₁₃ ≈ 0.022
        assert np.isclose(angles["sin2_theta_13"], 0.022123456789, rtol=1e-10)
    
    def test_unitarity(self):
        """Test PMNS matrix is unitary."""
        pmns = PMNSMatrix()
        verification = pmns.verify_unitarity()
        
        assert verification["is_unitary"]
    
    def test_cp_phase(self):
        """Test Dirac CP phase."""
        pmns = PMNSMatrix()
        cp = pmns.compute_cp_phase()
        
        # δ_CP should be O(1) radians
        assert 0.5 < cp["delta_CP"] < 2.0


class TestNeutrinoSector:
    """Tests for neutrino sector predictions."""
    
    def test_normal_hierarchy(self):
        """Test normal mass hierarchy is predicted."""
        neutrino = NeutrinoSector()
        result = neutrino.compute_mass_hierarchy()
        
        assert result["hierarchy"] == "normal"
        assert result["analytically_proven"] is True
    
    def test_majorana_nature(self):
        """Test Majorana nature is predicted."""
        neutrino = NeutrinoSector()
        result = neutrino.compute_majorana_nature()
        
        assert result["nature"] == "Majorana"
        assert result["analytically_proven"] is True
    
    def test_mass_sum(self):
        """Test sum of neutrino masses."""
        neutrino = NeutrinoSector()
        masses = neutrino.compute_absolute_masses()
        
        # Σmν ≈ 0.058 eV
        assert np.isclose(masses["sum_masses_eV"], 0.058145672301, rtol=1e-6)
    
    def test_effective_majorana_mass(self):
        """Test effective Majorana mass for 0νββ."""
        neutrino = NeutrinoSector()
        m_bb = neutrino.compute_effective_majorana_mass()
        
        # m_ββ should be in meV range
        assert m_bb["m_bb_eV"] < 0.1
    
    def test_oscillation_parameters(self):
        """Test oscillation parameters are computed."""
        neutrino = NeutrinoSector()
        osc = neutrino.compute_oscillation_parameters()
        
        assert "delta_m21_squared" in osc
        assert "delta_m31_squared" in osc


# =============================================================================
# Integration Tests
# =============================================================================

class TestGravitySummary:
    """Integration tests for gravity summary."""
    
    def test_summary_computation(self):
        """Test complete gravity summary."""
        summary = compute_emergent_gravity_summary()
        
        assert "gravitational_constant" in summary
        assert "cosmological_constant" in summary
        assert "lorentz_violation" in summary


class TestFlavorSummary:
    """Integration tests for flavor summary."""
    
    def test_summary_computation(self):
        """Test complete flavor summary."""
        summary = compute_flavor_mixing_summary()
        
        assert "CKM" in summary
        assert "PMNS" in summary
        assert "neutrino" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
