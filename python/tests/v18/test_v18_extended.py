"""
Tests for IRH v18.0 Additional Modules
======================================

Tests for:
- dark_energy.py: Holographic Hum, w₀, vacuum energy
- emergent_spacetime.py: Lorentzian signature, time emergence
- emergent_qft.py: Particle spectrum, effective Lagrangian

References:
    docs/manuscripts/IRHv18.md
"""

import pytest
import numpy as np
import sys
import os

# Add the source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from irh.core.v18 import (
    # Dark energy
    HolographicHum,
    DarkEnergyEquationOfState,
    VacuumEnergyDensity,
    CosmologicalEvolution,
    DarkEnergyModule,
    compute_dark_energy_summary,
    DARK_ENERGY_OBSERVATIONS,
    
    # Emergent spacetime
    LorentzianSignatureEmergence,
    TimeEmergence,
    DiffeomorphismInvariance,
    EmergentSpacetime,
    compute_spacetime_summary,
    
    # Emergent QFT
    ParticleType,
    GaugeGroup,
    GravitonIdentification,
    GaugeBosonIdentification,
    FermionIdentification,
    EffectiveLagrangian,
    EmergentQFT,
    compute_qft_summary,
    
    # Core
    CosmicFixedPoint,
)


# =============================================================================
# Dark Energy Tests
# =============================================================================

class TestHolographicHum:
    """Tests for Holographic Hum."""
    
    def test_hum_amplitude(self):
        """Test Hum amplitude is computed."""
        hum = HolographicHum()
        result = hum.compute_hum_amplitude()
        
        assert "amplitude" in result
        assert result["amplitude"] > 0
    
    def test_running_hum(self):
        """Test running Hum at different scales."""
        hum = HolographicHum()
        
        # UV limit (k → ∞)
        result_uv = hum.compute_running_hum(k=100)
        
        # IR limit (k → 0)
        result_ir = hum.compute_running_hum(k=0.01)
        
        assert result_ir["mu_running"] < result_uv["mu_running"]


class TestDarkEnergyEquationOfState:
    """Tests for dark energy equation of state."""
    
    def test_w0_value(self):
        """Test w₀ ≈ -1 (with small correction)."""
        w_eos = DarkEnergyEquationOfState()
        result = w_eos.compute_w0()
        
        # w₀ should be very close to -1
        assert np.isclose(result["w0"], -1.0, atol=0.01)
        assert result["w0"] > -1.0  # Slightly above -1
    
    def test_w0_formula(self):
        """Test w₀ formula structure."""
        w_eos = DarkEnergyEquationOfState()
        result = w_eos.compute_w0()
        
        # w₀ = -1 + C_H/(3×8π²)
        C_H = result["C_H"]
        expected_correction = C_H / (3 * 8 * np.pi**2)
        
        assert np.isclose(result["w0_correction"], expected_correction)
    
    def test_observational_consistency(self):
        """Test consistency with Planck observations."""
        w_eos = DarkEnergyEquationOfState()
        result = w_eos.is_consistent_with_observations()
        
        # Should be within 2σ of observations
        assert result["within_2sigma"]


class TestVacuumEnergyDensity:
    """Tests for vacuum energy density."""
    
    def test_lambda_star(self):
        """Test Λ_* is computed."""
        vacuum = VacuumEnergyDensity()
        result = vacuum.compute_lambda_star()
        
        assert "Lambda_star" in result
        assert result["Lambda_star"] > 0
    
    def test_vacuum_energy(self):
        """Test vacuum energy is positive."""
        vacuum = VacuumEnergyDensity()
        result = vacuum.compute_vacuum_energy()
        
        assert result["rho_vacuum"] > 0


class TestCosmologicalEvolution:
    """Tests for cosmological evolution."""
    
    def test_hubble_today(self):
        """Test H(z=0) ≈ H₀."""
        cosmo = CosmologicalEvolution()
        result = cosmo.compute_hubble_parameter(z=0)
        
        H0 = DARK_ENERGY_OBSERVATIONS["H0_km_s_Mpc"]
        assert np.isclose(result["H_z"], H0, rtol=0.01)
    
    def test_deceleration_today(self):
        """Test q₀ < 0 (accelerating expansion)."""
        cosmo = CosmologicalEvolution()
        result = cosmo.compute_deceleration_parameter(z=0)
        
        assert result["q"] < 0
        assert result["accelerating"]


class TestDarkEnergyModule:
    """Integration tests for dark energy module."""
    
    def test_full_analysis(self):
        """Test complete analysis is returned."""
        module = DarkEnergyModule()
        result = module.compute_full_analysis()
        
        assert "holographic_hum" in result
        assert "equation_of_state" in result
        assert "vacuum_energy" in result
        assert "cosmological_constant" in result


# =============================================================================
# Emergent Spacetime Tests
# =============================================================================

class TestLorentzianSignatureEmergence:
    """Tests for Lorentzian signature emergence."""
    
    def test_metric_signature(self):
        """Test signature is (-,+,+,+)."""
        lorentz = LorentzianSignatureEmergence()
        result = lorentz.get_metric_signature()
        
        assert result["signature"] == (-1, +1, +1, +1)
        assert result["lorentzian"]
        assert result["dimension"] == 4
    
    def test_ssb_mechanism(self):
        """Test SSB mechanism is verified."""
        lorentz = LorentzianSignatureEmergence()
        result = lorentz.verify_ssb_mechanism()
        
        assert result["verified"]
        assert result["symmetry_broken"] == "Z_2 (complex conjugation)"
    
    def test_effective_metric(self):
        """Test Minkowski metric is returned."""
        lorentz = LorentzianSignatureEmergence()
        eta = lorentz.compute_effective_metric()
        
        # η_μν = diag(-1, +1, +1, +1)
        assert eta[0, 0] == -1
        assert eta[1, 1] == +1
        assert eta[2, 2] == +1
        assert eta[3, 3] == +1


class TestTimeEmergence:
    """Tests for time emergence."""
    
    def test_arrow_of_time(self):
        """Test arrow of time is emergent."""
        time = TimeEmergence()
        result = time.get_arrow_of_time()
        
        assert result["emergent"]
        assert not result["fundamental"]
        assert result["thermodynamic"]
    
    def test_timelike_progression_vector(self):
        """Test TPV is timelike."""
        time = TimeEmergence()
        result = time.compute_timelike_progression_vector()
        
        assert result["is_timelike"]
        assert result["norm_squared"] == -1.0
    
    def test_proper_time(self):
        """Test proper time computation."""
        time = TimeEmergence()
        
        # Stationary worldline (only time changes)
        worldline = [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)]
        tau = time.compute_proper_time(worldline)
        
        assert tau == 2.0  # Δτ = Δt for stationary observer


class TestDiffeomorphismInvariance:
    """Tests for diffeomorphism invariance."""
    
    def test_theorem_2_8(self):
        """Test Theorem 2.8 verification."""
        diff = DiffeomorphismInvariance()
        result = diff.verify_theorem_2_8()
        
        assert result["verified"]
        assert "Reparametrization Invariance" in result["theorem"]
    
    def test_diff_group(self):
        """Test Diff(M⁴) description."""
        diff = DiffeomorphismInvariance()
        result = diff.describe_diff_group()
        
        assert result["group"] == "Diff(M⁴)"
        assert result["dimension"] == "infinite"
    
    def test_general_covariance(self):
        """Test all physics is covariant."""
        diff = DiffeomorphismInvariance()
        result = diff.verify_general_covariance()
        
        assert all(result.values())


class TestEmergentSpacetime:
    """Integration tests for emergent spacetime."""
    
    def test_full_analysis(self):
        """Test complete analysis is returned."""
        spacetime = EmergentSpacetime()
        result = spacetime.get_full_analysis()
        
        assert "signature" in result
        assert "arrow_of_time" in result
        assert "diffeomorphisms" in result
    
    def test_all_properties_verified(self):
        """Test all properties are verified."""
        spacetime = EmergentSpacetime()
        result = spacetime.verify_all_properties()
        
        assert result["all_verified"]
        assert result["lorentzian_signature"]
        assert result["four_dimensional"]


# =============================================================================
# Emergent QFT Tests
# =============================================================================

class TestGravitonIdentification:
    """Tests for graviton identification."""
    
    def test_graviton_properties(self):
        """Test graviton is massless spin-2."""
        graviton = GravitonIdentification()
        result = graviton.get_properties()
        
        assert result["spin"] == 2
        assert result["mass_GeV"] == 0.0
        assert result["massless"]
    
    def test_helicity_states(self):
        """Test graviton has ±2 helicities."""
        graviton = GravitonIdentification()
        helicities = graviton.get_helicity_states()
        
        assert +2 in helicities
        assert -2 in helicities
        assert len(helicities) == 2


class TestGaugeBosonIdentification:
    """Tests for gauge boson identification."""
    
    def test_gluons(self):
        """Test 8 massless gluons."""
        gauge = GaugeBosonIdentification()
        result = gauge.get_gluons()
        
        assert result["spin"] == 1
        assert result["massless"]
        assert result["generators"] == 8
    
    def test_weak_bosons(self):
        """Test W and Z masses."""
        gauge = GaugeBosonIdentification()
        result = gauge.get_weak_bosons()
        
        assert np.isclose(result["W_boson"]["mass_GeV"], 80.377, rtol=0.01)
        assert np.isclose(result["Z_boson"]["mass_GeV"], 91.1876, rtol=0.01)
    
    def test_photon(self):
        """Test photon is massless."""
        gauge = GaugeBosonIdentification()
        result = gauge.get_photon()
        
        assert result["massless"]
        assert result["mass_GeV"] == 0.0
    
    def test_total_gauge_bosons(self):
        """Test 12 total gauge bosons."""
        gauge = GaugeBosonIdentification()
        result = gauge.get_all_gauge_bosons()
        
        assert result["total_gauge_bosons"] == 12


class TestFermionIdentification:
    """Tests for fermion identification."""
    
    def test_three_quark_generations(self):
        """Test 3 quark generations."""
        fermion = FermionIdentification()
        result = fermion.get_quarks()
        
        assert result["n_generations"] == 3
        assert "generation_1" in result
        assert "generation_2" in result
        assert "generation_3" in result
    
    def test_three_lepton_generations(self):
        """Test 3 lepton generations."""
        fermion = FermionIdentification()
        result = fermion.get_leptons()
        
        assert result["n_generations"] == 3
        assert result["neutrino_nature"] == "Majorana"
    
    def test_fermion_emergence(self):
        """Test fermions emerge as VWPs."""
        fermion = FermionIdentification()
        result = fermion.get_all_fermions()
        
        assert result["emergence"] == "Vortex Wave Patterns (VWPs)"
        assert result["spin"] == 0.5


class TestEffectiveLagrangian:
    """Tests for effective Lagrangian."""
    
    def test_gravity_sector(self):
        """Test gravity Lagrangian."""
        lagrangian = EffectiveLagrangian()
        result = lagrangian.get_gravity_sector()
        
        assert "Einstein_Hilbert" in result
        assert "cosmological_constant" in result
    
    def test_gauge_sector(self):
        """Test gauge Lagrangian."""
        lagrangian = EffectiveLagrangian()
        result = lagrangian.get_gauge_sector()
        
        assert "SU3" in result
        assert "SU2" in result
        assert "U1" in result
    
    def test_full_lagrangian(self):
        """Test complete Lagrangian."""
        lagrangian = EffectiveLagrangian()
        result = lagrangian.get_full_lagrangian()
        
        assert "gravity" in result
        assert "gauge" in result
        assert "fermion" in result
        assert "higgs" in result
        assert "yukawa" in result


class TestEmergentQFT:
    """Integration tests for emergent QFT."""
    
    def test_particle_spectrum(self):
        """Test complete particle spectrum."""
        qft = EmergentQFT()
        result = qft.get_particle_spectrum()
        
        assert "graviton" in result
        assert "gauge_bosons" in result
        assert "fermions" in result
    
    def test_standard_model_verification(self):
        """Test Standard Model structure."""
        qft = EmergentQFT()
        result = qft.verify_standard_model()
        
        assert result["gauge_group"]
        assert result["three_generations"]
        assert result["ewsb"]
        assert result["ckm_matrix"]
        assert result["pmns_matrix"]
    
    def test_full_analysis(self):
        """Test complete QFT analysis."""
        qft = EmergentQFT()
        result = qft.get_full_analysis()
        
        assert "particle_spectrum" in result
        assert "effective_lagrangian" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
