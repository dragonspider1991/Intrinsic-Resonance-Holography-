"""
Tests for IRH v18.0 Additional Physics Modules
==============================================

Tests for:
- electroweak.py: Higgs, W/Z masses, Weinberg angle
- strong_cp.py: Algorithmic axion, θ = 0
- quantum_mechanics.py: Born rule, decoherence, Lindblad

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
    # Electroweak
    HiggsBoson,
    GaugeBosonMasses,
    WeinbergAngle,
    FermiConstant,
    ElectroweakSector,
    EW_EXPERIMENTAL,
    
    # Strong CP
    ThetaAngle,
    PecceiQuinnSymmetry,
    AlgorithmicAxion,
    StrongCPResolution,
    STRONG_CP_CONSTANTS,
    
    # Quantum mechanics
    ElementaryAlgorithmicTransformation,
    QuantumAmplitudeEmergence,
    BornRule,
    Decoherence,
    LindbladEquation,
    EmergentQuantumMechanics,
    
    # Core
    CosmicFixedPoint,
)


# =============================================================================
# Electroweak Tests
# =============================================================================

class TestHiggsBoson:
    """Tests for Higgs boson properties."""
    
    def test_vev_value(self):
        """Test Higgs VEV ≈ 246 GeV."""
        higgs = HiggsBoson()
        vev = higgs.compute_vev()
        
        assert np.isclose(vev["v_GeV"], 246.219651, rtol=1e-4)
    
    def test_mass_value(self):
        """Test Higgs mass ≈ 125 GeV."""
        higgs = HiggsBoson()
        mass = higgs.compute_mass()
        
        assert np.isclose(mass["m_H_GeV"], 125.25, rtol=1e-2)
    
    def test_self_coupling(self):
        """Test Higgs self-coupling is computed."""
        higgs = HiggsBoson()
        coupling = higgs.compute_self_coupling()
        
        assert "lambda" in coupling
        assert coupling["lambda"] > 0


class TestGaugeBosonMasses:
    """Tests for W and Z boson masses."""
    
    def test_w_mass(self):
        """Test W mass ≈ 80.4 GeV."""
        gauge = GaugeBosonMasses()
        w = gauge.compute_w_mass()
        
        assert np.isclose(w["m_W_GeV"], 80.377, rtol=1e-3)
    
    def test_z_mass(self):
        """Test Z mass ≈ 91.2 GeV."""
        gauge = GaugeBosonMasses()
        z = gauge.compute_z_mass()
        
        assert np.isclose(z["m_Z_GeV"], 91.1876, rtol=1e-4)
    
    def test_mass_ratio(self):
        """Test W/Z mass ratio gives ρ ≈ 1."""
        gauge = GaugeBosonMasses()
        ratio = gauge.compute_mass_ratio()
        
        # ρ is close to 1 but has small radiative corrections
        assert np.isclose(ratio["rho"], 1.0, rtol=0.02)


class TestWeinbergAngle:
    """Tests for weak mixing angle."""
    
    def test_sin2_theta_w(self):
        """Test sin²θ_W ≈ 0.231."""
        weinberg = WeinbergAngle()
        angle = weinberg.compute_sin2_theta_w()
        
        assert np.isclose(angle["sin2_theta_W"], 0.23122, rtol=1e-3)
    
    def test_cos2_theta_w(self):
        """Test cos²θ_W = 1 - sin²θ_W."""
        weinberg = WeinbergAngle()
        angle = weinberg.compute_sin2_theta_w()
        
        assert np.isclose(
            angle["sin2_theta_W"] + angle["cos2_theta_W"], 
            1.0, 
            atol=1e-10
        )
    
    def test_from_masses(self):
        """Test sin²θ_W from W/Z mass ratio."""
        weinberg = WeinbergAngle()
        from_masses = weinberg.compute_from_masses()
        
        assert "sin2_theta_W_from_masses" in from_masses


class TestFermiConstant:
    """Tests for Fermi constant."""
    
    def test_G_F_value(self):
        """Test G_F ≈ 1.166 × 10^-5 GeV^-2."""
        fermi = FermiConstant()
        result = fermi.compute_G_F()
        
        # G_F from v = 246 GeV: G_F = 1/(√2 × v²)
        v = EW_EXPERIMENTAL["v_GeV"]
        expected = 1 / (np.sqrt(2) * v**2)
        assert np.isclose(result["G_F"], expected, rtol=1e-3)


class TestElectroweakSector:
    """Integration tests for electroweak sector."""
    
    def test_full_sector(self):
        """Test complete EW sector computation."""
        ew = ElectroweakSector()
        result = ew.compute_full_sector()
        
        assert "higgs" in result
        assert "gauge_bosons" in result
        assert "weinberg_angle" in result
        assert "fermi_constant" in result
    
    def test_consistency(self):
        """Test internal consistency of EW sector."""
        ew = ElectroweakSector()
        consistency = ew.verify_consistency()
        
        assert consistency["m_W_consistent"]
        assert consistency["G_F_consistent"]


# =============================================================================
# Strong CP Tests
# =============================================================================

class TestThetaAngle:
    """Tests for QCD θ-angle."""
    
    def test_effective_theta_zero(self):
        """Test θ_eff = 0 after axion relaxation."""
        theta = ThetaAngle()
        result = theta.compute_effective_theta()
        
        assert result["theta_effective"] == 0.0
    
    def test_precision(self):
        """Test θ precision is < 10^-10."""
        theta = ThetaAngle()
        result = theta.compute_effective_theta()
        
        assert result["precision"] == "< 10^-10"


class TestPecceiQuinnSymmetry:
    """Tests for emergent PQ symmetry."""
    
    def test_symmetry_emergence(self):
        """Test PQ symmetry emerges from cGFT."""
        pq = PecceiQuinnSymmetry()
        result = pq.verify_symmetry_emergence()
        
        assert result["symmetry"] == "U(1)_PQ"
        assert result["spontaneously_broken"]
        assert result["verified"]
    
    def test_breaking_scale(self):
        """Test f_a ~ 10^12 GeV."""
        pq = PecceiQuinnSymmetry()
        result = pq.compute_breaking_scale()
        
        assert result["f_a_GeV"] == 1e12
        assert result["within_bounds"]


class TestAlgorithmicAxion:
    """Tests for algorithmic axion."""
    
    def test_mass(self):
        """Test axion mass ≈ 5.7 μeV."""
        axion = AlgorithmicAxion()
        mass = axion.compute_mass()
        
        assert np.isclose(mass["m_a_muev"], 5.7, rtol=0.1)
    
    def test_photon_coupling(self):
        """Test axion-photon coupling is computed."""
        axion = AlgorithmicAxion()
        coupling = axion.compute_photon_coupling()
        
        assert "g_agamma_GeV_inv" in coupling
    
    def test_dark_matter(self):
        """Test axion dark matter properties."""
        axion = AlgorithmicAxion()
        dm = axion.compute_dark_matter_density()
        
        assert "omega_a_h2" in dm
        assert "is_dark_matter" in dm


class TestStrongCPResolution:
    """Tests for strong CP problem resolution."""
    
    def test_resolution_verified(self):
        """Test strong CP problem is resolved."""
        cp = StrongCPResolution()
        result = cp.verify_resolution()
        
        assert result["resolved"]
        assert result["theta_effective"] == 0.0
    
    def test_neutron_edm(self):
        """Test neutron EDM consistent with experiment."""
        cp = StrongCPResolution()
        edm = cp.compute_neutron_edm()
        
        assert edm["d_n_ecm"] == 0.0
        assert edm["consistent"]
    
    def test_full_analysis(self):
        """Test complete analysis is returned."""
        cp = StrongCPResolution()
        result = cp.get_full_analysis()
        
        assert "theta_angle" in result
        assert "peccei_quinn" in result
        assert "algorithmic_axion" in result


# =============================================================================
# Quantum Mechanics Tests
# =============================================================================

class TestElementaryAlgorithmicTransformation:
    """Tests for EATs."""
    
    def test_identity(self):
        """Test identity EAT."""
        eat = ElementaryAlgorithmicTransformation.identity()
        
        assert np.isclose(eat.amplitude, 1.0)
        assert np.isclose(eat.phase, 0.0)
    
    def test_composition(self):
        """Test EAT composition (phase addition)."""
        eat1 = ElementaryAlgorithmicTransformation(phase=np.pi/4)
        eat2 = ElementaryAlgorithmicTransformation(phase=np.pi/4)
        
        composed = eat1.compose(eat2)
        assert np.isclose(composed.phase, np.pi/2)
    
    def test_inverse(self):
        """Test EAT inverse."""
        eat = ElementaryAlgorithmicTransformation(phase=np.pi/3)
        inv = eat.inverse()
        
        composed = eat.compose(inv)
        # Phase should be 0 (mod 2π)
        assert np.isclose(composed.phase % (2*np.pi), 0.0, atol=1e-10)


class TestBornRule:
    """Tests for Born rule emergence."""
    
    def test_transition_probability(self):
        """Test |ψ_n|² probability."""
        born = BornRule()
        psi = np.array([1, 0], dtype=complex)
        
        p0 = born.compute_transition_probability(psi, 0)
        p1 = born.compute_transition_probability(psi, 1)
        
        assert np.isclose(p0, 1.0)
        assert np.isclose(p1, 0.0)
    
    def test_superposition(self):
        """Test equal superposition gives 50/50."""
        born = BornRule()
        psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
        
        p0 = born.compute_transition_probability(psi, 0)
        p1 = born.compute_transition_probability(psi, 1)
        
        assert np.isclose(p0, 0.5)
        assert np.isclose(p1, 0.5)
    
    def test_born_rule_verification(self):
        """Test Born rule is verified by sampling."""
        born = BornRule()
        psi = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
        
        verification = born.verify_born_rule(psi, num_samples=5000)
        
        assert verification["verified"]
        assert verification["max_deviation"] < 0.05


class TestDecoherence:
    """Tests for decoherence mechanism."""
    
    def test_decoherence_rate(self):
        """Test decoherence rate is computed."""
        dec = Decoherence()
        gamma = dec.compute_decoherence_rate(
            system_size=2,
            environment_size=100
        )
        
        assert gamma > 0
    
    def test_density_matrix_evolution(self):
        """Test off-diagonal elements decay."""
        dec = Decoherence()
        
        # Start with superposition (off-diagonal elements)
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        
        # Evolve
        rho_evolved = dec.evolve_density_matrix(rho, gamma=1.0, dt=1.0)
        
        # Off-diagonal should decrease
        assert np.abs(rho_evolved[0, 1]) < np.abs(rho[0, 1])
    
    def test_measurement(self):
        """Test measurement gives definite outcome."""
        dec = Decoherence()
        
        rho = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        outcome, rho_post = dec.measure(rho)
        
        # Post-measurement should be pure
        assert outcome in [0, 1]
        assert np.isclose(np.trace(rho_post @ rho_post), 1.0, atol=1e-10)


class TestLindbladEquation:
    """Tests for Lindblad dynamics."""
    
    def test_cptp(self):
        """Test evolution preserves CPTP properties."""
        lindblad = LindbladEquation()
        
        # Simple 2-level system
        H = np.zeros((2, 2), dtype=complex)
        L = np.array([[0, 1], [0, 0]], dtype=complex)  # Lowering operator
        
        rho_init = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=complex)
        
        rho_final = lindblad.evolve(
            rho_init, H, [L], [0.1], t_final=0.1, dt=0.01
        )
        
        verification = lindblad.verify_cptp(rho_final)
        assert verification["is_valid_state"]
    
    def test_trace_preservation(self):
        """Test trace is preserved."""
        lindblad = LindbladEquation()
        
        H = np.diag([0, 1]).astype(complex)
        rho_init = np.eye(2, dtype=complex) / 2
        
        rho_final = lindblad.evolve(rho_init, H, [], [], t_final=0.1)
        
        assert np.isclose(np.trace(rho_final), 1.0, atol=1e-6)


class TestEmergentQuantumMechanics:
    """Integration tests for emergent QM."""
    
    def test_summary(self):
        """Test summary is complete."""
        qm = EmergentQuantumMechanics()
        summary = qm.get_summary()
        
        assert "foundation" in summary
        assert "born_rule" in summary
        assert "measurement" in summary
        assert "lindblad" in summary
    
    def test_demonstration(self):
        """Test EAT demonstration works."""
        qm = EmergentQuantumMechanics()
        result = qm.demonstrate_emergence(dim=2, num_eats=50)
        
        assert "collective_amplitude" in result
        assert "born_rule_verified" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
