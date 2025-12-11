"""
Dark Energy and Holographic Hum for IRH v18.0
==============================================

Implements the dark energy predictions from the cGFT:
- Holographic Hum (vacuum energy from fixed point)
- Dark energy equation of state w₀
- Running vacuum energy
- Cosmological evolution

THEORETICAL COMPLIANCE:
    This implementation strictly follows docs/manuscripts/IRHv18.md
    - Section 2.3: Dynamically Quantized Holographic Hum
    - Section 2.3.1-2.3.3: Holographic Hum and w₀
    - Eq. 2.23-2.24: Exact w₀ derivation

Key Results:
    - w₀ = -1 + C_H/(3×8π²) ≈ -0.9998
    - Λ_* emerges from fixed-point vacuum energy
    - Dark energy is dynamical, not a bare cosmological constant

References:
    docs/manuscripts/IRHv18.md:
        - §2.3: The Holographic Hum
        - §2.3.3: Equation of State w₀
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from .rg_flow import CosmicFixedPoint, PI_SQUARED


# =============================================================================
# Constants
# =============================================================================

# Universal constant
C_H = 0.045935703598

# Cosmological observations (Planck 2018)
DARK_ENERGY_OBSERVATIONS = {
    "w0_planck": -1.03,           # Planck 2018 constraint
    "w0_uncertainty": 0.03,       # 1σ uncertainty
    "Omega_Lambda": 0.6847,       # Dark energy density fraction
    "Omega_m": 0.3153,            # Matter density fraction
    "H0_km_s_Mpc": 67.4,          # Hubble constant
}


# =============================================================================
# Holographic Hum
# =============================================================================

@dataclass
class HolographicHum:
    """
    The Holographic Hum - fixed-point vacuum energy.
    
    The Holographic Hum represents the vacuum energy density at the
    Cosmic Fixed Point. It is not a bare cosmological constant but
    emerges dynamically from the optimization of algorithmic coherence.
    
    The Hum encodes the "zero-point energy" of the informational substrate,
    which manifests as dark energy in the emergent spacetime.
    
    References:
        IRHv18.md §2.3.1: Holographic Hum as fixed-point vacuum
        IRHv18.md Eq. 2.20: Hum definition
    """
    
    fixed_point: CosmicFixedPoint = field(default_factory=CosmicFixedPoint)
    
    def compute_hum_amplitude(self) -> Dict[str, float]:
        """
        Compute the Holographic Hum amplitude.
        
        The Hum amplitude is determined by the fixed-point couplings.
        
        Returns:
            Dictionary with Hum properties
        """
        fp = self.fixed_point
        
        # The Hum amplitude from μ̃* (holographic measure coupling)
        hum_amplitude = fp.mu_star / (16 * PI_SQUARED)
        
        return {
            "amplitude": hum_amplitude,
            "mu_star": fp.mu_star,
            "formula": "H = μ̃*/(16π²)",
            "interpretation": "Vacuum energy density at fixed point"
        }
    
    def compute_running_hum(self, k: float) -> Dict[str, float]:
        """
        Compute the running Hum at RG scale k.
        
        The Hum runs with the RG scale due to quantum corrections.
        
        Args:
            k: RG scale (0 = IR, large = UV)
            
        Returns:
            Dictionary with running Hum
        """
        fp = self.fixed_point
        
        # At the IR fixed point, μ̃(k→0) = μ̃*
        # In the UV, μ̃(k→∞) → 0 (asymptotic freedom)
        mu_running = fp.mu_star * (1 - np.exp(-k))
        
        return {
            "mu_running": mu_running,
            "k": k,
            "mu_star": fp.mu_star,
            "uv_limit": 0.0,
            "ir_limit": fp.mu_star
        }


# =============================================================================
# Dark Energy Equation of State
# =============================================================================

@dataclass
class DarkEnergyEquationOfState:
    """
    Dark energy equation of state from Holographic Hum.
    
    The equation of state parameter w₀ relates the pressure and
    density of dark energy: P = w₀ρ.
    
    In IRH, w₀ is not exactly -1 (cosmological constant) but receives
    a small correction from the universal constant C_H:
    
    w₀ = -1 + C_H/(3×8π²)
    
    This prediction is testable by precision cosmology.
    
    References:
        IRHv18.md §2.3.3: w₀ derivation
        IRHv18.md Eq. 2.23-2.24
    """
    
    fixed_point: CosmicFixedPoint = field(default_factory=CosmicFixedPoint)
    
    def compute_w0(self) -> Dict[str, float]:
        """
        Compute dark energy equation of state w₀.
        
        w₀ = -1 + C_H/(3×8π²) ≈ -0.9998
        
        Returns:
            Dictionary with w₀ prediction
        """
        # Exact formula from manuscript
        w0_correction = C_H / (3 * 8 * PI_SQUARED)
        w0 = -1 + w0_correction
        
        return {
            "w0": w0,
            "w0_correction": w0_correction,
            "C_H": C_H,
            "deviation_from_minus_1": w0_correction,
            "experimental": DARK_ENERGY_OBSERVATIONS["w0_planck"],
            "experimental_uncertainty": DARK_ENERGY_OBSERVATIONS["w0_uncertainty"],
            "formula": "w₀ = -1 + C_H/(3×8π²)"
        }
    
    def compute_w_running(self, z: float) -> Dict[str, float]:
        """
        Compute running equation of state w(z) at redshift z.
        
        w(z) ≈ w₀ + w_a × z/(1+z)
        
        where w_a is the time variation (small in IRH).
        
        Args:
            z: Redshift
            
        Returns:
            Dictionary with w(z)
        """
        w0_result = self.compute_w0()
        w0 = w0_result["w0"]
        
        # In IRH, w_a is suppressed by additional factors of C_H
        w_a = C_H**2 / (16 * PI_SQUARED)  # Very small
        
        w_z = w0 + w_a * z / (1 + z)
        
        return {
            "w_z": w_z,
            "w0": w0,
            "w_a": w_a,
            "z": z,
            "formula": "w(z) = w₀ + w_a × z/(1+z)"
        }
    
    def is_consistent_with_observations(self) -> Dict[str, bool]:
        """
        Check consistency with cosmological observations.
        
        Returns:
            Dictionary with consistency checks
        """
        w0_result = self.compute_w0()
        w0 = w0_result["w0"]
        
        w0_obs = DARK_ENERGY_OBSERVATIONS["w0_planck"]
        w0_err = DARK_ENERGY_OBSERVATIONS["w0_uncertainty"]
        
        # Check within 2σ
        within_1sigma = abs(w0 - w0_obs) < w0_err
        within_2sigma = abs(w0 - w0_obs) < 2 * w0_err
        
        return {
            "w0_predicted": w0,
            "w0_observed": w0_obs,
            "within_1sigma": within_1sigma,
            "within_2sigma": within_2sigma,
            "tension": abs(w0 - w0_obs) / w0_err
        }


# =============================================================================
# Vacuum Energy Density
# =============================================================================

@dataclass
class VacuumEnergyDensity:
    """
    Vacuum energy density from cGFT fixed point.
    
    The vacuum energy is not a free parameter but emerges from
    the fixed-point structure. This resolves the cosmological
    constant problem by deriving Λ from first principles.
    
    References:
        IRHv18.md §2.3.1: Vacuum energy from fixed point
        IRHv18.md Eq. 2.21-2.22
    """
    
    fixed_point: CosmicFixedPoint = field(default_factory=CosmicFixedPoint)
    
    def compute_lambda_star(self) -> Dict[str, float]:
        """
        Compute the cosmological constant Λ_*.
        
        Returns:
            Dictionary with Λ_* prediction
        """
        fp = self.fixed_point
        
        # From emergent_gravity.py - Einstein equations
        # Λ_* ≈ 1.1 × 10^-52 m^-2
        Lambda_star = fp.mu_star / (8 * np.pi * fp.lambda_star)
        
        # In Planck units (l_P = 1)
        Lambda_planck = Lambda_star  # Dimensionless
        
        # Convert to physical units (m^-2)
        l_P = 1.616255e-35  # Planck length in meters
        Lambda_physical = Lambda_star * l_P**(-2)
        
        return {
            "Lambda_star": Lambda_star,
            "Lambda_planck": Lambda_planck,
            "Lambda_m2": Lambda_physical,
            "mu_star": fp.mu_star,
            "lambda_star": fp.lambda_star
        }
    
    def compute_vacuum_energy(self) -> Dict[str, float]:
        """
        Compute vacuum energy density ρ_vac.
        
        ρ_vac = Λ_* / (8πG) in natural units
        
        Returns:
            Dictionary with vacuum energy
        """
        Lambda_result = self.compute_lambda_star()
        Lambda_star = Lambda_result["Lambda_star"]
        
        # ρ_vac = Λ/(8πG) in Planck units where G = 1
        rho_vac = Lambda_star / (8 * np.pi)
        
        return {
            "rho_vacuum": rho_vac,
            "Lambda_star": Lambda_star,
            "units": "Planck units (ρ_P = 1)"
        }


# =============================================================================
# Cosmological Evolution
# =============================================================================

@dataclass
class CosmologicalEvolution:
    """
    Cosmological evolution from IRH predictions.
    
    Computes the expansion history of the universe using
    the IRH dark energy predictions.
    
    References:
        IRHv18.md §2.3: Dark energy dynamics
    """
    
    fixed_point: CosmicFixedPoint = field(default_factory=CosmicFixedPoint)
    
    def compute_hubble_parameter(self, z: float) -> Dict[str, float]:
        """
        Compute Hubble parameter H(z) at redshift z.
        
        H²(z)/H₀² = Ω_m(1+z)³ + Ω_Λ(1+z)^(3(1+w))
        
        Args:
            z: Redshift
            
        Returns:
            Dictionary with H(z)
        """
        Omega_m = DARK_ENERGY_OBSERVATIONS["Omega_m"]
        Omega_Lambda = DARK_ENERGY_OBSERVATIONS["Omega_Lambda"]
        H0 = DARK_ENERGY_OBSERVATIONS["H0_km_s_Mpc"]
        
        # Get w from IRH
        w_eos = DarkEnergyEquationOfState(self.fixed_point)
        w0 = w_eos.compute_w0()["w0"]
        
        # Friedmann equation
        H2_over_H02 = (
            Omega_m * (1 + z)**3 
            + Omega_Lambda * (1 + z)**(3 * (1 + w0))
        )
        
        H_z = H0 * np.sqrt(H2_over_H02)
        
        return {
            "H_z": H_z,
            "H0": H0,
            "z": z,
            "w0": w0,
            "units": "km/s/Mpc"
        }
    
    def compute_deceleration_parameter(self, z: float = 0.0) -> Dict[str, float]:
        """
        Compute deceleration parameter q(z).
        
        q = -1 - Ḣ/H²
        
        Args:
            z: Redshift (default: today)
            
        Returns:
            Dictionary with q(z)
        """
        Omega_m = DARK_ENERGY_OBSERVATIONS["Omega_m"]
        Omega_Lambda = DARK_ENERGY_OBSERVATIONS["Omega_Lambda"]
        
        w_eos = DarkEnergyEquationOfState(self.fixed_point)
        w0 = w_eos.compute_w0()["w0"]
        
        # q = (Ω_m/2) + Ω_Λ(1 + 3w)/2 at z=0
        q = 0.5 * Omega_m + 0.5 * Omega_Lambda * (1 + 3 * w0)
        
        return {
            "q": q,
            "z": z,
            "accelerating": q < 0,
            "w0": w0
        }


# =============================================================================
# Complete Dark Energy Module
# =============================================================================

@dataclass
class DarkEnergyModule:
    """
    Complete dark energy predictions from IRH v18.0.
    
    Combines all dark energy calculations:
    - Holographic Hum
    - Equation of state w₀
    - Vacuum energy density
    - Cosmological evolution
    
    References:
        IRHv18.md §2.3: Complete dark energy derivation
    """
    
    fixed_point: CosmicFixedPoint = field(default_factory=CosmicFixedPoint)
    
    def compute_full_analysis(self) -> Dict[str, any]:
        """
        Compute complete dark energy analysis.
        
        Returns:
            Dictionary with all predictions
        """
        hum = HolographicHum(self.fixed_point)
        w_eos = DarkEnergyEquationOfState(self.fixed_point)
        vacuum = VacuumEnergyDensity(self.fixed_point)
        cosmo = CosmologicalEvolution(self.fixed_point)
        
        return {
            "holographic_hum": hum.compute_hum_amplitude(),
            "equation_of_state": w_eos.compute_w0(),
            "vacuum_energy": vacuum.compute_vacuum_energy(),
            "cosmological_constant": vacuum.compute_lambda_star(),
            "observational_consistency": w_eos.is_consistent_with_observations(),
            "hubble_today": cosmo.compute_hubble_parameter(0),
            "deceleration": cosmo.compute_deceleration_parameter(0),
            "status": "Dark energy fully derived from Cosmic Fixed Point"
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_dark_energy_summary() -> Dict[str, any]:
    """
    Compute summary of all dark energy predictions.
    
    Returns:
        Dictionary with complete dark energy summary
    """
    module = DarkEnergyModule()
    return module.compute_full_analysis()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HolographicHum',
    'DarkEnergyEquationOfState',
    'VacuumEnergyDensity',
    'CosmologicalEvolution',
    'DarkEnergyModule',
    'compute_dark_energy_summary',
    'DARK_ENERGY_OBSERVATIONS',
    'C_H',
]
