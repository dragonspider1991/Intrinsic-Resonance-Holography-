"""
Physical constant predictions from IRH v10.0

All constants derived from optimized Cymatic Resonance Network:
    - Fine structure constant: α⁻¹ = 137.035999084
    - Planck constant: ℏ (derived from phase space quantization)
    - Newton's constant: G (from emergent gravity)
    - Proton-electron mass ratio: m_p/m_e = 1836.15...

Zero free parameters - all values emergent from network geometry.
"""

from .fine_structure_alpha import derive_alpha, quick_alpha_demo
from .planck_constant import derive_planck_constant, planck_length, planck_mass
from .newton_G import derive_newton_G, gravitational_coupling_strength
from .proton_electron_mass_ratio import derive_mass_ratio, mass_from_winding

__all__ = [
    "derive_alpha",
    "quick_alpha_demo",
    "derive_planck_constant",
    "planck_length",
    "planck_mass",
    "derive_newton_G",
    "gravitational_coupling_strength",
    "derive_mass_ratio",
    "mass_from_winding",
]
