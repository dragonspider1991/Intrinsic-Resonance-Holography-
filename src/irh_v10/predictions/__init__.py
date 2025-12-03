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

__all__ = [
    "derive_alpha",
    "quick_alpha_demo",
]
