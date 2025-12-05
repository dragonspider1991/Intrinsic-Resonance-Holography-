"""
constants.py - Physical Constants for IRH Suite

This module provides consistent physical constants used throughout
the IRH Suite, avoiding duplication across modules.

Values are from CODATA 2022 where applicable.
"""

# Planck units
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
HBAR_G = HBAR  # Alias for geometric units
C = 299792458  # Speed of light (m/s)
G_N = 6.67430e-11  # Newton's gravitational constant (m³/(kg·s²))

# Planck scale
L_P = 1.616255e-35  # Planck length (m)
L_G = L_P  # Alias for graph length scale
T_P = 5.391247e-44  # Planck time (s)
M_P = 2.176434e-8  # Planck mass (kg)
E_P = 1.9561e9  # Planck energy (J)

# Standard Model parameters (PDG 2024)
ALPHA_EM = 7.2973525693e-3  # Fine structure constant α
ALPHA_EM_INVERSE = 137.035999084  # α⁻¹
ALPHA_S_MZ = 0.1180  # Strong coupling at M_Z

# Gauge coupling constants at M_Z
ALPHA_1_MZ = 0.0169  # U(1) hypercharge
ALPHA_2_MZ = 0.0337  # SU(2) weak
ALPHA_3_MZ = 0.118  # SU(3) strong (QCD)

# Particle masses (PDG 2024, in GeV)
M_ELECTRON = 0.000511  # Electron mass
M_MUON = 0.1057  # Muon mass
M_TAU = 1.777  # Tau mass
M_Z = 91.1876  # Z boson mass
M_W = 80.377  # W boson mass
M_HIGGS = 125.25  # Higgs boson mass

# Neutrino parameters
NEUTRINO_MASS_SUM = 0.0583  # eV (cosmological bound / prediction)
DELTA_M21_SQ = 7.53e-5  # eV² (solar)
DELTA_M31_SQ = 2.453e-3  # eV² (atmospheric)

# Cosmological parameters
HUBBLE_CONSTANT = 67.4  # km/s/Mpc
OMEGA_LAMBDA = 0.685  # Dark energy density parameter
W_LAMBDA_LCDM = -1.0  # Λ-CDM dark energy EoS
W_LAMBDA_IRH = -0.75  # IRH prediction for dark energy EoS

# CKM matrix elements (PDG 2024)
CKM_MATRIX = {
    "V_ud": 0.97373,
    "V_us": 0.2243,
    "V_ub": 0.00382,
    "V_cd": 0.221,
    "V_cs": 0.975,
    "V_cb": 0.0408,
    "V_td": 0.0086,
    "V_ts": 0.0415,
    "V_tb": 1.014,
}

# Beta function coefficients (1-loop)
# For SU(N) with N_f fermions: b0 = (11/3)N - (2/3)N_f
# With sign convention β = -b0 * g³/(16π²)
BETA_COEFFICIENTS = {
    "U(1)": 41 / 10,  # Hypercharge
    "SU(2)": -19 / 6,  # Weak isospin
    "SU(3)": -7,  # QCD (negative for asymptotic freedom)
}

# IRH-specific parameters
TARGET_SPECTRAL_DIMENSION = 4.0  # d_s ≈ 4 for physical spacetime
TARGET_BETTI_1 = 12  # β₁ = 12 for Standard Model fermion generations
TARGET_NEGATIVE_EIGENVALUES = 1  # Lorentzian signature (-,+,+,+)
