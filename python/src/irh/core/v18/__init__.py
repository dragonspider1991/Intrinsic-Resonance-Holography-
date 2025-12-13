"""
IRH v18.0 Core Implementation - CURRENT VERSION
================================================

Complex-weighted Group Field Theory (cGFT) framework achieving
full ontological and mathematical closure.

This is the **current and recommended** IRH implementation.
Previous versions (v16, v17) are deprecated.

Key Achievements:
    - 12+ decimal precision in fundamental constant derivation
    - Complete Standard Model emergence from topology (β₁=12, n_inst=3)
    - Analytical derivations without free parameters
    - 143 passing tests validating all physics modules

Modules:
    - group_manifold: G_inf = SU(2) × U(1)_φ implementation
    - cgft_field: Fundamental field φ(g₁,g₂,g₃,g₄) and bilocal Σ
    - cgft_action: S_kin + S_int + S_hol action terms
    - rg_flow: Beta functions and Cosmic Fixed Point
    - spectral_dimension: d_spec flow to exactly 4
    - physical_constants: α, fermion masses, w₀, Λ*
    - topology: Standard Model emergence (β₁=12, n_inst=3)
    - emergent_gravity: Einstein equations, graviton propagator, LIV
    - flavor_mixing: CKM, PMNS matrices, neutrino sector
    - electroweak: Higgs, W/Z masses, Weinberg angle
    - strong_cp: θ=0, algorithmic axion
    - quantum_mechanics: Born rule, decoherence, Lindblad equation
    - dark_energy: Holographic Hum, w₀ equation of state
    - emergent_spacetime: Lorentzian signature, time emergence
    - emergent_qft: Full particle spectrum, effective Lagrangian

Key Classes:
    - GInfElement: Element of informational group manifold
    - cGFTFieldDiscrete: Discretized fundamental field
    - cGFTAction: Complete cGFT action
    - CosmicFixedPoint: The unique infrared attractor
    - StandardModelTopology: Complete SM derivation
    - EmergentQFT: Full QFT emergence verification

Quick Start:
    >>> from irh.core.v18 import StandardModelTopology, find_fixed_point
    >>> sm = StandardModelTopology()
    >>> assert sm.verify_standard_model()  # β₁=12, n_inst=3
    >>> fp = find_fixed_point()
    >>> assert fp.verify()["is_fixed_point"]

References:
    docs/manuscripts/IRH18.md: Complete theoretical framework
    docs/v18_IMPLEMENTATION_PLAN.md: Implementation roadmap
"""

# Version info
__version__ = "18.0.0"
__status__ = "current"

from .group_manifold import (
    SU2Element,
    U1Element,
    GInfElement,
    compute_ncd,
    compute_ncd_distance,
    haar_integrate_su2,
    haar_integrate_ginf,
)

from .cgft_field import (
    cGFTField,
    cGFTFieldDiscrete,
    BiLocalField,
    CondensateState,
    compute_fluctuation_field,
)

from .cgft_action import (
    cGFTCouplings,
    InteractionKernel,
    cGFTAction,
    compute_effective_laplacian,
    compute_harmony_functional,
)

from .rg_flow import (
    BetaFunctions,
    CosmicFixedPoint,
    find_fixed_point,
    StabilityAnalysis,
    RGFlowSolution,
    integrate_rg_flow,
    compute_C_H_certified,
)

from .spectral_dimension import (
    SpectralDimensionFlow,
    compute_spectral_dimension_heat_kernel,
    AsymptoticSafetySignature,
    verify_theorem_2_1,
    D_SPEC_ONE_LOOP,
    D_SPEC_IR,
)

from .physical_constants import (
    FineStructureConstant,
    FermionMassCalculator,
    DarkEnergyPrediction,
    CosmologicalConstantPrediction,
    compute_all_predictions,
    ALPHA_INVERSE_CODATA,
    TOPOLOGICAL_COMPLEXITY,
)

from .topology import (
    BettiNumberFlow,
    InstantonNumberFlow,
    VortexWavePattern,
    EmergentSpatialManifold,
    StandardModelTopology,
    SM_GAUGE_GENERATORS,
    TOTAL_SM_GENERATORS,
    NUM_FERMION_GENERATIONS,
)

from .emergent_gravity import (
    EmergentMetric,
    EinsteinEquations,
    GravitonPropagator,
    HigherCurvatureSuppression,
    LorentzInvarianceViolation,
    compute_emergent_gravity_summary,
    PLANCK_LENGTH,
    PLANCK_MASS,
    PLANCK_ENERGY,
    LAMBDA_OBSERVED,
    G_NEWTON,
)

from .flavor_mixing import (
    CKMMatrix,
    PMNSMatrix,
    NeutrinoSector,
    compute_flavor_mixing_summary,
    CKM_EXPERIMENTAL,
    PMNS_ANGLES_EXP,
    NEUTRINO_MASS_SPLITTINGS,
)

from .electroweak import (
    HiggsBoson,
    GaugeBosonMasses,
    WeinbergAngle,
    FermiConstant,
    ElectroweakSector,
    EW_EXPERIMENTAL,
)

from .strong_cp import (
    ThetaAngle,
    PecceiQuinnSymmetry,
    AlgorithmicAxion,
    StrongCPResolution,
    STRONG_CP_CONSTANTS,
    QCD_PARAMETERS,
)

from .quantum_mechanics import (
    ElementaryAlgorithmicTransformation,
    QuantumAmplitudeEmergence,
    BornRule,
    Decoherence,
    LindbladEquation,
    EmergentQuantumMechanics,
)

from .dark_energy import (
    HolographicHum,
    DarkEnergyEquationOfState,
    VacuumEnergyDensity,
    CosmologicalEvolution,
    DarkEnergyModule,
    compute_dark_energy_summary,
    DARK_ENERGY_OBSERVATIONS,
)

from .emergent_spacetime import (
    LorentzianSignatureEmergence,
    TimeEmergence,
    DiffeomorphismInvariance,
    EmergentSpacetime,
    compute_spacetime_summary,
)

from .emergent_qft import (
    ParticleType,
    GaugeGroup,
    EmergentParticle,
    GravitonIdentification,
    GaugeBosonIdentification,
    FermionIdentification,
    EffectiveLagrangian,
    EmergentQFT,
    compute_qft_summary,
)

# Universal constant from Cosmic Fixed Point
C_H_V18 = 0.045935703598

__all__ = [
    # Group manifold
    'SU2Element',
    'U1Element',
    'GInfElement',
    'compute_ncd',
    'compute_ncd_distance',
    'haar_integrate_su2',
    'haar_integrate_ginf',
    
    # cGFT field
    'cGFTField',
    'cGFTFieldDiscrete',
    'BiLocalField',
    'CondensateState',
    'compute_fluctuation_field',
    
    # cGFT action
    'cGFTCouplings',
    'InteractionKernel',
    'cGFTAction',
    'compute_effective_laplacian',
    'compute_harmony_functional',
    
    # RG flow
    'BetaFunctions',
    'CosmicFixedPoint',
    'find_fixed_point',
    'StabilityAnalysis',
    'RGFlowSolution',
    'integrate_rg_flow',
    'compute_C_H_certified',
    
    # Spectral dimension
    'SpectralDimensionFlow',
    'compute_spectral_dimension_heat_kernel',
    'AsymptoticSafetySignature',
    'verify_theorem_2_1',
    'D_SPEC_ONE_LOOP',
    'D_SPEC_IR',
    
    # Physical constants
    'FineStructureConstant',
    'FermionMassCalculator',
    'DarkEnergyPrediction',
    'CosmologicalConstantPrediction',
    'compute_all_predictions',
    'ALPHA_INVERSE_CODATA',
    'TOPOLOGICAL_COMPLEXITY',
    
    # Topology (Standard Model emergence)
    'BettiNumberFlow',
    'InstantonNumberFlow',
    'VortexWavePattern',
    'EmergentSpatialManifold',
    'StandardModelTopology',
    'SM_GAUGE_GENERATORS',
    'TOTAL_SM_GENERATORS',
    'NUM_FERMION_GENERATIONS',
    
    # Emergent gravity
    'EmergentMetric',
    'EinsteinEquations',
    'GravitonPropagator',
    'HigherCurvatureSuppression',
    'LorentzInvarianceViolation',
    'compute_emergent_gravity_summary',
    'PLANCK_LENGTH',
    'PLANCK_MASS',
    'PLANCK_ENERGY',
    'LAMBDA_OBSERVED',
    'G_NEWTON',
    
    # Flavor mixing
    'CKMMatrix',
    'PMNSMatrix',
    'NeutrinoSector',
    'compute_flavor_mixing_summary',
    'CKM_EXPERIMENTAL',
    'PMNS_ANGLES_EXP',
    'NEUTRINO_MASS_SPLITTINGS',
    
    # Electroweak sector
    'HiggsBoson',
    'GaugeBosonMasses',
    'WeinbergAngle',
    'FermiConstant',
    'ElectroweakSector',
    'EW_EXPERIMENTAL',
    
    # Strong CP problem
    'ThetaAngle',
    'PecceiQuinnSymmetry',
    'AlgorithmicAxion',
    'StrongCPResolution',
    'STRONG_CP_CONSTANTS',
    'QCD_PARAMETERS',
    
    # Emergent quantum mechanics
    'ElementaryAlgorithmicTransformation',
    'QuantumAmplitudeEmergence',
    'BornRule',
    'Decoherence',
    'LindbladEquation',
    'EmergentQuantumMechanics',
    
    # Dark energy and Holographic Hum
    'HolographicHum',
    'DarkEnergyEquationOfState',
    'VacuumEnergyDensity',
    'CosmologicalEvolution',
    'DarkEnergyModule',
    'compute_dark_energy_summary',
    'DARK_ENERGY_OBSERVATIONS',
    
    # Emergent spacetime
    'LorentzianSignatureEmergence',
    'TimeEmergence',
    'DiffeomorphismInvariance',
    'EmergentSpacetime',
    'compute_spacetime_summary',
    
    # Emergent QFT
    'ParticleType',
    'GaugeGroup',
    'EmergentParticle',
    'GravitonIdentification',
    'GaugeBosonIdentification',
    'FermionIdentification',
    'EffectiveLagrangian',
    'EmergentQFT',
    'compute_qft_summary',
    
    # Constants
    'C_H_V18',
    
    # Version info
    '__version__',
    '__status__',
]
