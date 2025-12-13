# IRH 20.3 Theory-to-Code Traceability

This document maps equations and results from IRH20.3.md (root) to their implementations in the v18 codebase.

## Governing Theory

**Source**: `IRH20.3.md` (repository root) - Intrinsic Resonance Holography v20.3: The Unified Theory of Emergent Reality

**Prior Baseline**: `docs/manuscripts/IRH18.md` - For historical reference

---

## 1. RG Flow and Cosmic Fixed Point

### 1.1 β-Functions (Eq. 1.13)

| IRH20.3 Equation | Code Location | Test |
|------------------|---------------|------|
| β_λ = -2λ̃ + (9/8π²)λ̃² | `rg_flow.py:BetaFunctions.beta_lambda()` | `test_cgft_core.py::TestBetaFunctions` |
| β_γ = 0γ̃ + (3/4π²)λ̃γ̃ | `rg_flow.py:BetaFunctions.beta_gamma()` | `test_cgft_core.py::TestBetaFunctions` |
| β_μ = 2μ̃ + (1/2π²)λ̃μ̃ | `rg_flow.py:BetaFunctions.beta_mu()` | `test_cgft_core.py::TestBetaFunctions` |

### 1.2 Fixed Point Values (Eq. 1.14)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| λ̃* | 48π²/9 ≈ 52.64 | `rg_flow.py:CosmicFixedPoint.lambda_star` | `test_cgft_core.py::TestCosmicFixedPoint` |
| γ̃* | 32π²/3 ≈ 105.28 | `rg_flow.py:CosmicFixedPoint.gamma_star` | `test_cgft_core.py::TestCosmicFixedPoint` |
| μ̃* | 16π² ≈ 157.91 | `rg_flow.py:CosmicFixedPoint.mu_star` | `test_cgft_core.py::TestCosmicFixedPoint` |

### 1.3 Universal Exponent C_H (Eq. 1.15-1.16)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| C_H = 3λ̃*/2γ̃* | 0.045935703598 | `rg_flow.py:CosmicFixedPoint.C_H` | `test_cgft_core.py::TestCosmicFixedPoint` |

### 1.4 Stability Matrix (Sec. 1.3.1)

| Matrix Element | IRH20.3 Value | Code Location | Test |
|----------------|---------------|---------------|------|
| M[0,0] | 10 | `rg_flow.py:StabilityAnalysis.compute_stability_matrix()` | `test_cgft_core.py::TestStabilityAnalysis` |
| M[1,0] | 8 | `rg_flow.py:StabilityAnalysis.compute_stability_matrix()` | `test_cgft_core.py::TestStabilityAnalysis` |
| M[1,1] | 4 | `rg_flow.py:StabilityAnalysis.compute_stability_matrix()` | `test_cgft_core.py::TestStabilityAnalysis` |
| M[2,0] | 8 | `rg_flow.py:StabilityAnalysis.compute_stability_matrix()` | `test_cgft_core.py::TestStabilityAnalysis` |
| M[2,2] | 14/3 ≈ 4.67 | `rg_flow.py:StabilityAnalysis.compute_stability_matrix()` | `test_cgft_core.py::TestStabilityAnalysis` |

**Eigenvalues (Sec. 1.3.2)**:
- λ₁ = 10 (relevant, positive)
- λ₂ = 4 (relevant, positive)
- λ₃ = 14/3 ≈ 4.67 (relevant, positive)

**Note**: All three eigenvalues are positive, confirming IR-attractiveness for all couplings.

---

## 2. Spacetime and Gravity

### 2.1 Spectral Dimension Flow (Eq. 2.8-2.9)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| d_spec (one-loop) | 42/11 ≈ 3.818 | `spectral_dimension.py:SpectralDimensionFlow` | `test_cgft_core.py::TestSpectralDimension` |
| d_spec (IR, exact) | 4.0000000000(1) | `spectral_dimension.py:SpectralDimensionFlow.compute_ir_value()` | `test_cgft_core.py::TestSpectralDimension` |
| Δ_grav correction | -2/11 → 0 in IR | `spectral_dimension.py` | `test_cgft_core.py` |

### 2.2 Dark Energy and w₀ (Sec. 2.3, Eq. 2.21-2.23)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| w₀ (one-loop, Eq. 2.22) | -5/6 ≈ -0.833 | `dark_energy.py:DarkEnergyEquationOfState.compute_w0()` | `test_v18_extended.py::TestDarkEnergyEquationOfState` |
| w₀ (final, Eq. 2.23) | **-0.91234567(8)** | `dark_energy.py:DarkEnergyEquationOfState.compute_w0()` | `test_v18_extended.py::TestDarkEnergyEquationOfState` |
| Λ* (Eq. 2.19) | 1.1056 × 10⁻⁵² m⁻² | `dark_energy.py:VacuumEnergyDensity.compute_lambda_star()` | `test_v18_extended.py::TestVacuumEnergyDensity` |

### 2.3 Lorentz Invariance Violation (Eq. 2.24-2.26)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| ξ = C_H/(24π²) | 1.933355051 × 10⁻⁴ | `emergent_gravity.py:LorentzInvarianceViolation.compute_xi()` | `test_v18_new_modules.py::TestLorentzInvarianceViolation` |

---

## 3. Standard Model Topology

### 3.1 Gauge Symmetry (Eq. 3.1, Appendix D)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| β₁ (First Betti number) | 12 | `topology.py:StandardModelTopology.compute_beta_1()` | `test_v18_new_modules.py::TestBettiNumber` |
| Gauge decomposition | SU(3)×SU(2)×U(1) → 8+3+1 | `topology.py:StandardModelTopology` | `test_v18_new_modules.py::TestBettiNumber` |

### 3.2 Fermion Generations (Eq. 3.2, Appendix D)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| n_inst (instanton number) | 3 | `topology.py:StandardModelTopology.compute_n_inst()` | `test_v18_new_modules.py::TestInstantonNumber` |
| N_gen (generations) | 3 | `topology.py:StandardModelTopology` | `test_v18_new_modules.py::TestInstantonNumber` |

### 3.3 Topological Complexity K_f (Eq. 3.3, Appendix E)

| Generation | IRH20.3 Value | Code Location | Test |
|------------|---------------|---------------|------|
| K₁ (electron) | 1.000 ± 0.001 | `topology.py:VortexWavePattern` | `test_v18_new_modules.py::TestVortexWavePattern` |
| K₂ (muon) | 206.77 ± 0.02 | `topology.py:VortexWavePattern` | `test_v18_new_modules.py::TestVortexWavePattern` |
| K₃ (tau) | 3477.15 ± 0.35 | `topology.py:VortexWavePattern` | `test_v18_new_modules.py::TestVortexWavePattern` |

### 3.4 Fine Structure Constant (Eq. 3.4-3.5)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| α⁻¹ | 137.035999084(1) | `physical_constants.py:FineStructureConstant` | `test_cgft_core.py::TestPhysicalConstants` |

---

## 4. Harmony Functional (Sec. 1.4, Eq. 1.5)

| Quantity | IRH20.3 Equation | Code Location | Test |
|----------|------------------|---------------|------|
| Γ[Σ] | Tr(L²) - C_H log det' L | `cgft_action.py:compute_harmony_functional()` | `test_cgft_core.py::TestHarmonyFunctional` |
| Bilocal field Σ(g,g') | Definition 1.4.1 | `cgft_field.py:BiLocalField` | `test_cgft_core.py::TestBiLocalField` |

---

## 5. Electroweak and Strong Sector

### 5.1 Electroweak (Sec. 3.3)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| v (Higgs VEV) | 246.22 GeV | `electroweak.py:HiggsBoson` | `test_v18_physics.py::TestHiggsBoson` |
| m_H (Higgs mass) | 125.25(10) GeV | `electroweak.py:HiggsBoson` | `test_v18_physics.py::TestHiggsBoson` |
| sin²θ_W | 0.231 | `electroweak.py:WeinbergAngle` | `test_v18_physics.py::TestWeinbergAngle` |

### 5.2 Strong CP (Sec. 3.4)

| Quantity | IRH20.3 Value | Code Location | Test |
|----------|---------------|---------------|------|
| θ_eff | 0 | `strong_cp.py:StrongCPResolution` | `test_v18_physics.py::TestStrongCPResolution` |
| m_a (axion mass) | ~5.7 μeV | `strong_cp.py:AlgorithmicAxion` | `test_v18_physics.py::TestAlgorithmicAxion` |

---

## 6. Emergent Quantum Mechanics (Sec. 5)

| Feature | IRH20.3 Section | Code Location | Test |
|---------|-----------------|---------------|------|
| Born rule | §5.1 | `quantum_mechanics.py:BornRule` | `test_v18_physics.py::TestBornRule` |
| Decoherence | §5.2 | `quantum_mechanics.py:Decoherence` | `test_v18_physics.py::TestDecoherence` |
| Lindblad equation | §5.3 | `quantum_mechanics.py:LindbladEquation` | `test_v18_physics.py::TestLindbladEquation` |

---

## Version History

| Date | Change | Author |
|------|--------|--------|
| 2025-12-13 | Initial traceability document for IRH20.3 alignment | Copilot Agent |

---

## References

1. **IRH20.3.md** (root): Primary theory document
2. **docs/manuscripts/IRH18.md**: Prior v18 baseline
3. **python/src/irh/core/v18/**: Implementation modules
4. **python/tests/v18/**: Test suite
