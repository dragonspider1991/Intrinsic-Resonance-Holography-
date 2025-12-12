# HarmonyOptimizer Numerical Methods & Fixed-Point Transparency

This note makes the numerical workflow for locating and characterizing the Cosmic Fixed Point fully explicit and reproducible. It documents the algorithms, truncation scheme, regulator assumptions, and projection operators used in the publicly available codebase, with runnable examples.

## Code Locations

- **RG flow, fixed point, stability:** [`python/src/irh/core/v18/rg_flow.py`](../python/src/irh/core/v18/rg_flow.py)
- **Action and coupling container:** [`python/src/irh/core/v18/cgft_action.py`](../python/src/irh/core/v18/cgft_action.py)
- **Spectral dimension flow:** [`python/src/irh/core/v18/spectral_dimension.py`](../python/src/irh/core/v18/spectral_dimension.py)
- **Physical constants (α, masses, w₀):** [`python/src/irh/core/v18/physical_constants.py`](../python/src/irh/core/v18/physical_constants.py)
- **Topology and SM derivation:** [`python/src/irh/core/v18/topology.py`](../python/src/irh/core/v18/topology.py)
- **Validation tests:** [`python/tests/v18/`](../python/tests/v18/)

No external datasets are required; all calculations are derived from the analytical beta functions and solved with open-source SciPy/Numpy routines.

## Numerical Algorithms (public API)

| Task | API | Method |
| --- | --- | --- |
| Evaluate β-functions | `BetaFunctions.evaluate(...)` | One-loop Wetterich truncation (IRHv18 Eq. 1.13) |
| Solve for fixed point | `find_fixed_point()` | `scipy.optimize.fsolve` on β=0 |
| Integrate RG flow | `integrate_rg_flow(...)` | `scipy.integrate.solve_ivp` (RK45) with user-supplied span |
| Stability analysis | `StabilityAnalysis.compute_stability_matrix()` / `.compute_eigenvalues()` | Analytic Jacobian projection onto {λ̃, γ̃, μ̃} |
| Certified C_H | `compute_C_H_certified()` | Closed-form ratio 3λ̃*/2γ̃* with stored 12+ decimal value |

### Truncation and Projection
- **Truncation:** One-loop β-functions for the three couplings (λ̃, γ̃, μ̃) matching IRHv18 Eq. 1.13. Higher operators are excluded by construction; the lower-triangular Jacobian captures their projected influence.
- **Regulator choice:** Fixed canonical scaling (no scheme parameter in code). The coefficients COEFF_* encode the regulator-dependent prefactors used in the manuscript; changing schemes requires only updating these constants.
- **Projection operators:** Stability matrix is explicitly projected onto the coupling subspace with analytically derived partial derivatives (`compute_stability_matrix`). This yields eigenvalues/eigendirections used for relevance classification.

## Reproduction Examples

Run from the repo root:

```bash
cd python
export PYTHONPATH=$(pwd)/src
python - <<'PY'
from irh.core.v18.rg_flow import (
    find_fixed_point,
    integrate_rg_flow,
    StabilityAnalysis,
    compute_C_H_certified,
)

# Fixed point solve (β=0)
fp = find_fixed_point()
print("Fixed point couplings:", fp.lambda_star, fp.gamma_star, fp.mu_star)
print("C_H:", fp.C_H)

# RG trajectory toward IR
traj = integrate_rg_flow((60.0, 90.0, 120.0), t_span=(0, -6), num_points=25)
print("Final couplings (IR):", traj.couplings_final)

# Stability and projection
analysis = StabilityAnalysis(fp).full_analysis()
print("Eigenvalues:", analysis["eigenvalues"])
print("Operator classes:", analysis["operator_classifications"])

# Certified universal exponent
print("Certified C_H payload:", compute_C_H_certified())
PY
```

## Other Key Computational Modules

- **Spectral dimension flow:** `SpectralDimensionFlow` (`spectral_dimension.py`) integrates the return probability and enforces d_spec → 4.
- **Physical constants:** `FineStructureConstant`, `FermionMassCalculator` (`physical_constants.py`) provide closed-form evaluations linked to the fixed point.
- **Topology & Standard Model:** `StandardModelTopology` (`topology.py`) computes β₁ = 12 and n_inst = 3 with deterministic combinatorics.
- **Gravity & LIV:** `EinsteinEquations` and `LorentzInvarianceViolation` (`emergent_gravity.py`) derive Λ_* and ξ with no external data.
- **Dark energy & spacetime:** `DarkEnergyModule` (`dark_energy.py`) and `EmergentSpacetime` (`emergent_spacetime.py`) carry the holographic and geometric calculations.

Each module is fully contained in `python/src/irh/core/v18/` and exercised by the public tests in `python/tests/v18/`. No hidden datasets or proprietary binaries are used.
