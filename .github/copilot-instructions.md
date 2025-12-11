# Intrinsic Resonance Holography (IRH) - Copilot Instructions

## Project Overview

IRH is a theoretical physics framework that derives fundamental constants and physical laws from first principles using Algorithmic Holonomic States (AHS). This is a research codebase implementing computational validation of the IRH v16.0 and v18.0 theories.

**Key Features:**
- Derives fundamental constants (fine structure constant α, dark energy w₀, etc.) from algorithmic information theory
- Exascale-ready implementation with certified numerical precision (12+ decimal places)
- Multi-version architecture (v15, v16, v18) with incremental enhancements
- Complete physics derivations: Quantum Mechanics, General Relativity, Standard Model

**Target Audience:** Theoretical physicists, computational scientists, researchers in quantum gravity and emergent spacetime

## Technology Stack

### Core Technologies
- **Python**: 3.11, 3.12 (primary implementation language)
- **NumPy**: >=1.24.0 (numerical computations)
- **SciPy**: >=1.11.0 (scientific computing, optimization)
- **NetworkX**: >=3.1 (graph-based algorithms for Cymatic Resonance Networks)
- **QuTip**: >=5.0.0 (quantum mechanics simulations)
- **SymPy**: >=1.12 (symbolic mathematics)
- **mpmath**: >=1.3.0 (arbitrary precision arithmetic)

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting (line length: 100)
- **mypy**: Type checking
- **ruff**: Fast linting

### Web Interface (Optional)
- **FastAPI**: Backend REST API
- **Streamlit**: Interactive frontend
- **Uvicorn**: ASGI server

### Build System
- **setuptools**: Package management
- **pyproject.toml**: Modern Python packaging

## Repository Structure

```
Intrinsic-Resonance-Holography-/
├── python/
│   ├── src/irh/
│   │   ├── core/
│   │   │   ├── v16/          # v16 implementation (ACW, AHS, ARO, NCD, Harmony)
│   │   │   └── v18/          # v18 cGFT implementation (analytical derivations)
│   │   ├── predictions/      # Physical constant predictions
│   │   └── ...
│   └── tests/
│       ├── v16/              # v16 unit tests
│       └── v18/              # v18 unit tests
├── docs/
│   ├── manuscripts/          # Theory manuscripts (IRHv16.md, IRHv18.md)
│   └── ...
├── webapp/                   # Web interface (FastAPI + Streamlit)
├── notebooks/                # Jupyter notebooks for demonstrations
└── pyproject.toml
```

## Coding Standards

### Python Style
- Follow **PEP 8** guidelines strictly
- Use **type hints** for all function parameters and return values
- Maximum line length: **100 characters**
- Format code with `black --line-length 100`
- Use `ruff` for linting before committing

### Naming Conventions
- **Variables**: `snake_case` (e.g., `holonomic_phase`, `binary_string`)
- **Functions**: `snake_case` (e.g., `compute_ncd`, `create_ahs_network`)
- **Classes**: `PascalCase` (e.g., `AlgorithmicHolonomicState`, `CymaticResonanceNetwork`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `PHASE_TOLERANCE`, `FIXED_POINT_LAMBDA`)
- **Private methods/attributes**: prefix with `_` (e.g., `_to_bytes`, `_wrapped_phase_difference`)

### Docstring Style
Use **NumPy-style docstrings** for all public functions and classes:

```python
def compute_harmony(
    network: CymaticResonanceNetwork,
    epsilon: float = 0.730129,
) -> float:
    """
    Compute the Harmony Functional H(ε) for a given network (Eq.4.1).
    
    Parameters
    ----------
    network : CymaticResonanceNetwork
        The network of AHS nodes to evaluate.
    epsilon : float, optional
        Edge density threshold parameter (default: 0.730129).
    
    Returns
    -------
    float
        The harmony value H(ε) with certified precision.
    
    Notes
    -----
    Implements the spectral zeta-regularized harmony functional with
    certified numerical precision (error < 10^-12).
    
    References
    ----------
    IRH v16.0 Manuscript, Eq.4.1
    """
```

### Type Hints
Always use type hints from `typing` and `numpy.typing`:

```python
from typing import Optional, Union, List, Tuple
from numpy.typing import NDArray
import numpy as np

def create_ahs_network(
    binary_strings: List[Union[str, bytes]],
    phases: Optional[NDArray[np.float64]] = None,
) -> List[AlgorithmicHolonomicState]:
    """Create a network of AHS nodes."""
    ...
```

### Mathematical Constants and Precision
- Use certified precision constants from theory (12+ decimal places)
- Reference equation numbers from manuscripts in comments
- Use `np.float64` for standard precision, `mpmath` for arbitrary precision when needed
- Always track numerical error bounds for critical calculations

### Data Validation Patterns
- **Input normalization**: Use helper functions like `_to_bytes()` to normalize str/bytes/bytearray inputs
- **Phase wrapping**: Use `np.mod(angle, 2*np.pi)` to wrap phases to [0, 2π)
- **Phase differences**: Use `_wrapped_phase_difference()` returning values in [-π, π]
- **Tolerance comparisons**: Use `PHASE_TOLERANCE = 1e-10` for phase equality, `np.isclose()` for floating-point comparisons

### Error Handling
- Raise appropriate exceptions (`TypeError`, `ValueError`) with descriptive messages
- Detect degenerate cases early (e.g., trace < 1e-12 in harmony functional)
- Validate inputs in `__post_init__` for dataclasses

## Version-Specific Guidelines

### v16.0 (Current Implementation)
- **Focus**: Exascale readiness, certified precision, non-circular derivations
- **Key modules**: `ahs.py`, `acw.py`, `aro.py`, `harmony.py`, `ncd_multifidelity.py`, `distributed_ahs.py`
- **Constants**: Universal constant C_H = 0.045935703598 (12+ decimals)
- **Data types**: AlgorithmicHolonomicState accepts str/bytes/bytearray, preserves original type
- **NCD**: Multi-fidelity adaptive selection (HIGH <10^4, MEDIUM <10^6, LOW >=10^6 bytes)

### v18.0 (Analytical Derivations from cGFT)
- **Focus**: Analytical derivations from complex-weighted Group Field Theory (cGFT)
- **Core modules**: 
  - `group_manifold.py`: G_inf = SU(2) × U(1)_φ implementation
  - `cgft_field.py`: Fundamental field φ(g₁,g₂,g₃,g₄) and bilocal Σ
  - `cgft_action.py`: Complete action S_kin + S_int + S_hol
  - `rg_flow.py`: Beta functions and Cosmic Fixed Point
  - `spectral_dimension.py`: d_spec flow to exactly 4
  - `physical_constants.py`: α, fermion masses, w₀, Λ*
- **Physics modules**:
  - `topology.py`: β₁ = 12 (gauge group), n_inst = 3 (fermion generations)
  - `emergent_gravity.py`: Einstein equations, graviton propagator, LIV
  - `flavor_mixing.py`: CKM, PMNS matrices, neutrino sector
  - `electroweak.py`: Higgs boson, W/Z masses, Weinberg angle
  - `strong_cp.py`: θ = 0, algorithmic axion, PQ symmetry
  - `quantum_mechanics.py`: Born rule, decoherence, Lindblad equation
- **Approach**: Analytical over stochastic - constants derived, not fitted
- **Fixed point values** (Eq. 1.14): λ̃* = 48π²/9 ≈ 52.64, γ̃* = 32π²/3 ≈ 105.28, μ̃* = 16π² ≈ 157.91
- **Key classes**:
  - `CosmicFixedPoint`: The unique IR attractor, use via `find_fixed_point()`
  - `BetaFunctions`: One-loop β-functions, evaluate with `.evaluate(λ, γ, μ)`
  - `StandardModelTopology`: Complete SM derivation from β₁ and n_inst
  - `NeutrinoSector`: Normal hierarchy, Majorana nature, 12-digit masses
  - `ElectroweakSector`: Higgs VEV, W/Z masses, Weinberg angle
  - `StrongCPResolution`: θ = 0, algorithmic axion predictions
  - `EmergentQuantumMechanics`: Born rule and measurement emergence

#### v18 Code Patterns
```python
# Import v18 components
from irh.core.v18 import (
    CosmicFixedPoint, find_fixed_point, BetaFunctions,
    StandardModelTopology, compute_emergent_gravity_summary,
    CKMMatrix, PMNSMatrix, NeutrinoSector,
    ElectroweakSector, StrongCPResolution, EmergentQuantumMechanics
)

# Get fixed point and verify
fp = find_fixed_point()
assert fp.verify()["is_fixed_point"]

# Compute Standard Model emergence
sm = StandardModelTopology()
result = sm.compute_full_derivation()
assert result["gauge_sector"]["beta_1"] == 12  # SU(3)×SU(2)×U(1)
assert result["matter_sector"]["n_inst"] == 3   # 3 generations

# Get neutrino predictions
neutrino = NeutrinoSector()
assert neutrino.compute_mass_hierarchy()["hierarchy"] == "normal"
masses = neutrino.compute_absolute_masses()
print(f"Σmν = {masses['sum_masses_eV']:.6f} eV")  # ≈ 0.058 eV

# Electroweak sector
ew = ElectroweakSector()
print(ew.compute_full_sector()["higgs"]["mass"])  # m_H ≈ 125 GeV

# Strong CP resolution
cp = StrongCPResolution()
print(cp.verify_resolution())  # θ = 0, resolved = True

# Emergent QM
qm = EmergentQuantumMechanics()
print(qm.get_summary()["born_rule"])  # Derived, not postulated
```

## Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest python/tests/ -v

# Run v16 tests only
pytest python/tests/v16/ -v

# Run v18 tests only
pytest python/tests/v18/ -v

# Run with coverage
pytest python/tests/ --cov=irh --cov-report=html
```

### Writing Tests
- **Location**: Place tests in `python/tests/v16/` or `python/tests/v18/` matching the module
- **Naming**: Use `test_*.py` for files, `test_*` for functions
- **Structure**: Use pytest classes to group related tests (e.g., `class TestAlgorithmicHolonomicState`)
- **Docstrings**: Include equation references in test docstrings
- **Assertions**: Use `np.isclose()` or `assert_allclose()` for floating-point comparisons

Example test pattern:
```python
def test_beta_lambda_at_fixed_point():
    """β_λ should vanish at the fixed point (Eq.1.14)."""
    from irh.core.v18.rg_flow import BetaFunctions, find_fixed_point
    
    fp = find_fixed_point()
    beta = BetaFunctions()
    result = beta.beta_lambda(fp.lambda_star, fp.gamma_star, fp.mu_star)
    assert np.isclose(result, 0.0, atol=1e-10)
```

### Testing Best Practices
- Test both normal cases and edge cases
- Test phase normalization and wrapping
- Verify numerical precision (12+ decimals for constants)
- Test input validation and error handling
- Reference manuscript equations in test names and docstrings

## Important References

### Theory Documentation
- **v16 Manuscript**: `docs/manuscripts/IRHv16.md` - Core v16.0 theory
- **v16 Supplementary**: `docs/manuscripts/IRHv16_Supplementary_Vol_1-5.md` - Detailed derivations
- **v18 Manuscript**: `docs/manuscripts/IRHv18.md` - Analytical cGFT theory
- **Architecture**: `docs/v16_ARCHITECTURE.md` - v16 system architecture
- **Roadmap**: `docs/v16_IMPLEMENTATION_ROADMAP.md` - Implementation phases

### Development Guides
- **Contributing**: `CONTRIBUTING.md` - Contribution guidelines
- **Quickstart**: `docs/QUICKSTART.md` - Getting started guide
- **README**: `README.md` - Project overview and status

### Phase Status Documents
- **Phase 1**: `PHASE_1_STATUS.md`, `PHASE_1_COMPLETION_SUMMARY.md`
- **Phase 2**: `PHASE_2_STATUS.md` - Current phase (Exascale Infrastructure)

## Code Examples

### Creating an Algorithmic Holonomic State (v16)
```python
import numpy as np
from irh.core.v16.ahs import AlgorithmicHolonomicState

# Create AHS with binary string and phase
ahs = AlgorithmicHolonomicState(
    binary_string="101010",  # or b"101010" or bytearray(b"101010")
    holonomic_phase=np.pi / 4
)

# Access properties
amplitude = ahs.complex_amplitude  # e^{iφ}
info = ahs.information_content     # 6 bits
phase = ahs.holonomic_phase        # normalized to [0, 2π)
```

### Computing Normalized Compression Distance (NCD)
```python
from irh.core.v16.acw import normalized_compression_distance

# Multi-fidelity NCD with adaptive selection
ncd = normalized_compression_distance(
    binary_string1="10101010",
    binary_string2="11001100",
    fidelity="AUTO"  # or "HIGH", "MEDIUM", "LOW"
)
```

### Fixed Point Computation (v18)
```python
from irh.core.v18.rg_flow import find_fixed_point, BetaFunctions

# Compute analytical fixed point (Eq.1.14)
fixed_point = find_fixed_point()
# Returns CosmicFixedPoint with: λ̃* = 48π²/9, γ̃* = 32π²/3, μ̃* = 16π²

# Verify β-functions vanish at fixed point
beta = BetaFunctions()
beta_vals = beta.evaluate(fixed_point.lambda_star, fixed_point.gamma_star, fixed_point.mu_star)
assert all(abs(b) < 1e-8 for b in beta_vals)  # All β ≈ 0
```

### Standard Model Derivation (v18)
```python
from irh.core.v18 import StandardModelTopology, NeutrinoSector

# Derive complete Standard Model structure
sm = StandardModelTopology()
result = sm.compute_full_derivation()

# Gauge group from β₁ = 12
assert result["gauge_sector"]["beta_1"] == 12  # SU(3)×SU(2)×U(1)
print(result["gauge_sector"]["decomposition"])  # {"SU3": 8, "SU2": 3, "U1": 1}

# Fermion generations from n_inst = 3
assert result["matter_sector"]["n_inst"] == 3

# Neutrino predictions
neutrino = NeutrinoSector()
print(neutrino.compute_mass_hierarchy())  # {"hierarchy": "normal", ...}
print(neutrino.compute_majorana_nature())  # {"nature": "Majorana", ...}
masses = neutrino.compute_absolute_masses()
print(f"Σmν = {masses['sum_masses_eV']:.6f} eV")  # ≈ 0.058 eV
```

### Emergent Gravity and LIV (v18)
```python
from irh.core.v18 import compute_emergent_gravity_summary, LorentzInvarianceViolation

# Get complete gravity predictions
gravity = compute_emergent_gravity_summary()
print(f"Λ_* = {gravity['cosmological_constant']['Lambda_star']:.4e} m⁻²")

# Lorentz Invariance Violation parameter
liv = LorentzInvarianceViolation()
xi = liv.compute_xi()
print(f"ξ = {xi['xi']:.6e}")  # ≈ 1.93 × 10⁻⁴ (testable!)
```

### Phase Handling Patterns
```python
import numpy as np

# Wrap phase to [0, 2π)
phase = np.mod(angle, 2 * np.pi)

# Compute wrapped phase difference in [-π, π]
from irh.core.v16.ahs import _wrapped_phase_difference
diff = _wrapped_phase_difference(phase1, phase2)

# Compare phases with tolerance
PHASE_TOLERANCE = 1e-10
are_equal = abs(diff) <= PHASE_TOLERANCE
```

## Common Patterns and Conventions

### Input Normalization
Always normalize string/bytes inputs using helper functions:
```python
def _to_bytes(data: Union[str, bytes, bytearray]) -> bytes:
    """Normalize str/bytes/bytearray to bytes."""
    if isinstance(data, bytes):
        return data
    elif isinstance(data, bytearray):
        return bytes(data)
    elif isinstance(data, str):
        return data.encode('ascii')
    else:
        raise TypeError("Expected str, bytes, or bytearray")
```

### Equation References
Always reference manuscript equations in comments and docstrings:
```python
# Implements Eq.4.1 from IRH v16.0 Manuscript
harmony = compute_spectral_zeta_functional(network, epsilon)

# Fixed point values from Eq.1.14 (IRH v18.0)
FIXED_POINT_LAMBDA = 48 * np.pi**2 / 9
```

### Complex Number Handling
Wrap `np.exp()` results when returning from `__complex__` to avoid deprecation warnings:
```python
def __complex__(self) -> complex:
    """Return complex amplitude e^{iφ}."""
    return complex(np.exp(1j * self.holonomic_phase))
```

### Dataclass Patterns
Use `__post_init__` for validation and normalization:
```python
from dataclasses import dataclass

@dataclass
class AlgorithmicHolonomicState:
    binary_string: Union[str, bytes, bytearray]
    holonomic_phase: float
    
    def __post_init__(self):
        # Validate and normalize inputs
        # Compute derived properties
        pass
```

## Domain-Specific Knowledge

### Algorithmic Holonomic States (AHS)
- Fundamental ontological primitive in IRH
- Pair of (binary_string, holonomic_phase)
- Phase φ ∈ [0, 2π) derived from non-commutative algebra
- Two states distinguishable if binary_string OR phase differs

### Normalized Compression Distance (NCD)
- Measures algorithmic similarity between binary strings
- Multi-fidelity evaluation: HIGH, MEDIUM, LOW based on data size
- Adaptive selection for optimal performance/accuracy trade-off

### Harmony Functional
- Spectral zeta-regularized functional H(ε)
- Computed from complex Laplacian eigenvalues
- Critical for deriving universal constant C_H
- Detects degenerate networks (trace < 1e-12 OR det < 1e-12)

### Universal Constant C_H
- Derived as renormalization group fixed point
- v16 value: 0.045935703598 (12+ decimal precision)
- NOT fitted - emerges from network criticality

### Phase Quantization
- Occurs through Adaptive Resonance Optimization (ARO)
- Uses genetic algorithms for network optimization
- Unitary evolution on complex-valued states

### v18-Specific Concepts

#### Cosmic Fixed Point
- The unique infrared attractor of the RG flow
- Analytically derived values: λ̃* = 48π²/9, γ̃* = 32π²/3, μ̃* = 16π²
- All physics emerges at this fixed point
- Access via `CosmicFixedPoint()` or `find_fixed_point()`

#### Informational Group Manifold G_inf
- G_inf = SU(2) × U(1)_φ
- SU(2): Encodes minimal non-commutative algebra (quaternions)
- U(1)_φ: Holonomic phase φ ∈ [0, 2π)
- Use `GInfElement.random(rng)` for sampling

#### Topological Invariants
- **First Betti number β₁ = 12**: Determines gauge group SU(3)×SU(2)×U(1)
- **Instanton number n_inst = 3**: Determines three fermion generations
- Both are topologically protected integers

#### Vortex Wave Pattern (VWP)
- Stable topological defects = fermions
- Topological complexity K_f determines mass hierarchy
- K_e = 1, K_μ ≈ 207, K_τ ≈ 3477

#### Lorentz Invariance Violation
- Testable prediction: ξ = C_H/(24π²) ≈ 1.93 × 10⁻⁴
- Modified dispersion: E² = p²c² + ξ × E³/(E_Planck × c²)
- Detectable via high-energy gamma-ray astronomy

#### Electroweak Sector
- Higgs VEV v = 246.22 GeV emerges from μ̃*/λ̃*
- W mass (80.4 GeV), Z mass (91.2 GeV) from Higgs mechanism
- Weinberg angle sin²θ_W = 0.231 from gauge coupling unification

#### Strong CP Problem
- θ_QCD = 0 via emergent algorithmic axion
- Peccei-Quinn symmetry emerges from cGFT
- Axion mass m_a ≈ 5.7 μeV, decay constant f_a ≈ 10¹² GeV

#### Emergent Quantum Mechanics
- Born rule (P = |ψ|²) derived from EAT collective dynamics
- Measurement = decoherence in environment
- Lindblad equation for open systems
- Unitarity preserved at substrate level

## Key Principles

1. **Reproducibility**: Use fixed random seeds, well-defined precision
2. **Certification**: Track error bounds for all critical calculations
3. **Non-circularity**: Derive constants from first principles, never fit them
4. **Verification**: Test against known analytical results
5. **Documentation**: Link all code to manuscript equations
6. **Backward compatibility**: v16 extends v15 without breaking changes
7. **Incremental implementation**: Test each component before integration

## Security and Performance

### Dependency Management
- Before adding new dependencies, check for security vulnerabilities
- Supported ecosystems: pip, npm, actions (use gh-advisory-database)
- Pin version numbers in requirements.txt and pyproject.toml

### Performance Considerations
- Use NumPy vectorized operations over Python loops
- For exascale: stub MPI/GPU backends for future parallelization
- Multi-fidelity NCD: automatic selection based on data size
- Memory-efficient: streaming algorithms for large datasets

### Python Compatibility
- Support Python 3.11 and 3.12
- Use `datetime.timezone.utc` (not `datetime.UTC`) for broader compatibility
- Test with both supported versions in CI

## Warnings and Known Issues

- **Phase equality**: Use absolute tolerance (PHASE_TOLERANCE = 1e-10), not relative
- **numpy subclass deprecation**: Wrap `np.exp()` results in `complex()` when needed
- **Git operations**: Never use `git reset` or `git rebase` (no force push available)
- **Temporary files**: Create in `/tmp` directory to avoid committing build artifacts
- **Build artifacts**: Ensure `.gitignore` excludes `node_modules`, `dist`, `__pycache__`, etc.

## Getting Help

- **Open an issue** on GitHub for bugs or feature requests
- **Check manuscripts** in `docs/manuscripts/` for theoretical questions
- **Review CONTRIBUTING.md** for contribution guidelines
- **Check phase status docs** for current development roadmap

---

*"A complete, exascale-ready computational framework achieving 12+ decimal precision in fundamental constant derivations, with definitive empirical verification at the theoretical and computational frontiers."*

---

## Addendum: Fast operational checklist (v18-first)

- **What this repo is**: IRH research code; active Python package in `python/src/irh` (v16–v18), legacy `src/` + `.wl` in root, webapp in `webapp/`, docs in `docs/`.
- **Bootstrap (validated)**: `python -m pip install -e .[dev]` (Python 3.11/3.12). Set PYTHONPATH: in `python/` use `export PYTHONPATH=$(pwd)/src`; in repo root use `export PYTHONPATH=$PWD` for legacy/tests.
- **Tests**:  
  - **v18 (72 tests)**: `cd python && export PYTHONPATH=$(pwd)/src && pytest tests/v18/ -v` (passes ~0.6s)
  - v16: `cd python && export PYTHONPATH=$(pwd)/src && pytest tests/v16/ -v`
  - Legacy: `cd /home/runner/work/Intrinsic-Resonance-Holography-/Intrinsic-Resonance-Holography- && export PYTHONPATH=$PWD && pytest tests/test_v16_core.py`
- **Lint/type/build**:  
  - `cd python && export PYTHONPATH=$(pwd)/src && ruff check src/`  
  - `black --check src/ tests/ --line-length 100` (within `python/`)  
  - `mypy src/irh/ --ignore-missing-imports`  
  - Root legacy (if touched): `ruff check src/ --ignore E501`, `black --check src/ tests/`.  
  - Build: `python -m build && twine check dist/*`.
- **Run/demo**: `python project_irh_v16.py` (root; set PYTHONPATH); web backend `cd webapp/backend && pip install -r requirements.txt && python app.py`; frontend `cd webapp/frontend && npm install && npm run dev` (Vite :5173).
- **v18 quick verification**:
  ```python
  from irh.core.v18 import StandardModelTopology, NeutrinoSector
  sm = StandardModelTopology()
  assert sm.verify_standard_model()  # β₁=12, n_inst=3
  neutrino = NeutrinoSector()
  assert neutrino.compute_mass_hierarchy()["hierarchy"] == "normal"
  ```
- **Conventions**: PEP 8, line length 100, NumPy docstrings with equation refs; phase wrapping via `np.mod(angle, 2*np.pi)` and `_wrapped_phase_difference` with `PHASE_TOLERANCE=1e-10`; input normalization via `_to_bytes`; wrap `np.exp(...)` in `complex(...)`.
- **CI signals**: `.github/workflows/ci.yml` (pytest on `tests/`, ruff on `src/`, mypy on `src/irh_v10`) and `ci-cd.yml` (black/mypy, v16 legacy tests, python package tests/coverage, docs check, benchmarks, Wolfram notice, release stub). Prefer Python 3.12 and correct PYTHONPATH to mirror CI.
- **Agent reminders**: keep changes minimal, place new code in `python/src/irh/...` with matching tests in `python/tests/...`, avoid new deps unless required, and trust these instructions before searching.

## v18 Module Summary (12 modules)

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `group_manifold.py` | G_inf = SU(2)×U(1) | `SU2Element`, `GInfElement`, `compute_ncd_distance` |
| `cgft_field.py` | φ(g₁,g₂,g₃,g₄) field | `cGFTFieldDiscrete`, `BiLocalField`, `CondensateState` |
| `cgft_action.py` | S_kin + S_int + S_hol | `cGFTAction`, `compute_harmony_functional` |
| `rg_flow.py` | β-functions, fixed point | `BetaFunctions`, `CosmicFixedPoint`, `find_fixed_point` |
| `spectral_dimension.py` | d_spec → 4 | `SpectralDimensionFlow`, `verify_theorem_2_1` |
| `physical_constants.py` | α, masses, w₀, Λ | `FineStructureConstant`, `FermionMassCalculator` |
| `topology.py` | β₁=12, n_inst=3 | `StandardModelTopology`, `VortexWavePattern` |
| `emergent_gravity.py` | Einstein, graviton, LIV | `EinsteinEquations`, `LorentzInvarianceViolation` |
| `flavor_mixing.py` | CKM, PMNS, neutrinos | `CKMMatrix`, `PMNSMatrix`, `NeutrinoSector` |
| `electroweak.py` | Higgs, W/Z, θ_W | `HiggsBoson`, `ElectroweakSector`, `WeinbergAngle` |
| `strong_cp.py` | θ=0, axion | `AlgorithmicAxion`, `StrongCPResolution` |
| `quantum_mechanics.py` | Born rule, Lindblad | `BornRule`, `EmergentQuantumMechanics` |
