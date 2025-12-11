# Contributing to Intrinsic Resonance Holography

Thank you for your interest in contributing to IRH! This document provides guidelines for contributing to the codebase.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [IRH v17.0 Module Architecture](#irh-v170-module-architecture)

## Code of Conduct

This project adheres to scientific integrity and open collaboration. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- Git
- Basic understanding of quantum field theory and renormalization group methods (for physics contributions)

### Repository Structure

```
Intrinsic-Resonance-Holography-/
├── python/
│   ├── src/irh/
│   │   ├── core/
│   │   │   ├── v16/          # v16 implementation (ACW, AHS, ARO)
│   │   │   └── v17/          # v17 cGFT implementation (NEW)
│   │   ├── predictions/      # Physical constant predictions
│   │   └── ...
│   └── tests/
│       ├── v16/              # v16 tests
│       └── v17/              # v17 tests (NEW)
├── notebooks/                # Jupyter notebooks
├── webapp/                   # Web interface (FastAPI + Streamlit)
├── legacy/                   # Deprecated code
├── docs/
│   └── manuscripts/          # IRH theory manuscripts
└── pyproject.toml
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-.git
   cd Intrinsic-Resonance-Holography-
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev,webapp,notebooks]"
   ```

4. Install pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Maximum line length: 100 characters
- Use `black` for formatting:
  ```bash
  black python/src/irh/ --line-length 100
  ```

### Docstrings

Use NumPy-style docstrings:

```python
def compute_fixed_point(
    initial_guess: Optional[Tuple[float, float, float]] = None,
    tol: float = 1e-14,
) -> Tuple[float, float, float]:
    """
    Compute the unique non-Gaussian infrared fixed point (Eq.1.14).
    
    Parameters
    ----------
    initial_guess : tuple of 3 floats, optional
        Initial guess for [λ̃, γ̃, μ̃].
    tol : float, optional
        Tolerance for the solver.
    
    Returns
    -------
    tuple of 3 floats
        The fixed-point values (λ̃*, γ̃*, μ̃*).
    
    Notes
    -----
    The analytic solution (Eq.1.14) is:
        λ̃* = 48π²/9
        γ̃* = 32π²/3
        μ̃* = 16π²
    
    References
    ----------
    IRH v17.0 Manuscript, Eq.1.14
    """
```

### Equation References

Always cite equation numbers from the IRH manuscripts:
- v17: `docs/manuscripts/IRHv17.md`
- v16: `docs/manuscripts/IRHv16.md`

Example: "Implements Eq.1.13 from IRH v17.0"

## Testing

### Running Tests

```bash
# Run all tests
pytest python/tests/ -v

# Run v17 tests only
pytest python/tests/v17/ -v

# Run with coverage
pytest python/tests/ --cov=irh --cov-report=html
```

### Writing Tests

- Place tests in the appropriate `tests/` subdirectory
- Use descriptive test names: `test_beta_lambda_at_fixed_point`
- Test both normal cases and edge cases
- Reference equation numbers in test docstrings

Example:
```python
def test_beta_lambda_at_fixed_point():
    """β_λ should vanish at the fixed point (Eq.1.14)."""
    from irh.core.v17.beta_functions import beta_lambda, FIXED_POINT_LAMBDA
    
    result = beta_lambda(FIXED_POINT_LAMBDA)
    assert_allclose(result, 0.0, atol=1e-10)
```

## Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear commit messages

3. Run tests and ensure they pass:
   ```bash
   pytest python/tests/ -v
   ```

4. Update documentation if needed

5. Submit a pull request with:
   - Clear description of changes
   - Reference to relevant issues
   - Reference to manuscript equations

## IRH v17.0 Module Architecture

### Core Modules

#### `irh.core.v17.beta_functions`
- One-loop β-functions (Eq.1.13)
- Fixed-point computation (Eq.1.14)
- Stability matrix analysis

#### `irh.core.v17.constants`
- Universal constant C_H (Eq.1.15-1.16)
- Fine-structure constant α⁻¹ (Eq.3.4-3.5)
- Dark energy equation of state w₀ (Eq.2.22-2.23)
- Topological invariants (Eq.3.1-3.2)
- Fermion masses (Eq.3.6-3.8)

#### `irh.core.v17.spectral_dimension`
- Spectral dimension flow (Eq.2.8-2.9)
- Graviton anomalous dimension
- Asymptotic safety verification

#### `irh.core.v17.cgft_action`
- cGFT action on G_inf = SU(2) × U(1)
- Kinetic term S_kin (Eq.1.1)
- Interaction term S_int (Eq.1.2-1.3)
- Holographic term S_hol (Eq.1.4)

### Extending the Code

To add new physics modules:

1. Create a new file in `python/src/irh/core/v17/`
2. Implement classes/functions with full docstrings
3. Add to `__init__.py` exports
4. Write comprehensive tests
5. Reference manuscript equations

### Key Principles

1. **Analytical over stochastic**: v17 derives constants analytically, not through optimization
2. **Reproducibility**: Use fixed seeds and well-defined precision
3. **Verification**: Test against known analytic results
4. **Documentation**: Link all code to manuscript equations

## Questions?

Open an issue on GitHub or refer to the manuscripts in `docs/manuscripts/`.

---

*All constants of Nature are now derived, not discovered.*
