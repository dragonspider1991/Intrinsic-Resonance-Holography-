# IRH v13.0 Quick Reference Guide

## Quick Start

### Test the Framework
```bash
cd /home/runner/work/Intrinsic-Resonance-Holography-/Intrinsic-Resonance-Holography-

# Test imports
python -c "from src.core import AROOptimizer, harmony_functional; print('✓ OK')"

# Run basic test
python -c "
from src.core import AROOptimizer
opt = AROOptimizer(N=100, rng_seed=42)
W = opt.initialize_network(scheme='geometric', connectivity_param=0.1)
print(f'Network: N={opt.N}, edges={W.nnz}')
"

# Run full integration test
python tests/integration/test_v13_core.py
```

### Common Code Patterns

#### Initialize and Optimize Network
```python
from src.core import AROOptimizer, harmony_functional

# Create optimizer
opt = AROOptimizer(N=200, rng_seed=42)

# Initialize network (try higher connectivity)
W = opt.initialize_network(
    scheme='geometric',
    connectivity_param=0.1,  # Increase if too sparse
    d_initial=4
)

# Check initial harmony
S_H_initial = harmony_functional(W)
print(f"Initial S_H = {S_H_initial}")

# Optimize
opt.optimize(
    iterations=100,
    learning_rate=0.01,
    mutation_rate=0.05,
    temp_start=1.0,
    verbose=True
)

# Get best network
W_optimal = opt.best_W
print(f"Best S_H = {opt.best_S}")
```

#### Compute Topological Invariants
```python
from src.topology import (
    calculate_frustration_density,
    derive_fine_structure_constant
)

# Compute frustration
rho_frust = calculate_frustration_density(W_optimal, max_cycles=1000)

# Derive fine-structure constant
alpha_inv, match = derive_fine_structure_constant(rho_frust)

print(f"ρ_frust = {rho_frust:.6f}")
print(f"α⁻¹ = {alpha_inv:.3f} (experimental: 137.036)")
print(f"Match: {match}")
```

#### Compute Dimensional Metrics
```python
from src.metrics import (
    spectral_dimension,
    dimensional_coherence_index
)

# Compute spectral dimension
d_spec, info = spectral_dimension(W_optimal, method='heat_kernel')
print(f"d_spec = {d_spec:.3f} (target: 4.0)")
print(f"Status: {info['status']}")

# Compute coherence index
chi_D, components = dimensional_coherence_index(W_optimal, target_d=4)
print(f"χ_D = {chi_D:.3f}")
print(f"E_R = {components['E_R']:.3f} (dimensional residue)")
```

## Debugging Issues

### Issue: S_H returns -inf

**Cause**: Network too sparse, not enough eigenvalues

**Solutions**:
1. Increase `connectivity_param` (try 0.1 to 0.2)
2. Use larger N (try N >= 200)
3. Check edge count: `W.nnz` should be > N*log(N)

```python
# Diagnostic
print(f"N = {N}")
print(f"Edges = {W.nnz}")
print(f"Edge density = {W.nnz / (N*N):.4f}")
print(f"Target: ~{np.log(N) / N:.4f} or higher")
```

### Issue: Optimization doesn't improve S_H

**Cause**: Perturbations may be breaking network connectivity

**Solutions**:
1. Reduce `learning_rate` (try 0.001 to 0.01)
2. Reduce `mutation_rate` (try 0.01 to 0.05)
3. Start with higher `temp_start` (try 2.0 to 5.0)

```python
opt.optimize(
    iterations=200,
    learning_rate=0.005,      # Lower
    mutation_rate=0.02,       # Lower
    temp_start=2.0,           # Higher
    verbose=True
)
```

### Issue: α⁻¹ prediction way off

**Cause**: Network hasn't converged to Cosmic Fixed Point

**Solutions**:
1. Run more iterations (try 10,000+)
2. Use larger network (N >= 1000)
3. Check S_H is improving (should increase over time)

## File Locations

### Source Code
- `src/core/harmony.py` - Harmony Functional
- `src/core/aro_optimizer.py` - ARO Engine
- `src/topology/invariants.py` - Frustration, Betti numbers
- `src/metrics/dimensions.py` - Spectral dimension, χ_D

### Tests
- `tests/integration/test_v13_core.py` - Main integration tests

### Documentation
- `docs/STRUCTURE_v13.md` - Repository structure
- `docs/manuscripts/IRH_v13_0_Theory.md` - Theory (placeholder)
- `AGENT_HANDOFF.md` - Full implementation status

## Next Steps for Development

1. **Fix network initialization** (src/core/aro_optimizer.py:97)
2. **Fix eigenvalue computation** (src/core/harmony.py:108)
3. **Add convergence diagnostics** to AROOptimizer
4. **Implement Betti numbers** (src/topology/invariants.py:199)
5. **Implement full ℰ_H, ℰ_C** (src/metrics/dimensions.py:84-88)

## Performance Tips

- For quick tests: N = 50-100, iterations = 20-50
- For validation: N = 500-1000, iterations = 500-1000
- For production: N = 10,000+, iterations = 10,000+
- Use `verbose=False` for batch runs
- Profile with: `python -m cProfile -s tottime script.py`

## Testing Commands

```bash
# Run specific test
pytest tests/integration/test_v13_core.py::TestV13Framework::test_aro_initialization -v

# Run all integration tests
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run in verbose mode
python tests/integration/test_v13_core.py
```

## Git Workflow

```bash
# Check status
git status

# See changes
git diff

# Stage changes (done automatically by report_progress)
# Commit (done automatically by report_progress)

# View commits
git log --oneline -10

# View specific commit
git show d33d0cf
```

---

**Last Updated**: 2025-12-06 (Commit d33d0cf)
