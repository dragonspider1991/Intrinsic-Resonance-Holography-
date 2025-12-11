# Legacy Code Directory

This directory contains deprecated code from earlier IRH versions that has been superseded by the v17.0 analytical framework.

## Deprecation Notice

**As of IRH v17.0, the following approaches are deprecated:**

1. **Stochastic/Global Optimizers**: The "HarmonyOptimizer" and genetic algorithm-based approaches from v16 and earlier are no longer used. IRH v17.0 replaces stochastic parameter search with **analytically derived constants** from the unique infrared fixed point of the cGFT.

2. **ARO (Algorithmic Resonance Optimization)**: While the ARO module remains useful for exploration, the core physics constants are no longer discovered through optimization but are **derived from first principles**.

## Why Legacy?

IRH v17.0 represents a fundamental paradigm shift:

- **Before (v16 and earlier)**: Physical constants were "discovered" through optimization algorithms searching parameter space.
- **After (v17.0)**: All constants are **analytically computed** from the unique Cosmic Fixed Point of the complex-weighted Group Field Theory.

As stated in the v17.0 manuscript:
> "All constants of Nature are now derived, not discovered."

## Retained Modules

The following legacy modules are preserved for reference and backward compatibility:

- `v16/` - Complete v16 implementation (ACW, AHS, ARO, etc.)
- `harmony_optimizer/` - Original stochastic optimizer (deprecated)
- `genetic_search/` - Genetic algorithm implementations (deprecated)

## Migration Guide

To migrate from v16 to v17:

1. Replace `HarmonyOptimizer.optimize()` calls with direct use of fixed-point constants from `irh.core.v17.constants`
2. Replace iterative searches with analytical computations from `irh.core.v17.beta_functions`
3. Use the new `CGFTAction` class for field-theoretic computations

## Usage

Legacy code can still be imported:

```python
# Old way (deprecated)
from irh.legacy.harmony_optimizer import HarmonyOptimizer

# New way (recommended)
from irh.core.v17.constants import (
    FIXED_POINT_LAMBDA,
    FIXED_POINT_GAMMA,
    FIXED_POINT_MU,
    compute_alpha_inverse,
    compute_w0,
)
```

## References

- IRH v17.0 Manuscript: `docs/manuscripts/IRHv17.md`
- IRH v16.0 Manuscript: `docs/manuscripts/IRHv16.md`
- v16 Implementation Status: `docs/V16_STATUS.md`
