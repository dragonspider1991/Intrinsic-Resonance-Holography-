# API Documentation

This directory will contain auto-generated API documentation for IRH v13.0.

## Generating Documentation

To generate the API documentation, run:

```bash
# Using Sphinx (recommended)
sphinx-apidoc -o docs/api src/

# Or using pdoc
pdoc --html --output-dir docs/api src/
```

## Documentation Structure

The API documentation will cover:

- `src/core/` - Core ARO Engine and Harmony Functional
- `src/topology/` - Topological invariant calculations
- `src/cosmology/` - Cosmological simulations
- `src/metrics/` - Dimensional coherence metrics
- `src/utils/` - Mathematical utilities

---

*Documentation will be auto-generated from docstrings in the source code.*
