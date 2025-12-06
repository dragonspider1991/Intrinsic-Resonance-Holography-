# IRH v13.0 Examples

This directory contains example scripts demonstrating the use of the IRH v13.0 framework.

## Examples

### Example 1: Basic ARO Optimization
**File**: `example1_basic_optimization.py`  
**Runtime**: ~2 minutes  
**Demonstrates**: Basic network initialization and optimization

```bash
python examples/v13/example1_basic_optimization.py
```

### Example 2: Computing Physical Predictions
**File**: `example2_compute_predictions.py`  
**Runtime**: ~3 minutes  
**Demonstrates**: Computing all v13.0 predictions from optimized network

```bash
python examples/v13/example2_compute_predictions.py
```

### Example 3: Parameter Exploration
**File**: `example3_parameter_exploration.py`  
**Runtime**: ~5-10 minutes  
**Demonstrates**: Testing different network configurations

```bash
python examples/v13/example3_parameter_exploration.py
```

## Requirements

All examples require:
- Python 3.8+
- NumPy, SciPy, NetworkX
- IRH v13.0 framework installed

Optional:
- Matplotlib (for plotting in Example 1)

## Expected Outputs

### Example 1
- Optimization history
- Final Harmony value
- Optional plot saved to `examples/v13/example1_optimization_history.png`

### Example 2
- All 4 v13.0 predictions computed
- Comparison with experimental values
- Interpretation guidance

### Example 3
- Comparison table of different configurations
- Analysis of parameter effects
- Recommendations for use cases

## Notes

- Small network examples (N < 500) won't produce converged predictions
- They validate that the framework works correctly
- For actual v13.0 validation, use the Cosmic Fixed Point Test:
  ```bash
  python experiments/cosmic_fixed_point_test.py --N 1000 --iterations 5000
  ```
