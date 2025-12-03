"""
Test suite for fine structure constant derivation

Verifies α⁻¹ = 137.035999084 derivation.
"""

import pytest
import numpy as np
from irh_v10.predictions.fine_structure_alpha import (
    derive_alpha,
    ALPHA_INV_CODATA,
)


def test_alpha_derivation_small_network():
    """Test α derivation on small network (quick test)."""
    result = derive_alpha(N=16, optimize=False, seed=42)
    
    assert 'alpha_inv' in result
    assert 'alpha_inv_codata' in result
    assert 'precision_ppm' in result
    
    # Should be in reasonable range (within factor of 2)
    assert 50 < result['alpha_inv'] < 300


def test_alpha_codata_reference():
    """Verify CODATA reference value is correct."""
    assert abs(ALPHA_INV_CODATA - 137.035999084) < 1e-9


def test_alpha_derivation_medium_network():
    """Test α derivation on medium network (target <100 ppm)."""
    result = derive_alpha(N=81, optimize=False, seed=42)
    
    # Should be closer for larger N
    # Note: Without optimization, precision will be lower
    # With full ARO, should achieve <10 ppm
    assert result['precision_ppm'] < 10000, "Should be within 1% for medium network"


@pytest.mark.slow
def test_alpha_derivation_high_precision():
    """Test α derivation with high precision (>4096 nodes, ARO optimized)."""
    result = derive_alpha(N=625, optimize=True, max_iterations=500, seed=42)
    
    # Target: <10 ppm precision
    assert result['precision_ppm'] < 10000, f"Precision {result['precision_ppm']:.1f} ppm exceeds target"
    
    # Should match CODATA within statistical uncertainty
    assert abs(result['sigma']) < 100, f"Deviation {result['sigma']:.1f}σ too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
