"""
Unit Tests for IRH v16.0 Numerics Module

Tests the certified numerics, precision constants, and error tracking
components of the v16.0 numerical precision framework.
"""

import pytest
import numpy as np
from src.numerics import (
    CertifiedValue,
    interval_arithmetic,
    certified_sum,
    certified_product,
    track_floating_point_error,
    C_H_CERTIFIED,
    EPSILON_THRESHOLD_CERTIFIED,
    Q_HOLONOMIC_CERTIFIED,
    ErrorBudget,
    create_error_budget,
    combine_errors,
    get_precision_target,
    validate_precision,
)


class TestCertifiedNumerics:
    """Tests for certified numerical operations."""
    
    def test_certified_value_creation(self):
        """Test creating a certified value."""
        val = CertifiedValue.from_value_and_error(1.0, 0.01, "test")
        assert val.value == 1.0
        assert val.error == 0.01
        assert val.lower_bound == 0.99
        assert val.upper_bound == 1.01
        assert 0.01 <= val.relative_error <= 0.011  # Approximately 1%
    
    def test_certified_addition(self):
        """Test addition with error propagation."""
        a = CertifiedValue.from_value_and_error(1.0, 0.01, "a")
        b = CertifiedValue.from_value_and_error(2.0, 0.01, "b")
        c = a + b
        
        assert c.value == 3.0
        assert c.error == 0.02  # Errors add in worst case
    
    def test_certified_addition_with_constant(self):
        """Test adding a constant (exact value)."""
        a = CertifiedValue.from_value_and_error(1.0, 0.01, "a")
        b = a + 5.0
        
        assert b.value == 6.0
        assert b.error == 0.01  # Error unchanged
    
    def test_certified_multiplication(self):
        """Test multiplication with error propagation."""
        a = CertifiedValue.from_value_and_error(2.0, 0.02, "a")
        b = CertifiedValue.from_value_and_error(3.0, 0.03, "b")
        c = a * b
        
        assert c.value == 6.0
        # Relative errors add: (0.02/2) + (0.03/3) = 0.01 + 0.01 = 0.02
        # Absolute error: 6 * 0.02 = 0.12
        assert abs(c.error - 0.12) < 1e-10
    
    def test_certified_multiplication_with_constant(self):
        """Test multiplying by a constant."""
        a = CertifiedValue.from_value_and_error(2.0, 0.02, "a")
        b = a * 3.0
        
        assert b.value == 6.0
        assert b.error == 0.06  # Error scales linearly
    
    def test_interval_arithmetic_sqrt(self):
        """Test square root with interval arithmetic."""
        a = CertifiedValue.from_value_and_error(4.0, 0.1, "a")
        b = interval_arithmetic('sqrt', a)
        
        assert abs(b.value - 2.0) < 1e-10
        assert b.error > 0  # Should have propagated error
    
    def test_interval_arithmetic_division(self):
        """Test division with interval arithmetic."""
        a = CertifiedValue.from_value_and_error(6.0, 0.1, "a")
        b = CertifiedValue.from_value_and_error(2.0, 0.02, "b")
        c = interval_arithmetic('div', a, b)
        
        assert abs(c.value - 3.0) < 1e-10
        assert c.error > 0
    
    def test_certified_sum_array(self):
        """Test certified sum of array."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = certified_sum(values)
        
        assert abs(result.value - 15.0) < 1e-10
        assert result.error > 0
    
    def test_certified_product_array(self):
        """Test certified product of array."""
        values = np.array([2.0, 3.0, 4.0])
        result = certified_product(values)
        
        assert abs(result.value - 24.0) < 1e-10
        assert result.error > 0
    
    def test_track_floating_point_error(self):
        """Test floating-point error tracking."""
        result = 1000.0
        operands = np.ones(100)
        certified = track_floating_point_error("test_sum", result, operands, n_operations=100)
        
        assert certified.value == result
        assert certified.error > 0


class TestPrecisionConstants:
    """Tests for precision constants."""
    
    def test_harmony_constant_precision(self):
        """Test C_H has correct value and precision."""
        assert abs(C_H_CERTIFIED.value - 0.045935703598) < 1e-12
        assert C_H_CERTIFIED.error == 1e-12
    
    def test_epsilon_threshold_precision(self):
        """Test epsilon threshold value."""
        assert abs(EPSILON_THRESHOLD_CERTIFIED.value - 0.730129) < 1e-6
        assert EPSILON_THRESHOLD_CERTIFIED.error == 1e-6
    
    def test_holonomic_quantization(self):
        """Test q constant derived from fine structure."""
        expected_q = 1.0 / 137.035999084
        assert abs(Q_HOLONOMIC_CERTIFIED.value - expected_q) < 1e-10
    
    def test_get_precision_target(self):
        """Test retrieving precision targets."""
        assert get_precision_target('fine_structure_constant') == 12
        assert get_precision_target('spectral_dimension') == 3
    
    def test_validate_precision_pass(self):
        """Test precision validation with matching values."""
        computed = CertifiedValue.from_value_and_error(137.035999, 1e-6, "computed")
        target = CertifiedValue.from_value_and_error(137.035999084, 1e-9, "target")
        
        # Should pass at 5 decimal places
        assert validate_precision(computed, target, required_decimals=5)
    
    def test_validate_precision_fail(self):
        """Test precision validation with non-matching values."""
        computed = CertifiedValue.from_value_and_error(137.037, 1e-5, "computed")
        target = CertifiedValue.from_value_and_error(137.035999084, 1e-9, "target")
        
        # Should fail at 3 decimal places (difference is ~0.001)
        assert not validate_precision(computed, target, required_decimals=3)


class TestErrorTracking:
    """Tests for error budget tracking."""
    
    def test_error_budget_creation(self):
        """Test creating an error budget."""
        budget = ErrorBudget(
            numerical_error=1e-6,
            statistical_error=1e-5,
            finite_size_error=1e-4,
            theoretical_error=1e-3
        )
        
        total = budget.total_error()
        expected = np.sqrt(1e-6**2 + 1e-5**2 + 1e-4**2 + 1e-3**2)
        assert abs(total - expected) < 1e-10
    
    def test_dominant_error_source(self):
        """Test identifying dominant error source."""
        budget = ErrorBudget(
            numerical_error=1e-8,
            statistical_error=1e-7,
            finite_size_error=1e-6,
            theoretical_error=1e-4
        )
        
        assert budget.dominant_error_source() == 'theoretical'
    
    def test_create_error_budget_auto(self):
        """Test automatic error budget creation."""
        budget = create_error_budget(
            N=10000,
            n_samples=1000,
            n_operations=5000
        )
        
        assert budget.finite_size_error > 0
        assert budget.statistical_error > 0
        assert budget.numerical_error > 0
    
    def test_combine_errors(self):
        """Test combining multiple error budgets."""
        budget1 = ErrorBudget(numerical_error=1e-6, statistical_error=1e-5)
        budget2 = ErrorBudget(numerical_error=2e-6, finite_size_error=1e-5)
        
        combined = combine_errors([budget1, budget2])
        
        # Numerical: sqrt(1e-6^2 + 2e-6^2) = sqrt(5) * 1e-6
        expected_num = np.sqrt(5) * 1e-6
        assert abs(combined.numerical_error - expected_num) < 1e-10
        
        # Statistical: just from budget1
        assert abs(combined.statistical_error - 1e-5) < 1e-10
    
    def test_relative_error(self):
        """Test relative error calculation."""
        budget = ErrorBudget(
            numerical_error=0.01,
            statistical_error=0.01
        )
        
        rel_err = budget.relative_error(100.0)
        expected = np.sqrt(0.01**2 + 0.01**2) / 100.0
        assert abs(rel_err - expected) < 1e-10


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_certified_calculation_with_error_budget(self):
        """Test a complete certified calculation with error tracking."""
        # Setup
        a = CertifiedValue.from_value_and_error(1.0, 1e-10, "measurement_a")
        b = CertifiedValue.from_value_and_error(2.0, 1e-10, "measurement_b")
        
        # Calculate
        c = a + b
        d = c * c
        
        # Create error budget
        budget = ErrorBudget(
            numerical_error=d.error,
            statistical_error=1e-9,
            finite_size_error=1e-8,
            theoretical_error=1e-7
        )
        
        # Validate
        assert d.value == 9.0  # (1+2)^2 = 9
        assert budget.total_error() > 0
        assert budget.dominant_error_source() == 'theoretical'
    
    def test_precision_validation_workflow(self):
        """Test complete precision validation workflow."""
        # Compute a "predicted" value
        predicted = CertifiedValue.from_value_and_error(
            137.035999,
            1e-6,
            "irh_prediction"
        )
        
        # Compare to CODATA (simulated)
        experimental = CertifiedValue.from_value_and_error(
            137.035999084,
            2.1e-8,
            "codata_2022"
        )
        
        # Check precision requirement
        required_decimals = 6
        is_valid = validate_precision(predicted, experimental, required_decimals)
        
        assert is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
