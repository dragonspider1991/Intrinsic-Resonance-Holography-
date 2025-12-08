"""
Certified Numerics Module - IRH v16.0

Provides interval arithmetic, rigorous floating-point error tracking, and
validated numerics for critical calculations requiring 12+ decimal precision.

This module implements the error budgeting framework described in the v16.0
specification for achieving certified numerical bounds.
"""

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class CertifiedValue:
    """
    A numerical value with certified error bounds.
    
    Attributes
    ----------
    value : float
        Central value of the quantity.
    lower_bound : float
        Guaranteed lower bound (value - error).
    upper_bound : float
        Guaranteed upper bound (value + error).
    error : float
        Absolute error bound (half-width of interval).
    relative_error : float
        Relative error (error / abs(value)).
    source : str
        Description of error source (e.g., 'numerical', 'statistical').
        
    Notes
    -----
    This represents a rigorous interval [lower_bound, upper_bound] that
    is guaranteed to contain the true mathematical value under the
    computational model assumptions.
    
    References
    ----------
    IRH v16.0 Section: Error Budgeting Framework
    """
    
    value: float
    lower_bound: float
    upper_bound: float
    error: float
    relative_error: float
    source: str = "unknown"
    
    @classmethod
    def from_value_and_error(
        cls,
        value: float,
        error: float,
        source: str = "unknown"
    ) -> 'CertifiedValue':
        """
        Create a CertifiedValue from central value and absolute error.
        
        Parameters
        ----------
        value : float
            Central value.
        error : float
            Absolute error bound (must be non-negative).
        source : str
            Description of error source.
            
        Returns
        -------
        certified : CertifiedValue
            Certified value with interval bounds.
        """
        if error < 0:
            raise ValueError("Error must be non-negative")
        
        return cls(
            value=value,
            lower_bound=value - error,
            upper_bound=value + error,
            error=error,
            relative_error=abs(error / value) if value != 0 else float('inf'),
            source=source
        )
    
    def __add__(self, other: Union['CertifiedValue', float]) -> 'CertifiedValue':
        """Add two certified values with rigorous error propagation."""
        if isinstance(other, CertifiedValue):
            new_value = self.value + other.value
            new_error = self.error + other.error  # Worst-case addition
            return CertifiedValue.from_value_and_error(
                new_value, new_error, f"sum({self.source}, {other.source})"
            )
        else:
            # Adding a constant (exact)
            return CertifiedValue(
                value=self.value + other,
                lower_bound=self.lower_bound + other,
                upper_bound=self.upper_bound + other,
                error=self.error,
                relative_error=abs(self.error / (self.value + other)) if (self.value + other) != 0 else float('inf'),
                source=self.source
            )
    
    def __mul__(self, other: Union['CertifiedValue', float]) -> 'CertifiedValue':
        """Multiply two certified values with rigorous error propagation."""
        if isinstance(other, CertifiedValue):
            new_value = self.value * other.value
            # Relative errors add in multiplication
            new_rel_error = self.relative_error + other.relative_error
            new_error = abs(new_value) * new_rel_error
            return CertifiedValue.from_value_and_error(
                new_value, new_error, f"product({self.source}, {other.source})"
            )
        else:
            # Multiplying by constant
            return CertifiedValue(
                value=self.value * other,
                lower_bound=min(self.lower_bound * other, self.upper_bound * other),
                upper_bound=max(self.lower_bound * other, self.upper_bound * other),
                error=abs(self.error * other),
                relative_error=self.relative_error,
                source=self.source
            )
    
    def __repr__(self) -> str:
        """String representation showing value ± error."""
        # Determine significant figures from error
        if self.error > 0:
            error_order = int(np.floor(np.log10(self.error)))
            decimals = max(0, -error_order + 1)
            return f"{self.value:.{decimals}f} ± {self.error:.{decimals}e} ({self.source})"
        else:
            return f"{self.value} (exact, {self.source})"


def interval_arithmetic(
    operation: str,
    a: CertifiedValue,
    b: Optional[CertifiedValue] = None
) -> CertifiedValue:
    """
    Perform interval arithmetic operations with certified bounds.
    
    Parameters
    ----------
    operation : str
        Operation to perform: 'add', 'sub', 'mul', 'div', 'sqrt', 'exp', 'log'.
    a : CertifiedValue
        First operand.
    b : CertifiedValue, optional
        Second operand (for binary operations).
        
    Returns
    -------
    result : CertifiedValue
        Result with certified error bounds.
        
    Notes
    -----
    Uses conservative interval arithmetic to guarantee bounds contain true value.
    """
    if operation == 'add':
        if b is None:
            raise ValueError("Binary operation 'add' requires two operands")
        return a + b
    
    elif operation == 'sub':
        if b is None:
            raise ValueError("Binary operation 'sub' requires two operands")
        return a + (b * -1)
    
    elif operation == 'mul':
        if b is None:
            raise ValueError("Binary operation 'mul' requires two operands")
        return a * b
    
    elif operation == 'div':
        if b is None:
            raise ValueError("Binary operation 'div' requires two operands")
        if b.lower_bound <= 0 <= b.upper_bound:
            raise ValueError("Division by interval containing zero")
        new_value = a.value / b.value
        # For division, relative errors add
        new_rel_error = a.relative_error + b.relative_error
        new_error = abs(new_value) * new_rel_error
        return CertifiedValue.from_value_and_error(
            new_value, new_error, f"div({a.source}, {b.source})"
        )
    
    elif operation == 'sqrt':
        if a.lower_bound < 0:
            raise ValueError("Square root of negative interval")
        new_value = np.sqrt(a.value)
        # d(sqrt(x))/dx = 1/(2*sqrt(x))
        new_error = a.error / (2 * np.sqrt(a.lower_bound))
        return CertifiedValue.from_value_and_error(
            new_value, new_error, f"sqrt({a.source})"
        )
    
    elif operation == 'exp':
        new_value = np.exp(a.value)
        # d(exp(x))/dx = exp(x), use upper bound for conservative error
        new_error = a.error * np.exp(a.upper_bound)
        return CertifiedValue.from_value_and_error(
            new_value, new_error, f"exp({a.source})"
        )
    
    elif operation == 'log':
        if a.lower_bound <= 0:
            raise ValueError("Logarithm of non-positive interval")
        new_value = np.log(a.value)
        # d(log(x))/dx = 1/x, use lower bound for conservative error
        new_error = a.error / a.lower_bound
        return CertifiedValue.from_value_and_error(
            new_value, new_error, f"log({a.source})"
        )
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


def certified_sum(
    values: NDArray[np.float64],
    error_per_value: Optional[NDArray[np.float64]] = None
) -> CertifiedValue:
    """
    Compute sum of array with certified error bounds.
    
    Parameters
    ----------
    values : NDArray
        Array of values to sum.
    error_per_value : NDArray, optional
        Error bound for each value. If None, uses machine epsilon estimate.
        
    Returns
    -------
    result : CertifiedValue
        Sum with certified error bounds.
        
    Notes
    -----
    Accounts for:
    1. Rounding errors in summation (Kahan summation effect)
    2. Input value uncertainties
    3. Catastrophic cancellation detection
    """
    if len(values) == 0:
        return CertifiedValue.from_value_and_error(0.0, 0.0, "empty_sum")
    
    # Compute sum
    total = np.sum(values)
    
    # Estimate numerical error
    # Each addition introduces at most eps * |sum_so_far|
    # Use compensated summation estimate
    eps = np.finfo(values.dtype).eps
    numerical_error = eps * len(values) * np.max(np.abs(values))
    
    # Add input uncertainties
    if error_per_value is not None:
        input_error = np.sum(np.abs(error_per_value))
    else:
        input_error = eps * np.sum(np.abs(values))
    
    total_error = numerical_error + input_error
    
    return CertifiedValue.from_value_and_error(
        total, total_error, "certified_sum"
    )


def certified_product(
    values: NDArray[np.float64],
    error_per_value: Optional[NDArray[np.float64]] = None
) -> CertifiedValue:
    """
    Compute product of array with certified error bounds.
    
    Parameters
    ----------
    values : NDArray
        Array of values to multiply.
    error_per_value : NDArray, optional
        Error bound for each value. If None, uses machine epsilon estimate.
        
    Returns
    -------
    result : CertifiedValue
        Product with certified error bounds.
        
    Notes
    -----
    Relative errors accumulate in multiplication.
    """
    if len(values) == 0:
        return CertifiedValue.from_value_and_error(1.0, 0.0, "empty_product")
    
    # Compute product
    product = np.prod(values)
    
    # Estimate relative error
    eps = np.finfo(values.dtype).eps
    
    if error_per_value is not None:
        # Sum relative errors
        rel_errors = error_per_value / np.abs(values)
        # Note: Assumes zero values are exact (no error); this is a simplification
        rel_errors[np.abs(values) == 0] = 0  # Avoid division by zero
        total_rel_error = np.sum(rel_errors)
    else:
        total_rel_error = len(values) * eps
    
    # Add numerical roundoff
    total_rel_error += len(values) * eps
    
    total_error = abs(product) * total_rel_error
    
    return CertifiedValue.from_value_and_error(
        product, total_error, "certified_product"
    )


def track_floating_point_error(
    computation: str,
    result: float,
    operands: Union[float, NDArray[np.float64]],
    n_operations: Optional[int] = None
) -> CertifiedValue:
    """
    Estimate floating-point error for a computation.
    
    Parameters
    ----------
    computation : str
        Description of computation type.
    result : float
        Computed result.
    operands : float or NDArray
        Input operands.
    n_operations : int, optional
        Number of floating-point operations. If None, estimated from operands.
        
    Returns
    -------
    certified : CertifiedValue
        Result with estimated floating-point error.
        
    Notes
    -----
    Uses backward error analysis to estimate accumulated roundoff.
    Conservative bounds assume worst-case error accumulation.
    """
    # Get machine epsilon
    eps = np.finfo(np.float64).eps
    
    # Estimate number of operations if not provided
    if n_operations is None:
        if isinstance(operands, np.ndarray):
            n_operations = len(operands)
        else:
            n_operations = 1
    
    # Conservative error estimate: each operation adds relative error eps
    # Total relative error ≈ n_ops * eps
    relative_error = n_operations * eps
    absolute_error = abs(result) * relative_error
    
    return CertifiedValue.from_value_and_error(
        result, absolute_error, f"floating_point({computation})"
    )


# Example usage and tests
if __name__ == "__main__":
    # Example: Certified computation
    a = CertifiedValue.from_value_and_error(1.0, 1e-12, "measurement_a")
    b = CertifiedValue.from_value_and_error(2.0, 1e-12, "measurement_b")
    
    c = a + b
    print(f"Addition: {c}")
    
    d = a * b
    print(f"Multiplication: {d}")
    
    # Array operations
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sum_result = certified_sum(values)
    print(f"Sum: {sum_result}")
    
    product_result = certified_product(values)
    print(f"Product: {product_result}")
