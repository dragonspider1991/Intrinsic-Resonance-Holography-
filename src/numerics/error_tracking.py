"""
Error Tracking Module - IRH v16.0

Provides real-time error budgeting and propagation tracking across all
computational modules for certified numerical precision.

This module implements the error classification and budgeting framework:
- Numerical errors (floating-point roundoff, truncation)
- Statistical errors (finite sampling, ensemble averaging)
- Finite-size errors (O(1/√N) convergence terms)
- Theoretical errors (model approximations, higher-order terms)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class ErrorBudget:
    """
    Comprehensive error budget for a physical quantity calculation.
    
    Attributes
    ----------
    numerical_error : float
        Floating-point roundoff and truncation errors.
    statistical_error : float
        Errors from finite sampling and ensemble averaging.
    finite_size_error : float
        O(1/√N) and higher-order convergence terms.
    theoretical_error : float
        Model approximations and higher-order corrections.
    metadata : dict
        Additional information about error sources.
        
    Notes
    -----
    Total error is combined in quadrature assuming independence:
    σ_total = √(σ_num² + σ_stat² + σ_fss² + σ_theory²)
    
    References
    ----------
    IRH v16.0 Section: Error Budgeting Framework
    """
    
    numerical_error: float = 0.0
    statistical_error: float = 0.0
    finite_size_error: float = 0.0
    theoretical_error: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def total_error(self) -> float:
        """
        Compute total error combining all sources in quadrature.
        
        Returns
        -------
        total : float
            Combined error (assumes uncorrelated sources).
        """
        return np.sqrt(
            self.numerical_error**2 +
            self.statistical_error**2 +
            self.finite_size_error**2 +
            self.theoretical_error**2
        )
    
    def relative_error(self, value: float) -> float:
        """
        Compute relative error fraction.
        
        Parameters
        ----------
        value : float
            Central value of quantity.
            
        Returns
        -------
        rel_error : float
            Total error divided by absolute value.
        """
        if value == 0:
            return float('inf')
        return self.total_error() / abs(value)
    
    def dominant_error_source(self) -> str:
        """
        Identify the dominant source of error.
        
        Returns
        -------
        source : str
            Name of dominant error source.
        """
        errors = {
            'numerical': self.numerical_error,
            'statistical': self.statistical_error,
            'finite_size': self.finite_size_error,
            'theoretical': self.theoretical_error,
        }
        return max(errors, key=errors.get)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'numerical_error': self.numerical_error,
            'statistical_error': self.statistical_error,
            'finite_size_error': self.finite_size_error,
            'theoretical_error': self.theoretical_error,
            'total_error': self.total_error(),
            'dominant_source': self.dominant_error_source(),
            'metadata': self.metadata,
        }
    
    def __repr__(self) -> str:
        """String representation of error budget."""
        total = self.total_error()
        dominant = self.dominant_error_source()
        return (
            f"ErrorBudget(total={total:.3e}, dominant={dominant}, "
            f"num={self.numerical_error:.3e}, "
            f"stat={self.statistical_error:.3e}, "
            f"fss={self.finite_size_error:.3e}, "
            f"theory={self.theoretical_error:.3e})"
        )


def create_error_budget(
    N: Optional[int] = None,
    n_samples: Optional[int] = None,
    n_operations: Optional[int] = None,
    higher_order_estimate: Optional[float] = None
) -> ErrorBudget:
    """
    Create an error budget with automatic estimates.
    
    Parameters
    ----------
    N : int, optional
        Network size for finite-size error estimate.
    n_samples : int, optional
        Number of ensemble samples for statistical error.
    n_operations : int, optional
        Number of floating-point operations for numerical error.
    higher_order_estimate : float, optional
        Estimate of theoretical higher-order corrections.
        
    Returns
    -------
    budget : ErrorBudget
        Error budget with estimated components.
        
    Notes
    -----
    Estimates use conservative bounds:
    - Numerical: O(eps * n_operations)
    - Statistical: O(1/√n_samples)
    - Finite-size: O(1/√N)
    - Theoretical: user-provided or 0
    """
    budget = ErrorBudget()
    
    # Numerical error estimate
    if n_operations is not None:
        eps = np.finfo(np.float64).eps
        budget.numerical_error = eps * n_operations
        budget.metadata['n_operations'] = n_operations
    
    # Statistical error estimate
    if n_samples is not None:
        budget.statistical_error = 1.0 / np.sqrt(n_samples)
        budget.metadata['n_samples'] = n_samples
    
    # Finite-size error estimate
    if N is not None:
        budget.finite_size_error = 1.0 / np.sqrt(N)
        budget.metadata['N'] = N
    
    # Theoretical error
    if higher_order_estimate is not None:
        budget.theoretical_error = higher_order_estimate
    
    return budget


def combine_errors(budgets: List[ErrorBudget]) -> ErrorBudget:
    """
    Combine multiple error budgets.
    
    Parameters
    ----------
    budgets : List[ErrorBudget]
        List of error budgets to combine.
        
    Returns
    -------
    combined : ErrorBudget
        Combined error budget (errors added in quadrature).
        
    Notes
    -----
    Assumes error sources are uncorrelated. For correlated errors,
    use worst-case addition instead.
    """
    if not budgets:
        return ErrorBudget()
    
    # Combine in quadrature for each category
    combined = ErrorBudget(
        numerical_error=np.sqrt(sum(b.numerical_error**2 for b in budgets)),
        statistical_error=np.sqrt(sum(b.statistical_error**2 for b in budgets)),
        finite_size_error=np.sqrt(sum(b.finite_size_error**2 for b in budgets)),
        theoretical_error=np.sqrt(sum(b.theoretical_error**2 for b in budgets)),
    )
    
    # Merge metadata
    combined.metadata['n_budgets_combined'] = len(budgets)
    combined.metadata['sources'] = [b.metadata.get('source', 'unknown') for b in budgets]
    
    return combined


def estimate_numerical_error(
    result: float,
    operands: NDArray[np.float64],
    operation_type: str = "general"
) -> float:
    """
    Estimate numerical roundoff error for a computation.
    
    Parameters
    ----------
    result : float
        Computed result.
    operands : NDArray
        Input operands used in computation.
    operation_type : str
        Type of operation: 'sum', 'product', 'matmul', 'eigen', 'general'.
        
    Returns
    -------
    error : float
        Estimated absolute numerical error.
        
    Notes
    -----
    Uses backward error analysis principles for conservative estimates.
    """
    eps = np.finfo(np.float64).eps
    n = len(operands)
    
    if operation_type == "sum":
        # Each addition: eps * |running_sum|
        error = eps * n * np.max(np.abs(operands))
    
    elif operation_type == "product":
        # Relative errors accumulate
        error = abs(result) * n * eps
    
    elif operation_type == "matmul":
        # Matrix multiplication: n^3 operations for n×n matrices
        # Conservative: each element has n multiplications + n-1 additions
        error = abs(result) * (2 * n - 1) * eps
    
    elif operation_type == "eigen":
        # Eigenvalue computation: O(n^3) complexity
        # Wilkinson's backward error bound
        error = abs(result) * (n**3) * eps
    
    else:  # general
        error = abs(result) * n * eps
    
    return error


def estimate_finite_size_error(
    N: int,
    observable_value: float,
    correction_order: float = 0.5
) -> float:
    """
    Estimate finite-size scaling error.
    
    Parameters
    ----------
    N : int
        System size (number of nodes/states).
    observable_value : float
        Value of observable at finite N.
    correction_order : float
        Leading correction exponent (default: 0.5 for O(1/√N)).
        
    Returns
    -------
    error : float
        Estimated finite-size correction.
        
    Notes
    -----
    For observables with O(N^(-ω)) corrections:
    σ_fss ≈ |observable| × N^(-ω)
    
    Default assumes O(1/√N) scaling common in statistical mechanics.
    """
    if N <= 1:
        return abs(observable_value)  # Meaningless for N=1
    
    return abs(observable_value) * (N ** (-correction_order))


def estimate_statistical_error(
    samples: NDArray[np.float64],
    method: str = "standard"
) -> float:
    """
    Estimate statistical error from ensemble samples.
    
    Parameters
    ----------
    samples : NDArray
        Array of sample values.
    method : str
        Method for error estimation:
        - 'standard': Standard error of mean
        - 'bootstrap': Bootstrap resampling (not implemented)
        - 'jackknife': Jackknife resampling (not implemented)
        
    Returns
    -------
    error : float
        Estimated statistical error.
    """
    if len(samples) <= 1:
        return 0.0
    
    if method == "standard":
        # Standard error of the mean
        return np.std(samples, ddof=1) / np.sqrt(len(samples))
    
    else:
        raise NotImplementedError(f"Method '{method}' not yet implemented")


def check_error_budget_compliance(
    budget: ErrorBudget,
    target_error: float,
    strict: bool = True
) -> bool:
    """
    Check if error budget meets target requirements.
    
    Parameters
    ----------
    budget : ErrorBudget
        Error budget to check.
    target_error : float
        Maximum allowable total error.
    strict : bool
        If True, requires all individual sources to be within target.
        If False, only checks total error.
        
    Returns
    -------
    compliant : bool
        True if error budget meets requirements.
    """
    if strict:
        # All sources must be within target
        return (
            budget.numerical_error <= target_error and
            budget.statistical_error <= target_error and
            budget.finite_size_error <= target_error and
            budget.theoretical_error <= target_error
        )
    else:
        # Only total error matters
        return budget.total_error() <= target_error


# Example usage and testing
if __name__ == "__main__":
    print("Error Budget Example")
    print("=" * 60)
    
    # Create error budget for a calculation
    budget = create_error_budget(
        N=10000,
        n_samples=1000,
        n_operations=50000,
        higher_order_estimate=1e-6
    )
    
    print(f"Error Budget: {budget}")
    print(f"\nTotal Error: {budget.total_error():.3e}")
    print(f"Dominant Source: {budget.dominant_error_source()}")
    
    # Check compliance with target
    target = 1e-3
    compliant = check_error_budget_compliance(budget, target, strict=False)
    print(f"\nComplies with target ({target:.3e}): {compliant}")
    
    # Example: Combine multiple budgets
    budget1 = ErrorBudget(numerical_error=1e-6, statistical_error=1e-5)
    budget2 = ErrorBudget(numerical_error=2e-6, finite_size_error=1e-5)
    
    combined = combine_errors([budget1, budget2])
    print(f"\nCombined Budget: {combined}")
