"""
Numerics Module for IRH v16.0

This module provides certified numerical methods for achieving 12+ decimal
place precision in fundamental constant calculations.
"""

from .certified_numerics import (
    CertifiedValue,
    interval_arithmetic,
    certified_sum,
    certified_product,
    track_floating_point_error,
)

from .precision_constants import (
    C_H_CERTIFIED,
    EPSILON_THRESHOLD_CERTIFIED,
    C_RESIDUAL_CERTIFIED,
    Q_HOLONOMIC_CERTIFIED,
    PRECISION_TARGET,
    NUMERICAL_SETTINGS,
    ERROR_BUDGET_ALPHA,
    CODATA_2022,
    OBSERVATIONAL_DATA,
    FALSIFIABILITY_THRESHOLDS,
    get_precision_target,
    validate_precision,
)

from .error_tracking import (
    ErrorBudget,
    create_error_budget,
    combine_errors,
    estimate_numerical_error,
    estimate_finite_size_error,
    estimate_statistical_error,
    check_error_budget_compliance,
)

__all__ = [
    # Certified numerics
    'CertifiedValue',
    'interval_arithmetic',
    'certified_sum',
    'certified_product',
    'track_floating_point_error',
    
    # Constants
    'C_H_CERTIFIED',
    'EPSILON_THRESHOLD_CERTIFIED',
    'C_RESIDUAL_CERTIFIED',
    'Q_HOLONOMIC_CERTIFIED',
    'PRECISION_TARGET',
    'NUMERICAL_SETTINGS',
    'ERROR_BUDGET_ALPHA',
    'CODATA_2022',
    'OBSERVATIONAL_DATA',
    'FALSIFIABILITY_THRESHOLDS',
    'get_precision_target',
    'validate_precision',
    
    # Error tracking
    'ErrorBudget',
    'create_error_budget',
    'combine_errors',
    'estimate_numerical_error',
    'estimate_finite_size_error',
    'estimate_statistical_error',
    'check_error_budget_compliance',
]

__version__ = "16.0.0-alpha"
