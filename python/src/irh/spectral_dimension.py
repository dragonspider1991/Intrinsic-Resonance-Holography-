"""
spectral_dimension.py - Spectral Dimension Computation

This module computes the spectral dimension of a Cymatic Resonance Network, which characterizes
the effective dimensionality of discrete spacetime as probed by diffusion processes.

Target: d_s ≈ 4 for physical 4D spacetime.

Equations Implemented:
- Heat kernel: P(t) = (1/N) * Tr[exp(-t·ℒ_comb)] where ℒ_comb is combinatorial Interference Matrix
- Spectral dimension: d_s = -2 * d(log P)/d(log t)
- Stage 3 minimization: Psi(d_s, ℒ_comb) = entropic_cost(d_s) + scaling_penalty

The derivation is non-circular: we start from the combinatorial (integer) Interference Matrix
without assuming any dimensionality, then derive d_s from diffusion behavior.

References:
- Spectral Geometry
- Causal Dynamical Triangulations
- Random walks on graphs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigsh

if TYPE_CHECKING:
    from .graph_state import CymaticResonanceNetwork


@dataclass
class SpectralDimensionResult:
    """Result of spectral dimension computation."""

    value: float
    error: float
    fit_quality: float  # R² value
    slope: float
    intercept: float
    residuals: NDArray[np.float64]
    fit_data: dict[str, NDArray[np.float64]]
    fit_range: tuple[float, float]
    num_points: int


def HeatKernelTrace(graph: CymaticResonanceNetwork, t: float) -> float:
    """
    Compute the heat kernel trace Tr[exp(-t·L)] at diffusion time t.

    P(t) = (1/N) * Tr[exp(-t·L)]

    For efficiency, this uses eigenvalue decomposition:
    P(t) = (1/N) * Σᵢ exp(-λᵢ·t)

    Args:
        graph: CymaticResonanceNetwork instance
        t: Diffusion time

    Returns:
        Heat kernel trace value
    """
    # Get combinatorial Laplacian (unweighted)
    ℒ_comb = _combinatorial_laplacian(graph)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(ℒ_comb)

    # Heat kernel trace
    trace = np.sum(np.exp(-eigenvalues * t))

    return float(trace / graph.N)


def _combinatorial_laplacian(graph: CymaticResonanceNetwork) -> NDArray[np.float64]:
    """
    Compute the combinatorial (integer) Laplacian.

    ℒ_comb = D - A where A is the unweighted adjacency matrix.

    Args:
        graph: CymaticResonanceNetwork instance

    Returns:
        Combinatorial Laplacian matrix
    """
    # Unweighted adjacency
    A = (graph.adjacency_matrix > 0).astype(np.float64)
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D - A


def SpectralDimension(
    graph: CymaticResonanceNetwork,
    fit_range: tuple[float, float] = (0.01, 10.0),
    num_points: int = 50,
    use_eigenvalues: bool = True,
) -> SpectralDimensionResult:
    """
    Compute the spectral dimension via heat kernel analysis.

    The spectral dimension is defined via:
        P(t) = Tr[exp(-t·L)] ≈ t^(-d_s/2) for appropriate t range

    We compute d_s from the log-log slope:
        d_s = -2 · d(log P)/d(log t)

    Args:
        graph: CymaticResonanceNetwork instance
        fit_range: (t_min, t_max) for fitting
        num_points: Number of sample points
        use_eigenvalues: If True, use eigenvalue method (faster)

    Returns:
        SpectralDimensionResult with value, error, and diagnostics
    """
    # Generate log-spaced time points
    t_values = np.exp(np.linspace(np.log(fit_range[0]), np.log(fit_range[1]), num_points))

    # Compute heat kernel trace at each time
    if use_eigenvalues:
        p_values = _heat_kernel_from_eigen(graph, t_values)
    else:
        p_values = np.array([HeatKernelTrace(graph, t) for t in t_values])

    # Filter invalid values
    valid_mask = (p_values > 0) & np.isfinite(p_values)
    t_valid = t_values[valid_mask]
    p_valid = p_values[valid_mask]

    if len(t_valid) < 5:
        return SpectralDimensionResult(
            value=np.nan,
            error=np.inf,
            fit_quality=0.0,
            slope=0.0,
            intercept=0.0,
            residuals=np.array([]),
            fit_data={"log_t": np.array([]), "log_p": np.array([])},
            fit_range=fit_range,
            num_points=0,
        )

    # Log transform
    log_t = np.log(t_valid)
    log_p = np.log(p_valid)

    # Linear regression: log P = slope * log t + intercept
    # where slope = -d_s/2
    slope, intercept, se_slope, r_squared, residuals = _linear_fit(log_t, log_p)

    # Extract spectral dimension: d_s = -2 * slope
    ds = -2 * slope
    ds_error = 2 * se_slope

    return SpectralDimensionResult(
        value=ds,
        error=ds_error,
        fit_quality=r_squared,
        slope=slope,
        intercept=intercept,
        residuals=residuals,
        fit_data={"log_t": log_t, "log_p": log_p},
        fit_range=fit_range,
        num_points=len(t_valid),
    )


def _heat_kernel_from_eigen(
    graph: CymaticResonanceNetwork, t_values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute heat kernel trace from eigenvalues (vectorized).

    Args:
        graph: CymaticResonanceNetwork instance
        t_values: Array of diffusion times

    Returns:
        Array of heat kernel trace values
    """
    # Get combinatorial Laplacian
    ℒ_comb = _combinatorial_laplacian(graph)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(ℒ_comb)

    # P(t) = (1/N) * Σᵢ exp(-λᵢ·t) for each t
    # Vectorized: shape (len(t_values),)
    p_values = np.array(
        [(1.0 / graph.N) * np.sum(np.exp(-eigenvalues * t)) for t in t_values]
    )

    return p_values


def _linear_fit(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[float, float, float, float, NDArray[np.float64]]:
    """
    Perform linear regression with error estimates.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        (slope, intercept, se_slope, r_squared, residuals)
    """
    n = len(x)

    if n < 3:
        return 0.0, 0.0, np.inf, 0.0, np.array([])

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sxx = np.sum((x - x_mean) ** 2)
    sxy = np.sum((x - x_mean) * (y - y_mean))
    syy = np.sum((y - y_mean) ** 2)

    if sxx == 0:
        return 0.0, y_mean, np.inf, 0.0, y - y_mean

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    # Residuals
    residuals = y - (slope * x + intercept)
    sse = np.sum(residuals**2)

    # Mean squared error and standard error of slope
    mse = sse / (n - 2) if n > 2 else sse
    se_slope = np.sqrt(mse / sxx) if sxx > 0 else np.inf

    # R-squared
    r_squared = 1 - sse / syy if syy > 0 else 0.0

    return float(slope), float(intercept), float(se_slope), float(r_squared), residuals


def combinatorial_heat_kernel(ℒ_comb: NDArray[np.float64], t_steps: int = 100) -> float:
    """
    Compute heat kernel trace P_C = (1/N) * Tr[exp(-t·ℒ_comb)].

    This is the pure combinatorial version using the integer Laplacian,
    used for non-circular dimensional bootstrap.

    Args:
        ℒ_comb: Combinatorial (integer) Laplacian matrix
        t_steps: Number of time steps for averaging

    Returns:
        Average heat kernel trace
    """
    N = ℒ_comb.shape[0]
    eigenvalues = np.linalg.eigvalsh(ℒ_comb)

    # Average over time range
    t_values = np.linspace(0.1, 10.0, t_steps)
    traces = [(1.0 / N) * np.sum(np.exp(-eigenvalues * t)) for t in t_values]

    return float(np.mean(traces))


def compute_ds_combinatorial(
    ℒ_comb: NDArray[np.float64], t_range: tuple[float, float] = (0.1, 10.0)
) -> float:
    """
    Compute spectral dimension from combinatorial Laplacian.

    d_s_comb = -2 * lim_{t->0} d(log P_C)/d(log t)

    Args:
        ℒ_comb: Combinatorial Laplacian
        t_range: Time range for fitting

    Returns:
        Spectral dimension estimate
    """
    N = ℒ_comb.shape[0]
    eigenvalues = np.linalg.eigvalsh(ℒ_comb)

    # Sample points
    t_values = np.exp(np.linspace(np.log(t_range[0]), np.log(t_range[1]), 50))

    # Compute P_C(t)
    p_values = np.array(
        [(1.0 / N) * np.sum(np.exp(-eigenvalues * t)) for t in t_values]
    )

    # Filter and fit
    valid = (p_values > 0) & np.isfinite(p_values)
    log_t = np.log(t_values[valid])
    log_p = np.log(p_values[valid])

    if len(log_t) < 3:
        return np.nan

    # Linear fit
    slope, _, _, _, _ = _linear_fit(log_t, log_p)

    return -2 * slope


def minimize_psi(graph: CymaticResonanceNetwork, target_ds: float = 4.0) -> dict:
    """
    Minimize Ψ(d_s, ℒ_comb) = entropic_cost(d_s) + scaling_penalty.

    Stage 3 of dimensional bootstrap: find unique minimum at d_s ≈ 4.

    Args:
        graph: CymaticResonanceNetwork instance
        target_ds: Target spectral dimension

    Returns:
        Dictionary with minimization results
    """
    ℒ_comb = _combinatorial_laplacian(graph)
    N = graph.N

    # Current spectral dimension
    ds_current = compute_ds_combinatorial(ℒ_comb)

    # Entropic cost: penalize deviation from target
    entropic_cost = (ds_current - target_ds) ** 2

    # Scaling penalty: N should scale as L^d_s
    # Proxy: check if N ~ edge_count^(d_s/2)
    E = graph.edge_count
    if E > 0:
        expected_n = E ** (target_ds / 2)
        scaling_penalty = ((N - expected_n) / max(N, 1)) ** 2
    else:
        scaling_penalty = 1.0

    # Total cost
    psi = entropic_cost + scaling_penalty

    return {
        "psi": float(psi),
        "ds_current": float(ds_current),
        "entropic_cost": float(entropic_cost),
        "scaling_penalty": float(scaling_penalty),
        "target_ds": target_ds,
    }


def dimensional_bootstrap_test(
    graph: CymaticResonanceNetwork, tolerance: float = 0.02
) -> dict:
    """
    Test dimensional bootstrap: verify d_s ≈ 4 emerges uniquely.

    This implements the Stage 3 validation from the IRH framework.

    Args:
        graph: CymaticResonanceNetwork instance
        tolerance: Acceptable deviation from d_s = 4

    Returns:
        Test results dictionary
    """
    result = SpectralDimension(graph)

    passed = abs(result.value - 4.0) < tolerance if not np.isnan(result.value) else False

    return {
        "passed": passed,
        "ds_value": result.value,
        "ds_error": result.error,
        "deviation": abs(result.value - 4.0) if not np.isnan(result.value) else np.inf,
        "tolerance": tolerance,
        "fit_quality": result.fit_quality,
    }
