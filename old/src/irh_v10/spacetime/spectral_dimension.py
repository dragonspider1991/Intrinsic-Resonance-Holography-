"""
Spectral Dimension - Heat kernel analysis for dimensional emergence

Computes the spectral dimension d_s from heat kernel trace:
    K(t) = Tr(exp(-tℒ)) ~ t^(-d_s/2)

For 4D spacetime, d_s → 4 under ARO optimization.

Reference: IRH v10.0 manuscript, Section IV.A "Dimensional Bootstrap"
"""

import numpy as np
from typing import Tuple, Optional
import scipy.sparse as sp


def compute_heat_kernel_trace(
    eigenvalues: np.ndarray,
    t: float,
) -> float:
    """
    Compute heat kernel trace K(t) = Tr(exp(-tℒ)).
    
    Args:
        eigenvalues: Eigenvalues of Interference Matrix ℒ
        t: Time parameter
    
    Returns:
        K_t: Heat kernel trace
    
    Example:
        >>> eigenvalues = network.compute_spectrum()
        >>> K_t = compute_heat_kernel_trace(eigenvalues, t=0.1)
    """
    # K(t) = Σ_i exp(-t λ_i)
    K_t = np.sum(np.exp(-t * eigenvalues))
    return K_t


def estimate_spectral_dimension(
    eigenvalues: np.ndarray,
    t_min: float = 0.01,
    t_max: float = 1.0,
    n_points: int = 20,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate spectral dimension from heat kernel scaling.
    
    Fits K(t) ~ t^(-d_s/2) to extract d_s.
    
    Args:
        eigenvalues: Eigenvalues of ℒ
        t_min: Minimum time for fitting
        t_max: Maximum time for fitting
        n_points: Number of time points
    
    Returns:
        d_s: Spectral dimension
        t_values: Time values used
        K_values: Heat kernel values
    
    Example:
        >>> d_s, t_vals, K_vals = estimate_spectral_dimension(eigenvalues)
        >>> print(f"Spectral dimension: {d_s:.3f}")
    """
    # Generate time values (log-spaced)
    t_values = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
    
    # Compute heat kernel for each time
    K_values = np.array([compute_heat_kernel_trace(eigenvalues, t) for t in t_values])
    
    # Fit log(K) = (-d_s/2) * log(t) + const
    log_t = np.log(t_values)
    log_K = np.log(K_values)
    
    # Linear regression
    coeffs = np.polyfit(log_t, log_K, 1)
    slope = coeffs[0]
    
    # Extract d_s
    d_s = -2 * slope
    
    return d_s, t_values, K_values


def spectral_dimension_evolution(
    eigenvalues_list: list,
    labels: list,
    t_min: float = 0.01,
    t_max: float = 1.0,
) -> dict:
    """
    Track spectral dimension evolution across multiple networks.
    
    Useful for monitoring ARO convergence toward d_s = 4.
    
    Args:
        eigenvalues_list: List of eigenvalue arrays
        labels: Labels for each network state
        t_min: Minimum time
        t_max: Maximum time
    
    Returns:
        results: Dictionary with d_s values and fit parameters
    """
    results = {
        'd_s_values': [],
        't_values': None,
        'K_values_list': [],
        'labels': labels,
    }
    
    for eigenvalues in eigenvalues_list:
        d_s, t_vals, K_vals = estimate_spectral_dimension(
            eigenvalues, t_min=t_min, t_max=t_max
        )
        results['d_s_values'].append(d_s)
        results['K_values_list'].append(K_vals)
        
        if results['t_values'] is None:
            results['t_values'] = t_vals
    
    return results


def growth_dimension_estimate(
    adjacency: np.ndarray | sp.spmatrix,
    max_distance: int = 10,
) -> float:
    """
    Estimate growth dimension from volume scaling.
    
    For d-dimensional space: V(r) ~ r^d
    
    Args:
        adjacency: Adjacency or coupling matrix
        max_distance: Maximum graph distance to probe
    
    Returns:
        d_g: Growth dimension
    
    Notes:
        This is computationally expensive for large graphs.
        Requires BFS from multiple seed nodes.
    """
    import networkx as nx
    
    # Convert to NetworkX graph
    if sp.issparse(adjacency):
        G = nx.from_scipy_sparse_array(adjacency)
    else:
        G = nx.from_numpy_array(adjacency)
    
    # Sample a few seed nodes
    N = len(adjacency)
    n_seeds = min(10, N)
    seed_nodes = np.random.choice(N, n_seeds, replace=False)
    
    # Compute average volume at each distance
    distances = range(1, max_distance + 1)
    volumes = []
    
    for r in distances:
        vol_sum = 0
        for seed in seed_nodes:
            # Find all nodes at distance <= r
            lengths = nx.single_source_shortest_path_length(G, seed, cutoff=r)
            vol_sum += len(lengths)
        volumes.append(vol_sum / n_seeds)
    
    # Fit V(r) ~ r^d_g
    log_r = np.log(np.array(distances))
    log_V = np.log(np.array(volumes))
    
    coeffs = np.polyfit(log_r, log_V, 1)
    d_g = coeffs[0]
    
    return d_g


def verify_4d_emergence(
    eigenvalues: np.ndarray,
    tolerance: float = 0.1,
) -> bool:
    """
    Verify that spectral dimension is close to 4.
    
    Args:
        eigenvalues: Eigenvalues of ℒ
        tolerance: Acceptable deviation from 4.0
    
    Returns:
        is_4d: True if |d_s - 4| < tolerance
    """
    d_s, _, _ = estimate_spectral_dimension(eigenvalues)
    
    deviation = abs(d_s - 4.0)
    is_4d = deviation < tolerance
    
    return is_4d
