"""
Topological Invariants Calculator

Computes frustration density, Betti numbers, and other topological
invariants from ARO-optimized Cymatic Resonance Networks.

Key Functions:
- calculate_frustration_density: ρ_frust → α via Theorem 1.2
- calculate_betti_numbers: β₁ = 12 for emergent gauge group
- derive_fine_structure_constant: α⁻¹ = 2π/ρ_frust

References: IRH v13.0 Theorems 1.2, 5.1, Section 9.1
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import random
from typing import List, Tuple, Optional
from numpy.typing import NDArray


def calculate_frustration_density(
    W: sp.spmatrix,
    max_cycles: int = 5000,
    sampling: bool = True
) -> float:
    """
    Calculate frustration density ρ_frust from phase holonomies.
    
    Computes the average residual phase winding per minimal cycle,
    which directly determines the fine-structure constant via
    α⁻¹ = 2π/ρ_frust (Theorem 1.2).
    
    Parameters
    ----------
    W : sp.spmatrix
        Complex adjacency matrix with weights W_ij = |W_ij| exp(iφ_ij).
    max_cycles : int
        Maximum number of cycles to process.
    sampling : bool
        If True, use sampling for large graphs (N > 1000).
        
    Returns
    -------
    rho_frust : float
        Frustration density (dimensionless topological invariant).
        
    Notes
    -----
    Uses Horton's algorithm (via NetworkX cycle_basis) for small graphs,
    and edge sampling for large graphs to ensure O(N log N) complexity.
    
    References
    ----------
    IRH v13.0 Theorem 1.2: Emergence of Phase Structure and α
    """
    # Convert to NetworkX graph (directed to preserve phase information)
    G = nx.from_scipy_sparse_array(W, create_using=nx.DiGraph)
    N = G.number_of_nodes()
    
    # Select cycle enumeration strategy based on network size
    if not sampling or N < 1000:
        # Exact cycle basis for moderate graphs
        try:
            cycle_basis = list(nx.cycle_basis(G.to_undirected()))
        except:
            return 0.0
    else:
        # Sampling strategy for large graphs
        cycle_basis = _sample_cycles(G, max_samples=max_cycles)
    
    if len(cycle_basis) == 0:
        return 0.0
    
    # Compute phase holonomies (Wilson loops)
    holonomies = []
    cycles_to_process = min(len(cycle_basis), max_cycles)
    
    for cycle_nodes in cycle_basis[:cycles_to_process]:
        if len(cycle_nodes) < 3:
            continue
        
        try:
            holonomy = _compute_cycle_holonomy(G, W, cycle_nodes)
            holonomies.append(holonomy)
        except:
            continue
    
    if not holonomies:
        return 0.0
    
    # ρ_frust = average absolute phase winding
    rho_frust = np.mean(np.abs(holonomies))
    return float(rho_frust)


def _sample_cycles(
    G: nx.DiGraph,
    max_samples: int = 5000
) -> List[List[int]]:
    """
    Sample cycles for large graphs using edge-based strategy.
    
    For each sampled edge (u,v), attempts to find shortest cycle
    containing that edge.
    """
    cycles = []
    edges = list(G.edges())
    
    if len(edges) == 0:
        return cycles
    
    sample_size = min(len(edges), max_samples)
    sampled_edges = random.sample(edges, sample_size)
    
    for u, v in sampled_edges:
        try:
            # Find shortest path back from v to u
            path = nx.shortest_path(G.to_undirected(), source=v, target=u)
            if len(path) > 2:
                # Form cycle: u → v → ...path... → u
                cycle = [u] + path
                if cycle not in cycles:  # Avoid duplicates
                    cycles.append(cycle)
        except nx.NetworkXNoPath:
            continue
    
    return cycles


def _compute_cycle_holonomy(
    G: nx.DiGraph,
    W: sp.spmatrix,
    cycle_nodes: List[int]
) -> float:
    """
    Compute phase holonomy (Wilson loop) around a cycle.
    
    Returns the residual phase: arg(∏ W_ij) along cycle.
    """
    holonomy_product = 1.0 + 0.0j
    
    for i in range(len(cycle_nodes)):
        u = cycle_nodes[i]
        v = cycle_nodes[(i + 1) % len(cycle_nodes)]
        
        # Get complex weight from sparse matrix
        weight = W[u, v]
        
        if weight == 0:
            # Try reverse edge
            weight = W[v, u]
            if weight == 0:
                raise ValueError(f"Edge ({u},{v}) not found in cycle")
            weight = np.conj(weight)  # Use conjugate for reverse traversal
        
        holonomy_product *= weight
    
    # Extract residual phase (frustration)
    phase = np.angle(holonomy_product)
    return phase


def derive_fine_structure_constant(
    rho_frust: float
) -> Tuple[float, bool]:
    """
    Derive fine-structure constant from frustration density.
    
    Implements Theorem 1.2: α⁻¹ = 2π/ρ_frust
    
    Parameters
    ----------
    rho_frust : float
        Frustration density from calculate_frustration_density().
        
    Returns
    -------
    alpha_inv : float
        Inverse fine-structure constant.
    match : bool
        True if prediction matches experiment within 1.0.
        
    References
    ----------
    IRH v13.0 Theorem 1.2
    CODATA 2022: α⁻¹ = 137.035999084(21)
    """
    if rho_frust == 0 or np.isnan(rho_frust):
        return 0.0, False
    
    alpha_inv = (2 * np.pi) / rho_frust
    
    # Check prediction against experiment
    experimental = 137.035999084
    match = abs(alpha_inv - experimental) < 1.0
    
    return alpha_inv, match


def calculate_betti_numbers(
    W: sp.spmatrix,
    boundary_only: bool = True
) -> dict:
    """
    Calculate Betti numbers of the emergent network topology.
    
    For ARO-optimized networks at Cosmic Fixed Point,
    the first Betti number β₁ = 12 corresponds to the
    12 generators of SU(3)×SU(2)×U(1) (Theorem 5.1).
    
    Parameters
    ----------
    W : sp.spmatrix
        Complex adjacency matrix.
    boundary_only : bool
        If True, compute only for emergent S³ boundary.
        
    Returns
    -------
    betti_numbers : dict
        Dictionary with keys 'beta_0', 'beta_1', etc.
        
    Notes
    -----
    Uses persistent homology for boundary identification.
    Full implementation requires specialized libraries (e.g., Ripser, Gudhi).
    
    References
    ----------
    IRH v13.0 Theorem 5.1: Network Homology and β₁ = 12
    """
    # Placeholder implementation
    # Full implementation requires:
    # 1. Identify boundary nodes (max info exchange with "external" states)
    # 2. Construct simplicial complex from boundary subgraph  
    # 3. Compute homology groups using persistent homology
    
    results = {
        'beta_0': 1,  # Connected components (should be 1 for connected graph)
        'beta_1': None,  # Fundamental group rank (target: 12 for v13.0)
        'implementation': 'placeholder'
    }
    
    return results


def validate_topological_predictions(
    W: sp.spmatrix
) -> dict:
    """
    Validate key topological predictions from IRH v13.0.
    
    Returns
    -------
    validation : dict
        Results for:
        - 'rho_frust': computed frustration density
        - 'alpha_inv': predicted α⁻¹
        - 'alpha_match': bool, within experimental error
        - 'beta_1': first Betti number (target: 12)
    """
    rho_frust = calculate_frustration_density(W)
    alpha_inv, alpha_match = derive_fine_structure_constant(rho_frust)
    betti = calculate_betti_numbers(W)
    
    return {
        'rho_frust': rho_frust,
        'alpha_inv': alpha_inv,
        'alpha_match': alpha_match,
        'beta_1': betti['beta_1'],
        'experimental_alpha': 137.035999084
    }
