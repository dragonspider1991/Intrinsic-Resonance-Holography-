"""
Axiom 2: Network Emergence Principle - Cymatic Resonance Network

This module implements the Cymatic Resonance Network (CRN) as defined in
IRHv16.md §1 Axiom 2 (Network Emergence Principle).

Key Concepts (from IRHv16.md §1 Axiom 2):
    - CRN G = (V, E, W) is the unique minimal representation of AHS relationships
    - V = S (nodes are Algorithmic Holonomic States)
    - (s_i, s_j) ∈ E iff |W_ij| > ε_threshold
    - W_ij ∈ ℂ as defined in Axiom 1
    - ε_threshold = 0.730129 ± 10^{-6} (rigorously derived, not a free parameter)

Implementation Status: Phase 3 Implementation
    - CRN class: IMPLEMENTED
    - Network metrics: IMPLEMENTED
    - Holonomy computation: IMPLEMENTED (basic)
    - Frustration density: IMPLEMENTED

References:
    IRHv16.md §1 Axiom 2: Network Emergence Principle
    IRHv16.md Theorem 1.2: Necessity of Network Representation
    IRHv16.md §2 Definition 2.1: Frustration Density
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .ahs import AlgorithmicHolonomicState, create_ahs_network
from .acw import build_acw_matrix


# Universal threshold from IRHv16.md
EPSILON_THRESHOLD = 0.730129  # ± 10^{-6}
EPSILON_THRESHOLD_ERROR = 1e-6


@dataclass
class CymaticResonanceNetwork:
    """
    Cymatic Resonance Network (CRN) - the fundamental network structure.
    
    Per IRHv16.md §1 Axiom 2 (Network Emergence Principle):
        "Any system of Algorithmic Holonomic States satisfying Axiom 1 can be
        represented uniquely and minimally as a complex-weighted, directed
        Cymatic Resonance Network (CRN) G = (V, E, W)"
    
    Attributes:
        states: List of AHS nodes (V = S)
        W: Complex-valued ACW matrix W_ij
        epsilon_threshold: Edge inclusion threshold
        
    Properties:
        N: Number of nodes
        num_edges: Number of edges (|W_ij| > epsilon)
        edge_density: Fraction of possible edges present
        
    References:
        IRHv16.md §1 Axiom 2: Network Emergence Principle
        IRHv16.md Theorem 1.2: Necessity of Network Representation
    """
    
    states: List[AlgorithmicHolonomicState]
    W: NDArray[np.complex128] = field(repr=False)  # ACW matrix
    epsilon_threshold: float = EPSILON_THRESHOLD
    
    def __post_init__(self):
        """Validate CRN structure."""
        N = len(self.states)
        if N == 0:
            raise ValueError("CRN must have at least one node")
        if self.W.shape != (N, N):
            raise ValueError(f"W matrix shape {self.W.shape} doesn't match N={N}")
    
    @property
    def N(self) -> int:
        """Number of nodes in the network."""
        return len(self.states)
    
    @property
    def num_edges(self) -> int:
        """Number of edges (non-zero, non-diagonal entries)."""
        # Count entries where |W_ij| > epsilon and i != j
        if sp.issparse(self.W):
            W_dense = self.W.toarray()
        else:
            W_dense = self.W
        
        mask = (np.abs(W_dense) > self.epsilon_threshold)
        np.fill_diagonal(mask, False)  # Exclude self-loops
        return int(np.sum(mask))
    
    @property
    def edge_density(self) -> float:
        """Fraction of possible edges present."""
        max_edges = self.N * (self.N - 1)  # Directed graph, no self-loops
        if max_edges == 0:
            return 0.0
        return self.num_edges / max_edges
    
    @classmethod
    def from_states(
        cls,
        states: List[AlgorithmicHolonomicState],
        epsilon_threshold: float = EPSILON_THRESHOLD,
        compression_level: int = 6,
    ) -> CymaticResonanceNetwork:
        """
        Construct CRN from a list of AHS.
        
        Per IRHv16.md §1 Axiom 2, this is the unique minimal representation.
        
        Args:
            states: List of AlgorithmicHolonomicState nodes
            epsilon_threshold: Edge inclusion threshold (default: 0.730129)
            compression_level: zlib compression level for NCD
            
        Returns:
            CymaticResonanceNetwork instance
            
        References:
            IRHv16.md §1 Axiom 2: Network construction rules
        """
        W = build_acw_matrix(
            states,
            epsilon_threshold=epsilon_threshold,
            compression_level=compression_level,
            sparse=False,  # Dense for now
        )
        return cls(states=states, W=W, epsilon_threshold=epsilon_threshold)
    
    @classmethod
    def create_random(
        cls,
        N: int,
        epsilon_threshold: float = EPSILON_THRESHOLD,
        seed: Optional[int] = None,
    ) -> CymaticResonanceNetwork:
        """
        Create a random CRN for testing/demonstration.
        
        Args:
            N: Number of AHS nodes
            epsilon_threshold: Edge inclusion threshold
            seed: Random seed for reproducibility
            
        Returns:
            CymaticResonanceNetwork with random AHS
        """
        states = create_ahs_network(N=N, seed=seed)
        return cls.from_states(states, epsilon_threshold=epsilon_threshold)
    
    def get_adjacency_matrix(self) -> NDArray[np.bool_]:
        """
        Get binary adjacency matrix A where A_ij = (|W_ij| > epsilon).
        
        Returns:
            N×N boolean adjacency matrix
        """
        if sp.issparse(self.W):
            W_dense = self.W.toarray()
        else:
            W_dense = self.W
        
        A = np.abs(W_dense) > self.epsilon_threshold
        np.fill_diagonal(A, False)  # No self-loops
        return A
    
    def get_degree_distribution(self) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
        """
        Get in-degree and out-degree for each node.
        
        Returns:
            (in_degrees, out_degrees) arrays
        """
        A = self.get_adjacency_matrix()
        in_degrees = np.sum(A, axis=0)
        out_degrees = np.sum(A, axis=1)
        return in_degrees, out_degrees
    
    def is_connected(self) -> bool:
        """
        Check if the CRN is weakly connected.
        
        Per IRHv16.md, a valid CRN should be connected for
        global coherence to be possible.
        
        Returns:
            True if network is weakly connected
        """
        A = self.get_adjacency_matrix()
        # Make symmetric for weak connectivity
        A_sym = A | A.T
        n_components, _ = connected_components(
            sp.csr_matrix(A_sym), directed=False
        )
        return n_components == 1
    
    def compute_cycle_holonomy(self, cycle: List[int]) -> complex:
        """
        Compute holonomy (coherent transfer product) around a cycle.
        
        Per IRHv16.md §2 Theorem 2.1:
            "The coherent transfer product for a cycle C is Π_C = ∏ W_ij"
        
        Args:
            cycle: List of node indices forming a cycle [i, j, k, ..., i]
            
        Returns:
            Complex holonomy Π_C
            
        References:
            IRHv16.md §2 Theorem 2.1: Algorithmic Quantization of Holonomic Phases
        """
        if len(cycle) < 2:
            raise ValueError("Cycle must have at least 2 nodes")
        if cycle[0] != cycle[-1]:
            raise ValueError("Cycle must start and end at same node")
        
        if sp.issparse(self.W):
            W_dense = self.W.toarray()
        else:
            W_dense = self.W
        
        holonomy = 1.0 + 0j
        for k in range(len(cycle) - 1):
            i, j = cycle[k], cycle[k + 1]
            holonomy *= W_dense[i, j]
        
        return holonomy
    
    def compute_cycle_phase(self, cycle: List[int]) -> float:
        """
        Compute total phase winding around a cycle.
        
        Per IRHv16.md §2:
            Φ_C = Σ_{(i,j) ∈ C} φ_ij mod 2π
        
        Args:
            cycle: List of node indices forming a cycle
            
        Returns:
            Phase winding in [0, 2π)
        """
        holonomy = self.compute_cycle_holonomy(cycle)
        return np.angle(holonomy) % (2 * np.pi)
    
    def find_triangular_cycles(self, max_cycles: int = 1000) -> List[List[int]]:
        """
        Find triangular cycles (length 3) in the network.
        
        These are the fundamental cycles for frustration computation.
        
        Args:
            max_cycles: Maximum number of cycles to return
            
        Returns:
            List of cycles as [i, j, k, i]
        """
        A = self.get_adjacency_matrix()
        cycles = []
        
        for i in range(self.N):
            if len(cycles) >= max_cycles:
                break
            # Find neighbors
            neighbors_i = np.where(A[i, :])[0]
            for j in neighbors_i:
                if j <= i:
                    continue
                neighbors_j = np.where(A[j, :])[0]
                # Find common neighbors that close the triangle
                for k in neighbors_j:
                    if k <= j:
                        continue
                    if A[k, i]:
                        cycles.append([i, j, k, i])
                        if len(cycles) >= max_cycles:
                            break
                if len(cycles) >= max_cycles:
                    break
        
        return cycles
    
    def compute_frustration_density(self, max_cycles: int = 1000) -> float:
        """
        Compute frustration density ρ_frust.
        
        Per IRHv16.md §2 Definition 2.1:
            ρ_frust := (1/|C_min|) Σ_{C ∈ C_min} |Φ_C|
            
        This is the average absolute value of the minimal non-zero
        holonomic phase winding per fundamental cycle.
        
        Args:
            max_cycles: Maximum number of cycles to consider
            
        Returns:
            Frustration density ρ_frust
            
        References:
            IRHv16.md §2 Definition 2.1: Frustration Density
            IRHv16.md Theorem 2.2: Fine-structure constant derivation
        """
        cycles = self.find_triangular_cycles(max_cycles=max_cycles)
        
        if not cycles:
            return 0.0
        
        phase_windings = [self.compute_cycle_phase(c) for c in cycles]
        
        # Compute absolute values, handling the branch cut at 2π
        # |Φ| should be the minimal distance from 0 or 2π
        abs_phases = []
        for phi in phase_windings:
            # Distance to 0 or 2π
            min_dist = min(phi, 2 * np.pi - phi)
            abs_phases.append(min_dist)
        
        return float(np.mean(abs_phases))
    
    def get_interference_matrix(self) -> NDArray[np.complex128]:
        """
        Compute the Interference Matrix L from the ACW matrix W.
        
        Per IRHv16.md, the Interference Matrix is the discrete Laplacian
        analog for the CRN:
            L_ij = D_ij - W_ij
        where D is the degree matrix.
        
        Returns:
            N×N complex interference matrix L
            
        References:
            IRHv16.md §4: Harmony Functional definition
        """
        if sp.issparse(self.W):
            W_dense = self.W.toarray()
        else:
            W_dense = self.W.copy()
        
        # Compute degree matrix (sum of absolute weights)
        degrees = np.sum(np.abs(W_dense), axis=1)
        D = np.diag(degrees)
        
        # L = D - W
        L = D - W_dense
        
        return L
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (f"CRN(N={self.N}, edges={self.num_edges}, "
                f"density={self.edge_density:.4f}, "
                f"ε={self.epsilon_threshold})")
    
    def __str__(self) -> str:
        """User-friendly string."""
        return f"Cymatic Resonance Network: {self.N} nodes, {self.num_edges} edges"


def derive_epsilon_threshold(
    N_samples: int = 100,
    N_per_sample: int = 50,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Derive optimal epsilon threshold from phase transition analysis.
    
    Per IRHv16.md §1 Axiom 2:
        "ε_threshold is the value that maximizes the Algorithmic Network
        Entropy, ensuring global connectivity (percolation) while minimizing
        redundant connections."
    
    This is a simplified derivation for demonstration. Full v16.0 requires
    exascale computation with N >= 10^12.
    
    Args:
        N_samples: Number of random networks to sample
        N_per_sample: Network size for each sample
        seed: Random seed
        
    Returns:
        (epsilon_optimal, error_estimate) tuple
        
    References:
        IRHv16.md §1 Axiom 2: Epsilon threshold derivation
        [IRH-MATH-2025-01]: Full mathematical derivation
    """
    rng = np.random.default_rng(seed)
    
    # Test range around expected value
    epsilon_values = np.linspace(0.5, 0.9, 41)
    connectivity_rates = []
    
    for eps in epsilon_values:
        connected_count = 0
        for _ in range(N_samples):
            sample_seed = rng.integers(0, 2**31)
            try:
                crn = CymaticResonanceNetwork.create_random(
                    N=N_per_sample,
                    epsilon_threshold=eps,
                    seed=sample_seed,
                )
                if crn.is_connected():
                    connected_count += 1
            except Exception:
                # Network creation may fail for edge cases (e.g., invalid parameters)
                # Skip failed samples and continue with remaining trials
                pass
        connectivity_rates.append(connected_count / N_samples)
    
    connectivity_rates = np.array(connectivity_rates)
    
    # Find critical point (50% connectivity = percolation threshold)
    # The optimal epsilon is at the critical point
    target = 0.5
    idx = np.argmin(np.abs(connectivity_rates - target))
    epsilon_optimal = epsilon_values[idx]
    
    # Error estimate from step size
    step_size = epsilon_values[1] - epsilon_values[0]
    error_estimate = step_size / 2
    
    return float(epsilon_optimal), float(error_estimate)


__version__ = "16.0.0-dev"
__status__ = "Phase 3 Implementation - CRN and network metrics"

__all__ = [
    "CymaticResonanceNetwork",
    "EPSILON_THRESHOLD",
    "EPSILON_THRESHOLD_ERROR",
    "derive_epsilon_threshold",
]
