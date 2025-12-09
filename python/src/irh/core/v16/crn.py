"""
Cymatic Resonance Network (CRN) for IRH v16.0

Implements Axiom 2 (Network Emergence Principle) with complex-valued
Algorithmic Coherence Weights from Axiom 1.

THEORETICAL COMPLIANCE:
    This implementation strictly follows docs/manuscripts/IRHv16.md
    - Axiom 2 (§1): Network Emergence Principle with ε_threshold
    - Section on CRN construction from AHS and ACW
    - Complex graph Laplacian ℒ for Harmony Functional (§4)

Key Concepts:
    - Nodes: Algorithmic Holonomic States (AHS) [IRHv16.md Axiom 0]
    - Edges: |W_ij| > ε_threshold = 0.730129 ± 10^-6 [IRHv16.md Axiom 2]
    - Weights: W_ij ∈ ℂ (complex Algorithmic Coherence Weights) [IRHv16.md Axiom 1]
    - Laplacian: Complex graph Laplacian (Interference Matrix ℒ) [IRHv16.md §4]

References:
    docs/manuscripts/IRHv16.md:
        - §1 Axiom 2: Network Emergence Principle
        - §1 lines 87-100: ε_threshold = 0.730129 ± 10^-6 derivation
        - §4 lines 254-269: Harmony Functional S_H = Tr(ℒ²) / [det'(ℒ)]^{C_H}
        - §1 lines 66-83: Complex-valued W_ij definition
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
import networkx as nx

from .ahs import AlgorithmicHolonomicState
from .acw import AlgorithmicCoherenceWeight, compute_acw


@dataclass
class CymaticResonanceNetworkV16:
    """
    Cymatic Resonance Network with complex-valued ACWs for IRH v16.0.
    
    Implements Axiom 2: Network Emergence Principle
    
    Attributes:
        states: List of Algorithmic Holonomic States (nodes)
        epsilon_threshold: Edge inclusion threshold (ε from Axiom 2)
        adjacency_matrix: Complex-valued adjacency matrix W_ij
        graph: NetworkX DiGraph for standard graph operations
        
    Properties:
        N: Number of nodes
        interference_matrix: Complex graph Laplacian ℒ
        num_edges: Number of edges (|W_ij| > ε)
        
    References:
        IRHv16.md Axiom 2: ε = 0.730129 ± 10^-6 (derived from network criticality)
    """
    
    states: List[AlgorithmicHolonomicState]
    epsilon_threshold: float = 0.730129  # From Axiom 2
    adjacency_matrix: Optional[NDArray[np.complex128]] = None
    graph: Optional[nx.DiGraph] = None
    
    def __post_init__(self):
        """Initialize network structure after creation."""
        if self.adjacency_matrix is None:
            self.build_network()
    
    @property
    def N(self) -> int:
        """Number of nodes in the network."""
        return len(self.states)
    
    @property
    def interference_matrix(self) -> NDArray[np.complex128]:
        """
        Complex graph Laplacian (Interference Matrix ℒ).
        
        For directed graphs with complex weights:
        ℒ_ii = Σ_j W_ij (out-degree sum)
        ℒ_ij = -W_ij for i ≠ j
        
        This is the operator in the Harmony Functional:
        S_H = Tr(ℒ²) / [det'(ℒ)]^{C_H}
        
        Returns:
            Complex Laplacian matrix (N x N)
        """
        if self.adjacency_matrix is None:
            raise ValueError("Network not built yet. Call build_network() first.")
        
        N = self.N
        L = np.zeros((N, N), dtype=np.complex128)
        
        # Off-diagonal: -W_ij
        for i in range(N):
            for j in range(N):
                if i != j:
                    L[i, j] = -self.adjacency_matrix[i, j]
        
        # Diagonal: sum of outgoing weights
        for i in range(N):
            L[i, i] = np.sum(self.adjacency_matrix[i, :])
        
        return L
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the network (|W_ij| > ε)."""
        if self.adjacency_matrix is None:
            return 0
        return np.count_nonzero(np.abs(self.adjacency_matrix) > self.epsilon_threshold)
    
    def build_network(self, method: str = "lzw") -> None:
        """
        Build the network by computing ACWs between all AHS pairs.
        
        Creates edges where |W_ij| > ε_threshold (Axiom 2).
        
        Args:
            method: NCD computation method ("lzw")
            
        Notes:
            For N states, computes N² ACW values. For large N (> 1000),
            consider using sparse methods or distributed computation.
        """
        N = self.N
        
        # Initialize adjacency matrix
        self.adjacency_matrix = np.zeros((N, N), dtype=np.complex128)
        
        # Compute ACW for all pairs
        for i in range(N):
            for j in range(N):
                if i == j:
                    # No self-loops
                    continue
                
                # Compute W_ij
                acw = compute_acw(self.states[i], self.states[j], method=method)
                
                # Only add edge if |W_ij| > ε_threshold
                if acw.magnitude > self.epsilon_threshold:
                    self.adjacency_matrix[i, j] = acw.complex_value
        
        # Build NetworkX graph for visualization and analysis
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(N))
        
        for i in range(N):
            for j in range(N):
                if np.abs(self.adjacency_matrix[i, j]) > 0:
                    self.graph.add_edge(
                        i, j,
                        weight=np.abs(self.adjacency_matrix[i, j]),
                        phase=np.angle(self.adjacency_matrix[i, j])
                    )
    
    def get_weight(self, i: int, j: int) -> complex:
        """
        Get complex weight W_ij between nodes i and j.
        
        Args:
            i: Source node index
            j: Target node index
            
        Returns:
            Complex weight W_ij (0 if no edge exists)
        """
        if self.adjacency_matrix is None:
            raise ValueError("Network not built yet.")
        
        return self.adjacency_matrix[i, j]
    
    def get_neighbors(self, i: int, direction: str = "out") -> List[int]:
        """
        Get neighbors of node i.
        
        Args:
            i: Node index
            direction: "out" for outgoing, "in" for incoming, "both" for all
            
        Returns:
            List of neighbor indices
        """
        if self.adjacency_matrix is None:
            raise ValueError("Network not built yet.")
        
        if direction == "out":
            # Outgoing: j where W_ij exists
            return [j for j in range(self.N) if np.abs(self.adjacency_matrix[i, j]) > 0]
        elif direction == "in":
            # Incoming: j where W_ji exists
            return [j for j in range(self.N) if np.abs(self.adjacency_matrix[j, i]) > 0]
        elif direction == "both":
            out_neighbors = set(self.get_neighbors(i, "out"))
            in_neighbors = set(self.get_neighbors(i, "in"))
            return list(out_neighbors | in_neighbors)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def compute_spectral_properties(self) -> Dict[str, any]:
        """
        Compute spectral properties of the Interference Matrix.
        
        Returns dictionary with:
            - eigenvalues: Complex eigenvalues of ℒ
            - trace_L2: Tr(ℒ²) for Harmony Functional
            - det_prime: det'(ℒ) excluding zero eigenvalues
            
        Notes:
            For preliminary Harmony Functional computation.
            Full exascale implementation in Phase 2.
        """
        L = self.interference_matrix
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(L)
        
        # Tr(ℒ²) = Tr(ℒ * ℒ)
        L2 = L @ L
        trace_L2 = np.trace(L2)
        
        # det'(ℒ): product of non-zero eigenvalues
        # Zero threshold
        zero_threshold = 1e-10
        nonzero_eigs = eigenvalues[np.abs(eigenvalues) > zero_threshold]
        
        if len(nonzero_eigs) > 0:
            det_prime = np.prod(nonzero_eigs)
        else:
            det_prime = 1.0  # Degenerate case
        
        return {
            "eigenvalues": eigenvalues,
            "trace_L2": trace_L2,
            "det_prime": det_prime,
            "num_zero_eigenvalues": len(eigenvalues) - len(nonzero_eigs),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        if self.adjacency_matrix is None:
            return f"CRNv16(N={self.N}, not built)"
        return f"CRNv16(N={self.N}, edges={self.num_edges}, ε={self.epsilon_threshold:.6f})"


def create_crn_from_states(
    states: List[AlgorithmicHolonomicState],
    epsilon_threshold: float = 0.730129
) -> CymaticResonanceNetworkV16:
    """
    Create a Cymatic Resonance Network from a list of AHS.
    
    Convenience function for network creation.
    
    Args:
        states: List of Algorithmic Holonomic States
        epsilon_threshold: Edge inclusion threshold (default from Axiom 2)
        
    Returns:
        Initialized CymaticResonanceNetworkV16
        
    Example:
        >>> from .ahs import create_ahs_network
        >>> states = create_ahs_network(N=10, seed=42)
        >>> crn = create_crn_from_states(states)
        >>> print(crn)
        CRNv16(N=10, edges=45, ε=0.730129)
    """
    return CymaticResonanceNetworkV16(
        states=states,
        epsilon_threshold=epsilon_threshold
    )


__all__ = [
    "CymaticResonanceNetworkV16",
    "create_crn_from_states",
]
