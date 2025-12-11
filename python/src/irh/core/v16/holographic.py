"""
Axiom 3: Combinatorial Holographic Principle

This module implements the Combinatorial Holographic Principle as defined in
IRHv16.md §1 Axiom 3.

Key Concepts (from IRHv16.md §1 Axiom 3):
    - For any subnetwork G_A ⊂ G, the maximum algorithmic information content I_A
      is bounded by the combinatorial capacity of its boundary:
      
      I_A(G_A) ≤ K · Σ_{v ∈ ∂G_A} deg(v)
      
    - ∂G_A is the boundary (nodes with edges crossing to outside G_A)
    - deg(v) is the degree of node v
    - K is a universal dimensionless constant

    Theorem 1.3 (Optimal Holographic Scaling):
        Linear scaling (β = 1) is the unique globally stable fixed point
        for holographic information scaling under ARO dynamics.

Implementation Status: Phase 4 Implementation
    - Subnetwork extraction: IMPLEMENTED
    - Boundary computation: IMPLEMENTED
    - Information content estimation: IMPLEMENTED
    - Holographic bound verification: IMPLEMENTED

References:
    IRHv16.md §1 Axiom 3: Combinatorial Holographic Principle
    IRHv16.md Theorem 1.3: Optimal Holographic Scaling
    [IRH-MATH-2025-01]: Free energy functional analysis
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Set, Dict, Any
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from .crn import CymaticResonanceNetwork


# Universal holographic constant K (placeholder - should be derived)
# Per IRHv16.md, K is "a universal dimensionless constant"
# TODO v16.0: Derive K from free energy functional analysis per [IRH-MATH-2025-01] Theorem 1.3
#             This requires the full ARO dynamics proof showing β=1 is the unique fixed point.
#             Current value is a placeholder for testing the implementation structure.
HOLOGRAPHIC_CONSTANT_K = 1.0  # Placeholder, needs derivation from theory
HOLOGRAPHIC_CONSTANT_K_ERROR = 0.1  # Placeholder uncertainty

# Holographic entropy factor: S_holo = HOLOGRAPHIC_ENTROPY_FACTOR * A_boundary
# Per Bekenstein-Hawking formula: S = (1/4) * A / l_p^2
HOLOGRAPHIC_ENTROPY_FACTOR = 0.25

# Linear scaling tolerance: β is considered linear if |β - 1| < LINEAR_SCALING_TOLERANCE
LINEAR_SCALING_TOLERANCE = 0.2  # Within 20% of β=1


@dataclass
class Subnetwork:
    """
    A subnetwork G_A ⊂ G extracted from a CRN.
    
    Attributes:
        node_indices: Set of node indices in the subnetwork
        parent_crn: Reference to the parent CRN
        boundary_nodes: Nodes on the boundary ∂G_A
        interior_nodes: Nodes in the interior (not on boundary)
    """
    node_indices: Set[int]
    parent_crn: CymaticResonanceNetwork
    
    def __post_init__(self):
        """Compute boundary and interior."""
        if not self.node_indices:
            raise ValueError("Subnetwork must have at least one node")
        
        N_parent = self.parent_crn.N
        for idx in self.node_indices:
            if idx < 0 or idx >= N_parent:
                raise ValueError(f"Node index {idx} out of range [0, {N_parent})")
    
    @property
    def N(self) -> int:
        """Number of nodes in subnetwork."""
        return len(self.node_indices)
    
    @property
    def boundary_nodes(self) -> Set[int]:
        """
        Nodes on the boundary ∂G_A.
        
        Per IRHv16.md: The boundary consists of nodes with edges
        crossing to nodes outside G_A.
        """
        A = self.parent_crn.get_adjacency_matrix()
        boundary = set()
        
        for i in self.node_indices:
            # Check if node i has neighbors outside subnetwork
            # For complex matrices, check |W| > 0
            out_neighbors = set(np.where(np.abs(A[i, :]) > 0)[0])
            in_neighbors = set(np.where(np.abs(A[:, i]) > 0)[0])
            neighbors = out_neighbors | in_neighbors
            if neighbors - self.node_indices:
                boundary.add(i)
        
        return boundary
    
    @property
    def interior_nodes(self) -> Set[int]:
        """Nodes not on the boundary."""
        return self.node_indices - self.boundary_nodes
    
    @property
    def boundary_degree_sum(self) -> int:
        """
        Sum of degrees for boundary nodes.
        
        Per IRHv16.md: Σ_{v ∈ ∂G_A} deg(v)
        """
        in_deg, out_deg = self.parent_crn.get_degree_distribution()
        
        total_degree = 0
        for v in self.boundary_nodes:
            # Degree is sum of in-degree and out-degree
            total_degree += int(in_deg[v]) + int(out_deg[v])
        
        return total_degree
    
    def get_subnetwork_matrix(self) -> NDArray[np.complex128]:
        """
        Extract the W matrix for just this subnetwork.
        
        Returns:
            |G_A| × |G_A| complex matrix
        """
        if sp.issparse(self.parent_crn.W):
            W_full = self.parent_crn.W.toarray()
        else:
            W_full = self.parent_crn.W
        
        # Convert to sorted list for consistent indexing
        indices = sorted(self.node_indices)
        W_sub = W_full[np.ix_(indices, indices)]
        
        return W_sub
    
    def compute_information_content(self) -> float:
        """
        Estimate the algorithmic information content I_A of the subnetwork.
        
        Per IRHv16.md: This is the total algorithmic information
        stored in the AHS within G_A.
        
        For now, we estimate this as the sum of K_t complexities
        of the states in the subnetwork.
        
        Returns:
            Estimated information content I_A
        """
        total_info = 0.0
        for idx in self.node_indices:
            state = self.parent_crn.states[idx]
            # Use complexity estimate, or information content as fallback
            if state.complexity_Kt is not None:
                total_info += state.complexity_Kt
            else:
                total_info += state.information_content
        
        return total_info


class HolographicAnalyzer:
    """
    Analyzer for the Combinatorial Holographic Principle.
    
    Per IRHv16.md §1 Axiom 3:
        "For any subnetwork G_A ⊂ G, the maximum algorithmic information
        content I_A is bounded by the combinatorial capacity of its boundary."
    
    This class provides tools to:
    1. Extract subnetworks from a CRN
    2. Compute boundary properties
    3. Verify the holographic bound
    4. Test holographic scaling (Theorem 1.3)
    
    References:
        IRHv16.md §1 Axiom 3: Combinatorial Holographic Principle
        IRHv16.md Theorem 1.3: Optimal Holographic Scaling
    """
    
    def __init__(self, crn: CymaticResonanceNetwork, K: float = HOLOGRAPHIC_CONSTANT_K):
        """
        Initialize holographic analyzer.
        
        Args:
            crn: The Cymatic Resonance Network to analyze
            K: Universal holographic constant (default: HOLOGRAPHIC_CONSTANT_K)
        """
        self.crn = crn
        self.K = K
    
    def extract_subnetwork(self, node_indices: Set[int]) -> Subnetwork:
        """
        Extract a subnetwork containing the specified nodes.
        
        Args:
            node_indices: Set of node indices to include
            
        Returns:
            Subnetwork object
        """
        return Subnetwork(node_indices=node_indices, parent_crn=self.crn)
    
    def extract_random_subnetwork(
        self, 
        size: int, 
        seed: Optional[int] = None
    ) -> Subnetwork:
        """
        Extract a random subnetwork of specified size.
        
        Args:
            size: Number of nodes to include
            seed: Random seed for reproducibility
            
        Returns:
            Random Subnetwork object
        """
        rng = np.random.default_rng(seed)
        
        if size > self.crn.N:
            raise ValueError(f"Requested size {size} > network size {self.crn.N}")
        
        indices = set(rng.choice(self.crn.N, size=size, replace=False))
        return self.extract_subnetwork(indices)
    
    def compute_holographic_bound(self, subnetwork: Subnetwork) -> float:
        """
        Compute the holographic bound for a subnetwork.
        
        Per IRHv16.md: I_A(G_A) ≤ K · Σ_{v ∈ ∂G_A} deg(v)
        
        Args:
            subnetwork: The subnetwork to analyze
            
        Returns:
            The holographic bound value
        """
        return self.K * subnetwork.boundary_degree_sum
    
    def verify_holographic_bound(self, subnetwork: Subnetwork) -> Tuple[bool, float, float]:
        """
        Verify the holographic bound is satisfied.
        
        Args:
            subnetwork: The subnetwork to check
            
        Returns:
            (is_satisfied, information_content, bound) tuple
        """
        I_A = subnetwork.compute_information_content()
        bound = self.compute_holographic_bound(subnetwork)
        
        is_satisfied = I_A <= bound
        
        return is_satisfied, I_A, bound
    
    def test_holographic_scaling(
        self,
        n_samples: int = 20,
        size_range: Tuple[int, int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test the holographic scaling relationship.
        
        Per IRHv16.md Theorem 1.3:
            Linear scaling (β = 1) is the unique globally stable fixed point.
        
        This tests I_A ~ boundary_capacity^β and estimates β.
        
        Args:
            n_samples: Number of random subnetworks to sample
            size_range: (min_size, max_size) for subnetworks
            seed: Random seed
            
        Returns:
            Dictionary with scaling analysis results
        """
        rng = np.random.default_rng(seed)
        
        if size_range is None:
            min_size = max(2, self.crn.N // 10)
            max_size = max(3, self.crn.N // 2)
            size_range = (min_size, max_size)
        
        # Collect data points
        boundary_capacities = []
        information_contents = []
        
        for i in range(n_samples):
            size = rng.integers(size_range[0], size_range[1] + 1)
            try:
                sub = self.extract_random_subnetwork(size, seed=rng.integers(0, 2**31))
                
                I_A = sub.compute_information_content()
                boundary_cap = sub.boundary_degree_sum
                
                if boundary_cap > 0 and I_A > 0:
                    boundary_capacities.append(boundary_cap)
                    information_contents.append(I_A)
            except Exception:
                continue
        
        if len(boundary_capacities) < 3:
            return {
                "success": False,
                "message": "Not enough valid samples",
                "n_samples": len(boundary_capacities)
            }
        
        # Fit power law: I_A = a * boundary^β
        # log(I_A) = log(a) + β * log(boundary)
        log_boundary = np.log(boundary_capacities)
        log_info = np.log(information_contents)
        
        # Linear regression
        A = np.vstack([log_boundary, np.ones(len(log_boundary))]).T
        result = np.linalg.lstsq(A, log_info, rcond=None)
        beta, log_a = result[0]
        
        # Compute R² for goodness of fit
        predicted = beta * log_boundary + log_a
        ss_res = np.sum((log_info - predicted) ** 2)
        ss_tot = np.sum((log_info - np.mean(log_info)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            "success": True,
            "beta": float(beta),
            "log_a": float(log_a),
            "r_squared": float(r_squared),
            "n_samples": len(boundary_capacities),
            "is_linear_scaling": abs(beta - 1.0) < LINEAR_SCALING_TOLERANCE,
            "boundary_capacities": boundary_capacities,
            "information_contents": information_contents
        }
    
    def compute_holographic_entropy(self, subnetwork: Subnetwork) -> float:
        """
        Compute the holographic entropy of a subnetwork.
        
        The holographic entropy is related to the boundary area
        in the emergent geometry.
        
        S_holo = HOLOGRAPHIC_ENTROPY_FACTOR * A_boundary
        
        For discrete networks, we use boundary degree sum as area proxy.
        
        Args:
            subnetwork: The subnetwork to analyze
            
        Returns:
            Holographic entropy estimate
        """
        # Using boundary degree sum as discrete analog of boundary area
        area = subnetwork.boundary_degree_sum
        return HOLOGRAPHIC_ENTROPY_FACTOR * area
    
    def analyze_all_sizes(
        self,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive holographic analysis across all subnetwork sizes.
        
        Args:
            seed: Random seed
            
        Returns:
            Analysis results dictionary
        """
        results = {
            "sizes": [],
            "avg_info_content": [],
            "avg_boundary_degree": [],
            "bound_satisfied_ratio": [],
        }
        
        rng = np.random.default_rng(seed)
        
        for size in range(2, min(self.crn.N, 20)):
            n_trials = 10
            info_contents = []
            boundary_degrees = []
            satisfied = 0
            
            for _ in range(n_trials):
                try:
                    sub = self.extract_random_subnetwork(size, seed=rng.integers(0, 2**31))
                    is_sat, I_A, bound = self.verify_holographic_bound(sub)
                    
                    info_contents.append(I_A)
                    boundary_degrees.append(sub.boundary_degree_sum)
                    if is_sat:
                        satisfied += 1
                except Exception:
                    continue
            
            if info_contents:
                results["sizes"].append(size)
                results["avg_info_content"].append(np.mean(info_contents))
                results["avg_boundary_degree"].append(np.mean(boundary_degrees))
                results["bound_satisfied_ratio"].append(satisfied / len(info_contents))
        
        return results


def verify_holographic_principle(
    crn: CymaticResonanceNetwork,
    n_tests: int = 100,
    seed: Optional[int] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Verify the Combinatorial Holographic Principle on a CRN.
    
    Per IRHv16.md §1 Axiom 3, for all subnetworks G_A:
        I_A(G_A) ≤ K · Σ_{v ∈ ∂G_A} deg(v)
    
    Args:
        crn: The CRN to test
        n_tests: Number of random subnetworks to test
        seed: Random seed
        
    Returns:
        (satisfaction_ratio, details) where satisfaction_ratio is
        the fraction of tests where the bound was satisfied
    """
    analyzer = HolographicAnalyzer(crn)
    rng = np.random.default_rng(seed)
    
    satisfied_count = 0
    total_tests = 0
    violations = []
    
    for i in range(n_tests):
        size = rng.integers(2, max(3, crn.N // 2 + 1))
        try:
            sub = analyzer.extract_random_subnetwork(size, seed=rng.integers(0, 2**31))
            is_satisfied, I_A, bound = analyzer.verify_holographic_bound(sub)
            
            total_tests += 1
            if is_satisfied:
                satisfied_count += 1
            else:
                violations.append({
                    "size": size,
                    "I_A": I_A,
                    "bound": bound,
                    "ratio": I_A / bound if bound > 0 else float('inf')
                })
        except Exception:
            continue
    
    satisfaction_ratio = satisfied_count / total_tests if total_tests > 0 else 0.0
    
    details = {
        "total_tests": total_tests,
        "satisfied": satisfied_count,
        "violations": violations,
        "K_used": HOLOGRAPHIC_CONSTANT_K
    }
    
    return satisfaction_ratio, details


__version__ = "16.0.0-dev"
__status__ = "Phase 4 Implementation - Axiom 3 Holographic Principle"

__all__ = [
    "Subnetwork",
    "HolographicAnalyzer",
    "verify_holographic_principle",
    "HOLOGRAPHIC_CONSTANT_K",
    "HOLOGRAPHIC_CONSTANT_K_ERROR",
]
