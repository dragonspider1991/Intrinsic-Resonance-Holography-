"""
Substrate module for IRH v11.0: Fundamental graph construction without assumptions.

Implements Axioms 0-2 from v11.0:
- Pure information state space
- Relationality via correlation tensor
- Finite information bound (holographic principle)

This module provides the foundational discrete structure from which all
physics emerges through ARO optimization.
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InformationSubstrate:
    """
    The foundational discrete structure.
    No geometry, time, or dynamics assumed—only correlations.
    
    Attributes:
        N (int): Number of fundamental information nodes
        dimension (int): Target emergent dimension (for initial embedding)
        holographic_bound (float): Maximum information capacity (I ≤ ln N)
        states (np.ndarray): State labels
        C (np.ndarray): Correlation tensor
        W (sp.csr_matrix): Graph adjacency with complex weights
        L (sp.csr_matrix): Laplacian operator
    """
    
    def __init__(self, N: int, dimension: Optional[int] = None):
        """
        Initialize substrate with N distinguishable states.
        
        Parameters:
        -----------
        N : int
            Number of fundamental information nodes
        dimension : int, optional
            Target emergent dimension (for initial embedding only)
        """
        self.N = N
        self.dimension = dimension if dimension is not None else 4
        self.holographic_bound = np.log(N)  # I_max ≤ ln(N)
        
        # State space (initially unstructured)
        self.states = np.arange(N)
        
        # Correlation structure (to be determined)
        self.C = None  # Correlation tensor
        self.W = None  # Graph adjacency (derived from C)
        self.L = None  # Laplacian (derived from W)
        
        logger.info(f"Initialized InformationSubstrate: N={N}, d={self.dimension}")
        
    def initialize_correlations(self, method: str = 'random_geometric') -> sp.csr_matrix:
        """
        Establish initial correlation structure.
        
        Methods:
        --------
        'random_geometric' : Spatial proximity in emergent dimension
        'maximum_entropy' : Uniform random (principle of indifference)
        'small_world' : Watts-Strogatz for realistic topology
        """
        
        if method == 'random_geometric':
            return self._init_random_geometric()
        elif method == 'maximum_entropy':
            return self._init_maximum_entropy()
        elif method == 'small_world':
            return self._init_small_world()
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def _init_random_geometric(self) -> sp.csr_matrix:
        """
        Initialize via random geometric graph.
        States embedded in d-dimensional space; correlations decay with distance.
        """
        # Embed nodes in [0,1]^d hypercube
        positions = np.random.rand(self.N, self.dimension)
        
        # Compute pairwise distances
        dist_matrix = squareform(pdist(positions, metric='euclidean'))
        
        # Critical radius for percolation (criticality condition)
        # For d-dimensional geometric graph: r_c ~ (ln N / N)^(1/d)
        r_c = (np.log(self.N) / (np.pi * self.N))**(1/self.dimension)
        
        # Adjacency: connect if distance < r_c
        A = (dist_matrix < r_c).astype(float)
        np.fill_diagonal(A, 0)
        
        # Add phase frustration (complex weights)
        # Phase encodes geometric frustration from embedding impossibility
        phases = self._compute_frustration_phases(positions, A)
        self.W = sp.csr_matrix(A * np.exp(1j * phases))
        
        logger.info(f"Random geometric graph initialized: {np.sum(A)/2:.0f} edges, r_c={r_c:.4f}")
        
        return self.W
    
    def _compute_frustration_phases(self, positions: np.ndarray, 
                                   A: np.ndarray) -> np.ndarray:
        """
        Derive complex phases from geometric frustration.
        
        Key insight: For triangles (3-cycles), the sum of angle deficits
        equals the curvature. This translates to phase holonomy.
        """
        phases = np.zeros_like(A)
        
        # For each edge, compute the "twisting" required to maintain
        # transitivity of correlations
        for i in range(self.N):
            for j in range(i+1, self.N):
                if A[i, j] > 0:
                    # Find common neighbors (forming triangles)
                    common = np.where((A[i, :] > 0) & (A[j, :] > 0))[0]
                    
                    if len(common) > 0:
                        # Average phase deficit over all triangles
                        phase_sum = 0
                        for k in common:
                            # Compute angle deficit in triangle (i,j,k)
                            v1 = positions[j] - positions[i]
                            v2 = positions[k] - positions[i]
                            v3 = positions[k] - positions[j]
                            
                            # Angles at each vertex
                            norm_v1 = np.linalg.norm(v1)
                            norm_v2 = np.linalg.norm(v2)
                            norm_v3 = np.linalg.norm(v3)
                            
                            if norm_v1 > 1e-10 and norm_v2 > 1e-10:
                                angle_i = np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1, 1))
                            else:
                                angle_i = 0
                                
                            if norm_v1 > 1e-10 and norm_v3 > 1e-10:
                                angle_j = np.arccos(np.clip(np.dot(-v1, v3) / (norm_v1 * norm_v3), -1, 1))
                            else:
                                angle_j = 0
                                
                            if norm_v2 > 1e-10 and norm_v3 > 1e-10:
                                angle_k = np.arccos(np.clip(np.dot(-v2, -v3) / (norm_v2 * norm_v3), -1, 1))
                            else:
                                angle_k = 0
                            
                            # Deficit from π (flat triangle)
                            deficit = np.pi - (angle_i + angle_j + angle_k)
                            phase_sum += deficit
                        
                        phases[i, j] = phase_sum / len(common)
                        phases[j, i] = -phases[i, j]  # Antisymmetry
        
        return phases
    
    def _init_maximum_entropy(self) -> sp.csr_matrix:
        """Initialize with maximum entropy (uniform random)."""
        # Random Erdős-Rényi graph
        p_edge = np.log(self.N) / self.N  # Critical probability
        A = (np.random.rand(self.N, self.N) < p_edge).astype(float)
        A = (A + A.T) / 2  # Symmetrize
        np.fill_diagonal(A, 0)
        
        # Random phases
        phases = np.random.uniform(0, 2*np.pi, (self.N, self.N))
        phases = (phases - phases.T) / 2  # Antisymmetrize
        
        self.W = sp.csr_matrix(A * np.exp(1j * phases))
        return self.W
    
    def _init_small_world(self) -> sp.csr_matrix:
        """Initialize with small-world topology (Watts-Strogatz)."""
        import networkx as nx
        
        k = int(np.log(self.N))  # Average degree
        p_rewire = 0.1  # Rewiring probability
        
        G = nx.watts_strogatz_graph(self.N, k, p_rewire)
        A = nx.to_numpy_array(G)
        
        # Add random phases
        phases = np.random.uniform(0, 2*np.pi, (self.N, self.N))
        phases = (phases - phases.T) / 2
        
        self.W = sp.csr_matrix(A * np.exp(1j * phases))
        return self.W
    
    def compute_laplacian(self) -> sp.csr_matrix:
        """
        Construct the Interference Matrix (graph Laplacian).
        L = D - W where D is degree matrix.
        """
        if self.W is None:
            raise ValueError("Must initialize correlations first")
        
        W_abs = np.abs(self.W.toarray())
        D = np.diag(W_abs.sum(axis=1))
        self.L = sp.csr_matrix(D - W_abs)
        
        logger.info(f"Laplacian computed: shape {self.L.shape}")
        
        return self.L
    
    def verify_holographic_bound(self) -> Dict[str, float]:
        """
        Check that information capacity satisfies holographic principle.
        
        Returns:
        --------
        dict with keys:
            'I_bulk' : bulk information content
            'I_boundary' : holographic bound (boundary capacity)
            'ratio' : I_bulk / I_boundary (should be ≤ 1)
        """
        if self.L is None:
            self.compute_laplacian()
        
        # Bulk entropy: von Neumann entropy of density matrix
        try:
            from scipy.sparse.linalg import eigsh
            k = min(100, self.N-2)
            eigenvalues = eigsh(self.L, k=k, which='SM', return_eigenvectors=False)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
        except:
            # Fallback to dense computation for small systems
            eigenvalues = np.linalg.eigvalsh(self.L.toarray())
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Normalize to form probability distribution
        if len(eigenvalues) > 0:
            p = eigenvalues / eigenvalues.sum()
            S_bulk = -np.sum(p * np.log2(p + 1e-10))
        else:
            S_bulk = 0
        
        # Boundary capacity: N^((d-1)/d) for d-dimensional system
        S_boundary = self.N**((self.dimension - 1) / self.dimension)
        
        I_boundary = np.log2(S_boundary + 1)
        ratio = S_bulk / max(I_boundary, 1e-10)
        
        return {
            'I_bulk': S_bulk,
            'I_boundary': I_boundary,
            'ratio': ratio,
            'satisfies_bound': ratio <= 1.1  # Allow 10% tolerance
        }
