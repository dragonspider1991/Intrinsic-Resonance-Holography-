"""
Cymatic Resonance Network - Real-valued oscillator substrate

This module implements the fundamental substrate of IRH v10.0:
A network of REAL-VALUED coupled harmonic oscillators.

Complex structure emerges via symplectic geometry → U(N) theorem.
NO complex weights from the beginning (major change from v9.5).

Mathematical Framework:
    - Position coordinates: q_i(t) ∈ ℝ^N
    - Momentum coordinates: p_i(t) ∈ ℝ^N
    - Coupling matrix: K_ij ∈ ℝ^(N×N), symmetric, positive semi-definite
    - Hamiltonian: H = Σ_i p_i²/2m + Σ_ij K_ij q_i q_j / 2

Reference: IRH v10.0 manuscript, Section II.A "The Real Substrate"
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple
import networkx as nx


class CymaticResonanceNetwork:
    """
    A network of real-valued coupled harmonic oscillators.
    
    The substrate consists of N oscillators with real-valued coupling.
    Complex structure emerges through the symplectic structure of phase space.
    
    Attributes:
        N (int): Number of oscillators
        K (np.ndarray or sp.spmatrix): Real symmetric coupling matrix N×N
        masses (np.ndarray): Mass array (default: all equal to 1.0)
        seed (int): Random seed for reproducibility
    
    Example:
        >>> network = CymaticResonanceNetwork(N=100, topology="grid_4d")
        >>> L = network.get_interference_matrix()
        >>> eigenvalues = network.compute_spectrum()
    """
    
    def __init__(
        self,
        N: int,
        topology: str = "random",
        coupling_strength: float = 1.0,
        seed: Optional[int] = None,
        sparse: bool = True,
    ):
        """
        Initialize a Cymatic Resonance Network.
        
        Args:
            N: Number of oscillators
            topology: Network topology ("random", "grid_4d", "toroidal_4d", "small_world")
            coupling_strength: Overall coupling scale
            seed: Random seed for reproducibility
            sparse: Use sparse matrix representation for large N
        """
        self.N = N
        self.topology = topology
        self.coupling_strength = coupling_strength
        self.seed = seed
        self.sparse = sparse and (N > 500)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize masses (all equal for now, can be made non-uniform)
        self.masses = np.ones(N)
        
        # Build coupling matrix K based on topology
        self.K = self._build_coupling_matrix()
        
    def _build_coupling_matrix(self) -> np.ndarray | sp.spmatrix:
        """
        Build the real symmetric coupling matrix K_ij based on topology.
        
        Returns:
            K: Real symmetric coupling matrix
        """
        if self.topology == "random":
            return self._random_coupling()
        elif self.topology == "grid_4d":
            return self._grid_4d_coupling()
        elif self.topology == "toroidal_4d":
            return self._toroidal_4d_coupling()
        elif self.topology == "small_world":
            return self._small_world_coupling()
        else:
            raise ValueError(f"Unknown topology: {self.topology}")
    
    def _random_coupling(self) -> np.ndarray:
        """Generate random coupling matrix (Erdős-Rényi style)."""
        p = min(0.1, 10.0 / self.N)  # Connection probability
        K = np.random.rand(self.N, self.N) < p
        K = K.astype(float) * self.coupling_strength
        K = (K + K.T) / 2  # Symmetrize
        np.fill_diagonal(K, 0)  # No self-coupling
        return K
    
    def _grid_4d_coupling(self) -> np.ndarray | sp.spmatrix:
        """
        Generate 4D hypercubic lattice coupling.
        
        This is the target topology for dimensional bootstrap → 4D spacetime.
        Reference: Manuscript Equation (23)
        """
        # Determine 4D grid dimensions
        n_per_dim = int(np.ceil(self.N ** 0.25))
        actual_N = n_per_dim ** 4
        
        if actual_N != self.N:
            print(f"Warning: Adjusting N from {self.N} to {actual_N} for 4D grid")
            self.N = actual_N
        
        if self.sparse:
            # Build sparse adjacency for 4D grid
            row, col, data = [], [], []
            for i in range(self.N):
                # Convert linear index to 4D coordinates
                coords = self._linear_to_4d(i, n_per_dim)
                # Add edges to nearest neighbors
                for dim in range(4):
                    for delta in [-1, 1]:
                        neighbor_coords = list(coords)
                        neighbor_coords[dim] += delta
                        if 0 <= neighbor_coords[dim] < n_per_dim:
                            j = self._coords_4d_to_linear(neighbor_coords, n_per_dim)
                            row.append(i)
                            col.append(j)
                            data.append(self.coupling_strength)
            
            K = sp.csr_matrix((data, (row, col)), shape=(self.N, self.N))
            K = (K + K.T) / 2  # Symmetrize
            return K
        else:
            # Dense version for small N
            K = np.zeros((self.N, self.N))
            for i in range(self.N):
                coords = self._linear_to_4d(i, n_per_dim)
                for dim in range(4):
                    for delta in [-1, 1]:
                        neighbor_coords = list(coords)
                        neighbor_coords[dim] += delta
                        if 0 <= neighbor_coords[dim] < n_per_dim:
                            j = self._coords_4d_to_linear(neighbor_coords, n_per_dim)
                            K[i, j] = self.coupling_strength
            return K
    
    def _toroidal_4d_coupling(self) -> np.ndarray | sp.spmatrix:
        """4D toroidal lattice (periodic boundary conditions)."""
        n_per_dim = int(np.ceil(self.N ** 0.25))
        actual_N = n_per_dim ** 4
        
        if actual_N != self.N:
            self.N = actual_N
        
        if self.sparse:
            row, col, data = [], [], []
            for i in range(self.N):
                coords = self._linear_to_4d(i, n_per_dim)
                for dim in range(4):
                    for delta in [-1, 1]:
                        neighbor_coords = list(coords)
                        neighbor_coords[dim] = (neighbor_coords[dim] + delta) % n_per_dim
                        j = self._coords_4d_to_linear(neighbor_coords, n_per_dim)
                        row.append(i)
                        col.append(j)
                        data.append(self.coupling_strength)
            
            K = sp.csr_matrix((data, (row, col)), shape=(self.N, self.N))
            K = (K + K.T) / 2
            return K
        else:
            K = np.zeros((self.N, self.N))
            for i in range(self.N):
                coords = self._linear_to_4d(i, n_per_dim)
                for dim in range(4):
                    for delta in [-1, 1]:
                        neighbor_coords = list(coords)
                        neighbor_coords[dim] = (neighbor_coords[dim] + delta) % n_per_dim
                        j = self._coords_4d_to_linear(neighbor_coords, n_per_dim)
                        K[i, j] = self.coupling_strength
            return K
    
    def _small_world_coupling(self) -> np.ndarray:
        """Small-world network (Watts-Strogatz model)."""
        k = 4  # Each node connected to k nearest neighbors
        p = 0.1  # Rewiring probability
        G = nx.watts_strogatz_graph(self.N, k, p, seed=self.seed)
        K = nx.to_numpy_array(G) * self.coupling_strength
        return K
    
    def _linear_to_4d(self, idx: int, n_per_dim: int) -> Tuple[int, int, int, int]:
        """Convert linear index to 4D coordinates."""
        i3 = idx % n_per_dim
        idx //= n_per_dim
        i2 = idx % n_per_dim
        idx //= n_per_dim
        i1 = idx % n_per_dim
        i0 = idx // n_per_dim
        return (i0, i1, i2, i3)
    
    def _coords_4d_to_linear(self, coords: list, n_per_dim: int) -> int:
        """Convert 4D coordinates to linear index."""
        return (coords[0] * n_per_dim**3 + 
                coords[1] * n_per_dim**2 + 
                coords[2] * n_per_dim + 
                coords[3])
    
    def get_interference_matrix(self) -> np.ndarray | sp.spmatrix:
        """
        Get the Interference Matrix (Graph Laplacian ℒ).
        
        The Interference Matrix is the graph Laplacian:
            ℒ = D - K
        where D is the degree matrix.
        
        Reference: Manuscript Equation (12)
        
        Returns:
            L: Interference matrix (Graph Laplacian)
        """
        if self.sparse and sp.issparse(self.K):
            degrees = np.array(self.K.sum(axis=1)).flatten()
            D = sp.diags(degrees)
            L = D - self.K
            return L
        else:
            degrees = self.K.sum(axis=1)
            D = np.diag(degrees)
            L = D - self.K
            return L
    
    def compute_spectrum(self, k: Optional[int] = None) -> np.ndarray:
        """
        Compute eigenvalue spectrum of the Interference Matrix.
        
        Args:
            k: Number of eigenvalues to compute (None = all for dense, min(N, 100) for sparse)
        
        Returns:
            eigenvalues: Sorted eigenvalues of ℒ
        """
        L = self.get_interference_matrix()
        
        if self.sparse and sp.issparse(L):
            # For sparse matrices, compute partial spectrum
            if k is None:
                k = min(self.N - 1, 100)
            try:
                eigenvalues = sp.linalg.eigsh(L, k=k, which='SM', return_eigenvectors=False)
                return np.sort(eigenvalues)
            except:
                # Fallback to dense computation for small matrices
                L_dense = L.toarray()
                eigenvalues = np.linalg.eigvalsh(L_dense)
                return np.sort(eigenvalues)
        else:
            # Dense computation
            eigenvalues = np.linalg.eigvalsh(L)
            return np.sort(eigenvalues)
    
    def get_degree_distribution(self) -> np.ndarray:
        """Get the degree distribution of the network."""
        if sp.issparse(self.K):
            return np.array(self.K.sum(axis=1)).flatten()
        else:
            return self.K.sum(axis=1)
