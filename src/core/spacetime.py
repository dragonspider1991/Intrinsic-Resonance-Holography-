"""
spacetime.py - Dimensional Bootstrap for Spacetime Emergence

RIRH v9.5 Spacetime Emergence Framework

This module implements the Dimensional Bootstrap mechanism for computing
intrinsic dimensions of emergent spacetime from hypergraph dynamics.

Key Components:
- Dimensional_Bootstrap: Class for computing intrinsic dimensions via
  spectral and geometric methods
- Heat kernel trace analysis for spectral dimension d_spectral
- BFS volume scaling for growth dimension d_growth
- SOTE penalty functional for dimension consistency

References:
- Heat kernel methods on graphs
- Spectral dimension in quantum gravity
- SOTE principle for dimensional emergence
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import issparse
from collections import deque


class Dimensional_Bootstrap:
    """
    Dimensional Bootstrap for Spacetime Emergence.

    Computes intrinsic dimensions of a hypergraph using multiple methods:
    1. Spectral dimension (d_spectral) from heat kernel trace slope
    2. Growth dimension (d_growth) from BFS volume scaling

    The SOTE principle drives dimension consistency: at equilibrium,
    d_spectral ≈ d_growth ≈ 4 for physically relevant configurations.

    Attributes:
        adj_matrix: Adjacency matrix of the hypergraph
        N: Number of nodes
        laplacian: Graph Laplacian matrix
        eigenvalues: Laplacian eigenvalues
        eigenvectors: Laplacian eigenvectors
    """

    def __init__(self, adj_matrix=None):
        """
        Initialize Dimensional Bootstrap.

        Args:
            adj_matrix: Optional adjacency matrix. If provided, computes
                       Laplacian spectrum upon initialization.
        """
        self.adj_matrix = None
        self.laplacian = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.N = 0

        if adj_matrix is not None:
            self._initialize_from_adjacency(adj_matrix)

    def _initialize_from_adjacency(self, adj_matrix):
        """Initialize from adjacency matrix."""
        if issparse(adj_matrix):
            self.adj_matrix = adj_matrix.toarray()
        else:
            self.adj_matrix = np.asarray(adj_matrix, dtype=float)

        self.N = self.adj_matrix.shape[0]

        # Compute Laplacian: L = D - A
        degrees = np.sum(self.adj_matrix, axis=1)
        D = np.diag(degrees)
        self.laplacian = D - self.adj_matrix

        # Compute eigendecomposition
        self.eigenvalues, self.eigenvectors = eigh(self.laplacian)

    def compute_intrinsic_dims(self, adj_matrix):
        """
        Calculate intrinsic dimensions from the hypergraph structure.

        Computes two independent dimension estimates:
        1. d_spectral: From heat kernel trace slope (log-log fit)
           K(t) = Tr(exp(-tL)) ~ t^(-d_s/2) for small t
        2. d_growth: From BFS volume scaling
           V(r) ~ r^d_g for large r

        Args:
            adj_matrix: Adjacency matrix (N x N)

        Returns:
            dict: Contains:
                - 'd_spectral': Spectral dimension from heat kernel
                - 'd_growth': Growth dimension from BFS volume scaling
                - 'd_average': Average of the two dimensions
                - 'heat_kernel_data': (t_values, K_values) for diagnostics
                - 'volume_data': (r_values, V_values) for diagnostics
        """
        self._initialize_from_adjacency(adj_matrix)

        # Compute spectral dimension from heat kernel trace
        d_spectral, heat_data = self._compute_spectral_dimension()

        # Compute growth dimension from BFS volume scaling
        d_growth, volume_data = self._compute_growth_dimension()

        return {
            'd_spectral': float(d_spectral),
            'd_growth': float(d_growth),
            'd_average': float((d_spectral + d_growth) / 2),
            'heat_kernel_data': heat_data,
            'volume_data': volume_data
        }

    def _compute_spectral_dimension(self):
        """
        Compute spectral dimension from heat kernel trace.

        The heat kernel K(t) = Tr(exp(-tL)) has asymptotic behavior:
        K(t) ~ t^(-d_s/2) for small t
        
        Taking log: log(K(t)) ~ -(d_s/2) * log(t)
        So d_s = -2 * slope in log-log plot.

        Returns:
            tuple: (d_spectral, (t_values, K_values))
        """
        if self.eigenvalues is None:
            return 0.0, (np.array([]), np.array([]))

        # Filter out zero eigenvalues (connected components)
        nonzero_eigs = self.eigenvalues[self.eigenvalues > 1e-10]

        if len(nonzero_eigs) < 2:
            # Not enough eigenvalues for reliable estimate
            return 2.0, (np.array([]), np.array([]))

        # Time range for heat kernel (use small t regime)
        # t should be small enough that K(t) ~ t^(-d_s/2) holds
        lambda_max = np.max(nonzero_eigs)
        lambda_min = np.min(nonzero_eigs)

        # Use times in the range where heat kernel behavior is well-defined
        t_min = 0.01 / lambda_max
        t_max = 1.0 / lambda_min
        t_values = np.logspace(np.log10(t_min), np.log10(t_max), 50)

        # Compute heat kernel trace: K(t) = sum_i exp(-lambda_i * t)
        K_values = np.zeros_like(t_values)
        for i, t in enumerate(t_values):
            K_values[i] = np.sum(np.exp(-self.eigenvalues * t))

        # Log-log fit to extract spectral dimension
        # log(K) = -(d_s/2) * log(t) + const
        log_t = np.log(t_values)
        log_K = np.log(K_values)

        # Filter valid values (avoid NaN/Inf)
        valid = np.isfinite(log_t) & np.isfinite(log_K)
        if np.sum(valid) < 5:
            return 2.0, (t_values, K_values)

        log_t = log_t[valid]
        log_K = log_K[valid]

        # Linear regression: log_K = slope * log_t + intercept
        try:
            coeffs = np.polyfit(log_t, log_K, 1)
            slope = coeffs[0]
            d_spectral = -2.0 * slope
        except (np.linalg.LinAlgError, ValueError):
            d_spectral = 2.0

        # Clamp to reasonable range [1, 10]
        d_spectral = np.clip(d_spectral, 1.0, 10.0)

        return d_spectral, (t_values, K_values)

    def _compute_growth_dimension(self):
        """
        Compute growth dimension from BFS volume scaling.

        For a d-dimensional lattice, the volume at distance r scales as:
        V(r) ~ r^d

        Taking log: log(V) ~ d * log(r)

        Returns:
            tuple: (d_growth, (r_values, V_values))
        """
        if self.adj_matrix is None:
            return 2.0, (np.array([]), np.array([]))

        N = self.N

        # Sample multiple starting nodes for averaging
        n_samples = min(10, N)
        sample_nodes = np.random.choice(N, n_samples, replace=False)

        # Maximum radius to consider
        max_radius = int(np.sqrt(N))

        # Aggregate volume profiles
        all_r = []
        all_V = []

        for start in sample_nodes:
            r_vals, V_vals = self._bfs_volume_profile(start, max_radius)
            if len(r_vals) > 1:
                all_r.extend(r_vals[1:])  # Skip r=0
                all_V.extend(V_vals[1:])

        if len(all_r) < 3:
            return 2.0, (np.array([]), np.array([]))

        r_values = np.array(all_r, dtype=float)
        V_values = np.array(all_V, dtype=float)

        # Log-log fit
        log_r = np.log(r_values)
        log_V = np.log(V_values)

        valid = np.isfinite(log_r) & np.isfinite(log_V)
        if np.sum(valid) < 3:
            return 2.0, (r_values, V_values)

        log_r = log_r[valid]
        log_V = log_V[valid]

        try:
            coeffs = np.polyfit(log_r, log_V, 1)
            d_growth = coeffs[0]
        except (np.linalg.LinAlgError, ValueError):
            d_growth = 2.0

        # Clamp to reasonable range [1, 10]
        d_growth = np.clip(d_growth, 1.0, 10.0)

        return d_growth, (r_values, V_values)

    def _bfs_volume_profile(self, start, max_radius):
        """
        Compute BFS volume profile from a starting node.

        Args:
            start: Starting node index
            max_radius: Maximum BFS radius

        Returns:
            tuple: (r_values, V_values) where V[i] is the number of
                   nodes within distance r[i] of start
        """
        N = self.N
        visited = np.zeros(N, dtype=bool)
        distance = np.full(N, -1)

        queue = deque([start])
        visited[start] = True
        distance[start] = 0

        r_values = [0]
        V_values = [1]  # V(0) = 1 (just the starting node)

        while queue:
            node = queue.popleft()
            current_dist = distance[node]

            if current_dist >= max_radius:
                continue

            # Find neighbors
            neighbors = np.where(self.adj_matrix[node] > 0)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    distance[neighbor] = current_dist + 1
                    queue.append(neighbor)

                    # Update volume at this radius
                    r = distance[neighbor]
                    if r > r_values[-1]:
                        r_values.append(r)
                        V_values.append(np.sum(distance <= r))
                    else:
                        # Update cumulative count
                        V_values[-1] = np.sum((distance >= 0) & (distance <= r))

        return np.array(r_values), np.array(V_values)

    def compute_sote_penalty(self, d_spectral, d_growth, d_volume=None):
        """
        Compute SOTE (Self-Organizing Topological Entropy) penalty.

        The SOTE principle requires dimension consistency: all intrinsic
        dimension measures should agree. The penalty functional is:
        
        P = sum_i,j (d_i - d_j)^2

        This penalty is minimized when all dimensions converge to the
        same value (ideally d = 4 for 4D spacetime emergence).

        Args:
            d_spectral: Spectral dimension from heat kernel
            d_growth: Growth dimension from BFS volume scaling
            d_volume: Optional third dimension estimate (e.g., from 
                     Hausdorff measure). If None, only uses d_spectral
                     and d_growth.

        Returns:
            float: The SOTE penalty value. Zero indicates perfect
                  dimension consistency.
        """
        dimensions = [d_spectral, d_growth]
        if d_volume is not None:
            dimensions.append(d_volume)

        # Compute sum of squared differences
        penalty = 0.0
        n = len(dimensions)
        for i in range(n):
            for j in range(i + 1, n):
                penalty += (dimensions[i] - dimensions[j]) ** 2

        return float(penalty)


if __name__ == "__main__":
    # Verification test for Dimensional Bootstrap
    print("=" * 60)
    print("Dimensional Bootstrap Verification")
    print("=" * 60)
    
    # Test 1: 2D grid (4x4)
    print("\nTest 1: 2D grid (4x4 = 16 nodes)")
    N = 16
    adj = np.zeros((N, N))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if j < 3:
                adj[idx, idx + 1] = 1
                adj[idx + 1, idx] = 1
            if i < 3:
                adj[idx, idx + 4] = 1
                adj[idx + 4, idx] = 1
    
    bootstrap = Dimensional_Bootstrap()
    result = bootstrap.compute_intrinsic_dims(adj)
    print(f"  Spectral dimension: {result['d_spectral']:.4f}")
    print(f"  Growth dimension: {result['d_growth']:.4f}")
    print(f"  Average dimension: {result['d_average']:.4f}")
    
    penalty = bootstrap.compute_sote_penalty(result['d_spectral'], result['d_growth'])
    print(f"  SOTE penalty: {penalty:.6f}")
    
    # Test 2: 1D cycle
    print("\nTest 2: 1D cycle (20 nodes)")
    N = 20
    adj = np.zeros((N, N))
    for i in range(N):
        adj[i, (i + 1) % N] = 1
        adj[(i + 1) % N, i] = 1
    
    result = bootstrap.compute_intrinsic_dims(adj)
    print(f"  Spectral dimension: {result['d_spectral']:.4f} (expected ~1)")
    print(f"  Growth dimension: {result['d_growth']:.4f}")
    
    # Test 3: 4D grid (2x2x2x2 = 16 nodes)
    print("\nTest 3: 4D grid (2x2x2x2 = 16 nodes)")
    N = 16
    adj = np.zeros((N, N))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    idx = i + 2*j + 4*k + 8*l
                    if i < 1:
                        neighbor = (i+1) + 2*j + 4*k + 8*l
                        adj[idx, neighbor] = 1
                        adj[neighbor, idx] = 1
                    if j < 1:
                        neighbor = i + 2*(j+1) + 4*k + 8*l
                        adj[idx, neighbor] = 1
                        adj[neighbor, idx] = 1
                    if k < 1:
                        neighbor = i + 2*j + 4*(k+1) + 8*l
                        adj[idx, neighbor] = 1
                        adj[neighbor, idx] = 1
                    if l < 1:
                        neighbor = i + 2*j + 4*k + 8*(l+1)
                        adj[idx, neighbor] = 1
                        adj[neighbor, idx] = 1
    
    result = bootstrap.compute_intrinsic_dims(adj)
    print(f"  Spectral dimension: {result['d_spectral']:.4f}")
    print(f"  Growth dimension: {result['d_growth']:.4f}")
    print(f"  Average dimension: {result['d_average']:.4f}")
    
    # Test 4: SOTE penalty gradient
    print("\nTest 4: SOTE penalty gradient")
    penalties = []
    for d in [2.0, 3.0, 4.0, 5.0]:
        p = bootstrap.compute_sote_penalty(4.0, d)
        penalties.append((d, p))
        print(f"  d_growth={d:.1f}, d_spectral=4.0 -> penalty={p:.4f}")
    
    print("\n" + "=" * 60)
    print("Dimensional Bootstrap Verification Complete")
    print("=" * 60)
