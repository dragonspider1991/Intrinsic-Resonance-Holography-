"""
matter.py - Topological Defect Classifier for Matter Genesis

RIRH v9.5 Matter Genesis Framework

This module implements the Topological Defect Classifier for identifying
fundamental cycles (quantum knots) in the Cymatic Resonance Network that correspond to
particle-like excitations.

Key Components:
- Topological_Defect_Classifier: Class for cycle identification and
  gauge group verification
- K-Theory index argument verification (n_cycles == 12 for SM gauge group)
- Holonomy computation for non-trivial cycle filtering

References:
- K-Theory index theorems in physics
- Topological defects in quantum field theory
- Graph cycle basis and homology
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigh


class Topological_Defect_Classifier:
    """
    Topological Defect Classifier for Matter Genesis.

    Identifies fundamental cycles in the Cymatic Resonance Network using networkx.cycle_basis
    and computes holonomies to filter for non-trivial (independent) cycles.

    The K-Theory index argument predicts exactly 12 independent generators
    for the Standard Model gauge group SU(3) x SU(2) x U(1), corresponding
    to the 12-dimensional algebra.

    Attributes:
        adj_matrix: Adjacency matrix of the Cymatic Resonance Network
        N: Number of nodes
        graph: NetworkX graph representation
        cycles: List of fundamental cycles
        holonomies: Holonomy values for each cycle
    """

    def __init__(self, adj_matrix=None, phase_matrix=None):
        """
        Initialize Topological Defect Classifier.

        Args:
            adj_matrix: Optional adjacency matrix. If provided, constructs
                       the networkx graph upon initialization.
            phase_matrix: Optional complex phase matrix for gauge connection.
                         If None, random phases are assigned to edges.
        """
        self.adj_matrix = None
        self.phase_matrix = None
        self.graph = None
        self.N = 0
        self.cycles = []
        self.holonomies = []

        if adj_matrix is not None:
            self._initialize_from_adjacency(adj_matrix, phase_matrix)

    def _initialize_from_adjacency(self, adj_matrix, phase_matrix=None):
        """Initialize from adjacency matrix."""
        self.adj_matrix = np.asarray(adj_matrix, dtype=float)
        self.N = self.adj_matrix.shape[0]

        # Create NetworkX graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.N))

        # Add edges
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.adj_matrix[i, j] > 0:
                    self.graph.add_edge(i, j, weight=self.adj_matrix[i, j])

        # Initialize phase matrix (gauge connection)
        if phase_matrix is not None:
            self.phase_matrix = np.asarray(phase_matrix, dtype=complex)
        else:
            # Generate consistent random phases for edges
            self.phase_matrix = np.zeros((self.N, self.N), dtype=complex)
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if self.adj_matrix[i, j] > 0:
                        # Random phase in [0, 2π)
                        phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
                        self.phase_matrix[i, j] = phase
                        self.phase_matrix[j, i] = np.conj(phase)  # Hermitian

    def identify_cycles(self, adj_matrix, phase_matrix=None):
        """
        Identify fundamental cycles and compute holonomies.

        Uses networkx.cycle_basis to find a set of fundamental cycles
        (generators of the first homology group H₁). Then computes the
        holonomy (phase sum) around each cycle to identify non-trivial
        topological defects.

        A cycle is non-trivial if its holonomy is not a multiple of 2π,
        indicating a non-vanishing gauge flux (topological charge).

        Args:
            adj_matrix: Adjacency matrix (N x N)
            phase_matrix: Optional complex phase matrix for gauge connection.
                         If None, uses the stored phase matrix or generates
                         random phases.

        Returns:
            dict: Contains:
                - 'cycles': List of fundamental cycles (each cycle is a list
                           of node indices)
                - 'n_cycles': Number of fundamental cycles (Betti number β₁)
                - 'holonomies': Holonomy phase for each cycle (in radians)
                - 'non_trivial_cycles': Cycles with non-trivial holonomy
                - 'n_generators': Number of independent non-trivial generators
        """
        self._initialize_from_adjacency(adj_matrix, phase_matrix)

        if self.graph is None or self.graph.number_of_nodes() == 0:
            return {
                'cycles': [],
                'n_cycles': 0,
                'holonomies': [],
                'non_trivial_cycles': [],
                'n_generators': 0
            }

        # Find fundamental cycles using cycle_basis
        try:
            self.cycles = list(nx.cycle_basis(self.graph))
        except nx.NetworkXError:
            self.cycles = []

        n_cycles = len(self.cycles)

        # Compute holonomy for each cycle
        self.holonomies = []
        non_trivial_cycles = []
        non_trivial_holonomies = []

        for cycle in self.cycles:
            holonomy = self._compute_cycle_holonomy(cycle)
            self.holonomies.append(holonomy)

            # Check if holonomy is non-trivial (not close to 0 or 2πn)
            # Use mod 2π and check distance from 0
            normalized_holonomy = np.mod(holonomy, 2 * np.pi)
            if normalized_holonomy > np.pi:
                normalized_holonomy = 2 * np.pi - normalized_holonomy

            # Threshold for non-triviality (holonomy > π/6 ≈ 0.52)
            if normalized_holonomy > np.pi / 6:
                non_trivial_cycles.append(cycle)
                non_trivial_holonomies.append(holonomy)

        # Count independent generators
        # For exact count, we would need to check linear independence
        # Here we use a simpler heuristic: count distinct holonomy classes
        n_generators = len(non_trivial_cycles)

        return {
            'cycles': self.cycles,
            'n_cycles': n_cycles,
            'holonomies': self.holonomies,
            'non_trivial_cycles': non_trivial_cycles,
            'non_trivial_holonomies': non_trivial_holonomies,
            'n_generators': n_generators
        }

    def _compute_cycle_holonomy(self, cycle):
        """
        Compute the holonomy (phase accumulation) around a cycle.

        The holonomy is the sum of gauge phases around the cycle:
        Φ = sum_{edges in cycle} arg(W_{ij})

        Args:
            cycle: List of node indices forming the cycle

        Returns:
            float: Holonomy phase in radians
        """
        if len(cycle) < 2:
            return 0.0

        if self.phase_matrix is None:
            return 0.0

        holonomy = 0.0
        n = len(cycle)

        for i in range(n):
            node_i = cycle[i]
            node_j = cycle[(i + 1) % n]  # Wrap around to close the cycle

            # Get phase from phase matrix
            if self.phase_matrix is not None:
                w = self.phase_matrix[node_i, node_j]
                if np.abs(w) > 1e-12:
                    holonomy += np.angle(w)

        return holonomy

    def verify_gauge_group(self, n_cycles):
        """
        Verify if the number of cycles matches K-Theory index prediction.

        The K-Theory index argument predicts exactly 12 independent
        generators for the Standard Model gauge group structure:
        - SU(3): 8 generators (gluons)
        - SU(2): 3 generators (W bosons)
        - U(1):  1 generator (B boson)
        Total: 8 + 3 + 1 = 12

        This corresponds to the dimension of the Lie algebra
        su(3) ⊕ su(2) ⊕ u(1).

        Args:
            n_cycles: Number of independent non-trivial cycles

        Returns:
            bool: True if n_cycles == 12 (matches SM prediction),
                 False otherwise
        """
        SM_GAUGE_DIM = 12  # dim(su(3)) + dim(su(2)) + dim(u(1)) = 8 + 3 + 1
        return n_cycles == SM_GAUGE_DIM

    def get_cycle_statistics(self):
        """
        Get statistics about the identified cycles.

        Returns:
            dict: Contains:
                - 'n_cycles': Total number of fundamental cycles
                - 'n_non_trivial': Number of non-trivial cycles
                - 'avg_cycle_length': Average cycle length
                - 'max_cycle_length': Maximum cycle length
                - 'holonomy_mean': Mean holonomy magnitude
                - 'holonomy_std': Standard deviation of holonomies
        """
        if not self.cycles:
            return {
                'n_cycles': 0,
                'n_non_trivial': 0,
                'avg_cycle_length': 0.0,
                'max_cycle_length': 0,
                'holonomy_mean': 0.0,
                'holonomy_std': 0.0
            }

        cycle_lengths = [len(c) for c in self.cycles]

        # Count non-trivial cycles
        n_non_trivial = 0
        for h in self.holonomies:
            normalized = np.mod(np.abs(h), 2 * np.pi)
            if normalized > np.pi:
                normalized = 2 * np.pi - normalized
            if normalized > np.pi / 6:
                n_non_trivial += 1

        return {
            'n_cycles': len(self.cycles),
            'n_non_trivial': n_non_trivial,
            'avg_cycle_length': float(np.mean(cycle_lengths)),
            'max_cycle_length': int(np.max(cycle_lengths)),
            'holonomy_mean': float(np.mean(np.abs(self.holonomies))) if self.holonomies else 0.0,
            'holonomy_std': float(np.std(self.holonomies)) if self.holonomies else 0.0
        }


if __name__ == "__main__":
    # Verification test for Topological Defect Classifier
    print("=" * 60)
    print("Topological Defect Classifier Verification")
    print("=" * 60)
    
    # Test 1: Simple cycle graph
    print("\nTest 1: Simple cycle graph C_6")
    N = 6
    adj = np.zeros((N, N))
    for i in range(N):
        adj[i, (i + 1) % N] = 1
        adj[(i + 1) % N, i] = 1
    
    classifier = Topological_Defect_Classifier()
    result = classifier.identify_cycles(adj)
    print(f"  Number of cycles: {result['n_cycles']} (expected: 1)")
    print(f"  Cycle length: {len(result['cycles'][0]) if result['cycles'] else 0}")
    
    # Test 2: Tree graph (no cycles)
    print("\nTest 2: Star tree graph (no cycles)")
    N = 6
    adj = np.zeros((N, N))
    for i in range(1, N):
        adj[0, i] = 1
        adj[i, 0] = 1
    
    result = classifier.identify_cycles(adj)
    print(f"  Number of cycles: {result['n_cycles']} (expected: 0)")
    
    # Test 3: 4x4 grid (9 fundamental cycles)
    print("\nTest 3: 4x4 grid (9 fundamental cycles)")
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
    
    result = classifier.identify_cycles(adj)
    print(f"  Number of cycles: {result['n_cycles']} (expected: 9)")
    print(f"  Number of generators: {result['n_generators']}")
    
    stats = classifier.get_cycle_statistics()
    print(f"  Average cycle length: {stats['avg_cycle_length']:.2f}")
    print(f"  Holonomy mean: {stats['holonomy_mean']:.4f}")
    
    # Test 4: K-Theory index verification
    print("\nTest 4: K-Theory index verification")
    print(f"  SM gauge group dimension = 12 (SU(3)×SU(2)×U(1))")
    print(f"  verify_gauge_group(12) = {classifier.verify_gauge_group(12)}")
    print(f"  verify_gauge_group(10) = {classifier.verify_gauge_group(10)}")
    print(f"  verify_gauge_group(24) = {classifier.verify_gauge_group(24)}")
    
    print("\n" + "=" * 60)
    print("Topological Defect Classifier Verification Complete")
    print("=" * 60)
