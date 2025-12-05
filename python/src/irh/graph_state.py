"""
graph_state.py - Cymatic Resonance Network Substrate for Intrinsic Resonance Holography

This module implements the Cymatic Resonance Network substrate that forms
the foundational data structure for discrete quantum spacetime.

A CymaticResonanceNetwork encapsulates:
- Vertex set V = {0, 1, ..., N-1}
- Hyperedge set E (k-tuples for k <= 4)
- Complex weights W: E -> C with |W| in [0,1] and arg(W) in [0, 2π)
- L_U self-consistency derivation

Equations Implemented:
- Random Cymatic Resonance Network generation via Erdős-Rényi model
- Complex weight initialization: w = |w| * exp(i*φ)
- L_U = L_G * sqrt(hbar_G * G_N) (stub numericals from ARO)
- Holographic bound enforcement: S <= A/4 (with Lagrange multiplier)

References:
- IRH Theory: Discrete Cymatic Resonance Network as quantum spacetime substrate
- Holographic principle and entropy bounds
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray


@dataclass
class CymaticResonanceNetworkMetadata:
    """Metadata for CymaticResonanceNetwork creation and tracking."""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    seed: int | None = None
    precision: int = 50
    topology: str = "Random"
    edge_probability: float = 0.3
    weight_distribution: str = "Uniform"
    phase_distribution: str = "Uniform"
    version: str = "10.0"


class CymaticResonanceNetwork:
    """
    Cymatic Resonance Network representing discrete quantum spacetime.

    This class implements a network with:
    - N nodes (vertices)
    - Hyperedges as k-tuples (k <= 4) with complex weights
    - NetworkX HyperDiGraph backend for standard graph operations

    Attributes:
        N: Number of nodes
        V: Vertex set {0, 1, ..., N-1}
        E: List of hyperedges (tuples)
        W: Dictionary mapping edges to complex weights
        interference_matrix: Interference Matrix (Graph Laplacian ℒ)
        phases: Phase factors on edges (anti-symmetric)
        metadata: Creation metadata

    Physical Constants (stub values from ARO):
        hbar_G: Reduced Planck constant in geometric units
        G_N: Newton's gravitational constant
        L_G: Graph length scale
    """

    # Physical constants (stub values - to be derived from ARO)
    hbar_G: float = 1.054571817e-34  # J·s (Planck's reduced constant)
    G_N: float = 6.67430e-11  # m³/(kg·s²) (Newton's constant)
    L_G: float = 1.616255e-35  # m (Planck length scale)

    def __init__(
        self,
        N: int,
        edges: list[tuple[int, ...]] | None = None,
        weights: dict[tuple[int, ...], complex] | None = None,
        seed: int | None = None,
        topology: str = "Random",
        edge_probability: float = 0.3,
        weight_distribution: str = "Uniform",
        phase_distribution: str = "Uniform",
    ) -> None:
        """
        Initialize a CymaticResonanceNetwork.

        Args:
            N: Number of nodes (must be >= 2)
            edges: Optional list of hyperedges (k-tuples, k <= 4)
            weights: Optional dict mapping edges to complex weights
            seed: Random seed for reproducibility
            topology: Initial topology type ("Random", "Complete", "Cycle", "Lattice")
            edge_probability: Probability for random edges (Erdős-Rényi)
            weight_distribution: "Uniform" or "Gaussian" for weight initialization
            phase_distribution: "Uniform" for phase initialization
        """
        if N < 2:
            raise ValueError(f"Node count {N} is too small. Minimum is 2.")
        if N > 10000:
            raise ValueError(f"Node count {N} is too large. Maximum is 10000.")

        self.N = N
        self.V = list(range(N))
        self.E: list[tuple[int, ...]] = []
        self.W: dict[tuple[int, ...], complex] = {}

        # Set random seed for reproducibility
        self._rng = np.random.default_rng(seed)

        # Initialize metadata
        self.metadata = CymaticResonanceNetworkMetadata(
            seed=seed,
            topology=topology,
            edge_probability=edge_probability,
            weight_distribution=weight_distribution,
            phase_distribution=phase_distribution,
        )

        # Generate or set edges and weights
        if edges is not None and weights is not None:
            self.E = list(edges)
            self.W = dict(weights)
            self._validate_edges()
        else:
            self._generate_topology(topology, edge_probability)
            self._generate_weights(weight_distribution)
            self._generate_phases(phase_distribution)

        # Build matrices
        self._build_matrices()

    def _generate_topology(self, topology: str, p: float) -> None:
        """Generate initial topology based on type."""
        if topology == "Random":
            self._generate_random(p)
        elif topology == "Complete":
            self._generate_complete()
        elif topology == "Cycle":
            self._generate_cycle()
        elif topology == "Lattice":
            self._generate_lattice()
        else:
            raise ValueError(f"Unknown topology type: {topology}")

    def _generate_random(self, p: float) -> None:
        """Generate Erdős-Rényi random graph."""
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self._rng.random() < p:
                    edge = (i, j)
                    self.E.append(edge)
                    self.W[edge] = complex(1.0, 0.0)  # Placeholder

    def _generate_complete(self) -> None:
        """Generate complete graph K_N."""
        for i in range(self.N):
            for j in range(i + 1, self.N):
                edge = (i, j)
                self.E.append(edge)
                self.W[edge] = complex(1.0, 0.0)

    def _generate_cycle(self) -> None:
        """Generate cycle graph C_N."""
        for i in range(self.N):
            edge = (i, (i + 1) % self.N)
            # Normalize edge ordering
            edge = tuple(sorted(edge))
            if edge not in self.W:
                self.E.append(edge)
                self.W[edge] = complex(1.0, 0.0)

    def _generate_lattice(self) -> None:
        """Generate 2D lattice graph."""
        side = int(np.floor(np.sqrt(self.N)))
        for i in range(side):
            for j in range(side):
                idx1 = i * side + j
                if idx1 >= self.N:
                    continue
                # Right neighbor
                if j < side - 1:
                    idx2 = i * side + (j + 1)
                    if idx2 < self.N:
                        edge = tuple(sorted([idx1, idx2]))
                        if edge not in self.W:
                            self.E.append(edge)
                            self.W[edge] = complex(1.0, 0.0)
                # Bottom neighbor
                if i < side - 1:
                    idx2 = (i + 1) * side + j
                    if idx2 < self.N:
                        edge = tuple(sorted([idx1, idx2]))
                        if edge not in self.W:
                            self.E.append(edge)
                            self.W[edge] = complex(1.0, 0.0)

    def _generate_weights(self, distribution: str) -> None:
        """Generate complex weights for edges."""
        for edge in self.E:
            if distribution == "Uniform":
                magnitude = self._rng.uniform(0.1, 1.0)
            elif distribution == "Gaussian":
                magnitude = max(0.01, self._rng.normal(0.5, 0.2))
            else:
                magnitude = self._rng.uniform(0.1, 1.0)

            # Clamp magnitude to [0, 1]
            magnitude = np.clip(magnitude, 0.0, 1.0)
            self.W[edge] = complex(magnitude, 0.0)

    def _generate_phases(self, distribution: str) -> None:
        """Generate phase factors for edges."""
        for edge in self.E:
            if distribution == "Uniform":
                phase = self._rng.uniform(0, 2 * np.pi)
            else:
                phase = self._rng.uniform(0, 2 * np.pi)

            magnitude = abs(self.W[edge])
            self.W[edge] = magnitude * np.exp(1j * phase)

    def _validate_edges(self) -> None:
        """Validate edge format and weights."""
        for edge in self.E:
            if not isinstance(edge, tuple):
                raise ValueError(f"Edge {edge} must be a tuple")
            if len(edge) > 4:
                raise ValueError(f"Hyperedge {edge} has more than 4 nodes")
            if any(v < 0 or v >= self.N for v in edge):
                raise ValueError(f"Edge {edge} contains invalid node index")
            if edge not in self.W:
                self.W[edge] = complex(1.0, 0.0)

    def _build_matrices(self) -> None:
        """Build coupling matrix and phase matrix from edges."""
        self.adjacency_matrix: NDArray[np.float64] = np.zeros(
            (self.N, self.N), dtype=np.float64
        )
        self._weights_matrix: NDArray[np.float64] = np.zeros(
            (self.N, self.N), dtype=np.float64
        )
        self._phases_matrix: NDArray[np.float64] = np.zeros(
            (self.N, self.N), dtype=np.float64
        )

        for edge, weight in self.W.items():
            if len(edge) == 2:
                i, j = edge
                magnitude = abs(weight)
                phase = np.angle(weight)

                self.adjacency_matrix[i, j] = 1.0
                self.adjacency_matrix[j, i] = 1.0
                self._weights_matrix[i, j] = magnitude
                self._weights_matrix[j, i] = magnitude
                # Phases are anti-symmetric: φ_ji = -φ_ij
                self._phases_matrix[i, j] = phase
                self._phases_matrix[j, i] = -phase

    def add_hyperedge(
        self, nodes: tuple[int, ...], weight: complex = complex(1, 0)
    ) -> None:
        """
        Add a hyperedge to the graph.

        Args:
            nodes: Tuple of node indices (k-tuple, k <= 4)
            weight: Complex weight with |weight| in [0, 1]
        """
        if len(nodes) > 4:
            raise ValueError("Hyperedges can have at most 4 nodes")
        if any(v < 0 or v >= self.N for v in nodes):
            raise ValueError("Invalid node index in edge")
        if abs(weight) > 1.0:
            raise ValueError("Weight magnitude must be in [0, 1]")

        # Normalize edge representation
        edge = tuple(sorted(nodes)) if len(nodes) == 2 else nodes

        if edge not in self.W:
            self.E.append(edge)

        self.W[edge] = weight
        self._build_matrices()

    def validate_substrate(self) -> bool:
        """
        Validate the Cymatic Resonance Network substrate.

        Checks:
        - Finiteness: N < infinity
        - Dimensionless: All weights normalized
        - Symmetry: Adjacency is symmetric
        - Anti-symmetry: Phases are anti-symmetric

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check finiteness
        if self.N <= 0 or self.N > 10000:
            raise ValueError("Graph must have finite, bounded node count")

        # Check weight normalization
        for edge, weight in self.W.items():
            if abs(weight) > 1.0:
                raise ValueError(f"Weight magnitude > 1 for edge {edge}")

        # Check matrix symmetry
        if not np.allclose(
            self.adjacency_matrix, self.adjacency_matrix.T, atol=1e-10
        ):
            raise ValueError("Adjacency matrix must be symmetric")

        if not np.allclose(
            self._weights_matrix, self._weights_matrix.T, atol=1e-10
        ):
            raise ValueError("Weight matrix must be symmetric")

        # Check phase anti-symmetry
        if not np.allclose(
            self._phases_matrix, -self._phases_matrix.T, atol=1e-10
        ):
            raise ValueError("Phase matrix must be anti-symmetric")

        return True

    def derive_lu(self) -> float:
        """
        Derive L_U self-consistency scale.

        From ARO: L_U = L_G * sqrt(hbar_G * G_N)

        Returns:
            L_U scale factor
        """
        return self.L_G * np.sqrt(self.hbar_G * self.G_N)

    def enforce_holography(self, lambda_holo: float = 1.0) -> float:
        """
        Enforce holographic bound constraint.

        Adds Lagrange term to action when S_bulk > A/4.

        Args:
            lambda_holo: Lagrange multiplier strength

        Returns:
            Holographic penalty term
        """
        # Compute bulk entropy (proxy: graph entropy)
        s_bulk = self._compute_bulk_entropy()

        # Compute boundary area (proxy: boundary edge count)
        a_boundary = self._compute_boundary_area()

        # Holographic bound: S_bulk <= A/4
        bound = a_boundary / 4.0

        if s_bulk > bound:
            penalty = lambda_holo * (s_bulk - bound) ** 2
        else:
            penalty = 0.0

        return penalty

    def _compute_bulk_entropy(self) -> float:
        """Compute proxy for bulk entropy from eigenvalue distribution."""
        laplacian = self.get_laplacian()
        eigenvalues = np.linalg.eigvalsh(laplacian)

        # Filter positive eigenvalues
        pos_eigs = eigenvalues[eigenvalues > 1e-10]
        if len(pos_eigs) == 0:
            return 0.0

        # Normalize to probability distribution
        p = pos_eigs / np.sum(pos_eigs)

        # Shannon entropy
        entropy = -np.sum(p * np.log2(p + 1e-15))
        return float(entropy)

    def _compute_boundary_area(self) -> float:
        """Compute proxy for boundary area from edge count."""
        # Use total edge weight as proxy for area
        return float(np.sum(self._weights_matrix) / 2)

    def get_laplacian(self) -> NDArray[np.float64]:
        """
        Compute the Interference Matrix (Graph Laplacian ℒ = D - A).

        Returns:
            Interference Matrix
        """
        degrees = np.sum(self.adjacency_matrix, axis=1)
        D = np.diag(degrees)
        return D - self.adjacency_matrix

    def get_weighted_laplacian(self) -> NDArray[np.complex128]:
        """
        Compute the weighted Laplacian with complex weights.

        L_W[i,j] = -W[i,j] for i != j
        L_W[i,i] = sum_j |W[i,j]|

        Returns:
            Complex weighted Laplacian matrix
        """
        L = np.zeros((self.N, self.N), dtype=np.complex128)

        for edge, weight in self.W.items():
            if len(edge) == 2:
                i, j = edge
                L[i, j] = -weight
                L[j, i] = -np.conj(weight)

        # Diagonal: sum of edge magnitudes
        for i in range(self.N):
            L[i, i] = np.sum(np.abs(L[i, :]))

        return L

    def to_networkx(self) -> nx.Graph:
        """
        Convert to NetworkX Graph.

        Returns:
            NetworkX graph with edge weights
        """
        G = nx.Graph()
        G.add_nodes_from(self.V)

        for edge, weight in self.W.items():
            if len(edge) == 2:
                i, j = edge
                G.add_edge(i, j, weight=abs(weight), phase=np.angle(weight))

        return G

    def to_networkx_digraph(self) -> nx.DiGraph:
        """
        Convert to NetworkX DiGraph for directed operations.

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.V)

        for edge, weight in self.W.items():
            if len(edge) == 2:
                i, j = edge
                G.add_edge(i, j, weight=weight)
                G.add_edge(j, i, weight=np.conj(weight))

        return G

    @property
    def edge_count(self) -> int:
        """Return number of edges."""
        return len(self.E)

    @property
    def node_count(self) -> int:
        """Return number of nodes."""
        return self.N

    def save(self, filepath: str | Path) -> None:
        """
        Save CymaticResonanceNetwork to JSON file.

        Args:
            filepath: Path to save file
        """
        data = {
            "type": "CymaticResonanceNetwork",
            "version": self.metadata.version,
            "N": self.N,
            "edges": [list(e) for e in self.E],
            "weights_real": {str(k): v.real for k, v in self.W.items()},
            "weights_imag": {str(k): v.imag for k, v in self.W.items()},
            "metadata": {
                "created_at": self.metadata.created_at,
                "seed": self.metadata.seed,
                "topology": self.metadata.topology,
                "edge_probability": self.metadata.edge_probability,
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> CymaticResonanceNetwork:
        """
        Load CymaticResonanceNetwork from JSON file.

        Args:
            filepath: Path to saved file

        Returns:
            Loaded CymaticResonanceNetwork
        """
        with open(filepath) as f:
            data = json.load(f)

        edges = [tuple(e) for e in data["edges"]]
        weights = {}
        for k, v_real in data["weights_real"].items():
            v_imag = data["weights_imag"][k]
            weights[eval(k)] = complex(v_real, v_imag)

        graph = cls(
            N=data["N"],
            edges=edges,
            weights=weights,
            seed=data["metadata"].get("seed"),
        )

        return graph

    def __repr__(self) -> str:
        return (
            f"CymaticResonanceNetwork(N={self.N}, edges={len(self.E)}, "
            f"topology={self.metadata.topology})"
        )


def create_graph_state(
    n: int,
    seed: int | None = None,
    topology: str = "Random",
    edge_probability: float = 0.3,
) -> CymaticResonanceNetwork:
    """
    Convenience function to create a CymaticResonanceNetwork.

    This mirrors the Wolfram CreateGraphState function.

    Args:
        n: Number of nodes
        seed: Random seed
        topology: Graph topology type
        edge_probability: Edge probability for random graphs

    Returns:
        CymaticResonanceNetwork instance
    """
    return CymaticResonanceNetwork(
        N=n,
        seed=seed,
        topology=topology,
        edge_probability=edge_probability,
    )
