"""
IRH Suite v9.2 Unit Tests

Comprehensive test suite for all Python modules.
"""

import numpy as np
import pytest

# Test fixtures


@pytest.fixture
def small_graph():
    """Create a small test graph."""
    from irh.graph_state import HyperGraph

    return HyperGraph(N=16, seed=42, topology="Random", edge_probability=0.3)


@pytest.fixture
def lattice_graph():
    """Create a lattice test graph."""
    from irh.graph_state import HyperGraph

    return HyperGraph(N=16, seed=42, topology="Lattice")


@pytest.fixture
def complete_graph():
    """Create a complete graph."""
    from irh.graph_state import HyperGraph

    return HyperGraph(N=8, seed=42, topology="Complete")


@pytest.fixture
def cycle_graph():
    """Create a cycle graph."""
    from irh.graph_state import HyperGraph

    return HyperGraph(N=10, seed=42, topology="Cycle")


# GraphState Tests


class TestGraphState:
    """Tests for graph_state module."""

    def test_create_random_graph(self, small_graph):
        """Test random graph creation."""
        assert small_graph.N == 16
        assert small_graph.edge_count > 0
        assert len(small_graph.E) == small_graph.edge_count

    def test_create_complete_graph(self, complete_graph):
        """Test complete graph creation."""
        N = complete_graph.N
        expected_edges = N * (N - 1) // 2
        assert complete_graph.edge_count == expected_edges

    def test_create_cycle_graph(self, cycle_graph):
        """Test cycle graph creation."""
        assert cycle_graph.edge_count == cycle_graph.N

    def test_adjacency_symmetry(self, small_graph):
        """Test adjacency matrix is symmetric."""
        A = small_graph.adjacency_matrix
        assert np.allclose(A, A.T)

    def test_phase_antisymmetry(self, small_graph):
        """Test phase matrix is anti-symmetric."""
        P = small_graph._phases_matrix
        assert np.allclose(P, -P.T)

    def test_validate_substrate(self, small_graph):
        """Test substrate validation."""
        assert small_graph.validate_substrate() is True

    def test_laplacian(self, small_graph):
        """Test Laplacian computation."""
        L = small_graph.get_laplacian()
        assert L.shape == (small_graph.N, small_graph.N)
        # Laplacian should have row sums = 0
        assert np.allclose(np.sum(L, axis=1), 0, atol=1e-10)

    def test_add_hyperedge(self, small_graph):
        """Test adding a hyperedge."""
        initial_count = small_graph.edge_count
        small_graph.add_hyperedge((0, 1), weight=complex(0.5, 0.1))
        # Edge count may stay same if edge already exists
        assert small_graph.edge_count >= initial_count

    def test_save_load(self, small_graph, tmp_path):
        """Test save and load functionality."""
        filepath = tmp_path / "test_graph.irh"
        small_graph.save(filepath)

        from irh.graph_state import HyperGraph

        loaded = HyperGraph.load(filepath)
        assert loaded.N == small_graph.N
        assert loaded.edge_count == small_graph.edge_count


# SpectralDimension Tests


class TestSpectralDimension:
    """Tests for spectral_dimension module."""

    def test_heat_kernel_trace(self, small_graph):
        """Test heat kernel trace computation."""
        from irh.spectral_dimension import HeatKernelTrace

        P = HeatKernelTrace(small_graph, t=0.1)
        assert isinstance(P, float)
        assert P > 0

    def test_spectral_dimension(self, small_graph):
        """Test spectral dimension computation."""
        from irh.spectral_dimension import SpectralDimension

        result = SpectralDimension(small_graph)
        assert hasattr(result, "value")
        assert hasattr(result, "error")
        # Value should be positive (or NaN for small graphs)
        if not np.isnan(result.value):
            assert result.value > 0

    def test_dimensional_bootstrap(self, small_graph):
        """Test dimensional bootstrap test."""
        from irh.spectral_dimension import dimensional_bootstrap_test

        result = dimensional_bootstrap_test(small_graph, tolerance=1.0)
        assert "passed" in result
        assert "ds_value" in result


# ScalingFlows Tests


class TestScalingFlows:
    """Tests for scaling_flows module."""

    def test_gsrg_decimate(self, small_graph):
        """Test GSRG coarse-graining."""
        from irh.scaling_flows import GSRGDecimate

        result = GSRGDecimate(small_graph, scale=2)
        assert result.coarsened_n < small_graph.N
        assert result.decimated_modes > 0

    def test_metric_emergence(self, small_graph):
        """Test metric emergence."""
        from irh.scaling_flows import MetricEmergence

        result = MetricEmergence(small_graph)
        assert result.metric_tensor is not None
        assert hasattr(result, "signature")

    def test_lorentz_signature(self, small_graph):
        """Test Lorentz signature computation."""
        from irh.scaling_flows import LorentzSignature

        result = LorentzSignature(small_graph)
        assert result.negative_count >= 0
        assert result.positive_count >= 0


# GTEC Tests


class TestGTEC:
    """Tests for gtec module."""

    def test_gtec_computation(self, small_graph):
        """Test GTEC computation."""
        from irh.gtec import gtec

        result = gtec(small_graph)
        assert hasattr(result, "complexity")
        assert hasattr(result, "shannon_global")

    def test_shannon_entropy(self):
        """Test Shannon entropy function."""
        from irh.gtec import shannon_entropy

        # Uniform distribution over 4 outcomes: H = log2(4) = 2 bits
        P = np.array([0.25, 0.25, 0.25, 0.25])
        H = shannon_entropy(P)
        assert np.isclose(H, 2.0, atol=0.01)  # Expected: 2 bits


# NCGG Tests


class TestNCGG:
    """Tests for ncgg module."""

    def test_ncgg_operators(self, lattice_graph):
        """Test NCGG operator construction."""
        from irh.ncgg import NCGG

        ncgg = NCGG(lattice_graph)
        ops = ncgg.get_operators()
        assert len(ops.X) > 0
        assert len(ops.P) > 0

    def test_ccr_verification(self, lattice_graph):
        """Test CCR verification."""
        from irh.ncgg import NCGG

        ncgg = NCGG(lattice_graph)
        result = ncgg.verify_ccr(0, 0)
        assert hasattr(result, "passed")

    def test_frustration(self, small_graph):
        """Test frustration computation."""
        from irh.ncgg import frustration

        result = frustration(small_graph)
        assert hasattr(result, "total_frustration")


# DHGA/GSRG Tests


class TestDHGAGSRG:
    """Tests for dhga_gsrg module."""

    def test_discrete_homotopy(self, small_graph):
        """Test discrete homotopy computation."""
        from irh.dhga_gsrg import discrete_homotopy

        result = discrete_homotopy(small_graph)
        assert hasattr(result, "betti_1")
        assert result.betti_1 >= 0

    def test_vary_action_graph(self, small_graph):
        """Test graph action variation."""
        from irh.dhga_gsrg import vary_action_graph

        result = vary_action_graph(small_graph)
        assert hasattr(result, "ricci_scalar")
        assert hasattr(result, "einstein_tensor")


# Asymptotics Tests


class TestAsymptotics:
    """Tests for asymptotics module."""

    def test_newton_limit(self, small_graph):
        """Test Newtonian limit recovery."""
        from irh.asymptotics import newton_from_geodesic

        result = newton_from_geodesic(small_graph, tolerance=1.0)
        assert hasattr(result, "potential")
        assert hasattr(result, "passed")

    def test_born_typicality(self, small_graph):
        """Test Born rule typicality."""
        from irh.asymptotics import born_typicality

        result = born_typicality(small_graph, ensemble_size=100)
        assert hasattr(result, "passed")
        assert hasattr(result, "chi_squared")


# Recovery Tests


class TestRecovery:
    """Tests for recovery modules."""

    def test_entanglement_test(self, small_graph):
        """Test entanglement recovery."""
        from irh.recovery.quantum_mechanics import entanglement_test

        result = entanglement_test(small_graph)
        assert hasattr(result, "entropy")

    def test_efe_solver(self, small_graph):
        """Test EFE solver."""
        from irh.recovery.general_relativity import efe_solver

        result = efe_solver(small_graph)
        assert hasattr(result, "residual")

    def test_beta_functions(self, small_graph):
        """Test beta functions."""
        from irh.recovery.standard_model import beta_functions

        result = beta_functions(small_graph)
        assert hasattr(result, "qcd_b0")


# Predictions Tests


class TestPredictions:
    """Tests for predictions module."""

    def test_alpha_inverse(self, small_graph):
        """Test α⁻¹ prediction."""
        from irh.predictions.constants import predict_alpha_inverse

        result = predict_alpha_inverse(small_graph)
        assert hasattr(result, "value")
        assert result.value > 100  # Should be around 137

    def test_neutrino_masses(self, small_graph):
        """Test neutrino mass prediction."""
        from irh.predictions.constants import predict_neutrino_masses

        result = predict_neutrino_masses(small_graph)
        assert hasattr(result, "sum_masses")
        assert result.sum_masses > 0

    def test_ckm_matrix(self, small_graph):
        """Test CKM matrix prediction."""
        from irh.predictions.constants import predict_ckm_matrix

        result = predict_ckm_matrix(small_graph)
        assert result.matrix.shape == (3, 3)


# Grand Audit Tests


class TestGrandAudit:
    """Tests for grand_audit module."""

    def test_grand_audit(self, small_graph):
        """Test grand audit."""
        from irh.grand_audit import grand_audit

        report = grand_audit(small_graph)
        assert report.total_checks > 0
        assert report.pass_count >= 0


# DAG Validator Tests


class TestDAGValidator:
    """Tests for dag_validator module."""

    def test_validate_dag(self):
        """Test DAG validation."""
        from irh.dag_validator import validate_dag

        result = validate_dag()
        assert result["is_acyclic"] is True

    def test_assert_acyclic(self):
        """Test acyclic assertion."""
        from irh.dag_validator import assert_acyclic

        assert assert_acyclic() is True

    def test_check_no_adhoc(self):
        """Test no ad hoc parameters."""
        from irh.dag_validator import check_no_adhoc

        result = check_no_adhoc()
        assert result["all_grounded"] is True


# Golden Tests (Known Analytic Spectra)


class TestGoldenSpectra:
    """Golden tests with known analytic spectra."""

    def test_cycle_spectrum(self, cycle_graph):
        """Test cycle graph has known spectrum."""
        # C_n eigenvalues: λ_k = 2(1 - cos(2πk/n))
        N = cycle_graph.N
        expected = [2 * (1 - np.cos(2 * np.pi * k / N)) for k in range(N)]
        expected = sorted(expected)

        L = cycle_graph.get_laplacian()
        actual = sorted(np.linalg.eigvalsh(L))

        assert np.allclose(actual, expected, atol=1e-6)

    def test_complete_spectrum(self, complete_graph):
        """Test complete graph has known spectrum."""
        # K_n eigenvalues: 0 (mult 1), n (mult n-1)
        N = complete_graph.N
        expected = [0] + [N] * (N - 1)
        expected = sorted(expected)

        L = complete_graph.get_laplacian()
        actual = sorted(np.linalg.eigvalsh(L))

        assert np.allclose(actual, expected, atol=1e-6)


# Reproducibility Tests


class TestReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_same_graph(self):
        """Test same seed produces identical graphs."""
        from irh.graph_state import HyperGraph

        g1 = HyperGraph(N=20, seed=12345)
        g2 = HyperGraph(N=20, seed=12345)

        assert np.allclose(g1.adjacency_matrix, g2.adjacency_matrix)
        assert np.allclose(g1._weights_matrix, g2._weights_matrix)

    def test_different_seeds_different_graphs(self):
        """Test different seeds produce different graphs."""
        from irh.graph_state import HyperGraph

        g1 = HyperGraph(N=20, seed=12345)
        g2 = HyperGraph(N=20, seed=54321)

        assert not np.allclose(g1.adjacency_matrix, g2.adjacency_matrix)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
