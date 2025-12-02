"""
sote_scaling_verification.py - SOTE Scaling Verification Script

This module verifies the scaling arguments used in the SOTE Principle derivation
for the Intrinsic Resonance Holography (RIRH) formalism v9.5.

The script computes:
1. Random geometric graphs with varying N and connectivity
2. Holographic action S_holo from Laplacian eigenvalues
3. Entropic cost C_E (von Neumann entropy)
4. Verification that S_holo / C_E stabilizes when weighted by xi = 1/ln(N)

References:
- SOTE_Derivation.md for theoretical foundations
- IRH Formalism v9.5 Technical Appendix
"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def generate_random_geometric_graph(N: int, dim: int = 2, k: float = 6.0) -> np.ndarray:
    """
    Generate a random geometric graph using a radius threshold.

    The radius is set to r = sqrt(k * ln(N) / (pi * N)) which ensures
    expected degree ~ k * ln(N), near the connectivity threshold.

    Args:
        N: Number of nodes
        dim: Embedding dimension (default: 2)
        k: Connectivity constant (default: 6.0)

    Returns:
        adj_matrix: NxN symmetric adjacency matrix (0/1 entries)
    """
    if N < 2:
        raise ValueError("N must be at least 2")

    # Random points in [0, 1]^dim
    rng = np.random.default_rng()
    points = rng.random((N, dim))

    # Compute pairwise distances
    dist = distance_matrix(points, points)

    # Connectivity radius for expected degree ~ k * ln(N)
    # For d=2: expected neighbors ~ pi * r^2 * N
    # So r = sqrt(k * ln(N) / (pi * N))
    if N > 1:
        r = np.sqrt(k * np.log(N) / (np.pi * N))
    else:
        r = 1.0

    # Create adjacency matrix (connect if distance < r)
    adj_matrix = (dist < r).astype(np.float64)

    # Remove self-loops
    np.fill_diagonal(adj_matrix, 0.0)

    return adj_matrix


def compute_holographic_action(adj_matrix: np.ndarray) -> dict:
    """
    Compute the holographic action S_holo from the graph Laplacian.

    S_holo = Tr(L^2) / exp(log_det / (N * ln(N)))

    where:
    - L is the Laplacian matrix
    - Tr(L^2) = sum of eigenvalues squared
    - log_det = sum of log of non-zero eigenvalues

    Args:
        adj_matrix: NxN symmetric adjacency matrix

    Returns:
        Dictionary containing:
        - s_holo: The holographic action value
        - trace_L2: Tr(L^2)
        - log_det: Log pseudo-determinant
        - exponent: log_det / (N * ln(N))
        - eigenvalues: Full eigenvalue spectrum
        - von_neumann_entropy: S_vN = -sum(p_i * log(p_i))
    """
    N = adj_matrix.shape[0]

    # Compute Laplacian L = D - A
    degrees = np.sum(adj_matrix, axis=1)
    D = np.diag(degrees)
    L = D - adj_matrix

    # Compute eigenvalues (full spectrum for accuracy)
    eigenvalues = np.linalg.eigvalsh(L)

    # Sort eigenvalues
    eigenvalues = np.sort(eigenvalues)

    # Tr(L^2) = sum of eigenvalues squared
    trace_L2 = np.sum(eigenvalues ** 2)

    # Filter non-zero eigenvalues for log-determinant
    # Use threshold to handle numerical zeros
    threshold = 1e-10
    non_zero_eigs = eigenvalues[eigenvalues > threshold]

    if len(non_zero_eigs) == 0:
        # Disconnected or trivial graph
        return {
            "s_holo": np.nan,
            "trace_L2": trace_L2,
            "log_det": np.nan,
            "exponent": np.nan,
            "eigenvalues": eigenvalues,
            "von_neumann_entropy": 0.0,
        }

    # Log pseudo-determinant
    log_det = np.sum(np.log(non_zero_eigs))

    # Exponent: log_det / (N * ln(N))
    if N > 1:
        exponent = log_det / (N * np.log(N))
    else:
        exponent = 0.0

    # Holographic action
    if np.isfinite(exponent):
        s_holo = trace_L2 / np.exp(exponent)
    else:
        s_holo = np.nan

    # Von Neumann entropy from normalized eigenvalue distribution
    # p_i = lambda_i / sum(lambda_j) for lambda_i > 0
    p = non_zero_eigs / np.sum(non_zero_eigs)
    von_neumann_entropy = -np.sum(p * np.log(p + 1e-15))

    return {
        "s_holo": s_holo,
        "trace_L2": trace_L2,
        "log_det": log_det,
        "exponent": exponent,
        "eigenvalues": eigenvalues,
        "von_neumann_entropy": von_neumann_entropy,
    }


def verify_criticality(
    N_values: list[int] | None = None,
    k_range: tuple[float, float] = (4.0, 10.0),
    k_steps: int = 5,
    dim: int = 2,
    n_trials: int = 10,
) -> list[dict]:
    """
    Verify SOTE scaling by checking if S_holo / C_E stabilizes when weighted by xi = 1/ln(N).

    For each N, generate graphs with varying connectivity radius and compute:
    - S_holo (holographic action)
    - C_E ≈ S_vonNeumann (entropic cost)
    - Ratio S_holo / C_E
    - Weighted ratio: (S_holo / C_E) * xi where xi = 1/ln(N)

    Args:
        N_values: List of graph sizes to test (default: [100, 500, 1000, 2000])
        k_range: Range of connectivity constants (k_min, k_max)
        k_steps: Number of k values to test
        dim: Embedding dimension
        n_trials: Number of random trials per (N, k) pair

    Returns:
        List of result dictionaries for each (N, k) combination
    """
    if N_values is None:
        N_values = [100, 500, 1000, 2000]

    k_values = np.linspace(k_range[0], k_range[1], k_steps)

    results = []

    for N in N_values:
        xi = 1.0 / np.log(N)  # RG flow parameter

        for k in k_values:
            s_holo_trials = []
            c_e_trials = []
            ratio_trials = []
            weighted_ratio_trials = []

            for _ in range(n_trials):
                # Generate random geometric graph
                adj = generate_random_geometric_graph(N, dim=dim, k=k)

                # Compute holographic action
                result = compute_holographic_action(adj)

                if np.isnan(result["s_holo"]) or result["von_neumann_entropy"] < 1e-10:
                    continue

                s_holo = result["s_holo"]
                c_e = result["von_neumann_entropy"]

                ratio = s_holo / c_e if c_e > 0 else np.nan
                weighted_ratio = ratio * xi if np.isfinite(ratio) else np.nan

                s_holo_trials.append(s_holo)
                c_e_trials.append(c_e)
                ratio_trials.append(ratio)
                weighted_ratio_trials.append(weighted_ratio)

            if len(ratio_trials) > 0:
                results.append({
                    "N": N,
                    "k": k,
                    "xi": xi,
                    "s_holo_mean": np.mean(s_holo_trials),
                    "s_holo_std": np.std(s_holo_trials),
                    "c_e_mean": np.mean(c_e_trials),
                    "c_e_std": np.std(c_e_trials),
                    "ratio_mean": np.mean(ratio_trials),
                    "ratio_std": np.std(ratio_trials),
                    "weighted_ratio_mean": np.mean(weighted_ratio_trials),
                    "weighted_ratio_std": np.std(weighted_ratio_trials),
                    "n_valid_trials": len(ratio_trials),
                })

    return results


def print_results_table(results: list[dict]) -> None:
    """
    Print verification results as a formatted table.

    Args:
        results: List of result dictionaries from verify_criticality()
    """
    # Header
    print("\n" + "=" * 100)
    print("SOTE SCALING VERIFICATION RESULTS")
    print("=" * 100)
    print(
        f"{'N':>6} | {'k':>5} | {'ξ=1/ln(N)':>10} | "
        f"{'S_holo':>12} | {'C_E':>10} | "
        f"{'S/C_E':>10} | {'(S/C_E)·ξ':>12} | {'trials':>6}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['N']:>6} | {r['k']:>5.2f} | {r['xi']:>10.4f} | "
            f"{r['s_holo_mean']:>12.2f} | {r['c_e_mean']:>10.4f} | "
            f"{r['ratio_mean']:>10.2f} | {r['weighted_ratio_mean']:>12.4f} | "
            f"{r['n_valid_trials']:>6}"
        )

    print("=" * 100)

    # Summary: Check if weighted ratio stabilizes
    print("\nSCALING ANALYSIS:")
    print("-" * 60)

    # Group by k and check weighted ratio stability across N
    k_values = sorted(set(r["k"] for r in results))
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        weighted_ratios = [r["weighted_ratio_mean"] for r in k_results]

        if len(weighted_ratios) > 1:
            mean_wr = np.mean(weighted_ratios)
            std_wr = np.std(weighted_ratios)
            cv = std_wr / mean_wr if mean_wr > 0 else np.inf  # coefficient of variation

            stability = "STABLE" if cv < 0.3 else "UNSTABLE"
            print(f"k={k:.2f}: Weighted ratio = {mean_wr:.4f} ± {std_wr:.4f} (CV={cv:.2f}) -> {stability}")

    print("-" * 60)


def main():
    """
    Run the SOTE scaling verification and print results.
    """
    print("Running SOTE Scaling Verification...")
    print("This verifies the scaling arguments from SOTE_Derivation.md")
    print()

    # Run verification with default parameters
    results = verify_criticality(
        N_values=[100, 500, 1000, 2000],
        k_range=(5.0, 8.0),
        k_steps=4,
        dim=2,
        n_trials=5,
    )

    # Print formatted table
    print_results_table(results)

    return results


if __name__ == "__main__":
    main()
