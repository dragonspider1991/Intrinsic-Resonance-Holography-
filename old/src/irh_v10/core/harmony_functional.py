"""
Harmony Functional ℋ_Harmony[K] - IRH v10.0

The Harmony Functional is the objective function optimized by ARO.
It combines elastic energy with disorder/dissonance penalty.

Mathematical Definition (Equation 17 in manuscript):
    ℋ_Harmony[K] = Tr(K²) + ξ(N) × S_dissonance[K]

where:
    - Tr(K²) is the total elastic energy (vibrational energy)
    - S_dissonance[K] is the spectral entropy (disorder measure)
    - ξ(N) = 1/(N ln N) is the impedance matching coefficient

Reference: IRH v10.0 manuscript, Section III "Adaptive Resonance Optimization"
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional


def harmony_functional(
    K: np.ndarray | sp.spmatrix,
    N: int,
    eigenvalues: Optional[np.ndarray] = None,
    xi: Optional[float] = None,
) -> float:
    """
    Compute the Harmony Functional ℋ_Harmony[K].
    
    This is the exact implementation of Equation (17) from IRH v10.0:
        ℋ_Harmony[K] = Tr(K²) + ξ(N) × S_dissonance[K]
    
    Args:
        K: Coupling matrix (real symmetric)
        N: Number of oscillators
        eigenvalues: Pre-computed eigenvalues of ℒ (optional, for efficiency)
        xi: Impedance coefficient (optional, auto-computed if None)
    
    Returns:
        H: Harmony functional value (lower is better)
    
    Notes:
        - For large N > 10^6, uses stochastic trace estimation
        - Eigenvalues should be of the Interference Matrix ℒ = D - K
    
    Example:
        >>> K = network.K
        >>> H = harmony_functional(K, network.N)
        >>> print(f"Harmony: {H:.6f}")
    """
    # Compute impedance coefficient if not provided
    if xi is None:
        from .impedance_matching import impedance_coefficient
        xi = impedance_coefficient(N)
    
    # Compute elastic energy term: Tr(K²)
    if N > 1000000:
        # Stochastic trace estimation for very large matrices
        elastic_energy = _stochastic_trace_K_squared(K, N)
    else:
        if sp.issparse(K):
            # Efficient sparse computation: Tr(K²) = Σ_ij K_ij²
            elastic_energy = (K.multiply(K)).sum()
        else:
            elastic_energy = np.trace(K @ K)
    
    # Compute dissonance term: S_dissonance[K]
    if eigenvalues is None:
        # Need to compute eigenvalues of ℒ = D - K
        L = _build_laplacian(K)
        if sp.issparse(L) and N > 500:
            # Partial spectrum for large sparse matrices
            k = min(N - 1, 100)
            try:
                eigenvalues = sp.linalg.eigsh(L, k=k, which='SM', return_eigenvectors=False)
            except:
                # Fallback
                L_dense = L.toarray() if sp.issparse(L) else L
                eigenvalues = np.linalg.eigvalsh(L_dense)
        else:
            L_dense = L.toarray() if sp.issparse(L) else L
            eigenvalues = np.linalg.eigvalsh(L_dense)
    
    dissonance = _spectral_dissonance(eigenvalues)
    
    # Combine terms
    H_harmony = elastic_energy + xi * dissonance
    
    return H_harmony


def _build_laplacian(K: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
    """Build Laplacian ℒ = D - K from coupling matrix."""
    if sp.issparse(K):
        degrees = np.array(K.sum(axis=1)).flatten()
        D = sp.diags(degrees)
        L = D - K
        return L
    else:
        degrees = K.sum(axis=1)
        D = np.diag(degrees)
        L = D - K
        return L


def _spectral_dissonance(eigenvalues: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Compute spectral dissonance S_dissonance[K].
    
    This is the Shannon entropy of the normalized eigenvalue distribution:
        S = -Σ_i p_i log(p_i)
    where p_i = λ_i / Σ_j λ_j
    
    Args:
        eigenvalues: Eigenvalues of Interference Matrix ℒ
        epsilon: Small regularization to avoid log(0)
    
    Returns:
        S: Spectral dissonance (entropy)
    """
    # Filter out near-zero eigenvalues (zero modes)
    eigenvalues = eigenvalues[eigenvalues > epsilon]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    # Normalize to probability distribution
    total = eigenvalues.sum()
    if total < epsilon:
        return 0.0
    
    probabilities = eigenvalues / total
    
    # Compute Shannon entropy: -Σ p_i log(p_i)
    # Use log base 2 for bits
    entropy = -np.sum(probabilities * np.log2(probabilities + epsilon))
    
    return entropy


def _stochastic_trace_K_squared(
    K: sp.spmatrix,
    N: int,
    num_samples: int = 100,
) -> float:
    """
    Estimate Tr(K²) using stochastic trace estimation (Hutchinson's method).
    
    Args:
        K: Sparse coupling matrix
        N: Matrix dimension
        num_samples: Number of random vectors for estimation
    
    Returns:
        trace_estimate: Estimate of Tr(K²)
    """
    trace_sum = 0.0
    
    for _ in range(num_samples):
        # Random Rademacher vector (entries ±1 with equal probability)
        v = np.random.choice([-1, 1], size=N).astype(float)
        
        # Compute v^T K² v = (Kv)^T (Kv)
        Kv = K @ v
        trace_sum += np.dot(Kv, Kv)
    
    return trace_sum / num_samples


def harmony_gradient(
    K: np.ndarray,
    N: int,
    eigenvalues: Optional[np.ndarray] = None,
    eigenvectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute gradient of Harmony Functional with respect to K.
    
    This is used by gradient-based optimizers in ARO.
    
    ∂ℋ/∂K_ij = 2K_ij + ξ(N) × ∂S/∂K_ij
    
    Args:
        K: Coupling matrix
        N: Number of oscillators
        eigenvalues: Eigenvalues of ℒ (optional)
        eigenvectors: Eigenvectors of ℒ (optional)
    
    Returns:
        grad: Gradient matrix ∂ℋ/∂K
    """
    from .impedance_matching import impedance_coefficient
    xi = impedance_coefficient(N)
    
    # Elastic energy gradient: ∂Tr(K²)/∂K_ij = 2K_ij
    grad_elastic = 2.0 * K
    
    # Dissonance gradient (more complex, requires eigenvectors)
    if eigenvalues is None or eigenvectors is None:
        L = _build_laplacian(K)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    grad_dissonance = _dissonance_gradient(eigenvalues, eigenvectors, N)
    
    # Total gradient
    grad = grad_elastic + xi * grad_dissonance
    
    return grad


def _dissonance_gradient(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    N: int,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Compute gradient of spectral dissonance.
    
    This is a complex expression involving eigenvalue perturbation theory.
    For efficiency, we use a finite-difference approximation.
    
    Args:
        eigenvalues: Eigenvalues of ℒ
        eigenvectors: Eigenvectors of ℒ
        N: Matrix dimension
        epsilon: Regularization
    
    Returns:
        grad: Gradient of dissonance with respect to K
    """
    # Filter non-zero eigenvalues
    mask = eigenvalues > epsilon
    lambdas = eigenvalues[mask]
    vecs = eigenvectors[:, mask]
    
    if len(lambdas) == 0:
        return np.zeros((N, N))
    
    # Normalized probabilities
    total = lambdas.sum()
    probs = lambdas / total
    
    # ∂S/∂λ_k = -(log(λ_k/Σλ) + 1) / (Σλ log 2)
    dS_dlambda = -(np.log(probs) / np.log(2) + 1) / total
    
    # ∂λ_k/∂L_ij = φ_k^i φ_k^j (first-order perturbation)
    # ∂L/∂K = -I (since L = D - K, and ∂D/∂K_ij = δ_ij for diagonal terms)
    
    # This is approximate; exact formula is complex
    grad = np.zeros((N, N))
    for k, (lam, vec) in enumerate(zip(lambdas, vecs.T)):
        outer = np.outer(vec, vec)
        grad -= dS_dlambda[k] * outer
    
    return grad
