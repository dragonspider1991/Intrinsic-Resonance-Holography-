"""
Interference Matrix Builder - Graph Laplacian ℒ

The Interference Matrix is the graph Laplacian of the Cymatic Resonance Network.
It governs wave interference patterns and spectral properties.

Definition (Equation 12 in manuscript):
    ℒ = D - K
where:
    - K is the real symmetric coupling matrix
    - D is the degree matrix: D_ii = Σ_j K_ij

The eigenvalues of ℒ determine:
    - Spectral dimension (via heat kernel)
    - Lorentzian signature (negative eigenvalue count)
    - Emergent spacetime geometry

Reference: IRH v10.0 manuscript, Section II.C "Interference Matrix"
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional


def build_interference_matrix(
    K: np.ndarray | sp.spmatrix,
    normalized: bool = False,
) -> np.ndarray | sp.spmatrix:
    """
    Build the Interference Matrix (Graph Laplacian) from coupling matrix.
    
    Args:
        K: Real symmetric coupling matrix N×N
        normalized: If True, return normalized Laplacian
    
    Returns:
        L: Interference Matrix (Graph Laplacian)
            - Standard: ℒ = D - K
            - Normalized: ℒ_norm = D^(-1/2) ℒ D^(-1/2)
    
    Example:
        >>> K = network.K
        >>> L = build_interference_matrix(K)
        >>> eigenvalues = np.linalg.eigvalsh(L)
    """
    if sp.issparse(K):
        degrees = np.array(K.sum(axis=1)).flatten()
        D = sp.diags(degrees)
        L = D - K
        
        if normalized:
            # Normalized Laplacian: D^(-1/2) L D^(-1/2)
            degrees_inv_sqrt = np.zeros_like(degrees)
            mask = degrees > 1e-12
            degrees_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
            D_inv_sqrt = sp.diags(degrees_inv_sqrt)
            L = D_inv_sqrt @ L @ D_inv_sqrt
        
        return L
    else:
        degrees = K.sum(axis=1)
        D = np.diag(degrees)
        L = D - K
        
        if normalized:
            degrees_inv_sqrt = np.zeros_like(degrees)
            mask = degrees > 1e-12
            degrees_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
            D_inv_sqrt = np.diag(degrees_inv_sqrt)
            L = D_inv_sqrt @ L @ D_inv_sqrt
        
        return L


def compute_spectrum_full(
    L: np.ndarray | sp.spmatrix,
    return_eigenvectors: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Compute full eigenspectrum of Interference Matrix.
    
    Args:
        L: Interference Matrix
        return_eigenvectors: If True, return (eigenvalues, eigenvectors)
    
    Returns:
        eigenvalues: Sorted eigenvalues
        eigenvectors: Eigenvectors (if requested)
    """
    if sp.issparse(L):
        L_dense = L.toarray()
    else:
        L_dense = L
    
    if return_eigenvectors:
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]
    else:
        eigenvalues = np.linalg.eigvalsh(L_dense)
        return np.sort(eigenvalues)


def compute_spectrum_partial(
    L: sp.spmatrix,
    k: int,
    which: str = 'SM',
    return_eigenvectors: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Compute partial eigenspectrum (for large sparse matrices).
    
    Args:
        L: Sparse Interference Matrix
        k: Number of eigenvalues to compute
        which: Which eigenvalues ('SM'=smallest magnitude, 'LM'=largest, 'SA'=smallest algebraic)
        return_eigenvectors: If True, return (eigenvalues, eigenvectors)
    
    Returns:
        eigenvalues: k eigenvalues
        eigenvectors: k eigenvectors (if requested)
    """
    try:
        if return_eigenvectors:
            eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=k, which=which)
            idx = np.argsort(eigenvalues)
            return eigenvalues[idx], eigenvectors[:, idx]
        else:
            eigenvalues = sp.linalg.eigsh(L, k=k, which=which, return_eigenvectors=False)
            return np.sort(eigenvalues)
    except Exception as e:
        # Fallback to dense computation
        print(f"Warning: Sparse eigensolver failed ({e}), falling back to dense")
        return compute_spectrum_full(L, return_eigenvectors)


def spectral_gap(eigenvalues: np.ndarray, threshold: float = 1e-10) -> float:
    """
    Compute spectral gap (difference between first two non-zero eigenvalues).
    
    The spectral gap is important for:
        - Connectivity of the network
        - Mixing time of random walks
        - Quantum to classical transition
    
    Args:
        eigenvalues: Sorted eigenvalues of ℒ
        threshold: Minimum value to consider non-zero
    
    Returns:
        gap: Spectral gap λ₂ - λ₁ (where λ₁ ≈ 0)
    """
    # Filter out zero modes
    nonzero = eigenvalues[eigenvalues > threshold]
    
    if len(nonzero) < 2:
        return 0.0
    
    # Gap between first and second non-zero eigenvalues
    gap = nonzero[1] - nonzero[0]
    return gap


def count_negative_eigenvalues(eigenvalues: np.ndarray, threshold: float = -1e-10) -> int:
    """
    Count negative eigenvalues (for Lorentzian signature).
    
    In 4D spacetime, we expect exactly 1 negative eigenvalue
    (signature -+++ or metric signature -2).
    
    Args:
        eigenvalues: Eigenvalues of ℒ (or modified ℒ)
        threshold: Threshold for considering negative
    
    Returns:
        count: Number of negative eigenvalues
    """
    count = np.sum(eigenvalues < threshold)
    return int(count)
