"""
Finite Spectral Triple Implementation

This module implements the FiniteSpectralTriple class, which represents
a discrete approximation to a continuous noncommutative geometry.

A spectral triple (A, H, D) consists of:
- A: An algebra (represented implicitly through commutators)
- H: A Hilbert space (finite dimensional: C^N)
- D: A Dirac operator (Hermitian matrix)

Additional structures:
- J: Real structure (antilinear operator)
- γ: Grading operator (chirality)

Key axioms enforced:
- [D, γ] = 0 (D anticommutes with grading)
- [D, J] compatible with real structure
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class FiniteSpectralTriple:
    """
    A finite spectral triple representing discrete quantum geometry.
    
    Attributes
    ----------
    N : int
        Dimension of the Hilbert space
    D : NDArray[np.complex128]
        Dirac operator (Hermitian matrix)
    J : Optional[NDArray[np.complex128]]
        Real structure operator (antilinear)
    gamma : Optional[NDArray[np.complex128]]
        Grading operator (chirality)
    
    The Dirac operator D is the dynamical variable optimized to minimize
    the spectral action.
    """
    
    def __init__(
        self,
        N: int,
        D: Optional[NDArray[np.complex128]] = None,
        J: Optional[NDArray[np.complex128]] = None,
        gamma: Optional[NDArray[np.complex128]] = None,
        seed: Optional[int] = None,
        enforce_axioms_on_init: bool = True,
    ):
        """
        Initialize a finite spectral triple.
        
        Parameters
        ----------
        N : int
            Size of the Hilbert space
        D : Optional[NDArray], default=None
            Initial Dirac operator. If None, initialized randomly.
        J : Optional[NDArray], default=None
            Real structure operator. If None, uses standard conjugation.
        gamma : Optional[NDArray], default=None
            Grading operator. If None, uses balanced chirality split.
        seed : Optional[int], default=None
            Random seed for reproducibility
        enforce_axioms_on_init : bool, default=True
            Whether to enforce axioms on initialization
        """
        self.N = N
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize Dirac operator
        if D is None:
            # Random Hermitian matrix
            D_random = np.random.randn(N, N) + 1j * np.random.randn(N, N)
            self.D = (D_random + D_random.conj().T) / 2.0
        else:
            self.D = D.copy()
        
        # Initialize real structure (standard conjugation)
        if J is None:
            self.J = np.eye(N, dtype=np.complex128)
        else:
            self.J = J.copy()
        
        # Initialize grading (balanced chirality)
        if gamma is None:
            gamma_diag = np.ones(N)
            gamma_diag[N//2:] = -1
            self.gamma = np.diag(gamma_diag)
        else:
            self.gamma = gamma.copy()
        
        # Enforce axioms on initial state
        if enforce_axioms_on_init:
            self.enforce_axioms()
    
    def enforce_axioms(self) -> None:
        """
        Project D back to the subspace satisfying spectral triple axioms.
        
        Enforces:
        1. {D, γ} = 0  (D anticommutes with grading)
        2. D remains Hermitian
        
        This is done by projecting D onto the subspace where these
        conditions hold.
        
        For {D, γ} = 0, we need γDγ = -D.
        The projection is: D_new = (D - γDγ) / 2
        """
        # Enforce Hermiticity: D = (D + D†)/2
        self.D = (self.D + self.D.conj().T) / 2.0
        
        # Enforce {D, γ} = 0
        # The anticommutation relation {D, γ} = Dγ + γD = 0 means γDγ = -D
        # To project: D_new = (D - γDγ) / 2 gives the component that anticommutes
        gamma_D_gamma = self.gamma @ self.D @ self.gamma
        self.D = (self.D - gamma_D_gamma) / 2.0
        
        # Re-enforce Hermiticity after projection
        self.D = (self.D + self.D.conj().T) / 2.0
    
    def spectrum(self) -> NDArray[np.float64]:
        """
        Compute the eigenvalues of the Dirac operator.
        
        Returns
        -------
        eigenvalues : NDArray[np.float64]
            Sorted eigenvalues (real, since D is Hermitian)
        """
        eigvals = np.linalg.eigvalsh(self.D)
        return np.sort(eigvals)
    
    def apply_J(self, psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Apply the real structure J (antilinear operator).
        
        For standard real structure, J(ψ) = J̄ψ̄ where ψ̄ is complex conjugate.
        
        Parameters
        ----------
        psi : NDArray[np.complex128]
            State vector
        
        Returns
        -------
        J_psi : NDArray[np.complex128]
            J-transformed state
        """
        return self.J @ psi.conj()
    
    def count_zero_modes(self, threshold: float = 1e-6) -> int:
        """
        Count the number of near-zero eigenvalues (chiral zero modes).
        
        Parameters
        ----------
        threshold : float, default=1e-6
            Eigenvalues with |λ| < threshold are considered zero
        
        Returns
        -------
        n_zero : int
            Number of zero modes
        """
        eigvals = self.spectrum()
        return np.sum(np.abs(eigvals) < threshold)
    
    def get_zero_mode_chiralities(self, threshold: float = 1e-6) -> Tuple[int, int]:
        """
        Get the chirality distribution of zero modes.
        
        Parameters
        ----------
        threshold : float, default=1e-6
            Eigenvalues with |λ| < threshold are considered zero
        
        Returns
        -------
        n_plus, n_minus : Tuple[int, int]
            Number of positive and negative chirality zero modes
        """
        eigvals, eigvecs = np.linalg.eigh(self.D)
        zero_indices = np.where(np.abs(eigvals) < threshold)[0]
        
        n_plus = 0
        n_minus = 0
        
        for idx in zero_indices:
            psi = eigvecs[:, idx]
            chirality_expectation = np.real(psi.conj() @ self.gamma @ psi)
            if chirality_expectation > 0:
                n_plus += 1
            else:
                n_minus += 1
        
        return n_plus, n_minus
    
    def D_squared(self) -> NDArray[np.complex128]:
        """
        Compute D² efficiently.
        
        Returns
        -------
        D2 : NDArray[np.complex128]
            D² operator
        """
        return self.D @ self.D
    
    def to_dict(self) -> dict:
        """
        Serialize the spectral triple to a dictionary.
        
        Returns
        -------
        data : dict
            Dictionary with N, D, J, gamma as arrays
        """
        return {
            "N": self.N,
            "D": self.D,
            "J": self.J,
            "gamma": self.gamma,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FiniteSpectralTriple":
        """
        Deserialize from a dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with N, D, J, gamma
        
        Returns
        -------
        triple : FiniteSpectralTriple
            Reconstructed spectral triple
        """
        return cls(
            N=data["N"],
            D=data["D"],
            J=data["J"],
            gamma=data["gamma"],
            enforce_axioms_on_init=False,  # Don't re-enforce, assume data is valid
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FiniteSpectralTriple(N={self.N}, "
            f"n_zero_modes={self.count_zero_modes()})"
        )
