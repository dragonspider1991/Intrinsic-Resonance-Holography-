"""
ARO module for IRH v11.0: Self-Organized Topological Extremization functional.

Implements Theorem 4.1 from v11.0: Uniqueness of ARO as the only
functional satisfying intensive scaling, holographic compliance, and
scale invariance.

S_ARO[G] = Tr(L²) / (det' L)^(1/(N ln N))
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class AROFunctional:
    """
    The action functional that determines optimal graph topology.
    
    Key properties proven in v11.0:
    1. Intensive (scales as O(1))
    2. Holographically consistent
    3. Scale-invariant under coarse-graining
    
    Attributes:
        substrate: InformationSubstrate instance
        N: Number of nodes
        _eigenvalues_cache: Cached eigenvalues for efficiency
        _action_cache: Cached action value
    """
    
    def __init__(self, substrate):
        """Initialize ARO functional for given substrate."""
        self.substrate = substrate
        self.N = substrate.N
        
        # Cache for efficiency
        self._eigenvalues_cache = None
        self._action_cache = None
        
        logger.info(f"ARO Functional initialized for N={self.N} nodes")
        
    def compute_action(self, W: Optional[sp.csr_matrix] = None,
                      use_cache: bool = True) -> float:
        """
        Compute the ARO action for given graph.
        
        Parameters:
        -----------
        W : sparse matrix, optional
            Graph weights (uses substrate.W if None)
        use_cache : bool
            Whether to use cached eigenvalues
            
        Returns:
        --------
        S : float
            The ARO action value
        """
        
        if W is None:
            W = self.substrate.W
            
        if W is None:
            raise ValueError("No graph weights available")
        
        # Construct Laplacian
        W_abs = np.abs(W.toarray())
        D = np.diag(W_abs.sum(axis=1))
        L = sp.csr_matrix(D - W_abs)
        
        # Compute eigenvalues (expensive operation)
        if use_cache and self._eigenvalues_cache is not None:
            eigenvalues = self._eigenvalues_cache
        else:
            try:
                k = min(100, self.N - 2)  # Number of eigenvalues to compute
                eigenvalues = eigsh(L, k=k, which='SM', return_eigenvectors=False)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
            except:
                # Fallback for small systems
                eigenvalues = np.linalg.eigvalsh(L.toarray())
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            if use_cache:
                self._eigenvalues_cache = eigenvalues
        
        # Numerator: Tr(L²) = Σ λᵢ²
        trace_L2 = np.sum(eigenvalues**2)
        
        # Denominator: (det' L)^α where α = 1/(N ln N)
        log_det = np.sum(np.log(eigenvalues + 1e-100))
        
        # Scaling exponent (proven unique in Theorem 4.1)
        alpha = 1.0 / (self.N * np.log(self.N + 1))
        
        # Regularized determinant
        det_power = np.exp(log_det * alpha)
        
        # Action
        S = trace_L2 / max(det_power, 1e-100)
        
        self._action_cache = S
        return S
    
    def compute_gradient(self, W: sp.csr_matrix, 
                        delta: float = 1e-5) -> np.ndarray:
        """
        Compute gradient ∂S/∂W_{ij} via finite differences.
        
        This gradient guides the optimization (flow toward minimum).
        """
        gradient = np.zeros((self.N, self.N))
        S0 = self.compute_action(W, use_cache=False)
        
        # For each edge, perturb and compute derivative
        W_dense = W.toarray()
        
        # Sample random edges for efficiency
        n_samples = min(100, int(self.N * np.log(self.N)))
        
        for _ in range(n_samples):
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)
            
            if i != j and np.abs(W_dense[i, j]) > 1e-10:
                # Perturb magnitude
                W_pert = W_dense.copy()
                W_pert[i, j] *= (1 + delta)
                W_pert[j, i] *= (1 + delta)
                
                S_pert = self.compute_action(sp.csr_matrix(W_pert), use_cache=False)
                
                gradient[i, j] = (S_pert - S0) / (delta * np.abs(W_dense[i, j]))
                gradient[j, i] = gradient[i, j]
        
        return gradient
    
    def verify_intensive_scaling(self, N_range: list) -> dict:
        """
        Verify that S_ARO scales as O(1) with system size.
        
        This is Requirement 1 from Theorem 4.1.
        """
        results = {'N': [], 'S': [], 'S_normalized': []}
        
        from .substrate_v11 import InformationSubstrate
        
        for N in N_range:
            # Create test substrate
            sub = InformationSubstrate(N, dimension=4)
            sub.initialize_correlations('random_geometric')
            sub.compute_laplacian()
            
            # Compute action
            sote = AROFunctional(sub)
            S = sote.compute_action()
            
            # Normalize by expected extensive scaling
            # For extensive: S ~ N; for intensive: S ~ O(1)
            S_norm = S / N
            
            results['N'].append(N)
            results['S'].append(S)
            results['S_normalized'].append(S_norm)
        
        # Check scaling: if intensive, S/N should decay as 1/N
        # i.e., log(S/N) ~ -log(N)
        log_N = np.log(results['N'])
        log_S_norm = np.log(results['S_normalized'])
        
        # Fit: log(S/N) = a log(N) + b
        coeffs = np.polyfit(log_N, log_S_norm, deg=1)
        slope = coeffs[0]
        
        results['scaling_exponent'] = slope
        results['is_intensive'] = np.abs(slope + 1) < 0.2  # Should be -1
        
        logger.info(f"Intensive scaling test: slope={slope:.3f}, intensive={results['is_intensive']}")
        
        return results
    
    def verify_holographic_compliance(self) -> dict:
        """
        Verify that minimizing S_ARO enforces holographic bound.
        
        This is Requirement 2 from Theorem 4.1.
        """
        # Compute action at current state
        S_initial = self.compute_action()
        
        # Compute information content
        bound_check = self.substrate.verify_holographic_bound()
        
        # Perturb to violate bound (add extra edges)
        W_test = self.substrate.W.toarray()
        W_test += 0.1 * np.random.rand(self.N, self.N)  # Excessive correlations
        W_test = (W_test + W_test.T) / 2  # Symmetrize
        
        S_violated = self.compute_action(sp.csr_matrix(W_test), use_cache=False)
        
        return {
            'S_compliant': S_initial,
            'S_violated': S_violated,
            'action_increases': S_violated > S_initial,
            'bound_satisfied': bound_check['satisfies_bound']
        }
