"""
Adaptive Resonance Optimization (ARO) - Core Optimizer

Implements the hybrid optimization strategy for maximizing the Harmony Functional:
1. Gradient-like perturbation (complex phase rotation)
2. Topological mutation (edge addition/removal)  
3. Simulated annealing (temperature-driven acceptance)

References: IRH v13.0 Section 4.2, Section 9.1
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, List
from numpy.typing import NDArray
from math import gamma

from .harmony import harmony_functional, compute_information_transfer_matrix


class AROOptimizer:
    """
    Adaptive Resonance Optimization Engine.
    
    Drives the Cymatic Resonance Network toward the Cosmic Fixed Point
    by maximizing the Harmony Functional through hybrid optimization.
    
    Parameters
    ----------
    N : int
        Number of nodes in the network.
    rng_seed : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    current_W : sp.spmatrix
        Current complex adjacency matrix.
    harmony_history : List[float]
        History of Harmony values during optimization.
    best_W : sp.spmatrix
        Best network configuration found.
    best_S : float
        Best Harmony value achieved.
        
    References
    ----------
    IRH v13.0 Section 4.2: ARO Optimization Loop
    """
    
    def __init__(self, N: int, rng_seed: Optional[int] = None, connection_probability: Optional[float] = None):
        self.N = int(N)
        self.rng = np.random.default_rng(rng_seed)
        self.current_W: Optional[sp.spmatrix] = None
        self.harmony_history: List[float] = []
        self.best_W: Optional[sp.spmatrix] = None
        self.best_S: float = -np.inf
        self.convergence_metric: List[dict] = []
        
        # Auto-initialize if connection_probability is provided
        if connection_probability is not None:
            self.initialize_network(scheme='random', connectivity_param=connection_probability)
        
    def initialize_network(
        self,
        scheme: str = 'geometric',
        connectivity_param: float = 0.01,
        d_initial: int = 4
    ) -> sp.spmatrix:
        """
        Initialize Cymatic Resonance Network for ARO optimization.
        
        Parameters
        ----------
        scheme : str, default 'geometric'
            Initialization scheme: 'geometric', 'random', 'lattice'.
        connectivity_param : float
            Controls initial edge density.
        d_initial : int
            Dimensionality hint for geometric initialization.
            
        Returns
        -------
        W : sp.spmatrix
            Initial complex adjacency matrix.
            
        Notes
        -----
        Complex phases φ_ij are initialized randomly to enable
        emergence of frustration (Theorem 1.2).
        """
        from scipy.spatial import KDTree
        
        adj_matrix = sp.lil_matrix((self.N, self.N), dtype=np.complex128)
        
        if scheme == 'geometric':
            # Geometric random graph with complex weights
            coords = self.rng.random((self.N, d_initial))
            tree = KDTree(coords)
            
            # Critical connectivity radius heuristic
            vol_factor = (np.pi**(d_initial/2) / gamma(d_initial/2 + 1))
            radius = (connectivity_param * np.log(self.N) / (self.N * vol_factor))**(1/d_initial)
            
            pairs = tree.query_pairs(radius)
            for i, j in pairs:
                magnitude = self.rng.uniform(0.1, 1.0)
                phase = self.rng.uniform(0, 2 * np.pi)
                adj_matrix[i, j] = magnitude * np.exp(1j * phase)
                adj_matrix[j, i] = magnitude * np.exp(-1j * phase)  # Hermitian-like
                
        elif scheme == 'random':
            # Erdős-Rényi with complex weights
            p = 2 * np.log(self.N) / self.N
            mask = sp.random(self.N, self.N, density=p, format='lil', random_state=self.rng)
            rows, cols = mask.nonzero()
            for r, c in zip(rows, cols):
                if r < c:
                    magnitude = self.rng.uniform(0.1, 1.0)
                    phase = self.rng.uniform(0, 2 * np.pi)
                    adj_matrix[r, c] = magnitude * np.exp(1j * phase)
                    adj_matrix[c, r] = magnitude * np.exp(-1j * phase)
        
        self.current_W = adj_matrix.tocsr()
        print(f"[ARO] Initialized {scheme} network: N={self.N}, edges={self.current_W.nnz}")
        return self.current_W
    
    def optimize(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.01,
        mutation_rate: float = 0.05,
        temp: float = 1.0,
        temp_start: Optional[float] = None,
        cooling_rate: float = 0.99,
        convergence_tol: float = 1e-6,
        verbose: bool = True,
        log_rg_invariants: bool = True
    ) -> np.ndarray:
        """
        Execute ARO optimization loop.
        
        Parameters
        ----------
        iterations : int
            Maximum number of optimization cycles.
        learning_rate : float
            Step size for weight perturbations.
        mutation_rate : float
            Probability of topological mutations.
        temp : float
            Initial temperature (for compatibility with main.py interface).
        temp_start : float, optional
            Alternative parameter for initial temperature.
        cooling_rate : float
            Temperature cooling rate per iteration.
        convergence_tol : float
            Convergence criterion for early stopping.
        verbose : bool
            Print progress updates.
        log_rg_invariants : bool, default True
            If True, log RG-invariant scalings at checkpoints (v15.0+).
            
        Returns
        -------
        best_W : np.ndarray
            Optimized network configuration as dense array.
            
        Notes
        -----
        Implements hybrid optimization (Theorem 4.2):
        - Perturbative phase adjustment
        - Topological rewiring  
        - Metropolis-Hastings acceptance
        
        v15.0+ Enhancement: Logs RG-invariant scalings to ensure parameters
        flow to self-consistent resonances without tuning.
        """
        if self.current_W is None:
            raise ValueError("Network not initialized. Call initialize_network() first.")
        
        # Use temp_start if provided, otherwise use temp
        if temp_start is None:
            temp_start = temp
        
        current_S = harmony_functional(self.current_W)
        self.best_S = current_S
        self.best_W = self.current_W.copy()
        self.harmony_history = [current_S]
        
        if verbose:
            print(f"[ARO] Starting optimization: {iterations} iterations")
            print(f"[ARO] Initial S_H = {current_S:.5f}")
            
            # Log RG invariants at start (v15.0+)
            if log_rg_invariants:
                try:
                    from .harmony import C_H
                    from .rigor_enhancements import rg_flow_beta
                    beta_val = rg_flow_beta(C_H)
                    print(f"[ARO] RG Flow: C_H = {C_H:.9f}, β(C_H) = {beta_val:.6e}")
                except ImportError:
                    pass
        
        no_improvement = 0
        max_no_improvement = 50
        
        # RG logging checkpoints
        rg_log_interval = max(1, iterations // 10)
        
        for i in range(iterations):
            # Dynamic annealing temperature with cooling_rate
            T = temp_start * (cooling_rate ** i)
            
            # Stage 1: Weight perturbation (complex rotation + magnitude scaling)
            W_candidate = self._perturb_weights(self.current_W, learning_rate)
            
            # Stage 2: Topological mutation (edge add/remove)
            if self.rng.random() < mutation_rate:
                W_candidate = self._mutate_topology(W_candidate)
            
            # Evaluate candidate
            new_S = harmony_functional(W_candidate)
            
            # Metropolis-Hastings acceptance
            delta_S = new_S - current_S
            accept = False
            
            if delta_S > 0:
                accept = True
            elif T > 1e-4:
                prob = np.exp(delta_S / T)
                if self.rng.random() < prob:
                    accept = True
            
            if accept and new_S > -np.inf:
                self.current_W = W_candidate
                current_S = new_S
                if current_S > self.best_S:
                    self.best_S = current_S
                    self.best_W = W_candidate.copy()
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                no_improvement += 1
            
            self.harmony_history.append(current_S)
            
            # Log RG-invariant scalings at checkpoints (v15.0+)
            if log_rg_invariants and verbose and (i % rg_log_interval == 0) and i > 0:
                try:
                    from .rigor_enhancements import compute_nondimensional_resonance_density
                    from .harmony import compute_information_transfer_matrix
                    
                    M = compute_information_transfer_matrix(self.current_W)
                    # Quick eigenvalue estimate for resonance density
                    if self.N < 500:
                        evals = np.linalg.eigvalsh(M.toarray())
                    else:
                        from scipy.sparse.linalg import eigsh
                        k = min(self.N - 1, 100)
                        evals = eigsh(M, k=k, which='LM', return_eigenvectors=False)
                    
                    rho_res, _ = compute_nondimensional_resonance_density(evals, self.N)
                    print(f"[ARO] Iter {i}: ρ_res = {rho_res:.6f} (nondimensional resonance density)")
                except:
                    pass
            
            # Convergence check
            if no_improvement > max_no_improvement:
                recent_mean = np.mean(self.harmony_history[-max_no_improvement:])
                if abs(self.best_S - recent_mean) < convergence_tol:
                    if verbose:
                        print(f"[ARO] Converged at iteration {i+1}")
                    break
            
            if verbose and (i % (iterations // 10 + 1) == 0 or i == iterations - 1):
                print(f"[ARO] Iter {i}/{iterations}: S_H={current_S:.5f}, Best={self.best_S:.5f}, T={T:.3f}")
        
        self.current_W = self.best_W
        if verbose:
            print(f"[ARO] Optimization complete. Final S_H = {self.best_S:.5f}")
        
        # Return dense array for compatibility with main.py
        return self.best_W.toarray()
    
    def _perturb_weights(
        self,
        W: sp.spmatrix,
        learning_rate: float
    ) -> sp.spmatrix:
        """Apply complex rotation and magnitude scaling to edge weights."""
        W_new = W.copy().tolil()
        rows, cols = W_new.nonzero()
        
        # Perturb ~10% of edges per iteration
        n_perturb = max(1, int(len(rows) * 0.1))
        perturb_idx = self.rng.choice(len(rows), size=n_perturb, replace=False)
        
        for idx in perturb_idx:
            r, c = rows[idx], cols[idx]
            val = W_new[r, c]
            
            # Complex rotation: W_ij → W_ij exp(iδθ)
            d_theta = self.rng.normal(0, learning_rate)
            # Magnitude scaling
            d_mag = self.rng.normal(0, learning_rate * 0.1)
            
            new_val = val * np.exp(1j * d_theta) * (1 + d_mag)
            
            # Normalize to keep weights bounded
            if np.abs(new_val) > 1.0:
                new_val /= np.abs(new_val)
            
            W_new[r, c] = new_val
            if r != c:
                W_new[c, r] = np.conj(new_val)  # Maintain hermiticity
        
        return W_new.tocsr()
    
    def _mutate_topology(
        self,
        W: sp.spmatrix
    ) -> sp.spmatrix:
        """Probabilistically add or remove edges."""
        W_new = W.copy().tolil()
        
        if self.rng.random() < 0.5:
            # Add edge
            u, v = self.rng.integers(0, self.N, size=2)
            if u != v and W_new[u, v] == 0:
                mag = 0.1
                phase = self.rng.uniform(0, 2 * np.pi)
                W_new[u, v] = mag * np.exp(1j * phase)
                W_new[v, u] = mag * np.exp(-1j * phase)
        else:
            # Remove edge
            rows, cols = W_new.nonzero()
            if len(rows) > 0:
                idx = self.rng.choice(len(rows))
                r, c = rows[idx], cols[idx]
                W_new[r, c] = 0
                W_new[c, r] = 0
        
        return W_new.tocsr()
