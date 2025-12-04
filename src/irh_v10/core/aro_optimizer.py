"""
Adaptive Resonance Optimization (ARO) - IRH v10.0

ARO is the optimization algorithm that evolves random networks into
4D spacetime by minimizing the Harmony Functional.

Algorithm (Section III in manuscript):
    1. Initialize: Random network K₀
    2. Iterate:
        a. Compute ℋ_Harmony[K_t]
        b. Propose mutation: K_t → K'
        c. Accept if ℋ[K'] < ℋ[K_t] + thermal noise
        d. Anneal temperature: T_t → T_{t+1}
    3. Convergence: When ℋ stabilizes

Mutation Kernels:
    - Edge weight perturbation
    - Topology modification (add/remove edges)
    - Large-scale rewiring

ARO replaces ARO, ARO, ARO from v9.5.

Reference: IRH v10.0 manuscript, Section III "Adaptive Resonance Optimization"
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
from tqdm import tqdm

from .harmony_functional import harmony_functional
from .substrate import CymaticResonanceNetwork


@dataclass
class AROResult:
    """Results from ARO optimization."""
    K_final: np.ndarray | sp.spmatrix
    harmony_history: List[float]
    acceptance_rate: float
    final_harmony: float
    iterations: int
    converged: bool
    convergence_iteration: Optional[int] = None


class AdaptiveResonanceOptimizer:
    """
    Adaptive Resonance Optimization engine.
    
    Attributes:
        network: CymaticResonanceNetwork to optimize
        T_initial: Initial temperature for simulated annealing
        T_final: Final temperature
        max_iterations: Maximum number of iterations
        cooling_schedule: Temperature cooling ("exponential" or "logarithmic")
        mutation_rate: Probability of mutation per iteration
        convergence_threshold: Harmony change threshold for convergence
    
    Example:
        >>> network = CymaticResonanceNetwork(N=100, topology="random")
        >>> aro = AdaptiveResonanceOptimizer(network, max_iterations=1000)
        >>> result = aro.optimize()
        >>> print(f"Final harmony: {result.final_harmony:.6f}")
    """
    
    def __init__(
        self,
        network: CymaticResonanceNetwork,
        T_initial: float = 1.0,
        T_final: float = 0.01,
        max_iterations: int = 1000,
        cooling_schedule: str = "exponential",
        mutation_rate: float = 0.1,
        convergence_threshold: float = 1e-6,
        verbose: bool = True,
    ):
        self.network = network
        self.T_initial = T_initial
        self.T_final = T_final
        self.max_iterations = max_iterations
        self.cooling_schedule = cooling_schedule
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        
        # State
        self.K_current = network.K.copy()
        self.harmony_current = None
        self.iteration = 0
        
    def optimize(self) -> AROResult:
        """
        Run ARO optimization to minimize Harmony Functional.
        
        Returns:
            result: AROResult with optimization results
        """
        harmony_history = []
        accepted = 0
        total_proposed = 0
        
        # Initialize
        self.harmony_current = harmony_functional(self.K_current, self.network.N)
        harmony_history.append(self.harmony_current)
        
        # Progress bar
        iterator = range(self.max_iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc="ARO")
        
        converged = False
        convergence_iteration = None
        
        for iteration in iterator:
            self.iteration = iteration
            
            # Temperature annealing
            T = self._get_temperature(iteration)
            
            # Propose mutation
            K_proposed = self._propose_mutation(self.K_current)
            total_proposed += 1
            
            # Compute harmony of proposed state
            harmony_proposed = harmony_functional(K_proposed, self.network.N)
            
            # Acceptance criterion (Metropolis-Hastings)
            delta_H = harmony_proposed - self.harmony_current
            
            if delta_H < 0 or np.random.rand() < np.exp(-delta_H / T):
                # Accept
                self.K_current = K_proposed
                self.harmony_current = harmony_proposed
                accepted += 1
            
            harmony_history.append(self.harmony_current)
            
            # Check convergence
            if iteration > 100:
                recent_change = np.abs(harmony_history[-1] - harmony_history[-100])
                if recent_change < self.convergence_threshold:
                    converged = True
                    convergence_iteration = iteration
                    if self.verbose:
                        print(f"\nConverged at iteration {iteration}")
                    break
        
        acceptance_rate = accepted / total_proposed if total_proposed > 0 else 0.0
        
        result = AROResult(
            K_final=self.K_current,
            harmony_history=harmony_history,
            acceptance_rate=acceptance_rate,
            final_harmony=self.harmony_current,
            iterations=iteration + 1,
            converged=converged,
            convergence_iteration=convergence_iteration,
        )
        
        return result
    
    def _get_temperature(self, iteration: int) -> float:
        """Compute temperature for current iteration."""
        progress = iteration / self.max_iterations
        
        if self.cooling_schedule == "exponential":
            # T(t) = T_0 × (T_f/T_0)^t
            T = self.T_initial * (self.T_final / self.T_initial) ** progress
        elif self.cooling_schedule == "logarithmic":
            # T(t) = T_0 / ln(1 + t)
            T = self.T_initial / np.log(2 + iteration)
        else:
            # Linear
            T = self.T_initial + (self.T_final - self.T_initial) * progress
        
        return T
    
    def _propose_mutation(self, K: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
        """
        Propose a mutation to the coupling matrix.
        
        Mutation types:
            1. Edge weight perturbation (Gaussian noise)
            2. Edge addition/removal
            3. Large-scale topology change
        
        Args:
            K: Current coupling matrix
        
        Returns:
            K_new: Mutated coupling matrix
        """
        mutation_type = np.random.choice(
            ["perturbation", "topology", "rewiring"],
            p=[0.7, 0.2, 0.1]
        )
        
        if mutation_type == "perturbation":
            K_new = self._mutate_weights(K)
        elif mutation_type == "topology":
            K_new = self._mutate_topology(K)
        else:
            K_new = self._mutate_rewiring(K)
        
        return K_new
    
    def _mutate_weights(self, K: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
        """Perturb edge weights with Gaussian noise."""
        if sp.issparse(K):
            K_new = K.copy()
            # Perturb non-zero elements
            K_new.data += np.random.normal(0, 0.1 * self.mutation_rate, size=len(K_new.data))
            # Keep positive and symmetric
            K_new.data = np.abs(K_new.data)
            K_new = (K_new + K_new.T) / 2
        else:
            K_new = K.copy()
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1 * self.mutation_rate, size=K.shape)
            noise = (noise + noise.T) / 2  # Symmetrize
            K_new += noise
            # Keep positive
            K_new = np.maximum(K_new, 0)
            np.fill_diagonal(K_new, 0)
        
        return K_new
    
    def _mutate_topology(self, K: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
        """Add or remove edges."""
        K_new = K.copy() if not sp.issparse(K) else K.toarray()
        
        # Choose random edge
        i = np.random.randint(0, self.network.N)
        j = np.random.randint(0, self.network.N)
        if i == j:
            return K
        
        # Flip edge (add if absent, remove if present)
        if K_new[i, j] > 1e-10:
            # Remove
            K_new[i, j] = 0
            K_new[j, i] = 0
        else:
            # Add with random weight
            weight = np.random.uniform(0.5, 1.5) * self.network.coupling_strength
            K_new[i, j] = weight
            K_new[j, i] = weight
        
        return K_new
    
    def _mutate_rewiring(self, K: np.ndarray | sp.spmatrix) -> np.ndarray | sp.spmatrix:
        """Rewire edges (remove one, add another)."""
        K_new = K.copy() if not sp.issparse(K) else K.toarray()
        
        # Find existing edges
        if sp.issparse(K):
            edges = list(zip(*K.nonzero()))
        else:
            edges = list(zip(*np.nonzero(K)))
        
        if len(edges) < 2:
            return K
        
        # Remove random edge
        edge_to_remove = edges[np.random.randint(len(edges))]
        i, j = edge_to_remove
        weight = K_new[i, j]
        K_new[i, j] = 0
        K_new[j, i] = 0
        
        # Add new edge elsewhere
        i_new = np.random.randint(0, self.network.N)
        j_new = np.random.randint(0, self.network.N)
        if i_new != j_new:
            K_new[i_new, j_new] = weight
            K_new[j_new, i_new] = weight
        
        return K_new


def run_aro_demo(N: int = 100, iterations: int = 500) -> AROResult:
    """
    Demo function to run ARO on a random network.
    
    Args:
        N: Number of oscillators
        iterations: Number of optimization iterations
    
    Returns:
        result: Optimization result
    """
    print(f"Running ARO demo: N={N}, iterations={iterations}")
    
    network = CymaticResonanceNetwork(N=N, topology="random", seed=42)
    aro = AdaptiveResonanceOptimizer(
        network,
        max_iterations=iterations,
        T_initial=1.0,
        T_final=0.01,
    )
    
    result = aro.optimize()
    
    print(f"\nResults:")
    print(f"  Final harmony: {result.final_harmony:.6f}")
    print(f"  Acceptance rate: {result.acceptance_rate:.2%}")
    print(f"  Converged: {result.converged}")
    
    return result
