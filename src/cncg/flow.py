"""
Riemannian Gradient Flow on the Manifold of Hermitian Matrices

This module implements the Adaptive Resonance Optimization (ARO) flow:

    dD/dτ = -∇_D S[D] + η(τ)

where η is Langevin noise and the gradient is projected to maintain
the spectral triple axioms.

The flow evolves D on the manifold of Hermitian matrices satisfying
the commutation relations with J and γ.
"""

from typing import Callable, Optional, Tuple, Dict, List
import numpy as np
from numpy.typing import NDArray
from .spectral import FiniteSpectralTriple
from .action import spectral_action, spectral_action_gradient


class AdaptiveResonanceOptimizer:
    """
    Adaptive Resonance Optimization (ARO) flow for spectral triples.
    
    This class implements gradient descent on the manifold of Hermitian
    matrices with adaptive learning rate and optional Langevin dynamics.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        noise_strength: float = 0.0,
        adaptive: bool = True,
        momentum: float = 0.0,
    ):
        """
        Initialize the optimizer.
        
        Parameters
        ----------
        learning_rate : float, default=0.01
            Initial learning rate (step size)
        noise_strength : float, default=0.0
            Strength of Langevin noise η(τ)
        adaptive : bool, default=True
            Use adaptive learning rate (Armijo rule)
        momentum : float, default=0.0
            Momentum parameter (0 = no momentum)
        """
        self.learning_rate = learning_rate
        self.noise_strength = noise_strength
        self.adaptive = adaptive
        self.momentum = momentum
        self.velocity: Optional[NDArray[np.complex128]] = None
    
    def step(
        self,
        triple: FiniteSpectralTriple,
        gradient: NDArray[np.complex128],
    ) -> None:
        """
        Perform one optimization step.
        
        Parameters
        ----------
        triple : FiniteSpectralTriple
            The spectral triple to update
        gradient : NDArray[np.complex128]
            The gradient ∇_D S[D]
        """
        N = triple.N
        
        # Initialize velocity for momentum
        if self.velocity is None:
            self.velocity = np.zeros((N, N), dtype=np.complex128)
        
        # Add Langevin noise if requested
        if self.noise_strength > 0:
            noise = np.random.randn(N, N) + 1j * np.random.randn(N, N)
            noise = (noise + noise.conj().T) / 2.0  # Hermitian noise
            gradient = gradient + self.noise_strength * noise
        
        # Update velocity (momentum)
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        
        # Update D
        triple.D += self.velocity
        
        # Project back to constraint manifold
        triple.enforce_axioms()
    
    def adapt_learning_rate(
        self,
        S_old: float,
        S_new: float,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
    ) -> None:
        """
        Adapt learning rate based on progress.
        
        If S decreased, increase learning rate.
        If S increased, decrease learning rate.
        
        Parameters
        ----------
        S_old : float
            Previous action value
        S_new : float
            New action value
        increase_factor : float, default=1.1
            Factor to increase learning rate
        decrease_factor : float, default=0.5
            Factor to decrease learning rate
        """
        if not self.adaptive:
            return
        
        if S_new < S_old:
            # Good step, increase learning rate
            self.learning_rate *= increase_factor
        else:
            # Bad step, decrease learning rate
            self.learning_rate *= decrease_factor


def riemannian_gradient_descent(
    triple: FiniteSpectralTriple,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    Lambda: float = 1.0,
    cutoff: str = "heat",
    cutoff_param: float = 1.0,
    sparsity_weight: float = 0.0,
    noise_strength: float = 0.0,
    momentum: float = 0.0,
    adaptive: bool = True,
    tolerance: float = 1e-8,
    log_interval: int = 10,
    callback: Optional[Callable[[int, float, FiniteSpectralTriple], None]] = None,
) -> Dict[str, List]:
    """
    Run Riemannian gradient descent to minimize the spectral action.
    
    Parameters
    ----------
    triple : FiniteSpectralTriple
        Initial spectral triple (modified in place)
    max_iterations : int, default=1000
        Maximum number of iterations
    learning_rate : float, default=0.01
        Initial learning rate
    Lambda : float, default=1.0
        Energy scale for spectral action
    cutoff : str, default="heat"
        Cutoff function type
    cutoff_param : float, default=1.0
        Cutoff parameter
    sparsity_weight : float, default=0.0
        Weight for sparsity penalty
    noise_strength : float, default=0.0
        Langevin noise strength
    momentum : float, default=0.0
        Momentum parameter
    adaptive : bool, default=True
        Use adaptive learning rate
    tolerance : float, default=1e-8
        Convergence tolerance on |∇S|
    log_interval : int, default=10
        Log progress every N iterations
    callback : Optional[Callable], default=None
        Callback function called at each logged iteration
    
    Returns
    -------
    history : Dict[str, List]
        Dictionary containing:
        - "iteration": List of iteration numbers
        - "action": List of action values
        - "grad_norm": List of gradient norms
        - "learning_rate": List of learning rates
    """
    optimizer = AdaptiveResonanceOptimizer(
        learning_rate=learning_rate,
        noise_strength=noise_strength,
        adaptive=adaptive,
        momentum=momentum,
    )
    
    history: Dict[str, List] = {
        "iteration": [],
        "action": [],
        "grad_norm": [],
        "learning_rate": [],
    }
    
    for iteration in range(max_iterations):
        # Compute action
        S = spectral_action(
            triple.D,
            Lambda=Lambda,
            cutoff=cutoff,
            cutoff_param=cutoff_param,
            sparsity_weight=sparsity_weight,
        )
        
        # Compute gradient
        grad = spectral_action_gradient(
            triple.D,
            Lambda=Lambda,
            cutoff=cutoff,
            cutoff_param=cutoff_param,
            sparsity_weight=sparsity_weight,
        )
        
        # Gradient norm
        grad_norm = np.linalg.norm(grad)
        
        # Log progress
        if iteration % log_interval == 0:
            history["iteration"].append(iteration)
            history["action"].append(S)
            history["grad_norm"].append(grad_norm)
            history["learning_rate"].append(optimizer.learning_rate)
            
            print(
                f"Iter {iteration:5d}: S = {S:12.6f}, "
                f"|∇S| = {grad_norm:10.6e}, "
                f"lr = {optimizer.learning_rate:.6e}"
            )
            
            if callback is not None:
                callback(iteration, S, triple)
        
        # Check convergence
        if grad_norm < tolerance:
            print(f"Converged at iteration {iteration}")
            break
        
        # Store old action for adaptive learning rate
        S_old = S
        
        # Perform optimization step
        optimizer.step(triple, grad)
        
        # Compute new action for adaptation
        S_new = spectral_action(
            triple.D,
            Lambda=Lambda,
            cutoff=cutoff,
            cutoff_param=cutoff_param,
            sparsity_weight=sparsity_weight,
        )
        
        # Adapt learning rate
        optimizer.adapt_learning_rate(S_old, S_new)
    
    return history


def simulated_annealing_flow(
    triple: FiniteSpectralTriple,
    max_iterations: int = 1000,
    T_initial: float = 1.0,
    T_final: float = 0.01,
    Lambda: float = 1.0,
    cutoff: str = "heat",
    cutoff_param: float = 1.0,
    sparsity_weight: float = 0.0,
    log_interval: int = 10,
    callback: Optional[Callable[[int, float, FiniteSpectralTriple], None]] = None,
) -> Dict[str, List]:
    """
    Run simulated annealing with temperature schedule.
    
    This is an alternative to gradient descent that can escape local minima.
    
    Parameters
    ----------
    triple : FiniteSpectralTriple
        Initial spectral triple
    max_iterations : int, default=1000
        Number of iterations
    T_initial : float, default=1.0
        Initial temperature
    T_final : float, default=0.01
        Final temperature
    Lambda : float, default=1.0
        Energy scale
    cutoff : str, default="heat"
        Cutoff type
    cutoff_param : float, default=1.0
        Cutoff parameter
    sparsity_weight : float, default=0.0
        Sparsity weight
    log_interval : int, default=10
        Logging interval
    callback : Optional[Callable], default=None
        Callback function
    
    Returns
    -------
    history : Dict[str, List]
        Optimization history
    """
    # Exponential cooling schedule
    cooling_rate = (T_final / T_initial) ** (1.0 / max_iterations)
    
    history: Dict[str, List] = {
        "iteration": [],
        "action": [],
        "temperature": [],
    }
    
    # Current and best states
    S_current = spectral_action(
        triple.D,
        Lambda=Lambda,
        cutoff=cutoff,
        cutoff_param=cutoff_param,
        sparsity_weight=sparsity_weight,
    )
    D_best = triple.D.copy()
    S_best = S_current
    
    T = T_initial
    
    for iteration in range(max_iterations):
        # Propose a random perturbation
        N = triple.N
        perturbation = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        perturbation = (perturbation + perturbation.conj().T) / 2.0
        perturbation *= T  # Scale by temperature
        
        D_old = triple.D.copy()
        triple.D += perturbation
        triple.enforce_axioms()
        
        # Compute new action
        S_new = spectral_action(
            triple.D,
            Lambda=Lambda,
            cutoff=cutoff,
            cutoff_param=cutoff_param,
            sparsity_weight=sparsity_weight,
        )
        
        # Metropolis acceptance
        delta_S = S_new - S_current
        if delta_S < 0 or np.random.rand() < np.exp(-delta_S / T):
            # Accept
            S_current = S_new
            if S_current < S_best:
                S_best = S_current
                D_best = triple.D.copy()
        else:
            # Reject
            triple.D = D_old
        
        # Cool down
        T *= cooling_rate
        
        # Log
        if iteration % log_interval == 0:
            history["iteration"].append(iteration)
            history["action"].append(S_current)
            history["temperature"].append(T)
            
            print(
                f"Iter {iteration:5d}: S = {S_current:12.6f}, "
                f"S_best = {S_best:12.6f}, T = {T:.6e}"
            )
            
            if callback is not None:
                callback(iteration, S_current, triple)
    
    # Restore best solution
    triple.D = D_best
    
    return history
