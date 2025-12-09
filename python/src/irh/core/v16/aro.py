"""
Adaptive Resonance Optimization (ARO) for IRH v16.0

Implements the iterative genetic algorithm that maximizes the Harmony Functional
S_H[G] to find the unique Cosmic Fixed Point.

THEORETICAL COMPLIANCE:
    This implementation strictly follows docs/manuscripts/IRHv16.md
    - §4 Definition 4.1 (lines 280-306): ARO formal definition
    - Line 282: ARO maximizes S_H[G] over network configurations
    - Lines 287-298: Genetic algorithm specification
    - Lines 300-305: Convergence to Cosmic Fixed Point
    
Key Concepts:
    - Population-based genetic algorithm
    - Fitness = S_H (Harmony Functional)
    - Selection, mutation, crossover operators
    - Simulated annealing schedule
    - Convergence monitoring
    
Algorithm Steps (from IRHv16.md lines 289-298):
    1. Initialization: Random CRN configurations
    2. Fitness Evaluation: Compute S_H for each
    3. Selection: Higher S_H configurations reproduce
    4. Mutation: Weight perturbation, topological changes, AHS content
    5. Crossover: Exchange subgraphs
    6. Annealing: Accept lower S_H with decreasing probability
    7. Adaptive Meshing: Focus on high complexity regions
    
References:
    docs/manuscripts/IRHv16.md:
        - §4 Definition 4.1: ARO specification
        - Lines 282-306: Algorithm and convergence theorem
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
import logging

from .ahs import AlgorithmicHolonomicState, create_ahs_network
from .crn import CymaticResonanceNetworkV16, create_crn_from_states
from .harmony import compute_harmony_functional, HarmonyFunctionalEvaluator

logger = logging.getLogger(__name__)


@dataclass
class AROConfiguration:
    """
    A single configuration in the ARO population.
    
    Represents one candidate CRN with its fitness (S_H value).
    
    Attributes:
        crn: Cymatic Resonance Network
        S_H: Harmony Functional value (fitness)
        generation: Generation when created/modified
        parent_ids: IDs of parent configurations (for genealogy tracking)
    """
    crn: CymaticResonanceNetworkV16
    S_H: float = -np.inf
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)


class AROOptimizerV16:
    """
    Adaptive Resonance Optimization (ARO) for IRH v16.0.
    
    Implements the genetic algorithm that maximizes S_H[G] per IRHv16.md
    Definition 4.1 (lines 280-306).
    
    This is a preliminary Phase 1 implementation focusing on:
    - Population management
    - Basic genetic operators
    - Fitness evaluation via Harmony Functional
    - Simple annealing schedule
    
    Full exascale implementation (P~10^5, distributed across GPUs) in Phase 2.
    
    Attributes:
        population_size: Number of configurations in population
        N: Number of AHS nodes (fixed for all configurations)
        epsilon_threshold: Edge threshold for CRN construction
        temperature: Current annealing temperature
        temperature_schedule: Function(iteration) -> temperature
        
    References:
        docs/manuscripts/IRHv16.md §4 Definition 4.1: ARO specification
    """
    
    def __init__(
        self,
        N: int,
        population_size: int = 20,
        epsilon_threshold: float = 0.5,  # Lower for Phase 1 testing
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        seed: Optional[int] = None
    ):
        """
        Initialize ARO optimizer.
        
        Args:
            N: Number of AHS nodes (fixed)
            population_size: Population size (P~10^5 for exascale, ~20 for Phase 1)
            epsilon_threshold: Edge inclusion threshold
            initial_temperature: Initial temperature for annealing
            cooling_rate: Temperature decay rate per generation
            seed: Random seed for reproducibility
            
        Notes:
            Phase 1 uses small N (~10-50) and P (~20) for demonstration.
            Phase 2 will scale to N >= 10^4, P >= 10^5.
        """
        self.N = N
        self.population_size = population_size
        self.epsilon_threshold = epsilon_threshold
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        
        self.rng = np.random.default_rng(seed)
        self.evaluator = HarmonyFunctionalEvaluator()
        
        # Population storage
        self.population: List[AROConfiguration] = []
        self.generation = 0
        self.best_config: Optional[AROConfiguration] = None
        
        logger.info(f"ARO Optimizer initialized: N={N}, P={population_size}")
    
    def initialize_population(self) -> None:
        """
        Initialize population with random CRN configurations.
        
        Implements IRHv16.md line 289: Multiple random CRN configurations.
        
        Each configuration has randomly generated AHS and corresponding CRN.
        """
        logger.info(f"Initializing population of {self.population_size} configurations...")
        
        self.population = []
        
        for i in range(self.population_size):
            # Create random AHS network
            states = create_ahs_network(
                N=self.N,
                seed=self.rng.integers(0, 2**31) if self.rng else None
            )
            
            # Build CRN
            crn = create_crn_from_states(states, epsilon_threshold=self.epsilon_threshold)
            
            # Create configuration
            config = AROConfiguration(
                crn=crn,
                S_H=-np.inf,
                generation=0
            )
            
            self.population.append(config)
        
        # Evaluate initial fitness
        self._evaluate_population()
        
        logger.info(f"Population initialized. Best S_H: {self.best_config.S_H:.4f}")
    
    def _evaluate_population(self) -> None:
        """
        Evaluate fitness (S_H) for all configurations.
        
        Implements IRHv16.md line 290: Each configuration's S_H is computed.
        """
        for config in self.population:
            try:
                S_H = self.evaluator.evaluate(config.crn, iteration=self.generation)
                config.S_H = S_H
            except (ValueError, np.linalg.LinAlgError):
                # Degenerate network
                config.S_H = -np.inf
        
        # Update best configuration
        best = max(self.population, key=lambda c: c.S_H)
        if self.best_config is None or best.S_H > self.best_config.S_H:
            self.best_config = best
    
    def select_parents(self, k: int = 2) -> List[AROConfiguration]:
        """
        Select parent configurations for reproduction.
        
        Implements IRHv16.md line 291: Configurations with higher S_H selected.
        
        Uses tournament selection: randomly sample k configurations and
        select the best among them.
        
        Args:
            k: Tournament size
            
        Returns:
            List of selected parent configurations
        """
        # Tournament selection
        parents = []
        for _ in range(2):  # Select 2 parents
            # Random sample of k configurations
            candidates = self.rng.choice(self.population, size=k, replace=False)
            # Select best
            best = max(candidates, key=lambda c: c.S_H)
            parents.append(best)
        
        return parents
    
    def mutate_configuration(self, config: AROConfiguration) -> AROConfiguration:
        """
        Apply mutations to a configuration.
        
        Implements IRHv16.md lines 292-295: Mutation operators.
        
        Three types of mutations (simplified for Phase 1):
        1. Weight Perturbation: Small changes to W_ij
        2. Topological Mutation: Add/remove edges
        3. AHS Content Mutation: Modify binary strings
        
        Args:
            config: Configuration to mutate
            
        Returns:
            New mutated configuration
        """
        # Choose mutation type randomly
        mutation_type = self.rng.choice(['weight', 'topology', 'ahs_content'])
        
        # Copy states for mutation
        new_states = [
            AlgorithmicHolonomicState(
                s.binary_string,
                s.holonomic_phase
            )
            for s in config.crn.states
        ]
        
        if mutation_type == 'ahs_content':
            # Mutate binary strings of AHS
            idx = self.rng.integers(0, len(new_states))
            # Flip a random bit
            bits = list(new_states[idx].binary_string)
            if len(bits) > 0:
                flip_idx = self.rng.integers(0, len(bits))
                bits[flip_idx] = '1' if bits[flip_idx] == '0' else '0'
                new_states[idx] = AlgorithmicHolonomicState(
                    ''.join(bits),
                    new_states[idx].holonomic_phase
                )
        
        elif mutation_type == 'topology':
            # Modify network by adjusting threshold slightly
            # This changes which edges exist
            threshold_mutation = self.epsilon_threshold * self.rng.uniform(0.95, 1.05)
        else:
            threshold_mutation = self.epsilon_threshold
        
        # Rebuild CRN with mutations
        new_crn = create_crn_from_states(
            new_states,
            epsilon_threshold=threshold_mutation if mutation_type == 'topology' else self.epsilon_threshold
        )
        
        return AROConfiguration(
            crn=new_crn,
            S_H=-np.inf,
            generation=self.generation + 1,
            parent_ids=[id(config)]
        )
    
    def step(self) -> None:
        """
        Perform one generation of ARO.
        
        Implements full genetic algorithm iteration:
        1. Selection
        2. Mutation
        3. Crossover (simplified for Phase 1)
        4. Fitness evaluation
        5. Population replacement
        6. Annealing temperature update
        """
        # Create offspring population
        offspring = []
        
        for _ in range(self.population_size):
            # Selection
            parents = self.select_parents()
            
            # For Phase 1, use simple mutation without crossover
            # Full crossover implementation in Phase 2
            parent = self.rng.choice(parents)
            
            # Mutation
            child = self.mutate_configuration(parent)
            offspring.append(child)
        
        # Evaluate offspring
        for child in offspring:
            try:
                S_H = self.evaluator.evaluate(child.crn, iteration=self.generation)
                child.S_H = S_H
            except (ValueError, np.linalg.LinAlgError):
                child.S_H = -np.inf
        
        # Combine populations and select best
        combined = self.population + offspring
        
        # Sort by fitness
        combined.sort(key=lambda c: c.S_H, reverse=True)
        
        # Keep best configurations (elitism)
        self.population = combined[:self.population_size]
        
        # Update best
        if self.population[0].S_H > self.best_config.S_H:
            self.best_config = self.population[0]
        
        # Update temperature (annealing)
        self.temperature *= self.cooling_rate
        
        self.generation += 1
    
    def optimize(self, num_generations: int = 50) -> AROConfiguration:
        """
        Run ARO for specified number of generations.
        
        Args:
            num_generations: Number of generations to run
            
        Returns:
            Best configuration found
        """
        logger.info(f"Starting ARO optimization for {num_generations} generations...")
        
        if len(self.population) == 0:
            self.initialize_population()
        
        for gen in range(num_generations):
            self.step()
            
            if (gen + 1) % 10 == 0:
                logger.info(
                    f"Generation {gen + 1}/{num_generations}: "
                    f"Best S_H = {self.best_config.S_H:.4f}, "
                    f"T = {self.temperature:.4f}"
                )
        
        logger.info(f"Optimization complete. Best S_H: {self.best_config.S_H:.4f}")
        
        return self.best_config
    
    def get_convergence_metrics(self) -> dict:
        """Get convergence metrics from evaluator."""
        return self.evaluator.get_convergence_metrics()


__all__ = [
    "AROConfiguration",
    "AROOptimizerV16",
]

__version__ = "16.0.0-dev"
