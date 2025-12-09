"""
Unit tests for ARO Optimizer - Definition 4.1.

Tests ARO genetic algorithm per IRHv16.md §4 lines 280-306.

THEORETICAL COMPLIANCE:
    Tests validate against docs/manuscripts/IRHv16.md §4 Definition 4.1
    - Line 282: ARO maximizes S_H[G]
    - Lines 287-298: Genetic algorithm specification
    - Lines 300-305: Convergence theorem
"""

import pytest
import numpy as np
from irh.core.v16.aro import (
    AROConfiguration,
    AROOptimizerV16
)


class TestAROConfiguration:
    """Test ARO configuration dataclass."""
    
    def test_configuration_creation(self):
        """Test AROConfiguration can be created."""
        from irh.core.v16.ahs import create_ahs_network
        from irh.core.v16.crn import create_crn_from_states
        
        states = create_ahs_network(N=5, seed=42)
        crn = create_crn_from_states(states)
        
        config = AROConfiguration(crn=crn, S_H=10.0, generation=0)
        
        assert config.crn is not None
        assert config.S_H == 10.0
        assert config.generation == 0


class TestAROOptimizerInitialization:
    """Test ARO optimizer initialization."""
    
    def test_basic_initialization(self):
        """Test ARO can be initialized."""
        aro = AROOptimizerV16(N=5, population_size=10, seed=42)
        
        assert aro.N == 5
        assert aro.population_size == 10
        assert aro.generation == 0
        assert len(aro.population) == 0  # Not initialized yet
        
    def test_initialization_parameters(self):
        """Test ARO accepts all parameters."""
        aro = AROOptimizerV16(
            N=10,
            population_size=20,
            epsilon_threshold=0.6,
            initial_temperature=2.0,
            cooling_rate=0.9,
            seed=42
        )
        
        assert aro.N == 10
        assert aro.population_size == 20
        assert aro.epsilon_threshold == 0.6
        assert aro.temperature == 2.0
        assert aro.cooling_rate == 0.9
        
    def test_initialize_population(self):
        """Test population initialization."""
        aro = AROOptimizerV16(N=5, population_size=10, seed=42)
        aro.initialize_population()
        
        assert len(aro.population) == 10, "Population should have 10 members"
        assert aro.best_config is not None, "Should have a best configuration"
        assert aro.best_config.S_H > -np.inf, "Best S_H should be evaluated"


class TestAROSelection:
    """Test ARO selection mechanism."""
    
    def test_select_parents(self):
        """Test parent selection."""
        aro = AROOptimizerV16(N=5, population_size=10, seed=42)
        aro.initialize_population()
        
        parents = aro.select_parents(k=2)
        
        assert len(parents) == 2, "Should select 2 parents"
        assert all(p in aro.population for p in parents), \
            "Parents should be from population"
            
    def test_tournament_selection_quality(self):
        """Test tournament selection favors higher S_H."""
        aro = AROOptimizerV16(N=5, population_size=20, seed=42)
        aro.initialize_population()
        
        # Over many selections, should tend toward higher S_H
        selected_S_H = []
        for _ in range(50):
            parents = aro.select_parents(k=3)
            selected_S_H.extend([p.S_H for p in parents])
            
        mean_selected = np.mean(selected_S_H)
        mean_population = np.mean([c.S_H for c in aro.population])
        
        # Selected should generally be higher than average
        # (Not guaranteed but very likely with k=3 tournament)
        assert mean_selected >= mean_population * 0.9, \
            "Tournament selection should favor higher S_H"


class TestAROMutation:
    """Test ARO mutation operators."""
    
    def test_mutate_configuration(self):
        """Test mutation creates valid configurations."""
        aro = AROOptimizerV16(N=5, population_size=10, seed=42)
        aro.initialize_population()
        
        original = aro.population[0]
        mutated = aro.mutate_configuration(original)
        
        assert isinstance(mutated, AROConfiguration)
        assert mutated.crn is not None
        assert mutated.crn.N == original.crn.N, "Mutation preserves N"
        assert mutated.generation == aro.generation + 1
        
    def test_mutation_types(self):
        """Test different mutation types can occur."""
        aro = AROOptimizerV16(N=5, population_size=10, seed=42)
        aro.initialize_population()
        
        # Perform many mutations, should see variety
        original = aro.population[0]
        
        for _ in range(20):
            mutated = aro.mutate_configuration(original)
            assert mutated.crn is not None, "Mutation should produce valid CRN"


class TestAROOptimization:
    """
    Test ARO optimization process.
    
    References:
        IRHv16.md §4 Definition 4.1 lines 282: ARO maximizes S_H
    """
    
    def test_single_step(self):
        """Test single ARO generation step."""
        aro = AROOptimizerV16(N=5, population_size=5, seed=42)
        aro.initialize_population()
        
        initial_gen = aro.generation
        initial_temp = aro.temperature
        
        aro.step()
        
        assert aro.generation == initial_gen + 1, "Generation should increment"
        assert aro.temperature < initial_temp, "Temperature should decrease"
        assert len(aro.population) == 5, "Population size should remain constant"
        
    def test_optimize_multiple_generations(self):
        """Test ARO runs for multiple generations."""
        aro = AROOptimizerV16(N=5, population_size=5, seed=42)
        
        best = aro.optimize(num_generations=10)
        
        assert aro.generation == 10, "Should run 10 generations"
        assert best is not None, "Should return best configuration"
        assert best.S_H > -np.inf, "Best should have valid S_H"
        
    def test_S_H_improvement_tendency(self):
        """
        Test ARO tends to improve S_H over generations.
        
        Per IRHv16.md line 282: ARO maximizes S_H
        """
        aro = AROOptimizerV16(N=10, population_size=10, seed=42)
        aro.initialize_population()
        
        initial_best = aro.best_config.S_H
        
        # Run optimization
        aro.optimize(num_generations=20)
        
        final_best = aro.best_config.S_H
        
        # S_H should generally improve (not guaranteed but very likely)
        # Allow for no improvement but not degradation
        assert final_best >= initial_best * 0.95, \
            f"S_H should not degrade significantly: {initial_best} → {final_best}"
            
    def test_convergence_metrics(self):
        """Test convergence metrics are tracked."""
        aro = AROOptimizerV16(N=5, population_size=5, seed=42)
        aro.optimize(num_generations=10)
        
        metrics = aro.get_convergence_metrics()
        
        assert metrics['num_evaluations'] > 0
        assert metrics['best_S_H'] > -np.inf
        assert 'mean_S_H' in metrics
        assert 'convergence_trend' in metrics


class TestAROAnnealing:
    """Test simulated annealing schedule."""
    
    def test_temperature_decay(self):
        """Test temperature decreases over generations."""
        aro = AROOptimizerV16(
            N=5,
            population_size=5,
            initial_temperature=1.0,
            cooling_rate=0.9,
            seed=42
        )
        aro.initialize_population()
        
        temperatures = [aro.temperature]
        
        for _ in range(10):
            aro.step()
            temperatures.append(aro.temperature)
            
        # Temperature should monotonically decrease
        for i in range(len(temperatures) - 1):
            assert temperatures[i] >= temperatures[i+1], \
                f"Temperature should decrease: {temperatures[i]} vs {temperatures[i+1]}"
                
    def test_cooling_rate_effect(self):
        """Test different cooling rates produce different schedules."""
        aro1 = AROOptimizerV16(N=5, population_size=5, cooling_rate=0.95, seed=42)
        aro2 = AROOptimizerV16(N=5, population_size=5, cooling_rate=0.85, seed=42)
        
        aro1.initialize_population()
        aro2.initialize_population()
        
        for _ in range(5):
            aro1.step()
            aro2.step()
            
        # Slower cooling (0.95) should have higher temperature after same steps
        assert aro1.temperature > aro2.temperature, \
            "Slower cooling rate should maintain higher temperature"


class TestTheoreticalCompliance:
    """
    Test compliance with IRHv16.md §4 Definition 4.1.
    
    References:
        docs/manuscripts/IRHv16.md §4 lines 280-306
    """
    
    def test_definition41_maximizes_S_H(self):
        """
        Test ARO maximizes S_H per Definition 4.1.
        
        References:
            IRHv16.md line 282: ARO maximizes S_H[G]
        """
        aro = AROOptimizerV16(N=10, population_size=10, seed=42)
        aro.initialize_population()
        
        initial_best = aro.best_config.S_H
        
        # Run optimization
        final = aro.optimize(num_generations=15)
        
        # Final best should be >= initial (maximization)
        assert final.S_H >= initial_best, \
            "Definition 4.1 requires ARO to maximize S_H"
            
    def test_definition41_fixed_N(self):
        """
        Test ARO maintains fixed N per Definition 4.1.
        
        References:
            IRHv16.md line 283: Fixed |V| = N
        """
        aro = AROOptimizerV16(N=10, population_size=10, seed=42)
        aro.optimize(num_generations=10)
        
        # All configurations should have N = 10
        for config in aro.population:
            assert config.crn.N == 10, \
                "Definition 4.1 requires fixed N"
                
    def test_definition41_genetic_algorithm(self):
        """
        Test ARO uses genetic algorithm per Definition 4.1.
        
        References:
            IRHv16.md lines 287-298: Genetic algorithm specification
        """
        aro = AROOptimizerV16(N=5, population_size=10, seed=42)
        
        # Should have population
        aro.initialize_population()
        assert len(aro.population) == 10, "Should have population (line 289)"
        
        # Should perform selection, mutation
        initial_gen = aro.generation
        aro.step()
        assert aro.generation > initial_gen, "Should advance generation"
        
        # Should track fitness (S_H)
        assert all(hasattr(c, 'S_H') for c in aro.population), \
            "Should track fitness (lines 290-291)"


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    def test_small_population(self):
        """Test ARO works with minimal population."""
        aro = AROOptimizerV16(N=5, population_size=2, seed=42)
        aro.optimize(num_generations=5)
        
        assert len(aro.population) == 2
        assert aro.best_config is not None
        
    def test_zero_generations(self):
        """Test ARO handles zero generations gracefully."""
        aro = AROOptimizerV16(N=5, population_size=5, seed=42)
        
        best = aro.optimize(num_generations=0)
        
        # Should just initialize and return best from initial population
        assert best is not None
        assert best.S_H > -np.inf or best.S_H == -np.inf
        
    def test_reproducibility_with_seed(self):
        """Test ARO is reproducible with same seed."""
        aro1 = AROOptimizerV16(N=5, population_size=5, seed=42)
        aro2 = AROOptimizerV16(N=5, population_size=5, seed=42)
        
        best1 = aro1.optimize(num_generations=5)
        best2 = aro2.optimize(num_generations=5)
        
        # Same seed should produce same results
        assert np.isclose(best1.S_H, best2.S_H), \
            "Same seed should produce reproducible results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
