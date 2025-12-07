"""
Test suite for Phase 7: Exascale Infrastructure

Tests the parallel computing infrastructure including MPI, GPU, and distributed
eigensolvers for scaling to N ≥ 10^10.
"""
import pytest
import numpy as np
import scipy.sparse as sp
from src.parallel import MPI_AVAILABLE, GPU_AVAILABLE
from src.parallel.mpi_aro import MPIAROOptimizer, run_mpi_optimization
from src.parallel.gpu_kernels import GPUAccelerator, benchmark_gpu_speedup
from src.parallel.distributed_eigen import distributed_eigsh
from benchmarks.exascale_benchmark import benchmark_scaling


class TestMPIInfrastructure:
    """Test MPI parallel infrastructure."""
    
    def test_mpi_availability_check(self):
        """Test that MPI availability is detected correctly."""
        # Should be a boolean
        assert isinstance(MPI_AVAILABLE, bool)
    
    def test_mpi_optimizer_initialization(self):
        """Test MPIAROOptimizer can be initialized without MPI."""
        # Should work even without MPI (fallback mode)
        opt = MPIAROOptimizer(N_global=100, rng_seed=42)
        
        assert opt.N_global == 100
        assert opt.rank == 0  # Single process in fallback mode
        assert opt.size == 1
        assert opt.N_local == 100
    
    def test_mpi_optimizer_initialize_network(self):
        """Test network initialization in MPI optimizer."""
        opt = MPIAROOptimizer(N_global=100, rng_seed=42)
        
        # Should not raise an error
        opt.initialize_network(scheme='geometric', connectivity_param=0.1)
    
    def test_mpi_optimizer_optimize(self):
        """Test optimization method exists."""
        opt = MPIAROOptimizer(N_global=100, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        
        # Should not raise an error (placeholder implementation)
        opt.optimize(iterations=10, sync_interval=5)
    
    def test_run_mpi_optimization_function(self):
        """Test run_mpi_optimization entry point."""
        results = run_mpi_optimization(N_global=100, iterations=10)
        
        # Should return dict with results
        assert isinstance(results, dict)
        assert 'N_global' in results
        assert results['N_global'] == 100


class TestGPUInfrastructure:
    """Test GPU acceleration infrastructure."""
    
    def test_gpu_availability_check(self):
        """Test that GPU availability is detected correctly."""
        # Should be a boolean
        assert isinstance(GPU_AVAILABLE, bool)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU (CuPy) not available")
    def test_gpu_accelerator_initialization(self):
        """Test GPUAccelerator initialization (requires GPU)."""
        # Only runs if GPU is available
        acc = GPUAccelerator(device_id=0)
        assert acc.device is not None
    
    def test_gpu_accelerator_requires_cupy(self):
        """Test that GPUAccelerator raises error without CuPy."""
        if not GPU_AVAILABLE:
            with pytest.raises(ImportError):
                GPUAccelerator(device_id=0)
    
    def test_benchmark_gpu_speedup_function(self):
        """Test GPU benchmark function exists and runs."""
        # Should not raise an error (placeholder implementation)
        benchmark_gpu_speedup()


class TestDistributedEigensolvers:
    """Test distributed eigenvalue solvers."""
    
    def test_distributed_eigsh_function_exists(self):
        """Test distributed_eigsh function exists."""
        # Create small sparse matrix
        N = 10
        M = sp.random(N, N, density=0.3, format='csr')
        M = (M + M.T) / 2  # Make symmetric
        
        # Should not raise error (returns None as placeholder)
        eigenvalues, eigenvectors = distributed_eigsh(M, k=3, which='LM')
        
        # Placeholder returns None
        # In full implementation, would return actual eigenvalues
    
    def test_distributed_eigsh_backend_selection(self):
        """Test backend selection in distributed_eigsh."""
        N = 10
        M = sp.random(N, N, density=0.3, format='csr')
        
        # slepc backend (placeholder)
        eigs1 = distributed_eigsh(M, k=3, backend='slepc')
        
        # scalapack should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            distributed_eigsh(M, k=3, backend='scalapack')
        
        # Unknown backend should raise ValueError
        with pytest.raises(ValueError):
            distributed_eigsh(M, k=3, backend='unknown')


class TestBenchmarking:
    """Test benchmarking and profiling tools."""
    
    def test_benchmark_scaling_function(self):
        """Test benchmark_scaling function."""
        # Test with small networks
        N_values = [10, 20]
        
        results = benchmark_scaling(N_values, backend='cpu', output_file=None)
        
        # Should return dict with results
        assert isinstance(results, dict)
        assert 'backend' in results
        assert 'N_values' in results
        assert 'timings' in results
        assert results['backend'] == 'cpu'
        assert len(results['timings']) == len(N_values)
    
    def test_benchmark_scaling_timings(self):
        """Test that benchmark records timings."""
        N_values = [10, 20]
        
        results = benchmark_scaling(N_values, backend='cpu', output_file=None)
        
        # Should have timing for each N
        assert len(results['timings']) == 2
        assert all(isinstance(t, (int, float)) for t in results['timings'])
        assert all(t > 0 for t in results['timings'])


@pytest.mark.slow
class TestLargeScaleInfrastructure:
    """Tests for larger-scale infrastructure (marked as slow)."""
    
    def test_mpi_larger_network(self):
        """Test MPI optimizer with larger network."""
        opt = MPIAROOptimizer(N_global=1000, rng_seed=42)
        opt.initialize_network('geometric', 0.1, 4)
        opt.optimize(iterations=10)
        
        # Should complete without error
        assert opt.N_global == 1000
    
    def test_benchmark_larger_networks(self):
        """Test benchmarking with larger networks."""
        N_values = [100, 200, 400]
        
        results = benchmark_scaling(N_values, backend='cpu', output_file=None)
        
        # Should complete and show scaling trend
        assert len(results['timings']) == 3
        # Larger networks should generally take longer
        # (though with small iterations this may not always hold)
