"""
Tests for Distributed AHS Manager

Phase 2 test suite for distributed AHS management infrastructure.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from irh.core.v16.ahs import AlgorithmicHolonomicState
from irh.core.v16.distributed_ahs import (
    DistributedAHSManager,
    AHSMetadata,
    create_distributed_network
)


class TestAHSMetadata:
    """Test AHS metadata functionality."""
    
    def test_create_metadata(self):
        """Should create valid metadata."""
        meta = AHSMetadata(
            global_id="test_id_123",
            local_id=0,
            owner_rank=0
        )
        assert meta.global_id == "test_id_123"
        assert meta.local_id == 0
        assert meta.owner_rank == 0
    
    def test_to_dict(self):
        """Should serialize to dictionary."""
        meta = AHSMetadata(global_id="id1", local_id=5, owner_rank=2)
        d = meta.to_dict()
        assert d['global_id'] == "id1"
        assert d['local_id'] == 5
        assert d['owner_rank'] == 2
    
    def test_from_dict(self):
        """Should deserialize from dictionary."""
        d = {
            'global_id': 'id2',
            'local_id': 10,
            'owner_rank': 1,
            'creation_time': '2025-01-01T00:00:00',
            'complexity_Kt': 100.0,
            'checksum': 'abc123'
        }
        meta = AHSMetadata.from_dict(d)
        assert meta.global_id == 'id2'
        assert meta.local_id == 10


class TestDistributedAHSManager:
    """Test distributed AHS manager."""
    
    def test_init_single_node(self):
        """Should initialize in single-node mode."""
        manager = DistributedAHSManager()
        assert manager.rank == 0
        assert manager.size == 1
        assert len(manager) == 0
    
    def test_generate_global_id(self):
        """Should generate deterministic global IDs."""
        manager = DistributedAHSManager()
        
        state1 = AlgorithmicHolonomicState("0110", 1.5)
        state2 = AlgorithmicHolonomicState("0110", 1.5)
        
        id1 = manager.generate_global_id(state1)
        id2 = manager.generate_global_id(state2)
        
        # Same state should give same ID
        assert id1 == id2
    
    def test_different_states_different_ids(self):
        """Different states should have different IDs."""
        manager = DistributedAHSManager()
        
        state1 = AlgorithmicHolonomicState("0110", 1.5)
        state2 = AlgorithmicHolonomicState("1001", 2.0)
        
        id1 = manager.generate_global_id(state1)
        id2 = manager.generate_global_id(state2)
        
        assert id1 != id2
    
    def test_add_state(self):
        """Should add state and return global ID."""
        manager = DistributedAHSManager()
        state = AlgorithmicHolonomicState("0110", 1.5)
        
        global_id = manager.add_state(state)
        
        assert global_id is not None
        assert len(manager) == 1
        assert global_id in manager
    
    def test_add_duplicate_state(self):
        """Adding duplicate state should return same ID."""
        manager = DistributedAHSManager()
        state = AlgorithmicHolonomicState("0110", 1.5)
        
        id1 = manager.add_state(state)
        id2 = manager.add_state(state)
        
        assert id1 == id2
        assert len(manager) == 1  # Should not add twice
    
    def test_get_state(self):
        """Should retrieve state by global ID."""
        manager = DistributedAHSManager()
        state = AlgorithmicHolonomicState("0110", 1.5)
        
        global_id = manager.add_state(state)
        retrieved = manager.get_state(global_id)
        
        assert retrieved is not None
        assert retrieved.binary_string == state.binary_string
        assert retrieved.holonomic_phase == state.holonomic_phase
    
    def test_get_nonexistent_state(self):
        """Should return None for nonexistent state."""
        manager = DistributedAHSManager()
        result = manager.get_state("nonexistent_id")
        assert result is None
    
    def test_remove_state(self):
        """Should remove state."""
        manager = DistributedAHSManager()
        state = AlgorithmicHolonomicState("0110", 1.5)
        
        global_id = manager.add_state(state)
        assert len(manager) == 1
        
        removed = manager.remove_state(global_id)
        assert removed is True
        assert len(manager) == 0
        assert global_id not in manager
    
    def test_remove_nonexistent_state(self):
        """Should return False for nonexistent state removal."""
        manager = DistributedAHSManager()
        removed = manager.remove_state("nonexistent_id")
        assert removed is False
    
    def test_list_local_states(self):
        """Should list all local state IDs."""
        manager = DistributedAHSManager()
        
        states = [
            AlgorithmicHolonomicState("0110", 1.5),
            AlgorithmicHolonomicState("1001", 2.0),
            AlgorithmicHolonomicState("1111", 2.5)
        ]
        
        ids = [manager.add_state(s) for s in states]
        local_ids = manager.list_local_states()
        
        assert len(local_ids) == 3
        assert set(local_ids) == set(ids)
    
    def test_get_local_count(self):
        """Should return correct local count."""
        manager = DistributedAHSManager()
        assert manager.get_local_count() == 0
        
        manager.add_state(AlgorithmicHolonomicState("0110", 1.5))
        assert manager.get_local_count() == 1
        
        manager.add_state(AlgorithmicHolonomicState("1001", 2.0))
        assert manager.get_local_count() == 2
    
    def test_get_global_count_single_node(self):
        """Global count should equal local count in single-node mode."""
        manager = DistributedAHSManager()
        
        manager.add_state(AlgorithmicHolonomicState("0110", 1.5))
        manager.add_state(AlgorithmicHolonomicState("1001", 2.0))
        
        assert manager.get_global_count() == manager.get_local_count()
        assert manager.get_global_count() == 2
    
    def test_checkpoint_and_restore(self):
        """Should checkpoint and restore state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DistributedAHSManager(checkpoint_dir=Path(tmpdir))
            
            # Add states
            state1 = AlgorithmicHolonomicState("0110", 1.5)
            state2 = AlgorithmicHolonomicState("1001", 2.0)
            id1 = manager.add_state(state1)
            id2 = manager.add_state(state2)
            
            # Checkpoint
            checkpoint_path = manager.checkpoint("test_checkpoint")
            assert checkpoint_path.exists()
            
            # Create new manager and restore
            manager2 = DistributedAHSManager(checkpoint_dir=Path(tmpdir))
            restored = manager2.restore("test_checkpoint")
            
            assert restored is True
            assert manager2.get_local_count() == 2
            assert id1 in manager2
            assert id2 in manager2
    
    def test_restore_nonexistent_checkpoint(self):
        """Should return False for nonexistent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DistributedAHSManager(checkpoint_dir=Path(tmpdir))
            restored = manager.restore("nonexistent")
            assert restored is False
    
    def test_get_statistics(self):
        """Should return statistics dictionary."""
        manager = DistributedAHSManager()
        manager.add_state(AlgorithmicHolonomicState("0110", 1.5))
        
        stats = manager.get_statistics()
        
        assert 'rank' in stats
        assert 'size' in stats
        assert 'local_count' in stats
        assert 'global_count' in stats
        assert stats['local_count'] == 1
    
    def test_iteration(self):
        """Should iterate over states."""
        manager = DistributedAHSManager()
        
        states = [
            AlgorithmicHolonomicState("0110", 1.5),
            AlgorithmicHolonomicState("1001", 2.0)
        ]
        
        for s in states:
            manager.add_state(s)
        
        count = 0
        for global_id, state in manager:
            assert isinstance(global_id, str)
            assert isinstance(state, AlgorithmicHolonomicState)
            count += 1
        
        assert count == 2


class TestCreateDistributedNetwork:
    """Test network creation utility."""
    
    def test_create_network(self):
        """Should create network of N states."""
        manager = DistributedAHSManager()
        N = 10
        
        global_ids = create_distributed_network(N, manager, seed=42)
        
        assert len(global_ids) == N
        assert manager.get_local_count() == N
    
    def test_reproducible_with_seed(self):
        """Should be reproducible with seed."""
        manager1 = DistributedAHSManager()
        ids1 = create_distributed_network(5, manager1, seed=42)
        
        manager2 = DistributedAHSManager()
        ids2 = create_distributed_network(5, manager2, seed=42)
        
        # IDs should be same (deterministic generation)
        assert ids1 == ids2
    
    def test_gaussian_phase_distribution(self):
        """Should support Gaussian phase distribution."""
        manager = DistributedAHSManager()
        
        global_ids = create_distributed_network(
            20, manager, seed=42, phase_distribution="gaussian"
        )
        
        assert len(global_ids) == 20
        
        # Check phases are distributed
        phases = []
        for gid in global_ids:
            state = manager.get_state(gid)
            phases.append(state.holonomic_phase)
        
        # Should have some variance
        assert np.std(phases) > 0
    
    def test_uniform_phase_distribution(self):
        """Should support uniform phase distribution."""
        manager = DistributedAHSManager()
        
        global_ids = create_distributed_network(
            20, manager, seed=42, phase_distribution="uniform"
        )
        
        assert len(global_ids) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
