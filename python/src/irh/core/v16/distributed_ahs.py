"""
Distributed Algorithmic Holonomic State Manager

This module implements Phase 2 distributed AHS management infrastructure
for exascale simulations (N >= 10^12).

Key Features:
    - MPI-ready skeleton for distributed hash table (DHT)
    - Global ID generation and management
    - Checkpointing and fault tolerance hooks
    - CUDA array interface support for GPU transfer

Implementation Status: Phase 2 - MPI-Ready Skeleton (CPU baseline)

References:
    PHASE_2_STATUS.md: Distributed AHS/CRN interfaces requirement
    v16_IMPLEMENTATION_ROADMAP.md Phase 3, §3.2: Distributed computing
    [IRH-COMP-2025-02] §3: Full exascale implementation (future)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path
import pickle
import hashlib
import json
from datetime import datetime, timezone

# Python 3.10 compatibility: datetime.UTC only exists in 3.11+
UTC = timezone.utc

from .ahs import AlgorithmicHolonomicState


@dataclass
class AHSMetadata:
    """
    Metadata for distributed AHS management.
    
    Attributes:
        global_id: Globally unique identifier
        local_id: Local ID within this process/node
        owner_rank: MPI rank that owns this AHS (0 for single-node)
        creation_time: Timestamp of AHS creation
        complexity_Kt: Cached Kolmogorov complexity
        checksum: SHA256 checksum for validation
    """
    global_id: str
    local_id: int
    owner_rank: int = 0
    creation_time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    complexity_Kt: Optional[float] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'global_id': self.global_id,
            'local_id': self.local_id,
            'owner_rank': self.owner_rank,
            'creation_time': self.creation_time,
            'complexity_Kt': self.complexity_Kt,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> AHSMetadata:
        """Create from dictionary."""
        return cls(**data)


class DistributedAHSManager:
    """
    Manager for distributed Algorithmic Holonomic States.
    
    Phase 2 Implementation: CPU baseline with MPI-ready interface
    - Single-node DHT (dict-based)
    - Serialization and checkpointing
    - Global ID generation
    - Future: MPI-based distributed DHT
    
    Attributes:
        rank: MPI rank (0 for single-node)
        size: Number of MPI ranks (1 for single-node)
        local_states: Dictionary of locally-owned AHS
        metadata: Dictionary of AHS metadata
        checkpoint_dir: Directory for checkpoints
        
    Phase 3 Upgrade Path:
        - Replace local_states dict with MPI-based DHT
        - Implement ghost cell communication
        - Add dynamic load balancing
        - Enable fault tolerance with process recovery
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        enable_mpi: bool = False
    ):
        """
        Initialize distributed AHS manager.
        
        Args:
            checkpoint_dir: Directory for checkpointing (None for temp)
            enable_mpi: If True, initialize MPI (requires mpi4py)
        """
        # MPI configuration
        self.rank = 0
        self.size = 1
        self.comm = None
        
        if enable_mpi:
            try:
                from mpi4py import MPI
                self.comm = MPI.COMM_WORLD
                self.rank = self.comm.Get_rank()
                self.size = self.comm.Get_size()
            except ImportError:
                import warnings
                warnings.warn("MPI requested but mpi4py not available. Running in single-node mode.")
        
        # Local storage (Phase 2: dict-based, Phase 3: distributed DHT)
        self.local_states: Dict[str, AlgorithmicHolonomicState] = {}
        self.metadata: Dict[str, AHSMetadata] = {}
        
        # Checkpointing
        self.checkpoint_dir = checkpoint_dir or Path.cwd() / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.next_local_id = 0
        self.num_operations = 0
        
    def generate_global_id(self, state: AlgorithmicHolonomicState) -> str:
        """
        Generate globally unique ID for AHS.
        
        Uses content-based addressing: hash of (binary_string, phase)
        
        Args:
            state: AHS to generate ID for
            
        Returns:
            Global ID (hex string)
        """
        # Create deterministic ID from state content
        content = f"{state.binary_string}:{state.holonomic_phase:.15f}"
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def add_state(self, state: AlgorithmicHolonomicState) -> str:
        """
        Add AHS to distributed manager.
        
        Args:
            state: AHS to add
            
        Returns:
            Global ID of added state
            
        Phase 3: Will use MPI collective to determine owner rank
        """
        # Generate global ID
        global_id = self.generate_global_id(state)
        
        # Check if already exists
        if global_id in self.local_states:
            return global_id
        
        # Create metadata
        local_id = self.next_local_id
        self.next_local_id += 1
        
        # Compute checksum
        if isinstance(state.binary_string, bytes):
            checksum = hashlib.sha256(state.binary_string).hexdigest()
        else:
            checksum = hashlib.sha256(state.binary_string.encode('ascii')).hexdigest()
        
        metadata = AHSMetadata(
            global_id=global_id,
            local_id=local_id,
            owner_rank=self.rank,
            complexity_Kt=state.complexity_Kt,
            checksum=checksum
        )
        
        # Store locally
        self.local_states[global_id] = state
        self.metadata[global_id] = metadata
        self.num_operations += 1
        
        return global_id
    
    def get_state(self, global_id: str) -> Optional[AlgorithmicHolonomicState]:
        """
        Retrieve AHS by global ID.
        
        Args:
            global_id: Global ID to lookup
            
        Returns:
            AHS if found, None otherwise
            
        Phase 3: Will use MPI communication to fetch from remote ranks
        """
        return self.local_states.get(global_id)
    
    def remove_state(self, global_id: str) -> bool:
        """
        Remove AHS from manager.
        
        Args:
            global_id: Global ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if global_id in self.local_states:
            del self.local_states[global_id]
            del self.metadata[global_id]
            self.num_operations += 1
            return True
        return False
    
    def list_local_states(self) -> List[str]:
        """
        List all global IDs of locally-owned states.
        
        Returns:
            List of global IDs
        """
        return list(self.local_states.keys())
    
    def get_local_count(self) -> int:
        """Get number of locally-owned states."""
        return len(self.local_states)
    
    def get_global_count(self) -> int:
        """
        Get total number of states across all ranks.
        
        Phase 2: Same as local count (single node)
        Phase 3: MPI_Allreduce to sum counts
        """
        if self.comm is not None:
            from mpi4py import MPI
            local_count = self.get_local_count()
            global_count = self.comm.allreduce(local_count, op=MPI.SUM)
            return global_count
        else:
            return self.get_local_count()
    
    def checkpoint(self, checkpoint_name: str = "ahs_checkpoint") -> Path:
        """
        Save current state to checkpoint file.
        
        Args:
            checkpoint_name: Name for checkpoint file
            
        Returns:
            Path to checkpoint file
            
        Phase 3: Coordinate checkpointing across all ranks
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_rank{self.rank}.pkl"
        
        checkpoint_data = {
            'rank': self.rank,
            'size': self.size,
            'states': self.local_states,
            'metadata': {k: v.to_dict() for k, v in self.metadata.items()},
            'next_local_id': self.next_local_id,
            'num_operations': self.num_operations,
            'timestamp': datetime.now(UTC).isoformat()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        return checkpoint_path
    
    def restore(self, checkpoint_name: str = "ahs_checkpoint") -> bool:
        """
        Restore state from checkpoint file.
        
        Args:
            checkpoint_name: Name of checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_rank{self.rank}.pkl"
        
        if not checkpoint_path.exists():
            return False
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Restore state
        self.local_states = checkpoint_data['states']
        self.metadata = {k: AHSMetadata.from_dict(v) for k, v in checkpoint_data['metadata'].items()}
        self.next_local_id = checkpoint_data['next_local_id']
        self.num_operations = checkpoint_data['num_operations']
        
        return True
    
    def get_statistics(self) -> dict:
        """
        Get manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'rank': self.rank,
            'size': self.size,
            'local_count': self.get_local_count(),
            'global_count': self.get_global_count(),
            'num_operations': self.num_operations,
            'checkpoint_dir': str(self.checkpoint_dir)
        }
    
    def __iter__(self) -> Iterator[Tuple[str, AlgorithmicHolonomicState]]:
        """Iterate over locally-owned states."""
        return iter(self.local_states.items())
    
    def __len__(self) -> int:
        """Return number of locally-owned states."""
        return len(self.local_states)
    
    def __contains__(self, global_id: str) -> bool:
        """Check if global ID exists locally."""
        return global_id in self.local_states


# Utility functions for distributed operations

def create_distributed_network(
    N: int,
    manager: DistributedAHSManager,
    seed: Optional[int] = None,
    phase_distribution: str = "uniform"
) -> List[str]:
    """
    Create network of N distributed AHS.
    
    Args:
        N: Number of states to create
        manager: DistributedAHSManager instance
        seed: Random seed for reproducibility
        phase_distribution: Phase distribution ("uniform", "gaussian")
        
    Returns:
        List of global IDs for created states
        
    Phase 3: Will distribute creation across MPI ranks
    """
    import numpy as np
    
    rng = np.random.RandomState(seed)
    global_ids = []
    
    # Determine how many states this rank should create
    # Phase 2: Single rank creates all
    # Phase 3: Distribute N across ranks
    if manager.comm is not None:
        states_per_rank = N // manager.size
        extra = N % manager.size
        my_count = states_per_rank + (1 if manager.rank < extra else 0)
    else:
        my_count = N
    
    for i in range(my_count):
        # Generate random binary string (50-200 bits)
        length = rng.randint(50, 201)
        binary_str = ''.join(rng.choice(['0', '1']) for _ in range(length))
        
        # Generate phase
        if phase_distribution == "uniform":
            phase = rng.uniform(0, 2 * np.pi)
        elif phase_distribution == "gaussian":
            phase = rng.normal(np.pi, np.pi / 3) % (2 * np.pi)
        else:
            phase = rng.uniform(0, 2 * np.pi)
        
        # Create state
        state = AlgorithmicHolonomicState(binary_str, phase)
        
        # Add to manager
        global_id = manager.add_state(state)
        global_ids.append(global_id)
    
    return global_ids
