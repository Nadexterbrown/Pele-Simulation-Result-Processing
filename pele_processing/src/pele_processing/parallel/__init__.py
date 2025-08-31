"""
Parallel processing module for the Pele processing system.
"""

# Coordination
from .coordinator import (
    MPICoordinator, SequentialCoordinator, ThreadPoolCoordinator,
    create_coordinator
)

# Workers
from .workers import (
    WorkerBase, MPIWorker, ThreadWorker, ProcessWorker,
    WorkerManager, TaskQueue, WorkerPool
)

# Distribution
from .distribution import (
    DistributionStrategy, RoundRobinDistribution, ChunkDistribution,
    LoadBalancedDistribution, RandomDistribution, AdaptiveDistribution,
    MPIDistributor, ThreadDistributor,
    create_distributor, create_strategy
)

# Strategies
from .strategies import (
    ProcessingStrategy, SequentialStrategy, MPIStrategy, ThreadPoolStrategy,
    HybridStrategy, AdaptiveStrategy,
    create_processing_strategy, create_default_adaptive_strategy
)

__all__ = [
    # Coordinators
    'MPICoordinator',
    'SequentialCoordinator',
    'ThreadPoolCoordinator',
    'create_coordinator',

    # Workers
    'WorkerBase',
    'MPIWorker',
    'ThreadWorker',
    'ProcessWorker',
    'WorkerManager',
    'TaskQueue',
    'WorkerPool',

    # Distribution
    'DistributionStrategy',
    'RoundRobinDistribution',
    'ChunkDistribution',
    'LoadBalancedDistribution',
    'RandomDistribution',
    'AdaptiveDistribution',
    'MPIDistributor',
    'ThreadDistributor',
    'create_distributor',
    'create_strategy',

    # Strategies
    'ProcessingStrategy',
    'SequentialStrategy',
    'MPIStrategy',
    'ThreadPoolStrategy',
    'HybridStrategy',
    'AdaptiveStrategy',
    'create_processing_strategy',
    'create_default_adaptive_strategy'
]

__version__ = "1.0.0"