"""
High-level parallel processing strategies.
"""
from typing import List, Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import time

from ..core.domain import ProcessingResult, ProcessingBatch
from ..core.interfaces import Logger
from ..core.exceptions import ParallelProcessingError
from .coordinator import MPICoordinator, SequentialCoordinator, ThreadPoolCoordinator
from .distribution import DistributionStrategy, RoundRobinDistribution, ChunkDistribution
from .workers import WorkerManager


class ProcessingStrategy(ABC):
    """Base class for processing strategies."""

    @abstractmethod
    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute processing strategy."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        pass


class SequentialStrategy(ProcessingStrategy):
    """Sequential processing strategy."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.coordinator = SequentialCoordinator(logger)
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute sequential processing."""
        start_time = time.time()

        batch = self.coordinator.coordinate_processing(work_items, processor_func)

        end_time = time.time()
        self._stats = {
            'strategy': 'sequential',
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(batch.get_successful_results()),
            'throughput': len(batch.results) / (end_time - start_time)
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()


class MPIStrategy(ProcessingStrategy):
    """MPI-based parallel processing strategy."""

    def __init__(self, distribution_strategy: Optional[DistributionStrategy] = None,
                 timeout: float = 300.0, logger: Optional[Logger] = None):
        self.distribution_strategy = distribution_strategy or RoundRobinDistribution()
        self.logger = logger
        self.coordinator = MPICoordinator(logger, timeout)
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute MPI parallel processing."""
        start_time = time.time()

        # Let coordinator handle MPI distribution and gathering
        batch = self.coordinator.coordinate_processing(work_items, processor_func)

        end_time = time.time()

        # Only root process computes final statistics
        if self.coordinator.rank == 0:
            self._stats = {
                'strategy': 'mpi',
                'total_time': end_time - start_time,
                'items_processed': len(batch.results),
                'successful_items': len(batch.get_successful_results()),
                'throughput': len(batch.results) / (end_time - start_time),
                'mpi_size': self.coordinator.size,
                'distribution': type(self.distribution_strategy).__name__
            }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()

    def gather_worker_statistics(self) -> Dict[str, Any]:
        """Gather statistics from all MPI ranks."""
        local_stats = self._stats
        all_stats = self.coordinator.comm.gather(local_stats, root=0)

        if self.coordinator.rank == 0:
            return {
                'per_rank_stats': all_stats,
                'aggregate_stats': self._compute_aggregate_stats(all_stats)
            }
        return {}

    def _compute_aggregate_stats(self, rank_stats: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics across ranks."""
        total_time = max(stats.get('total_time', 0) for stats in rank_stats)
        total_items = sum(stats.get('items_processed', 0) for stats in rank_stats)
        total_successful = sum(stats.get('successful_items', 0) for stats in rank_stats)

        return {
            'total_time': total_time,
            'total_items_processed': total_items,
            'total_successful_items': total_successful,
            'overall_success_rate': total_successful / total_items if total_items > 0 else 0,
            'overall_throughput': total_items / total_time if total_time > 0 else 0
        }


class ThreadPoolStrategy(ProcessingStrategy):
    """Thread pool processing strategy."""

    def __init__(self, max_workers: Optional[int] = None,
                 distribution_strategy: Optional[DistributionStrategy] = None,
                 logger: Optional[Logger] = None):
        self.max_workers = max_workers
        self.distribution_strategy = distribution_strategy or ChunkDistribution()
        self.logger = logger
        self.coordinator = ThreadPoolCoordinator(max_workers, logger)
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute thread pool processing."""
        start_time = time.time()

        batch = self.coordinator.coordinate_processing(work_items, processor_func)

        end_time = time.time()
        self._stats = {
            'strategy': 'thread_pool',
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(batch.get_successful_results()),
            'throughput': len(batch.results) / (end_time - start_time),
            'max_workers': self.max_workers,
            'distribution': type(self.distribution_strategy).__name__
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()


class HybridStrategy(ProcessingStrategy):
    """Hybrid strategy combining multiple approaches."""

    def __init__(self, primary_strategy: ProcessingStrategy,
                 fallback_strategy: ProcessingStrategy,
                 failure_threshold: float = 0.5,
                 logger: Optional[Logger] = None):
        self.primary_strategy = primary_strategy
        self.fallback_strategy = fallback_strategy
        self.failure_threshold = failure_threshold
        self.logger = logger
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute hybrid processing with fallback."""
        start_time = time.time()

        try:
            # Try primary strategy first
            batch = self.primary_strategy.execute(work_items, processor_func, **kwargs)

            # Check success rate
            success_rate = (len(batch.get_successful_results()) /
                            len(batch.results) if batch.results else 0)

            if success_rate < self.failure_threshold:
                if self.logger:
                    self.logger.log_warning(
                        f"Primary strategy success rate {success_rate:.2%} below threshold {self.failure_threshold:.2%}, "
                        "falling back to secondary strategy"
                    )

                # Get failed items
                failed_items = [
                    result.dataset_info.path
                    for result in batch.results
                    if not result.is_successful()
                ]

                # Retry with fallback strategy
                fallback_batch = self.fallback_strategy.execute(failed_items, processor_func, **kwargs)

                # Merge successful results
                final_batch = ProcessingBatch()
                for result in batch.results:
                    if result.is_successful():
                        final_batch.add_result(result)

                for result in fallback_batch.results:
                    final_batch.add_result(result)

                batch = final_batch
                strategy_used = 'hybrid'
            else:
                strategy_used = 'primary'

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Primary strategy failed: {e}, using fallback")
            batch = self.fallback_strategy.execute(work_items, processor_func, **kwargs)
            strategy_used = 'fallback'

        end_time = time.time()
        self._stats = {
            'strategy': 'hybrid',
            'strategy_used': strategy_used,
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(batch.get_successful_results()),
            'throughput': len(batch.results) / (end_time - start_time),
            'primary_stats': self.primary_strategy.get_statistics(),
            'fallback_stats': self.fallback_strategy.get_statistics()
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()


class AdaptiveStrategy(ProcessingStrategy):
    """Adaptive strategy that chooses best approach based on workload."""

    def __init__(self, available_strategies: Dict[str, ProcessingStrategy],
                 workload_analyzer: Optional[Callable] = None,
                 logger: Optional[Logger] = None):
        self.available_strategies = available_strategies
        self.workload_analyzer = workload_analyzer or self._default_workload_analyzer
        self.logger = logger
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute adaptive processing."""
        start_time = time.time()

        # Analyze workload to choose strategy
        workload_analysis = self.workload_analyzer(work_items)
        chosen_strategy = self._choose_strategy(workload_analysis)

        if self.logger:
            self.logger.log_info(f"Adaptive strategy chose: {chosen_strategy}")

        # Execute chosen strategy
        strategy = self.available_strategies[chosen_strategy]
        batch = strategy.execute(work_items, processor_func, **kwargs)

        end_time = time.time()
        self._stats = {
            'strategy': 'adaptive',
            'chosen_strategy': chosen_strategy,
            'workload_analysis': workload_analysis,
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(batch.get_successful_results()),
            'throughput': len(batch.results) / (end_time - start_time),
            'strategy_stats': strategy.get_statistics()
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()

    def _default_workload_analyzer(self, work_items: List[str]) -> Dict[str, Any]:
        """Default workload analysis."""
        return {
            'item_count': len(work_items),
            'estimated_total_size': len(work_items),  # Could analyze file sizes
            'complexity': 'medium'  # Could analyze dataset complexity
        }

    def _choose_strategy(self, workload_analysis: Dict[str, Any]) -> str:
        """Choose strategy based on workload analysis."""
        item_count = workload_analysis['item_count']

        if item_count < 5:
            return 'sequential'
        elif item_count < 20:
            return 'thread_pool'
        else:
            return 'mpi' if 'mpi' in self.available_strategies else 'thread_pool'


def create_processing_strategy(strategy_type: str, **kwargs) -> ProcessingStrategy:
    """Factory function to create processing strategy."""
    strategies = {
        'sequential': SequentialStrategy,
        'mpi': MPIStrategy,
        'thread_pool': ThreadPoolStrategy,
        'hybrid': HybridStrategy,
        'adaptive': AdaptiveStrategy
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategies[strategy_type](**kwargs)


def create_default_adaptive_strategy(logger: Optional[Logger] = None) -> AdaptiveStrategy:
    """Create adaptive strategy with default options."""
    strategies = {
        'sequential': SequentialStrategy(logger),
        'thread_pool': ThreadPoolStrategy(logger=logger)
    }

    # Add MPI if available
    try:
        strategies['mpi'] = MPIStrategy(logger=logger)
    except:
        pass  # MPI not available

    return AdaptiveStrategy(strategies, logger=logger)