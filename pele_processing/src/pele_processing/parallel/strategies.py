"""
High-level parallel processing strategies for Pele simulation result processing.
"""
from typing import List, Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import time
import re
from pathlib import Path
import numpy as np

from ..core.domain import ProcessingResult, ProcessingBatch, DatasetInfo, BoundingBox
from ..core.interfaces import Logger
from ..core.exceptions import ParallelProcessingError, DataLoadError
from .coordinator import MPICoordinator, SequentialCoordinator, ThreadPoolCoordinator
from .distribution import DistributionStrategy, RoundRobinDistribution, ChunkDistribution
from .workers import WorkerManager

# Import Pele-specific components
try:
    from .workers import PeleWorkerPool, create_pele_worker_pool
    PELE_WORKERS_AVAILABLE = True
except ImportError:
    PELE_WORKERS_AVAILABLE = False
    # Fallback to basic worker pool
    PeleWorkerPool = None
    create_pele_worker_pool = None


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
    """Sequential processing strategy for Pele datasets."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.coordinator = SequentialCoordinator(logger)
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute sequential processing of Pele datasets."""
        start_time = time.time()

        # Validate Pele dataset paths
        validated_items = self._validate_pele_datasets(work_items)
        if len(validated_items) != len(work_items):
            if self.logger:
                self.logger.log_warning(f"Filtered {len(work_items) - len(validated_items)} invalid datasets")

        batch = self.coordinator.coordinate_processing(validated_items, processor_func)

        end_time = time.time()

        # Calculate Pele-specific statistics
        successful_results = batch.get_successful_results()
        flame_analyses = sum(1 for r in successful_results if r.flame_data)
        shock_analyses = sum(1 for r in successful_results if r.shock_data)

        self._stats = {
            'strategy': 'sequential',
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(successful_results),
            'flame_analyses': flame_analyses,
            'shock_analyses': shock_analyses,
            'throughput': len(batch.results) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'success_rate': len(successful_results) / len(batch.results) if batch.results else 0
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()

    def _validate_pele_datasets(self, dataset_paths: List[str]) -> List[str]:
        """Validate Pele dataset paths."""
        valid_paths = []
        for path in dataset_paths:
            path_obj = Path(path)
            if (path_obj.exists() and
                path_obj.is_dir() and
                (path_obj / "Header").exists()):
                valid_paths.append(path)
            elif self.logger:
                self.logger.log_warning(f"Invalid Pele dataset: {path}")
        return valid_paths


class MPIStrategy(ProcessingStrategy):
    """MPI-based parallel processing strategy for Pele datasets."""

    def __init__(self, distribution_strategy: Optional[DistributionStrategy] = None,
                 timeout: float = 300.0, logger: Optional[Logger] = None):
        self.distribution_strategy = distribution_strategy or RoundRobinDistribution()
        self.logger = logger
        self.coordinator = MPICoordinator(logger, timeout)
        self.worker_pool = create_pele_worker_pool("mpi", logger=logger)
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute MPI parallel processing of Pele datasets."""
        start_time = time.time()

        # Validate and sort Pele datasets
        validated_items = self._validate_and_sort_datasets(work_items)

        # Use coordinator for MPI processing
        batch = self.coordinator.coordinate_processing(validated_items, processor_func)

        end_time = time.time()

        # Only root process computes final statistics
        if self.coordinator.rank == 0:
            successful_results = batch.get_successful_results()
            flame_analyses = sum(1 for r in successful_results if r.flame_data and r.flame_data.position)
            shock_analyses = sum(1 for r in successful_results if r.shock_data and r.shock_data.position)

            # Calculate time series analysis if possible
            time_series_stats = self._calculate_time_series_stats(batch)

            self._stats = {
                'strategy': 'mpi',
                'total_time': end_time - start_time,
                'items_processed': len(batch.results),
                'successful_items': len(successful_results),
                'flame_analyses': flame_analyses,
                'shock_analyses': shock_analyses,
                'throughput': len(batch.results) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                'mpi_size': self.coordinator.size,
                'distribution': type(self.distribution_strategy).__name__,
                'success_rate': len(successful_results) / len(batch.results) if batch.results else 0,
                **time_series_stats
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

    def _validate_and_sort_datasets(self, dataset_paths: List[str]) -> List[str]:
        """Validate and sort Pele datasets by plt number."""
        valid_paths = []
        for path in dataset_paths:
            path_obj = Path(path)
            if (path_obj.exists() and
                path_obj.is_dir() and
                (path_obj / "Header").exists()):
                valid_paths.append(path)

        # Sort by plt number for consistent processing
        return self._sort_by_plt_number(valid_paths)

    def _sort_by_plt_number(self, dataset_paths: List[str]) -> List[str]:
        """Sort datasets by plt number."""
        def extract_plt_number(path: str) -> int:
            match = re.search(r'plt(\d+)', Path(path).name)
            return int(match.group(1)) if match else 0

        return sorted(dataset_paths, key=extract_plt_number)

    def _calculate_time_series_stats(self, batch: ProcessingBatch) -> Dict[str, Any]:
        """Calculate time series analysis statistics."""
        successful = batch.get_successful_results()
        if len(successful) < 2:
            return {}

        try:
            # Extract time series
            times = batch.get_timestamps()
            flame_positions = batch.get_flame_positions()
            shock_positions = batch.get_shock_positions()

            stats = {}

            # Flame velocity analysis
            valid_flame = ~np.isnan(flame_positions)
            if np.sum(valid_flame) > 1:
                import numpy as np
                flame_velocities = np.gradient(flame_positions[valid_flame], times[valid_flame])
                stats.update({
                    'mean_flame_velocity': float(np.mean(flame_velocities)),
                    'flame_velocity_std': float(np.std(flame_velocities)),
                    'flame_position_range': float(np.max(flame_positions[valid_flame]) - np.min(flame_positions[valid_flame]))
                })

            # Shock velocity analysis
            valid_shock = ~np.isnan(shock_positions)
            if np.sum(valid_shock) > 1:
                shock_velocities = np.gradient(shock_positions[valid_shock], times[valid_shock])
                stats.update({
                    'mean_shock_velocity': float(np.mean(shock_velocities)),
                    'shock_velocity_std': float(np.std(shock_velocities)),
                    'shock_position_range': float(np.max(shock_positions[valid_shock]) - np.min(shock_positions[valid_shock]))
                })

            return stats

        except Exception:
            return {}

    def _compute_aggregate_stats(self, rank_stats: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics across ranks."""
        total_time = max(stats.get('total_time', 0) for stats in rank_stats if stats)
        total_items = sum(stats.get('items_processed', 0) for stats in rank_stats if stats)
        total_successful = sum(stats.get('successful_items', 0) for stats in rank_stats if stats)
        total_flame = sum(stats.get('flame_analyses', 0) for stats in rank_stats if stats)
        total_shock = sum(stats.get('shock_analyses', 0) for stats in rank_stats if stats)

        return {
            'total_time': total_time,
            'total_items_processed': total_items,
            'total_successful_items': total_successful,
            'total_flame_analyses': total_flame,
            'total_shock_analyses': total_shock,
            'overall_success_rate': total_successful / total_items if total_items > 0 else 0,
            'overall_throughput': total_items / total_time if total_time > 0 else 0,
            'flame_analysis_rate': total_flame / total_successful if total_successful > 0 else 0,
            'shock_analysis_rate': total_shock / total_successful if total_successful > 0 else 0
        }


class ThreadPoolStrategy(ProcessingStrategy):
    """Thread pool processing strategy for Pele datasets."""

    def __init__(self, max_workers: Optional[int] = None,
                 distribution_strategy: Optional[DistributionStrategy] = None,
                 logger: Optional[Logger] = None):
        self.max_workers = max_workers
        self.distribution_strategy = distribution_strategy or ChunkDistribution()
        self.logger = logger
        self.coordinator = ThreadPoolCoordinator(max_workers, logger)
        self.worker_pool = create_pele_worker_pool("thread", max_workers, logger)
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute thread pool processing of Pele datasets."""
        start_time = time.time()

        # Validate Pele datasets
        validated_items = self._validate_pele_datasets(work_items)

        # Process using worker pool for better task distribution
        results = self.worker_pool.process_datasets(validated_items, processor_func)

        # Create batch from results
        batch = ProcessingBatch()
        for result in results:
            batch.add_result(result)

        end_time = time.time()

        # Calculate comprehensive statistics
        successful_results = batch.get_successful_results()
        flame_analyses = sum(1 for r in successful_results if r.flame_data and r.flame_data.position)
        shock_analyses = sum(1 for r in successful_results if r.shock_data and r.shock_data.position)

        self._stats = {
            'strategy': 'thread_pool',
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(successful_results),
            'flame_analyses': flame_analyses,
            'shock_analyses': shock_analyses,
            'throughput': len(batch.results) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'max_workers': self.max_workers,
            'distribution': type(self.distribution_strategy).__name__,
            'success_rate': len(successful_results) / len(batch.results) if batch.results else 0,
            'worker_pool_stats': self.worker_pool.get_comprehensive_statistics()
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()

    def _validate_pele_datasets(self, dataset_paths: List[str]) -> List[str]:
        """Validate Pele dataset paths."""
        valid_paths = []
        for path in dataset_paths:
            path_obj = Path(path)
            if (path_obj.exists() and
                path_obj.is_dir() and
                (path_obj / "Header").exists()):
                valid_paths.append(path)
            elif self.logger:
                self.logger.log_warning(f"Invalid Pele dataset: {path}")
        return valid_paths


class HybridStrategy(ProcessingStrategy):
    """Hybrid strategy with fallback for failed Pele datasets."""

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
        """Execute hybrid processing with intelligent fallback."""
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
                        "falling back to secondary strategy for failed datasets"
                    )

                # Get failed dataset paths
                failed_paths = [
                    str(result.dataset_info.path)
                    for result in batch.results
                    if not result.is_successful()
                ]

                if failed_paths:
                    # Retry with fallback strategy
                    fallback_batch = self.fallback_strategy.execute(failed_paths, processor_func, **kwargs)

                    # Merge successful results from both strategies
                    final_batch = ProcessingBatch()

                    # Add successful results from primary
                    for result in batch.results:
                        if result.is_successful():
                            final_batch.add_result(result)

                    # Add all results from fallback
                    for result in fallback_batch.results:
                        final_batch.add_result(result)

                    batch = final_batch
                    strategy_used = 'hybrid'
                else:
                    strategy_used = 'primary'
            else:
                strategy_used = 'primary'

        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Primary strategy failed: {e}, using fallback")
            batch = self.fallback_strategy.execute(work_items, processor_func, **kwargs)
            strategy_used = 'fallback'

        end_time = time.time()

        # Calculate final statistics
        successful_results = batch.get_successful_results()
        flame_analyses = sum(1 for r in successful_results if r.flame_data and r.flame_data.position)
        shock_analyses = sum(1 for r in successful_results if r.shock_data and r.shock_data.position)

        self._stats = {
            'strategy': 'hybrid',
            'strategy_used': strategy_used,
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(successful_results),
            'flame_analyses': flame_analyses,
            'shock_analyses': shock_analyses,
            'throughput': len(batch.results) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'success_rate': len(successful_results) / len(batch.results) if batch.results else 0,
            'primary_stats': self.primary_strategy.get_statistics(),
            'fallback_stats': self.fallback_strategy.get_statistics()
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()


class AdaptiveStrategy(ProcessingStrategy):
    """Adaptive strategy that chooses best approach based on Pele workload characteristics."""

    def __init__(self, available_strategies: Dict[str, ProcessingStrategy],
                 workload_analyzer: Optional[Callable] = None,
                 logger: Optional[Logger] = None):
        self.available_strategies = available_strategies
        self.workload_analyzer = workload_analyzer or self._analyze_pele_workload
        self.logger = logger
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute adaptive processing based on Pele dataset characteristics."""
        start_time = time.time()

        # Analyze Pele workload to choose optimal strategy
        workload_analysis = self.workload_analyzer(work_items)
        chosen_strategy = self._choose_strategy(workload_analysis)

        if self.logger:
            self.logger.log_info(f"Adaptive strategy analysis: {workload_analysis}")
            self.logger.log_info(f"Chosen strategy: {chosen_strategy}")

        # Execute chosen strategy
        strategy = self.available_strategies[chosen_strategy]
        batch = strategy.execute(work_items, processor_func, **kwargs)

        end_time = time.time()

        # Calculate comprehensive statistics
        successful_results = batch.get_successful_results()
        flame_analyses = sum(1 for r in successful_results if r.flame_data and r.flame_data.position)
        shock_analyses = sum(1 for r in successful_results if r.shock_data and r.shock_data.position)

        self._stats = {
            'strategy': 'adaptive',
            'chosen_strategy': chosen_strategy,
            'workload_analysis': workload_analysis,
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(successful_results),
            'flame_analyses': flame_analyses,
            'shock_analyses': shock_analyses,
            'throughput': len(batch.results) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'success_rate': len(successful_results) / len(batch.results) if batch.results else 0,
            'strategy_stats': strategy.get_statistics()
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()

    def _analyze_pele_workload(self, dataset_paths: List[str]) -> Dict[str, Any]:
        """Analyze Pele dataset workload characteristics."""
        analysis = {
            'dataset_count': len(dataset_paths),
            'estimated_total_size': 0,
            'complexity': 'medium',
            'has_time_series': False,
            'plt_number_range': None,
            'estimated_memory_usage': 0
        }

        if not dataset_paths:
            return analysis

        # Analyze dataset characteristics
        plt_numbers = []
        total_estimated_size = 0

        for path in dataset_paths:
            path_obj = Path(path)

            # Extract plt number
            match = re.search(r'plt(\d+)', path_obj.name)
            if match:
                plt_numbers.append(int(match.group(1)))

            # Estimate dataset size
            if path_obj.exists() and path_obj.is_dir():
                try:
                    # Quick size estimation
                    level_dirs = list(path_obj.glob('Level_*'))
                    estimated_size = len(level_dirs) * 50 * 1024 * 1024  # 50MB per level
                    total_estimated_size += estimated_size
                except:
                    total_estimated_size += 100 * 1024 * 1024  # Default 100MB

        # Update analysis
        if plt_numbers:
            analysis['plt_number_range'] = (min(plt_numbers), max(plt_numbers))
            analysis['has_time_series'] = len(set(plt_numbers)) == len(plt_numbers)  # Consecutive

        analysis['estimated_total_size'] = total_estimated_size
        analysis['estimated_memory_usage'] = total_estimated_size * 0.5  # Rough estimate

        # Determine complexity
        if len(dataset_paths) > 100 or total_estimated_size > 10 * 1024**3:  # >10GB
            analysis['complexity'] = 'high'
        elif len(dataset_paths) > 20 or total_estimated_size > 1 * 1024**3:  # >1GB
            analysis['complexity'] = 'medium'
        else:
            analysis['complexity'] = 'low'

        return analysis

    def _choose_strategy(self, workload_analysis: Dict[str, Any]) -> str:
        """Choose optimal strategy based on Pele workload analysis."""
        dataset_count = workload_analysis['dataset_count']
        complexity = workload_analysis['complexity']
        estimated_size = workload_analysis['estimated_total_size']

        # Decision logic for Pele datasets
        if dataset_count < 3:
            return 'sequential'
        elif complexity == 'low' and dataset_count < 10:
            return 'sequential'
        elif complexity == 'high' or dataset_count > 50:
            # Use MPI for large workloads if available
            return 'mpi' if 'mpi' in self.available_strategies else 'thread_pool'
        else:
            return 'thread_pool'


class PeleOptimizedStrategy(ProcessingStrategy):
    """Specialized strategy optimized for Pele simulation processing."""

    def __init__(self, max_workers: Optional[int] = None,
                 enable_caching: bool = True,
                 logger: Optional[Logger] = None):
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.logger = logger
        self.worker_pool = create_pele_worker_pool("thread", max_workers, logger)
        self._stats = {}
        self._processed_datasets = set()  # Track processed datasets

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute Pele-optimized processing."""
        start_time = time.time()

        # Pre-process dataset list
        validated_items = self._preprocess_dataset_list(work_items)

        # Apply caching if enabled
        if self.enable_caching:
            validated_items = self._filter_cached_datasets(validated_items)

        # Process with optimization
        results = self._process_with_optimization(validated_items, processor_func)

        # Create batch
        batch = ProcessingBatch()
        for result in results:
            batch.add_result(result)
            if result.is_successful():
                self._processed_datasets.add(str(result.dataset_info.path))

        end_time = time.time()

        # Calculate optimization statistics
        successful_results = batch.get_successful_results()
        self._stats = {
            'strategy': 'pele_optimized',
            'total_time': end_time - start_time,
            'items_processed': len(batch.results),
            'successful_items': len(successful_results),
            'cache_hits': len(work_items) - len(validated_items) if self.enable_caching else 0,
            'throughput': len(batch.results) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'success_rate': len(successful_results) / len(batch.results) if batch.results else 0,
            'worker_pool_stats': self.worker_pool.get_comprehensive_statistics()
        }

        return batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()

    def _preprocess_dataset_list(self, dataset_paths: List[str]) -> List[str]:
        """Preprocess and validate Pele dataset list."""
        valid_paths = []

        for path in dataset_paths:
            path_obj = Path(path)

            # Validate Pele dataset structure
            if not path_obj.exists():
                if self.logger:
                    self.logger.log_warning(f"Dataset does not exist: {path}")
                continue

            if not path_obj.is_dir():
                if self.logger:
                    self.logger.log_warning(f"Dataset is not a directory: {path}")
                continue

            # Check for required AMReX files
            header_file = path_obj / "Header"
            if not header_file.exists():
                if self.logger:
                    self.logger.log_warning(f"Dataset missing Header file: {path}")
                continue

            valid_paths.append(path)

        # Sort by plt number for optimal processing order
        return self._sort_by_plt_number(valid_paths)

    def _filter_cached_datasets(self, dataset_paths: List[str]) -> List[str]:
        """Filter out already processed datasets if caching enabled."""
        if not self.enable_caching:
            return dataset_paths

        return [path for path in dataset_paths if path not in self._processed_datasets]

    def _process_with_optimization(self, dataset_paths: List[str],
                                  processor_func: Callable) -> List[ProcessingResult]:
        """Process with Pele-specific optimizations."""
        # Group datasets by time proximity for better cache locality
        grouped_datasets = self._group_datasets_by_time(dataset_paths)

        all_results = []
        for group in grouped_datasets:
            group_results = self.worker_pool.process_datasets(group, processor_func)
            all_results.extend(group_results)

        return all_results

    def _group_datasets_by_time(self, dataset_paths: List[str]) -> List[List[str]]:
        """Group datasets by time proximity for cache optimization."""
        # Simple grouping by consecutive plt numbers
        sorted_paths = self._sort_by_plt_number(dataset_paths)

        groups = []
        current_group = []
        last_plt_num = None

        for path in sorted_paths:
            plt_num = self._extract_plt_number(path)

            if last_plt_num is None or plt_num == last_plt_num + 1:
                current_group.append(path)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [path]

            last_plt_num = plt_num

        if current_group:
            groups.append(current_group)

        return groups

    def _sort_by_plt_number(self, dataset_paths: List[str]) -> List[str]:
        """Sort datasets by plt number."""
        def extract_plt_number(path: str) -> int:
            match = re.search(r'plt(\d+)', Path(path).name)
            return int(match.group(1)) if match else 0

        return sorted(dataset_paths, key=extract_plt_number)

    def _extract_plt_number(self, path: str) -> int:
        """Extract plt number from dataset path."""
        match = re.search(r'plt(\d+)', Path(path).name)
        return int(match.group(1)) if match else 0


class BatchStrategy(ProcessingStrategy):
    """Strategy for processing large batches of Pele datasets efficiently."""

    def __init__(self, batch_size: int = 50, max_workers: Optional[int] = None,
                 checkpoint_interval: Optional[int] = None,
                 logger: Optional[Logger] = None):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.checkpoint_interval = checkpoint_interval
        self.logger = logger
        self.worker_pool = create_pele_worker_pool("thread", max_workers, logger)
        self._stats = {}

    def execute(self, work_items: List[str], processor_func: Callable,
                **kwargs) -> ProcessingBatch:
        """Execute batch processing of Pele datasets."""
        start_time = time.time()

        # Validate datasets
        validated_items = self._validate_datasets(work_items)

        # Process in batches
        final_batch = ProcessingBatch()
        total_batches = (len(validated_items) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(validated_items))
            batch_items = validated_items[batch_start:batch_end]

            if self.logger:
                self.logger.log_info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_items)} datasets)")

            # Process batch
            batch_results = self.worker_pool.process_datasets(batch_items, processor_func)

            for result in batch_results:
                final_batch.add_result(result)

            # Checkpoint if configured
            if (self.checkpoint_interval and
                (batch_idx + 1) % self.checkpoint_interval == 0):
                self._save_checkpoint(final_batch, batch_idx + 1)

        end_time = time.time()

        # Calculate batch processing statistics
        successful_results = final_batch.get_successful_results()

        self._stats = {
            'strategy': 'batch',
            'total_time': end_time - start_time,
            'total_batches': total_batches,
            'batch_size': self.batch_size,
            'items_processed': len(final_batch.results),
            'successful_items': len(successful_results),
            'throughput': len(final_batch.results) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'success_rate': len(successful_results) / len(final_batch.results) if final_batch.results else 0,
            'average_batch_time': (end_time - start_time) / total_batches if total_batches > 0 else 0
        }

        return final_batch

    def get_statistics(self) -> Dict[str, Any]:
        return self._stats.copy()

    def _validate_datasets(self, dataset_paths: List[str]) -> List[str]:
        """Validate Pele datasets."""
        valid_paths = []
        for path in dataset_paths:
            path_obj = Path(path)
            if (path_obj.exists() and
                path_obj.is_dir() and
                (path_obj / "Header").exists()):
                valid_paths.append(path)
        return valid_paths

    def _save_checkpoint(self, batch: ProcessingBatch, batch_number: int) -> None:
        """Save processing checkpoint."""
        try:
            checkpoint_file = Path(f"checkpoint_batch_{batch_number}.json")
            batch.save_to_file(checkpoint_file)
            if self.logger:
                self.logger.log_info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to save checkpoint: {e}")


def create_processing_strategy(strategy_type: str, **kwargs) -> ProcessingStrategy:
    """Factory function to create Pele processing strategy."""
    strategies = {
        'sequential': SequentialStrategy,
        'mpi': MPIStrategy,
        'thread_pool': ThreadPoolStrategy,
        'hybrid': HybridStrategy,
        'adaptive': AdaptiveStrategy,
        'pele_optimized': PeleOptimizedStrategy,
        'batch': BatchStrategy
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategies[strategy_type](**kwargs)


def create_default_adaptive_strategy(logger: Optional[Logger] = None) -> AdaptiveStrategy:
    """Create adaptive strategy optimized for Pele processing."""
    strategies = {
        'sequential': SequentialStrategy(logger),
        'thread_pool': ThreadPoolStrategy(logger=logger),
        'pele_optimized': PeleOptimizedStrategy(logger=logger)
    }

    # Add MPI if available
    try:
        strategies['mpi'] = MPIStrategy(logger=logger)
    except:
        pass  # MPI not available

    return AdaptiveStrategy(strategies, logger=logger)


def create_pele_strategy_suite(logger: Optional[Logger] = None) -> Dict[str, ProcessingStrategy]:
    """Create complete suite of Pele processing strategies."""
    return {
        'sequential': SequentialStrategy(logger),
        'thread_pool': ThreadPoolStrategy(logger=logger),
        'mpi': MPIStrategy(logger=logger) if MPI_AVAILABLE else None,
        'pele_optimized': PeleOptimizedStrategy(logger=logger),
        'batch': BatchStrategy(logger=logger),
        'adaptive': create_default_adaptive_strategy(logger)
    }