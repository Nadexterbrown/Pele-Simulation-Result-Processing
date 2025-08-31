"""
MPI coordination for parallel processing.
"""
from typing import List, Dict, Any, Callable, Optional, Union
import time
from pathlib import Path

from ..core.interfaces import ParallelCoordinator, Logger
from ..core.domain import ProcessingResult, ProcessingBatch, DatasetInfo
from ..core.exceptions import MPIError, ProcessSynchronizationError, WorkDistributionError

try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI = None
    MPI_AVAILABLE = False


class MPICoordinator(ParallelCoordinator):
    """MPI-based parallel coordinator."""

    def __init__(self, logger: Optional[Logger] = None, timeout: float = 300.0):
        if not MPI_AVAILABLE:
            raise MPIError("initialization", 0, "mpi4py not available")

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.timeout = timeout
        self.logger = logger

        self._log_info(f"MPI initialized: rank {self.rank} of {self.size}")

    def coordinate_processing(self, dataset_paths: List[str],
                              processor_function: Callable[[str], ProcessingResult]) -> ProcessingBatch:
        """Coordinate parallel processing across MPI ranks."""
        self._log_info(f"Starting parallel processing of {len(dataset_paths)} datasets")

        # Distribute work across ranks
        local_paths = self._distribute_work(dataset_paths)
        self._log_info(f"Rank {self.rank} assigned {len(local_paths)} datasets")

        # Process local datasets
        local_results = []
        for i, path in enumerate(local_paths):
            try:
                self._log_info(f"Processing dataset {i + 1}/{len(local_paths)}: {Path(path).name}")
                result = processor_function(path)
                if result:
                    local_results.append(result)
            except Exception as e:
                self._log_error(f"Failed to process {path}: {e}")
                # Create failed result
                failed_result = ProcessingResult(
                    dataset_info=DatasetInfo.from_path(path),
                    success=False,
                    error_message=str(e)
                )
                local_results.append(failed_result)

        # Synchronize before gathering
        self.synchronize_processes()

        # Gather all results to root
        all_results = self.gather_results(local_results)

        # Root creates and returns batch
        if self.rank == 0:
            batch = ProcessingBatch()
            for result in all_results:
                batch.add_result(result)
            self._log_info(f"Processing complete: {len(batch.get_successful_results())}/{len(all_results)} successful")
            return batch
        else:
            return ProcessingBatch()  # Non-root returns empty batch

    def synchronize_processes(self) -> None:
        """Synchronize all processes at barrier."""
        try:
            self._log_debug("Synchronizing processes...")
            self.comm.Barrier()
            self._log_debug("Synchronization complete")
        except Exception as e:
            raise ProcessSynchronizationError("barrier", self.timeout) from e

    def gather_results(self, local_results: List[ProcessingResult]) -> List[ProcessingResult]:
        """Gather results from all processes to root."""
        try:
            self._log_debug(f"Gathering {len(local_results)} local results")
            all_results = self.comm.gather(local_results, root=0)

            if self.rank == 0:
                # Flatten list of lists
                flattened = []
                for rank_results in all_results:
                    flattened.extend(rank_results)
                self._log_debug(f"Gathered {len(flattened)} total results")
                return flattened
            else:
                return []

        except Exception as e:
            raise MPIError("gather", self.rank, str(e)) from e

    def get_process_info(self) -> Dict[str, int]:
        """Get MPI process information."""
        return {
            'rank': self.rank,
            'size': self.size,
            'is_root': self.rank == 0
        }

    def broadcast_config(self, config: Any) -> Any:
        """Broadcast configuration from root to all processes."""
        try:
            return self.comm.bcast(config, root=0)
        except Exception as e:
            raise MPIError("broadcast", self.rank, str(e)) from e

    def reduce_statistics(self, local_stats: Dict[str, float]) -> Dict[str, float]:
        """Reduce statistics across all processes."""
        try:
            # Sum statistics
            global_stats = {}
            for key, value in local_stats.items():
                total = self.comm.reduce(value, op=MPI.SUM, root=0)
                if self.rank == 0:
                    global_stats[key] = total

            return global_stats
        except Exception as e:
            raise MPIError("reduce", self.rank, str(e)) from e

    def _distribute_work(self, work_items: List[str]) -> List[str]:
        """Distribute work items across MPI ranks."""
        # Simple round-robin distribution
        local_items = []
        for i, item in enumerate(work_items):
            if i % self.size == self.rank:
                local_items.append(item)

        return local_items

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.log_info(message, rank=self.rank)

    def _log_debug(self, message: str) -> None:
        if self.logger:
            self.logger.log_debug(message, rank=self.rank)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.log_error(message, rank=self.rank)


class SequentialCoordinator(ParallelCoordinator):
    """Sequential processing coordinator (no parallelization)."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def coordinate_processing(self, dataset_paths: List[str],
                              processor_function: Callable[[str], ProcessingResult]) -> ProcessingBatch:
        """Process datasets sequentially."""
        self._log_info(f"Starting sequential processing of {len(dataset_paths)} datasets")

        batch = ProcessingBatch()
        for i, path in enumerate(dataset_paths):
            try:
                self._log_info(f"Processing dataset {i + 1}/{len(dataset_paths)}: {Path(path).name}")
                result = processor_function(path)
                if result:
                    batch.add_result(result)
            except Exception as e:
                self._log_error(f"Failed to process {path}: {e}")
                failed_result = ProcessingResult(
                    dataset_info=DatasetInfo.from_path(path),
                    success=False,
                    error_message=str(e)
                )
                batch.add_result(failed_result)

        successful = len(batch.get_successful_results())
        self._log_info(f"Processing complete: {successful}/{len(batch.results)} successful")
        return batch

    def synchronize_processes(self) -> None:
        """No-op for sequential processing."""
        pass

    def gather_results(self, local_results: List[Any]) -> List[Any]:
        """Return results as-is for sequential processing."""
        return local_results

    def get_process_info(self) -> Dict[str, int]:
        """Get process information for sequential execution."""
        return {
            'rank': 0,
            'size': 1,
            'is_root': True
        }

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.log_info(message)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.log_error(message)


class ThreadPoolCoordinator(ParallelCoordinator):
    """Thread-based parallel coordinator."""

    def __init__(self, max_workers: Optional[int] = None, logger: Optional[Logger] = None):
        self.max_workers = max_workers
        self.logger = logger

        try:
            from concurrent.futures import ThreadPoolExecutor
            self._executor_class = ThreadPoolExecutor
        except ImportError:
            raise WorkDistributionError("ThreadPoolExecutor not available", 0, 0)

    def coordinate_processing(self, dataset_paths: List[str],
                              processor_function: Callable[[str], ProcessingResult]) -> ProcessingBatch:
        """Process datasets using thread pool."""
        self._log_info(f"Starting threaded processing of {len(dataset_paths)} datasets with {self.max_workers} workers")

        batch = ProcessingBatch()

        with self._executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(processor_function, path): path
                for path in dataset_paths
            }

            # Collect results
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        batch.add_result(result)
                except Exception as e:
                    self._log_error(f"Failed to process {path}: {e}")
                    failed_result = ProcessingResult(
                        dataset_info=DatasetInfo.from_path(path),
                        success=False,
                        error_message=str(e)
                    )
                    batch.add_result(failed_result)

        successful = len(batch.get_successful_results())
        self._log_info(f"Processing complete: {successful}/{len(batch.results)} successful")
        return batch

    def synchronize_processes(self) -> None:
        """No-op for thread-based processing."""
        pass

    def gather_results(self, local_results: List[Any]) -> List[Any]:
        """Return results as-is for thread-based processing."""
        return local_results

    def get_process_info(self) -> Dict[str, int]:
        """Get process information for threaded execution."""
        return {
            'rank': 0,
            'size': self.max_workers or 1,
            'is_root': True
        }

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.log_info(message)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.log_error(message)


def create_coordinator(mode: str, **kwargs) -> ParallelCoordinator:
    """Factory function to create appropriate coordinator."""
    if mode == "sequential":
        return SequentialCoordinator(**kwargs)
    elif mode == "parallel_mpi":
        return MPICoordinator(**kwargs)
    elif mode == "parallel_threads":
        return ThreadPoolCoordinator(**kwargs)
    else:
        raise ValueError(f"Unknown coordination mode: {mode}")