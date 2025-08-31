"""
Worker implementations for parallel processing.
"""
from typing import Any, Dict, List, Optional, Callable
import time
import traceback
import threading
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
from pathlib import Path

from ..core.domain import ProcessingResult, DatasetInfo, BoundingBox
from ..core.exceptions import MPIError, ParallelProcessingError, DataLoadError
from ..core.interfaces import Logger

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI = None
    MPI_AVAILABLE = False


class WorkerBase(ABC):
    """Base class for worker implementations."""

    def __init__(self, worker_id: int, logger: Optional[Logger] = None):
        self.worker_id = worker_id
        self.logger = logger
        self._stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'datasets_processed': 0,
            'flame_analyses_completed': 0,
            'shock_analyses_completed': 0
        }
        self._lock = threading.Lock()

    @abstractmethod
    def process_work_item(self, work_item: Any, processor_func: Callable) -> Any:
        """Process a single work item."""
        pass

    def get_statistics(self) -> Dict[str, float]:
        """Get worker statistics."""
        with self._lock:
            return self._stats.copy()

    def _record_success(self, processing_time: float, result: ProcessingResult = None) -> None:
        """Record successful task completion."""
        with self._lock:
            self._stats['tasks_completed'] += 1
            self._stats['total_processing_time'] += processing_time
            self._stats['datasets_processed'] += 1

            # Track analysis types
            if result:
                if result.flame_data:
                    self._stats['flame_analyses_completed'] += 1
                if result.shock_data:
                    self._stats['shock_analyses_completed'] += 1

    def _record_failure(self) -> None:
        """Record task failure."""
        with self._lock:
            self._stats['tasks_failed'] += 1

    def _create_dataset_info(self, dataset_path: str) -> DatasetInfo:
        """Create DatasetInfo for Pele plotfile."""
        try:
            path_obj = Path(dataset_path)

            # Extract plt number for timestamp estimation
            match = re.search(r'plt(\d+)', path_obj.name)
            plt_number = int(match.group(1)) if match else 0
            estimated_time = plt_number * 1e-6  # Rough estimate

            # Basic bounds (will be updated by actual data loader)
            bounds = BoundingBox(0.0, 0.1, 0.0, 0.05)  # Typical Pele domain

            return DatasetInfo(
                path=path_obj,
                basename=path_obj.name,
                timestamp=estimated_time,
                domain_bounds=bounds,
                max_refinement_level=2,  # Default
                grid_spacing=1e-5  # Default
            )
        except Exception:
            # Fallback for any parsing issues
            return DatasetInfo.from_path(dataset_path)

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.log_info(f"Worker {self.worker_id}: {message}")

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.log_error(f"Worker {self.worker_id}: {message}")


class MPIWorker(WorkerBase):
    """MPI-based worker implementation."""

    def __init__(self, logger: Optional[Logger] = None):
        if not MPI_AVAILABLE:
            raise MPIError("initialization", 0, "mpi4py not available")

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        super().__init__(worker_id=self.rank, logger=logger)
        self._log_info(f"Initialized on rank {self.rank} of {self.size}")

    def process_work_item(self, work_item: str, processor_func: Callable) -> Optional[ProcessingResult]:
        """Process dataset path using provided function."""
        start_time = time.time()

        try:
            # Validate dataset path exists
            path_obj = Path(work_item)
            if not path_obj.exists():
                raise DataLoadError(work_item, "Dataset directory does not exist")

            if not path_obj.is_dir():
                raise DataLoadError(work_item, "Dataset path is not a directory")

            self._log_info(f"Processing: {path_obj.name}")
            result = processor_func(work_item)

            processing_time = time.time() - start_time

            if result and result.is_successful():
                self._record_success(processing_time, result)

                # Log analysis results
                analysis_info = []
                if result.flame_data and result.flame_data.position:
                    analysis_info.append(f"flame@{result.flame_data.position:.3e}m")
                if result.shock_data and result.shock_data.position:
                    analysis_info.append(f"shock@{result.shock_data.position:.3e}m")

                analysis_str = ", ".join(analysis_info) if analysis_info else "basic"
                self._log_info(f"Successfully processed {path_obj.name} ({analysis_str}) in {processing_time:.2f}s")
            else:
                self._record_failure()
                error_msg = result.error_message if result else "Unknown error"
                self._log_error(f"Processing failed for {path_obj.name}: {error_msg}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_failure()

            error_msg = f"Exception processing {work_item}: {e}"
            self._log_error(error_msg)
            self._log_error(f"Traceback: {traceback.format_exc()}")

            return ProcessingResult(
                dataset_info=self._create_dataset_info(work_item),
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    def run_worker_loop(self, processor_func: Callable, work_items: List[str]) -> List[ProcessingResult]:
        """Run main worker processing loop."""
        results = []

        for work_item in work_items:
            result = self.process_work_item(work_item, processor_func)
            if result:
                results.append(result)

        return results


class ThreadWorker(WorkerBase):
    """Thread-based worker implementation."""

    def __init__(self, worker_id: int, logger: Optional[Logger] = None):
        super().__init__(worker_id, logger)
        self._thread_id = threading.get_ident()
        self._log_info(f"Initialized on thread {self._thread_id}")

    def process_work_item(self, work_item: str, processor_func: Callable) -> Optional[ProcessingResult]:
        """Process dataset path using provided function."""
        start_time = time.time()

        try:
            path_obj = Path(work_item)
            self._log_info(f"Processing: {path_obj.name}")

            result = processor_func(work_item)

            processing_time = time.time() - start_time

            if result and result.is_successful():
                self._record_success(processing_time, result)
                self._log_info(f"Successfully processed {path_obj.name} in {processing_time:.2f}s")
            else:
                self._record_failure()
                self._log_error(f"Processing failed for {path_obj.name}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_failure()

            error_msg = f"Exception processing {work_item}: {e}"
            self._log_error(error_msg)

            return ProcessingResult(
                dataset_info=self._create_dataset_info(work_item),
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )


class ProcessWorker(WorkerBase):
    """Process-based worker implementation."""

    def __init__(self, worker_id: int, logger: Optional[Logger] = None):
        super().__init__(worker_id, logger)
        import os
        self._process_id = os.getpid()
        self._log_info(f"Initialized in process {self._process_id}")

    def process_work_item(self, work_item: str, processor_func: Callable) -> Optional[ProcessingResult]:
        """Process dataset path using provided function."""
        start_time = time.time()

        try:
            path_obj = Path(work_item)
            self._log_info(f"Processing: {path_obj.name}")

            result = processor_func(work_item)

            processing_time = time.time() - start_time

            if result and result.is_successful():
                self._record_success(processing_time, result)
                self._log_info(f"Successfully processed {path_obj.name} in {processing_time:.2f}s")
            else:
                self._record_failure()
                self._log_error(f"Processing failed for {path_obj.name}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self._record_failure()

            error_msg = f"Exception processing {work_item}: {e}"
            self._log_error(error_msg)

            return ProcessingResult(
                dataset_info=self._create_dataset_info(work_item),
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )


class WorkerManager:
    """Manages a pool of workers."""

    def __init__(self, worker_type: str = "thread", num_workers: Optional[int] = None,
                 logger: Optional[Logger] = None):
        self.worker_type = worker_type
        self.num_workers = num_workers or self._get_default_worker_count()
        self.logger = logger
        self.workers = []

        self._create_workers()

    def _get_default_worker_count(self) -> int:
        """Get default number of workers based on system."""
        if self.worker_type == "mpi":
            if MPI_AVAILABLE:
                return MPI.COMM_WORLD.Get_size()
            return 1
        else:
            import os
            return os.cpu_count() or 1

    def _create_workers(self) -> None:
        """Create worker instances."""
        if self.worker_type == "mpi":
            self.workers = [MPIWorker(self.logger)]
        elif self.worker_type == "thread":
            self.workers = [
                ThreadWorker(i, self.logger)
                for i in range(self.num_workers)
            ]
        elif self.worker_type == "process":
            self.workers = [
                ProcessWorker(i, self.logger)
                for i in range(self.num_workers)
            ]
        else:
            raise ValueError(f"Unknown worker type: {self.worker_type}")

    def get_worker_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics from all workers."""
        stats = {}
        for worker in self.workers:
            stats[f"worker_{worker.worker_id}"] = worker.get_statistics()
        return stats

    def get_aggregate_statistics(self) -> Dict[str, float]:
        """Get aggregate statistics across all workers."""
        total_completed = sum(w.get_statistics()['tasks_completed'] for w in self.workers)
        total_failed = sum(w.get_statistics()['tasks_failed'] for w in self.workers)
        total_time = sum(w.get_statistics()['total_processing_time'] for w in self.workers)
        total_flame_analyses = sum(w.get_statistics()['flame_analyses_completed'] for w in self.workers)
        total_shock_analyses = sum(w.get_statistics()['shock_analyses_completed'] for w in self.workers)

        return {
            'total_tasks_completed': total_completed,
            'total_tasks_failed': total_failed,
            'total_processing_time': total_time,
            'success_rate': total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0,
            'average_task_time': total_time / total_completed if total_completed > 0 else 0,
            'flame_analyses_completed': total_flame_analyses,
            'shock_analyses_completed': total_shock_analyses,
            'datasets_per_second': total_completed / total_time if total_time > 0 else 0
        }


class TaskQueue:
    """Thread-safe task queue for worker coordination."""

    def __init__(self):
        self._queue = queue.Queue()
        self._results = queue.Queue()

    def add_task(self, task: Any) -> None:
        """Add task to queue."""
        self._queue.put(task)

    def get_task(self, timeout: Optional[float] = None) -> Any:
        """Get next task from queue."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def add_result(self, result: Any) -> None:
        """Add result to results queue."""
        self._results.put(result)

    def get_result(self, timeout: Optional[float] = None) -> Any:
        """Get result from results queue."""
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Mark task as completed."""
        self._queue.task_done()

    def wait_completion(self) -> None:
        """Wait for all tasks to complete."""
        self._queue.join()

    def empty(self) -> bool:
        """Check if task queue is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Get approximate queue size."""
        return self._queue.qsize()

    def get_all_tasks(self) -> List[Any]:
        """Get all remaining tasks from queue."""
        tasks = []
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                tasks.append(task)
            except queue.Empty:
                break
        return tasks


class WorkerPool:
    """High-level worker pool interface."""

    def __init__(self, worker_type: str = "thread", max_workers: Optional[int] = None,
                 logger: Optional[Logger] = None):
        self.worker_manager = WorkerManager(worker_type, max_workers, logger)
        self.task_queue = TaskQueue()
        self.logger = logger
        self.worker_type = worker_type
        self.max_workers = max_workers
        self._running = False

    def submit_tasks(self, tasks: List[Any]) -> None:
        """Submit multiple tasks for processing."""
        for task in tasks:
            self.task_queue.add_task(task)

    def process_all_tasks(self, processor_func: Callable) -> List[Any]:
        """Process all submitted tasks with proper parallel distribution."""
        # Get all tasks from queue
        tasks = self.task_queue.get_all_tasks()

        if not tasks:
            return []

        results = []

        if self.worker_type == "mpi":
            # MPI processing is handled by MPICoordinator
            for task in tasks:
                worker = self.worker_manager.workers[0]  # Single MPI worker per rank
                result = worker.process_work_item(task, processor_func)
                if result:
                    results.append(result)

        elif self.worker_type == "thread":
            # Use ThreadPoolExecutor for true parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._process_task_wrapper, task, processor_func): task
                    for task in tasks
                }

                # Collect results as they complete
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        if self.logger:
                            self.logger.log_error(f"Task failed: {task}, Error: {e}")

                        # Create failed result
                        failed_result = ProcessingResult(
                            dataset_info=DatasetInfo.from_path(str(task)) if isinstance(task, str) else DatasetInfo(name=str(task)),
                            success=False,
                            error_message=str(e),
                            processing_time=0.0
                        )
                        results.append(failed_result)

        elif self.worker_type == "process":
            # Use ProcessPoolExecutor for true multiprocess processing
            try:
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all tasks
                    future_to_task = {
                        executor.submit(self._process_task_static, task, processor_func): task
                        for task in tasks
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_task):
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                        except Exception as e:
                            task = future_to_task[future]
                            if self.logger:
                                self.logger.log_error(f"Process task failed: {task}, Error: {e}")

                            # Create failed result
                            failed_result = ProcessingResult(
                                dataset_info=DatasetInfo.from_path(str(task)) if isinstance(task, str) else DatasetInfo(name=str(task)),
                                success=False,
                                error_message=str(e),
                                processing_time=0.0
                            )
                            results.append(failed_result)

            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"ProcessPoolExecutor failed: {e}")
                # Fall back to sequential processing
                results = self._process_tasks_sequential(tasks, processor_func)
        else:
            # Sequential fallback
            results = self._process_tasks_sequential(tasks, processor_func)

        return results

    def _process_task_wrapper(self, task: Any, processor_func: Callable) -> Any:
        """Wrapper for processing individual tasks in threads."""
        # Create a temporary worker for this task
        worker = ThreadWorker(threading.get_ident(), self.logger)
        return worker.process_work_item(task, processor_func)

    @staticmethod
    def _process_task_static(task: Any, processor_func: Callable) -> Any:
        """Static method for processing tasks in separate processes."""
        import os
        start_time = time.time()

        try:
            result = processor_func(task)
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                dataset_info=DatasetInfo.from_path(str(task)) if isinstance(task, str) else DatasetInfo(name=str(task)),
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    def _process_tasks_sequential(self, tasks: List[Any], processor_func: Callable) -> List[Any]:
        """Sequential processing fallback."""
        results = []
        worker = self.worker_manager.workers[0] if self.worker_manager.workers else None

        if not worker:
            # Create a basic worker if none exist
            worker = ThreadWorker(0, self.logger)

        for task in tasks:
            result = worker.process_work_item(task, processor_func)
            if result:
                results.append(result)

        return results

    def process_tasks_with_callback(self, tasks: List[Any], processor_func: Callable,
                                   callback: Optional[Callable[[Any], None]] = None) -> List[Any]:
        """Process tasks with optional progress callback."""
        results = []
        completed = 0
        total = len(tasks)

        if self.worker_type == "thread" and self.max_workers and self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._process_task_wrapper, task, processor_func): task
                    for task in tasks
                }

                for future in as_completed(future_to_task):
                    completed += 1
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        failed_result = ProcessingResult(
                            dataset_info=DatasetInfo.from_path(str(task)) if isinstance(task, str) else DatasetInfo(name=str(task)),
                            success=False,
                            error_message=str(e),
                            processing_time=0.0
                        )
                        results.append(failed_result)

                    if callback:
                        callback({'completed': completed, 'total': total, 'progress': completed / total})
        else:
            # Sequential processing with progress
            worker = self.worker_manager.workers[0] if self.worker_manager.workers else ThreadWorker(0, self.logger)

            for i, task in enumerate(tasks):
                result = worker.process_work_item(task, processor_func)
                if result:
                    results.append(result)

                completed = i + 1
                if callback:
                    callback({'completed': completed, 'total': total, 'progress': completed / total})

        return results

    def shutdown(self) -> None:
        """Shutdown the worker pool."""
        self._running = False
        # Clean up any remaining tasks
        while not self.task_queue.empty():
            try:
                self.task_queue.get_task(timeout=0.1)
                self.task_queue.task_done()
            except:
                break

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'worker_type': self.worker_type,
            'max_workers': self.max_workers,
            'worker_stats': self.worker_manager.get_worker_statistics(),
            'aggregate_stats': self.worker_manager.get_aggregate_statistics(),
            'queue_size': self.task_queue.qsize(),
            'is_running': self._running
        }


# Utility functions for Windows compatibility
def setup_windows_multiprocessing():
    """Setup multiprocessing for Windows compatibility."""
    import multiprocessing
    import sys

    if sys.platform.startswith('win'):
        # Required for Windows multiprocessing
        multiprocessing.set_start_method('spawn', force=True)


def create_worker_pool(worker_type: str = "thread", max_workers: Optional[int] = None,
                      logger: Optional[Logger] = None) -> WorkerPool:
    """Factory function to create worker pool with Windows compatibility."""
    if worker_type == "process":
        setup_windows_multiprocessing()

    return WorkerPool(worker_type, max_workers, logger)