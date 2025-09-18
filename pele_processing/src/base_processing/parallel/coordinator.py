"""
MPI coordination for Pele dataset parallel processing.
"""
from typing import List, Dict, Any, Callable, Optional, Union
import time
import re
from pathlib import Path

from ..core.interfaces import ParallelCoordinator, Logger
from ..core.domain import ProcessingResult, ProcessingBatch, DatasetInfo, BoundingBox
from ..core.exceptions import MPIError, ProcessSynchronizationError, WorkDistributionError

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI = None
    MPI_AVAILABLE = False


class PeleMPICoordinator(ParallelCoordinator):
    """MPI-based parallel coordinator specifically for Pele plotfile processing."""

    def __init__(self, logger: Optional[Logger] = None, timeout: float = 300.0):
        if not MPI_AVAILABLE:
            raise MPIError("initialization", 0, "mpi4py not available")

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.timeout = timeout
        self.logger = logger

        self._log_info(f"Pele MPI coordinator initialized: rank {self.rank} of {self.size}")

    def coordinate_processing(self, dataset_paths: List[str],
                              processor_function: Callable[[str], ProcessingResult]) -> ProcessingBatch:
        """Coordinate parallel processing of Pele datasets."""
        self._log_info(f"Starting parallel processing of {len(dataset_paths)} Pele datasets")

        # Validate and filter dataset paths
        valid_paths = self._validate_pele_datasets(dataset_paths)
        if len(valid_paths) != len(dataset_paths):
            skipped = len(dataset_paths) - len(valid_paths)
            self._log_warning(f"Skipped {skipped} invalid Pele datasets")

        # Distribute work across ranks with load balancing
        local_paths = self._distribute_pele_datasets(valid_paths)
        self._log_info(f"Rank {self.rank} assigned {len(local_paths)} datasets")

        # Process local datasets
        local_results = []
        for i, dataset_path in enumerate(local_paths):
            try:
                plt_name = Path(dataset_path).name
                self._log_info(f"Processing dataset {i + 1}/{len(local_paths)}: {plt_name}")

                result = processor_function(dataset_path)
                if result:
                    local_results.append(result)
                    self._log_analysis_summary(result)

            except Exception as e:
                self._log_error(f"Failed to process {dataset_path}: {e}")
                failed_result = ProcessingResult(
                    dataset_info=self._create_pele_dataset_info(dataset_path),
                    success=False,
                    error_message=str(e)
                )
                local_results.append(failed_result)

        # Synchronize processes and gather results
        self.synchronize_processes()
        all_results = self.gather_results(local_results)

        # Root process creates final batch
        if self.rank == 0:
            batch = ProcessingBatch()
            for result in all_results:
                batch.add_result(result)

            self._log_final_summary(batch)
            return batch
        else:
            return ProcessingBatch()

    def synchronize_processes(self) -> None:
        """Synchronize all MPI processes at barrier."""
        try:
            self._log_debug("Synchronizing MPI processes...")
            self.comm.Barrier()
            self._log_debug("MPI synchronization complete")
        except Exception as e:
            raise ProcessSynchronizationError("barrier", self.timeout) from e

    def gather_results(self, local_results: List[ProcessingResult]) -> List[ProcessingResult]:
        """Gather results from all MPI processes to root."""
        try:
            self._log_debug(f"Gathering {len(local_results)} local results")
            all_results = self.comm.gather(local_results, root=0)

            if self.rank == 0:
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
        """Reduce statistics across all MPI processes."""
        try:
            global_stats = {}
            for key, value in local_stats.items():
                total = self.comm.reduce(value, op=MPI.SUM, root=0)
                if self.rank == 0:
                    global_stats[key] = total
            return global_stats
        except Exception as e:
            raise MPIError("reduce", self.rank, str(e)) from e

    def _validate_pele_datasets(self, dataset_paths: List[str]) -> List[str]:
        """Validate Pele dataset paths and filter valid ones."""
        valid_paths = []

        for path_str in dataset_paths:
            path_obj = Path(path_str)

            # Check if it's a directory
            if not path_obj.exists():
                self._log_warning(f"Dataset path does not exist: {path_str}")
                continue

            if not path_obj.is_dir():
                self._log_warning(f"Dataset path is not a directory: {path_str}")
                continue

            # Check for AMReX Header file
            header_file = path_obj / "Header"
            if not header_file.exists():
                self._log_warning(f"Dataset missing Header file: {path_str}")
                continue

            # Check for Level_ directories
            level_dirs = list(path_obj.glob('Level_*'))
            if not level_dirs:
                self._log_warning(f"Dataset missing Level_ directories: {path_str}")
                continue

            valid_paths.append(path_str)

        return valid_paths

    def _distribute_pele_datasets(self, dataset_paths: List[str]) -> List[str]:
        """Distribute Pele datasets across MPI ranks with load balancing."""
        # Sort by plt number for consistent ordering
        sorted_paths = self._sort_datasets_by_plt_number(dataset_paths)

        # Round-robin distribution
        local_paths = []
        for i, path in enumerate(sorted_paths):
            if i % self.size == self.rank:
                local_paths.append(path)

        return local_paths

    def _sort_datasets_by_plt_number(self, dataset_paths: List[str]) -> List[str]:
        """Sort dataset paths by plt number."""
        def extract_plt_number(path: str) -> int:
            match = re.search(r'plt(\d+)', Path(path).name)
            return int(match.group(1)) if match else 0

        return sorted(dataset_paths, key=extract_plt_number)

    def _create_pele_dataset_info(self, dataset_path: str) -> DatasetInfo:
        """Create DatasetInfo for Pele plotfile with proper metadata."""
        path_obj = Path(dataset_path)

        # Extract plt number and estimate physical time
        match = re.search(r'plt(\d+)', path_obj.name)
        plt_number = int(match.group(1)) if match else 0
        estimated_time = plt_number * 1e-6  # Rough time estimate (1μs per step)

        # Read domain bounds from Header file if possible
        bounds = self._read_domain_bounds(path_obj)
        refinement_level, grid_spacing = self._read_grid_info(path_obj)

        return DatasetInfo(
            path=path_obj,
            basename=path_obj.name,
            timestamp=estimated_time,
            domain_bounds=bounds,
            max_refinement_level=refinement_level,
            grid_spacing=grid_spacing
        )

    def _read_domain_bounds(self, dataset_path: Path) -> BoundingBox:
        """Read domain bounds from AMReX Header file."""
        try:
            header_file = dataset_path / "Header"
            if not header_file.exists():
                return BoundingBox(0.0, 0.1, 0.0, 0.05)  # Default bounds

            with open(header_file, 'r') as f:
                lines = f.readlines()

            # Parse AMReX header format
            for line in lines:
                if 'prob_lo' in line.lower():
                    # Parse prob_lo coordinates
                    coords = [float(x) for x in line.split()[1:3]]  # x, y coordinates
                    prob_lo_x, prob_lo_y = coords[0] / 100, coords[1] / 100  # Convert cm to m
                elif 'prob_hi' in line.lower():
                    # Parse prob_hi coordinates
                    coords = [float(x) for x in line.split()[1:3]]
                    prob_hi_x, prob_hi_y = coords[0] / 100, coords[1] / 100  # Convert cm to m

                    return BoundingBox(prob_lo_x, prob_hi_x, prob_lo_y, prob_hi_y)

            return BoundingBox(0.0, 0.1, 0.0, 0.05)  # Default if parsing fails

        except Exception:
            return BoundingBox(0.0, 0.1, 0.0, 0.05)  # Default bounds

    def _read_grid_info(self, dataset_path: Path) -> tuple[int, float]:
        """Read grid refinement and spacing info from Header."""
        try:
            header_file = dataset_path / "Header"
            if not header_file.exists():
                return 2, 1e-5  # Defaults

            with open(header_file, 'r') as f:
                content = f.read()

            # Parse refinement levels
            max_level = 0
            if 'max_level' in content:
                match = re.search(r'max_level\s*=\s*(\d+)', content)
                if match:
                    max_level = int(match.group(1))

            # Parse grid spacing (simplified)
            grid_spacing = 1e-5  # Default
            if 'dx' in content:
                match = re.search(r'dx\[0\]\s*=\s*([\d.e-]+)', content)
                if match:
                    grid_spacing = float(match.group(1)) / 100  # Convert cm to m

            return max_level, grid_spacing

        except Exception:
            return 2, 1e-5  # Defaults

    def _log_analysis_summary(self, result: ProcessingResult) -> None:
        """Log analysis results summary."""
        if not result.is_successful():
            return

        analysis_info = []
        if result.flame_data and result.flame_data.position is not None:
            analysis_info.append(f"flame@{result.flame_data.position:.3e}m")
            if result.flame_data.velocity is not None:
                analysis_info.append(f"vel={result.flame_data.velocity:.3e}m/s")

        if result.shock_data and result.shock_data.position is not None:
            analysis_info.append(f"shock@{result.shock_data.position:.3e}m")

        if analysis_info:
            self._log_info(f"Analysis: {', '.join(analysis_info)}")

    def _log_final_summary(self, batch: ProcessingBatch) -> None:
        """Log final processing summary with Pele-specific metrics."""
        successful = batch.get_successful_results()
        failed = [r for r in batch.results if not r.is_successful()]

        # Count analysis types
        flame_analyses = sum(1 for r in successful if r.flame_data)
        shock_analyses = sum(1 for r in successful if r.shock_data)

        self._log_info(
            f"Pele processing complete: {len(successful)}/{len(batch.results)} datasets successful "
            f"({flame_analyses} flame, {shock_analyses} shock analyses)"
        )

        if failed:
            self._log_warning(f"Failed datasets: {[r.dataset_info.basename for r in failed[:5]]}")

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.log_info(message, rank=self.rank)

    def _log_debug(self, message: str) -> None:
        if self.logger:
            self.logger.log_debug(message, rank=self.rank)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.log_error(message, rank=self.rank)

    def _log_warning(self, message: str) -> None:
        if self.logger:
            self.logger.log_warning(message, rank=self.rank)


class SequentialCoordinator(ParallelCoordinator):
    """Sequential processing coordinator for Pele datasets."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def coordinate_processing(self, dataset_paths: List[str],
                              processor_function: Callable[[str], ProcessingResult]) -> ProcessingBatch:
        """Process Pele datasets sequentially."""
        self._log_info(f"Starting sequential processing of {len(dataset_paths)} Pele datasets")

        batch = ProcessingBatch()
        for i, dataset_path in enumerate(dataset_paths):
            try:
                plt_name = Path(dataset_path).name
                self._log_info(f"Processing dataset {i + 1}/{len(dataset_paths)}: {plt_name}")

                result = processor_function(dataset_path)
                if result:
                    batch.add_result(result)

            except Exception as e:
                self._log_error(f"Failed to process {dataset_path}: {e}")
                failed_result = ProcessingResult(
                    dataset_info=self._create_pele_dataset_info(dataset_path),
                    success=False,
                    error_message=str(e)
                )
                batch.add_result(failed_result)

        successful = len(batch.get_successful_results())
        self._log_info(f"Sequential processing complete: {successful}/{len(batch.results)} successful")
        return batch

    def synchronize_processes(self) -> None:
        """No-op for sequential processing."""
        pass

    def gather_results(self, local_results: List[Any]) -> List[Any]:
        """Return results as-is for sequential processing."""
        return local_results

    def get_process_info(self) -> Dict[str, int]:
        """Get process information for sequential execution."""
        return {'rank': 0, 'size': 1, 'is_root': True}

    def _create_pele_dataset_info(self, dataset_path: str) -> DatasetInfo:
        """Create DatasetInfo for failed Pele dataset."""
        path_obj = Path(dataset_path)
        match = re.search(r'plt(\d+)', path_obj.name)
        plt_number = int(match.group(1)) if match else 0

        return DatasetInfo(
            path=path_obj,
            basename=path_obj.name,
            timestamp=plt_number * 1e-6,
            domain_bounds=BoundingBox(0.0, 0.1, 0.0, 0.05),
            max_refinement_level=2,
            grid_spacing=1e-5
        )

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.log_info(message)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.log_error(message)


class ThreadPoolCoordinator(ParallelCoordinator):
    """Thread-based parallel coordinator for Pele datasets."""

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
        """Process Pele datasets using thread pool."""
        self._log_info(f"Starting threaded processing of {len(dataset_paths)} Pele datasets with {self.max_workers} workers")

        batch = ProcessingBatch()

        with self._executor_class(max_workers=self.max_workers) as executor:
            # Submit all dataset processing tasks
            future_to_path = {
                executor.submit(processor_function, path): path
                for path in dataset_paths
            }

            # Collect results as they complete
            from concurrent.futures import as_completed
            for future in as_completed(future_to_path):
                dataset_path = future_to_path[future]
                try:
                    result = future.result()
                    if result:
                        batch.add_result(result)
                except Exception as e:
                    self._log_error(f"Failed to process {dataset_path}: {e}")
                    failed_result = ProcessingResult(
                        dataset_info=self._create_pele_dataset_info(dataset_path),
                        success=False,
                        error_message=str(e)
                    )
                    batch.add_result(failed_result)

        successful = len(batch.get_successful_results())
        self._log_info(f"Threaded processing complete: {successful}/{len(batch.results)} successful")
        return batch

    def synchronize_processes(self) -> None:
        """No-op for thread-based processing."""
        pass

    def gather_results(self, local_results: List[Any]) -> List[Any]:
        """Return results as-is for thread-based processing."""
        return local_results

    def get_process_info(self) -> Dict[str, int]:
        """Get process information for threaded execution."""
        return {'rank': 0, 'size': self.max_workers or 1, 'is_root': True}

    def _create_pele_dataset_info(self, dataset_path: str) -> DatasetInfo:
        """Create DatasetInfo for failed dataset."""
        path_obj = Path(dataset_path)
        match = re.search(r'plt(\d+)', path_obj.name)
        plt_number = int(match.group(1)) if match else 0

        return DatasetInfo(
            path=path_obj,
            basename=path_obj.name,
            timestamp=plt_number * 1e-6,
            domain_bounds=BoundingBox(0.0, 0.1, 0.0, 0.05),
            max_refinement_level=2,
            grid_spacing=1e-5
        )

    def _log_info(self, message: str) -> None:
        if self.logger:
            self.logger.log_info(message)

    def _log_error(self, message: str) -> None:
        if self.logger:
            self.logger.log_error(message)


def create_coordinator(mode: str, **kwargs) -> ParallelCoordinator:
    """Factory function to create appropriate coordinator for Pele processing."""
    if mode == "sequential":
        return SequentialCoordinator(**kwargs)
    elif mode == "parallel_mpi":
        return PeleMPICoordinator(**kwargs)
    elif mode == "parallel_threads":
        return ThreadPoolCoordinator(**kwargs)
    else:
        raise ValueError(f"Unknown coordination mode: {mode}")


# Backward compatibility aliases
MPICoordinator = PeleMPICoordinator