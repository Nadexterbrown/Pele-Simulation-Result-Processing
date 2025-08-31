"""
Work distribution strategies for Pele parallel processing.
"""
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import random
import re
from pathlib import Path

from ..core.interfaces import WorkDistributor
from ..core.exceptions import WorkDistributionError


class DistributionStrategy(ABC):
    """Base class for work distribution strategies."""

    @abstractmethod
    def distribute(self, work_items: List[Any], num_workers: int) -> List[List[Any]]:
        """Distribute work items among workers."""
        pass


class RoundRobinDistribution(DistributionStrategy):
    """Round-robin distribution strategy."""

    def distribute(self, work_items: List[Any], num_workers: int) -> List[List[Any]]:
        """Distribute items in round-robin fashion."""
        if num_workers <= 0:
            raise WorkDistributionError("Invalid worker count", len(work_items), num_workers)

        worker_lists = [[] for _ in range(num_workers)]

        for i, item in enumerate(work_items):
            worker_idx = i % num_workers
            worker_lists[worker_idx].append(item)

        return worker_lists


class ChunkDistribution(DistributionStrategy):
    """Chunk-based distribution strategy."""

    def distribute(self, work_items: List[Any], num_workers: int) -> List[List[Any]]:
        """Distribute items in contiguous chunks."""
        if num_workers <= 0:
            raise WorkDistributionError("Invalid worker count", len(work_items), num_workers)

        chunk_size = len(work_items) // num_workers
        remainder = len(work_items) % num_workers

        worker_lists = []
        start_idx = 0

        for i in range(num_workers):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            worker_lists.append(work_items[start_idx:end_idx])
            start_idx = end_idx

        return worker_lists


class PeleDatasetDistribution(DistributionStrategy):
    """Distribution strategy optimized for Pele plotfile processing."""

    def distribute(self, work_items: List[str], num_workers: int) -> List[List[str]]:
        """Distribute Pele datasets with load balancing by plt number."""
        if num_workers <= 0:
            raise WorkDistributionError("Invalid worker count", len(work_items), num_workers)

        # Sort datasets by plt number for consistent distribution
        sorted_items = self._sort_by_plt_number(work_items)

        # Round-robin distribution to balance load across time steps
        worker_lists = [[] for _ in range(num_workers)]

        for i, item in enumerate(sorted_items):
            worker_idx = i % num_workers
            worker_lists[worker_idx].append(item)

        return worker_lists

    def _sort_by_plt_number(self, dataset_paths: List[str]) -> List[str]:
        """Sort dataset paths by plt number."""
        def extract_plt_number(path: str) -> int:
            match = re.search(r'plt(\d+)', Path(path).name)
            return int(match.group(1)) if match else 0

        return sorted(dataset_paths, key=extract_plt_number)


class PeleLoadBalancedDistribution(DistributionStrategy):
    """Load-balanced distribution for Pele datasets based on dataset characteristics."""

    def __init__(self):
        self.cost_estimator = self._estimate_pele_dataset_cost

    def distribute(self, work_items: List[str], num_workers: int) -> List[List[str]]:
        """Distribute Pele datasets based on estimated processing costs."""
        if num_workers <= 0:
            raise WorkDistributionError("Invalid worker count", len(work_items), num_workers)

        # Estimate costs for all datasets
        items_with_costs = [(item, self.cost_estimator(item)) for item in work_items]

        # Sort by cost (largest first) for better load balancing
        items_with_costs.sort(key=lambda x: x[1], reverse=True)

        # Initialize worker loads
        worker_lists = [[] for _ in range(num_workers)]
        worker_loads = [0.0] * num_workers

        # Assign each dataset to least loaded worker
        for item, cost in items_with_costs:
            min_load_idx = worker_loads.index(min(worker_loads))
            worker_lists[min_load_idx].append(item)
            worker_loads[min_load_idx] += cost

        return worker_lists

    def _estimate_pele_dataset_cost(self, dataset_path: str) -> float:
        """Estimate processing cost based on Pele dataset characteristics."""
        try:
            path_obj = Path(dataset_path)
            base_cost = 1.0

            if not path_obj.exists() or not path_obj.is_dir():
                return base_cost

            # Factor 1: Number of AMR levels (more levels = more data)
            level_dirs = list(path_obj.glob('Level_*'))
            amr_factor = len(level_dirs) * 0.3

            # Factor 2: Total dataset size
            try:
                total_size = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
                size_factor = total_size / (100 * 1024 * 1024)  # Normalize by 100MB
            except:
                size_factor = 1.0

            # Factor 3: Grid complexity from Header
            try:
                header_file = path_obj / "Header"
                if header_file.exists():
                    with open(header_file, 'r') as f:
                        content = f.read()

                    # Higher max_level = more processing
                    level_match = re.search(r'max_level\s*=\s*(\d+)', content)
                    if level_match:
                        max_level = int(level_match.group(1))
                        complexity_factor = max_level * 0.2
                    else:
                        complexity_factor = 0.5
                else:
                    complexity_factor = 0.5
            except:
                complexity_factor = 0.5

            total_cost = base_cost + amr_factor + size_factor + complexity_factor
            return max(total_cost, 0.1)  # Minimum cost

        except Exception:
            return 1.0  # Default cost


class LoadBalancedDistribution(DistributionStrategy):
    """General load-balanced distribution using estimated work costs."""

    def __init__(self, cost_estimator: Optional[Callable[[Any], float]] = None):
        self.cost_estimator = cost_estimator or self._default_cost_estimator

    def distribute(self, work_items: List[Any], num_workers: int) -> List[List[Any]]:
        """Distribute items based on estimated processing costs."""
        if num_workers <= 0:
            raise WorkDistributionError("Invalid worker count", len(work_items), num_workers)

        items_with_costs = [(item, self.cost_estimator(item)) for item in work_items]
        items_with_costs.sort(key=lambda x: x[1], reverse=True)

        worker_lists = [[] for _ in range(num_workers)]
        worker_loads = [0.0] * num_workers

        for item, cost in items_with_costs:
            min_load_idx = worker_loads.index(min(worker_loads))
            worker_lists[min_load_idx].append(item)
            worker_loads[min_load_idx] += cost

        return worker_lists

    def _default_cost_estimator(self, item: Any) -> float:
        """Default cost estimation (all items equal)."""
        return 1.0


class RandomDistribution(DistributionStrategy):
    """Random distribution strategy."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def distribute(self, work_items: List[Any], num_workers: int) -> List[List[Any]]:
        """Randomly distribute items among workers."""
        if num_workers <= 0:
            raise WorkDistributionError("Invalid worker count", len(work_items), num_workers)

        worker_lists = [[] for _ in range(num_workers)]

        for item in work_items:
            worker_idx = random.randint(0, num_workers - 1)
            worker_lists[worker_idx].append(item)

        return worker_lists


class AdaptiveDistribution(DistributionStrategy):
    """Adaptive distribution that adjusts based on worker performance."""

    def __init__(self):
        self.worker_performance_history = {}  # worker_id -> list of processing times

    def distribute(self, work_items: List[Any], num_workers: int) -> List[List[Any]]:
        """Distribute based on historical performance."""
        if num_workers <= 0:
            raise WorkDistributionError("Invalid worker count", len(work_items), num_workers)

        if not self.worker_performance_history:
            # No history, use round-robin
            return RoundRobinDistribution().distribute(work_items, num_workers)

        # Calculate worker speeds (inverse of average processing time)
        worker_speeds = {}
        for worker_id in range(num_workers):
            if worker_id in self.worker_performance_history:
                avg_time = sum(self.worker_performance_history[worker_id]) / len(
                    self.worker_performance_history[worker_id])
                worker_speeds[worker_id] = 1.0 / max(avg_time, 0.001)
            else:
                worker_speeds[worker_id] = 1.0

        # Normalize speeds to get distribution ratios
        total_speed = sum(worker_speeds.values())
        worker_ratios = {k: v / total_speed for k, v in worker_speeds.items()}

        # Distribute items based on speed ratios
        worker_lists = [[] for _ in range(num_workers)]

        for item in work_items:
            cumulative = 0.0
            rand_val = random.random()

            for worker_id in range(num_workers):
                cumulative += worker_ratios[worker_id]
                if rand_val <= cumulative:
                    worker_lists[worker_id].append(item)
                    break

        return worker_lists

    def update_performance(self, worker_id: int, processing_time: float):
        """Update performance history for adaptive distribution."""
        if worker_id not in self.worker_performance_history:
            self.worker_performance_history[worker_id] = []

        self.worker_performance_history[worker_id].append(processing_time)

        # Keep only recent history
        max_history = 10
        if len(self.worker_performance_history[worker_id]) > max_history:
            self.worker_performance_history[worker_id] = self.worker_performance_history[worker_id][-max_history:]


class MPIDistributor(WorkDistributor):
    """MPI-based work distributor for Pele datasets."""

    def __init__(self, strategy: DistributionStrategy = None):
        self.strategy = strategy or PeleDatasetDistribution()

        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        except ImportError:
            raise WorkDistributionError("MPI not available", 0, 0)

    def distribute_work(self, work_items: List[Any], worker_function: Callable) -> List[Any]:
        """Distribute work across MPI processes."""
        # Root distributes work
        if self.rank == 0:
            distributed_work = self.strategy.distribute(work_items, self.size)
        else:
            distributed_work = None

        # Scatter work to all processes
        local_work = self.comm.scatter(distributed_work, root=0)

        # Process local work
        local_results = []
        for item in local_work:
            try:
                result = worker_function(item)
                if result is not None:
                    local_results.append(result)
            except Exception:
                pass  # Failed items are handled by worker

        # Gather results
        all_results = self.comm.gather(local_results, root=0)

        if self.rank == 0:
            flattened_results = []
            for rank_results in all_results:
                flattened_results.extend([r for r in rank_results if r is not None])
            return flattened_results
        else:
            return []

    def get_process_info(self) -> Dict[str, int]:
        """Get MPI process information."""
        return {'rank': self.rank, 'size': self.size}


class ThreadDistributor(WorkDistributor):
    """Thread-based work distributor for Pele datasets."""

    def __init__(self, max_workers: Optional[int] = None, strategy: DistributionStrategy = None):
        self.max_workers = max_workers
        self.strategy = strategy or PeleDatasetDistribution()

    def distribute_work(self, work_items: List[Any], worker_function: Callable) -> List[Any]:
        """Distribute work across thread pool."""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
        except ImportError:
            raise WorkDistributionError("ThreadPoolExecutor not available", len(work_items), 0)

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(worker_function, item): item
                for item in work_items
            }

            for future in as_completed(future_to_item):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass  # Failed items handled by worker

        return results

    def get_process_info(self) -> Dict[str, int]:
        """Get thread pool information."""
        import os
        return {'rank': 0, 'size': self.max_workers or os.cpu_count() or 1}


def create_distributor(mode: str, **kwargs) -> WorkDistributor:
    """Factory function to create appropriate distributor."""
    if mode == "mpi":
        return MPIDistributor(**kwargs)
    elif mode == "threads":
        return ThreadDistributor(**kwargs)
    else:
        raise ValueError(f"Unknown distribution mode: {mode}")


def create_strategy(strategy_name: str, **kwargs) -> DistributionStrategy:
    """Factory function to create distribution strategy."""
    strategies = {
        "round_robin": RoundRobinDistribution,
        "chunk": ChunkDistribution,
        "load_balanced": LoadBalancedDistribution,
        "random": RandomDistribution,
        "adaptive": AdaptiveDistribution,
        "pele_dataset": PeleDatasetDistribution,
        "pele_load_balanced": PeleLoadBalancedDistribution
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name](**kwargs)