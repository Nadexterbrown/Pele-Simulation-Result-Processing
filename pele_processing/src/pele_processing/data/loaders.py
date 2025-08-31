"""
Data loading implementations for the Pele processing system.
"""
from typing import Any, List, Optional, Union
from pathlib import Path

from ..core.interfaces import DataLoader
from ..core.domain import DatasetInfo, BoundingBox
from ..core.exceptions import DataLoadError

try:
    import yt

    yt.set_log_level(0)
    YT_AVAILABLE = True
except ImportError:
    yt = None
    YT_AVAILABLE = False


class YTDataLoader(DataLoader):
    """YT-based data loader for Pele datasets."""

    def __init__(self):
        if not YT_AVAILABLE:
            raise DataLoadError("", "YT not available")
        yt.enable_parallelism()

    def load_dataset(self, path: Union[str, Path]) -> Any:
        """Load dataset using YT."""
        try:
            dataset = yt.load(str(path))
            dataset.force_periodicity()
            return dataset
        except Exception as e:
            raise DataLoadError(str(path), str(e))

    def get_dataset_info(self, dataset: Any) -> DatasetInfo:
        """Extract metadata from YT dataset."""
        try:
            bounds = BoundingBox(
                min_x=float(dataset.domain_left_edge[0].to_value()),
                max_x=float(dataset.domain_right_edge[0].to_value()),
                min_y=float(dataset.domain_left_edge[1].to_value()),
                max_y=float(dataset.domain_right_edge[1].to_value())
            )

            return DatasetInfo(
                path=Path(dataset.filename),
                basename=dataset.basename,
                timestamp=float(dataset.current_time.to_value()),
                domain_bounds=bounds,
                max_refinement_level=dataset.index.max_level,
                grid_spacing=float(dataset.index.get_smallest_dx().to_value())
            )
        except Exception as e:
            raise DataLoadError(dataset.filename, f"Metadata extraction failed: {e}")

    def list_available_fields(self, dataset: Any) -> List[str]:
        """List available fields in dataset."""
        try:
            return [field_name for field_type, field_name in dataset.field_list]
        except Exception as e:
            raise DataLoadError(dataset.filename, f"Field listing failed: {e}")


class CachedDataLoader(DataLoader):
    """Cached data loader wrapper."""

    def __init__(self, base_loader: DataLoader, cache_size: int = 10):
        self.base_loader = base_loader
        self.cache = {}
        self.cache_order = []
        self.max_cache_size = cache_size

    def load_dataset(self, path: Union[str, Path]) -> Any:
        """Load with caching."""
        path_str = str(path)

        if path_str in self.cache:
            return self.cache[path_str]

        dataset = self.base_loader.load_dataset(path)

        # Add to cache
        self.cache[path_str] = dataset
        self.cache_order.append(path_str)

        # Evict oldest if over limit
        if len(self.cache) > self.max_cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        return dataset

    def get_dataset_info(self, dataset: Any) -> DatasetInfo:
        return self.base_loader.get_dataset_info(dataset)

    def list_available_fields(self, dataset: Any) -> List[str]:
        return self.base_loader.list_available_fields(dataset)


def create_data_loader(loader_type: str = "yt", **kwargs) -> DataLoader:
    """Factory for data loaders."""
    if loader_type == "yt":
        base_loader = YTDataLoader()
        cache_size = kwargs.get('cache_size', 0)
        if cache_size > 0:
            return CachedDataLoader(base_loader, cache_size)
        return base_loader
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")