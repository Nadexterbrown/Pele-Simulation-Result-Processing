"""
Data loading implementations for the Pele post processing system.
"""

from typing import Any, List, Optional, Union
from pathlib import Path
import pandas as pd

class CSVDataLoader:
    """CSV-based data loader for Pele post-processing datasets."""

    def load_dataset(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load dataset from a CSV file."""
        try:
            data = pd.read_csv(path)
            return data
        except Exception as e:
            raise IOError(f"Failed to load dataset from {path}: {e}")

    def get_dataset_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata from the DataFrame."""
        try:
            info = {
                "num_rows": len(data),
                "num_columns": len(data.columns),
                "columns": list(data.columns)
            }
            return info
        except Exception as e:
            raise ValueError(f"Failed to extract metadata: {e}")

    def list_available_fields(self, data: pd.DataFrame) -> List[str]:
        """List available fields (columns) in the DataFrame."""
        try:
            return list(data.columns)
        except Exception as e:
            raise ValueError(f"Failed to list fields: {e}")

