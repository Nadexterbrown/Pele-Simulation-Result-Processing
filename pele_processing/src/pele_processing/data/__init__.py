"""
Data module for the Pele processing system.
"""

# Loaders
from .loaders import (
    YTDataLoader, CachedDataLoader, create_data_loader
)

# Extractors
from .extractors import (
    PeleDataExtractor, MultiLevelExtractor, create_data_extractor
)

# Processors
from .processors import (
    DataProcessor, FilterProcessor, DerivativeProcessor,
    DomainProcessor, BatchProcessor,
    create_standard_processor, create_analysis_processor
)

# Validators
from .validators import (
    PeleDataValidator, PhysicalConsistencyValidator,
    DataQualityMetrics, validate_dataset_batch
)

__all__ = [
    # Loaders
    'YTDataLoader',
    'CachedDataLoader',
    'create_data_loader',

    # Extractors
    'PeleDataExtractor',
    'MultiLevelExtractor',
    'create_data_extractor',

    # Processors
    'DataProcessor',
    'FilterProcessor',
    'DerivativeProcessor',
    'DomainProcessor',
    'BatchProcessor',
    'create_standard_processor',
    'create_analysis_processor',

    # Validators
    'PeleDataValidator',
    'PhysicalConsistencyValidator',
    'DataQualityMetrics',
    'validate_dataset_batch'
]

__version__ = "1.0.0"