"""
Core module for the Pele processing system.

This module provides the foundational components:
- Domain models representing business entities
- Abstract interfaces for dependency injection
- Custom exceptions for error handling
- Dependency injection container for service management
"""

# Domain models
from .domain import (
    # Enums
    WaveType, Direction,

    # Geometric types
    Point2D, Point3D, BoundingBox,

    # Physical properties
    ThermodynamicState, FlameProperties, ShockProperties, GasProperties, BurnedGasProperties,

    # Data structures
    SpeciesData, FieldData, DatasetInfo,

    # Processing results
    ProcessingResult, ProcessingBatch,

    # Visualization
    AnimationFrame, VisualizationRequest
)

# Abstract interfaces
from .interfaces import (
    # Data layer
    DataLoader, DataExtractor, DataValidator,

    # Analysis layer
    WaveTracker, FlameAnalyzer, ShockAnalyzer, ThermodynamicCalculator, BurnedGasAnalyzer,

    # Visualization layer
    FrameGenerator, AnimationBuilder, OutputFormatter,

    # Parallel processing
    WorkDistributor, ParallelCoordinator,

    # Utilities
    ConfigurationLoader, Logger, UnitConverter
)

# Exceptions
from .exceptions import (
    # Base exceptions
    PeleProcessingError,

    # Data layer exceptions
    DataError, DataLoadError, DataExtractionError, ValidationError, FieldNotFoundError,

    # Analysis layer exceptions
    AnalysisError, WaveNotFoundError, FlameAnalysisError, ShockAnalysisError,
    BurnedGasAnalysisError, ThermodynamicError, ConvergenceError,

    # Visualization exceptions
    VisualizationError, PlotGenerationError, AnimationError, OutputFormatError,

    # Parallel processing exceptions
    ParallelProcessingError, MPIError, WorkDistributionError, ProcessSynchronizationError,

    # Configuration exceptions
    ConfigurationError, DependencyError, ResourceError, FileSystemError,

    # Utility functions
    format_exception_chain, create_error_context
)

# Dependency injection
from .container import (
    Container, ServiceDescriptor, ServiceLifetime, ServiceRegistry, ServiceScope,
    inject, get_global_container, set_global_container, configure_default_services
)

__all__ = [
    # Domain
    'WaveType', 'Direction', 'Point2D', 'Point3D', 'BoundingBox',
    'ThermodynamicState', 'FlameProperties', 'ShockProperties', 'GasProperties', 'BurnedGasProperties',
    'SpeciesData', 'FieldData', 'DatasetInfo', 'ProcessingResult', 'ProcessingBatch',
    'AnimationFrame', 'VisualizationRequest',

    # Interfaces
    'DataLoader', 'DataExtractor', 'DataValidator',
    'WaveTracker', 'FlameAnalyzer', 'ShockAnalyzer', 'ThermodynamicCalculator', 'BurnedGasAnalyzer',
    'FrameGenerator', 'AnimationBuilder', 'OutputFormatter',
    'WorkDistributor', 'ParallelCoordinator',
    'ConfigurationLoader', 'Logger', 'UnitConverter',

    # Exceptions
    'PeleProcessingError', 'DataError', 'AnalysisError', 'VisualizationError',
    'ParallelProcessingError', 'ConfigurationError', 'format_exception_chain',

    # Container
    'Container', 'ServiceLifetime', 'ServiceRegistry', 'ServiceScope',
    'inject', 'get_global_container', 'set_global_container'
]

__version__ = "1.0.0"