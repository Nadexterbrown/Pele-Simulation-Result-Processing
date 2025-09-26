"""
Pele Simulation Result Processing System

A comprehensive parallel processing system for analyzing Pele combustion simulation data.
Supports flame analysis, shock detection, thermodynamic calculations, and visualization.
"""

# Core components
from .core import (
    # Domain models
    WaveType, Direction, Point2D, Point3D, BoundingBox,
    ThermodynamicState, FlameProperties, ShockProperties, GasProperties, BurnedGasProperties,
    PressureWaveProperties, SpeciesData, FieldData, DatasetInfo, ProcessingResult, ProcessingBatch,
    AnimationFrame, VisualizationRequest,

    # Interfaces
    DataLoader, DataExtractor, DataValidator,
    WaveTracker, FlameAnalyzer, ShockAnalyzer, ThermodynamicCalculator, BurnedGasAnalyzer,
    FrameGenerator, AnimationBuilder, OutputFormatter,
    WorkDistributor, ParallelCoordinator,
    ConfigurationLoader, Logger, UnitConverter,

    # Exceptions
    PeleProcessingError, DataError, AnalysisError, VisualizationError,
    ParallelProcessingError, ConfigurationError,

    # Container
    Container, ServiceLifetime, inject, get_global_container
)

# Configuration management
from .config import (
    AppConfig, ThermodynamicConfig, ProcessingConfig, VisualizationConfig,
    ParallelConfig, PathConfig, LoggingConfig, PerformanceConfig,
    ProcessingMode, AnimationFormat, LogLevel,
    DEFAULT_FLAME_CONFIG, DEFAULT_SHOCK_CONFIG, DEFAULT_FULL_CONFIG,
    ConfigLoader, load_config, create_default_config, save_config,
    validate_complete_config
)

# Data processing
from .data import (
    YTDataLoader, CachedDataLoader, create_data_loader,
    PeleDataExtractor, MultiLevelExtractor, create_data_extractor,
    DataProcessor, FilterProcessor, DerivativeProcessor,
    DomainProcessor, BatchProcessor,
    create_standard_processor, create_analysis_processor,
    PeleDataValidator, PhysicalConsistencyValidator,
    DataQualityMetrics, validate_dataset_batch
)

# Analysis components
from .analysis import (
    PeleFlameAnalyzer, create_flame_analyzer,
    PeleShockAnalyzer, create_shock_analyzer,
    PeleBurnedGasAnalyzer, create_burned_gas_analyzer,
    CanteraThermodynamicCalculator, create_thermodynamic_calculator,
    GeometryAnalyzer, FlameGeometryAnalyzer, create_geometry_analyzer
)

# Import pressure wave analyzer separately since it's in a different module
try:
    from .analysis.pressure_wave import (
        PelePressureWaveAnalyzer,
        create_pressure_wave_analyzer,
        DetectionMethod
    )
    PRESSURE_WAVE_AVAILABLE = True
except ImportError:
    PRESSURE_WAVE_AVAILABLE = False
    PelePressureWaveAnalyzer = None
    create_pressure_wave_analyzer = None
    DetectionMethod = None

# Parallel processing
from .parallel import (
    MPICoordinator, SequentialCoordinator, ThreadPoolCoordinator,
    create_coordinator,
    WorkerBase, MPIWorker, ThreadWorker, ProcessWorker,
    WorkerManager, TaskQueue, WorkerPool,
    DistributionStrategy, RoundRobinDistribution, ChunkDistribution,
    LoadBalancedDistribution, RandomDistribution, AdaptiveDistribution,
    MPIDistributor, ThreadDistributor,
    create_distributor, create_strategy,
    ProcessingStrategy, SequentialStrategy, MPIStrategy, ThreadPoolStrategy,
    HybridStrategy, AdaptiveStrategy,
    create_processing_strategy, create_default_adaptive_strategy
)

# Visualization
from .visualization import (
    StandardPlotter, LocalViewPlotter, StatisticalPlotter, ComparisonPlotter,
    FrameAnimator, BatchAnimator, InteractiveAnimator,
    TableFormatter, CSVFormatter, JSONFormatter, create_formatter,
    SchlierenVisualizer, StreamlineVisualizer, ContourVisualizer
)

# Utilities
from .utils import (
    PeleLogger, MPILogger, ProgressLogger, create_logger, setup_logging,
    ensure_long_path_prefix, safe_mkdir, safe_copy, find_files,
    load_dataset_paths, sort_dataset_paths, get_file_size, disk_usage,
    ensure_directory_exists, clean_filename, DirectoryManager, FileFilter,
    UnitConverter, PeleUnitConverter, create_unit_converter,
    UNIVERSAL_GAS_CONSTANT, STANDARD_TEMPERATURE, STANDARD_PRESSURE,
    DEFAULT_FLAME_TEMPERATURE, DEFAULT_SHOCK_PRESSURE_RATIO,
    CONVERGENCE_TOLERANCE, DEFAULT_DPI, DEFAULT_FRAME_RATE,
    COMMON_SPECIES, ERROR_CODES
)

# Version and metadata
__version__ = "1.0.0"
__author__ = "Nolan Dexter-Brown"
__email__ = "nadexterbrown@gmail.com"
__description__ = "Parallel processing system for Pele combustion simulation analysis"
__url__ = "https://github.com/pele-combustion/pele-processing"


# Package-level convenience functions
def quick_analysis(input_dir, output_dir, config=None, parallel=True):
    """
    Quick analysis setup for common use cases.

    Args:
        input_dir: Directory containing Pele plotfiles
        output_dir: Directory for output results
        config: Optional AppConfig object
        parallel: Whether to use parallel processing

    Returns:
        ProcessingBatch with results
    """
    if config is None:
        config = create_default_config(input_dir, output_dir)

    # Setup logging
    logger = setup_logging(config.logging.level.value)

    # Create processing strategy
    if parallel:
        strategy = create_default_adaptive_strategy(logger)
    else:
        strategy = SequentialStrategy(logger)

    # Load dataset paths
    dataset_paths = load_dataset_paths(config.paths.input_directory)

    # Create processor function
    def process_dataset(path):
        # Load dataset
        loader = create_data_loader("yt")
        dataset = loader.load_dataset(path)

        # Extract data
        extractor = create_data_extractor("pele",
                                          unit_converter=create_unit_converter("pele"))
        field_data = extractor.extract_ray_data(dataset,
                                                config.processing.extract_location or 0.025)

        # Analyze
        result = ProcessingResult(dataset_info=loader.get_dataset_info(dataset))

        if config.processing.analyze_flame_position:
            flame_analyzer = create_flame_analyzer(config.processing.flame_temperature)
            result.flame_data = flame_analyzer.analyze_flame_properties(dataset, field_data)

        if config.processing.analyze_shock_position:
            shock_analyzer = create_shock_analyzer(config.processing.shock_pressure_ratio)
            result.shock_data = shock_analyzer.analyze_shock_properties(field_data)

        result.success = True
        return result

    # Execute processing
    return strategy.execute([str(p) for p in dataset_paths], process_dataset)


def create_analysis_pipeline(config):
    """
    Create a complete analysis pipeline from configuration.

    Args:
        config: AppConfig object

    Returns:
        Configured processing pipeline
    """
    # Validate configuration
    issues = validate_complete_config(config)
    if any("ERROR" in issue for issue in issues):
        raise ConfigurationError("pipeline", "Configuration validation failed")

    # Setup components
    logger = create_logger({
        'type': 'mpi' if config.requires_mpi else 'standard',
        'level': config.logging.level.value,
        'log_directory': str(config.paths.log_directory)
    })

    # Data components
    loader = create_data_loader("yt", cache_size=config.performance.cache_size_mb // 10)
    extractor = create_data_extractor("pele",
                                      unit_converter=create_unit_converter("pele"))

    # Analysis components
    analyzers = {}
    if config.processing.analyze_flame_position:
        analyzers['flame'] = create_flame_analyzer(config.processing.flame_temperature)
    if config.processing.analyze_shock_position:
        analyzers['shock'] = create_shock_analyzer(config.processing.shock_pressure_ratio)

    # Processing strategy
    if config.is_parallel_enabled:
        if config.requires_mpi:
            strategy = MPIStrategy(logger=logger)
        else:
            strategy = ThreadPoolStrategy(config.parallel.max_workers, logger=logger)
    else:
        strategy = SequentialStrategy(logger=logger)

    return {
        'loader': loader,
        'extractor': extractor,
        'analyzers': analyzers,
        'strategy': strategy,
        'logger': logger
    }


# Main exports
__all__ = [
    # Core
    'WaveType', 'Direction', 'ThermodynamicState', 'FlameProperties', 'ShockProperties',
    'PressureWaveProperties', 'FieldData', 'ProcessingResult', 'ProcessingBatch', 'DatasetInfo',
    'Container', 'get_global_container',

    # Configuration
    'AppConfig', 'load_config', 'create_default_config', 'DEFAULT_FLAME_CONFIG',

    # Data
    'create_data_loader', 'create_data_extractor', 'create_standard_processor',

    # Analysis
    'PeleFlameAnalyzer', 'create_flame_analyzer',
    'PeleShockAnalyzer', 'create_shock_analyzer',
    'PeleBurnedGasAnalyzer', 'create_burned_gas_analyzer',
    'CanteraThermodynamicCalculator', 'create_thermodynamic_calculator',
    'PelePressureWaveAnalyzer', 'create_pressure_wave_analyzer', 'DetectionMethod',

    # Parallel
    'create_processing_strategy', 'create_default_adaptive_strategy',

    # Visualization
    'StandardPlotter', 'FrameAnimator', 'create_formatter',

    # Utilities
    'setup_logging', 'create_unit_converter', 'load_dataset_paths',

    # Constants
    'DEFAULT_FLAME_TEMPERATURE', 'DEFAULT_SHOCK_PRESSURE_RATIO',

    # Convenience functions
    'quick_analysis', 'create_analysis_pipeline',

    # Metadata
    '__version__', '__author__', '__description__'
]