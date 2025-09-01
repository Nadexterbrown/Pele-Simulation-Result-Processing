"""
Configuration settings for the Pele processing system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from enum import Enum


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL_MPI = "parallel_mpi"
    PARALLEL_THREADS = "parallel_threads"


class AnimationFormat(Enum):
    """Supported animation formats."""
    GIF = "gif"
    MP4 = "mp4"
    AVI = "avi"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ThermodynamicConfig:
    """Initial thermodynamic conditions."""
    temperature: float = 503.15  # K
    pressure: float = 1e6  # Pa
    equivalence_ratio: float = 1.0
    fuel: str = 'H2'
    oxidizer_composition: Dict[str, float] = field(default_factory=lambda: {'O2': 0.21, 'N2': 0.79})
    mechanism_file: Optional[Path] = None

    def __post_init__(self):
        if isinstance(self.mechanism_file, str):
            self.mechanism_file = Path(self.mechanism_file)


@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    # Wave detection
    flame_temperature: float = 2500.0  # K
    shock_pressure_ratio: float = 1.01
    transport_species: str = 'H2'

    # Extraction parameters
    extract_location: Optional[float] = None  # m, auto-detect if None
    extract_direction: str = 'x'

    # Analysis flags
    analyze_flame_position: bool = True
    analyze_flame_velocity: bool = True
    analyze_flame_thickness: bool = True
    analyze_flame_surface_length: bool = True
    analyze_flame_consumption_rate: bool = True
    analyze_flame_heat_release: bool = True

    analyze_shock_position: bool = True
    analyze_shock_velocity: bool = True
    analyze_shock_thermodynamics: bool = True

    analyze_burned_gas: bool = True
    analyze_unburned_gas: bool = False

    # Offsets for probe locations (meters)
    flame_offset: float = 0.0
    burned_gas_offset: float = -1e-5
    shock_pre_offset: float = 1e-5
    shock_post_offset: float = -1e-5


@dataclass
class VisualizationConfig:
    """Visualization and animation settings."""
    # Animation generation
    generate_animations: bool = True
    animation_formats: List[AnimationFormat] = field(default_factory=lambda: [AnimationFormat.GIF])
    frame_rate: float = 5.0

    # Field animations
    animate_temperature: bool = True
    animate_pressure: bool = True
    animate_velocity: bool = True
    animate_heat_release: bool = True
    animate_density: bool = False

    # Specialized visualizations
    animate_schlieren: bool = False
    animate_streamlines: bool = False
    animate_flame_geometry: bool = True
    animate_flame_thickness: bool = False

    # Local view settings
    enable_local_views: bool = True
    local_window_size: float = 1e-3  # m

    # Plot styling
    figure_dpi: int = 150
    figure_width: float = 10.0  # inches
    figure_height: float = 6.0  # inches
    font_size: int = 12

    # Output formats
    plot_formats: List[str] = field(default_factory=lambda: ['png'])
    high_quality_output: bool = False


@dataclass
class ParallelConfig:
    """Parallel processing configuration."""
    mode: ProcessingMode = ProcessingMode.SEQUENTIAL
    max_workers: Optional[int] = None  # Auto-detect if None

    # MPI settings
    use_mpi: bool = False
    mpi_timeout: float = 300.0  # seconds

    # Threading settings
    thread_pool_size: Optional[int] = None
    chunk_size: int = 1


@dataclass
class PathConfig:
    """File and directory paths."""
    # Input paths
    input_directory: Path
    mechanism_file: Optional[Path] = None

    # Output paths - now defaults to a new 'pele_results' folder in working directory
    output_directory: Path = Path("./pele_results")
    log_directory: Optional[Path] = None  # Defaults to output_directory/logs

    # Subdirectories (relative to output_directory)
    results_subdir: str = "results"
    animations_subdir: str = "animations"
    frames_subdir: str = "frames"
    plots_subdir: str = "plots"

    def __post_init__(self):
        # Convert strings to Path objects
        self.input_directory = Path(self.input_directory)
        self.output_directory = Path(self.output_directory)

        if isinstance(self.mechanism_file, str):
            self.mechanism_file = Path(self.mechanism_file)

        if self.log_directory is None:
            self.log_directory = self.output_directory / "logs"
        else:
            self.log_directory = Path(self.log_directory)

    def get_results_path(self) -> Path:
        return self.output_directory / self.results_subdir

    def get_animations_path(self) -> Path:
        return self.output_directory / self.animations_subdir

    def get_frames_path(self, field_name: str) -> Path:
        return self.output_directory / self.frames_subdir / f"{field_name}_frames"

    def get_plots_path(self) -> Path:
        return self.output_directory / self.plots_subdir


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_to_console: bool = True

    # File logging
    log_filename: str = "pele_processing.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    # Format settings
    include_timestamp: bool = True
    include_process_id: bool = True
    include_rank: bool = True  # For MPI

    # Per-rank logging (MPI)
    separate_rank_logs: bool = True
    rank_log_template: str = "pele_processing_rank_{rank}.log"


@dataclass
class PerformanceConfig:
    """Performance and resource management."""
    # Memory management
    memory_limit_gb: Optional[float] = None
    cache_size_mb: int = 512

    # Processing limits
    max_processing_time: Optional[float] = None  # seconds
    checkpoint_interval: Optional[float] = None  # seconds

    # Optimization flags
    use_numba: bool = True
    use_cython: bool = False
    optimize_memory_usage: bool = True

    # Progress reporting
    progress_reporting: bool = True
    progress_interval: int = 10  # datasets


@dataclass
class AppConfig:
    """Complete application configuration."""
    thermodynamics: ThermodynamicConfig = field(default_factory=ThermodynamicConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    paths: Optional[PathConfig] = None  # Must be provided
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Metadata
    version: str = "1.0.0"
    description: str = "Pele 2D processing configuration"

    def __post_init__(self):
        if self.paths is None:
            raise ValueError("PathConfig must be provided")

    @property
    def is_parallel_enabled(self) -> bool:
        return self.parallel.mode != ProcessingMode.SEQUENTIAL

    @property
    def requires_mpi(self) -> bool:
        return self.parallel.mode == ProcessingMode.PARALLEL_MPI

    def get_output_path(self, *parts: str) -> Path:
        """Get path relative to output directory."""
        return self.paths.output_directory.joinpath(*parts)

    def create_output_directories(self) -> None:
        """Create all required output directories."""
        dirs_to_create = [
            self.paths.output_directory,
            self.paths.log_directory,
            self.paths.get_results_path(),
            self.paths.get_animations_path(),
            self.paths.get_plots_path(),
        ]

        # Add frame directories for each animated field
        if self.visualization.animate_temperature:
            dirs_to_create.append(self.paths.get_frames_path("temperature"))
        if self.visualization.animate_pressure:
            dirs_to_create.append(self.paths.get_frames_path("pressure"))
        if self.visualization.animate_velocity:
            dirs_to_create.append(self.paths.get_frames_path("velocity"))
        if self.visualization.animate_heat_release:
            dirs_to_create.append(self.paths.get_frames_path("heat_release"))

        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)


# Configuration presets
DEFAULT_FLAME_CONFIG = AppConfig(
    paths=PathConfig(
        input_directory=Path("./data"),
        output_directory=Path("./pele_results/flame_analysis")
    ),
    processing=ProcessingConfig(
        analyze_shock_position=False,
        analyze_shock_velocity=False,
        analyze_shock_thermodynamics=False
    ),
    visualization=VisualizationConfig(
        animate_schlieren=False,
        animate_streamlines=False
    )
)

DEFAULT_SHOCK_CONFIG = AppConfig(
    paths=PathConfig(
        input_directory=Path("./data"),
        output_directory=Path("./pele_results/shock_analysis")
    ),
    processing=ProcessingConfig(
        analyze_flame_thickness=False,
        analyze_flame_surface_length=False,
        analyze_flame_consumption_rate=False
    )
)

DEFAULT_FULL_CONFIG = AppConfig(
    paths=PathConfig(
        input_directory=Path("./data"),
        output_directory=Path("./pele_results/full_analysis")
    ),
    visualization=VisualizationConfig(
        animate_schlieren=True,
        animate_streamlines=True,
        animate_flame_thickness=True
    ),
    parallel=ParallelConfig(
        mode=ProcessingMode.PARALLEL_MPI,
        use_mpi=True
    )
)