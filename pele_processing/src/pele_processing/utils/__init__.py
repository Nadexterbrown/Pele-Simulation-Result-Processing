"""
Utilities module for the Pele processing system.
"""

# Logging utilities
from .logging import (
    PeleLogger, MPILogger, ProgressLogger,
    create_logger, setup_logging
)

# Filesystem utilities
from .filesystem import (
    ensure_long_path_prefix, safe_mkdir, safe_copy,
    find_files, load_dataset_paths, sort_dataset_paths,
    get_file_size, disk_usage, ensure_directory_exists,
    clean_filename, DirectoryManager, FileFilter
)

# Unit conversion
from .units import (
    UnitConverter, PeleUnitConverter, create_unit_converter
)

# Constants
from .constants import (
    # Physical constants
    UNIVERSAL_GAS_CONSTANT, AVOGADRO_NUMBER, BOLTZMANN_CONSTANT,
    STANDARD_TEMPERATURE, STANDARD_PRESSURE,

    # Combustion constants
    STOICHIOMETRIC_AIR_FUEL_H2, HEATING_VALUE_H2,
    DEFAULT_FLAME_TEMPERATURE, DEFAULT_SHOCK_PRESSURE_RATIO,

    # Analysis parameters
    MIN_FLAME_THICKNESS_CELLS, CONVERGENCE_TOLERANCE,

    # System limits
    MAX_DATASET_SIZE, MAX_LOG_FILE_SIZE,

    # Visualization defaults
    DEFAULT_DPI, DEFAULT_FIGURE_SIZE, DEFAULT_FRAME_RATE,

    # Species and error codes
    COMMON_SPECIES, ERROR_CODES
)

__all__ = [
    # Logging
    'PeleLogger',
    'MPILogger',
    'ProgressLogger',
    'create_logger',
    'setup_logging',

    # Filesystem
    'ensure_long_path_prefix',
    'safe_mkdir',
    'safe_copy',
    'find_files',
    'load_dataset_paths',
    'sort_dataset_paths',
    'get_file_size',
    'disk_usage',
    'ensure_directory_exists',
    'clean_filename',
    'DirectoryManager',
    'FileFilter',

    # Units
    'UnitConverter',
    'PeleUnitConverter',
    'create_unit_converter',

    # Constants
    'UNIVERSAL_GAS_CONSTANT',
    'STANDARD_TEMPERATURE',
    'STANDARD_PRESSURE',
    'DEFAULT_FLAME_TEMPERATURE',
    'DEFAULT_SHOCK_PRESSURE_RATIO',
    'CONVERGENCE_TOLERANCE',
    'DEFAULT_DPI',
    'DEFAULT_FRAME_RATE',
    'COMMON_SPECIES',
    'ERROR_CODES'
]

__version__ = "1.0.0"