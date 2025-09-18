"""
Configuration management for the Pele processing system.
"""

# Core configuration classes
from .settings import (
    # Main configuration
    AppConfig,

    # Configuration sections
    ThermodynamicConfig, ProcessingConfig, VisualizationConfig,
    ParallelConfig, PathConfig, LoggingConfig, PerformanceConfig,

    # Enums
    ProcessingMode, AnimationFormat, LogLevel,

    # Presets
    DEFAULT_FLAME_CONFIG, DEFAULT_SHOCK_CONFIG, DEFAULT_FULL_CONFIG
)

# Configuration loading and validation
from .loader import (
    ConfigLoader, load_config, create_default_config, save_config
)

from .validation import (
    ConfigValidator, PathValidator, ConsistencyValidator,
    ValidationRule, validate_complete_config, raise_on_errors
)

__all__ = [
    # Main config classes
    'AppConfig',
    'ThermodynamicConfig',
    'ProcessingConfig',
    'VisualizationConfig',
    'ParallelConfig',
    'PathConfig',
    'LoggingConfig',
    'PerformanceConfig',

    # Enums
    'ProcessingMode',
    'AnimationFormat',
    'LogLevel',

    # Presets
    'DEFAULT_FLAME_CONFIG',
    'DEFAULT_SHOCK_CONFIG',
    'DEFAULT_FULL_CONFIG',

    # Loading/saving
    'ConfigLoader',
    'load_config',
    'create_default_config',
    'save_config',

    # Validation
    'ConfigValidator',
    'PathValidator',
    'ConsistencyValidator',
    'ValidationRule',
    'validate_complete_config',
    'raise_on_errors'
]

__version__ = "1.0.0"