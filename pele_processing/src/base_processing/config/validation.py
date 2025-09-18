"""
Configuration validation for the Pele processing system.
"""
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import os

from ..core.exceptions import ConfigurationError
from .settings import AppConfig, ProcessingConfig, VisualizationConfig, PathConfig


class ValidationRule:
    """Single validation rule."""

    def __init__(self, field_path: str, validator: Callable[[Any], bool],
                 message: str, severity: str = "error"):
        self.field_path = field_path
        self.validator = validator
        self.message = message
        self.severity = severity

    def validate(self, config: Dict[str, Any]) -> Optional[str]:
        """Validate rule against config."""
        try:
            value = self._get_nested_value(config, self.field_path)
            if not self.validator(value):
                return f"{self.severity.upper()}: {self.field_path} - {self.message}"
        except KeyError:
            return f"{self.severity.upper()}: Missing required field '{self.field_path}'"
        except Exception as e:
            return f"{self.severity.upper()}: Error validating '{self.field_path}': {e}"
        return None

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = path.split('.')
        value = config
        for key in keys:
            value = value[key]
        return value


class ConfigValidator:
    """Configuration validator with comprehensive rules."""

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._setup_validation_rules()

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        for rule in self.rules:
            issue = rule.validate(config)
            if issue:
                issues.append(issue)
        return issues

    def validate_config_object(self, config: AppConfig) -> List[str]:
        """Validate AppConfig object."""
        # Convert to dict for validation
        config_dict = self._config_to_dict(config)
        return self.validate(config_dict)

    def is_valid(self, config: Union[Dict[str, Any], AppConfig]) -> bool:
        """Check if configuration is valid."""
        if isinstance(config, AppConfig):
            issues = self.validate_config_object(config)
        else:
            issues = self.validate(config)

        # Only count errors, not warnings
        errors = [issue for issue in issues if issue.startswith("ERROR")]
        return len(errors) == 0

    def _setup_validation_rules(self):
        """Setup all validation rules."""
        # Path validation rules
        self.rules.extend([
            ValidationRule(
                "paths.input_directory",
                lambda x: x is not None,
                "Input directory must be specified"
            ),
            ValidationRule(
                "paths.output_directory",
                lambda x: x is not None,
                "Output directory must be specified"
            )
        ])

        # Thermodynamic validation
        self.rules.extend([
            ValidationRule(
                "thermodynamics.temperature",
                lambda x: 200 <= x <= 5000,
                "Temperature must be between 200K and 5000K"
            ),
            ValidationRule(
                "thermodynamics.pressure",
                lambda x: 1000 <= x <= 1e8,
                "Pressure must be between 1kPa and 100MPa"
            ),
            ValidationRule(
                "thermodynamics.equivalence_ratio",
                lambda x: 0.1 <= x <= 10.0,
                "Equivalence ratio must be between 0.1 and 10.0"
            )
        ])

        # Processing validation
        self.rules.extend([
            ValidationRule(
                "processing.flame_temperature",
                lambda x: 1000 <= x <= 4000,
                "Flame temperature must be between 1000K and 4000K"
            ),
            ValidationRule(
                "processing.shock_pressure_ratio",
                lambda x: 1.0 < x <= 100.0,
                "Shock pressure ratio must be > 1.0"
            ),
            ValidationRule(
                "processing.transport_species",
                lambda x: x and isinstance(x, str),
                "Transport species must be a non-empty string"
            )
        ])

        # Visualization validation
        self.rules.extend([
            ValidationRule(
                "visualization.frame_rate",
                lambda x: 0.1 <= x <= 60.0,
                "Frame rate must be between 0.1 and 60 fps"
            ),
            ValidationRule(
                "visualization.local_window_size",
                lambda x: 1e-6 <= x <= 0.1,
                "Local window size must be between 1μm and 10cm"
            )
        ])

        # Parallel processing validation
        self.rules.extend([
            ValidationRule(
                "parallel.max_workers",
                lambda x: x is None or (isinstance(x, int) and x > 0),
                "Max workers must be positive integer or None"
            ),
            ValidationRule(
                "parallel.mpi_timeout",
                lambda x: 1.0 <= x <= 3600.0,
                "MPI timeout must be between 1s and 1 hour"
            )
        ])

    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert AppConfig to nested dictionary."""
        import dataclasses

        def convert_value(value):
            if dataclasses.is_dataclass(value):
                return {field.name: convert_value(getattr(value, field.name))
                        for field in dataclasses.fields(value)}
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, Path):
                return str(value)
            else:
                return value

        return convert_value(config)


class PathValidator:
    """Specialized validator for path configurations."""

    @staticmethod
    def validate_input_paths(paths: PathConfig) -> List[str]:
        """Validate input paths exist and are readable."""
        issues = []

        # Input directory
        if not paths.input_directory.exists():
            issues.append(f"ERROR: Input directory does not exist: {paths.input_directory}")
        elif not paths.input_directory.is_dir():
            issues.append(f"ERROR: Input path is not a directory: {paths.input_directory}")
        elif not os.access(paths.input_directory, os.R_OK):
            issues.append(f"ERROR: No read permission for input directory: {paths.input_directory}")

        # Mechanism file
        if paths.mechanism_file:
            if not paths.mechanism_file.exists():
                issues.append(f"ERROR: Mechanism file does not exist: {paths.mechanism_file}")
            elif not paths.mechanism_file.is_file():
                issues.append(f"ERROR: Mechanism path is not a file: {paths.mechanism_file}")

        return issues

    @staticmethod
    def validate_output_paths(paths: PathConfig) -> List[str]:
        """Validate output paths are writable."""
        issues = []

        # Output directory
        try:
            paths.output_directory.mkdir(parents=True, exist_ok=True)
            if not os.access(paths.output_directory, os.W_OK):
                issues.append(f"ERROR: No write permission for output directory: {paths.output_directory}")
        except Exception as e:
            issues.append(f"ERROR: Cannot create output directory {paths.output_directory}: {e}")

        # Log directory
        if paths.log_directory:
            try:
                paths.log_directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"WARNING: Cannot create log directory {paths.log_directory}: {e}")

        return issues


class ConsistencyValidator:
    """Validates configuration consistency across modules."""

    @staticmethod
    def validate_processing_consistency(config: AppConfig) -> List[str]:
        """Validate processing configuration consistency."""
        issues = []

        # If analyzing flame thickness, need 2D data
        if config.processing.analyze_flame_thickness:
            if not any([
                config.processing.analyze_flame_surface_length,
                config.processing.analyze_flame_consumption_rate
            ]):
                issues.append("WARNING: Flame thickness analysis works best with surface length or consumption rate")

        # Shock analysis consistency
        if config.processing.analyze_shock_velocity and not config.processing.analyze_shock_position:
            issues.append("ERROR: Cannot analyze shock velocity without position tracking")

        return issues

    @staticmethod
    def validate_visualization_consistency(config: AppConfig) -> List[str]:
        """Validate visualization configuration consistency."""
        issues = []

        # Animation consistency
        if config.visualization.generate_animations:
            has_animated_fields = any([
                config.visualization.animate_temperature,
                config.visualization.animate_pressure,
                config.visualization.animate_velocity,
                config.visualization.animate_heat_release,
                config.visualization.animate_schlieren,
                config.visualization.animate_streamlines
            ])

            if not has_animated_fields:
                issues.append("WARNING: Animation enabled but no fields selected for animation")

        # Local view consistency
        if config.visualization.enable_local_views:
            if config.visualization.local_window_size <= 0:
                issues.append("ERROR: Local window size must be positive when local views enabled")

        return issues

    @staticmethod
    def validate_parallel_consistency(config: AppConfig) -> List[str]:
        """Validate parallel configuration consistency."""
        issues = []

        # MPI consistency
        if config.parallel.use_mpi:
            if config.parallel.mode != "parallel_mpi":
                issues.append("WARNING: MPI enabled but mode is not 'parallel_mpi'")

        # Worker count consistency
        if config.parallel.max_workers is not None:
            if config.parallel.max_workers > 64:
                issues.append("WARNING: Very high worker count may cause resource issues")

        return issues


def validate_complete_config(config: AppConfig) -> List[str]:
    """Perform complete validation of configuration."""
    all_issues = []

    # Basic validation
    validator = ConfigValidator()
    all_issues.extend(validator.validate_config_object(config))

    # Path validation
    all_issues.extend(PathValidator.validate_input_paths(config.paths))
    all_issues.extend(PathValidator.validate_output_paths(config.paths))

    # Consistency validation
    all_issues.extend(ConsistencyValidator.validate_processing_consistency(config))
    all_issues.extend(ConsistencyValidator.validate_visualization_consistency(config))
    all_issues.extend(ConsistencyValidator.validate_parallel_consistency(config))

    return all_issues


def raise_on_errors(issues: List[str]) -> None:
    """Raise ConfigurationError if any errors found."""
    errors = [issue for issue in issues if issue.startswith("ERROR")]
    if errors:
        raise ConfigurationError(
            "validation",
            f"Configuration validation failed with {len(errors)} errors",
            {"errors": errors}
        )