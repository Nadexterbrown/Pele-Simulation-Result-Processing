"""
Configuration loading from files and environment variables.
"""
from typing import Dict, Any, Optional, Union
from pathlib import Path
import os
import sys

from ..core.exceptions import ConfigurationError, FileSystemError
from .settings import (
    AppConfig, ThermodynamicConfig, ProcessingConfig, VisualizationConfig,
    ParallelConfig, PathConfig, LoggingConfig, PerformanceConfig,
    ProcessingMode, AnimationFormat, LogLevel
)
from .validation import validate_complete_config, raise_on_errors


class ConfigLoader:
    """Configuration loader supporting YAML, TOML, and environment variables."""

    def __init__(self):
        self._yaml_available = self._check_yaml()
        self._toml_available = self._check_toml()

    def load_from_file(self, config_path: Union[str, Path]) -> AppConfig:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileSystemError("read", str(config_path), "File does not exist")

        # Determine file format
        suffix = config_path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            config_dict = self._load_yaml(config_path)
        elif suffix in ['.toml']:
            config_dict = self._load_toml(config_path)
        elif suffix in ['.json']:
            config_dict = self._load_json(config_path)
        else:
            raise ConfigurationError(
                "file_format",
                f"Unsupported configuration format: {suffix}",
                {"supported_formats": [".yaml", ".yml", ".toml", ".json"]}
            )

        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)

        # Convert to AppConfig object
        return self._dict_to_config(config_dict)

    def load_from_dict(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Load configuration from dictionary."""
        return self._dict_to_config(config_dict)

    def load_from_env(self, prefix: str = "PELE_") -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        config_dict = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert PELE_PROCESSING_FLAME_TEMPERATURE to processing.flame_temperature
                config_key = key[len(prefix):].lower().replace('_', '.')
                config_dict[config_key] = self._parse_env_value(value)

        return config_dict

    def save_to_file(self, config: AppConfig, output_path: Union[str, Path],
                     format: str = "yaml") -> None:
        """Save configuration to file."""
        output_path = Path(output_path)
        config_dict = self._config_to_dict(config)

        if format.lower() == "yaml":
            self._save_yaml(config_dict, output_path)
        elif format.lower() == "toml":
            self._save_toml(config_dict, output_path)
        elif format.lower() == "json":
            self._save_json(config_dict, output_path)
        else:
            raise ConfigurationError("file_format", f"Unsupported save format: {format}")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self._yaml_available:
            raise ConfigurationError("dependency", "PyYAML not available for YAML config loading")

        import yaml

        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError("parsing", f"Invalid YAML in {path}: {e}")
        except Exception as e:
            raise FileSystemError("read", str(path), str(e))

    def _load_toml(self, path: Path) -> Dict[str, Any]:
        """Load TOML configuration file."""
        if not self._toml_available:
            raise ConfigurationError("dependency", "tomllib/toml not available for TOML config loading")

        try:
            if sys.version_info >= (3, 11):
                import tomllib
                with open(path, 'rb') as f:
                    return tomllib.load(f)
            else:
                import toml
                with open(path, 'r') as f:
                    return toml.load(f)
        except Exception as e:
            raise ConfigurationError("parsing", f"Invalid TOML in {path}: {e}")

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        import json

        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError("parsing", f"Invalid JSON in {path}: {e}")
        except Exception as e:
            raise FileSystemError("read", str(path), str(e))

    def _save_yaml(self, config_dict: Dict[str, Any], path: Path) -> None:
        """Save configuration as YAML."""
        if not self._yaml_available:
            raise ConfigurationError("dependency", "PyYAML not available for YAML saving")

        import yaml

        try:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise FileSystemError("write", str(path), str(e))

    def _save_toml(self, config_dict: Dict[str, Any], path: Path) -> None:
        """Save configuration as TOML."""
        if not self._toml_available:
            raise ConfigurationError("dependency", "toml not available for TOML saving")

        import toml

        try:
            with open(path, 'w') as f:
                toml.dump(config_dict, f)
        except Exception as e:
            raise FileSystemError("write", str(path), str(e))

    def _save_json(self, config_dict: Dict[str, Any], path: Path) -> None:
        """Save configuration as JSON."""
        import json

        try:
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
        except Exception as e:
            raise FileSystemError("write", str(path), str(e))

    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        env_overrides = self.load_from_env()

        for key, value in env_overrides.items():
            self._set_nested_value(config_dict, key, value)

        return config_dict

    def _set_nested_value(self, config_dict: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = config_dict

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Numeric values
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # List values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]

        # String value
        return value

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object."""
        # Extract sections with defaults
        thermodynamics_dict = config_dict.get('thermodynamics', {})
        processing_dict = config_dict.get('processing', {})
        visualization_dict = config_dict.get('visualization', {})
        parallel_dict = config_dict.get('parallel', {})
        paths_dict = config_dict.get('paths', {})
        logging_dict = config_dict.get('logging', {})
        performance_dict = config_dict.get('performance', {})

        # Convert paths
        if 'input_directory' in paths_dict:
            paths_dict['input_directory'] = Path(paths_dict['input_directory'])
        if 'output_directory' in paths_dict:
            paths_dict['output_directory'] = Path(paths_dict['output_directory'])
        if 'mechanism_file' in paths_dict:
            paths_dict['mechanism_file'] = Path(paths_dict['mechanism_file'])

        # Convert enums
        if 'mode' in parallel_dict and isinstance(parallel_dict['mode'], str):
            parallel_dict['mode'] = ProcessingMode(parallel_dict['mode'])

        if 'animation_formats' in visualization_dict:
            formats = visualization_dict['animation_formats']
            if isinstance(formats, list):
                visualization_dict['animation_formats'] = [
                    AnimationFormat(f) if isinstance(f, str) else f for f in formats
                ]

        if 'level' in logging_dict and isinstance(logging_dict['level'], str):
            logging_dict['level'] = LogLevel(logging_dict['level'])

        # Create config objects
        try:
            config = AppConfig(
                thermodynamics=ThermodynamicConfig(**thermodynamics_dict),
                processing=ProcessingConfig(**processing_dict),
                visualization=VisualizationConfig(**visualization_dict),
                parallel=ParallelConfig(**parallel_dict),
                paths=PathConfig(**paths_dict) if paths_dict else None,
                logging=LoggingConfig(**logging_dict),
                performance=PerformanceConfig(**performance_dict),
                version=config_dict.get('version', '1.0.0'),
                description=config_dict.get('description', 'Pele processing configuration')
            )
        except Exception as e:
            raise ConfigurationError("construction", f"Failed to create config object: {e}")

        # Validate configuration
        issues = validate_complete_config(config)
        raise_on_errors(issues)

        return config

    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
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
            elif hasattr(value, 'value'):  # Enum
                return value.value
            else:
                return value

        return convert_value(config)

    def _check_yaml(self) -> bool:
        """Check if PyYAML is available."""
        try:
            import yaml
            return True
        except ImportError:
            return False

    def _check_toml(self) -> bool:
        """Check if TOML support is available."""
        try:
            if sys.version_info >= (3, 11):
                import tomllib
                return True
            else:
                import toml
                return True
        except ImportError:
            return False


# Convenience functions
def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Load configuration from file."""
    loader = ConfigLoader()
    return loader.load_from_file(config_path)


def create_default_config(input_dir: Union[str, Path],
                          output_dir: Union[str, Path]) -> AppConfig:
    """Create default configuration with specified directories."""
    return AppConfig(
        paths=PathConfig(
            input_directory=Path(input_dir),
            output_directory=Path(output_dir)
        )
    )


def save_config(config: AppConfig, output_path: Union[str, Path],
                format: str = "yaml") -> None:
    """Save configuration to file."""
    loader = ConfigLoader()
    loader.save_to_file(config, output_path, format)