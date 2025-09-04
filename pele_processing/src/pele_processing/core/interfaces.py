"""
Abstract interfaces for the Pele processing system.

Defines contracts that concrete implementations must follow,
enabling dependency injection and testability.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path

from .domain import (
    DatasetInfo, FieldData, FlameProperties, ShockProperties,
    ProcessingResult, ProcessingBatch, ThermodynamicState,
    AnimationFrame, VisualizationRequest, Point2D, WaveType, Direction
)


# =============================================================================
# Data Layer Interfaces
# =============================================================================

class DataLoader(ABC):
    """Interface for loading datasets from storage."""

    @abstractmethod
    def load_dataset(self, path: Union[str, Path]) -> Any:
        """Load a dataset from the given path.

        Args:
            path: Path to dataset file or directory

        Returns:
            Loaded dataset object (implementation specific)

        Raises:
            FileNotFoundError: If dataset doesn't exist
            DataLoadError: If dataset is corrupted or invalid
        """
        pass

    @abstractmethod
    def get_dataset_info(self, dataset: Any) -> DatasetInfo:
        """Extract metadata from loaded dataset.

        Args:
            dataset: Loaded dataset object

        Returns:
            DatasetInfo containing metadata
        """
        pass

    @abstractmethod
    def list_available_fields(self, dataset: Any) -> List[str]:
        """List all available fields in the dataset.

        Args:
            dataset: Loaded dataset object

        Returns:
            List of field names
        """
        pass


class DataExtractor(ABC):
    """Interface for extracting data from loaded datasets."""

    @abstractmethod
    def extract_ray_data(self, dataset: Any, location: float,
                         direction: Direction = Direction.X) -> FieldData:
        """Extract 1D ray data from dataset.

        Args:
            dataset: Loaded dataset object
            location: Position along perpendicular axis (in meters)
            direction: Direction of ray extraction

        Returns:
            FieldData containing extracted 1D arrays

        Raises:
            ExtractionError: If extraction fails
        """
        pass

    @abstractmethod
    def extract_thermodynamic_state(self, dataset: Any, location: Point2D) -> ThermodynamicState:
        """Extract thermodynamic state at specific location.

        Args:
            dataset: Loaded dataset object
            location: 2D point for extraction

        Returns:
            ThermodynamicState at the location
        """
        pass


class DataValidator(ABC):
    """Interface for validating data quality."""

    @abstractmethod
    def validate_field_data(self, data: FieldData) -> bool:
        """Validate field data for completeness and physical consistency.

        Args:
            data: FieldData to validate

        Returns:
            True if data is valid

        Raises:
            ValidationError: If data is invalid
        """
        pass

    @abstractmethod
    def check_grid_convergence(self, dataset: Any) -> bool:
        """Check if grid is adequately resolved.

        Args:
            dataset: Loaded dataset object

        Returns:
            True if grid is converged
        """
        pass


# =============================================================================
# Analysis Layer Interfaces
# =============================================================================

class WaveTracker(ABC):
    """Interface for tracking wave fronts."""

    @abstractmethod
    def find_wave_position(self, data: FieldData, wave_type: WaveType) -> Tuple[int, float]:
        """Find position of wave front.

        Args:
            data: 1D field data
            wave_type: Type of wave to track

        Returns:
            Tuple of (grid_index, position_in_meters)

        Raises:
            WaveNotFoundError: If wave cannot be detected
        """
        pass

    @abstractmethod
    def calculate_wave_velocity(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate wave velocity from position time series.

        Args:
            positions: Wave positions over time (m)
            times: Time stamps (s)

        Returns:
            Wave velocities (m/s)
        """
        pass


class FlameAnalyzer(ABC):
    """Interface for flame-specific analysis."""

    @abstractmethod
    def analyze_flame_properties(self, dataset: Any, data: FieldData) -> FlameProperties:
        """Perform comprehensive flame analysis.

        Args:
            dataset: Full dataset for 2D analysis
            data: 1D ray data through flame

        Returns:
            FlameProperties with all computed metrics
        """
        pass

    @abstractmethod
    def calculate_flame_thickness(self, dataset: Any, contour_points: np.ndarray,
                                  center_location: float) -> float:
        """Calculate flame thickness using temperature gradient method.

        Args:
            dataset: Full dataset for 2D temperature field
            contour_points: Flame contour coordinates
            center_location: Y-coordinate of analysis location

        Returns:
            Flame thickness in meters
        """
        pass

    @abstractmethod
    def calculate_consumption_rate(self, dataset: Any, contour_points: np.ndarray,
                                   transport_species: str) -> Tuple[float, float]:
        """Calculate species consumption rate and burning velocity.

        Args:
            dataset: Full dataset
            contour_points: Flame contour coordinates
            transport_species: Species name for analysis

        Returns:
            Tuple of (consumption_rate_kg/s, burning_velocity_m/s)
        """
        pass


class ShockAnalyzer(ABC):
    """Interface for shock wave analysis."""

    @abstractmethod
    def analyze_shock_properties(self, data: FieldData) -> ShockProperties:
        """Analyze shock wave properties.

        Args:
            data: 1D field data containing shock

        Returns:
            ShockProperties with computed metrics
        """
        pass

    @abstractmethod
    def calculate_rankine_hugoniot_relations(self, pre_state: ThermodynamicState,
                                             post_state: ThermodynamicState) -> Dict[str, float]:
        """Verify shock using Rankine-Hugoniot relations.

        Args:
            pre_state: Pre-shock thermodynamic state
            post_state: Post-shock thermodynamic state

        Returns:
            Dictionary with shock validation metrics
        """
        pass


class ThermodynamicCalculator(ABC):
    """Interface for thermodynamic calculations."""

    @abstractmethod
    def calculate_state_from_TP(self, temperature: float, pressure: float,
                                composition: Dict[str, float]) -> ThermodynamicState:
        """Calculate thermodynamic state from T, P, and composition.

        Args:
            temperature: Temperature in K
            pressure: Pressure in Pa
            composition: Species mass fractions

        Returns:
            Complete thermodynamic state
        """
        pass

    @abstractmethod
    def calculate_transport_properties(self, state: ThermodynamicState,
                                       composition: Dict[str, float]) -> Dict[str, float]:
        """Calculate transport properties.

        Args:
            state: Thermodynamic state
            composition: Species mass fractions

        Returns:
            Dictionary with viscosity, conductivity, diffusion coeffs
        """
        pass


# =============================================================================
# Visualization Layer Interfaces
# =============================================================================

class FrameGenerator(ABC):
    """Interface for generating animation frames."""

    @abstractmethod
    def create_field_plot(self, frame: AnimationFrame) -> None:
        """Create a single field visualization frame.

        Args:
            frame: Animation frame specification
        """
        pass

    @abstractmethod
    def create_multi_field_plot(self, x_data: np.ndarray, field_data: Dict[str, np.ndarray],
                                output_path: Path, **kwargs) -> None:
        """Create plot with multiple fields.

        Args:
            x_data: X-axis coordinates
            field_data: Dictionary of field_name -> y_data
            output_path: Where to save the plot
            **kwargs: Additional plotting options
        """
        pass

    @abstractmethod
    def create_specialized_plot(self, dataset: Any, plot_type: str,
                                output_path: Path, **kwargs) -> None:
        """Create specialized plots (Schlieren, streamlines, etc.).

        Args:
            dataset: Full dataset for 2D plots
            plot_type: Type of specialized plot
            output_path: Where to save the plot
            **kwargs: Plot-specific options
        """
        pass


class AnimationBuilder(ABC):
    """Interface for building animations from frames."""

    @abstractmethod
    def create_animation(self, frame_directory: Path, output_path: Path,
                         frame_rate: float = 5.0, format: str = 'gif') -> None:
        """Create animation from frame directory.

        Args:
            frame_directory: Directory containing frame images
            output_path: Output animation file path
            frame_rate: Animation frame rate (fps)
            format: Output format ('gif', 'mp4', etc.)
        """
        pass


class OutputFormatter(ABC):
    """Interface for formatting output data."""

    @abstractmethod
    def format_results_table(self, batch: ProcessingBatch, output_path: Path) -> None:
        """Format processing results as structured table.

        Args:
            batch: Batch of processing results
            output_path: Where to save formatted output
        """
        pass

    @abstractmethod
    def format_summary_report(self, batch: ProcessingBatch, output_path: Path) -> None:
        """Generate summary report from processing batch.

        Args:
            batch: Batch of processing results
            output_path: Where to save report
        """
        pass


# =============================================================================
# Parallel Processing Interfaces
# =============================================================================

class WorkDistributor(ABC):
    """Interface for distributing work across processes."""

    @abstractmethod
    def distribute_work(self, work_items: List[Any],
                        worker_function: callable) -> List[Any]:
        """Distribute work items across available processes.

        Args:
            work_items: List of work items to process
            worker_function: Function to execute on each item

        Returns:
            List of results from processing
        """
        pass

    @abstractmethod
    def get_process_info(self) -> Dict[str, int]:
        """Get information about parallel environment.

        Returns:
            Dictionary with 'rank', 'size', and other process info
        """
        pass


class ParallelCoordinator(ABC):
    """Interface for coordinating parallel execution."""

    @abstractmethod
    def coordinate_processing(self, dataset_paths: List[str],
                              processor_function: callable) -> ProcessingBatch:
        """Coordinate parallel processing of multiple datasets.

        Args:
            dataset_paths: List of paths to process
            processor_function: Function to process single dataset

        Returns:
            ProcessingBatch with all results
        """
        pass

    @abstractmethod
    def synchronize_processes(self) -> None:
        """Synchronize all processes at barrier."""
        pass

    @abstractmethod
    def gather_results(self, local_results: List[Any]) -> List[Any]:
        """Gather results from all processes to root.

        Args:
            local_results: Results from current process

        Returns:
            All results (only valid on root process)
        """
        pass


# =============================================================================
# Configuration and Utilities
# =============================================================================

class ConfigurationLoader(ABC):
    """Interface for loading and validating configuration."""

    @abstractmethod
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Parsed configuration dictionary
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass


class Logger(ABC):
    """Interface for logging operations."""

    @abstractmethod
    def log_info(self, message: str, **kwargs) -> None:
        """Log informational message."""
        pass

    @abstractmethod
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        pass

    @abstractmethod
    def log_error(self, message: str, **kwargs) -> None:
        """Log error message."""
        pass

    @abstractmethod
    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        pass


class UnitConverter(ABC):
    """Interface for unit conversions."""

    @abstractmethod
    def convert_value(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        pass

    @abstractmethod
    def get_field_units(self, field_name: str) -> str:
        """Get standard units for a field.

        Args:
            field_name: Name of physical field

        Returns:
            Standard unit string
        """
        pass