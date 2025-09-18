"""
Data processing pipelines for the Pele processing system.
"""
from typing import Dict, List, Optional, Callable, Any
import numpy as np

from ..core.domain import FieldData, ProcessingResult, DatasetInfo
from ..core.exceptions import DataExtractionError


class DataProcessor:
    """Base data processing pipeline."""

    def __init__(self):
        self.processing_steps = []

    def add_step(self, step: Callable[[FieldData], FieldData]) -> 'DataProcessor':
        """Add processing step to pipeline."""
        self.processing_steps.append(step)
        return self

    def process(self, data: FieldData) -> FieldData:
        """Execute processing pipeline."""
        result = data
        for step in self.processing_steps:
            result = step(result)
        return result


class FilterProcessor:
    """Data filtering operations."""

    @staticmethod
    def smooth_data(data: FieldData, window_size: int = 5) -> FieldData:
        """Apply moving average smoothing."""
        from scipy import signal

        # Smooth each field
        smoothed_temp = signal.savgol_filter(data.temperature, window_size, 3)
        smoothed_pressure = signal.savgol_filter(data.pressure, window_size, 3)
        smoothed_density = signal.savgol_filter(data.density, window_size, 3)
        smoothed_vel = signal.savgol_filter(data.velocity_x, window_size, 3)

        return FieldData(
            coordinates=data.coordinates,
            temperature=smoothed_temp,
            pressure=smoothed_pressure,
            density=smoothed_density,
            velocity_x=smoothed_vel,
            velocity_y=data.velocity_y,
            heat_release_rate=data.heat_release_rate,
            species_data=data.species_data
        )

    @staticmethod
    def remove_outliers(data: FieldData, std_threshold: float = 3.0) -> FieldData:
        """Remove outliers using standard deviation."""

        def filter_outliers(values: np.ndarray) -> np.ndarray:
            mean_val = np.mean(values)
            std_val = np.std(values)
            mask = np.abs(values - mean_val) <= std_threshold * std_val
            return np.where(mask, values, mean_val)

        return FieldData(
            coordinates=data.coordinates,
            temperature=filter_outliers(data.temperature),
            pressure=filter_outliers(data.pressure),
            density=filter_outliers(data.density),
            velocity_x=filter_outliers(data.velocity_x),
            velocity_y=data.velocity_y,
            heat_release_rate=data.heat_release_rate,
            species_data=data.species_data
        )


class DerivativeProcessor:
    """Calculate derivatives and gradients."""

    @staticmethod
    def calculate_gradients(data: FieldData) -> Dict[str, np.ndarray]:
        """Calculate spatial gradients."""
        dx = np.gradient(data.coordinates)

        gradients = {
            'temperature_gradient': np.gradient(data.temperature, dx),
            'pressure_gradient': np.gradient(data.pressure, dx),
            'density_gradient': np.gradient(data.density, dx),
            'velocity_gradient': np.gradient(data.velocity_x, dx)
        }

        return gradients

    @staticmethod
    def calculate_strain_rate(data: FieldData) -> np.ndarray:
        """Calculate strain rate from velocity gradient."""
        dx = np.gradient(data.coordinates)
        return np.gradient(data.velocity_x, dx)


class DomainProcessor:
    """Domain-specific processing operations."""

    @staticmethod
    def extract_subdomain(data: FieldData, x_min: float, x_max: float) -> FieldData:
        """Extract subdomain between x_min and x_max."""
        mask = (data.coordinates >= x_min) & (data.coordinates <= x_max)

        if not np.any(mask):
            raise DataExtractionError("subdomain", None, "No data points in specified range")

        return FieldData(
            coordinates=data.coordinates[mask],
            temperature=data.temperature[mask],
            pressure=data.pressure[mask],
            density=data.density[mask],
            velocity_x=data.velocity_x[mask],
            velocity_y=data.velocity_y[mask] if data.velocity_y is not None else None,
            heat_release_rate=data.heat_release_rate[mask] if data.heat_release_rate is not None else None,
            species_data=data.species_data  # Species data handling would need more complex indexing
        )

    @staticmethod
    def interpolate_to_grid(data: FieldData, new_coordinates: np.ndarray) -> FieldData:
        """Interpolate data to new coordinate grid."""
        from scipy.interpolate import interp1d

        # Create interpolators
        temp_interp = interp1d(data.coordinates, data.temperature, bounds_error=False, fill_value='extrapolate')
        pres_interp = interp1d(data.coordinates, data.pressure, bounds_error=False, fill_value='extrapolate')
        dens_interp = interp1d(data.coordinates, data.density, bounds_error=False, fill_value='extrapolate')
        vel_interp = interp1d(data.coordinates, data.velocity_x, bounds_error=False, fill_value='extrapolate')

        return FieldData(
            coordinates=new_coordinates,
            temperature=temp_interp(new_coordinates),
            pressure=pres_interp(new_coordinates),
            density=dens_interp(new_coordinates),
            velocity_x=vel_interp(new_coordinates),
            velocity_y=data.velocity_y,
            heat_release_rate=data.heat_release_rate,
            species_data=data.species_data
        )


class BatchProcessor:
    """Process multiple datasets in batch."""

    def __init__(self, processor: DataProcessor):
        self.processor = processor

    def process_batch(self, dataset_results: List[ProcessingResult]) -> List[ProcessingResult]:
        """Process batch of results."""
        processed_results = []

        for result in dataset_results:
            if result.success and hasattr(result, 'field_data'):
                try:
                    processed_data = self.processor.process(result.field_data)
                    # Update result with processed data
                    result.field_data = processed_data
                    processed_results.append(result)
                except Exception as e:
                    result.success = False
                    result.error_message = f"Processing failed: {e}"
                    processed_results.append(result)
            else:
                processed_results.append(result)

        return processed_results


def create_standard_processor() -> DataProcessor:
    """Create standard processing pipeline."""
    processor = DataProcessor()
    processor.add_step(FilterProcessor.remove_outliers)
    processor.add_step(lambda data: FilterProcessor.smooth_data(data, window_size=3))
    return processor


def create_analysis_processor() -> DataProcessor:
    """Create processor for detailed analysis."""
    processor = DataProcessor()
    processor.add_step(FilterProcessor.remove_outliers)
    processor.add_step(lambda data: FilterProcessor.smooth_data(data, window_size=5))
    return processor