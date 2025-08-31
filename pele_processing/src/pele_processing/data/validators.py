"""
Data validation for the Pele processing system.
"""
from typing import List, Dict, Any, Optional
import numpy as np

from ..core.interfaces import DataValidator
from ..core.domain import FieldData
from ..core.exceptions import ValidationError
from ..utils.constants import STANDARD_TEMPERATURE, STANDARD_PRESSURE


class PeleDataValidator(DataValidator):
    """Validates Pele dataset quality and physical consistency."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_results = {}

    def validate_field_data(self, data: FieldData) -> bool:
        """Validate field data completeness and physical bounds."""
        issues = []

        # Check array consistency
        base_length = len(data.coordinates)
        for field_name, field_data in [
            ('temperature', data.temperature),
            ('pressure', data.pressure),
            ('density', data.density),
            ('velocity_x', data.velocity_x)
        ]:
            if len(field_data) != base_length:
                issues.append(f"Field {field_name} length mismatch")

        # Physical bounds checks
        if np.any(data.temperature <= 0):
            issues.append("Temperature must be positive")
        if np.any(data.pressure <= 0):
            issues.append("Pressure must be positive")
        if np.any(data.density <= 0):
            issues.append("Density must be positive")

        # Reasonable value ranges
        if np.any(data.temperature > 5000):
            issues.append("Temperature exceeds 5000K")
        if np.any(data.pressure > 1e8):
            issues.append("Pressure exceeds 100 MPa")
        if np.any(np.abs(data.velocity_x) > 10000):
            issues.append("Velocity exceeds 10 km/s")

        # NaN/Inf checks
        for field_name, field_data in [
            ('temperature', data.temperature),
            ('pressure', data.pressure),
            ('density', data.density),
            ('velocity_x', data.velocity_x)
        ]:
            if np.any(~np.isfinite(field_data)):
                issues.append(f"Invalid values in {field_name}")

        if issues and self.strict_mode:
            raise ValidationError("field_data", "; ".join(issues))

        self.validation_results['field_data'] = issues
        return len(issues) == 0

    def check_grid_convergence(self, dataset: Any) -> bool:
        """Check grid resolution adequacy."""
        try:
            min_dx = dataset.index.get_smallest_dx().to_value()
            max_level = dataset.index.max_level

            issues = []

            # Check minimum resolution
            if min_dx > 1e-5:  # 10 microns
                issues.append("Grid spacing too coarse for flame resolution")

            # Check refinement levels
            if max_level < 2:
                issues.append("Insufficient AMR levels for complex flows")

            self.validation_results['grid_convergence'] = issues
            return len(issues) == 0

        except Exception as e:
            if self.strict_mode:
                raise ValidationError("grid_convergence", str(e))
            return False


class PhysicalConsistencyValidator:
    """Validates thermodynamic and fluid mechanical consistency."""

    @staticmethod
    def check_equation_of_state(temperature: np.ndarray, pressure: np.ndarray,
                                density: np.ndarray, gas_constant: float = 287.0) -> List[str]:
        """Check ideal gas law consistency."""
        issues = []

        # Calculate density from ideal gas law
        calculated_density = pressure / (gas_constant * temperature)
        relative_error = np.abs(calculated_density - density) / density

        if np.any(relative_error > 0.1):  # 10% tolerance
            issues.append("Density inconsistent with ideal gas law")

        return issues

    @staticmethod
    def check_conservation_laws(data: FieldData) -> List[str]:
        """Check basic conservation principles."""
        issues = []

        # Mass conservation check (simplified)
        mass_flux = data.density * data.velocity_x
        mass_flux_variation = np.std(mass_flux) / np.mean(mass_flux)

        if mass_flux_variation > 0.2:  # 20% variation
            issues.append("Mass conservation violation detected")

        return issues


class DataQualityMetrics:
    """Calculate data quality metrics."""

    @staticmethod
    def calculate_smoothness(data: np.ndarray) -> float:
        """Calculate data smoothness metric."""
        if len(data) < 3:
            return 0.0

        second_derivative = np.gradient(np.gradient(data))
        smoothness = 1.0 / (1.0 + np.std(second_derivative))
        return smoothness

    @staticmethod
    def calculate_signal_noise_ratio(data: np.ndarray, window_size: int = 5) -> float:
        """Calculate signal-to-noise ratio."""
        from scipy import signal

        # Smooth signal
        smoothed = signal.savgol_filter(data, window_size, 3)
        noise = data - smoothed

        signal_power = np.var(smoothed)
        noise_power = np.var(noise)

        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    @staticmethod
    def generate_quality_report(data: FieldData) -> Dict[str, float]:
        """Generate comprehensive quality metrics."""
        return {
            'temperature_smoothness': DataQualityMetrics.calculate_smoothness(data.temperature),
            'pressure_smoothness': DataQualityMetrics.calculate_smoothness(data.pressure),
            'temperature_snr': DataQualityMetrics.calculate_signal_noise_ratio(data.temperature),
            'pressure_snr': DataQualityMetrics.calculate_signal_noise_ratio(data.pressure),
            'coordinate_spacing_uniformity': 1.0 - np.std(np.diff(data.coordinates)) / np.mean(
                np.diff(data.coordinates))
        }


def validate_dataset_batch(datasets: List[Any], validator: DataValidator) -> Dict[str, bool]:
    """Validate batch of datasets."""
    results = {}

    for dataset in datasets:
        try:
            is_converged = validator.check_grid_convergence(dataset)
            results[dataset.basename] = is_converged
        except Exception:
            results[dataset.basename] = False

    return results