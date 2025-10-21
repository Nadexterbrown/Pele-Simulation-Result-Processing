"""
Pressure wave analysis for the Pele processing system.
"""
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import numpy as np

from ..core.interfaces import PressureWaveAnalyzer, ThermodynamicCalculator
from ..core.domain import PressureWaveProperties, FieldData, WaveType, ThermodynamicState, FlameProperties
from ..core.exceptions import WaveNotFoundError


class DetectionMethod(Enum):
    """Methods for wave position detection."""
    THRESHOLD = "threshold"  # Find where field exceeds threshold value
    MAX_GRADIENT = "max_gradient"  # Find location of maximum gradient
    MAX_VALUE = "max_value"  # Find location of maximum value


class PelePressureWaveAnalyzer(PressureWaveAnalyzer):
    """Pressure wave analysis implementation for Pele datasets."""

    def __init__(self, detection_method: Union[str, DetectionMethod] = DetectionMethod.MAX_VALUE,
                 threshold_value: Optional[float] = None,
                 thermo_calculator: Optional['ThermodynamicCalculator'] = None):
        """
        Initialize pressure wave analyzer.

        Args:
            detection_method: Method for detecting wave position
            threshold_value: Value for threshold detection method
            field_name: Field to use for detection ('pressure', 'temperature', 'density')
            thermo_calculator: Optional thermodynamic calculator for sound speed
        """
        if isinstance(detection_method, str):
            detection_method = DetectionMethod(detection_method)
        self.detection_method = detection_method
        self.threshold_value = threshold_value
        self.field_name = 'pressure'
        self.thermo_calculator = thermo_calculator

    def analyze_pressure_wave_properties(self, dataset: Any, data: FieldData,
                                        extraction_location: float = None) -> PressureWaveProperties:
        """
        Analyze pressure wave properties.

        Args:
            dataset: Full dataset for 2D analysis
            data: 1D ray data through wave
            extraction_location: Y-location of extraction

        Returns:
            PressureWaveProperties with computed metrics
        """
        # Find pressure wave position
        wave_idx, wave_pos = self.find_wave_position(data, WaveType.PRESSUREWAVE)

        # Extract thermodynamic properties at wave position
        temp = data.temperature[wave_idx]
        pressure = data.pressure[wave_idx]
        density = data.density[wave_idx]

        # Get sound speed from data or calculate it
        if data.sound_speed is not None and wave_idx < len(data.sound_speed):
            sound_speed = data.sound_speed[wave_idx]
        elif self.thermo_calculator is not None and data.species_data is not None:
            # Use thermodynamic calculator to compute sound speed
            composition = {}
            if data.species_data and data.species_data.mass_fractions:
                for species, mass_fracs in data.species_data.mass_fractions.items():
                    if isinstance(mass_fracs, np.ndarray) and wave_idx < len(mass_fracs):
                        composition[species] = mass_fracs[wave_idx]

            # Calculate thermodynamic state with sound speed
            calculated_state = self.thermo_calculator.calculate_state(temp, pressure, composition)
            sound_speed = calculated_state.sound_speed
        else:
            # Fallback to ideal gas approximation
            # c = sqrt(gamma * P/rho)
            gamma = 1.4  # Typical value for diatomic gases
            sound_speed = np.sqrt(gamma * pressure / density)

        # Create thermodynamic state at wave position
        thermo_state = ThermodynamicState(
            temperature=temp,
            pressure=pressure,
            density=density,
            sound_speed=sound_speed
        )

        # Create pressure wave properties
        wave_props = PressureWaveProperties(
            position=wave_pos,
            index=wave_idx,
            thermodynamic_state=thermo_state
        )

        return wave_props

    def find_wave_position(self, data: FieldData, wave_type: WaveType,
                          detection_method: Optional[Union[str, DetectionMethod]] = None,
                          threshold_value: Optional[float] = None,
                          field_name: Optional[str] = None) -> Tuple[int, float]:
        """
        Find wave position using specified detection method.

        Args:
            data: Field data containing arrays
            wave_type: Type of wave to detect
            detection_method: Override default detection method
            threshold_value: Override default threshold value
            field_name: Override default field name

        Returns:
            Tuple of (grid_index, position_in_meters)

        Raises:
            WaveNotFoundError: If wave cannot be detected
        """
        # Use provided parameters or defaults
        method = detection_method if detection_method is not None else self.detection_method
        if isinstance(method, str):
            method = DetectionMethod(method)
        threshold = threshold_value if threshold_value is not None else self.threshold_value
        field = field_name if field_name is not None else self.field_name

        # Get the field data
        if field == 'pressure':
            field_data = data.pressure
        else:
            raise ValueError(f"Invalid or unavailable field: {field}")

        # Apply detection method
        if method == DetectionMethod.THRESHOLD:
            if threshold is None:
                raise ValueError("Threshold value required for threshold detection method")

            # Find indices where field exceeds threshold
            indices = np.where(field_data >= threshold)[0]

            if len(indices) == 0:
                raise WaveNotFoundError(wave_type.value,
                                      f"No points where {field} >= {threshold}")

            # Use downstream-most point above threshold
            wave_idx = indices[-1]

        elif method == DetectionMethod.MAX_GRADIENT:
            # Calculate gradient
            gradient = np.gradient(field_data, data.coordinates)

            # Find maximum gradient location
            wave_idx = np.argmax(np.abs(gradient))

            # Validate that gradient is significant
            max_grad = np.abs(gradient[wave_idx])
            mean_grad = np.mean(np.abs(gradient))
            if max_grad < 3 * mean_grad:  # Require gradient to be 3x mean
                raise WaveNotFoundError(wave_type.value,
                                      f"No significant gradient found in {field}")

        elif method == DetectionMethod.MAX_VALUE:
            # Find maximum value location
            wave_idx = np.argmax(field_data)
        else:
            raise ValueError(f"Unknown detection method: {method}")

        return wave_idx, data.coordinates[wave_idx]


def create_pressure_wave_analyzer(detection_method: str = "max_value",
                                 threshold_value: Optional[float] = None,
                                 field_name: str = "pressure",
                                 thermo_calculator: Optional['ThermodynamicCalculator'] = None) -> PelePressureWaveAnalyzer:
    """
    Factory for pressure wave analyzers.

    Args:
        detection_method: Method for detection ('threshold', 'max_gradient', 'max_value')
        threshold_value: Threshold value for threshold method
        field_name: Field to use for detection
        thermo_calculator: Optional thermodynamic calculator

    Returns:
        Configured PelePressureWaveAnalyzer instance
    """
    return PelePressureWaveAnalyzer(
        detection_method=detection_method,
        threshold_value=threshold_value,
        thermo_calculator=thermo_calculator
    )