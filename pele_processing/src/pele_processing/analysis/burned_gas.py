"""
Burned gas analysis for the Pele processing system.
"""
from typing import Dict, Any
import numpy as np

from ..core.interfaces import BurnedGasAnalyzer
from ..core.domain import FieldData, ThermodynamicState, BurnedGasProperties
from ..core.exceptions import BurnedGasAnalysisError


class PeleBurnedGasAnalyzer(BurnedGasAnalyzer):
    """Burned gas analysis implementation for Pele datasets."""

    def __init__(self, offset: float = -10e-6, transport_species: str = 'H2'):
        self.offset = offset  # Offset behind flame for burned gas analysis
        self.transport_species = transport_species

    def analyze_burned_gas_properties(self, data: FieldData, flame_position: float, 
                                     flame_index: int) -> BurnedGasProperties:
        """Analyze burned gas properties behind the flame using 1D field data."""
        try:
            # Find burned gas location behind flame
            burned_gas_position = flame_position + self.offset
            
            # Find index closest to burned gas position
            burned_gas_idx = np.argmin(np.abs(data.coordinates - burned_gas_position))
            
            # Extract properties at burned gas location from 1D data
            properties = BurnedGasProperties(
                position=burned_gas_position,
                index=burned_gas_idx
            )
            
            # Extract thermodynamic state from 1D field data
            if burned_gas_idx < len(data.coordinates):
                temp = data.temperature[burned_gas_idx]
                pressure = data.pressure[burned_gas_idx]
                density = data.density[burned_gas_idx]
                
                # Calculate sound speed from ideal gas relations
                sound_speed = np.sqrt(1.4 * pressure / density)
                
                properties.thermodynamic_state = ThermodynamicState(
                    temperature=temp,
                    pressure=pressure, 
                    density=density,
                    sound_speed=sound_speed
                )
            
            # Extract velocity if available
            if data.velocity_x is not None and burned_gas_idx < len(data.velocity_x):
                properties.velocity = data.velocity_x[burned_gas_idx]
            else:
                properties.velocity = 0.0
            
            return properties
            
        except Exception as e:
            raise BurnedGasAnalysisError("burned_gas_properties", str(e))


def create_burned_gas_analyzer(offset: float = -10e-6, **kwargs) -> BurnedGasAnalyzer:
    """Factory for burned gas analyzers."""
    return PeleBurnedGasAnalyzer(offset, **kwargs)