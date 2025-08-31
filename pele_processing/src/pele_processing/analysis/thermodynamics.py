"""
Thermodynamic calculations for the Pele processing system.
"""
from typing import Dict, Optional, Any
import numpy as np

from ..core.interfaces import ThermodynamicCalculator
from ..core.domain import ThermodynamicState
from ..core.exceptions import ThermodynamicError

try:
    import cantera as ct
    CANTERA_AVAILABLE = True
except ImportError:
    ct = None
    CANTERA_AVAILABLE = False


class CanteraThermodynamicCalculator(ThermodynamicCalculator):
    """Cantera-based thermodynamic calculator for filling missing data gaps."""

    def __init__(self, mechanism_file: str):
        if not CANTERA_AVAILABLE:
            raise ThermodynamicError("initialization", {"reason": "Cantera not available"})

        self.mechanism_file = mechanism_file
        self._gas = ct.Solution(mechanism_file)

    def calculate_state_from_TP(self, temperature: float, pressure: float,
                               composition: Dict[str, float]) -> ThermodynamicState:
        """Calculate complete thermodynamic state from T, P, Y."""
        try:
            self._gas.TPY = temperature, pressure, composition

            return ThermodynamicState(
                temperature=temperature,
                pressure=pressure,
                density=self._gas.density,
                sound_speed=self._gas.sound_speed,
                viscosity=self._gas.viscosity,
                conductivity=self._gas.thermal_conductivity,
                cp=self._gas.cp_mass,
                cv=self._gas.cv_mass
            )

        except Exception as e:
            raise ThermodynamicError("state_calculation", {"T": temperature, "P": pressure}) from e

    def calculate_transport_properties(self, state: ThermodynamicState,
                                     composition: Dict[str, float]) -> Dict[str, float]:
        """Calculate transport properties using Cantera."""
        try:
            self._gas.TPY = state.temperature, state.pressure, composition

            return {
                'viscosity': self._gas.viscosity,
                'thermal_conductivity': self._gas.thermal_conductivity,
                'binary_diff_coeffs': dict(zip(self._gas.species_names, self._gas.mix_diff_coeffs)),
                'thermal_diffusivity': self._gas.thermal_conductivity / (self._gas.density * self._gas.cp_mass)
            }

        except Exception as e:
            raise ThermodynamicError("transport_properties", {"T": state.temperature}) from e

    def fill_missing_properties(self, temperature: np.ndarray, pressure: np.ndarray,
                               species_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fill missing field properties using Cantera calculations."""
        n_points = len(temperature)

        calculated = {
            'density': np.zeros(n_points),
            'sound_speed': np.zeros(n_points),
            'viscosity': np.zeros(n_points),
            'thermal_conductivity': np.zeros(n_points),
            'cp': np.zeros(n_points),
            'cv': np.zeros(n_points)
        }

        for i in range(n_points):
            try:
                # Build composition dict for this point
                Y = {}
                for species, mass_fractions in species_data.items():
                    Y[species] = mass_fractions[i]

                self._gas.TPY = temperature[i], pressure[i], Y

                calculated['density'][i] = self._gas.density
                calculated['sound_speed'][i] = self._gas.sound_speed
                calculated['viscosity'][i] = self._gas.viscosity
                calculated['thermal_conductivity'][i] = self._gas.thermal_conductivity
                calculated['cp'][i] = self._gas.cp_mass
                calculated['cv'][i] = self._gas.cv_mass

            except Exception:
                # Use NaN for failed calculations
                for prop in calculated:
                    calculated[prop][i] = np.nan

        return calculated


def create_thermodynamic_calculator(mechanism_file: str, **kwargs) -> ThermodynamicCalculator:
    """Factory for thermodynamic calculators."""
    if not mechanism_file:
        raise ValueError("mechanism_file required for thermodynamic calculations")
    return CanteraThermodynamicCalculator(mechanism_file)