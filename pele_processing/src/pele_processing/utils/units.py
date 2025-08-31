"""
Unit conversion utilities for the Pele processing system.
"""
import re
from typing import Dict, Any, Union
from functools import reduce

from ..core.interfaces import UnitConverter as UnitConverterInterface
from ..core.exceptions import ValidationError


class UnitConverter(UnitConverterInterface):
    """Physical unit converter."""

    # Base unit conversion factors to SI
    UNIT_MAP = {
        # Length
        'cm': 1e-2, 'm': 1.0, 'mm': 1e-3, 'km': 1e3,
        # Mass
        'g': 1e-3, 'kg': 1.0, 'mg': 1e-6,
        # Time
        's': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9,
        # Temperature
        'K': 1.0, 'C': 1.0, 'F': 5 / 9,
        # Pressure
        'Pa': 1.0, 'kPa': 1e3, 'MPa': 1e6, 'bar': 1e5, 'atm': 101325,
        # Energy
        'J': 1.0, 'kJ': 1e3, 'MJ': 1e6, 'cal': 4.184, 'kcal': 4184,
        # Amount
        'mol': 1.0, 'kmol': 1e3,
    }

    # Field to standard unit mapping
    FIELD_UNITS = {
        'X': 'm', 'Y': 'm', 'Z': 'm',
        'Temperature': 'K',
        'Pressure': 'Pa',
        'Density': 'kg/m^3',
        'X Velocity': 'm/s',
        'Y Velocity': 'm/s',
        'Sound speed': 'm/s',
        'Viscosity': 'Pa*s',
        'Conductivity': 'W/(m*K)',
        'Heat Release Rate': 'W/m^3',
        'Cp': 'J/(kg*K)',
        'Cv': 'J/(kg*K)',
    }

    def convert_value(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between units."""
        if from_unit == to_unit:
            return value

        # Handle temperature conversions separately
        if from_unit in ['C', 'F'] or to_unit in ['C', 'F']:
            return self._convert_temperature(value, from_unit, to_unit)

        # Parse compound units
        from_factor = self._parse_unit_expression(from_unit)
        to_factor = self._parse_unit_expression(to_unit)

        # Convert via SI base units
        si_value = value * from_factor
        return si_value / to_factor

    def get_field_units(self, field_name: str) -> str:
        """Get standard units for field."""
        return self.FIELD_UNITS.get(field_name, '')

    def _parse_unit_expression(self, unit_expr: str) -> float:
        """Parse compound unit expression (e.g., 'kg/m^3')."""
        if not unit_expr or unit_expr == '':
            return 1.0

        # Split by '/' for numerator and denominator
        parts = unit_expr.split('/')
        numerator = parts[0].strip()
        denominator = parts[1].strip() if len(parts) > 1 else ''

        # Calculate numerator factor
        num_factor = self._parse_unit_group(numerator, 1)

        # Calculate denominator factor
        denom_factor = self._parse_unit_group(denominator, -1) if denominator else 1

        return num_factor * denom_factor

    def _parse_unit_group(self, unit_group: str, exponent: int) -> float:
        """Parse unit group with powers (e.g., 'm^2*s')."""
        if not unit_group:
            return 1.0

        # Split by '*' for multiple units
        units = [u.strip() for u in unit_group.split('*')]

        total_factor = 1.0
        for unit_str in units:
            if not unit_str:
                continue

            # Parse unit with power (e.g., 'm^2')
            match = re.match(r'([a-zA-Z]+)(?:\^(-?\d+))?', unit_str)
            if not match:
                raise ValidationError("unit_parsing", f"Invalid unit format: {unit_str}")

            base_unit = match.group(1)
            power = int(match.group(2)) if match.group(2) else 1

            if base_unit not in self.UNIT_MAP:
                raise ValidationError("unit_parsing", f"Unknown unit: {base_unit}")

            unit_factor = self.UNIT_MAP[base_unit] ** (power * exponent)
            total_factor *= unit_factor

        return total_factor

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Handle temperature conversions with offset."""
        # Convert to Kelvin first
        if from_unit == 'C':
            kelvin_value = value + 273.15
        elif from_unit == 'F':
            kelvin_value = (value - 32) * 5 / 9 + 273.15
        else:  # Kelvin
            kelvin_value = value

        # Convert from Kelvin to target
        if to_unit == 'C':
            return kelvin_value - 273.15
        elif to_unit == 'F':
            return (kelvin_value - 273.15) * 9 / 5 + 32
        else:  # Kelvin
            return kelvin_value


class PeleUnitConverter(UnitConverter):
    """Pele-specific unit converter with CGS support."""

    def __init__(self):
        super().__init__()
        # Add Pele-specific CGS units
        self.UNIT_MAP.update({
            'dyne': 1e-5,  # Force: 1 dyne = 1e-5 N
            'erg': 1e-7,  # Energy: 1 erg = 1e-7 J
        })

        # Update field units for Pele datasets
        self.FIELD_UNITS.update({
            'density': 'kg/m^3',
            'x_velocity': 'm/s',
            'y_velocity': 'm/s',
            'soundspeed': 'm/s',
            'viscosity': 'Pa*s',
            'conductivity': 'W/(m*K)',
            'heatRelease': 'W/m^3',
            'cp': 'J/(kg*K)',
            'cv': 'J/(kg*K)',
        })

    def convert_from_cgs(self, value: float, field_name: str) -> float:
        """Convert from CGS units to SI."""
        cgs_conversions = {
            'Pressure': 0.1,  # dyne/cm^2 to Pa
            'Density': 1000,  # g/cm^3 to kg/m^3
            'Viscosity': 0.1,  # g/(cm*s) to Pa*s
            'Conductivity': 418.4,  # cal/(s*cm*K) to W/(m*K)
            'Heat Release Rate': 4.184e7,  # erg/(s*cm^3) to W/m^3
            'X Velocity': 0.01,  # cm/s to m/s
            'Y Velocity': 0.01,  # cm/s to m/s
            'Sound speed': 0.01,  # cm/s to m/s
        }

        factor = cgs_conversions.get(field_name, 1.0)
        return value * factor


def create_unit_converter(system: str = "si") -> UnitConverterInterface:
    """Factory function for unit converters."""
    if system == "pele":
        return PeleUnitConverter()
    return UnitConverter()