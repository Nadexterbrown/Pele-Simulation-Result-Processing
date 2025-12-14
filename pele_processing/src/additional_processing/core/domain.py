"""
Domain classes for additional processing functionality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np
from pathlib import Path
from enum import Enum

class CJType(Enum):
    DETONATION = "detonation"
    DEFLAGRATION = "deflagration"

@dataclass
class CJProperties:
    """Data structure to hold Chapman-Jouguet properties."""
    cj_type: CJType
    pressure: float  # in Pa
    temperature: float  # in K
    density: float  # in kg/m^3
    reactant_velocity: float  # wave speed relative to reactants in m/s
    product_velocity: float  # wave speed relative to products in m/s
    specific_energy: float  # in J/kg
    specific_volume: float  # in m^3/kg
    sound_speed: float  # in m/s
    gamma: float  # dimensionless
    mach_number: float  # dimensionless
    product_mach_number: float  # dimensionless
    density_ratio: float  # rho_b/rho_u ratio
    enthalpy: float  # in J/kg
    r_squared: float  # quality of fit
    converged: bool  # convergence flag
    species_concentrations: Dict[str, float] = field(default_factory=dict)  # mole fractions

    def __str__(self) -> str:
        """String representation of CJ state."""
        return (f"CJ {self.cj_type.value}: v_reactant={self.reactant_velocity:.1f} m/s, "
                f"v_product={self.product_velocity:.1f} m/s, "
                f"T={self.temperature:.1f} K, P={self.pressure/1e5:.2f} bar, "
                f"ρ₂/ρ₁={self.density_ratio:.3f}")


@dataclass
class CanteraFreeFlameProperties:
    """Properties of a Cantera free flame solution."""

    initial_temperature: float  # Unburned gas temperature (K)
    initial_pressure: float  # Pressure (Pa)
    initial_density: float  # Unburned gas density (kg/m³)
    max_temperature: float  # Maximum temperature (K)
    product_temperature: float  # Burned gas temperature (K)
    product_density: float  # Burned gas density (kg/m³)
    flame_speed: float  # Laminar flame speed (m/s)
    flame_thickness: float  # Thermal thickness (m)
    max_heat_release_rate: float  # Maximum HRR (W/m³)
    density_ratio: Optional[float] = None  # rho_u/rho_b ratio

    # Raw profile data (optional)
    profiles: Optional[Dict[str, np.ndarray]] = None  # x, T, u, Y_i, etc.

    def __str__(self) -> str:
        """String representation of flame properties."""
        return (f"Cantera Flame: S_L={self.flame_speed*100:.1f} cm/s, "
                f"δ_th={self.flame_thickness*1000:.3f} mm, "
                f"T_max={self.max_temperature:.0f} K, "
                f"P={self.initial_pressure/1e5:.1f} bar")
