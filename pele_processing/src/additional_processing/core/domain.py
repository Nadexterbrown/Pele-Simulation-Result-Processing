"""


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
    velocity: float  # in m/s
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
        return (f"CJ {self.cj_type.value}: v={self.velocity:.1f} m/s, "
                f"T={self.temperature:.1f} K, P={self.pressure/1e5:.2f} bar, "
                f"ρ₂/ρ₁={self.density_ratio:.3f}")