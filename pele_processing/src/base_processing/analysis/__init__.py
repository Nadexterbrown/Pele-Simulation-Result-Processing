"""
Analysis module for the Pele processing system.
"""

# Flame analysis
from .flame import (
    PeleFlameAnalyzer, create_flame_analyzer
)

# Shock analysis
from .shock import (
    PeleShockAnalyzer, create_shock_analyzer
)

# Pressure wave analysis
from .pressure_wave import (
    PelePressureWaveAnalyzer, create_pressure_wave_analyzer
)

# Thermodynamics
from .thermodynamics import (
    CanteraThermodynamicCalculator, create_thermodynamic_calculator
)

# Burned gas analysis
from .burned_gas import (
    PeleBurnedGasAnalyzer, create_burned_gas_analyzer
)

# Geometry analysis
from .geometry import (
    GeometryAnalyzer, FlameGeometryAnalyzer, create_geometry_analyzer
)

__all__ = [
    # Flame
    'PeleFlameAnalyzer',
    'create_flame_analyzer',

    # Shock
    'PeleShockAnalyzer',
    'create_shock_analyzer',

    # Thermodynamics
    'CanteraThermodynamicCalculator',
    'create_thermodynamic_calculator',

    # Burned gas
    'PeleBurnedGasAnalyzer',
    'create_burned_gas_analyzer',

    # Geometry
    'GeometryAnalyzer',
    'FlameGeometryAnalyzer',
    'create_geometry_analyzer'
]

__version__ = "1.0.0"