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

# Thermodynamics
from .thermodynamics import (
    CanteraThermodynamicCalculator, IdealGasCalculator,
    EquilibriumCalculator, create_thermodynamic_calculator
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
    'IdealGasCalculator',
    'EquilibriumCalculator',
    'create_thermodynamic_calculator',

    # Geometry
    'GeometryAnalyzer',
    'FlameGeometryAnalyzer',
    'create_geometry_analyzer'
]

__version__ = "1.0.0"