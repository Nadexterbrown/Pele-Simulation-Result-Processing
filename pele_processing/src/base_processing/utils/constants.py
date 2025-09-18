"""
Physical and system constants for the Pele processing system.
"""

# Physical constants (SI units)
UNIVERSAL_GAS_CONSTANT = 8.314462618  # J/(mol·K)
AVOGADRO_NUMBER = 6.02214076e23       # 1/mol
BOLTZMANN_CONSTANT = 1.380649e-23     # J/K
STEFAN_BOLTZMANN = 5.670374419e-8     # W/(m²·K⁴)
PLANCK_CONSTANT = 6.62607015e-34      # J·s

# Standard conditions
STANDARD_TEMPERATURE = 273.15         # K (0°C)
STANDARD_PRESSURE = 101325            # Pa (1 atm)
STANDARD_DENSITY_AIR = 1.225          # kg/m³ at STP

# Combustion-related constants
STOICHIOMETRIC_AIR_FUEL_H2 = 34.3    # kg air / kg H2
STOICHIOMETRIC_AIR_FUEL_CH4 = 17.2   # kg air / kg CH4
STOICHIOMETRIC_AIR_FUEL_C2H6 = 16.1  # kg air / kg C2H6

HEATING_VALUE_H2 = 120e6              # J/kg (lower heating value)
HEATING_VALUE_CH4 = 50e6              # J/kg (lower heating value)
HEATING_VALUE_C2H6 = 47.8e6           # J/kg (lower heating value)

# Default analysis parameters
DEFAULT_FLAME_TEMPERATURE = 2500.0    # K
DEFAULT_SHOCK_PRESSURE_RATIO = 1.01
DEFAULT_EXTRACTION_LOCATION = None    # Auto-detect

# Grid resolution requirements
MIN_FLAME_THICKNESS_CELLS = 5
MIN_SHOCK_WIDTH_CELLS = 3
RECOMMENDED_PPW = 20  # Points per wavelength

# Numerical tolerances
CONVERGENCE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-8
MAX_ITERATIONS = 1000

# File size limits (bytes)
MAX_DATASET_SIZE = 50 * 1024**3       # 50 GB
MAX_LOG_FILE_SIZE = 100 * 1024**2     # 100 MB
MAX_ANIMATION_FRAMES = 10000

# Performance tuning
DEFAULT_CHUNK_SIZE = 1
DEFAULT_THREAD_COUNT = 4
MPI_TIMEOUT_DEFAULT = 300.0           # seconds

# Visualization defaults
DEFAULT_DPI = 150
DEFAULT_FIGURE_SIZE = (10, 6)         # inches
DEFAULT_FRAME_RATE = 5.0              # fps
DEFAULT_ANIMATION_FORMAT = 'gif'

# Species mapping for common mechanisms
COMMON_SPECIES = {
    'H2': ['H2'],
    'O2': ['O2'],
    'N2': ['N2'],
    'H2O': ['H2O'],
    'CO2': ['CO2'],
    'CO': ['CO'],
    'OH': ['OH'],
    'H': ['H'],
    'O': ['O'],
    'HO2': ['HO2'],
    'H2O2': ['H2O2']
}

# Error codes
ERROR_CODES = {
    'DATA_LOAD_FAILED': 1001,
    'EXTRACTION_FAILED': 1002,
    'ANALYSIS_FAILED': 1003,
    'VISUALIZATION_FAILED': 1004,
    'MPI_ERROR': 2001,
    'CONFIG_ERROR': 3001,
    'FILE_NOT_FOUND': 4001,
    'PERMISSION_DENIED': 4002,
    'DISK_FULL': 4003
}

# Version information
__version__ = "1.0.0"
__author__ = "Pele Processing Team"
__license__ = "MIT"