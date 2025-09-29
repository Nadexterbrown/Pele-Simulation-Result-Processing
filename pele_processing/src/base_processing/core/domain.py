"""
Domain models and data structures for Pele processing.

This module defines the core domain entities that represent the business
objects in the Pele processing system.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np
from pathlib import Path
from enum import Enum


class WaveType(Enum):
    """Types of waves that can be tracked."""
    FLAME = "flame"
    SHOCK = "shock"
    PRESSUREWAVE = "pressure wave"


class Direction(Enum):
    """Spatial directions for data extraction."""
    X = "x"
    Y = "y"
    Z = "z"


@dataclass(frozen=True)
class Point2D:
    """2D spatial point."""
    x: float
    y: float

    def distance_to(self, other: 'Point2D') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass(frozen=True)
class Point3D:
    """3D spatial point."""
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BoundingBox:
    """Rectangular bounding box."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def contains_point(self, point: Point2D) -> bool:
        return (self.min_x <= point.x <= self.max_x and
                self.min_y <= point.y <= self.max_y)


@dataclass(frozen=True)
class ThermodynamicState:
    """Thermodynamic state at a point."""
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m³
    sound_speed: float  # m/s
    viscosity: Optional[float] = None  # kg/(m·s)
    conductivity: Optional[float] = None  # W/(m·K)
    cp: Optional[float] = None  # J/(kg·K)
    cv: Optional[float] = None  # J/(kg·K)

    @property
    def mach_number(self) -> Optional[float]:
        """Calculate Mach number if velocity is provided."""
        return None  # Requires velocity context


@dataclass
class FlameProperties:
    """Properties of a flame front."""
    position: float  # m
    index: Optional[int] = None  # Grid index
    velocity: Optional[float] = None  # m/s
    relative_velocity: Optional[float] = None  # m/s
    gas_velocity: Optional[float] = None  # m/s - gas velocity at flame position
    thickness: Optional[float] = None  # m
    surface_length: Optional[float] = None  # m
    heat_release_rate: Optional[float] = None  # W/m³
    consumption_rate: Optional[float] = None  # kg/s
    burning_velocity: Optional[float] = None  # m/s
    reynolds_number: Optional[float] = None
    thermodynamic_state: Optional[ThermodynamicState] = None
    contour_points: Optional[np.ndarray] = None  # 2D contour coordinates
    skirt_pos: Optional[float] = None  # m - position of flame skirt
    # Additional data for thickness plotting
    region_grid: Optional[np.ndarray] = None  # Local grid around flame
    region_temperature: Optional[np.ndarray] = None  # Temperature field on grid
    normal_line: Optional[np.ndarray] = None  # Normal line coordinates
    interpolated_temperatures: Optional[np.ndarray] = None  # Temperatures along normal line
    # Contour fitting results
    contour_fits: Optional[Dict[str, Any]] = None  # All fitting results
    best_fit_type: Optional[str] = None  # Name of best fitting method
    best_fit_r_squared: Optional[float] = None  # R^2 of best fit
    best_fit_length: Optional[float] = None  # Arc length of best fit curve
    contour_length: Optional[float] = None  # Arc length of actual contour
    length_ratio: Optional[float] = None  # Ratio of contour to fitted length
    # Single fit results (from analyze_flame_properties)
    fitted_points: Optional[np.ndarray] = None  # Fitted curve points
    fit_parameters: Optional[Dict[str, Any]] = None  # Fitting parameters
    fit_quality: Optional[Dict[str, float]] = None  # R^2, RMSE, fit_type

    def is_valid(self) -> bool:
        """Check if flame has minimum required data."""
        return self.position is not None and not np.isnan(self.position)


@dataclass
class ShockProperties:
    """Properties of a shock front."""
    position: float  # m
    index: Optional[int] = None  # Grid index
    velocity: Optional[float] = None  # m/s
    pre_shock_state: Optional[ThermodynamicState] = None
    post_shock_state: Optional[ThermodynamicState] = None

    def is_valid(self) -> bool:
        """Check if shock has minimum required data."""
        return self.position is not None and not np.isnan(self.position)


@dataclass
class PressureWaveProperties:
    """Properties of a flame front."""
    position: float  # m
    index: Optional[int] = None  # Grid index
    thermodynamic_state: Optional[ThermodynamicState] = None

    def is_valid(self) -> bool:
        """Check if flame has minimum required data."""
        return self.position is not None and not np.isnan(self.position)


@dataclass
class GasProperties:
    """Properties of gas in a region."""
    velocity: Optional[float] = None  # m/s
    thermodynamic_state: Optional[ThermodynamicState] = None

    def is_valid(self) -> bool:
        """Check if gas properties are valid."""
        return (self.velocity is not None and
                not np.isnan(self.velocity))


@dataclass
class BurnedGasProperties:
    """Burned gas properties behind the flame."""
    position: Optional[float] = None  # m
    index: Optional[int] = None
    velocity: Optional[float] = None  # m/s
    thermodynamic_state: Optional[ThermodynamicState] = None

    def is_valid(self) -> bool:
        """Check if burned gas has minimum required data."""
        return self.position is not None and not np.isnan(self.position)


@dataclass
class SpeciesData:
    """Chemical species data."""
    mass_fractions: Dict[str, float] = field(default_factory=dict)  # Y_i
    mole_fractions: Dict[str, float] = field(default_factory=dict)  # X_i
    diffusion_coeffs: Dict[str, float] = field(default_factory=dict)  # D_i
    production_rates: Dict[str, float] = field(default_factory=dict)  # ω_i

    def get_mass_fraction(self, species: str) -> float:
        """Get mass fraction for species."""
        return self.mass_fractions.get(species, 0.0)


@dataclass
class FieldData:
    """1D field data extracted from dataset."""
    coordinates: np.ndarray  # Spatial coordinates
    temperature: np.ndarray  # K
    pressure: np.ndarray  # Pa
    density: np.ndarray  # kg/m³
    velocity_x: np.ndarray  # m/s
    velocity_y: Optional[np.ndarray] = None  # m/s
    sound_speed: Optional[np.ndarray] = None  # m/s
    heat_release_rate: Optional[np.ndarray] = None  # W/m³
    species_data: Optional[SpeciesData] = None

    def __post_init__(self):
        """Validate array lengths match."""
        base_length = len(self.coordinates)
        arrays = [self.temperature, self.pressure, self.density, self.velocity_x]

        for arr in arrays:
            if len(arr) != base_length:
                raise ValueError("All field arrays must have same length")


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    path: Path
    basename: str
    timestamp: float  # Physical time
    domain_bounds: BoundingBox
    max_refinement_level: int
    grid_spacing: float  # Finest grid spacing

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> 'DatasetInfo':
        """Create DatasetInfo from path (to be implemented by data loaders)."""
        path = Path(path)
        return cls(
            path=path,
            basename=path.name,
            timestamp=0.0,
            domain_bounds=BoundingBox(0, 1, 0, 1),
            max_refinement_level=0,
            grid_spacing=1e-6
        )


@dataclass
class ProcessingResult:
    """Result from processing a single dataset."""
    dataset_info: DatasetInfo
    flame_data: Optional[FlameProperties] = None
    shock_data: Optional[ShockProperties] = None
    burned_gas_data: Optional[GasProperties] = None
    unburned_gas_data: Optional[GasProperties] = None

    # Processing metadata
    processing_time: float = 0.0  # seconds
    success: bool = True
    error_message: Optional[str] = None

    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return self.success and self.error_message is None

    def has_flame_data(self) -> bool:
        """Check if flame analysis was successful."""
        return (self.flame_data is not None and
                self.flame_data.is_valid())

    def has_shock_data(self) -> bool:
        """Check if shock analysis was successful."""
        return (self.shock_data is not None and
                self.shock_data.is_valid())


@dataclass
class ProcessingBatch:
    """Collection of processing results."""
    results: List[ProcessingResult] = field(default_factory=list)

    def add_result(self, result: ProcessingResult) -> None:
        """Add a result to the batch."""
        self.results.append(result)

    def get_successful_results(self) -> List[ProcessingResult]:
        """Get only successful processing results."""
        return [r for r in self.results if r.is_successful()]

    def get_timestamps(self) -> np.ndarray:
        """Extract timestamps from all results."""
        return np.array([r.dataset_info.timestamp for r in self.results])

    def get_flame_positions(self) -> np.ndarray:
        """Extract flame positions where available."""
        positions = []
        for result in self.results:
            if result.has_flame_data():
                positions.append(result.flame_data.position)
            else:
                positions.append(np.nan)
        return np.array(positions)

    def get_shock_positions(self) -> np.ndarray:
        """Extract shock positions where available."""
        positions = []
        for result in self.results:
            if result.has_shock_data():
                positions.append(result.shock_data.position)
            else:
                positions.append(np.nan)
        return np.array(positions)

    def calculate_wave_velocities(self) -> Dict[str, np.ndarray]:
        """Calculate wave velocities from position time series."""
        times = self.get_timestamps()
        velocities = {}

        # Flame velocity
        flame_positions = self.get_flame_positions()
        valid_flame = ~np.isnan(flame_positions)
        if np.sum(valid_flame) > 1:
            velocities['flame'] = np.gradient(flame_positions[valid_flame],
                                              times[valid_flame])

        # Shock velocity
        shock_positions = self.get_shock_positions()
        valid_shock = ~np.isnan(shock_positions)
        if np.sum(valid_shock) > 1:
            velocities['shock'] = np.gradient(shock_positions[valid_shock],
                                              times[valid_shock])

        return velocities


@dataclass
class AnimationFrame:
    """Single frame for animation generation."""
    dataset_basename: str
    field_name: str
    x_data: np.ndarray
    y_data: np.ndarray
    output_path: Path

    # Optional metadata
    timestamp: Optional[float] = None
    flame_position: Optional[float] = None
    shock_position: Optional[float] = None


@dataclass
class VisualizationRequest:
    """Request for generating visualizations."""
    output_directory: Path
    field_names: List[str]
    animation_format: str = 'gif'
    frame_rate: float = 5.0
    local_window_size: Optional[float] = None
    generate_schlieren: bool = False
    generate_streamlines: bool = False