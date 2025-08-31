"""
Data extraction implementations for the Pele processing system.
"""
from typing import Dict, List, Optional, Any
import numpy as np

from ..core.interfaces import DataExtractor, UnitConverter
from ..core.domain import FieldData, ThermodynamicState, Point2D, Direction, SpeciesData
from ..core.exceptions import DataExtractionError, FieldNotFoundError
from ..utils.constants import COMMON_SPECIES

try:
    import cantera as ct

    CANTERA_AVAILABLE = True
except ImportError:
    ct = None
    CANTERA_AVAILABLE = False


class PeleDataExtractor(DataExtractor):
    """Pele-specific data extractor with unit conversion."""

    def __init__(self, unit_converter: UnitConverter, mechanism_file: Optional[str] = None):
        self.unit_converter = unit_converter
        self.mechanism_file = mechanism_file
        self._gas = None
        self._field_map = self._create_field_map()

    @property
    def gas(self):
        if self._gas is None and self.mechanism_file and CANTERA_AVAILABLE:
            self._gas = ct.Solution(self.mechanism_file)
        return self._gas

    def extract_ray_data(self, dataset: Any, location: float, direction: Direction = Direction.X) -> FieldData:
        """Extract 1D ray data."""
        try:
            # Create ray
            if direction == Direction.Y:
                ray = dataset.ortho_ray(1, (0, location * 100))  # Convert to cm
            else:
                ray = dataset.ortho_ray(0, (location * 100, 0))

            sort_idx = np.argsort(ray["boxlib", 'x'])

            # Extract coordinates
            coords = self.unit_converter.convert_value(
                ray["boxlib", 'x'][sort_idx].to_value(), 'cm', 'm'
            )

            # Extract basic fields
            temperature = ray["boxlib", "Temp"][sort_idx].to_value()
            pressure = self.unit_converter.convert_value(
                ray["boxlib", "pressure"][sort_idx].to_value(), 'dyne/cm^2', 'Pa'
            )
            velocity_x = self.unit_converter.convert_value(
                ray["boxlib", "x_velocity"][sort_idx].to_value(), 'cm/s', 'm/s'
            )

            # Extract species data if available
            species_data = self._extract_species_data(ray, sort_idx)

            # Calculate derived fields using Cantera if available
            density = self._calculate_density(temperature, pressure, species_data)

            return FieldData(
                coordinates=coords,
                temperature=temperature,
                pressure=pressure,
                density=density,
                velocity_x=velocity_x,
                species_data=species_data
            )

        except Exception as e:
            raise DataExtractionError("ray", location, str(e))

    def extract_2d_contour(self, dataset: Any, field_name: str, iso_value: float) -> np.ndarray:
        """Extract 2D iso-contour."""
        try:
            # Use YT's contour extraction
            if hasattr(dataset, 'all_data'):
                contours = dataset.all_data().extract_isocontours(field_name, iso_value)
                return self._process_contour_points(contours)
            else:
                # Manual contour extraction
                return self._manual_contour_extraction(dataset, field_name, iso_value)

        except Exception as e:
            raise DataExtractionError("contour", None, str(e))

    def extract_thermodynamic_state(self, dataset: Any, location: Point2D) -> ThermodynamicState:
        """Extract thermodynamic state at point."""
        try:
            # Extract data at specific point (simplified)
            ray = dataset.ortho_ray(0, (location.y * 100, 0))
            sort_idx = np.argsort(ray["boxlib", 'x'])

            x_coords = ray["boxlib", 'x'][sort_idx].to_value()
            target_idx = np.argmin(np.abs(x_coords - location.x * 100))

            temp = ray["boxlib", "Temp"][sort_idx][target_idx].to_value()
            pressure = self.unit_converter.convert_value(
                ray["boxlib", "pressure"][sort_idx][target_idx].to_value(), 'dyne/cm^2', 'Pa'
            )

            # Calculate other properties if gas available
            if self.gas:
                species_data = self._extract_species_data(ray, sort_idx)
                if species_data and species_data.mass_fractions:
                    self.gas.TPY = temp, pressure, species_data.mass_fractions
                    return ThermodynamicState(
                        temperature=temp,
                        pressure=pressure,
                        density=self.gas.density,
                        sound_speed=self.gas.sound_speed,
                        viscosity=self.gas.viscosity,
                        conductivity=self.gas.thermal_conductivity
                    )

            # Basic state without detailed properties
            return ThermodynamicState(
                temperature=temp,
                pressure=pressure,
                density=1.0,  # Placeholder
                sound_speed=300.0  # Placeholder
            )

        except Exception as e:
            raise DataExtractionError("thermodynamic_state", None, str(e))

    def _create_field_map(self) -> Dict[str, str]:
        """Map standard field names to Pele field names."""
        return {
            'temperature': 'Temp',
            'pressure': 'pressure',
            'density': 'density',
            'x_velocity': 'x_velocity',
            'y_velocity': 'y_velocity',
            'heat_release': 'heatRelease'
        }

    def _extract_species_data(self, ray: Any, sort_idx: np.ndarray) -> Optional[SpeciesData]:
        """Extract species mass fractions."""
        if not CANTERA_AVAILABLE or not self.gas:
            return None

        species_data = SpeciesData()

        for species in self.gas.species_names:
            field_name = f'Y({species})'
            if field_name in dict(ray.ds.field_list):
                mass_fractions = ray["boxlib", field_name][sort_idx].to_value()
                species_data.mass_fractions[species] = mass_fractions

        return species_data if species_data.mass_fractions else None

    def _calculate_density(self, temperature: np.ndarray, pressure: np.ndarray,
                           species_data: Optional[SpeciesData]) -> np.ndarray:
        """Calculate density using ideal gas law or Cantera."""
        if self.gas and species_data and species_data.mass_fractions:
            density = np.zeros_like(temperature)
            for i in range(len(temperature)):
                try:
                    Y = {species: fractions[i] for species, fractions in species_data.mass_fractions.items()}
                    self.gas.TPY = temperature[i], pressure[i], Y
                    density[i] = self.gas.density
                except:
                    density[i] = pressure[i] / (287 * temperature[i])  # Air approximation
            return density
        else:
            # Ideal gas approximation
            return pressure / (287 * temperature)

    def _process_contour_points(self, contours: np.ndarray) -> np.ndarray:
        """Process raw contour points."""
        if len(contours) == 0:
            return np.array([])

        # Convert from cm to m
        contours_m = self.unit_converter.convert_value(contours, 'cm', 'm')

        # Sort contour points by connectivity
        return self._sort_contour_points(contours_m)

    def _sort_contour_points(self, points: np.ndarray) -> np.ndarray:
        """Sort contour points for continuous curves."""
        if len(points) < 2:
            return points

        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        sorted_points = [points[0]]
        remaining = list(range(1, len(points)))

        current_point = points[0]
        while remaining:
            distances, indices = tree.query(current_point, k=len(points))

            for idx in indices[1:]:  # Skip current point
                if idx in remaining:
                    sorted_points.append(points[idx])
                    remaining.remove(idx)
                    current_point = points[idx]
                    break

        return np.array(sorted_points)

    def _manual_contour_extraction(self, dataset: Any, field_name: str, iso_value: float) -> np.ndarray:
        """Manual contour extraction for datasets without YT support."""
        try:
            from matplotlib.tri import Triangulation
            import matplotlib.pyplot as plt

            # Extract 2D data at highest level
            max_level = dataset.index.max_level
            x_coords, y_coords, field_values = [], [], []

            for grid in dataset.index.grids:
                if grid.Level == max_level:
                    x_coords.extend(grid["boxlib", "x"].to_value().flatten())
                    y_coords.extend(grid["boxlib", "y"].to_value().flatten())
                    field_values.extend(grid[field_name].flatten())

            x_coords, y_coords = np.array(x_coords), np.array(y_coords)
            field_values = np.array(field_values)

            # Create triangulation and extract contour
            tri = Triangulation(x_coords, y_coords)
            contour = plt.tricontour(tri, field_values, levels=[iso_value])

            if contour.collections:
                paths = contour.collections[0].get_paths()
                if paths:
                    contour_points = paths[0].vertices
                    return self.unit_converter.convert_value(contour_points, 'cm', 'm')

            return np.array([])

        except Exception as e:
            raise DataExtractionError("manual_contour", None, str(e))


class MultiLevelExtractor(DataExtractor):
    """Extract data from multiple refinement levels."""

    def __init__(self, base_extractor: DataExtractor):
        self.base_extractor = base_extractor

    def extract_ray_data(self, dataset: Any, location: float, direction: Direction = Direction.X) -> FieldData:
        """Extract data from all refinement levels."""
        # For now, just use base extractor at max level
        return self.base_extractor.extract_ray_data(dataset, location, direction)

    def extract_2d_contour(self, dataset: Any, field_name: str, iso_value: float) -> np.ndarray:
        return self.base_extractor.extract_2d_contour(dataset, field_name, iso_value)

    def extract_thermodynamic_state(self, dataset: Any, location: Point2D) -> ThermodynamicState:
        return self.base_extractor.extract_thermodynamic_state(dataset, location)


def create_data_extractor(extractor_type: str = "pele", **kwargs) -> DataExtractor:
    """Factory for data extractors."""
    if extractor_type == "pele":
        return PeleDataExtractor(**kwargs)
    elif extractor_type == "multilevel":
        base_extractor = create_data_extractor("pele", **kwargs)
        return MultiLevelExtractor(base_extractor)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")