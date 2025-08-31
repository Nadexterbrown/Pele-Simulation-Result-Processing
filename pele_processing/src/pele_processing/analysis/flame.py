"""
Flame analysis for the Pele processing system.
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np

from ..core.interfaces import FlameAnalyzer, WaveTracker
from ..core.domain import FlameProperties, FieldData, WaveType, ThermodynamicState
from ..core.exceptions import FlameAnalysisError, WaveNotFoundError


class PeleFlameAnalyzer(FlameAnalyzer, WaveTracker):
    """Flame analysis implementation for Pele datasets."""

    def __init__(self, flame_temperature: float = 2500.0, transport_species: str = 'H2'):
        self.flame_temperature = flame_temperature
        self.transport_species = transport_species

    def sort_contour_by_nearest_neighbors(self, points: np.ndarray, dataset: Any) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """Sort contour points using nearest neighbors matching original algorithm."""
        if len(points) == 0:
            return points, [], 0.0

        # Filter points within buffer zone (matching original)
        buffer = 0.0075 * (dataset.domain_right_edge[1].to_value() - dataset.domain_left_edge[1].to_value()) / 100  # Convert to m
        domain_min = dataset.domain_left_edge[1].to_value() / 100 + buffer
        domain_max = dataset.domain_right_edge[1].to_value() / 100 - buffer

        valid_indices = (points[:, 1] >= domain_min) & (points[:, 1] <= domain_max)
        points = points[valid_indices]

        if len(points) == 0:
            return points, [], 0.0

        # Use cKDTree for efficient nearest neighbor search
        from scipy.spatial import cKDTree
        tree = cKDTree(points)

        # Start from bottom-left point
        origin_idx = np.argmin(np.lexsort((points[:, 0], points[:, 1])))
        order = [origin_idx]
        distance_arr = []
        segments = []
        segment_length = []
        segment_start = 0

        # Build ordered path
        for i in range(1, len(points)):
            distances, indices = tree.query(points[order[i - 1]], k=len(points))
            for neighbor_idx in indices[1:]:  # Skip self
                if neighbor_idx not in order:
                    order.append(neighbor_idx)
                    break

            # Calculate distance to next point
            distance = np.linalg.norm(points[order[i]] - points[order[i - 1]])
            distance_arr.append(distance)

            # Check for segment break (matching original threshold)
            if distance > 50 * dataset.index.get_smallest_dx().to_value() / 100:  # Convert to m
                segment = points[np.array(order[segment_start:i])]
                segments.append(segment)
                segment_length.append(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))
                segment_start = i

        # Add final segment
        if segment_start < len(order):
            segment = points[np.array(order[segment_start:])]
            segments.append(segment)
            segment_length.append(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))

        # Fallback to sklearn if coverage is poor (matching original)
        if len(segments) == 0 or len(np.concatenate(segments)) < 0.95 * len(points):
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=len(points), algorithm='ball_tree').fit(points)
            distances, indices = nbrs.kneighbors(points)

            origin_idx = np.argmin(points[:, 1])
            order = [origin_idx]
            segments = []
            segment_length = []
            segment_start = 0

            for i in range(1, len(points)):
                temp_idx = np.argwhere(indices[:, 0] == order[i - 1])[0][0]
                for neighbor_idx in indices[temp_idx, 1:]:
                    if neighbor_idx not in order:
                        order.append(neighbor_idx)
                        break

                distance = np.linalg.norm(points[order[i]] - points[order[i - 1]])
                if distance > 50 * dataset.index.get_smallest_dx().to_value() / 100:
                    segment = points[np.array(order[segment_start:i])]
                    segments.append(segment)
                    segment_length.append(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))
                    segment_start = i

        total_length = np.sum(segment_length)
        return points[np.array(order)], segments, total_length
        """Extract flame contour matching original implementation."""
        try:
            # Try YT contour extraction first
            if hasattr(dataset, 'all_data'):
                contours = dataset.all_data().extract_isocontours("Temp", self.flame_temperature)
                if len(contours) > 0:
                    return contours / 100  # Convert cm to m

            # Fall back to manual extraction
            return self._manual_contour_extraction(dataset)

        except Exception:
            # Manual extraction as fallback
            return self._manual_contour_extraction(dataset)

    def _manual_contour_extraction(self, dataset: Any) -> np.ndarray:
        """Manual contour extraction matching original."""
        from matplotlib.tri import Triangulation
        import matplotlib.pyplot as plt
        from scipy.interpolate import griddata

        # Get maximum refinement level
        max_level = dataset.index.max_level
        x_coords, y_coords, temperatures = [], [], []

        # Extract data from highest level grids
        for grid in dataset.index.grids:
            if grid.Level == max_level:
                x_coords.append(grid["boxlib", "x"].to_value().flatten())
                y_coords.append(grid["boxlib", "y"].to_value().flatten())
                temperatures.append(grid["boxlib", "Temp"].flatten())

        x_coords = np.concatenate(x_coords)
        y_coords = np.concatenate(y_coords)
        temperatures = np.concatenate(temperatures)

        # Create triangulation
        triangulation = Triangulation(x_coords, y_coords)
        contour = plt.tricontour(triangulation, temperatures, levels=[self.flame_temperature])

        # Extract contour points
        if contour.collections:
            paths = contour.collections[0].get_paths()
            if paths:
                contour_points = np.vstack([path.vertices for path in paths])
                return contour_points / 100  # Convert cm to m

        # If no contour found, try grid interpolation
        xi = np.linspace(np.min(x_coords), np.max(x_coords), int(1e4))
        yi = np.linspace(np.min(y_coords), np.max(y_coords), int(1e4))
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        temperature_grid = griddata((x_coords, y_coords), temperatures, (xi_grid, yi_grid), method='cubic')
        triangulation = Triangulation(xi_grid.flatten(), yi_grid.flatten())
        contour = plt.tricontour(triangulation, temperature_grid.flatten(), levels=[self.flame_temperature])

        if contour.collections:
            paths = contour.collections[0].get_paths()
            contour_points = np.vstack([path.vertices for path in paths])
            return contour_points / 100  # Convert cm to m

        return np.array([])

    def find_wave_position(self, data: FieldData, wave_type: WaveType) -> Tuple[int, float]:
        """Find flame position using temperature and species criteria."""
        if wave_type != WaveType.FLAME:
            raise WaveNotFoundError(wave_type.value, "Only flame detection supported")

        # Primary criterion: temperature threshold
        flame_indices = np.where(data.temperature >= self.flame_temperature)[0]

        if len(flame_indices) == 0:
            raise WaveNotFoundError("flame", f"No points above {self.flame_temperature}K")

        # Use downstream-most point above threshold
        temp_flame_idx = flame_indices[-1]

        # Secondary validation with Y(HO2) species if available
        if data.species_data and 'HO2' in data.species_data.mass_fractions:
            species_flame_idx = np.argmax(data.species_data.mass_fractions['HO2'])

            # Check agreement between methods
            if abs(temp_flame_idx - species_flame_idx) > 10:
                print(f'Warning: Flame Location differs by more than 10 cells!\n'
                      f'Flame Temperature Location {data.coordinates[temp_flame_idx]}\n'
                      f'Flame Species Location {data.coordinates[species_flame_idx]}')

            flame_idx = species_flame_idx  # Prefer species-based detection
        else:
            flame_idx = temp_flame_idx

        return flame_idx, data.coordinates[flame_idx]

    def calculate_wave_velocity(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate flame velocity from position time series."""
        if len(positions) < 2:
            return np.array([])
        return np.gradient(positions, times)

    def calculate_flame_thickness(self, dataset: Any, contour_points: Optional[np.ndarray],
                                 center_location: float) -> float:
        """Calculate flame thickness matching original implementation."""
        try:
            if contour_points is None or len(contour_points) == 0:
                raise FlameAnalysisError("thickness", "No contour points provided")

            # Find flame point closest to center location
            flame_idx = np.argmin(np.abs(contour_points[:, 1] - center_location))
            flame_x, flame_y = contour_points[flame_idx]

            # Extract simulation grid (matching original)
            subgrid_x, subgrid_y, subgrid_temperatures = self._extract_simulation_grid(dataset, flame_x, flame_y)

            # Create symmetric subgrid around flame point
            region_grid, region_temperature, interpolator = self._create_subgrid(
                subgrid_x, subgrid_y, subgrid_temperatures, flame_x, flame_y)

            # Calculate contour normal vectors
            contour_normals = self._calculate_contour_normal(contour_points)

            # Generate normal line through flame point
            normal_line = self._calculate_normal_vector_line(
                contour_normals[flame_idx], contour_points[flame_idx], region_grid)

            # Interpolate temperature along normal
            normal_distances = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(normal_line, axis=0)**2, axis=1))), 0, 0)
            normal_temperatures = interpolator(normal_line)

            # Calculate thickness from temperature gradient
            temp_grad = np.abs(np.gradient(normal_temperatures, normal_distances))

            if len(temp_grad) == 0 or np.max(temp_grad) == 0:
                return np.nan

            thickness = (np.max(normal_temperatures) - np.min(normal_temperatures)) / np.max(temp_grad)
            return thickness  # Already in meters

        except Exception as e:
            raise FlameAnalysisError("thickness", str(e))

    def _extract_simulation_grid(self, dataset: Any, flame_x: float, flame_y: float):
        """Extract simulation grid matching original algorithm."""
        max_level = dataset.index.max_level
        grids = [grid for grid in dataset.index.grids if grid.Level == max_level]

        # Pre-extract grid data
        grid_data = []
        for grid in grids:
            x = grid["boxlib", "x"].to_value().flatten() / 100  # Convert to m
            y = grid["boxlib", "y"].to_value().flatten() / 100
            temp = grid["boxlib", "Temp"].flatten()
            grid_data.append((x, y, temp))

        # Filter grids by mean x difference (matching original)
        subgrid_x, subgrid_y, subgrid_temperatures = [], [], []

        for i, (x, y, temp) in enumerate(grid_data):
            current_mean_x = np.mean(x)
            if current_mean_x > flame_x + 1e-2:  # Skip if too far downstream
                continue

            subgrid_x.extend(x)
            subgrid_y.extend(y)
            subgrid_temperatures.extend(temp)

        return np.array(subgrid_x), np.array(subgrid_y), np.array(subgrid_temperatures)

    def _create_subgrid(self, subgrid_x, subgrid_y, subgrid_temperatures, flame_x, flame_y):
        """Create symmetric subgrid around flame point matching original."""
        # Find indices closest to flame position
        flame_x_idx = np.argmin(np.abs(subgrid_x - flame_x))
        flame_y_idx = np.argmin(np.abs(subgrid_y - flame_y))

        # Get arrays along flame position
        flame_x_arr = subgrid_x[np.abs(subgrid_y - subgrid_y[flame_y_idx]) <= 1e-12]
        flame_y_arr = subgrid_y[np.abs(subgrid_x - subgrid_x[flame_x_idx]) <= 1e-12]

        flame_x_arr_idx = np.argmin(np.abs(flame_x_arr - flame_x))
        flame_y_arr_idx = np.argmin(np.abs(flame_y_arr - flame_y))

        # Determine subgrid size (matching original logic)
        left_x_indices = flame_x_arr_idx
        right_x_indices = len(flame_x_arr) - flame_x_arr_idx - 1
        x_indices = min(left_x_indices, right_x_indices)

        top_y_indices = flame_y_arr_idx
        bottom_y_indices = len(flame_y_arr) - flame_y_arr_idx - 1
        y_indices = min(top_y_indices, bottom_y_indices)

        subgrid_bin_size = min(x_indices, y_indices, 11)

        # Create symmetric subgrid
        subgrid_flame_x = flame_x_arr[flame_x_arr_idx - subgrid_bin_size:flame_x_arr_idx + subgrid_bin_size + 1]
        subgrid_flame_y = flame_y_arr[flame_y_arr_idx - subgrid_bin_size:flame_y_arr_idx + subgrid_bin_size + 1]

        # Create temperature grid
        subgrid_temperatures = np.full((len(subgrid_flame_y), len(subgrid_flame_x)), np.nan)

        for i, y in enumerate(subgrid_flame_y):
            for j, x in enumerate(subgrid_flame_x):
                matching_indices = np.where((subgrid_x == x) & (subgrid_y == y))
                if len(matching_indices[0]) > 0:
                    subgrid_temperatures[i, j] = subgrid_temperatures[matching_indices[0][0]]

        region_grid = np.dstack(np.meshgrid(subgrid_flame_x, subgrid_flame_y)).reshape(-1, 2)
        region_temperature = subgrid_temperatures.reshape(np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape)

        # Create interpolator
        from scipy.interpolate import RegularGridInterpolator
        interpolator = RegularGridInterpolator(
            (np.unique(region_grid[:, 0]), np.unique(region_grid[:, 1])),
            region_temperature.T, bounds_error=False, fill_value=np.nan)

        return region_grid, region_temperature, interpolator

    def _calculate_contour_normal(self, contour_points: np.ndarray) -> np.ndarray:
        """Calculate normal vectors for contour points matching original."""
        dx = np.gradient(contour_points[:, 0])
        dy = np.gradient(contour_points[:, 1])

        normals = np.zeros_like(contour_points)
        for i in range(len(dx)):
            if dx[i] == 0:  # Vertical contour
                normals[i, 0] = 0
                normals[i, 1] = 1
            elif dy[i] == 0:  # Horizontal contour
                normals[i, 0] = 1
                normals[i, 1] = 0
            else:  # General case - rotate tangent 90 degrees
                normals[i, 0] = dy[i]
                normals[i, 1] = -dx[i]

        # Normalize
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms[:, np.newaxis]

        return normals

    def _calculate_normal_vector_line(self, normal_vector: np.ndarray, center_point: np.ndarray, region_grid: np.ndarray):
        """Calculate normal line points matching original algorithm."""
        # Determine step size
        dx = np.abs(np.unique(region_grid[:, 0])[1] - np.unique(region_grid[:, 0])[0])
        dy = np.abs(np.unique(region_grid[:, 1])[1] - np.unique(region_grid[:, 1])[0])
        t_step = min(dx, dy) / np.linalg.norm(normal_vector)

        # Calculate bounds
        t_min_x = (np.min(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[0] != 0 else -np.inf
        t_max_x = (np.max(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[0] != 0 else np.inf
        t_min_y = (np.min(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[1] != 0 else -np.inf
        t_max_y = (np.max(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[1] != 0 else np.inf

        t_start = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y))
        t_end = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y))

        # Generate line points
        t_range = np.arange(t_start, t_end, t_step / 100, dtype=np.float32)
        line_points = np.column_stack([
            center_point[0] + t_range * normal_vector[0],
            center_point[1] + t_range * normal_vector[1]
        ])

        # Filter within bounds
        min_x, max_x = np.min(region_grid[:, 0]), np.max(region_grid[:, 0])
        min_y, max_y = np.min(region_grid[:, 1]), np.max(region_grid[:, 1])

        filtered_points = line_points[
            (line_points[:, 0] >= min_x) & (line_points[:, 0] <= max_x) &
            (line_points[:, 1] >= min_y) & (line_points[:, 1] <= max_y)
        ]

    def analyze_flame_properties(self, dataset: Any, data: FieldData) -> FlameProperties:
        """Complete flame analysis matching original flame_geometry function."""
        # Find flame position from 1D data
        flame_idx, flame_pos = self.find_wave_position(data, WaveType.FLAME)

        properties = FlameProperties(position=flame_pos, index=flame_idx)

        # Extract 2D flame contour
        try:
            contour_points = self.extract_flame_contour(dataset)
            if len(contour_points) > 0:
                # Sort contour points (matching original algorithm)
                sorted_points, segments, surface_length = self.sort_contour_by_nearest_neighbors(contour_points, dataset)

                properties.surface_length = surface_length
                properties.contour_points = sorted_points

                # Calculate flame thickness if contour available
                try:
                    center_location = flame_pos  # Use 1D flame position as center
                    properties.thickness = self.calculate_flame_thickness(dataset, sorted_points, center_location)
                except Exception as e:
                    print(f"Flame thickness calculation failed: {e}")
                    properties.thickness = np.nan

                # Calculate consumption rate
                try:
                    consumption_rate, burning_velocity = self.calculate_consumption_rate(
                        dataset, sorted_points, self.transport_species)
                    properties.consumption_rate = consumption_rate
                    properties.burning_velocity = burning_velocity
                except Exception as e:
                    print(f"Consumption rate calculation failed: {e}")
                    properties.consumption_rate = np.nan
                    properties.burning_velocity = np.nan
            else:
                print("No flame contour found")
                properties.surface_length = np.nan
                properties.thickness = np.nan

        except Exception as e:
            print(f"2D flame analysis failed: {e}")
            properties.surface_length = np.nan
            properties.thickness = np.nan

        return properties

    def _extract_flame_region_detailed(self, dataset: Any, flame_point: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract detailed temperature field around flame point."""
        max_level = dataset.index.max_level
        flame_x, flame_y = flame_point  # Already in meters

        # Extract grids at max level near flame
        region_data = {'x': [], 'y': [], 'temp': []}

        for grid in dataset.index.grids:
            if grid.Level == max_level:
                x = grid["boxlib", "x"].to_value().flatten() / 100  # Convert cm to m
                y = grid["boxlib", "y"].to_value().flatten() / 100
                temp = grid["Temp"].flatten()

                # Filter points within reasonable distance
                current_mean_x = np.mean(x)
                if current_mean_x > flame_x + 1e-2:  # Skip grids too far downstream
                    continue

                region_data['x'].extend(x)
                region_data['y'].extend(y)
                region_data['temp'].extend(temp)

        return {
            'x': np.array(region_data['x']),
            'y': np.array(region_data['y']),
            'temp': np.array(region_data['temp'])
        }

    def _create_temperature_interpolator(self, region_data: Dict[str, np.ndarray]) -> callable:
        """Create 2D temperature interpolator."""
        from scipy.interpolate import RegularGridInterpolator, griddata

        points = np.column_stack((region_data['x'], region_data['y']))

        # Create regular grid
        x_min, x_max = region_data['x'].min(), region_data['x'].max()
        y_min, y_max = region_data['y'].min(), region_data['y'].max()

        xi = np.linspace(x_min, x_max, 50)
        yi = np.linspace(y_min, y_max, 50)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate to regular grid
        temp_grid = griddata(points, region_data['temp'], (xi_grid, yi_grid), method='cubic')

        return RegularGridInterpolator((yi, xi), temp_grid, bounds_error=False, fill_value=np.nan)

    def calculate_consumption_rate(self, dataset: Any, contour_points: Optional[np.ndarray],
                                  transport_species: str) -> Tuple[float, float]:
        """Calculate species consumption rate and burning velocity."""
        try:
            if contour_points is None or len(contour_points) == 0:
                raise FlameAnalysisError("consumption_rate", "No contour points provided")

            # Create bounding box around contour
            min_x = np.min(contour_points[:, 0]) - 10e-4
            max_x = np.max(contour_points[:, 0]) + 10e-4
            min_y = np.min(contour_points[:, 1])
            max_y = np.max(contour_points[:, 1])

            # Convert to cm for YT
            left_edge = [min_x * 100, min_y * 100, dataset.domain_left_edge[2].to_value()]
            right_edge = [max_x * 100, max_y * 100, dataset.domain_right_edge[2].to_value()]

            level = dataset.index.max_level
            dims = ((np.array(right_edge) - np.array(left_edge)) / dataset.index.get_smallest_dx().to_value()).astype(int)
            dims[2] = 1

            # Create covering grid
            cg = dataset.covering_grid(
                level=level,
                left_edge=left_edge,
                dims=dims,
                fields=[('boxlib', 'x'), ('boxlib', 'y'),
                       ('boxlib', f'rho_{transport_species}'),
                       ('boxlib', f'rho_omega_{transport_species}')]
            )

            # Calculate consumption rate by integrating production rates
            dx = cg.dds[0].to_value() / 100  # Convert cm to m
            dy = cg.dds[1].to_value() / 100

            total_consumption = 0.0
            num_rows = cg['boxlib', 'x'].shape[1]

            for i in range(num_rows):
                row_data = cg['boxlib', f'rho_omega_{transport_species}'][:, i, 0].to_value()
                # Convert from g/(cm³·s) to kg/(m³·s)
                row_data *= 1000
                total_consumption += np.sum(np.abs(row_data)) * dx * dy

            # Calculate burning velocity
            # Get reactant density at flame location
            rho_reactant = cg["boxlib", f"rho_{transport_species}"].to_value()[-1, 0, 0] * 1000  # Convert to kg/m³
            domain_height = (dataset.domain_right_edge[1].to_value() - dataset.domain_left_edge[1].to_value()) / 100

            burning_velocity = total_consumption / (rho_reactant * domain_height)

            return total_consumption, burning_velocity

        except Exception as e:
            raise FlameAnalysisError("consumption_rate", str(e))

    def _extract_flame_region(self, dataset: Any, flame_position: float) -> Dict[str, np.ndarray]:
        """Extract local temperature field around flame."""
        max_level = dataset.index.max_level
        region_size = 2e-3  # 2mm region

        x_coords, y_coords, temperatures = [], [], []

        for grid in dataset.index.grids:
            if grid.Level == max_level:
                x = grid["boxlib", "x"].to_value().flatten()
                y = grid["boxlib", "y"].to_value().flatten()
                temp = grid["Temp"].flatten()

                # Filter by distance from flame
                distances = np.abs(x - flame_position * 100)  # Convert to cm
                mask = distances < region_size * 100

                x_coords.extend(x[mask])
                y_coords.extend(y[mask])
                temperatures.extend(temp[mask])

        return {
            'x': np.array(x_coords),
            'y': np.array(y_coords),
            'temperature': np.array(temperatures)
        }

    def _calculate_flame_normal(self, contour_points: np.ndarray, center_idx: int,
                               region_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate normal line through flame at center point."""
        # Calculate contour gradients to get tangent
        dx = np.gradient(contour_points[:, 0])
        dy = np.gradient(contour_points[:, 1])

        # Normal vector (perpendicular to tangent)
        normal_x = dy[center_idx]
        normal_y = -dx[center_idx]

        # Normalize
        normal_length = np.sqrt(normal_x**2 + normal_y**2)
        if normal_length > 0:
            normal_x /= normal_length
            normal_y /= normal_length

        # Center point
        center_point = contour_points[center_idx]

        # Determine line extent
        x_span = region_data['x'].max() - region_data['x'].min()
        y_span = region_data['y'].max() - region_data['y'].min()
        max_extent = min(x_span, y_span) * 0.4

        # Generate line points
        t_values = np.linspace(-max_extent, max_extent, 100)
        line_points = np.column_stack([
            center_point[0] + t_values * normal_x,
            center_point[1] + t_values * normal_y
        ])

        # Filter to stay within region bounds
        min_x, max_x = region_data['x'].min(), region_data['x'].max()
        min_y, max_y = region_data['y'].min(), region_data['y'].max()

        valid_mask = ((line_points[:, 0] >= min_x) & (line_points[:, 0] <= max_x) &
                     (line_points[:, 1] >= min_y) & (line_points[:, 1] <= max_y))

        return line_points[valid_mask]

    def _interpolate_along_normal(self, region_data: Dict[str, np.ndarray],
                                 normal_line: np.ndarray) -> np.ndarray:
        """Interpolate temperature along normal line."""
        from scipy.interpolate import griddata

        points = np.column_stack([region_data['x'], region_data['y']])
        temperatures = griddata(points, region_data['temperature'], normal_line, method='linear')

        return temperatures

    def _compute_thickness_from_gradient(self, normal_line: np.ndarray,
                                        temperatures: np.ndarray) -> float:
        """Compute flame thickness from temperature profile along normal."""
        # Remove NaN values
        valid_mask = ~np.isnan(temperatures)
        if np.sum(valid_mask) < 10:
            return np.nan

        valid_temps = temperatures[valid_mask]
        valid_positions = normal_line[valid_mask]

        # Calculate cumulative distances along normal line
        distances = np.zeros(len(valid_positions))
        for i in range(1, len(valid_positions)):
            distances[i] = distances[i-1] + np.linalg.norm(valid_positions[i] - valid_positions[i-1])

        # Calculate temperature gradient
        temp_gradient = np.abs(np.gradient(valid_temps, distances))

        if len(temp_gradient) == 0 or np.max(temp_gradient) == 0:
            return np.nan

        # Flame thickness = temperature rise / max gradient
        temp_rise = np.max(valid_temps) - np.min(valid_temps)
        max_gradient = np.max(temp_gradient)

        return temp_rise / max_gradient  # Already in meters


def create_flame_analyzer(flame_temperature: float = 2500.0, **kwargs) -> FlameAnalyzer:
    """Factory for flame analyzers."""
    return PeleFlameAnalyzer(flame_temperature, **kwargs)