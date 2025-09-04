"""
Flame analysis for the Pele processing system.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ..core.interfaces import FlameAnalyzer, WaveTracker
from ..core.domain import FlameProperties, FieldData, WaveType, ThermodynamicState
from ..core.exceptions import FlameAnalysisError, WaveNotFoundError


class PeleFlameAnalyzer(FlameAnalyzer, WaveTracker):
    """Flame analysis implementation for Pele datasets."""

    def __init__(self, flame_temperature: float = 2500.0, transport_species: str = 'H2'):
        self.flame_temperature = flame_temperature
        self.transport_species = transport_species

    def extract_flame_contour(self, dataset: Any, flame_pos: float = None) -> np.ndarray:
        """Extract flame contour using YT parallel processing with fallbacks."""
        # Try YT covering grid extraction first (fastest when flame_pos known)
        try:
            return self._extract_contour(dataset, flame_pos)
        except Exception as e:
            print(f"YT extraction failed: {e}")
            return np.array([])

    def sort_contour_by_nearest_neighbors(self, points: np.ndarray, dataset: Any) -> Tuple[
        np.ndarray, List[np.ndarray], float]:
        """Sort contour points using simple nearest neighbor approach."""
        if len(points) == 0:
            return points, [], 0.0

        # Filter points within buffer zone
        buffer = 0.0075 * (dataset.domain_right_edge[1].to_value() - dataset.domain_left_edge[
            1].to_value()) / 100  # Convert to m
        domain_min = dataset.domain_left_edge[1].to_value() / 100 + buffer
        domain_max = dataset.domain_right_edge[1].to_value() / 100 - buffer

        valid_indices = (points[:, 1] >= domain_min) & (points[:, 1] <= domain_max)
        points = points[valid_indices]

        if len(points) == 0:
            return points, [], 0.0

        # Use simple nearest neighbor sorting
        return self._sort_by_nearest_neighbors(points, dataset)

    def calculate_flame_thickness(self, dataset: Any, contour_points: Optional[np.ndarray],
                                 center_location: float) -> tuple:
        """Calculate flame thickness using original Pele-2D-Data-Processing method.
        
        Returns:
            tuple: (flame_thickness, plotting_data_dict) where plotting_data_dict contains
                   region_grid, region_temperature, normal_line, interpolated_temperatures
        """
        try:
            if contour_points is None or len(contour_points) == 0:
                raise FlameAnalysisError("thickness", "No contour points provided")

            # Step 1: Extract the flame location from the contour 
            flame_idx = np.argmin(abs(contour_points[:, 1] - center_location))
            flame_x, flame_y = contour_points[flame_idx]

            # Step 2: Extract simulation grid
            subgrid_x, subgrid_y, subgrid_temperatures = self._extract_simulation_grid(dataset, flame_x, flame_y)

            # Step 3: Find the nearest index to the flame contour
            flame_x_idx = np.argmin(np.abs(subgrid_x - flame_x))
            flame_y_idx = np.argmin(np.abs(subgrid_y - flame_y))

            flame_x_arr = subgrid_x[np.abs(subgrid_y - subgrid_y[flame_y_idx]) <= 1e-12]
            flame_y_arr = subgrid_y[np.abs(subgrid_x - subgrid_x[flame_x_idx]) <= 1e-12]

            flame_x_arr_idx = np.argmin(np.abs(flame_x_arr - flame_x))
            flame_y_arr_idx = np.argmin(np.abs(flame_y_arr - flame_y))

            # Step 4: Create subgrid
            region_grid, region_temperature, interpolator = self._create_subgrid(
                subgrid_x, subgrid_y, subgrid_temperatures, flame_x, flame_y, 
                flame_x_arr, flame_y_arr, flame_x_arr_idx, flame_y_arr_idx)

            # Step 5: Calculate contour normal and normal line
            contour_normals = self._calculate_contour_normal(contour_points)
            normal_line = self._calculate_normal_vector_line(contour_normals[flame_idx], region_grid)
            
            # Step 6: Calculate thickness using original method
            normal_distances = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(normal_line, axis=0) ** 2, axis=1))), 0, 0)
            temp_grad = np.abs(np.gradient(interpolator(normal_line)) / np.gradient(normal_distances))
            
            flame_thickness_val = (np.max(interpolator(normal_line)) - np.min(interpolator(normal_line))) / np.max(temp_grad)
            
            # Store plotting data
            plotting_data = {
                'region_grid': region_grid,
                'region_temperature': region_temperature,
                'normal_line': normal_line,
                'interpolated_temperatures': interpolator(normal_line)
            }
            
            return flame_thickness_val, plotting_data

        except Exception as e:
            print(f"Error: Unable to calculate flame thickness: {e}")
            return np.nan, None

    def _extract_contour(self, dataset: Any, flame_pos: float = None) -> np.ndarray:
        """Extract flame contour using YT's covering grid + extract_isocontours."""
        import yt
        
        # Method 1: Smart box region extraction (focused around flame when known)
        try:
            domain_left = dataset.domain_left_edge.to_value()
            domain_right = dataset.domain_right_edge.to_value()
            
            if flame_pos is not None:
                # Create focused box region around flame (+/-5mm window)
                flame_window_left = 2.5e-3  # 5mm in meters
                flame_window_right = 0.5e-3  # 5mm in meters
                flame_pos_cm = flame_pos * 100  # Convert to cm for YT
                
                # Create focused left/right edges (+/-5mm around flame)
                # Use the full Y domain but ensure X bounds are properly clamped
                focused_left = [
                    max(domain_left[0], flame_pos_cm - flame_window_left * 100),  # Don't go outside domain
                    domain_left[1],
                    domain_left[2]
                ]
                focused_right = [
                    min(domain_right[0], flame_pos_cm + flame_window_right * 100),
                    domain_right[1], 
                    domain_right[2]
                ]
                
                print(f"Creating focused box region around flame at {flame_pos:.4f}m")
                print(f"Box bounds: x=[{focused_left[0]:.1f}, {focused_right[0]:.1f}]cm, window: -{flame_window_left*1000:.1f}/+{flame_window_right*1000:.1f}mm")
            else:
                # Fallback to broader regional extraction (2.5% buffer from each end)
                flame_buffer = 0.025
                focused_left = [domain_left[0] + flame_buffer * (domain_right[0] - domain_left[0]), domain_left[1], domain_left[2]]
                focused_right = [domain_right[0] - flame_buffer * (domain_right[0] - domain_left[0]), domain_right[1], domain_right[2]]
                
                print(f"Creating regional box region (no flame position known)")
                print(f"Box bounds: x=[{focused_left[0]:.1f}, {focused_right[0]:.1f}]cm, 2.5% buffer")
            
            # Create box region and extract contours with validation
            print(f"Domain bounds: x=[{domain_left[0]:.2f}, {domain_right[0]:.2f}]cm, y=[{domain_left[1]:.4f}, {domain_right[1]:.4f}]cm")
            print(f"Box region: x=[{focused_left[0]:.2f}, {focused_right[0]:.2f}]cm, y=[{focused_left[1]:.4f}, {focused_right[1]:.4f}]cm")
            
            # Validate bounds are within domain
            if (focused_left[0] < domain_left[0] or focused_right[0] > domain_right[0] or
                focused_left[1] < domain_left[1] or focused_right[1] > domain_right[1]):
                print("Warning: Box bounds exceed domain, falling back to full domain extraction")
                raise ValueError("Box bounds exceed domain")

            box_region = dataset.box(left_edge=focused_left, right_edge=focused_right)
            contours = box_region.extract_isocontours("Temp", self.flame_temperature)
            
            if len(contours) > 0:
                contour_2d = contours[:, :2] / 100  # cm to m, drop z
                valid_mask = np.isfinite(contour_2d).all(axis=1)
                clean_contours = contour_2d[valid_mask]
                
                if len(clean_contours) > 0:
                    method_type = "focused" if flame_pos is not None else "regional"
                    print(f"Extracted {len(clean_contours)} contour points using {method_type} box region")
                    return self._remove_duplicate_points(clean_contours)
                        
        except Exception as e:
            print(f"Box region extraction failed: {e}")
        
        # Method 2: Full domain extraction
        try:
            if hasattr(dataset, 'all_data'):
                print("Trying full domain isocontour extraction...")
                contours = dataset.all_data().extract_isocontours("Temp", self.flame_temperature)
                if len(contours) > 0:
                    contour_2d = contours[:, :2] / 100  # Convert cm to m, drop z
                    valid_mask = np.isfinite(contour_2d).all(axis=1)
                    clean_contours = contour_2d[valid_mask]
                    if len(clean_contours) > 0:
                        print(f"Extracted {len(clean_contours)} contour points using full domain")
                        return self._remove_duplicate_points(clean_contours)
        except Exception as e:
            print(f"Full domain isocontour extraction failed: {e}")
        
        
        raise ValueError("All YT contour extraction methods failed")

    def _sort_by_nearest_neighbors(self, points: np.ndarray, dataset: Any) -> Tuple[
        np.ndarray, List[np.ndarray], float]:
        """Sort contour points using efficient nearest neighbor approach with segment breaking."""
        if len(points) < 2:
            return points, [points] if len(points) > 0 else [], 0.0
        
        from scipy.spatial import cKDTree
        
        # Calculate segment break distance threshold
        max_segment_distance = 20 * dataset.index.get_smallest_dx().to_value() / 100  # Convert to m
        
        # Build spatial index for efficient nearest neighbor queries
        tree = cKDTree(points)
        
        # Start from the leftmost point
        start_idx = np.argmin(points[:, 0])
        
        # Track visited points efficiently using boolean array
        visited = np.zeros(len(points), dtype=bool)
        ordered_indices = []
        segments = []
        segment_start = 0
        
        current_idx = start_idx
        visited[current_idx] = True
        ordered_indices.append(current_idx)
        
        while len(ordered_indices) < len(points):
            current_point = points[current_idx]
            
            # Find k nearest neighbors (k=min(20, remaining_points))
            remaining_count = len(points) - len(ordered_indices)
            k = min(20, remaining_count + 1)  # +1 because query includes current point
            
            distances, neighbor_indices = tree.query(current_point, k=k)
            
            # Find the nearest unvisited neighbor
            nearest_idx = None
            min_distance = float('inf')
            
            for dist, idx in zip(distances, neighbor_indices):
                if not visited[idx] and dist < min_distance:
                    nearest_idx = idx
                    min_distance = dist
            
            # Fallback: if no neighbor found in k-nearest, search all unvisited
            if nearest_idx is None:
                unvisited_mask = ~visited
                if np.any(unvisited_mask):
                    unvisited_indices = np.where(unvisited_mask)[0]
                    unvisited_points = points[unvisited_indices]
                    distances_to_unvisited = np.linalg.norm(unvisited_points - current_point, axis=1)
                    min_idx = np.argmin(distances_to_unvisited)
                    nearest_idx = unvisited_indices[min_idx]
                    min_distance = distances_to_unvisited[min_idx]
            
            if nearest_idx is None:
                break  # All points visited
            
            # Check if we need to create a new segment
            if min_distance > max_segment_distance and len(ordered_indices) > segment_start + 1:
                # Create segment from segment_start to current position
                segment_indices = ordered_indices[segment_start:]
                segment_points = points[segment_indices]
                segments.append(segment_points)
                segment_start = len(ordered_indices)
            
            # Add point to path
            visited[nearest_idx] = True
            ordered_indices.append(nearest_idx)
            current_idx = nearest_idx
        
        # Add final segment
        if segment_start < len(ordered_indices):
            segment_indices = ordered_indices[segment_start:]
            segment_points = points[segment_indices]
            segments.append(segment_points)
        
        # If no segments were created, treat all points as one segment
        if not segments:
            segments = [points[ordered_indices]]
        
        # Calculate total path length efficiently
        ordered_points = points[ordered_indices]
        total_length = 0.0
        
        for segment in segments:
            if len(segment) > 1:
                # Vectorized distance calculation
                segment_vectors = np.diff(segment, axis=0)
                segment_distances = np.linalg.norm(segment_vectors, axis=1)
                total_length += np.sum(segment_distances)
        
        return ordered_points, segments, total_length


    def _remove_duplicate_points(self, points: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
        """Remove duplicate points within tolerance."""
        if len(points) <= 1:
            return points
            
        from scipy.spatial.distance import pdist, squareform
        
        try:
            distances = pdist(points)
            distance_matrix = squareform(distances)
            
            close_pairs = np.where((distance_matrix < tolerance) & (distance_matrix > 0))
            
            indices_to_remove = set()
            for i, j in zip(close_pairs[0], close_pairs[1]):
                if i < j:  # Only process each pair once
                    indices_to_remove.add(j)
            
            keep_indices = [i for i in range(len(points)) if i not in indices_to_remove]
            return points[keep_indices]
            
        except:
            # Fallback to simple approach
            unique_points = [points[0]]
            
            for point in points[1:]:
                distances = np.linalg.norm(np.array(unique_points) - point, axis=1)
                if np.min(distances) > tolerance:
                    unique_points.append(point)
            
            return np.array(unique_points)

    def _extract_simulation_grid(self, dataset: Any, flame_x: float, flame_y: float):
        """Extract simulation grid using original method."""
        # Step 1: Collect the max level grids
        max_level = dataset.index.max_level
        grids = [grid for grid in dataset.index.grids if grid.Level == max_level]

        # Step 2: Pre-allocate lists for subgrid data and filtered grids
        subgrid_x, subgrid_y, subgrid_temperatures = [], [], []
        filtered_grids = []

        # Step 3: Pre-extract the grid data once for efficiency
        grid_data = []
        for temp_grid in grids:
            x = temp_grid["boxlib", "x"].to_value().flatten()
            y = temp_grid["boxlib", "y"].to_value().flatten()
            temp = temp_grid["Temp"].flatten()
            grid_data.append((x, y, temp))

        # Step 4: Filter grids based on mean x difference (original method)
        for i, (x, y, temp) in enumerate(grid_data):
            # Calculate the mean x value for the current grid
            current_mean_x = np.mean(x)
            if i < len(grids) - 1:
                # If the difference in mean x values is too large, skip the current grid
                if current_mean_x > flame_x * 100 + 1e-2:  # Convert flame_x to cm for comparison
                    continue

            # If this grid is not skipped, append it to the filtered list
            filtered_grids.append(grids[i])
            # Collect the values from this grid
            subgrid_x.extend(x)
            subgrid_y.extend(y)
            subgrid_temperatures.extend(temp)

        # Convert to cm for consistency with original
        subgrid_x = np.array(subgrid_x) / 100  # Convert to m
        subgrid_y = np.array(subgrid_y) / 100  # Convert to m
        subgrid_temperatures = np.array(subgrid_temperatures)

        return subgrid_x, subgrid_y, subgrid_temperatures

    def _create_subgrid(self, subgrid_x, subgrid_y, subgrid_temperatures, flame_x, flame_y, 
                       flame_x_arr, flame_y_arr, flame_x_arr_idx, flame_y_arr_idx):
        """Create subgrid using original method."""
        # Step 1: Determine the number of indices to the left and right of the flame_x_idx
        left_x_indices = flame_x_arr_idx
        right_x_indices = len(flame_x_arr) - flame_x_arr_idx - 1
        # Determine the smallest number of cells for the x indices
        x_indices = min(left_x_indices, right_x_indices)

        # Step 2: Determine the number of indices to the top and bottom of the flame_y_idx
        top_y_indices = flame_y_arr_idx
        bottom_y_indices = len(flame_y_arr) - flame_y_arr_idx - 1
        # Determine the smallest number of cells for the y indices
        y_indices = min(top_y_indices, bottom_y_indices)

        # Step 3: Determine the subgrid bin size
        if min(x_indices, y_indices) < 11:
            subgrid_bin_size = min(x_indices, y_indices)
        else:
            subgrid_bin_size = 11

        # Step 4: Create subgrid with the appropriate number of indices on either side of flame_x_idx and flame_y_idx
        subgrid_flame_x = flame_x_arr[flame_x_arr_idx - subgrid_bin_size:flame_x_arr_idx + subgrid_bin_size + 1]
        subgrid_flame_y = flame_y_arr[flame_y_arr_idx - subgrid_bin_size:flame_y_arr_idx + subgrid_bin_size + 1]

        # Step 5: Create a grid of temperature values corresponding to the subgrid (subgrid_flame_x, subgrid_flame_y)
        subgrid_temperatures_grid = np.full((len(subgrid_flame_y), len(subgrid_flame_x)), np.nan)

        # Step 6: Create a grid of temperature values corresponding to the subgrid (subgrid_flame_x, subgrid_flame_y)
        # Iterate over the subgrid (x, y) pairs and find the corresponding temperature from the collective data
        for i, y in enumerate(subgrid_flame_y):
            for j, x in enumerate(subgrid_flame_x):
                # Find the index in the collective data that corresponds to the current (x, y)
                matching_indices = np.where((subgrid_x == x) & (subgrid_y == y))

                if len(matching_indices[0]) > 0:
                    # If a match is found, assign the temperature at the (x, y) position
                    try:
                        subgrid_temperatures_grid[i, j] = subgrid_temperatures[matching_indices[0][0]]
                    except:
                        subgrid_temperatures_grid[i, j] = np.nan

        region_grid = np.dstack(np.meshgrid(subgrid_flame_x, subgrid_flame_y)).reshape(-1, 2)
        region_temperature = subgrid_temperatures_grid.reshape(np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape)

        # Original rotation/flipping logic to match grid orientation
        from scipy.interpolate import RegularGridInterpolator
        
        break_outer = False
        for i in range(2):
            if i == 0:
                temp_arr = region_temperature
            else:
                temp_arr = np.flip(region_temperature, axis=i - 1)

            for j in range(4):
                temp_grid = np.rot90(temp_arr, k=j)

                # Compute alignment score (difference between grid and contour points)
                interpolator = RegularGridInterpolator((np.unique(region_grid[:, 0]), np.unique(region_grid[:, 1])),
                                                       temp_grid, bounds_error=False, fill_value=None)
                contour_temps = interpolator(region_grid).reshape(
                    np.meshgrid(subgrid_flame_x, subgrid_flame_y)[0].shape)

                if np.all(contour_temps == region_temperature):
                    break_outer = True
                    break  # Break out of the inner loop

            if break_outer:
                break  # Break out of the outer loop

        return region_grid, region_temperature, interpolator

    def _calculate_contour_normal(self, contour_points: np.ndarray) -> np.ndarray:
        """Calculate contour normals using original method."""
        # Step 1: Compute the gradient of the contour points
        dx = np.gradient(contour_points[:, 0])
        dy = np.gradient(contour_points[:, 1])

        # Step 2: Compute the normals
        normals = np.zeros_like(contour_points)
        # Case 1: If the contour is aligned with the x-axis, the normal should be along the y-axis
        for i in range(len(dx)):
            if (dx[i] == 0):  # No change in x-coordinates, thus the normal is along y-axis
                normals[i, 0] = 0  # Normal along the y-axis (positive direction)
                normals[i, 1] = 1  # No change in y for normal direction

            # Case 2: If the contour is aligned with the y-axis, the normal should be along the x-axis
            elif (dy[i] == 0):  # No change in y-coordinates, normal should be along x-axis
                normals[i, 0] = 1  # No change in x for normal direction
                normals[i, 1] = 0  # Normal along the x-axis (positive direction)

            # General case: Calculate the normal by rotating the tangent 90 degrees
            else:
                normals[i, 0] = dy[i]  # Rotate by 90 degrees
                normals[i, 1] = -dx[i]  # Invert the x-component of the tangent

        # Step 3: Normalize the normal vectors
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        return normals

    def _calculate_normal_vector_line(self, normal_vector: np.ndarray, region_grid: np.ndarray):
        """Calculate normal vector line using original method."""
        # Step 1: Determine the spacing to be used for the normal vector
        dx = np.abs(np.unique(region_grid[:, 0])[1] - np.unique(region_grid[:, 0])[0])
        dy = np.abs(np.unique(region_grid[:, 1])[1] - np.unique(region_grid[:, 1])[0])
        t_step = min(dx, dy) / np.linalg.norm(normal_vector)  # Adjust step size for resolution

        # Center point of the array
        center_point = region_grid[region_grid.shape[0] // 2]

        # Step 2: Determine the bounds for the normal vector
        t_min_x = (np.min(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[0] != 0 else -np.inf
        t_max_x = (np.max(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[0] != 0 else np.inf
        t_min_y = (np.min(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[1] != 0 else -np.inf
        t_max_y = (np.max(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[1] != 0 else np.inf

        t_start = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y))
        t_end = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y))

        # Step 3: Generate t_range
        t_range = np.arange(t_start, t_end, t_step / 1e2, dtype=np.float32)

        # Step 4: Generate line points along the normal vector
        x_line_points = np.array(center_point[0] + t_range * normal_vector[0], dtype=np.float32)
        y_line_points = np.array(center_point[1] + t_range * normal_vector[1], dtype=np.float32)
        line_points = np.column_stack((x_line_points, y_line_points))

        # Step 5: Filter line points to ensure they remain within bounds
        min_x, max_x = np.min(region_grid[:, 0]), np.max(region_grid[:, 0])
        min_y, max_y = np.min(region_grid[:, 1]), np.max(region_grid[:, 1])
        line_points_filtered = line_points[
            (line_points[:, 0] >= min_x) & (line_points[:, 0] <= max_x) &
            (line_points[:, 1] >= min_y) & (line_points[:, 1] <= max_y)
            ]

        return line_points_filtered

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
                # Convert from g/(cm^3*s) to kg/(m^3*s)
                row_data *= 1000
                total_consumption += np.sum(np.abs(row_data)) * dx * dy

            # Calculate burning velocity
            rho_reactant = cg["boxlib", f"rho_{transport_species}"].to_value()[-1, 0, 0] * 1000  # Convert to kg/m^3
            domain_height = (dataset.domain_right_edge[1].to_value() - dataset.domain_left_edge[1].to_value()) / 100

            burning_velocity = total_consumption / (rho_reactant * domain_height)

            return total_consumption, burning_velocity

        except Exception as e:
            raise FlameAnalysisError("consumption_rate", str(e))

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

    def analyze_flame_properties(self, dataset: Any, data: FieldData, extraction_location: float = None) -> FlameProperties:
        """Complete flame analysis matching original flame_geometry function."""
        # Find flame position from 1D data
        flame_idx, flame_pos = self.find_wave_position(data, WaveType.FLAME)

        properties = FlameProperties(position=flame_pos, index=flame_idx)

        # Extract thermodynamic state at flame location from 1D data
        try:
            from ..core.domain import ThermodynamicState
            
            # Extract state directly from 1D data at flame index
            if flame_idx is not None and flame_idx < len(data.coordinates):
                temp = data.temperature[flame_idx]
                pressure = data.pressure[flame_idx]
                density = data.density[flame_idx]
                
                # Calculate sound speed from ideal gas relations
                # c = sqrt(gamma * R * T), assuming gamma = 1.4 and R = pressure/(density*T)
                sound_speed = np.sqrt(1.4 * pressure / density)
                
                properties.thermodynamic_state = ThermodynamicState(
                    temperature=temp,
                    pressure=pressure, 
                    density=density,
                    sound_speed=sound_speed
                )
                
                print(f"  Flame thermodynamic state: T={temp:.1f}K, P={pressure:.0f}Pa, rho={density:.3f}kg/m^3")
            else:
                print("  Could not extract thermodynamic state: invalid flame index")
                
        except Exception as e:
            print(f"  Thermodynamic state extraction failed: {e}")

        # Extract 2D flame contour using known flame position for local search
        try:
            contour_points = self.extract_flame_contour(dataset, flame_pos)
            if len(contour_points) > 0:
                # Sort contour points
                sorted_points, segments, surface_length = self.sort_contour_by_nearest_neighbors(contour_points, dataset)

                properties.surface_length = surface_length
                properties.contour_points = sorted_points

                # Calculate flame thickness if contour available
                try:
                    # Use the y-coordinate from the 1D extraction location for thickness calculation
                    if extraction_location is not None:
                        center_location = extraction_location  # This is the y-coordinate of the 1D ray
                    else:
                        # Fallback: use flame x-position (not ideal, but maintains backward compatibility)
                        center_location = flame_pos
                        print("Warning: No extraction location provided for thickness calculation, using flame position as fallback")
                    
                    thickness_result, plotting_data = self.calculate_flame_thickness(dataset, sorted_points, center_location)
                    properties.thickness = thickness_result
                    
                    # Store plotting data for later visualization
                    if plotting_data is not None:
                        properties.region_grid = plotting_data.get('region_grid')
                        properties.region_temperature = plotting_data.get('region_temperature')
                        properties.normal_line = plotting_data.get('normal_line')
                        properties.interpolated_temperatures = plotting_data.get('interpolated_temperatures')
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


def create_flame_analyzer(flame_temperature: float = 2500.0, **kwargs) -> FlameAnalyzer:
    """Factory for flame analyzers."""
    return PeleFlameAnalyzer(flame_temperature, **kwargs)