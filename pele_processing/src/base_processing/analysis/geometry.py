"""
2D geometry analysis for the Pele processing system.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ..core.exceptions import AnalysisError


class GeometryAnalyzer:
    """2D geometry analysis for flame fronts."""

    def __init__(self, flame_temperature: float = 2500.0):
        self.flame_temperature = flame_temperature

    def extract_flame_contour(self, dataset: Any) -> np.ndarray:
        """Extract flame contour from 2D dataset."""
        try:
            # Try YT contour extraction first
            if hasattr(dataset, 'all_data'):
                contours = dataset.all_data().extract_isocontours("Temp", self.flame_temperature)
                return self._process_yt_contours(contours)
            else:
                return self._manual_contour_extraction(dataset)
        except Exception as e:
            raise AnalysisError(f"Contour extraction failed: {e}")

    def calculate_surface_length(self, contour_points: np.ndarray) -> float:
        """Calculate total surface length."""
        if len(contour_points) < 2:
            return 0.0

        segments = contour_points[1:] - contour_points[:-1]
        segment_lengths = np.sqrt(np.sum(segments ** 2, axis=1))
        return np.sum(segment_lengths)

    def calculate_flame_curvature(self, contour_points: np.ndarray) -> np.ndarray:
        """Calculate local curvature along flame front."""
        if len(contour_points) < 3:
            return np.array([])

        # Numerical curvature calculation
        dx = np.gradient(contour_points[:, 0])
        dy = np.gradient(contour_points[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        curvature = (dx * d2y - dy * d2x) / (dx ** 2 + dy ** 2) ** (3 / 2)
        return curvature

    def sort_contour_points(self, points: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Sort contour points into connected segments."""
        from scipy.spatial import cKDTree

        if len(points) < 2:
            return points, [points]

        tree = cKDTree(points)
        sorted_points = []
        segments = []
        remaining = set(range(len(points)))

        while remaining:
            # Start new segment
            start_idx = remaining.pop()
            segment = [points[start_idx]]
            current_idx = start_idx

            # Build segment by nearest neighbors
            while remaining:
                distances, indices = tree.query(points[current_idx], k=len(points))

                next_idx = None
                for idx in indices[1:]:  # Skip self
                    if idx in remaining:
                        next_idx = idx
                        break

                if next_idx is None or np.linalg.norm(points[next_idx] - points[current_idx]) > 50e-6:
                    break

                segment.append(points[next_idx])
                remaining.remove(next_idx)
                current_idx = next_idx

            segments.append(np.array(segment))
            sorted_points.extend(segment)

        return np.array(sorted_points), segments

    def _process_yt_contours(self, contours: np.ndarray) -> np.ndarray:
        """Process YT contour output."""
        if len(contours) == 0:
            return np.array([])

        # Convert cm to m
        contours_m = contours / 100.0

        # Filter out boundary points
        y_range = contours_m[:, 1].max() - contours_m[:, 1].min()
        buffer = 0.0075 * y_range
        y_min, y_max = contours_m[:, 1].min() + buffer, contours_m[:, 1].max() - buffer

        valid_mask = (contours_m[:, 1] >= y_min) & (contours_m[:, 1] <= y_max)
        return contours_m[valid_mask]

    def _manual_contour_extraction(self, dataset: Any) -> np.ndarray:
        """Manual contour extraction using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.tri import Triangulation

            # Extract highest level data
            max_level = dataset.index.max_level
            x_coords, y_coords, temperatures = [], [], []

            for grid in dataset.index.grids:
                if grid.Level == max_level:
                    x_coords.extend(grid["boxlib", "x"].to_value().flatten())
                    y_coords.extend(grid["boxlib", "y"].to_value().flatten())
                    temperatures.extend(grid["Temp"].flatten())

            # Create triangulation and contour
            tri = Triangulation(np.array(x_coords), np.array(y_coords))
            contour = plt.tricontour(tri, np.array(temperatures), levels=[self.flame_temperature])

            contour_points = []
            for collection in contour.collections:
                for path in collection.get_paths():
                    contour_points.extend(path.vertices)

            return np.array(contour_points) / 100.0 if contour_points else np.array([])

        except Exception as e:
            raise AnalysisError(f"Manual contour extraction failed: {e}")


class FlameGeometryAnalyzer:
    """Complete flame geometry analysis."""

    def __init__(self, geometry_analyzer: GeometryAnalyzer):
        self.geometry = geometry_analyzer

    def analyze_complete_geometry(self, dataset: Any, center_location: float) -> Dict[str, Any]:
        """Complete 2D flame geometry analysis."""
        results = {}

        try:
            # Extract contour
            contour_points = self.geometry.extract_flame_contour(dataset)
            if len(contour_points) == 0:
                return {'error': 'No contour found'}

            # Sort points
            sorted_points, segments = self.geometry.sort_contour_points(contour_points)

            # Calculate metrics
            results['surface_length'] = self.geometry.calculate_surface_length(sorted_points)
            results['curvature'] = self.geometry.calculate_flame_curvature(sorted_points)
            results['contour_points'] = sorted_points
            results['segments'] = segments

            return results

        except Exception as e:
            return {'error': str(e)}


def create_geometry_analyzer(flame_temperature: float = 2500.0) -> GeometryAnalyzer:
    """Factory for geometry analyzers."""
    return GeometryAnalyzer(flame_temperature)