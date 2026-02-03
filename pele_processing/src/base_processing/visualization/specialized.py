"""
Specialized visualization techniques for Pele processing.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

import matplotlib.pyplot as plt

from ..core.exceptions import PlotGenerationError


class SchlierenMode(Enum):
    """Schlieren knife-edge orientation.

    X: Vertical knife edge — sensitive to density gradients in the x-direction (dρ/dx).
       Highlights features like the leading shock/detonation front.
    Y: Horizontal knife edge — sensitive to density gradients in the y-direction (dρ/dy).
       Highlights transverse waves and structures perpendicular to the propagation direction.
    COMBINED: Uses the full gradient magnitude sqrt((dρ/dx)² + (dρ/dy)²).
       Shows all density gradient features without directional selectivity.
    """
    X = "x"
    Y = "y"
    COMBINED = "combined"


class SchlierenVisualizer:
    """Generate Schlieren-like visualizations from density gradients.

    Uses YT covering grid extraction (same approach as YTFieldPlotter) for
    consistent, high-resolution data on the AMR grid.
    """

    def __init__(self, figure_size: tuple = (10, 8), dpi: int = 150):
        self.figure_size = figure_size
        self.dpi = dpi

    def generate_schlieren(self, dataset, output_path: Path,
                           field: str = 'density',
                           mode: SchlierenMode = SchlierenMode.COMBINED,
                           center_point=None,
                           forward_bound: float = None,
                           backward_bound: float = None,
                           axis: str = 'z',
                           normal_axis: str = 'x',
                           enable_y_zoom: bool = False,
                           y_min: float = 0.0,
                           y_max: float = 0.001,
                           **kwargs) -> None:
        """Generate Schlieren visualization using covering grid extraction.

        Args:
            dataset: YT dataset to visualize.
            output_path: Path to save the output image.
            field: PeleC field name to compute gradients from (default: 'density').
            mode: SchlierenMode controlling the knife-edge orientation.
                SchlierenMode.X — vertical knife edge, highlights dρ/dx.
                SchlierenMode.Y — horizontal knife edge, highlights dρ/dy
                    (ideal for transverse wave tracking).
                SchlierenMode.COMBINED — full gradient magnitude (default).
            center_point: [x, y, z] center for localized extraction (dataset units).
            forward_bound: Distance forward of center along normal_axis (dataset units).
            backward_bound: Distance backward of center along normal_axis (dataset units).
            axis: Slice axis ('x', 'y', or 'z') — normal to the plotting plane.
            normal_axis: Axis along which forward/backward bounds are applied.
            enable_y_zoom: Enable y-direction zooming.
            y_min: Minimum y-coordinate (dataset units) for zoom.
            y_max: Maximum y-coordinate (dataset units) for zoom.
            **kwargs: sensitivity (float), cutoff_angle (float).
        """
        try:
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            axis_idx = axis_map[axis]
            normal_axis_idx = axis_map[normal_axis]

            domain_left = dataset.domain_left_edge.to_ndarray()
            domain_right = dataset.domain_right_edge.to_ndarray()

            left_edge = domain_left.copy()
            right_edge = domain_right.copy()

            # Apply localized bounds along normal axis
            if center_point is not None and forward_bound is not None and backward_bound is not None:
                left_edge[normal_axis_idx] = center_point[normal_axis_idx] - backward_bound
                right_edge[normal_axis_idx] = center_point[normal_axis_idx] + forward_bound
                left_edge[normal_axis_idx] = max(left_edge[normal_axis_idx], domain_left[normal_axis_idx])
                right_edge[normal_axis_idx] = min(right_edge[normal_axis_idx], domain_right[normal_axis_idx])

            # For non-slice, non-normal axes: use full domain or y-zoom
            for i in range(3):
                if i != normal_axis_idx and i != axis_idx:
                    if i == 1 and enable_y_zoom:
                        left_edge[i] = y_min
                        right_edge[i] = y_max
                    else:
                        left_edge[i] = domain_left[i]
                        right_edge[i] = domain_right[i]

            # Single cell in slice direction
            dx = dataset.index.get_smallest_dx().to_value()
            left_edge[axis_idx] = domain_left[axis_idx]
            right_edge[axis_idx] = domain_left[axis_idx] + dx

            # Calculate covering grid dimensions at max refinement
            max_level = dataset.index.max_level
            dims = [max(1, int((right_edge[i] - left_edge[i]) / dx)) for i in range(3)]
            dims[axis_idx] = 1

            # Create covering grid — same approach as YTFieldPlotter
            cg = dataset.covering_grid(
                level=max_level,
                left_edge=left_edge,
                dims=dims,
                fields=[('boxlib', 'x'), ('boxlib', 'y'), ('boxlib', 'z'), field]
            )

            # Extract and squeeze out singleton slice dimension → 2D arrays
            x_2d = np.squeeze(cg['boxlib', 'x'].to_ndarray())
            y_2d = np.squeeze(cg['boxlib', 'y'].to_ndarray())
            field_2d = np.squeeze(cg[field].to_ndarray())

            # Select plotting coordinates based on slice axis
            if axis_idx == 2:  # z-normal → plot x-y
                plot_x, plot_y = x_2d, y_2d
                xlabel, ylabel = 'X', 'Y'
            elif axis_idx == 1:  # y-normal → plot x-z
                z_2d = np.squeeze(cg['boxlib', 'z'].to_ndarray())
                plot_x, plot_y = x_2d, z_2d
                xlabel, ylabel = 'X', 'Z'
            else:  # x-normal → plot y-z
                z_2d = np.squeeze(cg['boxlib', 'z'].to_ndarray())
                plot_x, plot_y = y_2d, z_2d
                xlabel, ylabel = 'Y', 'Z'

            # Compute gradients using uniform grid spacing (dx is same in all directions
            # for the covering grid at max refinement level)
            # grad_0 = derivative along array axis 0, grad_1 = along axis 1
            grad_0, grad_1 = np.gradient(field_2d, dx, dx)

            # Select gradient component based on mode
            # For z-normal: axis 0 = x, axis 1 = y, so grad_0 = d/dx, grad_1 = d/dy
            if mode == SchlierenMode.X:
                gradient_field = np.abs(grad_0)
            elif mode == SchlierenMode.Y:
                gradient_field = np.abs(grad_1)
            else:
                gradient_field = np.sqrt(grad_0 ** 2 + grad_1 ** 2)

            # Apply knife-edge effect
            schlieren_2d = self._apply_knife_edge(gradient_field, **kwargs)

            # Flatten for tricontourf (same as YTFieldPlotter)
            self._plot_schlieren(
                plot_x.flatten(), plot_y.flatten(), schlieren_2d.flatten(),
                output_path, field, mode, xlabel, ylabel
            )

        except Exception as e:
            raise PlotGenerationError("schlieren", str(e))

    def _apply_knife_edge(self, gradient_magnitude: np.ndarray,
                          sensitivity: float = 1.0, cutoff_angle: float = 0.0) -> np.ndarray:
        """Apply Schlieren knife-edge effect."""
        max_val = np.nanmax(gradient_magnitude)
        if max_val == 0 or np.isnan(max_val):
            return np.zeros_like(gradient_magnitude)

        grad_norm = gradient_magnitude / max_val
        schlieren = np.tanh(sensitivity * grad_norm)

        if cutoff_angle > 0:
            schlieren = np.where(grad_norm > cutoff_angle, schlieren, 0)

        return schlieren

    def _plot_schlieren(self, plot_x: np.ndarray, plot_y: np.ndarray,
                        schlieren_data: np.ndarray, output_path: Path,
                        field: str, mode: SchlierenMode,
                        xlabel: str = 'X', ylabel: str = 'Y') -> None:
        """Create Schlieren plot using tricontourf (same as YTFieldPlotter)."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        contour = ax.tricontourf(plot_x, plot_y, schlieren_data,
                                 levels=50, cmap='gray')

        ax.set_xlabel(f'{xlabel} (cm)')
        ax.set_ylabel(f'{ylabel} (cm)')
        ax.set_aspect('equal')

        mode_labels = {
            SchlierenMode.X: 'Vertical Knife Edge — d/dx',
            SchlierenMode.Y: 'Horizontal Knife Edge — d/dy',
            SchlierenMode.COMBINED: 'Combined |∇|',
        }
        ax.set_title(f'Schlieren ({field}) — {mode_labels[mode]}')

        plt.colorbar(contour, ax=ax, label='Normalized Gradient')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)


class StreamlineVisualizer:
    """Generate streamline visualizations."""

    def __init__(self, figure_size: tuple = (10, 8), dpi: int = 150):
        self.figure_size = figure_size
        self.dpi = dpi

    def generate_streamlines(self, dataset, output_path: Path,
                             velocity_fields: Tuple[str, str] = ('x_velocity', 'y_velocity'),
                             **kwargs) -> None:
        """Generate streamline visualization."""
        try:
            # Extract velocity field data
            x_coords, y_coords, u_vel, v_vel = self._extract_velocity_data(dataset, velocity_fields)

            # Create regular grid
            x_grid, y_grid, u_grid, v_grid = self._create_velocity_grid(x_coords, y_coords, u_vel, v_vel)

            # Create streamline plot
            self._plot_streamlines(x_grid, y_grid, u_grid, v_grid, output_path, **kwargs)

        except Exception as e:
            raise PlotGenerationError("streamlines", str(e))

    def _extract_velocity_data(self, dataset, velocity_fields: Tuple[str, str]) -> Tuple:
        """Extract velocity field data."""
        max_level = dataset.index.max_level
        x_coords, y_coords, u_vel, v_vel = [], [], [], []

        u_field, v_field = velocity_fields

        for grid in dataset.index.grids:
            if grid.Level == max_level:
                x_coords.extend(grid["boxlib", "x"].to_value().flatten())
                y_coords.extend(grid["boxlib", "y"].to_value().flatten())
                u_vel.extend(grid[u_field].flatten())
                v_vel.extend(grid[v_field].flatten())

        return np.array(x_coords), np.array(y_coords), np.array(u_vel), np.array(v_vel)

    def _create_velocity_grid(self, x_coords: np.ndarray, y_coords: np.ndarray,
                              u_vel: np.ndarray, v_vel: np.ndarray) -> Tuple:
        """Create regular velocity grids."""
        from scipy.interpolate import griddata

        # Create regular grid
        xi = np.linspace(x_coords.min(), x_coords.max(), 100)
        yi = np.linspace(y_coords.min(), y_coords.max(), 50)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate velocities
        u_grid = griddata((x_coords, y_coords), u_vel, (xi_grid, yi_grid), method='linear')
        v_grid = griddata((x_coords, y_coords), v_vel, (xi_grid, yi_grid), method='linear')

        return xi_grid, yi_grid, u_grid, v_grid

    def _plot_streamlines(self, x_grid: np.ndarray, y_grid: np.ndarray,
                          u_grid: np.ndarray, v_grid: np.ndarray,
                          output_path: Path, **kwargs) -> None:
        """Create streamline plot."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        # Calculate velocity magnitude for color
        velocity_mag = np.sqrt(u_grid ** 2 + v_grid ** 2)

        # Create streamlines
        streams = ax.streamplot(x_grid, y_grid, u_grid, v_grid,
                                color=velocity_mag, cmap='viridis',
                                density=kwargs.get('density', 1.0),
                                linewidth=kwargs.get('linewidth', 1.0),
                                arrowsize=kwargs.get('arrowsize', 1.0))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Velocity Streamlines')

        plt.colorbar(streams.lines, ax=ax, label='Velocity Magnitude (m/s)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)


class ContourVisualizer:
    """Advanced contour visualizations."""

    def __init__(self, figure_size: tuple = (10, 8), dpi: int = 150):
        self.figure_size = figure_size
        self.dpi = dpi

    def generate_multi_contour(self, dataset, output_path: Path,
                               fields: list, iso_values: dict = None, **kwargs) -> None:
        """Generate multi-field contour plot."""
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            colors = plt.cm.tab10(np.linspace(0, 1, len(fields)))

            for i, field in enumerate(fields):
                # Extract and process field data
                x_coords, y_coords, field_values = self._extract_2d_data(dataset, field)
                x_grid, y_grid, field_grid = self._create_regular_grid(x_coords, y_coords, field_values)

                # Determine contour level
                if iso_values and field in iso_values:
                    levels = [iso_values[field]]
                else:
                    levels = [np.nanmean(field_values)]

                # Plot contour
                cs = ax.contour(x_grid, y_grid, field_grid, levels=levels,
                                colors=[colors[i]], linewidths=2, alpha=0.8)
                ax.clabel(cs, inline=True, fontsize=8, fmt=f'{field}: %.2e')

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Multi-Field Contours')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            raise PlotGenerationError("multi_contour", str(e))

    def _extract_2d_data(self, dataset, field: str):
        """Extract 2D field data."""
        max_level = dataset.index.max_level
        x_coords, y_coords, field_values = [], [], []

        for grid in dataset.index.grids:
            if grid.Level == max_level:
                x_coords.extend(grid["boxlib", "x"].to_value().flatten())
                y_coords.extend(grid["boxlib", "y"].to_value().flatten())
                field_values.extend(grid[field].flatten())

        return np.array(x_coords), np.array(y_coords), np.array(field_values)

    def _create_regular_grid(self, x_coords, y_coords, field_values, resolution=200):
        """Create regular grid from scattered data."""
        from scipy.interpolate import griddata

        xi = np.linspace(x_coords.min(), x_coords.max(), resolution)
        yi = np.linspace(y_coords.min(), y_coords.max(), resolution // 2)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        field_grid = griddata((x_coords, y_coords), field_values,
                              (xi_grid, yi_grid), method='linear')

        return xi_grid, yi_grid, field_grid