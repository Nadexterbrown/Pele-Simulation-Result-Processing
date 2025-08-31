"""
Specialized visualization techniques for Pele processing.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from ..core.exceptions import PlotGenerationError


class SchlierenVisualizer:
    """Generate Schlieren-like visualizations from density gradients."""

    def __init__(self, figure_size: tuple = (10, 8), dpi: int = 150):
        self.figure_size = figure_size
        self.dpi = dpi

    def generate_schlieren(self, dataset, output_path: Path,
                           field: str = 'density', **kwargs) -> None:
        """Generate Schlieren visualization."""
        try:
            # Extract 2D data
            x_coords, y_coords, field_values = self._extract_2d_data(dataset, field)

            # Create regular grid
            x_grid, y_grid, field_grid = self._create_regular_grid(x_coords, y_coords, field_values)

            # Calculate gradients
            grad_x, grad_y = np.gradient(field_grid)
            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Apply Schlieren knife-edge effect
            schlieren_image = self._apply_knife_edge(gradient_magnitude, **kwargs)

            # Create plot
            self._plot_schlieren(x_grid, y_grid, schlieren_image, output_path, field)

        except Exception as e:
            raise PlotGenerationError("schlieren", str(e))

    def _extract_2d_data(self, dataset, field: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 2D field data from dataset."""
        max_level = dataset.index.max_level
        x_coords, y_coords, field_values = [], [], []

        for grid in dataset.index.grids:
            if grid.Level == max_level:
                x_coords.extend(grid["boxlib", "x"].to_value().flatten())
                y_coords.extend(grid["boxlib", "y"].to_value().flatten())
                field_values.extend(grid[field].flatten())

        return np.array(x_coords), np.array(y_coords), np.array(field_values)

    def _create_regular_grid(self, x_coords: np.ndarray, y_coords: np.ndarray,
                             field_values: np.ndarray, resolution: int = 200) -> Tuple:
        """Create regular grid from scattered data."""
        from scipy.interpolate import griddata

        # Create regular grid
        xi = np.linspace(x_coords.min(), x_coords.max(), resolution)
        yi = np.linspace(y_coords.min(), y_coords.max(), resolution // 2)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate to regular grid
        field_grid = griddata((x_coords, y_coords), field_values,
                              (xi_grid, yi_grid), method='linear')

        return xi_grid, yi_grid, field_grid

    def _apply_knife_edge(self, gradient_magnitude: np.ndarray,
                          sensitivity: float = 1.0, cutoff_angle: float = 0.0) -> np.ndarray:
        """Apply Schlieren knife-edge effect."""
        # Normalize gradient magnitude
        grad_norm = gradient_magnitude / np.nanmax(gradient_magnitude)

        # Apply sensitivity and knife-edge cutoff
        schlieren = np.tanh(sensitivity * grad_norm)

        # Apply directional cutoff if specified
        if cutoff_angle > 0:
            schlieren = np.where(grad_norm > cutoff_angle, schlieren, 0)

        return schlieren

    def _plot_schlieren(self, x_grid: np.ndarray, y_grid: np.ndarray,
                        schlieren_image: np.ndarray, output_path: Path, field: str) -> None:
        """Create Schlieren plot."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        im = ax.imshow(schlieren_image, extent=[x_grid.min(), x_grid.max(),
                                                y_grid.min(), y_grid.max()],
                       cmap='gray', origin='lower', aspect='auto')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Schlieren ({field})')

        plt.colorbar(im, ax=ax, label='Normalized Gradient')
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