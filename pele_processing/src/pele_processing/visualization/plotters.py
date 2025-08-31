"""
Plot generation for the Pele processing system.
"""
from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..core.interfaces import FrameGenerator
from ..core.domain import AnimationFrame
from ..core.exceptions import PlotGenerationError


class StandardPlotter(FrameGenerator):
    """Standard 2D plotting functionality."""

    def __init__(self, figure_size: tuple = (10, 6), dpi: int = 150):
        self.figure_size = figure_size
        self.dpi = dpi
        plt.style.use('default')

    def create_field_plot(self, frame: AnimationFrame) -> None:
        """Create single field plot."""
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            ax.plot(frame.x_data, frame.y_data, 'b-', linewidth=2)
            ax.set_xlabel('Position (m)')
            ax.set_ylabel(frame.field_name)
            ax.grid(True, alpha=0.3)

            # Add markers for flame/shock if available
            if frame.flame_position is not None:
                ax.axvline(frame.flame_position, color='red', linestyle='--',
                           alpha=0.7, label='Flame')
            if frame.shock_position is not None:
                ax.axvline(frame.shock_position, color='orange', linestyle='--',
                           alpha=0.7, label='Shock')

            if frame.flame_position is not None or frame.shock_position is not None:
                ax.legend()

            if frame.timestamp is not None:
                ax.set_title(f'{frame.field_name} - t = {frame.timestamp:.2e}s')

            plt.tight_layout()
            plt.savefig(frame.output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            raise PlotGenerationError("field_plot", str(e))

    def create_multi_field_plot(self, x_data: np.ndarray, field_data: Dict[str, np.ndarray],
                                output_path: Path, **kwargs) -> None:
        """Create plot with multiple y-axes."""
        try:
            fig, ax1 = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            field_names = list(field_data.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(field_names)))

            # First field on primary axis
            if field_names:
                ax1.plot(x_data, field_data[field_names[0]],
                         color=colors[0], label=field_names[0])
                ax1.set_ylabel(field_names[0], color=colors[0])
                ax1.tick_params(axis='y', labelcolor=colors[0])

            axes = [ax1]

            # Additional fields on secondary axes
            for i, (field_name, y_data) in enumerate(list(field_data.items())[1:], 1):
                ax_new = ax1.twinx()

                # Offset secondary axes
                if i > 1:
                    ax_new.spines["right"].set_position(("axes", 1 + 0.1 * (i - 1)))

                ax_new.plot(x_data, y_data, color=colors[i], label=field_name)
                ax_new.set_ylabel(field_name, color=colors[i])
                ax_new.tick_params(axis='y', labelcolor=colors[i])
                axes.append(ax_new)

            ax1.set_xlabel('Position (m)')
            ax1.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            raise PlotGenerationError("multi_field_plot", str(e))

    def create_specialized_plot(self, dataset, plot_type: str, output_path: Path, **kwargs) -> None:
        """Create specialized plots."""
        if plot_type == "contour":
            self._create_contour_plot(dataset, output_path, **kwargs)
        elif plot_type == "vector_field":
            self._create_vector_plot(dataset, output_path, **kwargs)
        else:
            raise PlotGenerationError(plot_type, f"Unknown plot type: {plot_type}")

    def _create_contour_plot(self, dataset, output_path: Path, **kwargs):
        """Create 2D contour plot."""
        field_name = kwargs.get('field', 'Temp')
        levels = kwargs.get('levels', 20)

        try:
            # Extract 2D data at highest refinement
            max_level = dataset.index.max_level
            x_coords, y_coords, field_values = [], [], []

            for grid in dataset.index.grids:
                if grid.Level == max_level:
                    x_coords.extend(grid["boxlib", "x"].to_value().flatten())
                    y_coords.extend(grid["boxlib", "y"].to_value().flatten())
                    field_values.extend(grid[field_name].flatten())

            x_coords, y_coords = np.array(x_coords), np.array(y_coords)
            field_values = np.array(field_values)

            # Create regular grid
            from scipy.interpolate import griddata
            xi = np.linspace(x_coords.min(), x_coords.max(), 200)
            yi = np.linspace(y_coords.min(), y_coords.max(), 100)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            field_grid = griddata((x_coords, y_coords), field_values,
                                  (xi_grid, yi_grid), method='linear')

            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            contour = ax.contourf(xi_grid, yi_grid, field_grid, levels=levels, cmap='hot')
            plt.colorbar(contour, ax=ax, label=field_name)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{field_name} Contour')

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            raise PlotGenerationError("contour", str(e))


class LocalViewPlotter(StandardPlotter):
    """Plotter with local windowing capability."""

    def create_local_view(self, frame: AnimationFrame, center: float,
                          window_size: float) -> None:
        """Create local view around specified center."""
        # Find indices within window
        x_min, x_max = center - window_size / 2, center + window_size / 2
        mask = (frame.x_data >= x_min) & (frame.x_data <= x_max)

        if not np.any(mask):
            raise PlotGenerationError("local_view", "No data points in window")

        # Create local frame
        local_frame = AnimationFrame(
            dataset_basename=frame.dataset_basename,
            field_name=f"Local {frame.field_name}",
            x_data=frame.x_data[mask],
            y_data=frame.y_data[mask],
            output_path=frame.output_path,
            timestamp=frame.timestamp,
            flame_position=frame.flame_position if frame.flame_position and
                                                   x_min <= frame.flame_position <= x_max else None,
            shock_position=frame.shock_position if frame.shock_position and
                                                   x_min <= frame.shock_position <= x_max else None
        )

        self.create_field_plot(local_frame)


class StatisticalPlotter:
    """Plotter for statistical analysis."""

    def __init__(self, figure_size: tuple = (10, 6), dpi: int = 150):
        self.figure_size = figure_size
        self.dpi = dpi

    def plot_time_series(self, times: np.ndarray, values: np.ndarray,
                         label: str, output_path: Path,
                         show_trend: bool = True) -> None:
        """Plot time series data."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        ax.plot(times, values, 'bo-', markersize=4, linewidth=1, label=label)

        if show_trend and len(times) > 2:
            # Add linear trend
            coeffs = np.polyfit(times, values, 1)
            trend = np.poly1d(coeffs)
            ax.plot(times, trend(times), 'r--', alpha=0.7,
                    label=f'Trend: {coeffs[0]:.2e} m/s²')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_velocity_analysis(self, times: np.ndarray, positions: np.ndarray,
                               velocities: np.ndarray, output_path: Path,
                               title: str = "Wave Analysis") -> None:
        """Plot position and velocity analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figure_size[0], 8),
                                       dpi=self.dpi, sharex=True)

        # Position vs time
        ax1.plot(times, positions, 'bo-', markersize=4)
        ax1.set_ylabel('Position (m)')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{title} - Position')

        # Velocity vs time
        ax2.plot(times[:-1], velocities, 'ro-', markersize=4)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'{title} - Velocity')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)


class ComparisonPlotter:
    """Plotter for comparing multiple datasets."""

    def __init__(self, figure_size: tuple = (12, 8), dpi: int = 150):
        self.figure_size = figure_size
        self.dpi = dpi

    def plot_multiple_series(self, data_dict: Dict[str, Dict[str, np.ndarray]],
                             x_key: str, y_key: str, output_path: Path,
                             title: str = "Comparison") -> None:
        """Plot multiple data series."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))

        for i, (series_name, data) in enumerate(data_dict.items()):
            x_data = data[x_key]
            y_data = data[y_key]
            ax.plot(x_data, y_data, color=colors[i], label=series_name,
                    linewidth=2, alpha=0.8)

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

    def plot_parameter_sweep(self, parameter_values: np.ndarray,
                             results: Dict[str, np.ndarray],
                             output_path: Path,
                             parameter_name: str = "Parameter") -> None:
        """Plot results of parameter sweep."""
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        axes = axes.flatten()

        result_keys = list(results.keys())[:4]  # Limit to 4 subplots

        for i, key in enumerate(result_keys):
            axes[i].plot(parameter_values, results[key], 'bo-', markersize=4)
            axes[i].set_xlabel(parameter_name)
            axes[i].set_ylabel(key)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(key)

        # Hide unused subplots
        for i in range(len(result_keys), 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)