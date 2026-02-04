"""
Specialized visualization techniques for Pele processing.
"""
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from enum import Enum

import matplotlib.pyplot as plt
import yt

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


class SchlierenField(Enum):
    """Base field for Schlieren computation."""
    DENSITY = "density"
    PRESSURE = "pressure"


def register_schlieren_fields(ds, base_field: str = "density") -> None:
    """Register gradient fields with a YT dataset for Schlieren visualization.

    Uses YT's add_gradient_fields() which computes gradients using second-order
    centered differences with proper AMR handling. Also creates absolute value
    versions for log-scale visualization.

    Args:
        ds: YT dataset to register fields with.
        base_field: The base field name to compute gradients from ("density" or "pressure").
    """
    field_tuple = ("boxlib", base_field)

    # Check if the field exists in the dataset
    if field_tuple not in ds.field_list and field_tuple not in ds.derived_field_list:
        raise ValueError(f"Field {field_tuple} not found in dataset. "
                        f"Available fields: {[f for f in ds.field_list if f[0] == 'boxlib']}")

    # Check if gradient fields already exist (avoid calling add_gradient_fields twice)
    gradient_field_x = ("boxlib", f"{base_field}_gradient_x")
    if gradient_field_x not in ds.derived_field_list:
        # Use YT's built-in gradient field generation (second-order centered differences)
        ds.add_gradient_fields(field_tuple)

    # Create absolute value versions for log-scale visualization
    abs_field_x = ("boxlib", f"{base_field}_gradient_x_abs")
    if abs_field_x not in ds.derived_field_list:
        # Get units from the gradient fields
        grad_units = str(ds.field_info["boxlib", f"{base_field}_gradient_x"].units)

        # Register abs versions
        ds.add_field(
            ("boxlib", f"{base_field}_gradient_x_abs"),
            function=lambda field, data, bf=base_field: np.abs(data["boxlib", f"{bf}_gradient_x"]),
            sampling_type="cell",
            units=grad_units,
            force_override=True,
        )
        ds.add_field(
            ("boxlib", f"{base_field}_gradient_y_abs"),
            function=lambda field, data, bf=base_field: np.abs(data["boxlib", f"{bf}_gradient_y"]),
            sampling_type="cell",
            units=grad_units,
            force_override=True,
        )


def get_schlieren_field_name(base_field: SchlierenField, mode: SchlierenMode, use_abs: bool = False) -> str:
    """Get the YT gradient field name for a Schlieren configuration.

    Args:
        base_field: The base field (DENSITY or PRESSURE).
        mode: The Schlieren mode (X, Y, or COMBINED).
        use_abs: If True, return the absolute value field name (for log scale).

    Returns:
        The field name string, e.g., "pressure_gradient_y" or "pressure_gradient_y_abs".
    """
    if mode == SchlierenMode.COMBINED:
        # Magnitude is already positive, no abs needed
        return f"{base_field.value}_gradient_magnitude"
    else:
        suffix = "_abs" if use_abs else ""
        return f"{base_field.value}_gradient_{mode.value}{suffix}"


class SchlierenVisualizer:
    """Generate Schlieren-like visualizations using YT derived fields.

    Uses YT's native SlicePlot with derived gradient fields for proper AMR handling.
    This avoids covering grid interpolation artifacts and provides better resolution.
    """

    def __init__(self, figure_size: tuple = (10, 8), dpi: int = 300):
        self.figure_size = figure_size
        self.dpi = dpi
        self._fields_registered = False

    def generate_schlieren(
        self,
        dataset,
        output_path: Path,
        base_field: Union[SchlierenField, str] = SchlierenField.DENSITY,
        mode: SchlierenMode = SchlierenMode.COMBINED,
        center_point=None,
        forward_bound: float = None,
        backward_bound: float = None,
        axis: str = 'z',
        normal_axis: str = 'x',
        enable_y_zoom: bool = False,
        y_min: float = 0.0,
        y_max: float = 0.001,
        zmin: float = None,
        zmax: float = None,
        cmap: str = 'gray',
        log_scale: bool = False,
        interpolation: str = None,
        buff_size: int = 2048,
        normalize_to_shock: bool = False,
        shock_sample_width: float = 0.05,
        **kwargs
    ) -> None:
        """Generate Schlieren visualization using YT derived fields and SlicePlot.

        Args:
            dataset: YT dataset to visualize.
            output_path: Path to save the output image.
            base_field: Base field for gradient computation (SchlierenField.DENSITY
                or SchlierenField.PRESSURE, or string 'density'/'pressure').
            mode: SchlierenMode controlling the knife-edge orientation.
                SchlierenMode.X — vertical knife edge, highlights d/dx.
                SchlierenMode.Y — horizontal knife edge, highlights d/dy
                    (ideal for transverse wave tracking).
                SchlierenMode.COMBINED — full gradient magnitude (default).
            center_point: [x, y, z] center for localized view (dataset units).
            forward_bound: Distance forward of center along normal_axis (dataset units).
            backward_bound: Distance backward of center along normal_axis (dataset units).
            axis: Slice axis ('x', 'y', or 'z') — normal to the plotting plane.
            normal_axis: Axis along which forward/backward bounds are applied.
            enable_y_zoom: Enable y-direction zooming.
            y_min: Minimum y-coordinate (dataset units) for zoom.
            y_max: Maximum y-coordinate (dataset units) for zoom.
            zmin: Minimum value for colorbar (disables per-frame normalization).
            zmax: Maximum value for colorbar (disables per-frame normalization).
            cmap: Colormap name (default: 'gray').
            log_scale: Use logarithmic scaling for colorbar (default: False).
            interpolation: Image interpolation method (e.g., 'nearest', 'bilinear',
                'bicubic', 'gaussian', 'lanczos'). None uses default (nearest).
            buff_size: Resolution of the fixed resolution buffer (default: 2048).
                Higher values give more detail for interpolation to work with.
            normalize_to_shock: If True, normalize colorbar based on gradient values
                at the shock front (center_point) rather than the entire domain.
            shock_sample_width: Width (in cm) of the region around shock front to sample
                for normalization (default: 0.05 cm = 0.5 mm).
            **kwargs: Additional arguments (unused, for backwards compatibility).
        """
        try:
            # Convert string to enum if needed
            if isinstance(base_field, str):
                base_field = SchlierenField(base_field)

            # Register derived fields for the requested base field
            register_schlieren_fields(dataset, base_field.value)

            # Get the appropriate derived field name
            # Use abs fields when log_scale is enabled (can't log negative values)
            field_name = get_schlieren_field_name(base_field, mode, use_abs=log_scale)
            field_tuple = ("boxlib", field_name)

            # Get domain bounds
            domain_left = dataset.domain_left_edge.to_value()
            domain_right = dataset.domain_right_edge.to_value()

            # Normalize to shock front if requested
            if normalize_to_shock and center_point is not None and zmin is None and zmax is None:
                # Sample gradient in a small region around the shock front
                shock_x = center_point[0]
                sample_left = [
                    shock_x - shock_sample_width,
                    domain_left[1],
                    domain_left[2]
                ]
                sample_right = [
                    shock_x + shock_sample_width,
                    domain_right[1],
                    domain_right[2]
                ]

                # Create a region around the shock front
                shock_region = dataset.region(
                    center=center_point,
                    left_edge=sample_left,
                    right_edge=sample_right
                )

                # Get gradient values in the shock region
                try:
                    grad_data = shock_region[field_tuple].to_ndarray()
                    if len(grad_data) > 0:
                        if log_scale:
                            # For log scale (abs values), use max
                            zmax = float(np.nanmax(grad_data))
                            zmin = float(np.nanmin(grad_data[grad_data > 0])) if np.any(grad_data > 0) else zmax / 1000
                        else:
                            # For linear scale (signed), use symmetric around zero
                            max_abs = float(np.nanmax(np.abs(grad_data)))
                            zmin = -max_abs
                            zmax = max_abs
                except Exception as e:
                    # If sampling fails, fall back to auto-scaling
                    pass

            # Calculate plot center and width
            plot_center = None
            plot_width = None

            if center_point is not None and forward_bound is not None and backward_bound is not None:
                # Calculate width along normal axis
                width_normal = forward_bound + backward_bound

                # For z-normal slice: plotting x-y plane
                if axis == 'z':
                    if normal_axis == 'x':
                        # View center in x (offset from center_point based on bounds)
                        view_center_x = center_point[0] + (forward_bound - backward_bound) / 2

                        # Width in y (full domain or y_zoom)
                        if enable_y_zoom:
                            width_perp = y_max - y_min
                            center_y = (y_min + y_max) / 2
                        else:
                            width_perp = domain_right[1] - domain_left[1]
                            center_y = (domain_left[1] + domain_right[1]) / 2

                        plot_center = [view_center_x, center_y, (domain_left[2] + domain_right[2]) / 2]
                        plot_width = ((width_normal, 'cm'), (width_perp, 'cm'))

                    elif normal_axis == 'y':
                        # View center in y (offset from center_point based on bounds)
                        view_center_y = center_point[1] + (forward_bound - backward_bound) / 2

                        # Width in x (full domain)
                        width_perp = domain_right[0] - domain_left[0]
                        center_x = (domain_left[0] + domain_right[0]) / 2

                        plot_center = [center_x, view_center_y, (domain_left[2] + domain_right[2]) / 2]
                        plot_width = ((width_perp, 'cm'), (width_normal, 'cm'))

            elif enable_y_zoom:
                # Only y-zoom without x bounds
                width_x = domain_right[0] - domain_left[0]
                width_y = y_max - y_min
                center_x = (domain_left[0] + domain_right[0]) / 2
                center_y = (y_min + y_max) / 2

                plot_center = [center_x, center_y, (domain_left[2] + domain_right[2]) / 2]
                plot_width = ((width_x, 'cm'), (width_y, 'cm'))

            # Create SlicePlot with center and width
            # Use origin='native' to show absolute domain coordinates instead of centered
            if plot_center is not None and plot_width is not None:
                slc = yt.SlicePlot(dataset, axis, field_tuple, center=plot_center, width=plot_width, origin='native')
            else:
                slc = yt.SlicePlot(dataset, axis, field_tuple, origin='native')

            # Set buffer size for high-resolution output (needed for interpolation to work)
            slc.set_buff_size(buff_size)

            # Set colormap and scaling
            slc.set_cmap(field_tuple, cmap)
            slc.set_log(field_tuple, log_scale)

            # Set color limits if provided (disables per-frame normalization)
            if zmin is not None or zmax is not None:
                slc.set_zlim(field_tuple, zmin=zmin, zmax=zmax)

            # Set figure properties
            slc.set_figure_size(self.figure_size[0])

            # Set title
            mode_labels = {
                SchlierenMode.X: 'Vertical Knife Edge — d/dx',
                SchlierenMode.Y: 'Horizontal Knife Edge — d/dy',
                SchlierenMode.COMBINED: 'Combined |∇|',
            }
            title = f'Schlieren ({base_field.value}) — {mode_labels[mode]}'
            slc.annotate_title(title)

            # Set colorbar label
            slc.set_colorbar_label(field_tuple, 'Gradient Magnitude')

            # Prepare output path
            output_path = Path(output_path)
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Render and save the plot
            slc._setup_plots()

            # Set image interpolation if specified (must be after _setup_plots)
            if interpolation is not None:
                for field_key in slc.plots:
                    slc.plots[field_key].image.set_interpolation(interpolation)

            # Save directly with matplotlib to preserve interpolation setting
            # (YT's save() re-renders and resets interpolation)
            fig = slc.plots[field_tuple].figure
            fig.savefig(str(output_path), dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            raise PlotGenerationError("schlieren", str(e))

    def generate_schlieren_pressure_y(
        self,
        dataset,
        output_path: Path,
        zmin: float = None,
        zmax: float = None,
        **kwargs
    ) -> None:
        """Convenience method for pressure-based Y-direction Schlieren.

        Ideal for tracking transverse waves in detonation simulations.

        Args:
            dataset: YT dataset to visualize.
            output_path: Path to save the output image.
            zmin: Minimum value for colorbar (disables per-frame normalization).
            zmax: Maximum value for colorbar (disables per-frame normalization).
            **kwargs: Additional arguments passed to generate_schlieren.
        """
        self.generate_schlieren(
            dataset=dataset,
            output_path=output_path,
            base_field=SchlierenField.PRESSURE,
            mode=SchlierenMode.Y,
            zmin=zmin,
            zmax=zmax,
            **kwargs
        )


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