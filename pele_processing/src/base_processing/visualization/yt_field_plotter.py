"""
YT Field Plotter - ParaView-like field visualization for YT datasets.

This module provides flexible field plotting capabilities similar to ParaView,
allowing users to visualize arbitrary fields from YT datasets with various
plot types including slices, projections, line plots, and phase plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

from ..core.exceptions import PlotGenerationError


# Field name mapping: pretty name -> PeleC field name
FIELD_NAME_MAP = {
    # Thermodynamic fields
    'Temperature': 'Temp',
    'Density': 'density',
    'Pressure': 'pressure',

    # Velocity fields
    'X Velocity': 'x_velocity',
    'Y Velocity': 'y_velocity',
    'Z Velocity': 'z_velocity',
    'Velocity Magnitude': 'velocity_magnitude',

    # Energy fields
    'Total Energy': 'rho_E',
    'Internal Energy': 'rho_e',

    # Reaction fields
    'Heat Release Rate': 'heatRelease',

    # Derived fields
    'Mach Number': 'MachNumber',
    'Sound Speed': 'soundspeed',

}

# Reverse mapping: PeleC name -> pretty name
PRETTY_NAME_MAP = {v: k for k, v in FIELD_NAME_MAP.items()}

# Field units and symbols mapping with conversion factors
# 'conversion': multiply data by this factor to get target units
FIELD_UNITS_MAP = {
    'Temp': {'symbol': 'T', 'units': 'K', 'conversion': 1.0},
    'Temperature': {'symbol': 'T', 'units': 'K', 'conversion': 1.0},
    'density': {'symbol': 'ρ', 'units': 'kg/m³', 'conversion': 1000.0},  # g/cm³ to kg/m³
    'Density': {'symbol': 'ρ', 'units': 'kg/m³', 'conversion': 1000.0},
    'pressure': {'symbol': 'P', 'units': 'Pa', 'conversion': 0.1},  # dyn/cm² to Pa
    'Pressure': {'symbol': 'P', 'units': 'Pa', 'conversion': 0.1},
    'x_velocity': {'symbol': 'u', 'units': 'm/s', 'conversion': 0.01},  # cm/s to m/s
    'y_velocity': {'symbol': 'v', 'units': 'm/s', 'conversion': 0.01},
    'z_velocity': {'symbol': 'w', 'units': 'm/s', 'conversion': 0.01},
    'velocity_magnitude': {'symbol': '|V|', 'units': 'm/s', 'conversion': 0.01},
    'MachNumber': {'symbol': 'Ma', 'units': '', 'conversion': 1.0},
    'soundspeed': {'symbol': 'c', 'units': 'm/s', 'conversion': 0.01},
    'heatRelease': {'symbol': 'Q̇', 'units': 'W/m³', 'conversion': 0.1},  # erg/(cm³·s) to W/m³
}


def get_field_symbol_and_units(field: str) -> tuple:
    """
    Get the symbol, units, and conversion factor for a field.

    Args:
        field: Field name (PeleC format)

    Returns:
        Tuple of (symbol, units, conversion_factor). If not found, returns (field_name, '', 1.0)
    """
    # Check if it's in the mapping
    if field in FIELD_UNITS_MAP:
        info = FIELD_UNITS_MAP[field]
        return info['symbol'], info['units'], info.get('conversion', 1.0)

    # Handle species mass fractions: Y(H2) -> Y_H2
    if field.startswith('Y(') and field.endswith(')'):
        species = field[2:-1]
        return f'Y_{{{species}}}', '', 1.0  # LaTeX format for subscript, no conversion

    # Handle reaction rates: rho_omega_H2
    if field.startswith('rho_omega_'):
        species = field[10:]  # Remove 'rho_omega_'
        return f'ω_{{{species}}}', 'g/(cm³·s)', 1.0

    # Handle density-weighted species
    if field.startswith('rho_') and not field.startswith('rho_omega_'):
        species = field[4:]  # Remove 'rho_'
        return f'ρY_{{{species}}}', 'g/cm³', 1.0

    # Default: use pretty name
    return get_pretty_field_name(field), '', 1.0


def get_pele_field_name(field: str) -> str:
    """Convert pretty field name to PeleC field name."""
    # If it's already a PeleC field name, return as is
    if field in PRETTY_NAME_MAP or field.startswith('Y(') or field.startswith('rho_'):
        return field

    # Try to find in mapping
    return FIELD_NAME_MAP.get(field, field)


def get_pretty_field_name(field: str) -> str:
    """Convert PeleC field name to pretty name for folder organization."""
    # Check reverse mapping
    if field in PRETTY_NAME_MAP:
        return PRETTY_NAME_MAP[field]

    # Handle species mass fractions: Y(H2) -> Y_H2
    if field.startswith('Y(') and field.endswith(')'):
        species = field[2:-1]  # Extract species name
        return f"Y_{species}"

    # Handle reaction rates: rho_omega_H2 -> rho_omega_H2 (keep as is)
    if field.startswith('rho_omega_'):
        return field

    # Handle density-weighted species: rho_H2 -> rho_H2 (keep as is)
    if field.startswith('rho_') and not field.startswith('rho_omega_'):
        return field

    # Default: return the original string as-is if not in mapping
    return field


class PlotType(Enum):
    """Available plot types for field visualization."""
    SLICE = "slice"
    PROJECTION = "projection"
    LINE_PLOT = "line_plot"
    PHASE_PLOT = "phase_plot"
    PROFILE = "profile"
    PARTICLE_PLOT = "particle_plot"


class YTFieldPlotter:
    """
    YT Field Plotter for ParaView-like field visualization.

    Supports multiple plot types:
    - Slice plots: 2D slice through dataset at specified position
    - Projection plots: Integrated view along an axis
    - Line plots: 1D profile along a ray or axis
    - Phase plots: 2D histogram of field relationships
    - Profile plots: Radial or 1D average profiles
    """

    def __init__(self, figure_size: Tuple[int, int] = (12, 8), dpi: int = 150):
        """
        Initialize YT Field Plotter.

        Args:
            figure_size: Figure size in inches (width, height)
            dpi: Resolution in dots per inch
        """
        self.figure_size = figure_size
        self.dpi = dpi

        try:
            import yt
            yt.set_log_level(0)  # Suppress YT logging
            self.yt = yt
            self.yt_available = True
        except ImportError:
            self.yt = None
            self.yt_available = False
            print("Warning: YT not available. YTFieldPlotter will not function.")

    def _get_output_path(self, output_base: Union[Path, str], field: str,
                        dataset: Any, suffix: str = "") -> Path:
        """
        Generate organized output path: output_base/FieldName/plt00100_suffix.png

        Args:
            output_base: Base output directory
            field: Field name (will be converted to pretty name for folder)
            dataset: YT dataset (for extracting plt name)
            suffix: Optional suffix to add before extension

        Returns:
            Full output path
        """
        output_base = Path(output_base)

        # Get pretty field name for folder
        pretty_field = get_pretty_field_name(field)

        # Create field subdirectory
        field_dir = output_base / pretty_field
        field_dir.mkdir(parents=True, exist_ok=True)

        # Get dataset name (plt file name)
        dataset_name = dataset.basename

        # Build filename
        if suffix:
            filename = f"{dataset_name}_{suffix}.png"
        else:
            filename = f"{dataset_name}.png"

        return field_dir / filename

    def plot_slice(self, dataset: Any, field: str, axis: Union[str, int],
                   output_path: Union[Path, str], center: Union[str, List[float]] = 'c',
                   width: Optional[float] = None, zlim: Optional[Tuple[float, float]] = None,
                   colormap: str = 'viridis', log_scale: bool = True,
                   title: Optional[str] = None, auto_organize: bool = True, **kwargs) -> None:
        """
        Create a 2D slice plot through the dataset.

        Args:
            dataset: YT dataset to plot
            field: Field name to visualize (e.g., 'Temp', 'Temperature', 'density')
            axis: Axis normal to slice ('x', 'y', 'z' or 0, 1, 2)
            output_path: Base directory or full path to save the plot
            center: Center position ('c' for domain center, or [x, y, z] coordinates)
            width: Width of slice view (None for full domain)
            zlim: Color scale limits (min, max) or None for auto
            colormap: Matplotlib colormap name
            log_scale: Use logarithmic color scale
            title: Custom plot title
            auto_organize: If True, organize as output_path/FieldName/plt00100.png
            **kwargs: Additional YT SlicePlot options
        """
        if not self.yt_available:
            raise PlotGenerationError("slice", "YT not available")

        try:
            # Convert field name to PeleC format
            pele_field = get_pele_field_name(field)

            # Organize output path
            if auto_organize:
                output_path = self._get_output_path(output_path, pele_field, dataset, suffix="slice")
            else:
                output_path = Path(output_path)

            # Convert axis to YT format
            axis_map = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
            axis_idx = axis_map.get(axis, axis)

            # Create slice plot
            slc = self.yt.SlicePlot(dataset, axis_idx, pele_field, center=center, **kwargs)

            # Apply customizations
            if width is not None:
                slc.set_width(width)

            if zlim is not None:
                slc.set_zlim(pele_field, *zlim)

            slc.set_cmap(pele_field, colormap)

            if log_scale:
                slc.set_log(pele_field, True)
            else:
                slc.set_log(pele_field, False)

            if title:
                slc.annotate_title(title)

            # Save plot
            slc.save(str(output_path))
            print(f"Saved slice plot: {output_path}")

        except Exception as e:
            raise PlotGenerationError("slice", str(e))

    def plot_projection(self, dataset: Any, field: str, axis: Union[str, int],
                       output_path: Union[Path, str], weight_field: Optional[str] = None,
                       center: Union[str, List[float]] = 'c',
                       width: Optional[float] = None, zlim: Optional[Tuple[float, float]] = None,
                       colormap: str = 'viridis', log_scale: bool = True,
                       title: Optional[str] = None, auto_organize: bool = True, **kwargs) -> None:
        """
        Create a projection plot (integrated view along axis).

        Args:
            dataset: YT dataset to plot
            field: Field name to project
            axis: Axis to project along ('x', 'y', 'z' or 0, 1, 2)
            output_path: Base directory or full path to save the plot
            weight_field: Field to weight projection by (None for sum, or field name for average)
            center: Center position
            width: Width of view
            zlim: Color scale limits
            colormap: Matplotlib colormap name
            log_scale: Use logarithmic color scale
            title: Custom plot title
            auto_organize: If True, organize as output_path/FieldName/plt00100.png
            **kwargs: Additional YT ProjectionPlot options
        """
        if not self.yt_available:
            raise PlotGenerationError("projection", "YT not available")

        try:
            # Convert field name to PeleC format
            pele_field = get_pele_field_name(field)

            # Organize output path
            if auto_organize:
                suffix = "projection_weighted" if weight_field else "projection"
                output_path = self._get_output_path(output_path, pele_field, dataset, suffix=suffix)
            else:
                output_path = Path(output_path)

            axis_map = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
            axis_idx = axis_map.get(axis, axis)

            # Create projection plot
            proj = self.yt.ProjectionPlot(dataset, axis_idx, pele_field,
                                         center=center, weight_field=weight_field, **kwargs)

            # Apply customizations
            if width is not None:
                proj.set_width(width)

            if zlim is not None:
                proj.set_zlim(pele_field, *zlim)

            proj.set_cmap(pele_field, colormap)

            if log_scale:
                proj.set_log(pele_field, True)
            else:
                proj.set_log(pele_field, False)

            if title:
                proj.annotate_title(title)

            # Save plot
            proj.save(str(output_path))
            print(f"Saved projection plot: {output_path}")

        except Exception as e:
            raise PlotGenerationError("projection", str(e))

    def plot_line(self, dataset: Any, fields: Union[str, List[str]],
                 start_point: List[float], end_point: List[float],
                 output_path: Path, num_points: int = 512,
                 xlabel: str = 'Position', title: Optional[str] = None,
                 log_x: bool = False, log_y: bool = False, **kwargs) -> None:
        """
        Create a 1D line plot along a ray through the dataset.

        Args:
            dataset: YT dataset to plot
            fields: Field name(s) to plot along the line
            start_point: Starting coordinates [x, y, z]
            end_point: Ending coordinates [x, y, z]
            output_path: Path to save the plot
            num_points: Number of sampling points along ray
            xlabel: X-axis label
            title: Plot title
            log_x: Use log scale for x-axis
            log_y: Use log scale for y-axis
            **kwargs: Additional matplotlib plot options
        """
        if not self.yt_available:
            raise PlotGenerationError("line_plot", "YT not available")

        try:
            # Ensure fields is a list
            if isinstance(fields, str):
                fields = [fields]

            # Create ray through dataset
            ray = self.yt.YTRay(dataset, start_point, end_point, num_points=num_points)

            # Calculate distance along ray
            ray_length = np.sqrt(np.sum((np.array(end_point) - np.array(start_point))**2))
            distances = np.linspace(0, ray_length, num_points)

            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            # Plot each field
            for field in fields:
                field_data = ray[field].to_ndarray()
                ax.plot(distances, field_data, label=field, linewidth=2, **kwargs)

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Field Value')

            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'Line Plot: {", ".join(fields)}')

            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved line plot to {output_path}")

        except Exception as e:
            raise PlotGenerationError("line_plot", str(e))

    def plot_phase(self, dataset: Any, x_field: str, y_field: str,
                  z_field: str, output_path: Path,
                  weight_field: Optional[str] = None,
                  x_bins: int = 128, y_bins: int = 128,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  zlim: Optional[Tuple[float, float]] = None,
                  colormap: str = 'viridis', log_x: bool = False,
                  log_y: bool = False, log_z: bool = False,
                  title: Optional[str] = None, **kwargs) -> None:
        """
        Create a 2D phase plot showing relationship between fields.

        Args:
            dataset: YT dataset to plot
            x_field: Field for x-axis
            y_field: Field for y-axis
            z_field: Field for color (typically mass, volume, or count)
            output_path: Path to save the plot
            weight_field: Field to weight by
            x_bins: Number of bins in x direction
            y_bins: Number of bins in y direction
            xlim: X-axis limits
            ylim: Y-axis limits
            zlim: Color scale limits
            colormap: Matplotlib colormap name
            log_x: Use log scale for x-axis
            log_y: Use log scale for y-axis
            log_z: Use log scale for color
            title: Plot title
            **kwargs: Additional YT PhasePlot options
        """
        if not self.yt_available:
            raise PlotGenerationError("phase_plot", "YT not available")

        try:
            # Create phase plot
            ad = dataset.all_data()

            phase = self.yt.PhasePlot(ad, x_field, y_field, z_field,
                                     weight_field=weight_field,
                                     x_bins=x_bins, y_bins=y_bins, **kwargs)

            # Apply customizations
            if xlim is not None:
                phase.set_xlim(*xlim)

            if ylim is not None:
                phase.set_ylim(*ylim)

            if zlim is not None:
                phase.set_zlim(z_field, *zlim)

            phase.set_cmap(z_field, colormap)

            if log_x:
                phase.set_log(x_field, True)
            else:
                phase.set_log(x_field, False)

            if log_y:
                phase.set_log(y_field, True)
            else:
                phase.set_log(y_field, False)

            if log_z:
                phase.set_log(z_field, True)
            else:
                phase.set_log(z_field, False)

            if title:
                phase.annotate_title(title)

            # Save plot
            phase.save(str(output_path))
            print(f"Saved phase plot to {output_path}")

        except Exception as e:
            raise PlotGenerationError("phase_plot", str(e))

    def plot_profile(self, dataset: Any, fields: Union[str, List[str]],
                    output_path: Path, weight_field: Optional[str] = None,
                    n_bins: int = 64, extrema: Optional[Dict[str, Tuple[float, float]]] = None,
                    log_x: bool = False, log_y: bool = False,
                    title: Optional[str] = None, **kwargs) -> None:
        """
        Create 1D profile plot (radial or binned average).

        Args:
            dataset: YT dataset to plot
            fields: Field name(s) to profile
            output_path: Path to save the plot
            weight_field: Field to weight average by
            n_bins: Number of bins
            extrema: Dictionary of field extrema for binning
            log_x: Use log scale for x-axis
            log_y: Use log scale for y-axis
            title: Plot title
            **kwargs: Additional YT create_profile options
        """
        if not self.yt_available:
            raise PlotGenerationError("profile", "YT not available")

        try:
            # Ensure fields is a list
            if isinstance(fields, str):
                fields = [fields]

            # Get all data
            ad = dataset.all_data()

            # Create profile (binned average)
            profile = self.yt.create_profile(ad, 'radius', fields,
                                            weight_field=weight_field,
                                            n_bins=n_bins,
                                            extrema=extrema, **kwargs)

            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            # Get radius bins
            radius = profile.x.to_ndarray()

            # Plot each field
            for field in fields:
                field_data = profile[field].to_ndarray()
                ax.plot(radius, field_data, label=field, linewidth=2, marker='o', markersize=4)

            ax.set_xlabel('Radius')
            ax.set_ylabel('Field Value')

            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'Radial Profile: {", ".join(fields)}')

            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved profile plot to {output_path}")

        except Exception as e:
            raise PlotGenerationError("profile", str(e))

    def plot_ortho_ray(self, dataset: Any, fields: Union[str, List[str]],
                      axis: Union[str, int], coord: Tuple[float, float],
                      output_path: Path, title: Optional[str] = None,
                      xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                      log_x: bool = False, log_y: bool = False,
                      plot_kwargs: Optional[Dict] = None) -> None:
        """
        Create a plot along an orthogonal ray (like a 1D slice).

        Args:
            dataset: YT dataset to plot
            fields: Field name(s) to plot
            axis: Axis along which to extract ray (0=x, 1=y, 2=z)
            coord: Coordinates perpendicular to axis (e.g., (y, z) for x-axis ray)
            output_path: Path to save the plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_x: Use log scale for x-axis
            log_y: Use log scale for y-axis
            plot_kwargs: Additional matplotlib plot options
        """
        if not self.yt_available:
            raise PlotGenerationError("ortho_ray", "YT not available")

        try:
            # Ensure fields is a list
            if isinstance(fields, str):
                fields = [fields]

            if plot_kwargs is None:
                plot_kwargs = {}

            # Convert axis
            axis_map = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
            axis_idx = axis_map.get(axis, axis)
            axis_names = ['x', 'y', 'z']

            # Create orthogonal ray
            ray = dataset.ortho_ray(axis_idx, coord)

            # Extract coordinates along axis
            axis_field = f'boxlib,{axis_names[axis_idx]}'
            ray_coords = ray[axis_field].to_ndarray()

            # Sort by coordinate to ensure proper ordering
            sort_indices = np.argsort(ray_coords)
            ray_coords = ray_coords[sort_indices]

            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            # Plot each field
            for field in fields:
                field_data = ray[field].to_ndarray()[sort_indices]
                ax.plot(ray_coords, field_data, label=field, linewidth=2, **plot_kwargs)

            # Set labels
            if xlabel:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel(f'{axis_names[axis_idx]} coordinate')

            if ylabel:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel('Field Value')

            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')

            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'Orthogonal Ray: {", ".join(fields)}')

            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved orthogonal ray plot to {output_path}")

        except Exception as e:
            raise PlotGenerationError("ortho_ray", str(e))

    def list_available_fields(self, dataset: Any) -> List[str]:
        """
        List all available fields in the dataset.

        Args:
            dataset: YT dataset

        Returns:
            List of field names
        """
        if not self.yt_available:
            print("Warning: YT not available")
            return []

        try:
            # Get field list
            fields = [field[1] for field in dataset.field_list]

            print(f"\nAvailable fields ({len(fields)} total):")
            print("-" * 50)

            # Group by category
            native_fields = [f for f in fields if not f.startswith('Y(')]
            species_fields = [f for f in fields if f.startswith('Y(')]

            print(f"\nNative fields ({len(native_fields)}):")
            for i, field in enumerate(sorted(native_fields)):
                print(f"  {field}", end='')
                if (i + 1) % 3 == 0:
                    print()
            print()

            if species_fields:
                print(f"\nSpecies mass fractions ({len(species_fields)}):")
                for i, field in enumerate(sorted(species_fields)):
                    print(f"  {field}", end='')
                    if (i + 1) % 5 == 0:
                        print()
                print()

            return fields

        except Exception as e:
            print(f"Error listing fields: {e}")
            return []

    def plot_localized_contour(self, dataset: Any, field: str,
                              center_point: List[float],
                              forward_bound: float, backward_bound: float,
                              output_path: Union[Path, str], axis: Union[str, int] = 'z',
                              normal_axis: str = 'x',
                              levels: int = 50, colormap: str = 'viridis',
                              log_scale: bool = False,
                              zlim: Optional[Tuple[float, float]] = None,
                              title: Optional[str] = None,
                              contour_lines: Optional[Dict[str, np.ndarray]] = None,
                              auto_organize: bool = True,
                              **kwargs) -> None:
        """
        Create a localized 2D contour/surface plot around a specific point with bounds.

        This method extracts data from a box region around a center point with specified
        forward and backward extents, similar to zooming into a specific region in ParaView.

        Args:
            dataset: YT dataset to plot
            field: Field name to visualize
            center_point: Center point coordinates [x, y, z] in dataset units
            forward_bound: Distance forward (positive direction) from center along normal_axis
            backward_bound: Distance backward (negative direction) from center along normal_axis
            output_path: Base directory or full path to save the plot
            axis: Axis normal to plotting plane ('x', 'y', 'z' or 0, 1, 2)
            normal_axis: Reference axis for forward/backward bounds ('x', 'y', or 'z')
            levels: Number of contour levels
            colormap: Matplotlib colormap name
            log_scale: Use logarithmic color scale
            zlim: Color scale limits (min, max)
            title: Custom plot title
            contour_lines: Optional dictionary of named contour lines to overlay
                          (e.g., {'flame': flame_contour_points})
            auto_organize: If True, organize as output_path/FieldName/plt00100.png
            **kwargs: Additional matplotlib tricontourf options
        """
        if not self.yt_available:
            raise PlotGenerationError("localized_contour", "YT not available")

        try:
            # Convert field name to PeleC format
            pele_field = get_pele_field_name(field)

            # Organize output path
            if auto_organize:
                output_path = self._get_output_path(output_path, pele_field, dataset, suffix="localized_contour")
            else:
                output_path = Path(output_path)
            # Convert axes to indices
            axis_map = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
            axis_idx = axis_map.get(axis, axis)
            normal_axis_idx = axis_map.get(normal_axis, normal_axis)
            axis_names = ['x', 'y', 'z']

            # Get domain bounds
            domain_left = dataset.domain_left_edge.to_ndarray()
            domain_right = dataset.domain_right_edge.to_ndarray()

            # Calculate box bounds
            # Start with center point, then expand in all directions
            left_edge = list(center_point)
            right_edge = list(center_point)

            # Set bounds along normal axis (the tracking axis - typically X for flame tracking)
            left_edge[normal_axis_idx] = center_point[normal_axis_idx] - backward_bound
            right_edge[normal_axis_idx] = center_point[normal_axis_idx] + forward_bound

            # For the plotting plane dimensions (perpendicular to normal_axis),
            # use full domain extent
            for i in range(3):
                if i != normal_axis_idx:
                    left_edge[i] = domain_left[i]
                    right_edge[i] = domain_right[i]

            # Ensure bounds don't exceed domain
            for i in range(3):
                left_edge[i] = max(left_edge[i], domain_left[i])
                right_edge[i] = min(right_edge[i], domain_right[i])

            print(f"Extracting localized region:")
            print(f"  Center: {center_point}")
            print(f"  Box bounds: {left_edge} to {right_edge}")
            print(f"  Normal axis: {axis_names[normal_axis_idx]} (backward: {backward_bound}, forward: {forward_bound})")

            # Create box region
            box_region = dataset.box(left_edge=left_edge, right_edge=right_edge)

            # Get max refinement level for highest resolution
            max_level = dataset.index.max_level

            # Calculate dimensions for covering grid
            dx = dataset.index.get_smallest_dx().to_value()
            dims = [int((right_edge[i] - left_edge[i]) / dx) for i in range(3)]
            dims[axis_idx] = 1  # Single cell in the slice direction

            # Create covering grid to extract data
            cg = dataset.covering_grid(
                level=max_level,
                left_edge=left_edge,
                dims=dims,
                fields=[('boxlib', 'x'), ('boxlib', 'y'), ('boxlib', 'z'), pele_field]
            )

            # Extract coordinates and field data
            x_data = cg['boxlib', 'x'].to_ndarray().flatten()
            y_data = cg['boxlib', 'y'].to_ndarray().flatten()
            z_data = cg['boxlib', 'z'].to_ndarray().flatten()
            field_data = cg[pele_field].to_ndarray().flatten()

            # Select coordinates for the plotting plane
            if axis_idx == 0:  # x-normal, plot y-z
                plot_x, plot_y = y_data, z_data
                xlabel, ylabel = 'Y', 'Z'
            elif axis_idx == 1:  # y-normal, plot x-z
                plot_x, plot_y = x_data, z_data
                xlabel, ylabel = 'X', 'Z'
            else:  # z-normal, plot x-y
                plot_x, plot_y = x_data, y_data
                xlabel, ylabel = 'X', 'Y'

            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

            # Apply log scale if requested
            if log_scale:
                field_data = np.log10(np.abs(field_data) + 1e-20)
                field_label = f'log10({get_pretty_field_name(pele_field)})'
            else:
                field_label = get_pretty_field_name(pele_field)

            # Create contour plot
            contour = ax.tricontourf(plot_x, plot_y, field_data,
                                     levels=levels, cmap=colormap, **kwargs)

            # Add colorbar at the top
            cbar = plt.colorbar(contour, ax=ax, label=field_label, orientation='horizontal',
                               pad=0.05, location='top')

            # Apply color limits if specified
            if zlim is not None:
                contour.set_clim(*zlim)

            # Overlay contour lines if provided
            if contour_lines:
                for line_name, line_points in contour_lines.items():
                    if line_points is not None and len(line_points) > 0:
                        # Assume line_points is Nx2 or Nx3 array
                        # Extract appropriate coordinates based on plotting plane
                        if axis_idx == 0:  # y-z plane
                            line_x, line_y = line_points[:, 1], line_points[:, 2] if line_points.shape[1] > 2 else (line_points[:, 0], line_points[:, 1])
                        elif axis_idx == 1:  # x-z plane
                            line_x, line_y = line_points[:, 0], line_points[:, 2] if line_points.shape[1] > 2 else (line_points[:, 0], line_points[:, 1])
                        else:  # x-y plane (most common)
                            line_x, line_y = line_points[:, 0], line_points[:, 1]

                        ax.plot(line_x, line_y, label=line_name, linewidth=2, color='red', linestyle='--')

            # Set labels and title
            ax.set_xlabel(f'{xlabel} (dataset units)')
            ax.set_ylabel(f'{ylabel} (dataset units)')
            ax.set_aspect('equal', adjustable='box')

            if title:
                ax.set_title(title)
            else:
                pretty_name = get_pretty_field_name(pele_field)
                ax.set_title(f'Localized {pretty_name} - Center: {center_point[normal_axis_idx]:.6f}')

            if contour_lines:
                ax.legend()

            # Save plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved localized contour plot: {output_path}")

        except Exception as e:
            raise PlotGenerationError("localized_contour", str(e))

    def plot_multiple_fields(self, dataset: Any, fields: List[str],
                            plot_type: PlotType, output_dir: Path,
                            axis: Union[str, int] = 'z',
                            **common_kwargs) -> None:
        """
        Create multiple plots of different fields with the same settings.

        Args:
            dataset: YT dataset
            fields: List of field names to plot
            plot_type: Type of plot to create
            output_dir: Directory to save plots
            axis: Axis for slice/projection plots
            **common_kwargs: Common arguments for all plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for field in fields:
            # Clean field name for filename
            safe_field_name = field.replace('(', '_').replace(')', '_').replace(',', '_')
            output_path = output_dir / f"{plot_type.value}_{safe_field_name}.png"

            try:
                if plot_type == PlotType.SLICE:
                    self.plot_slice(dataset, field, axis, output_path, **common_kwargs)
                elif plot_type == PlotType.PROJECTION:
                    self.plot_projection(dataset, field, axis, output_path, **common_kwargs)
                else:
                    print(f"Plot type {plot_type} not supported for multiple field plotting")

            except Exception as e:
                print(f"Failed to plot {field}: {e}")

    def plot_multiple_localized_contours(self, dataset: Any, fields: List[Dict[str, Any]],
                                        center_point: List[float],
                                        forward_bound: float, backward_bound: float,
                                        output_path: Union[Path, str], axis: Union[str, int] = 'z',
                                        normal_axis: str = 'x',
                                        contour_lines: Optional[Dict[str, np.ndarray]] = None,
                                        auto_organize: bool = True,
                                        enable_y_zoom: bool = False,
                                        y_min: float = 0.0,
                                        y_max: float = 0.001) -> None:
        """
        OPTIMIZED: Extract localized region ONCE and plot multiple fields from it.

        This method extracts the localized region data only once, then plots multiple
        fields from the same extracted data. Much more efficient than calling
        plot_localized_contour() multiple times.

        Args:
            dataset: YT dataset to plot
            fields: List of field configurations, each containing:
                   {'field': str, 'colormap': str, 'log_scale': bool, 'levels': int, 'zlim': tuple}
            center_point: Center point coordinates [x, y, z] in dataset units
            forward_bound: Distance forward from center along normal_axis
            backward_bound: Distance backward from center along normal_axis
            output_path: Base directory for organized output
            axis: Axis normal to plotting plane ('x', 'y', 'z' or 0, 1, 2)
            normal_axis: Reference axis for forward/backward bounds ('x', 'y', or 'z')
            contour_lines: Optional dictionary of contour lines to overlay on all plots
            auto_organize: If True, organize as output_path/FieldName/plt00100.png
            enable_y_zoom: Enable y-direction zooming (limits y-axis extent)
            y_min: Minimum y-coordinate for zoom (dataset units)
            y_max: Maximum y-coordinate for zoom (dataset units)
        """
        if not self.yt_available:
            raise PlotGenerationError("multiple_localized_contours", "YT not available")

        try:
            # Convert axes to indices
            axis_map = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
            axis_idx = axis_map.get(axis, axis)
            normal_axis_idx = axis_map.get(normal_axis, normal_axis)
            axis_names = ['x', 'y', 'z']

            # Get domain bounds
            domain_left = dataset.domain_left_edge.to_ndarray()
            domain_right = dataset.domain_right_edge.to_ndarray()

            # Calculate box bounds
            left_edge = list(center_point)
            right_edge = list(center_point)

            # Set bounds along normal axis
            left_edge[normal_axis_idx] = center_point[normal_axis_idx] - backward_bound
            right_edge[normal_axis_idx] = center_point[normal_axis_idx] + forward_bound

            # For plotting plane dimensions, use full domain extent OR y-zoom if enabled
            for i in range(3):
                if i != normal_axis_idx:
                    # Check if this is the y-axis (index 1) and y-zoom is enabled
                    if i == 1 and enable_y_zoom:
                        # Use specified y-bounds instead of full domain
                        left_edge[i] = y_min
                        right_edge[i] = y_max
                    else:
                        # Use full domain extent
                        left_edge[i] = domain_left[i]
                        right_edge[i] = domain_right[i]

            # Ensure bounds don't exceed domain
            for i in range(3):
                left_edge[i] = max(left_edge[i], domain_left[i])
                right_edge[i] = min(right_edge[i], domain_right[i])

            print(f"Extracting localized region ONCE for {len(fields)} fields:")
            print(f"  Center: {center_point}")
            print(f"  Box bounds: {left_edge} to {right_edge}")
            print(f"  Normal axis: {axis_names[normal_axis_idx]} (backward: {backward_bound}, forward: {forward_bound})")

            # Get max refinement level
            max_level = dataset.index.max_level

            # Calculate dimensions for covering grid
            dx = dataset.index.get_smallest_dx().to_value()
            dims = [int((right_edge[i] - left_edge[i]) / dx) for i in range(3)]
            dims[axis_idx] = 1  # Single cell in the slice direction

            # Convert field names to PeleC format
            pele_fields = [get_pele_field_name(f['field']) for f in fields]

            # Create covering grid to extract ALL fields at once
            all_fields_list = [('boxlib', 'x'), ('boxlib', 'y'), ('boxlib', 'z')] + pele_fields

            print(f"  Creating covering grid for all {len(pele_fields)} fields...")
            cg = dataset.covering_grid(
                level=max_level,
                left_edge=left_edge,
                dims=dims,
                fields=all_fields_list
            )

            # Extract coordinates ONCE
            x_data = cg['boxlib', 'x'].to_ndarray().flatten()
            y_data = cg['boxlib', 'y'].to_ndarray().flatten()
            z_data = cg['boxlib', 'z'].to_ndarray().flatten()

            # Select coordinates for the plotting plane
            if axis_idx == 0:  # x-normal, plot y-z
                plot_x, plot_y = y_data, z_data
                xlabel, ylabel = 'Y', 'Z'
            elif axis_idx == 1:  # y-normal, plot x-z
                plot_x, plot_y = x_data, z_data
                xlabel, ylabel = 'X', 'Z'
            else:  # z-normal, plot x-y
                plot_x, plot_y = x_data, y_data
                xlabel, ylabel = 'X', 'Y'

            # Now plot each field from the extracted data
            print(f"  Creating {len(fields)} plots from extracted data...")
            for i, (field_config, pele_field) in enumerate(zip(fields, pele_fields)):
                try:
                    field_name = field_config['field']
                    colormap = field_config.get('colormap', 'viridis')
                    log_scale = field_config.get('log_scale', False)
                    levels = field_config.get('levels', 50)
                    zlim = field_config.get('zlim', None)

                    # Extract field data from covering grid
                    field_data = cg[pele_field].to_ndarray().flatten()

                    # Organize output path
                    if auto_organize:
                        plot_output_path = self._get_output_path(output_path, pele_field, dataset, suffix="localized_contour")
                    else:
                        plot_output_path = Path(output_path) / f"{pele_field}_localized_contour.png"

                    # Create plot
                    fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

                    # Apply log scale if requested
                    if log_scale:
                        field_data_plot = np.log10(np.abs(field_data) + 1e-20)
                        field_label = f'log10({get_pretty_field_name(pele_field)})'
                    else:
                        field_data_plot = field_data
                        field_label = get_pretty_field_name(pele_field)

                    # Create contour plot
                    contour = ax.tricontourf(plot_x, plot_y, field_data_plot,
                                            levels=levels, cmap=colormap)

                    # Add colorbar at the top
                    plt.colorbar(contour, ax=ax, label=field_label, orientation='horizontal',
                               pad=0.05, location='top')

                    # Apply color limits if specified
                    if zlim is not None:
                        contour.set_clim(*zlim)

                    # Overlay contour lines if provided
                    if contour_lines:
                        for line_name, line_points in contour_lines.items():
                            if line_points is not None and len(line_points) > 0:
                                if axis_idx == 0:  # y-z plane
                                    line_x, line_y = line_points[:, 1], line_points[:, 2] if line_points.shape[1] > 2 else (line_points[:, 0], line_points[:, 1])
                                elif axis_idx == 1:  # x-z plane
                                    line_x, line_y = line_points[:, 0], line_points[:, 2] if line_points.shape[1] > 2 else (line_points[:, 0], line_points[:, 1])
                                else:  # x-y plane
                                    line_x, line_y = line_points[:, 0], line_points[:, 1]

                                ax.plot(line_x, line_y, label=line_name, linewidth=2, color='red', linestyle='--')

                    # Set labels and title
                    ax.set_xlabel(f'{xlabel} (dataset units)')
                    ax.set_ylabel(f'{ylabel} (dataset units)')
                    ax.set_aspect('equal', adjustable='box')

                    pretty_name = get_pretty_field_name(pele_field)
                    ax.set_title(f'Localized {pretty_name} - Center: {center_point[normal_axis_idx]:.6f}')

                    if contour_lines:
                        ax.legend()

                    # Save plot
                    plt.tight_layout()
                    plt.savefig(plot_output_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close(fig)

                    print(f"    [OK] {field_name}")

                except Exception as e:
                    print(f"    [FAILED] {field_config['field']}: {e}")

        except Exception as e:
            raise PlotGenerationError("multiple_localized_contours", str(e))

    def plot_combined_localized_figure(self, dataset: Any, field: str,
                                      center_point: List[float],
                                      forward_bound: float, backward_bound: float,
                                      extraction_y: float,
                                      output_path: Union[Path, str], axis: Union[str, int] = 'z',
                                      normal_axis: str = 'x',
                                      levels: int = 50, colormap: str = 'viridis',
                                      log_scale: bool = False,
                                      zlim: Optional[Tuple[float, float]] = None,
                                      title: Optional[str] = None,
                                      contour_lines: Optional[Dict[str, np.ndarray]] = None,
                                      auto_organize: bool = True,
                                      time_offset: float = 0.0) -> None:
        """
        Create a combined multi-panel figure with both 1D ortho-ray and 2D surface plots.

        This method creates a publication-quality figure with two panels:
        - Top panel: 1D ortho-ray plot showing field variation along the normal axis
        - Bottom panel: 2D surface/contour plot in the orthogonal plane

        Args:
            dataset: YT dataset to plot
            field: Field name to visualize
            center_point: Center point coordinates [x, y, z] in dataset units
            forward_bound: Distance forward from center along normal_axis
            backward_bound: Distance backward from center along normal_axis
            extraction_y: Y-coordinate for extracting the 1D ray (in cm, dataset units)
            output_path: Base directory or full path to save the plot
            axis: Axis normal to plotting plane ('x', 'y', 'z' or 0, 1, 2)
            normal_axis: Reference axis for forward/backward bounds ('x', 'y', or 'z')
            levels: Number of contour levels for 2D plot
            colormap: Matplotlib colormap name
            log_scale: Use logarithmic color scale
            zlim: Color scale limits (min, max)
            title: Custom plot title (shown at the top)
            contour_lines: Optional dictionary of named contour lines to overlay
            auto_organize: If True, organize as output_path/FieldName/plt00100_combined.png
            time_offset: Time offset in seconds to add to dataset time (for correcting restart times, etc.)
        """
        if not self.yt_available:
            raise PlotGenerationError("combined_localized_figure", "YT not available")

        try:
            # Convert field name to PeleC format
            pele_field = get_pele_field_name(field)

            # Organize output path
            if auto_organize:
                output_path = self._get_output_path(output_path, pele_field, dataset, suffix="combined")
            else:
                output_path = Path(output_path)

            # Convert axes to indices
            axis_map = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
            axis_idx = axis_map.get(axis, axis)
            normal_axis_idx = axis_map.get(normal_axis, normal_axis)
            axis_names = ['x', 'y', 'z']

            # Get domain bounds
            domain_left = dataset.domain_left_edge.to_ndarray()
            domain_right = dataset.domain_right_edge.to_ndarray()

            # Calculate box bounds
            left_edge = list(center_point)
            right_edge = list(center_point)

            # Set bounds along normal axis
            left_edge[normal_axis_idx] = center_point[normal_axis_idx] - backward_bound
            right_edge[normal_axis_idx] = center_point[normal_axis_idx] + forward_bound

            # For plotting plane dimensions, use full domain extent
            for i in range(3):
                if i != normal_axis_idx:
                    left_edge[i] = domain_left[i]
                    right_edge[i] = domain_right[i]

            # Ensure bounds don't exceed domain
            for i in range(3):
                left_edge[i] = max(left_edge[i], domain_left[i])
                right_edge[i] = min(right_edge[i], domain_right[i])

            print(f"Creating combined figure for {field}:")
            print(f"  Center: {center_point}")
            print(f"  Box bounds: {left_edge} to {right_edge}")
            print(f"  Normal axis: {axis_names[normal_axis_idx]} (backward: {backward_bound}, forward: {forward_bound})")

            # ============================================================================
            # STEP 1: Extract 1D ortho-ray data for top panel
            # ============================================================================
            print(f"  Extracting 1D ortho-ray along {axis_names[normal_axis_idx]}-axis at y={extraction_y}...")

            # Create orthogonal ray along the normal axis
            # For x-axis ray, coord should be (y, z)
            if normal_axis_idx == 0:  # x-axis
                ray_coord = (extraction_y, (domain_left[2] + domain_right[2]) / 2)
            elif normal_axis_idx == 1:  # y-axis
                ray_coord = ((domain_left[0] + domain_right[0]) / 2, (domain_left[2] + domain_right[2]) / 2)
            else:  # z-axis
                ray_coord = ((domain_left[0] + domain_right[0]) / 2, extraction_y)

            ray = dataset.ortho_ray(normal_axis_idx, ray_coord)

            # Extract coordinates and field data along ray
            axis_field = ('boxlib', axis_names[normal_axis_idx])
            ray_coords = ray[axis_field].to_ndarray()
            ray_field_data = ray[pele_field].to_ndarray()

            # Sort by coordinate
            sort_indices = np.argsort(ray_coords)
            ray_coords = ray_coords[sort_indices]
            ray_field_data = ray_field_data[sort_indices]

            # Filter to localized region only
            mask = (ray_coords >= left_edge[normal_axis_idx]) & (ray_coords <= right_edge[normal_axis_idx])
            ray_coords = ray_coords[mask]
            ray_field_data = ray_field_data[mask]

            # ============================================================================
            # STEP 2: Extract 2D surface data for bottom panel
            # ============================================================================
            print(f"  Extracting 2D surface data...")

            # Get max refinement level
            max_level = dataset.index.max_level

            # Calculate dimensions for covering grid
            dx = dataset.index.get_smallest_dx().to_value()
            dims = [int((right_edge[i] - left_edge[i]) / dx) for i in range(3)]
            dims[axis_idx] = 1  # Single cell in the slice direction

            # Create covering grid to extract data
            cg = dataset.covering_grid(
                level=max_level,
                left_edge=left_edge,
                dims=dims,
                fields=[('boxlib', 'x'), ('boxlib', 'y'), ('boxlib', 'z'), pele_field]
            )

            # Extract coordinates and field data
            x_data = cg['boxlib', 'x'].to_ndarray().flatten()
            y_data = cg['boxlib', 'y'].to_ndarray().flatten()
            z_data = cg['boxlib', 'z'].to_ndarray().flatten()
            field_data_2d = cg[pele_field].to_ndarray().flatten()

            # Select coordinates for the plotting plane
            if axis_idx == 0:  # x-normal, plot y-z
                plot_x, plot_y = y_data, z_data
                xlabel_2d, ylabel_2d = 'Y', 'Z'
            elif axis_idx == 1:  # y-normal, plot x-z
                plot_x, plot_y = x_data, z_data
                xlabel_2d, ylabel_2d = 'X', 'Z'
            else:  # z-normal, plot x-y
                plot_x, plot_y = x_data, y_data
                xlabel_2d, ylabel_2d = 'X', 'Y'

            # ============================================================================
            # STEP 3: Create combined figure with two panels
            # ============================================================================
            print(f"  Creating combined figure...")

            # Create figure with 2 subplots (top and bottom)
            fig = plt.figure(figsize=(12, 10), dpi=self.dpi)

            # Use GridSpec for better control over layout
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.3)

            # TOP PANEL: 1D Ortho-Ray Plot
            ax1 = fig.add_subplot(gs[0])

            # Get field symbol, units, and conversion factor
            field_symbol, field_units, conversion_factor = get_field_symbol_and_units(pele_field)
            pretty_name = get_pretty_field_name(pele_field)

            # Apply unit conversion to 1D data
            ray_field_data_converted = ray_field_data * conversion_factor

            ax1.plot(ray_coords, ray_field_data_converted, linewidth=2, color='navy')

            # Set axis labels with symbols and units
            ax1.set_xlabel(f'{axis_names[normal_axis_idx].upper()} (cm)')
            if field_units:
                ax1.set_ylabel(f'{field_symbol} ({field_units})')
            else:
                ax1.set_ylabel(field_symbol)
            ax1.grid(True, alpha=0.3)

            # Store x-limits for alignment (will be applied after 2D plot is created)
            ray_xlim = (ray_coords.min(), ray_coords.max())

            # Add time annotation with full precision (apply time offset if provided)
            time_corrected = float(dataset.current_time) + time_offset  # Apply offset in seconds
            time_ms = time_corrected * 1000  # Convert to ms
            time_text = f't={time_ms} ms'
            ax1.text(0.98, 0.95, time_text, transform=ax1.transAxes,
                    fontsize=12, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # BOTTOM PANEL: 2D Surface Plot
            ax2 = fig.add_subplot(gs[1])

            # Apply unit conversion to 2D data
            field_data_2d_converted = field_data_2d * conversion_factor

            # Apply log scale if requested
            if log_scale:
                field_data_plot = np.log10(np.abs(field_data_2d_converted) + 1e-20)
                if field_units:
                    field_label = f'log10({field_symbol}) ({field_units})'
                else:
                    field_label = f'log10({field_symbol})'
            else:
                field_data_plot = field_data_2d_converted
                if field_units:
                    field_label = f'{field_symbol} ({field_units})'
                else:
                    field_label = field_symbol

            # Create contour plot
            contour = ax2.tricontourf(plot_x, plot_y, field_data_plot,
                                     levels=levels, cmap=colormap)

            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax2, label=field_label,
                              orientation='horizontal', pad=0.08, location='top')

            # Apply color limits if specified
            if zlim is not None:
                contour.set_clim(*zlim)

            # Overlay contour lines if provided
            if contour_lines:
                for line_name, line_points in contour_lines.items():
                    if line_points is not None and len(line_points) > 0:
                        if axis_idx == 0:  # y-z plane
                            line_x, line_y = line_points[:, 1], line_points[:, 2] if line_points.shape[1] > 2 else (line_points[:, 0], line_points[:, 1])
                        elif axis_idx == 1:  # x-z plane
                            line_x, line_y = line_points[:, 0], line_points[:, 2] if line_points.shape[1] > 2 else (line_points[:, 0], line_points[:, 1])
                        else:  # x-y plane
                            line_x, line_y = line_points[:, 0], line_points[:, 1]

                        ax2.plot(line_x, line_y, label=line_name, linewidth=2, color='red', linestyle='--')

            # Set labels and formatting with proper units
            ax2.set_xlabel(f'{xlabel_2d} (cm)')
            ax2.set_ylabel(f'{ylabel_2d} (cm)')
            ax2.set_aspect('equal', adjustable='box')

            if contour_lines:
                ax2.legend()

            # Align x-axes between top and bottom panels
            # Use the extent from the 2D data for both plots to ensure perfect alignment
            # The normal_axis (typically X) should match in both panels
            if normal_axis_idx == 0 and axis_idx == 2:  # Most common: X is normal axis, Z-normal plot (x-y plane)
                # Both plots show X on horizontal axis
                x_min, x_max = plot_x.min(), plot_x.max()
                ax1.set_xlim(x_min, x_max)
                ax2.set_xlim(x_min, x_max)
            elif normal_axis_idx == 0 and axis_idx == 1:  # X is normal axis, Y-normal plot (x-z plane)
                # Both plots show X on horizontal axis
                x_min, x_max = plot_x.min(), plot_x.max()
                ax1.set_xlim(x_min, x_max)
                ax2.set_xlim(x_min, x_max)

            # Add overall title - just the field name
            if title:
                fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            else:
                fig.suptitle(pretty_name, fontsize=16, fontweight='bold', y=0.98)

            # Apply tight layout before saving
            plt.tight_layout()

            # Save plot
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

            print(f"  Saved combined figure: {output_path}")

        except Exception as e:
            raise PlotGenerationError("combined_localized_figure", str(e))

    def plot_multiple_combined_figures(self, dataset: Any, fields: List[Dict[str, Any]],
                                      center_point: List[float],
                                      forward_bound: float, backward_bound: float,
                                      extraction_y: float,
                                      output_path: Union[Path, str], axis: Union[str, int] = 'z',
                                      normal_axis: str = 'x',
                                      contour_lines: Optional[Dict[str, np.ndarray]] = None,
                                      auto_organize: bool = True,
                                      time_offset: float = 0.0) -> None:
        """
        Batch create combined figures (1D+2D) for multiple fields.

        Convenience wrapper around plot_combined_localized_figure that processes
        multiple fields with the same spatial configuration. Creates publication-quality
        multi-panel figures for each field.

        Args:
            dataset: YT dataset to plot
            fields: List of field configurations, each containing:
                   {'field': str, 'colormap': str, 'log_scale': bool, 'levels': int, 'zlim': tuple}
            center_point: Center point coordinates [x, y, z] in dataset units
            forward_bound: Distance forward from center along normal_axis
            backward_bound: Distance backward from center along normal_axis
            extraction_y: Y-coordinate for extracting the 1D ray (in cm, dataset units)
            output_path: Base directory for organized output
            axis: Axis normal to plotting plane ('x', 'y', 'z' or 0, 1, 2)
            normal_axis: Reference axis for forward/backward bounds ('x', 'y', or 'z')
            contour_lines: Optional dictionary of contour lines to overlay on all plots
            auto_organize: If True, organize as output_path/FieldName/plt00100_combined.png
            time_offset: Time offset in seconds to add to dataset time (for correcting restart times, etc.)

        Example:
            >>> plotter = YTFieldPlotter(figure_size=(12, 10), dpi=150)
            >>> fields_to_plot = [
            ...     {'field': 'Temperature', 'colormap': 'plasma', 'log_scale': False},
            ...     {'field': 'Pressure', 'colormap': 'viridis', 'log_scale': False},
            ... ]
            >>> plotter.plot_multiple_combined_figures(
            ...     dataset=ds,
            ...     fields=fields_to_plot,
            ...     center_point=[flame_x_cm, extraction_y_cm, 0.0],
            ...     forward_bound=0.1,  # 1mm in cm
            ...     backward_bound=1.0,  # 10mm in cm
            ...     extraction_y=extraction_y_cm,
            ...     output_path='./output/Combined-Figures',
            ...     axis='z',
            ...     normal_axis='x'
            ... )
        """
        if not self.yt_available:
            raise PlotGenerationError("multiple_combined_figures", "YT not available")

        print(f"Creating combined figures for {len(fields)} fields...")

        for i, field_config in enumerate(fields):
            try:
                field_name = field_config['field']
                print(f"  [{i+1}/{len(fields)}] Creating combined figure for {field_name}...")

                # Call the combined figure method for each field
                self.plot_combined_localized_figure(
                    dataset=dataset,
                    field=field_name,
                    center_point=center_point,
                    forward_bound=forward_bound,
                    backward_bound=backward_bound,
                    extraction_y=extraction_y,
                    output_path=output_path,
                    axis=axis,
                    normal_axis=normal_axis,
                    levels=field_config.get('levels', 50),
                    colormap=field_config.get('colormap', 'viridis'),
                    log_scale=field_config.get('log_scale', False),
                    zlim=field_config.get('zlim', None),
                    title=None,  # Auto-generated title
                    contour_lines=contour_lines,
                    auto_organize=auto_organize,
                    time_offset=time_offset
                )
                print(f"    [OK] {field_name}")

            except Exception as e:
                print(f"    [FAILED] {field_config['field']}: {e}")

        print(f"Completed creating {len(fields)} combined figures")


def create_yt_field_plotter(figure_size: Tuple[int, int] = (12, 8), dpi: int = 150) -> YTFieldPlotter:
    """
    Factory function to create a YT Field Plotter.

    Args:
        figure_size: Figure size in inches
        dpi: Resolution in dots per inch

    Returns:
        Configured YTFieldPlotter instance
    """
    return YTFieldPlotter(figure_size, dpi)
