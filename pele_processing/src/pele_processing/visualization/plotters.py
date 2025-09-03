"""
General plotting functions for Pele processing system.
All functions are generic and don't hardcode any specific data.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod

from ..core.interfaces import FrameGenerator
from ..core.domain import AnimationFrame
from ..core.exceptions import PlotGenerationError


class BasePlotter(ABC):
    """Base class for all plotters with common functionality."""
    
    def __init__(self, figure_size: Tuple[int, int] = (10, 6), dpi: int = 150):
        self.figure_size = figure_size
        self.dpi = dpi
    
    def _setup_figure(self, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create and setup a matplotlib figure and axes."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        if title:
            ax.set_title(title)
        return fig, ax
    
    def _save_and_close(self, fig: plt.Figure, output_path: Path) -> None:
        """Save figure and close to free memory."""
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)


class LinePlotter(BasePlotter):
    """General line plotting functionality."""
    
    def plot_single_line(self, x_data: np.ndarray, y_data: np.ndarray, 
                        output_path: Path, title: str = None, 
                        xlabel: str = 'X', ylabel: str = 'Y',
                        line_style: str = '-', line_color: str = 'blue',
                        line_width: float = 2.0, **kwargs) -> None:
        """Plot a single line with customizable styling."""
        fig, ax = self._setup_figure(title)
        
        ax.plot(x_data, y_data, linestyle=line_style, color=line_color, 
                linewidth=line_width, **kwargs)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        self._save_and_close(fig, output_path)
    
    def plot_multiple_lines(self, data_dict: Dict[str, Dict[str, np.ndarray]], 
                           x_key: str, y_key: str, output_path: Path,
                           title: str = None, xlabel: str = None, ylabel: str = None,
                           colors: List[str] = None, styles: List[str] = None,
                           **kwargs) -> None:
        """Plot multiple lines from a data dictionary."""
        fig, ax = self._setup_figure(title)
        
        # Default colors and styles
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
        if styles is None:
            styles = ['-'] * len(data_dict)
        
        for i, (series_name, data) in enumerate(data_dict.items()):
            x_data = data[x_key]
            y_data = data[y_key]
            color = colors[i % len(colors)]
            style = styles[i % len(styles)]
            
            ax.plot(x_data, y_data, color=color, linestyle=style, 
                   label=series_name, linewidth=2, **kwargs)
        
        ax.set_xlabel(xlabel or x_key)
        ax.set_ylabel(ylabel or y_key)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self._save_and_close(fig, output_path)


class ContourPlotter(BasePlotter):
    """General contour plotting functionality."""
    
    def plot_contour_field(self, x_data: np.ndarray, y_data: np.ndarray, 
                          field_data: np.ndarray, output_path: Path,
                          title: str = None, xlabel: str = 'X', ylabel: str = 'Y',
                          field_label: str = 'Field', colormap: str = 'viridis',
                          levels: int = 50, **kwargs) -> None:
        """Create a contour plot of a field."""
        fig, ax = self._setup_figure(title)
        
        contour = ax.tricontourf(x_data, y_data, field_data, levels=levels, cmap=colormap, **kwargs)
        plt.colorbar(contour, ax=ax, label=field_label)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal', adjustable='box')
        
        self._save_and_close(fig, output_path)
    
    def plot_contour_with_lines(self, x_data: np.ndarray, y_data: np.ndarray,
                               field_data: np.ndarray, line_data: Dict[str, np.ndarray],
                               output_path: Path, title: str = None,
                               xlabel: str = 'X', ylabel: str = 'Y',
                               field_label: str = 'Field', colormap: str = 'viridis',
                               levels: int = 50, **kwargs) -> None:
        """Create a contour plot with overlay lines."""
        fig, ax = self._setup_figure(title)
        
        # Create contour
        contour = ax.tricontourf(x_data, y_data, field_data, levels=levels, cmap=colormap, **kwargs)
        plt.colorbar(contour, ax=ax, label=field_label)
        
        # Add lines
        for line_name, line_coords in line_data.items():
            if line_coords is not None and len(line_coords) > 0:
                ax.plot(line_coords[:, 0], line_coords[:, 1], 
                       label=line_name, linewidth=2)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Set axis limits based on data bounds
        x_margin = (np.max(x_data) - np.min(x_data)) * 0.02
        y_margin = (np.max(y_data) - np.min(y_data)) * 0.02
        ax.set_xlim(np.min(x_data) - x_margin, np.max(x_data) + x_margin)
        ax.set_ylim(np.min(y_data) - y_margin, np.max(y_data) + y_margin)
        
        # Set equal aspect ratio after setting limits
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self._save_and_close(fig, output_path)


class ScatterPlotter(BasePlotter):
    """General scatter plotting functionality."""
    
    def plot_scatter(self, x_data: np.ndarray, y_data: np.ndarray,
                    output_path: Path, title: str = None,
                    xlabel: str = 'X', ylabel: str = 'Y',
                    color_data: np.ndarray = None, color_label: str = 'Value',
                    marker: str = 'o', size: float = 50, 
                    colormap: str = 'viridis', **kwargs) -> None:
        """Create a scatter plot with optional color mapping."""
        fig, ax = self._setup_figure(title)
        
        if color_data is not None:
            scatter = ax.scatter(x_data, y_data, c=color_data, s=size, 
                               marker=marker, cmap=colormap, **kwargs)
            plt.colorbar(scatter, ax=ax, label=color_label)
        else:
            ax.scatter(x_data, y_data, s=size, marker=marker, **kwargs)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        self._save_and_close(fig, output_path)


class HistogramPlotter(BasePlotter):
    """General histogram plotting functionality."""
    
    def plot_histogram(self, data: np.ndarray, output_path: Path,
                      title: str = None, xlabel: str = 'Value', 
                      ylabel: str = 'Frequency', bins: Union[int, str] = 30,
                      color: str = 'blue', alpha: float = 0.7,
                      **kwargs) -> None:
        """Create a histogram plot."""
        fig, ax = self._setup_figure(title)
        
        ax.hist(data, bins=bins, color=color, alpha=alpha, **kwargs)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        self._save_and_close(fig, output_path)


class SubplotPlotter(BasePlotter):
    """General subplot functionality."""
    
    def plot_subplots(self, data_dict: Dict[str, Dict[str, Any]], 
                     output_path: Path, subplot_layout: Tuple[int, int] = None,
                     title: str = None, **kwargs) -> None:
        """Create subplots from data dictionary."""
        n_plots = len(data_dict)
        
        if subplot_layout is None:
            ncols = int(np.ceil(np.sqrt(n_plots)))
            nrows = int(np.ceil(n_plots / ncols))
        else:
            nrows, ncols = subplot_layout
        
        fig, axes = plt.subplots(nrows, ncols, figsize=self.figure_size, dpi=self.dpi)
        if title:
            fig.suptitle(title)
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        for i, (plot_name, plot_data) in enumerate(data_dict.items()):
            ax = axes[i]
            
            # Extract plot parameters
            x_data = plot_data.get('x_data')
            y_data = plot_data.get('y_data')
            plot_type = plot_data.get('plot_type', 'line')
            plot_title = plot_data.get('title', plot_name)
            xlabel = plot_data.get('xlabel', 'X')
            ylabel = plot_data.get('ylabel', 'Y')
            
            # Create appropriate plot
            if plot_type == 'line':
                ax.plot(x_data, y_data, **plot_data.get('plot_kwargs', {}))
            elif plot_type == 'scatter':
                ax.scatter(x_data, y_data, **plot_data.get('plot_kwargs', {}))
            elif plot_type == 'bar':
                ax.bar(x_data, y_data, **plot_data.get('plot_kwargs', {}))
            
            ax.set_title(plot_title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        self._save_and_close(fig, output_path)


class AnimationPlotter(BasePlotter):
    """Animation frame plotting functionality."""
    
    def create_animation_frame(self, x_data: np.ndarray, y_data: np.ndarray,
                             output_path: Path, title: str = None,
                             xlabel: str = 'X', ylabel: str = 'Y',
                             timestamp: float = None, markers: Dict[str, float] = None,
                             **kwargs) -> None:
        """Create a single animation frame."""
        fig, ax = self._setup_figure()
        
        # Main data plot
        ax.plot(x_data, y_data, **kwargs)
        
        # Add markers (like flame position, shock position, etc.)
        if markers:
            for marker_name, marker_pos in markers.items():
                if marker_pos is not None:
                    ax.axvline(x=marker_pos, linestyle='--', alpha=0.7, 
                              label=marker_name)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add title with timestamp if provided
        frame_title = title or 'Animation Frame'
        if timestamp is not None:
            frame_title += f' (t = {timestamp:.6f})'
        ax.set_title(frame_title)
        
        ax.grid(True, alpha=0.3)
        if markers:
            ax.legend()
        
        self._save_and_close(fig, output_path)


class StandardPlotter(FrameGenerator, LinePlotter, ContourPlotter, AnimationPlotter):
    """Standard plotter combining all plotting capabilities."""
    
    def __init__(self, figure_size: Tuple[int, int] = (12, 8), dpi: int = 150):
        super().__init__(figure_size, dpi)
    
    def create_field_plot(self, frame: AnimationFrame) -> None:
        """Create field plot from AnimationFrame - required by FrameGenerator interface."""
        try:
            markers = {}
            if hasattr(frame, 'flame_position') and frame.flame_position is not None:
                markers['Flame'] = frame.flame_position
            if hasattr(frame, 'shock_position') and frame.shock_position is not None:
                markers['Shock'] = frame.shock_position
            
            self.create_animation_frame(
                x_data=frame.x_data,
                y_data=frame.y_data,
                output_path=frame.output_path,
                title=frame.field_name,
                timestamp=frame.timestamp,
                markers=markers,
                linewidth=2,
                color='blue'
            )
        except Exception as e:
            raise PlotGenerationError(f"field plot for {frame.field_name}", str(e))
    
    def create_multi_field_plot(self, x_data: np.ndarray, field_data: Dict[str, np.ndarray],
                               output_path: Path, title: str = None, xlabel: str = 'X',
                               **kwargs) -> None:
        """Create plot with multiple fields - required by FrameGenerator interface."""
        try:
            plot_data = {}
            for field_name, y_data in field_data.items():
                plot_data[field_name] = {'x_data': x_data, 'y_data': y_data}
            
            self.plot_multiple_lines(plot_data, 'x_data', 'y_data', output_path, 
                                   title=title, xlabel=xlabel, **kwargs)
        except Exception as e:
            raise PlotGenerationError("multi-field plot", str(e))
    
    def create_specialized_plot(self, dataset: Any, plot_type: str, output_path: Path, 
                               **kwargs) -> None:
        """Create specialized plots - required by FrameGenerator interface."""
        try:
            if plot_type == "contour":
                self._create_contour_specialized(dataset, output_path, **kwargs)
            elif plot_type == "surface":
                self._create_surface_specialized(dataset, output_path, **kwargs)
            else:
                raise ValueError(f"Unknown specialized plot type: {plot_type}")
        except Exception as e:
            raise PlotGenerationError(f"specialized plot: {plot_type}", str(e))
    
    def _create_contour_specialized(self, dataset: Any, output_path: Path, **kwargs) -> None:
        """Create specialized contour plot."""
        field = kwargs.get('field', 'Temp')
        contour_points = kwargs.get('contour_points')
        normal_line = kwargs.get('normal_line')
        thickness_val = kwargs.get('thickness_val')
        timestamp = kwargs.get('timestamp')
        
        # Extract field data from dataset
        max_level = dataset.index.max_level
        x_data, y_data, field_data = [], [], []
        
        for grid in dataset.index.grids:
            if grid.Level == max_level:
                x_grid = grid["boxlib", "x"].to_value().flatten() / 100  # cm to m
                y_grid = grid["boxlib", "y"].to_value().flatten() / 100  # cm to m
                field_grid = grid[field].flatten()
                
                x_data.extend(x_grid)
                y_data.extend(y_grid)
                field_data.extend(field_grid)
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        field_data = np.array(field_data)
        
        # Prepare line data
        line_data = {}
        if contour_points is not None and len(contour_points) > 0:
            line_data['Flame Contour'] = contour_points
        if normal_line is not None and len(normal_line) > 0:
            line_data['Normal Line'] = normal_line
        
        # Create title
        title = f'Flame Analysis - {dataset.basename}'
        if timestamp is not None:
            title += f' (t = {timestamp:.6f})'
        if thickness_val is not None:
            title += f' - Thickness: {thickness_val:.2e} m'
        
        # Create plot
        self.plot_contour_with_lines(
            x_data, y_data, field_data, line_data, output_path,
            title=title, xlabel='X [m]', ylabel='Y [m]',
            field_label=f'{field} [K]' if field == 'Temp' else field,
            colormap='hot', levels=50
        )
    
    def _create_surface_specialized(self, dataset: Any, output_path: Path, **kwargs) -> None:
        """Create specialized surface plot."""
        # Implementation for surface plots
        pass


class LocalViewPlotter(StandardPlotter):
    """Plotter for localized views around features."""
    
    def create_local_view(self, frame: AnimationFrame, center: float, 
                         window_size: float) -> None:
        """Create localized view around a feature."""
        try:
            # Calculate window bounds
            x_min = center - window_size / 2
            x_max = center + window_size / 2
            
            # Filter data to window
            mask = (frame.x_data >= x_min) & (frame.x_data <= x_max)
            x_windowed = frame.x_data[mask]
            y_windowed = frame.y_data[mask]
            
            if len(x_windowed) == 0:
                print(f"No data points in local window [{x_min:.6f}, {x_max:.6f}]")
                return
            
            # Create markers for local view
            markers = {}
            if hasattr(frame, 'flame_position') and frame.flame_position is not None:
                if x_min <= frame.flame_position <= x_max:
                    markers['Flame'] = frame.flame_position
            if hasattr(frame, 'shock_position') and frame.shock_position is not None:
                if x_min <= frame.shock_position <= x_max:
                    markers['Shock'] = frame.shock_position
            
            self.create_animation_frame(
                x_data=x_windowed,
                y_data=y_windowed,
                output_path=frame.output_path,
                title=f"Local View - {frame.field_name}",
                timestamp=frame.timestamp,
                markers=markers,
                linewidth=2,
                color='red'
            )
        except Exception as e:
            raise PlotGenerationError(f"local view for {frame.field_name}", str(e))


class StatisticalPlotter(SubplotPlotter, HistogramPlotter):
    """Plotter for statistical analysis and summaries."""
    
    def plot_time_series(self, time_data: np.ndarray, value_data: np.ndarray,
                        value_name: str, output_path: Path, **kwargs) -> None:
        """Plot time series data."""
        self.plot_single_line(
            time_data, value_data, output_path,
            title=f'{value_name} vs Time',
            xlabel='Time [s]', ylabel=value_name,
            **kwargs
        )
    
    def plot_parameter_sweep(self, parameter_values: np.ndarray,
                           results: Dict[str, np.ndarray],
                           output_path: Path, parameter_name: str = "Parameter") -> None:
        """Plot results of parameter sweep."""
        subplot_data = {}
        for i, (result_name, result_values) in enumerate(results.items()):
            subplot_data[result_name] = {
                'x_data': parameter_values,
                'y_data': result_values,
                'plot_type': 'line',
                'xlabel': parameter_name,
                'ylabel': result_name,
                'plot_kwargs': {'marker': 'o', 'markersize': 4}
            }
        
        self.plot_subplots(subplot_data, output_path, 
                          title=f'{parameter_name} Sweep Results')


class ComparisonPlotter(LinePlotter, SubplotPlotter):
    """Plotter for comparing multiple datasets or conditions."""
    
    def plot_comparison(self, comparison_data: Dict[str, Dict[str, np.ndarray]],
                       x_key: str, y_key: str, output_path: Path,
                       title: str = "Comparison", **kwargs) -> None:
        """Plot comparison of multiple data series."""
        self.plot_multiple_lines(comparison_data, x_key, y_key, output_path,
                               title=title, **kwargs)
    
    def plot_multiple_series(self, data: Dict[str, Dict[str, np.ndarray]],
                           x_key: str, y_key: str, output_path: Path,
                           title: str = "Multiple Series", **kwargs) -> None:
        """Plot multiple data series on the same axes."""
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        
        for series_name, series_data in data.items():
            if x_key in series_data and y_key in series_data:
                plt.plot(series_data[x_key], series_data[y_key], 
                        label=series_name, **kwargs.get('plot_kwargs', {}))
        
        plt.xlabel(kwargs.get('xlabel', x_key))
        plt.ylabel(kwargs.get('ylabel', y_key))
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()