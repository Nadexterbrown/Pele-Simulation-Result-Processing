"""
Visualization module for the Pele processing system.
"""

# Plotters
from .plotters import (
    StandardPlotter, LocalViewPlotter, StatisticalPlotter, ComparisonPlotter
)

# Animators
from .animators import (
    FrameAnimator, BatchAnimator, InteractiveAnimator
)

# Formatters
from .formatters import (
    TableFormatter, CSVFormatter, JSONFormatter, create_formatter
)

# Specialized visualizations
from .specialized import (
    SchlierenVisualizer, StreamlineVisualizer, ContourVisualizer
)

# YT Field Plotter
from .yt_field_plotter import (
    YTFieldPlotter, PlotType, create_yt_field_plotter
)

__all__ = [
    # Plotters
    'StandardPlotter',
    'LocalViewPlotter',
    'StatisticalPlotter',
    'ComparisonPlotter',

    # Animators
    'FrameAnimator',
    'BatchAnimator',
    'InteractiveAnimator',

    # Formatters
    'TableFormatter',
    'CSVFormatter',
    'JSONFormatter',
    'create_formatter',

    # Specialized
    'SchlierenVisualizer',
    'StreamlineVisualizer',
    'ContourVisualizer',

    # YT Field Plotter
    'YTFieldPlotter',
    'PlotType',
    'create_yt_field_plotter'
]

__version__ = "1.0.0"