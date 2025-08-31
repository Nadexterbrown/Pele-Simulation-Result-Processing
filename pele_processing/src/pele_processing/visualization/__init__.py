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
    'ContourVisualizer'
]

__version__ = "1.0.0"