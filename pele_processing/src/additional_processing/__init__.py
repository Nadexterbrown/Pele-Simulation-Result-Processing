"""
Additional processing modules for advanced PeleC simulation analysis.
"""

# Import core components
from .core import (
    ChapmanJouguetAnalyzer,
    CJProperties,
    CJType
)

# Import analysis implementations
from .analysis import (
    CJAnalyzer,
    create_chapman_jouguet_analyzer
)

__all__ = [
    # Interfaces
    'ChapmanJouguetAnalyzer',

    # Domain models
    'CJProperties',
    'CJType',

    # Implementations
    'CJAnalyzer',
    'create_chapman_jouguet_analyzer'
]