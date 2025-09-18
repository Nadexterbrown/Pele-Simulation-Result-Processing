"""
Core components for additional processing modules.
"""

from .interfaces import (
    ChapmanJouguetAnalyzer
)

from .domain import (
    CJProperties,
    CJType
)

__all__ = [
    # Interfaces
    'ChapmanJouguetAnalyzer',

    # Domain models
    'CJProperties',
    'CJType'
]