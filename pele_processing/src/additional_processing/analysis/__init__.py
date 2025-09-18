"""
Additional analysis modules for advanced processing.
"""

# Import Chapman-Jouguet functionality from the new file
from .chapman_jouguet import (
    CJAnalyzer,
    create_chapman_jouguet_analyzer
)

__all__ = [
    # CJ Analyzer and factory
    'CJAnalyzer',
    'create_chapman_jouguet_analyzer'
]