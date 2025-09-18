"""
Additional analysis modules for advanced processing.
"""

# Import existing Chapman-Jouguet functionality
from .chapmam_jouguet import (
    CJState,
    CJSolver,
    CJDetonation,
    CJDeflagration
)

# Import interface implementations
from .chapman_jouguet import (
    DetonationAnalyzer,
    DeflagrationAnalyzer,
    UnifiedCJAnalyzer,
    CJParametricAnalyzerImpl,
    create_cj_analyzer,
    create_parametric_analyzer
)

__all__ = [
    # Original CJ classes
    'CJState',
    'CJSolver',
    'CJDetonation',
    'CJDeflagration',

    # Interface implementations
    'DetonationAnalyzer',
    'DeflagrationAnalyzer',
    'UnifiedCJAnalyzer',
    'CJParametricAnalyzerImpl',
    'create_cj_analyzer',
    'create_parametric_analyzer'
]