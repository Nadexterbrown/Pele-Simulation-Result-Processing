"""
Additional analysis modules for advanced processing.
"""

# Import Chapman-Jouguet functionality
from .chapman_jouguet import (
    CJAnalyzer,
    create_chapman_jouguet_analyzer
)

# Import flame analysis functionality
from .flame import (
    CanteraLaminarFlame,
    CanteraBurningRate,
    FlameContourAnalyzer
)

# Import flame baseline functionality
from .flame_baseline import (
    FlameBaselineFitter,
    FlameBaselineResult,
    create_synthetic_flame
)

__all__ = [
    # CJ Analyzer and factory
    'CJAnalyzer',
    'create_chapman_jouguet_analyzer',

    # Flame analysis
    'CanteraLaminarFlame',
    'CanteraBurningRate',
    'FlameContourAnalyzer',

    # Flame baseline fitting
    'FlameBaselineFitter',
    'FlameBaselineResult',
    'create_synthetic_flame'
]