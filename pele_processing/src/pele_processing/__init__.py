"""
Pele Processing Package - Separate submodules for base and additional processing.

This package provides two distinct submodules:
- pele_processing.base_processing: Core Pele simulation processing functionality
- pele_processing.additional_processing: Advanced analysis tools (Chapman-Jouguet, etc.)

Usage:
    from pele_processing.base_processing import PeleFlameAnalyzer, create_flame_analyzer
    from pele_processing.additional_processing import CJAnalyzer, create_chapman_jouguet_analyzer

Note: To avoid naming conflicts, components are ONLY available through their respective
submodules. Direct imports from pele_processing are not supported.
"""

import sys

# Import submodules
import base_processing
import additional_processing

# Register submodules in sys.modules so they can be imported as pele_processing.base_processing
sys.modules['pele_processing.base_processing'] = base_processing
sys.modules['pele_processing.additional_processing'] = additional_processing

# Import version info from base_processing for package metadata only
from base_processing import __version__, __author__, __email__, __description__, __url__

# Only expose the submodules and version info, nothing else
__all__ = ['base_processing', 'additional_processing', '__version__', '__author__', '__email__', '__description__', '__url__']