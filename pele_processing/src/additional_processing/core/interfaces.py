"""
Additional Processing Interfaces

This module provides clean interfaces for additional analysis capabilities,
including Chapman-Jouguet analysis and other advanced processing features.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Tuple, List
from dataclasses import dataclass
import numpy as np

from .domain import (
    CJProperties
)


class ChapmanJouguetAnalyzer(ABC):
    """Interface for Chapman-Jouguet analysis."""

    @abstractmethod
    def analyze_cj_deflagration(self) -> CJProperties:
        """Perform comprehensive CJ deflagration analysis.

        Args:

        Returns:
            CJProperties: Data structure containing CJ deflagration properties.
        """
        pass

    @abstractmethod
    def analyze_cj_detonation(self) -> CJProperties:
        """Perform comprehensive CJ detonation analysis.

        Args:

        Returns:
            CJProperties: Data structure containing CJ detonation properties.
        """
        pass

