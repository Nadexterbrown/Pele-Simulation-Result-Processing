"""
Shock wave analysis for the Pele processing system.
"""
from typing import Tuple, Dict, Any
import numpy as np

from ..core.interfaces import ShockAnalyzer, WaveTracker
from ..core.domain import ShockProperties, FieldData, WaveType, ThermodynamicState
from ..core.exceptions import ShockAnalysisError, WaveNotFoundError


class PeleShockAnalyzer(ShockAnalyzer, WaveTracker):
    """Shock analysis implementation for Pele datasets."""

    def __init__(self, pressure_ratio_threshold: float = 1.01):
        self.pressure_ratio_threshold = pressure_ratio_threshold

    def analyze_shock_properties(self, data: FieldData) -> ShockProperties:
        """Complete shock analysis."""
        # Find shock position
        shock_idx, shock_pos = self.find_wave_position(data, WaveType.SHOCK)

        properties = ShockProperties(position=shock_pos, index=shock_idx)

        # Calculate pre/post shock states
        try:
            properties.pre_shock_state = self._extract_pre_shock_state(data, shock_idx)
            properties.post_shock_state = self._extract_post_shock_state(data, shock_idx)
        except Exception as e:
            print(f"Shock state extraction failed: {e}")

        return properties

    def find_wave_position(self, data: FieldData, wave_type: WaveType) -> Tuple[int, float]:
        """Find shock position using pressure jump."""
        if wave_type != WaveType.SHOCK:
            raise WaveNotFoundError(wave_type.value, "Only shock detection supported")

        # Calculate pressure ratio relative to upstream
        upstream_pressure = data.pressure[-1]  # Last point (upstream)
        pressure_ratios = data.pressure / upstream_pressure

        # Find first point above threshold
        shock_candidates = np.where(pressure_ratios >= self.pressure_ratio_threshold)[0]

        if len(shock_candidates) == 0:
            raise WaveNotFoundError("shock", f"No pressure jump above {self.pressure_ratio_threshold}")

        shock_idx = shock_candidates[-1]  # Most downstream shock
        shock_position = data.coordinates[shock_idx]

        return shock_idx, shock_position

    def calculate_wave_velocity(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate shock velocity from position time series."""
        if len(positions) < 2:
            return np.array([])
        return np.gradient(positions, times)

    def calculate_rankine_hugoniot_relations(self, pre_state: ThermodynamicState,
                                             post_state: ThermodynamicState) -> Dict[str, float]:
        """Verify shock using Rankine-Hugoniot relations."""
        try:
            # Mass conservation: ρ₁u₁ = ρ₂u₂
            # Momentum conservation: P₁ + ρ₁u₁² = P₂ + ρ₂u₂²
            # Energy conservation: h₁ + u₁²/2 = h₂ + u₂²/2

            # Calculate pressure ratio
            pressure_ratio = post_state.pressure / pre_state.pressure

            # Calculate density ratio
            density_ratio = post_state.density / pre_state.density

            # Calculate temperature ratio
            temperature_ratio = post_state.temperature / pre_state.temperature

            # Mach number estimation (assuming ideal gas)
            gamma = 1.4  # Heat capacity ratio for air
            mach_squared = (2 / (gamma - 1)) * (pressure_ratio - 1) / (gamma + 1)
            mach_number = np.sqrt(max(0, mach_squared))

            return {
                'pressure_ratio': pressure_ratio,
                'density_ratio': density_ratio,
                'temperature_ratio': temperature_ratio,
                'mach_number': mach_number,
                'shock_strength': pressure_ratio - 1
            }

        except Exception as e:
            raise ShockAnalysisError("rankine_hugoniot", str(e))

    def _extract_pre_shock_state(self, data: FieldData, shock_idx: int) -> ThermodynamicState:
        """Extract pre-shock thermodynamic state."""
        # Use point upstream of shock
        upstream_idx = min(shock_idx + 5, len(data.coordinates) - 1)

        return ThermodynamicState(
            temperature=data.temperature[upstream_idx],
            pressure=data.pressure[upstream_idx],
            density=data.density[upstream_idx],
            sound_speed=np.sqrt(1.4 * data.pressure[upstream_idx] / data.density[upstream_idx])
        )

    def _extract_post_shock_state(self, data: FieldData, shock_idx: int) -> ThermodynamicState:
        """Extract post-shock thermodynamic state."""
        # Use point downstream of shock
        downstream_idx = max(shock_idx - 5, 0)

        return ThermodynamicState(
            temperature=data.temperature[downstream_idx],
            pressure=data.pressure[downstream_idx],
            density=data.density[downstream_idx],
            sound_speed=np.sqrt(1.4 * data.pressure[downstream_idx] / data.density[downstream_idx])
        )


def create_shock_analyzer(pressure_ratio_threshold: float = 1.01, **kwargs) -> ShockAnalyzer:
    """Factory for shock analyzers."""
    return PeleShockAnalyzer(pressure_ratio_threshold, **kwargs)