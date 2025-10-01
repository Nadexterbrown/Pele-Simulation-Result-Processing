"""
Data loading implementations for the Pele post processing system.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

from base_processing.core.domain import (
    ProcessingResult, FlameProperties, ShockProperties,
    PressureWaveProperties, GasProperties, ThermodynamicState
)

class CSVDataLoader:
    """CSV-based data loader for Pele post-processing datasets."""

    def load_dataset(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load dataset from a CSV file."""
        try:
            data = pd.read_csv(path)
            return data
        except Exception as e:
            raise IOError(f"Failed to load dataset from {path}: {e}")

    def get_dataset_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata from the DataFrame."""
        try:
            info = {
                "num_rows": len(data),
                "num_columns": len(data.columns),
                "columns": list(data.columns)
            }
            return info
        except Exception as e:
            raise ValueError(f"Failed to extract metadata: {e}")

    def list_available_fields(self, data: pd.DataFrame) -> List[str]:
        """List available fields (columns) in the DataFrame."""
        try:
            return list(data.columns)
        except Exception as e:
            raise ValueError(f"Failed to list fields: {e}")


class ProcessedResultsV3Loader:
    """Loader for Processed-MPI-Global-Results V3.0.0 format."""

    def __init__(self):
        self.columns = []
        self.column_groups = {}

    def load_dataset(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load V3.0.0 format processed results."""
        try:
            path = Path(path)

            # Read the file with specific parsing for the V3.0.0 format
            with open(path, 'r') as f:
                lines = f.readlines()

            # Extract column names from the second line
            if len(lines) < 3:
                raise ValueError("File does not contain enough lines for V3.0.0 format")

            # Parse header line (second line) - headers are separated by multiple spaces
            header_line = lines[1].strip('#').strip()
            # Split by multiple spaces and filter out empty strings
            columns = [col.strip() for col in header_line.split('  ') if col.strip()]

            # Store columns for later use
            self.columns = columns

            # Dynamically create column groups based on prefixes
            self._create_column_groups(columns)

            # Read the data starting from line 3
            data_lines = []
            for line in lines[2:]:
                if line.strip() and not line.startswith('#'):
                    values = line.strip().split()
                    if len(values) == len(columns):
                        data_lines.append(values)

            # Create DataFrame
            df = pd.DataFrame(data_lines, columns=columns)

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows with NaN values
            df = df.dropna()

            return df

        except Exception as e:
            raise IOError(f"Failed to load V3.0.0 dataset from {path}: {e}")

    def _create_column_groups(self, columns: List[str]):
        """Dynamically create column groups based on column prefixes."""
        self.column_groups = {
            'time': [],
            'flame': [],
            'burned_gas': [],
            'shock': [],
            'pressure_wave': [],
            'other': []
        }

        for col in columns:
            col_lower = col.lower()
            if 'time' in col_lower:
                self.column_groups['time'].append(col)
            elif col.startswith('Flame '):
                self.column_groups['flame'].append(col)
            elif col.startswith('Burned Gas '):
                self.column_groups['burned_gas'].append(col)
            elif col.startswith('Shock '):
                self.column_groups['shock'].append(col)
            elif col.startswith('Pressure Wave '):
                self.column_groups['pressure_wave'].append(col)
            else:
                self.column_groups['other'].append(col)

        # Remove empty groups
        self.column_groups = {k: v for k, v in self.column_groups.items() if v}

    def get_dataset_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata from the V3.0.0 DataFrame."""
        try:
            time_col = None
            for col in data.columns:
                if 'time' in col.lower():
                    time_col = col
                    break

            info = {
                "format_version": "V3.0.0",
                "num_timesteps": len(data),
                "num_columns": len(data.columns),
                "time_range": (data[time_col].min(), data[time_col].max()) if time_col else (None, None),
                "column_groups": self.column_groups,
                "columns": list(data.columns)
            }
            return info
        except Exception as e:
            raise ValueError(f"Failed to extract metadata: {e}")

    def get_column_group(self, data: pd.DataFrame, group_name: str) -> pd.DataFrame:
        """Extract a specific group of columns from the dataset."""
        if group_name not in self.column_groups:
            raise ValueError(f"Unknown column group: {group_name}. Available groups: {list(self.column_groups.keys())}")

        columns = self.column_groups[group_name]
        available_columns = [col for col in columns if col in data.columns]

        if not available_columns:
            raise ValueError(f"No columns from group '{group_name}' found in dataset")

        return data[available_columns]

    def list_available_fields(self, data: pd.DataFrame) -> List[str]:
        """List available fields (columns) in the DataFrame."""
        return list(data.columns)

    def get_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get statistical summary for each numeric column."""
        stats = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats[col] = {
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'median': data[col].median()
                }
        return stats

    def get_processing_results(self, data: pd.DataFrame) -> List[ProcessingResult]:
        """Convert DataFrame rows to ProcessingResult objects."""
        results = []

        for idx, row in data.iterrows():
            # Create flame properties
            flame_props = self._create_flame_properties(row)

            # Create shock properties
            shock_props = self._create_shock_properties(row)

            # Create pressure wave properties
            pressure_wave_props = self._create_pressure_wave_properties(row)

            # Create burned gas properties
            burned_gas_props = self._create_burned_gas_properties(row)

            # Create result object (simplified without DatasetInfo)
            result = ProcessingResult(
                dataset_info=None,  # Not available in V3 format
                flame_data=flame_props,
                shock_data=shock_props,
                burned_gas_data=burned_gas_props,
                processing_time=row.get('Time', 0.0),
                success=True
            )
            results.append(result)

        return results

    def _create_flame_properties(self, row: pd.Series) -> Optional[FlameProperties]:
        """Create FlameProperties from a DataFrame row."""
        if 'Flame Position [m]' not in row:
            return None

        # Create thermodynamic state
        thermo_state = None
        if 'Flame Thermodynamic Temperature [K]' in row:
            thermo_state = ThermodynamicState(
                temperature=row.get('Flame Thermodynamic Temperature [K]', np.nan),
                pressure=row.get('Flame Thermodynamic Pressure [kg / m / s^2]', np.nan),
                density=row.get('Flame Thermodynamic Density [kg / m^3]', np.nan),
                sound_speed=row.get('Flame Thermodynamic Sound Speed', np.nan)
            )

        return FlameProperties(
            position=row.get('Flame Position [m]', np.nan),
            index=row.get('Flame Index', None),
            velocity=row.get('Flame Velocity [m / s]', None),
            relative_velocity=row.get('Flame Relative Velocity [m / s]', None),
            thickness=row.get('Flame Flame Thickness', None),
            surface_length=row.get('Flame Surface Length [m]', None),
            heat_release_rate=row.get('Flame HRR', None),
            consumption_rate=row.get('Flame Consumption Rate [kg / s]', None),
            burning_velocity=row.get('Flame Burning Velocity [m / s]', None),
            thermodynamic_state=thermo_state
        )

    def _create_shock_properties(self, row: pd.Series) -> Optional[ShockProperties]:
        """Create ShockProperties from a DataFrame row."""
        if 'Shock Position [m]' not in row:
            return None

        # Create pre-shock thermodynamic state
        pre_shock_state = None
        if 'Shock PreShockThermodynamicState Temperature [K]' in row:
            pre_shock_state = ThermodynamicState(
                temperature=row.get('Shock PreShockThermodynamicState Temperature [K]', np.nan),
                pressure=row.get('Shock PreShockThermodynamicState Pressure [kg / m / s^2]', np.nan),
                density=row.get('Shock PreShockThermodynamicState Density [kg / m^3]', np.nan),
                sound_speed=row.get('Shock PreShockThermodynamicState Sound Speed', np.nan)
            )

        # Create post-shock thermodynamic state
        post_shock_state = None
        if 'Shock PostShockThermodynamicState Temperature [K]' in row:
            post_shock_state = ThermodynamicState(
                temperature=row.get('Shock PostShockThermodynamicState Temperature [K]', np.nan),
                pressure=row.get('Shock PostShockThermodynamicState Pressure [kg / m / s^2]', np.nan),
                density=row.get('Shock PostShockThermodynamicState Density [kg / m^3]', np.nan),
                sound_speed=row.get('Shock PostShockThermodynamicState Sound Speed', np.nan)
            )

        return ShockProperties(
            position=row.get('Shock Position [m]', np.nan),
            index=row.get('Shock Index', None),
            velocity=row.get('Shock Velocity [m / s]', None),
            pre_shock_state=pre_shock_state,
            post_shock_state=post_shock_state
        )

    def _create_pressure_wave_properties(self, row: pd.Series) -> Optional[PressureWaveProperties]:
        """Create PressureWaveProperties from a DataFrame row."""
        if 'Pressure Wave Position [m]' not in row:
            return None

        # Create thermodynamic state
        thermo_state = None
        if 'Pressure Wave Thermodynamic Temperature [K]' in row:
            thermo_state = ThermodynamicState(
                temperature=row.get('Pressure Wave Thermodynamic Temperature [K]', np.nan),
                pressure=row.get('Pressure Wave Thermodynamic Pressure [kg / m / s^2]', np.nan),
                density=row.get('Pressure Wave Thermodynamic Density [kg / m^3]', np.nan),
                sound_speed=row.get('Pressure Wave Thermodynamic Sound Speed', np.nan)
            )

        return PressureWaveProperties(
            position=row.get('Pressure Wave Position [m]', np.nan),
            index=row.get('Pressure Wave Index', None),
            thermodynamic_state=thermo_state
        )

    def _create_burned_gas_properties(self, row: pd.Series) -> Optional[GasProperties]:
        """Create burned GasProperties from a DataFrame row."""
        if 'Burned Gas Gas Velocity [m / s]' not in row:
            return None

        # Create thermodynamic state
        thermo_state = None
        if 'Burned Gas Thermodynamic Temperature [K]' in row:
            thermo_state = ThermodynamicState(
                temperature=row.get('Burned Gas Thermodynamic Temperature [K]', np.nan),
                pressure=row.get('Burned Gas Thermodynamic Pressure [kg / m / s^2]', np.nan),
                density=row.get('Burned Gas Thermodynamic Density [kg / m^3]', np.nan),
                sound_speed=row.get('Burned Gas Thermodynamic Sound Speed', np.nan)
            )

        return GasProperties(
            velocity=row.get('Burned Gas Gas Velocity [m / s]', None),
            thermodynamic_state=thermo_state
        )

