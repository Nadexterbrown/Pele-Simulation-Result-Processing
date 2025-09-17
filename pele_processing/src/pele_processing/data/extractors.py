"""
Data extraction implementations for the Pele processing system.
"""
from typing import Dict, List, Optional, Any
import numpy as np

from ..core.interfaces import DataExtractor, UnitConverter
from ..core.domain import FieldData, ThermodynamicState, Point2D, Direction, SpeciesData
from ..core.exceptions import DataExtractionError, FieldNotFoundError
from ..utils.constants import COMMON_SPECIES
from ..utils.pele_constants import PELE_VAR_MAP, get_missing_fields, add_species_vars

try:
    import cantera as ct

    CANTERA_AVAILABLE = True
except ImportError:
    ct = None
    CANTERA_AVAILABLE = False


class PeleDataExtractor(DataExtractor):
    """Pele-specific data extractor with unit conversion."""
    """Pele-specific data extractor with unit conversion."""

    def __init__(self, unit_converter: UnitConverter, mechanism_file: Optional[str] = None):
        self.unit_converter = unit_converter
        self.mechanism_file = mechanism_file
        self._gas = None
        self._field_map = self._create_field_map()

    @property
    def gas(self):
        if self._gas is None and self.mechanism_file and CANTERA_AVAILABLE:
            self._gas = ct.Solution(self.mechanism_file)
        return self._gas

    def extract_ray_data(self, dataset: Any, location: float, direction: Direction = Direction.X) -> FieldData:
        """Extract 1D ray data."""
        try:
            # Create ray
            if direction == Direction.Y:
                ray = dataset.ortho_ray(1, (0, location * 100))  # Convert to cm
            else:
                ray = dataset.ortho_ray(0, (location * 100, 0))

            sort_idx = np.argsort(ray["boxlib", 'x'])

            # Extract coordinates
            coords = self.unit_converter.convert_value(
                ray["boxlib", 'x'][sort_idx].to_value(), 'cm', 'm'
            )

            # Identify missing fields that need Cantera calculation
            field_names_in_data = [field_name for _, field_name in dataset.field_list]
            missing_fields = get_missing_fields(field_names_in_data)
            
            # Extract basic fields with fallback to Cantera
            temperature = ray["boxlib", "Temp"][sort_idx].to_value()
            pressure = self.unit_converter.convert_value(
                ray["boxlib", "pressure"][sort_idx].to_value(), 'dyne/cm^2', 'Pa'
            )
            
            # Extract species data first (needed for Cantera calculations)
            species_data = self._extract_species_data(ray, sort_idx)

            # Extract velocity data
            velocity_x, velocity_y = self._extract_velocity_data(ray, sort_idx)
            
            # Extract or calculate other variables
            density = self._get_density(ray, sort_idx, temperature, pressure, species_data, field_names_in_data, missing_fields)
            sound_speed = self._get_sound_speed(ray, sort_idx, temperature, pressure, species_data, field_names_in_data, missing_fields)
            heat_release_rate = self._get_heat_release_rate(ray, sort_idx, temperature, pressure, species_data, field_names_in_data, missing_fields)

            return FieldData(
                coordinates=coords,
                temperature=temperature,
                pressure=pressure,
                density=density,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                sound_speed=sound_speed,
                heat_release_rate=heat_release_rate,
                species_data=species_data
            )

        except Exception as e:
            raise DataExtractionError("ray", location, str(e))


    def extract_gas_velocity(self, dataset: Any, location: float, direction: Direction = Direction.X) -> float:
        """Extract gas velocity at specific location."""
        try:
            # Extract 1D ray data at the location
            if direction == Direction.Y:
                ray = dataset.ortho_ray(0, (location * 100, 0))  # Convert m to cm
                x_coords = ray["boxlib", 'x'].to_value()
                velocity_field = ray["x_velocity"].to_value()
            else:  # Direction.X
                ray = dataset.ortho_ray(1, (0, location * 100))  # Convert m to cm
                x_coords = ray["boxlib", 'y'].to_value()
                velocity_field = ray["y_velocity"].to_value()
            
            # Sort and find closest point
            sort_idx = np.argsort(x_coords)
            sorted_coords = x_coords[sort_idx]
            sorted_velocity = velocity_field[sort_idx]
            
            # Find target location in cm
            target_location_cm = location * 100
            target_idx = np.argmin(np.abs(sorted_coords - target_location_cm))
            
            # Convert from cm/s to m/s
            gas_velocity = self.unit_converter.convert_value(
                sorted_velocity[target_idx], 'cm/s', 'm/s'
            )
            
            return gas_velocity
            
        except Exception as e:
            print(f"Error extracting gas velocity: {e}")
            return 0.0

    def extract_thermodynamic_state(self, dataset: Any, location: Point2D) -> ThermodynamicState:
        """Extract thermodynamic state at point."""
        try:
            # Extract data at specific point (simplified)
            ray = dataset.ortho_ray(0, (location.y * 100, 0))
            sort_idx = np.argsort(ray["boxlib", 'x'])

            x_coords = ray["boxlib", 'x'][sort_idx].to_value()
            target_idx = np.argmin(np.abs(x_coords - location.x * 100))

            temp = ray["boxlib", "Temp"][sort_idx][target_idx].to_value()
            pressure = self.unit_converter.convert_value(
                ray["boxlib", "pressure"][sort_idx][target_idx].to_value(), 'dyne/cm^2', 'Pa'
            )

            # Calculate other properties if gas available
            if self.gas:
                species_data = self._extract_species_data(ray, sort_idx)
                if species_data and species_data.mass_fractions:
                    self.gas.TPY = temp, pressure, species_data.mass_fractions
                    return ThermodynamicState(
                        temperature=temp,
                        pressure=pressure,
                        density=self.gas.density,
                        sound_speed=self.gas.sound_speed,
                        viscosity=self.gas.viscosity,
                        conductivity=self.gas.thermal_conductivity
                    )

            # Basic state without detailed properties
            return ThermodynamicState(
                temperature=temp,
                pressure=pressure,
                density=1.0,  # Placeholder
                sound_speed=300.0  # Placeholder
            )

        except Exception as e:
            raise DataExtractionError("thermodynamic_state", None, str(e))

    def _create_field_map(self) -> Dict[str, str]:
        """Map standard field names to Pele field names."""
        return {
            'temperature': 'Temp',
            'pressure': 'pressure',
            'density': 'density',
            'velocity_x': 'x_velocity',  # FieldData.velocity_x maps to Pele's x_velocity
            'velocity_y': 'y_velocity',  # FieldData.velocity_y maps to Pele's y_velocity
            'heat_release_rate': 'heatRelease'
        }

    def _extract_species_data(self, ray: Any, sort_idx: np.ndarray) -> Optional[SpeciesData]:
        """Extract species mass fractions."""
        if not CANTERA_AVAILABLE or not self.gas:
            return None

        species_data = SpeciesData()

        for species in self.gas.species_names:
            field_name = f'Y({species})'
            if field_name in dict(ray.ds.field_list):
                mass_fractions = ray["boxlib", field_name][sort_idx].to_value()
                species_data.mass_fractions[species] = mass_fractions

        return species_data if species_data.mass_fractions else None

    def _extract_velocity_data(self, ray: Any, sort_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract x and y velocity components from Pele data."""
        # Extract x velocity
        try:
            x_velocity_raw = ray["boxlib", "x_velocity"][sort_idx].to_value()
            x_velocity = self.unit_converter.convert_value(x_velocity_raw, 'cm/s', 'm/s')
        except Exception as e:
            raise DataExtractionError("velocity_extraction", None, f"Failed to extract x_velocity: {str(e)}")
        
        # Extract y velocity
        try:
            y_velocity_raw = ray["boxlib", "y_velocity"][sort_idx].to_value()
            y_velocity = self.unit_converter.convert_value(y_velocity_raw, 'cm/s', 'm/s')
        except Exception as e:
            raise DataExtractionError("velocity_extraction", None, f"Failed to extract y_velocity: {str(e)}")
        
        return x_velocity, y_velocity

    def _calculate_density(self, temperature: np.ndarray, pressure: np.ndarray,
                           species_data: Optional[SpeciesData]) -> np.ndarray:
        """Calculate density using ideal gas law or Cantera."""
        if self.gas and species_data and species_data.mass_fractions:
            density = np.zeros_like(temperature)
            for i in range(len(temperature)):
                try:
                    Y = {species: fractions[i] for species, fractions in species_data.mass_fractions.items()}
                    self.gas.TPY = temperature[i], pressure[i], Y
                    density[i] = self.gas.density
                except Exception as e:
                    raise DataExtractionError("thermodynamic_state", None, str(e))
            return density
        else:
            raise DataExtractionError("thermodynamic_state", None, str(e))


    def _get_density(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
                    pressure: np.ndarray, species_data: Optional[SpeciesData],
                    field_names_in_data: List[str], missing_fields: Dict[str, str]) -> np.ndarray:
        """Get density (extract from data or calculate using Cantera)."""
        field_name = PELE_VAR_MAP['Density']['Name']  # 'density'
        
        # Try direct extraction first
        if field_name in field_names_in_data:
            try:
                density_raw = ray["boxlib", field_name][sort_idx].to_value()
                return self.unit_converter.convert_value(density_raw, 'g/cm^3', 'kg/m^3')
            except Exception as e:
                print(f"  Could not extract {field_name}: {e}")

        # Calculate using Cantera if available
        if 'Density' in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print("  Calculating density using Cantera...")
            return self._calculate_with_cantera(temperature, pressure, species_data, 'density')
        else:
            # Cannot calculate density without Cantera
            raise DataExtractionError("density_calculation", None, 
                                     "Density field not found in data and Cantera not available or species data missing. Cannot use ideal gas assumptions.")

    def _get_heat_release_rate(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
                              pressure: np.ndarray, species_data: Optional[SpeciesData],
                              field_names_in_data: List[str], missing_fields: Dict[str, str]) -> Optional[np.ndarray]:
        """Get heat release rate (extract from data or calculate using Cantera)."""
        field_name = PELE_VAR_MAP['Heat Release Rate']['Name']  # 'heatRelease'
        
        # Try direct extraction first
        if field_name in field_names_in_data:
            try:
                hrr_raw = ray["boxlib", field_name][sort_idx].to_value()
                print(f"  Successfully extracted {field_name}: {len(hrr_raw)} points, range [{np.min(hrr_raw):.2e}, {np.max(hrr_raw):.2e}] units")
                return hrr_raw
            except Exception as e:
                print(f"  Could not extract {field_name}: {e}")
        
        # Calculate using Cantera if available
        if 'Heat Release Rate' in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print("  Calculating heat release rate using Cantera...")
            hrr_calc = self._calculate_with_cantera(temperature, pressure, species_data, 'heat_release_rate')
            if hrr_calc is not None:
                print(f"  Cantera calculated heat release rate: {len(hrr_calc)} points, range [{np.min(hrr_calc):.2e}, {np.max(hrr_calc):.2e}] W/m³")
            return hrr_calc
        
        print(f"  Warning: {field_name} not available and cannot calculate with Cantera")
        return None

    def _get_viscosity(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
                      pressure: np.ndarray, species_data: Optional[SpeciesData],
                      field_names_in_data: List[str], missing_fields: Dict[str, str]) -> Optional[np.ndarray]:
        """Get viscosity (extract from data or calculate using Cantera)."""
        field_name = PELE_VAR_MAP['Viscosity']['Name']  # 'viscosity'
        
        # Try direct extraction first
        if field_name in field_names_in_data:
            try:
                viscosity_raw = ray["boxlib", field_name][sort_idx].to_value()
                return self.unit_converter.convert_value(viscosity_raw, 'g/(cm*s)', 'Pa*s')
            except Exception as e:
                print(f"  Could not extract {field_name}: {e}")
        
        # Calculate using Cantera if available
        if 'Viscosity' in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print("  Calculating viscosity using Cantera...")
            return self._calculate_with_cantera(temperature, pressure, species_data, 'viscosity')
        
        return None

    def _get_conductivity(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
                         pressure: np.ndarray, species_data: Optional[SpeciesData],
                         field_names_in_data: List[str], missing_fields: Dict[str, str]) -> Optional[np.ndarray]:
        """Get thermal conductivity (extract from data or calculate using Cantera)."""
        field_name = PELE_VAR_MAP['Conductivity']['Name']  # 'conductivity'
        
        # Try direct extraction first
        if field_name in field_names_in_data:
            try:
                conductivity_raw = ray["boxlib", field_name][sort_idx].to_value()
                return self.unit_converter.convert_value(conductivity_raw, 'W/(cm*K)', 'W/(m*K)')
            except Exception as e:
                print(f"  Could not extract {field_name}: {e}")
        
        # Calculate using Cantera if available
        if 'Conductivity' in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print("  Calculating thermal conductivity using Cantera...")
            return self._calculate_with_cantera(temperature, pressure, species_data, 'conductivity')
        
        return None

    def _get_sound_speed(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
                        pressure: np.ndarray, species_data: Optional[SpeciesData],
                        field_names_in_data: List[str], missing_fields: Dict[str, str]) -> Optional[np.ndarray]:
        """Get sound speed (extract from data or calculate using Cantera)."""
        field_name = PELE_VAR_MAP['Sound speed']['Name']  # 'soundspeed'
        
        # Try direct extraction first
        if field_name in field_names_in_data:
            try:
                sound_speed_raw = ray["boxlib", field_name][sort_idx].to_value()
                return self.unit_converter.convert_value(sound_speed_raw, 'cm/s', 'm/s')
            except Exception as e:
                print(f"  Could not extract {field_name}: {e}")
        
        # Calculate using Cantera if available
        if 'Sound speed' in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print("  Calculating sound speed using Cantera...")
            return self._calculate_with_cantera(temperature, pressure, species_data, 'sound_speed')
        
        return None

    def _get_cp(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
               pressure: np.ndarray, species_data: Optional[SpeciesData],
               field_names_in_data: List[str], missing_fields: Dict[str, str]) -> Optional[np.ndarray]:
        """Get specific heat at constant pressure (extract from data or calculate using Cantera)."""
        field_name = PELE_VAR_MAP['Cp']['Name']  # 'cp'
        
        # Try direct extraction first
        if field_name in field_names_in_data:
            try:
                cp_raw = ray["boxlib", field_name][sort_idx].to_value()
                return self.unit_converter.convert_value(cp_raw, 'cm^2/(s^2*K)', 'J/(kg*K)')
            except Exception as e:
                print(f"  Could not extract {field_name}: {e}")
        
        # Calculate using Cantera if available
        if 'Cp' in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print("  Calculating Cp using Cantera...")
            return self._calculate_with_cantera(temperature, pressure, species_data, 'cp')
        
        return None

    def _get_cv(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
               pressure: np.ndarray, species_data: Optional[SpeciesData],
               field_names_in_data: List[str], missing_fields: Dict[str, str]) -> Optional[np.ndarray]:
        """Get specific heat at constant volume (extract from data or calculate using Cantera)."""
        field_name = PELE_VAR_MAP['Cv']['Name']  # 'cv'
        
        # Try direct extraction first
        if field_name in field_names_in_data:
            try:
                cv_raw = ray["boxlib", field_name][sort_idx].to_value()
                return self.unit_converter.convert_value(cv_raw, 'cm^2/(s^2*K)', 'J/(kg*K)')
            except Exception as e:
                print(f"  Could not extract {field_name}: {e}")
        
        # Calculate using Cantera if available
        if 'Cv' in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print("  Calculating Cv using Cantera...")
            return self._calculate_with_cantera(temperature, pressure, species_data, 'cv')
        
        return None

    def _get_species_diffusion_coeff(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
                                    pressure: np.ndarray, species_data: Optional[SpeciesData],
                                    field_names_in_data: List[str], missing_fields: Dict[str, str],
                                    species: str) -> Optional[np.ndarray]:
        """Get species diffusion coefficient (extract from data or calculate using Cantera)."""
        field_name = f'D({species})'
        pele_field_name = PELE_VAR_MAP.get(field_name, {}).get('Name', field_name)
        
        # Try direct extraction first
        if pele_field_name in field_names_in_data:
            try:
                return ray["boxlib", pele_field_name][sort_idx].to_value()
            except Exception as e:
                print(f"  Could not extract {pele_field_name}: {e}")
        
        # Calculate using Cantera if available
        if field_name in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print(f"  Calculating {field_name} using Cantera...")
            return self._calculate_species_property_cantera(temperature, pressure, species_data, species, 'diffusion')
        
        return None

    def _get_species_density(self, ray: Any, sort_idx: np.ndarray, temperature: np.ndarray,
                            pressure: np.ndarray, species_data: Optional[SpeciesData],
                            field_names_in_data: List[str], missing_fields: Dict[str, str],
                            species: str) -> Optional[np.ndarray]:
        """Get species density (extract from data or calculate using Cantera)."""
        field_name = f'rho_{species}'
        pele_field_name = f'rho_{species}'
        
        # Try direct extraction first
        if pele_field_name in field_names_in_data:
            try:
                rho_species_raw = ray["boxlib", pele_field_name][sort_idx].to_value()
                return self.unit_converter.convert_value(rho_species_raw, 'g/cm^3', 'kg/m^3')
            except Exception as e:
                print(f"  Could not extract {pele_field_name}: {e}")
        
        # Calculate using Cantera if available
        if field_name in missing_fields and self.gas and species_data and species_data.mass_fractions:
            print(f"  Calculating {field_name} using Cantera...")
            return self._calculate_species_property_cantera(temperature, pressure, species_data, species, 'density')
        
        return None

    def _calculate_with_cantera(self, temperature: np.ndarray, pressure: np.ndarray,
                               species_data: SpeciesData, property_name: str) -> np.ndarray:
        """Unified Cantera calculation method for all properties."""
        result = np.zeros_like(temperature)
        
        for i in range(len(temperature)):
            try:
                Y = {species: fractions[i] for species, fractions in species_data.mass_fractions.items()}
                self.gas.TPY = temperature[i], pressure[i], Y
                
                if property_name == 'density':
                    result[i] = self.gas.density
                elif property_name == 'heat_release_rate':
                    result[i] = np.sum(self.gas.net_production_rates * self.gas.standard_enthalpies_RT * ct.gas_constant * self.gas.T)
                elif property_name == 'viscosity':
                    result[i] = self.gas.viscosity
                elif property_name == 'conductivity':
                    result[i] = self.gas.thermal_conductivity
                elif property_name == 'sound_speed':
                    result[i] = self.gas.sound_speed
                elif property_name == 'cp':
                    result[i] = self.gas.cp_mass
                elif property_name == 'cv':
                    result[i] = self.gas.cv_mass
                else:
                    result[i] = 0.0
                    
            except Exception as e:
                # Raise error instead of using fallback values
                raise DataExtractionError("cantera_calculation", None, 
                                         f"Cantera calculation failed for {property_name} at point {i}: {str(e)}")
        
        return result

    def _calculate_species_property_cantera(self, temperature: np.ndarray, pressure: np.ndarray,
                                          species_data: SpeciesData, species: str, property_name: str) -> np.ndarray:
        """Calculate species-specific properties using Cantera."""
        result = np.zeros_like(temperature)
        
        for i in range(len(temperature)):
            try:
                Y = {spec: fractions[i] for spec, fractions in species_data.mass_fractions.items()}
                self.gas.TPY = temperature[i], pressure[i], Y
                
                species_idx = self.gas.species_index(species)
                
                if property_name == 'diffusion':
                    result[i] = self.gas.mix_diff_coeffs_mass[species_idx]
                elif property_name == 'density':
                    result[i] = self.gas.density * self.gas.Y[species_idx]
                else:
                    result[i] = 0.0
                    
            except Exception as e:
                result[i] = 0.0
        
        return result


class MultiLevelExtractor(DataExtractor):
    """Extract data from multiple refinement levels."""

    def __init__(self, base_extractor: DataExtractor):
        self.base_extractor = base_extractor

    def extract_ray_data(self, dataset: Any, location: float, direction: Direction = Direction.X) -> FieldData:
        """Extract data from all refinement levels."""
        # For now, just use base extractor at max level
        return self.base_extractor.extract_ray_data(dataset, location, direction)


    def extract_thermodynamic_state(self, dataset: Any, location: Point2D) -> ThermodynamicState:
        return self.base_extractor.extract_thermodynamic_state(dataset, location)


def create_data_extractor(extractor_type: str = "pele", **kwargs) -> DataExtractor:
    """Factory for data extractors."""
    if extractor_type == "pele":
        return PeleDataExtractor(**kwargs)
    elif extractor_type == "multilevel":
        base_extractor = create_data_extractor("pele", **kwargs)
        return MultiLevelExtractor(base_extractor)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")