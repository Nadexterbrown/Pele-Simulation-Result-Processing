"""
Flame analysis for the Pele processing system.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ..core.interfaces import FlameAnalyzer, WaveTracker
from ..core.domain import FlameProperties, FieldData, WaveType, ThermodynamicState
from ..core.exceptions import FlameAnalysisError, WaveNotFoundError


class PeleFlameAnalyzer(FlameAnalyzer, WaveTracker):
    """Flame analysis implementation for Pele datasets."""

    def __init__(self, mechanism_file: str = None, thermo_calculator: Optional['ThermodynamicCalculator'] = None, **kwargs):
        """Initialize flame analyzer with configuration parameters.

        Args:
            mechanism_file: Path to Cantera mechanism file for reaction rate calculations
            thermo_calculator: Optional thermodynamic calculator for sound speed calculations
            **kwargs: Enable flags for different calculations:
                - enable_all: If True, enables all calculations (default: False)
                - enable_thickness: Enable flame thickness calculation (default: False, requires flame_temperature)
                - enable_consumption_rate: Enable consumption rate/burning velocity (default: False, requires transport_species)
                - enable_contour: Enable 2D flame contour extraction (default: False, requires flame_temperature)
                - enable_surface_length: Enable surface length calculation (default: False, requires flame_temperature)
                - enable_thermodynamic_state: Enable thermodynamic state extraction (default: False)
                - enable_cantera_fallback: Enable Cantera fallback for missing data (default: False, requires mechanism_file)
                - verbose: Enable verbose output (default: True)
        """
        # Store thermodynamic calculator
        self.thermo_calculator = thermo_calculator

        # Check for enable_all flag - if True, enables everything
        enable_all = kwargs.get('enable_all', False)

        # Extract enable flags from kwargs
        # If enable_all is True, enable everything; otherwise use individual flags with False defaults
        if enable_all:
            self.enable_thermodynamic_state = True
            self.enable_contour = True
            self.enable_surface_length = True
            self.enable_thickness = True
            self.enable_consumption_rate = True
            self.enable_flame_skirt = True
        else:
            # Individual flags default to False unless explicitly set
            self.enable_thermodynamic_state = kwargs.get('enable_thermodynamic_state', False)
            self.enable_contour = kwargs.get('enable_contour', False)
            self.enable_surface_length = kwargs.get('enable_surface_length', False)
            self.enable_thickness = kwargs.get('enable_thickness', False)
            self.enable_consumption_rate = kwargs.get('enable_consumption_rate', False)
            self.enable_flame_skirt = kwargs.get('enable_flame_skirt', False)

        # Verbose is always controlled separately
        self.verbose = kwargs.get('verbose', True)

        # Validate required parameters based on enabled features
        if (self.enable_thickness or self.enable_contour or self.enable_surface_length):
            flame_temperature = kwargs.get('flame_temperature', None)
            if flame_temperature is None:
                raise ValueError(
                    "flame_temperature is required when enable_thickness, enable_contour, or enable_surface_length is True"
                )
            self.flame_temperature = flame_temperature

        if self.enable_consumption_rate:
            transport_species = kwargs.get('transport_species', None)
            if transport_species is None:
                raise ValueError(
                    "transport_species is required when enable_consumption_rate is True"
                )
            self.transport_species = transport_species

        if self.enable_flame_skirt:
            # Validate flame_temperature is available for skirt detection
            if not hasattr(self, 'flame_temperature'):
                flame_temperature = kwargs.get('flame_temperature', None)
                if flame_temperature is None:
                    raise ValueError(
                        "flame_temperature is required when enable_flame_skirt is True"
                    )
                self.flame_temperature = flame_temperature

        # Initialize chemical mechanism for Cantera fallback
        self.mechanism_file = mechanism_file

        # Store any additional kwargs for future use
        self.additional_params = {k: v for k, v in kwargs.items()
                                 if k not in ['enable_thickness', 'enable_consumption_rate',
                                            'enable_contour', 'enable_surface_length',
                                            'enable_thermodynamic_state', 'enable_cantera_fallback',
                                            'verbose']}

    def analyze_flame_properties(self, dataset: Any, data: FieldData, extraction_location: float = None,
                                 thermo_offset: float = 10e-6) -> FlameProperties:
        """Complete flame analysis matching original flame_geometry function."""
        # Find flame position from 1D data
        flame_idx, flame_pos = self.find_wave_position(data, WaveType.FLAME)

        properties = FlameProperties(position=flame_pos, index=flame_idx)

        # Extract flame gas velocity
        gas_vel_position = flame_pos + thermo_offset
        gas_vel_idx = np.argmin(np.abs(data.coordinates - gas_vel_position))
        properties.gas_velocity = data.velocity_x[gas_vel_idx]

        # Extract thermodynamic state at flame location with offset from 1D data
        if self.enable_thermodynamic_state:
            try:
                from ..core.domain import ThermodynamicState

                # Calculate offset position (10 microns ahead of flame)
                thermo_position = flame_pos + thermo_offset

                # Find index closest to the offset position
                if flame_idx is not None and flame_idx < len(data.coordinates):
                    # Find the index closest to the thermodynamic offset position
                    thermo_idx = np.argmin(np.abs(data.coordinates - thermo_position))

                    # Ensure the index is valid
                    if thermo_idx < len(data.coordinates):
                        temp = data.temperature[thermo_idx]
                        pressure = data.pressure[thermo_idx]
                        density = data.density[thermo_idx]

                        # Try to get sound speed from data, or calculate it
                        if data.sound_speed is not None and thermo_idx < len(data.sound_speed):
                            sound_speed = data.sound_speed[thermo_idx]
                        elif self.thermo_calculator is not None and data.species_data is not None:
                            # Use thermodynamic calculator to compute sound speed
                            from ..core.domain import ThermodynamicState as ThermoStateTemp
                            temp_state = ThermoStateTemp(
                                temperature=temp,
                                pressure=pressure,
                                density=density,
                                sound_speed=0  # Placeholder
                            )
                            # Build species composition at this point
                            composition = {}
                            if data.species_data and data.species_data.mass_fractions:
                                for species, mass_fracs in data.species_data.mass_fractions.items():
                                    if isinstance(mass_fracs, np.ndarray) and thermo_idx < len(mass_fracs):
                                        composition[species] = mass_fracs[thermo_idx]

                            # Calculate thermodynamic state with sound speed
                            calculated_state = self.thermo_calculator.calculate_state(temp, pressure, composition)
                            sound_speed = calculated_state.sound_speed
                        else:
                            # Fallback to ideal gas approximation
                            # c = sqrt(gamma * R * T), where R = P/(rho*T), so c = sqrt(gamma * P/rho)
                            gamma = 1.4  # Typical value for diatomic gases
                            sound_speed = np.sqrt(gamma * pressure / density)

                        properties.thermodynamic_state = ThermodynamicState(
                            temperature=temp,
                            pressure=pressure,
                            density=density,
                            sound_speed=sound_speed
                        )

                        print(
                            f"  Flame thermodynamic state (offset +{thermo_offset * 1e6:.1f}um): T={temp:.1f}K, P={pressure:.0f}Pa, rho={density:.3f}kg/m^3")
                        print(f"  Extraction location: {data.coordinates[thermo_idx]:.6f}m vs flame at {flame_pos:.6f}m")
                    else:
                        print("  Could not extract thermodynamic state: offset position out of bounds")
                else:
                    print("  Could not extract thermodynamic state: invalid flame index")

            except Exception as e:
                print(f"  Thermodynamic state extraction failed: {e}")

        # Extract 2D flame contour using known flame position for local search
        if (self.enable_contour or self.enable_surface_length or self.enable_thickness or self.enable_consumption_rate):
            try:
                contour_points = self.extract_flame_contour(dataset, flame_pos)
                if len(contour_points) > 0:
                    # Sort contour points
                    sorted_points, segments, surface_length = self.sort_contour_by_nearest_neighbors(contour_points,
                                                                                                     dataset)

                    # Store contour points if enabled
                    if self.enable_contour:
                        properties.contour_points = sorted_points
                    # Store surface length if enabled
                    if self.enable_surface_length:
                        properties.surface_length = surface_length

                    # Calculate flame thickness if contour available
                    if self.enable_thickness:
                        try:
                            # Use the y-coordinate from the 1D extraction location for thickness calculation
                            if extraction_location is not None:
                                center_location = extraction_location  # This is the y-coordinate of the 1D ray
                            else:
                                # Fallback: use flame x-position (not ideal, but maintains backward compatibility)
                                center_location = flame_pos
                                print(
                                    "Warning: No extraction location provided for thickness calculation, using flame position as fallback")

                            thickness_result, plotting_data = self.calculate_flame_thickness(dataset, sorted_points,
                                                                                             center_location)
                            properties.thickness = thickness_result

                            # Store plotting data for later visualization
                            if plotting_data is not None:
                                properties.region_grid = plotting_data.get('region_grid')
                                properties.region_temperature = plotting_data.get('region_temperature')
                                properties.normal_line = plotting_data.get('normal_line')
                                properties.interpolated_temperatures = plotting_data.get('interpolated_temperatures')
                        except Exception as e:
                            print(f"Flame thickness calculation failed: {e}")
                            properties.thickness = np.nan

                    # Calculate consumption rate
                    if self.enable_consumption_rate:
                        try:
                            consumption_rate, burning_velocity = self.calculate_consumption_rate(
                                dataset, sorted_points, self.transport_species)
                            properties.consumption_rate = consumption_rate
                            properties.burning_velocity = burning_velocity
                        except Exception as e:
                            print(f"Consumption rate calculation failed: {e}")
                            properties.consumption_rate = np.nan
                            properties.burning_velocity = np.nan

                    if self.enable_flame_skirt:
                        try:
                            # Get the segment with the longest length (most points)
                            segment_lengths = [len(seg) for seg in segments]
                            longest_segment_idx = np.argmax(segment_lengths)
                            flame_front_points = segments[longest_segment_idx]
                            properties.skirt_pos = self.calculate_flame_skirt(dataset, flame_front_points)
                        except Exception as e:
                            print(f"Flame skirt calculation failed: {e}")
                            properties.skirt_pos = np.nan
                else:
                    print("No flame contour found")
                    properties.surface_length = np.nan
                    properties.thickness = np.nan
                    properties.consumption_rate = np.nan
                    properties.burning_velocity = np.nan
                    properties.skirt_pos = np.nan

            except Exception as e:
                print(f"2D flame analysis failed: {e}")
                properties.surface_length = np.nan
                properties.thickness = np.nan
                properties.consumption_rate = np.nan
                properties.burning_velocity = np.nan

        return properties

    def find_wave_position(self, data: FieldData, wave_type: WaveType) -> Tuple[int, float]:
        """Find flame position using temperature and species criteria."""
        if wave_type != WaveType.FLAME:
            raise WaveNotFoundError(wave_type.value, "Only flame detection supported")

        # Primary criterion: temperature threshold
        flame_indices = np.where(data.temperature >= self.flame_temperature)[0]

        if len(flame_indices) == 0:
            raise WaveNotFoundError("flame", f"No points above {self.flame_temperature}K")

        # Use downstream-most point above threshold
        temp_flame_idx = flame_indices[-1]

        # Secondary validation with Y(HO2) species if available
        if data.species_data and 'HO2' in data.species_data.mass_fractions:
            species_flame_idx = np.argmax(data.species_data.mass_fractions['HO2'])

            # Check agreement between methods
            if abs(temp_flame_idx - species_flame_idx) > 10:
                print(f'Warning: Flame Location differs by more than 10 cells!\n'
                      f'Flame Temperature Location {data.coordinates[temp_flame_idx]}\n'
                      f'Flame Species Location {data.coordinates[species_flame_idx]}')

            flame_idx = species_flame_idx  # Prefer species-based detection
        else:
            flame_idx = temp_flame_idx

        return flame_idx, data.coordinates[flame_idx]

    def calculate_wave_velocity(self, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate flame velocity from position time series."""
        if len(positions) < 2:
            return np.array([])
        return np.gradient(positions, times)

    def extract_flame_contour(self, dataset: Any, flame_pos: float = None) -> np.ndarray:
        """Extract flame contour using YT parallel processing with fallbacks."""
        # Try YT covering grid extraction first (fastest when flame_pos known)
        try:
            return self._extract_contour(dataset, flame_pos)
        except Exception as e:
            print(f"YT extraction failed: {e}")
            return np.array([])

    def sort_contour_by_nearest_neighbors(self, points: np.ndarray, dataset: Any) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """Sort contour points using simple nearest neighbor approach."""
        if len(points) == 0:
            return points, [], 0.0

        # Filter points within buffer zone
        buffer = 0.0075 * (dataset.domain_right_edge[1].to_value() - dataset.domain_left_edge[
            1].to_value()) / 100  # Convert to m
        domain_min = dataset.domain_left_edge[1].to_value() / 100 + buffer
        domain_max = dataset.domain_right_edge[1].to_value() / 100 - buffer

        valid_indices = (points[:, 1] >= domain_min) & (points[:, 1] <= domain_max)
        points = points[valid_indices]

        if len(points) == 0:
            return points, [], 0.0

        # Use simple nearest neighbor sorting
        return self._sort_by_nearest_neighbors(points, dataset)

    def calculate_flame_thickness(self, dataset: Any, contour_points: Optional[np.ndarray],
                                 center_location: float) -> tuple:
        """Calculate flame thickness using original Pele-2D-Data-Processing method.
        
        Returns:
            tuple: (flame_thickness, plotting_data_dict) where plotting_data_dict contains
                   region_grid, region_temperature, normal_line, interpolated_temperatures
        """
        try:
            if contour_points is None or len(contour_points) == 0:
                raise FlameAnalysisError("thickness", "No contour points provided")

            # Step 1: Extract the flame location from the contour 
            flame_idx = np.argmin(abs(contour_points[:, 1] - center_location))
            flame_x, flame_y = contour_points[flame_idx]

            # Step 2: Extract simulation grid
            subgrid_x, subgrid_y, subgrid_temperatures = self._extract_simulation_grid(dataset, flame_x, flame_y)

            # Step 3: Find the nearest index to the flame contour
            flame_x_idx = np.argmin(np.abs(subgrid_x - flame_x))
            flame_y_idx = np.argmin(np.abs(subgrid_y - flame_y))

            flame_x_arr = subgrid_x[np.abs(subgrid_y - subgrid_y[flame_y_idx]) <= 1e-12]
            flame_y_arr = subgrid_y[np.abs(subgrid_x - subgrid_x[flame_x_idx]) <= 1e-12]

            flame_x_arr_idx = np.argmin(np.abs(flame_x_arr - flame_x))
            flame_y_arr_idx = np.argmin(np.abs(flame_y_arr - flame_y))

            # Step 4: Create subgrid
            region_grid, region_temperature, interpolator = self._create_subgrid(
                subgrid_x, subgrid_y, subgrid_temperatures, flame_x, flame_y, 
                flame_x_arr, flame_y_arr, flame_x_arr_idx, flame_y_arr_idx)

            # Step 5: Calculate contour normal and normal line
            contour_normals = self._calculate_contour_normal(contour_points)
            normal_line = self._calculate_normal_vector_line(contour_normals[flame_idx], region_grid)
            
            # Step 6: Calculate thickness using original method
            normal_distances = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(normal_line, axis=0) ** 2, axis=1))), 0, 0)
            temp_grad = np.abs(np.gradient(interpolator(normal_line)) / np.gradient(normal_distances))
            
            flame_thickness_val = (np.max(interpolator(normal_line)) - np.min(interpolator(normal_line))) / np.max(temp_grad)
            
            # Store plotting data
            plotting_data = {
                'region_grid': region_grid,
                'region_temperature': region_temperature,
                'normal_line': normal_line,
                'interpolated_temperatures': interpolator(normal_line)
            }
            
            return flame_thickness_val, plotting_data

        except Exception as e:
            print(f"Error: Unable to calculate flame thickness: {e}")
            return np.nan, None

    def calculate_consumption_rate(self, dataset: Any, contour_points: Optional[np.ndarray],
                                  transport_species: str) -> Tuple[float, float]:
        """Calculate species consumption rate and burning velocity."""
        try:
            if contour_points is None or len(contour_points) == 0:
                raise FlameAnalysisError("consumption_rate", "No contour points provided")

            # Create bounding box around contour
            min_x = np.min(contour_points[:, 0]) - 10e-4
            max_x = np.max(contour_points[:, 0]) + 10e-4
            min_y = np.min(contour_points[:, 1])
            max_y = np.max(contour_points[:, 1])

            # Convert to cm for YT
            left_edge = [min_x * 100, min_y * 100, dataset.domain_left_edge[2].to_value()]
            right_edge = [max_x * 100, max_y * 100, dataset.domain_right_edge[2].to_value()]

            level = dataset.index.max_level
            dims = ((np.array(right_edge) - np.array(left_edge)) / dataset.index.get_smallest_dx().to_value()).astype(int)
            dims[2] = 1

            # Check if required reaction rate fields exist in dataset
            required_fields = [('boxlib', f'rho_{transport_species}'),
                             ('boxlib', f'rho_omega_{transport_species}')]
            use_dataset_fields = all(field in dataset.field_list for field in required_fields)

            # Create covering grid with appropriate fields
            if use_dataset_fields:
                print(f"  Using rho_omega_{transport_species} from dataset")
                cg = dataset.covering_grid(
                    level=level,
                    left_edge=left_edge,
                    dims=dims,
                    fields=[('boxlib', 'x'), ('boxlib', 'y'),
                           ('boxlib', f'rho_{transport_species}'),
                           ('boxlib', f'rho_omega_{transport_species}')]
                )
            else:
                # Will use Cantera - load temperature, pressure, and all species
                print(f"  rho_{transport_species} or rho_omega_{transport_species} not in dataset")
                print(f"  Will calculate consumption rate using Cantera...")

                # Build fields list with all available species
                fields_to_extract = [('boxlib', 'x'), ('boxlib', 'y'),
                                    ('boxlib', 'Temp'), ('boxlib', 'pressure')]

                species_fields = [f for f in dataset.field_list if f[0] == 'boxlib' and f[1].startswith('Y(')]
                if species_fields:
                    print(f"  Found {len(species_fields)} species mass fraction fields in dataset")
                    fields_to_extract.extend(species_fields)
                else:
                    print(f"  Warning: No Y(species) fields found in dataset")

                try:
                    cg = dataset.covering_grid(
                        level=level,
                        left_edge=left_edge,
                        dims=dims,
                        fields=fields_to_extract
                    )
                except Exception as e:
                    print(f"  Error creating covering grid with species fields: {e}")
                    print(f"  Retrying with basic fields only...")
                    cg = dataset.covering_grid(
                        level=level,
                        left_edge=left_edge,
                        dims=dims,
                        fields=[('boxlib', 'x'), ('boxlib', 'y'),
                               ('boxlib', 'Temp'), ('boxlib', 'pressure')]
                    )

                # Initialize Cantera for reaction rate calculations
                try:
                    import cantera as ct
                except ImportError:
                    raise FlameAnalysisError("consumption_rate",
                        f"rho_omega_{transport_species} not in dataset and Cantera not available")

                if not hasattr(self, 'gas'):
                    mechanism = self.mechanism_file if self.mechanism_file else 'gri30.yaml'
                    self.gas = ct.Solution(mechanism)
                    print(f"  Initialized Cantera with mechanism '{mechanism}'")
                    print(f"  Loaded {len(self.gas.species_names)} species")

            # Calculate consumption rate by integrating production rates
            dx = cg.dds[0].to_value() / 100  # Convert cm to m
            dy = cg.dds[1].to_value() / 100

            total_consumption = 0.0
            num_rows = cg['boxlib', 'x'].shape[1]

            # Process each row
            if use_dataset_fields:
                # Direct path: extract rho_omega from dataset
                for i in range(num_rows):
                    row_data = cg['boxlib', f'rho_omega_{transport_species}'][:, i, 0].to_value()
                    row_data *= 1000  # Convert from g/(cm^3*s) to kg/(m^3*s)
                    total_consumption += np.sum(np.abs(row_data)) * dx * dy
            else:
                # Cantera path: calculate rho_omega for each point
                for i in range(num_rows):
                    # Get temperature and pressure for this row
                    temp_arr = cg['boxlib', 'Temp'][:, i, 0].to_value()  # K
                    pressure_arr = cg['boxlib', 'pressure'][:, i, 0].to_value() * 0.1  # dyne/cm^2 to Pa

                    # Extract species mass fractions for this row
                    species_dict = {}
                    for species in self.gas.species_names:
                        field_name = f'Y({species})'
                        try:
                            species_arr = cg['boxlib', field_name][:, i, 0].to_value()
                            species_dict[species] = species_arr
                        except:
                            species_dict[species] = np.zeros_like(temp_arr)

                    # Calculate production rates at each point in this row
                    row_data = np.zeros_like(temp_arr)
                    for j in range(len(temp_arr)):
                        # Build mass fraction dictionary for this point
                        Y_dict = {species: species_dict[species][j] for species in self.gas.species_names}

                        # Set gas state
                        self.gas.TPY = temp_arr[j], pressure_arr[j], Y_dict

                        # Get species index and calculate production rate
                        try:
                            species_idx = self.gas.species_index(transport_species)
                        except:
                            raise FlameAnalysisError("consumption_rate",
                                f"Species {transport_species} not found in Cantera mechanism")

                        # rho_omega = omega_dot * MW (kg/m^3/s)
                        omega_dot = self.gas.net_production_rates[species_idx]  # kmol/m^3/s
                        molecular_weight = self.gas.molecular_weights[species_idx]  # kg/kmol
                        row_data[j] = omega_dot * molecular_weight

                    total_consumption += np.sum(np.abs(row_data)) * dx * dy

            # Calculate burning velocity using middle row
            middle_row = num_rows // 2
            print(f"  Extracting rho_{transport_species} from middle row (row {middle_row}/{num_rows})")

            if use_dataset_fields:
                # Get rho_species from dataset at middle row
                rho_reactant = cg["boxlib", f"rho_{transport_species}"].to_value()[-1, middle_row, 0] * 1000  # g/cm^3 to kg/m^3
            else:
                # Calculate rho_species using Cantera at middle row
                print(f"  Calculating using Cantera...")
                temp_inlet = cg['boxlib', 'Temp'][-1, middle_row, 0].to_value()
                pressure_inlet = cg['boxlib', 'pressure'][-1, middle_row, 0].to_value() * 0.1  # dyne/cm^2 to Pa

                # Extract species at inlet
                species_dict = {}
                for species in self.gas.species_names:
                    field_name = f'Y({species})'
                    try:
                        species_dict[species] = cg['boxlib', field_name][-1, middle_row, 0].to_value()
                    except:
                        species_dict[species] = 0.0

                self.gas.TPY = temp_inlet, pressure_inlet, species_dict

                try:
                    species_idx = self.gas.species_index(transport_species)
                except:
                    raise FlameAnalysisError("consumption_rate",
                        f"Species {transport_species} not found in Cantera mechanism")

                rho_reactant = self.gas.density_mass * self.gas.Y[species_idx]  # kg/m^3

            domain_height = (dataset.domain_right_edge[1].to_value() - dataset.domain_left_edge[1].to_value()) / 100
            burning_velocity = total_consumption / (rho_reactant * domain_height)

            return total_consumption, burning_velocity

        except Exception as e:
            raise FlameAnalysisError("consumption_rate", str(e))

    def calculate_flame_skirt(self, dataset: Any, flame_front_points: np.ndarray):
        """Calculate flame skirt position using ortho_ray at 95% domain height.

        Extracts an orthogonal ray along the x-axis at 95% of the domain height,
        then uses find_wave_position to locate the flame front along that ray.

        Args:
            dataset: The dataset containing domain information
            flame_front_points: Array of flame contour points (x, y) in meters (not used in new approach)

        Returns:
            float: X-position of flame front at 95% domain height, or NaN if not found
        """
        try:
            # Get domain height bounds in meters
            domain_bottom = dataset.domain_left_edge[1].to_value() / 100  # Convert cm to m
            domain_top = dataset.domain_right_edge[1].to_value() / 100  # Convert cm to m
            domain_height = domain_top - domain_bottom

            # Calculate 95% height position
            target_y = domain_bottom + 0.95 * domain_height

            # Extract ortho_ray along x-axis at the target y position
            # Create a ray object similar to what extract_ray_data would return
            try:
                # Get domain x bounds
                x_min = dataset.domain_left_edge[0].to_value() / 100  # Convert cm to m
                x_max = dataset.domain_right_edge[0].to_value() / 100  # Convert cm to m

                # Create an ortho_ray along x at the target y height
                # Using YT's ortho_ray functionality
                ray = dataset.ortho_ray(0, (target_y * 100, 0))  # axis=0 for x, coords in cm

                # Extract the fields we need for flame detection
                x_coords = ray['boxlib','x'].to_value() / 100  # Convert to meters
                temperature = ray['boxlib','Temp'].to_value()  # Temperature in K

                # Sort by x-coordinate to ensure proper ordering
                sort_indices = np.argsort(x_coords)
                x_coords = x_coords[sort_indices]
                temperature = temperature[sort_indices]

                # Create a minimal FieldData object for find_wave_position
                # We only need coordinates and temperature, plus optional species data
                class MinimalFieldData:
                    """Minimal FieldData for flame detection"""
                    def __init__(self, coordinates, temperature):
                        self.coordinates = coordinates
                        self.temperature = temperature
                        self.species_data = None

                ray_data = MinimalFieldData(x_coords, temperature)

                # Try to get species data if available for better flame detection
                try:
                    if 'Y(HO2)' in ray.field_list:
                        ho2_data = ray['boxlib','Y(HO2)'].to_value()[sort_indices]
                        # Create species data structure
                        from ..core.domain import SpeciesData
                        ray_data.species_data = SpeciesData(species_names=['HO2'])
                        ray_data.species_data.mass_fractions = {'HO2': ho2_data}
                except:
                    # Species data not available, will use temperature only
                    ray_data.species_data = None

                # Use find_wave_position to locate the flame along this ray
                flame_idx, flame_x = self.find_wave_position(ray_data, WaveType.FLAME)

                # Calculate actual relative height for reporting
                relative_height = (target_y - domain_bottom) / domain_height

                print(f"  Flame skirt: x={flame_x:.6f}m at {relative_height:.1%} domain height (Bottom Bnd: {domain_bottom}m, Top Bnd {domain_top}m)")
                print(f"    Using ortho_ray method with {len(x_coords)} points along x-axis")

                return flame_x

            except Exception as e:
                print(f"  Error extracting ortho_ray at 95% height: {e}")
                # Fallback to contour-based method if ortho_ray fails
                print(f"  Falling back to contour-based method")

                if flame_front_points is not None and len(flame_front_points) > 0:
                    # Find points near 95% height and take the rightmost
                    y_distances = np.abs(flame_front_points[:, 1] - target_y)
                    y_tolerance = 0.01 * domain_height  # 1% tolerance

                    near_target = y_distances <= y_tolerance
                    if np.any(near_target):
                        candidate_points = flame_front_points[near_target]
                        # Take rightmost point to ensure flame front
                        max_x_idx = np.argmax(candidate_points[:, 0])
                        return candidate_points[max_x_idx, 0]
                    else:
                        # Take closest point
                        closest_idx = np.argmin(y_distances)
                        return flame_front_points[closest_idx, 0]

                return np.nan

        except Exception as e:
            print(f"  Error calculating flame skirt: {e}")
            return np.nan

    def _extract_contour(self, dataset: Any, flame_pos: float = None) -> np.ndarray:
        """Extract flame contour using YT's covering grid + extract_isocontours."""
        import yt
        
        # Method 1: Smart box region extraction (focused around flame when known)
        try:
            domain_left = dataset.domain_left_edge.to_value()
            domain_right = dataset.domain_right_edge.to_value()
            
            if flame_pos is not None:
                # Create focused box region around flame (+/-5mm window)
                flame_window_left = 10e-3  # 10mm in meters
                flame_window_right = 0.5e-3  # 0.5mm in meters
                flame_pos_cm = flame_pos * 100  # Convert to cm for YT
                
                # Create focused left/right edges (+/-5mm around flame)
                # Use the full Y domain but ensure X bounds are properly clamped
                focused_left = [
                    max(domain_left[0], flame_pos_cm - flame_window_left * 100),  # Don't go outside domain
                    domain_left[1],
                    domain_left[2]
                ]
                focused_right = [
                    min(domain_right[0], flame_pos_cm + flame_window_right * 100),
                    domain_right[1], 
                    domain_right[2]
                ]
                
                print(f"Creating focused box region around flame at {flame_pos:.4f}m")
                print(f"Box bounds: x=[{focused_left[0]:.2f}, {focused_right[0]:.2f}]cm, window: -{flame_window_left*1000:.1f}/+{flame_window_right*1000:.1f}mm")
            else:
                # Fallback to broader regional extraction (2.5% buffer from each end)
                flame_buffer = 0.025
                focused_left = [domain_left[0] + flame_buffer * (domain_right[0] - domain_left[0]), domain_left[1], domain_left[2]]
                focused_right = [domain_right[0] - flame_buffer * (domain_right[0] - domain_left[0]), domain_right[1], domain_right[2]]
                
                print(f"Creating regional box region (no flame position known)")
                print(f"Box bounds: x=[{focused_left[0]:.1f}, {focused_right[0]:.1f}]cm, 2.5% buffer")
            
            # Create box region and extract contours with validation
            print(f"Domain bounds: x=[{domain_left[0]:.2f}, {domain_right[0]:.2f}]cm, y=[{domain_left[1]:.4f}, {domain_right[1]:.4f}]cm")
            print(f"Box region: x=[{focused_left[0]:.2f}, {focused_right[0]:.2f}]cm, y=[{focused_left[1]:.4f}, {focused_right[1]:.4f}]cm")
            
            # Validate bounds are within domain
            if (focused_left[0] < domain_left[0] or focused_right[0] > domain_right[0] or
                focused_left[1] < domain_left[1] or focused_right[1] > domain_right[1]):
                print("Warning: Box bounds exceed domain, falling back to full domain extraction")
                raise ValueError("Box bounds exceed domain")

            box_region = dataset.box(left_edge=focused_left, right_edge=focused_right)
            contours = box_region.extract_isocontours("Temp", self.flame_temperature)
            
            if len(contours) > 0:
                contour_2d = contours[:, :2] / 100  # cm to m, drop z
                valid_mask = np.isfinite(contour_2d).all(axis=1)
                clean_contours = contour_2d[valid_mask]
                
                if len(clean_contours) > 0:
                    method_type = "focused" if flame_pos is not None else "regional"
                    print(f"Extracted {len(clean_contours)} contour points using {method_type} box region")
                    return self._remove_duplicate_points(clean_contours)
                        
        except Exception as e:
            print(f"Box region extraction failed: {e}")
        
        # Method 2: Full domain extraction
        try:
            if hasattr(dataset, 'all_data'):
                print("Trying full domain isocontour extraction...")
                contours = dataset.all_data().extract_isocontours("Temp", self.flame_temperature)
                if len(contours) > 0:
                    contour_2d = contours[:, :2] / 100  # Convert cm to m, drop z
                    valid_mask = np.isfinite(contour_2d).all(axis=1)
                    clean_contours = contour_2d[valid_mask]
                    if len(clean_contours) > 0:
                        print(f"Extracted {len(clean_contours)} contour points using full domain")
                        return self._remove_duplicate_points(clean_contours)
        except Exception as e:
            print(f"Full domain isocontour extraction failed: {e}")
        
        
        raise ValueError("All YT contour extraction methods failed")

    def _sort_by_nearest_neighbors(self, points: np.ndarray, dataset: Any) -> Tuple[np.ndarray, List[np.ndarray], float]:
        """Sort contour points using efficient nearest neighbor approach with segment breaking."""
        if len(points) < 2:
            return points, [points] if len(points) > 0 else [], 0.0
        
        from scipy.spatial import cKDTree
        
        # Calculate segment break distance threshold
        max_segment_distance = 20 * dataset.index.get_smallest_dx().to_value() / 100  # Convert to m
        
        # Build spatial index for efficient nearest neighbor queries
        tree = cKDTree(points)
        
        # Start from the leftmost point
        start_idx = np.argmin(points[:, 0])
        
        # Track visited points efficiently using boolean array
        visited = np.zeros(len(points), dtype=bool)
        ordered_indices = []
        segments = []
        segment_start = 0
        
        current_idx = start_idx
        visited[current_idx] = True
        ordered_indices.append(current_idx)
        
        while len(ordered_indices) < len(points):
            current_point = points[current_idx]
            
            # Find k nearest neighbors (k=min(20, remaining_points))
            remaining_count = len(points) - len(ordered_indices)
            k = min(20, remaining_count + 1)  # +1 because query includes current point
            
            distances, neighbor_indices = tree.query(current_point, k=k)
            
            # Find the nearest unvisited neighbor
            nearest_idx = None
            min_distance = float('inf')
            
            for dist, idx in zip(distances, neighbor_indices):
                if not visited[idx] and dist < min_distance:
                    nearest_idx = idx
                    min_distance = dist
            
            # Fallback: if no neighbor found in k-nearest, search all unvisited
            if nearest_idx is None:
                unvisited_mask = ~visited
                if np.any(unvisited_mask):
                    unvisited_indices = np.where(unvisited_mask)[0]
                    unvisited_points = points[unvisited_indices]
                    distances_to_unvisited = np.linalg.norm(unvisited_points - current_point, axis=1)
                    min_idx = np.argmin(distances_to_unvisited)
                    nearest_idx = unvisited_indices[min_idx]
                    min_distance = distances_to_unvisited[min_idx]
            
            if nearest_idx is None:
                break  # All points visited
            
            # Check if we need to create a new segment
            if min_distance > max_segment_distance and len(ordered_indices) > segment_start + 1:
                # Create segment from segment_start to current position
                segment_indices = ordered_indices[segment_start:]
                segment_points = points[segment_indices]
                segments.append(segment_points)
                segment_start = len(ordered_indices)
            
            # Add point to path
            visited[nearest_idx] = True
            ordered_indices.append(nearest_idx)
            current_idx = nearest_idx
        
        # Add final segment
        if segment_start < len(ordered_indices):
            segment_indices = ordered_indices[segment_start:]
            segment_points = points[segment_indices]
            segments.append(segment_points)
        
        # If no segments were created, treat all points as one segment
        if not segments:
            segments = [points[ordered_indices]]
        
        # Calculate total path length efficiently
        ordered_points = points[ordered_indices]
        total_length = 0.0
        
        for segment in segments:
            if len(segment) > 1:
                # Vectorized distance calculation
                segment_vectors = np.diff(segment, axis=0)
                segment_distances = np.linalg.norm(segment_vectors, axis=1)
                total_length += np.sum(segment_distances)
        
        return ordered_points, segments, total_length

    def _remove_duplicate_points(self, points: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
        """Remove duplicate points within tolerance."""
        if len(points) <= 1:
            return points
            
        from scipy.spatial.distance import pdist, squareform
        
        try:
            distances = pdist(points)
            distance_matrix = squareform(distances)
            
            close_pairs = np.where((distance_matrix < tolerance) & (distance_matrix > 0))
            
            indices_to_remove = set()
            for i, j in zip(close_pairs[0], close_pairs[1]):
                if i < j:  # Only process each pair once
                    indices_to_remove.add(j)
            
            keep_indices = [i for i in range(len(points)) if i not in indices_to_remove]
            return points[keep_indices]
            
        except:
            # Fallback to simple approach
            unique_points = [points[0]]
            
            for point in points[1:]:
                distances = np.linalg.norm(np.array(unique_points) - point, axis=1)
                if np.min(distances) > tolerance:
                    unique_points.append(point)
            
            return np.array(unique_points)

    def _extract_simulation_grid(self, dataset: Any, flame_x: float, flame_y: float):
        """Extract simulation grid using local covering grid around flame location."""
        # Get max level and smallest grid spacing
        max_level = dataset.index.max_level
        dx = dataset.index.get_smallest_dx().to_value() / 100  # Convert cm to m
        dy = dataset.index.get_smallest_dx().to_value() / 100  # Convert cm to m
        
        # Create local box region around flame (25 grid points on each side)
        grid_extent = 25
        box_width_x = grid_extent * dx
        box_width_y = grid_extent * dy
        
        # Define box boundaries (flame_x, flame_y are already in meters)
        left_edge = [
            (flame_x - box_width_x) * 100,  # Convert back to cm for YT
            (flame_y - box_width_y) * 100,
            dataset.domain_left_edge[2].to_value()
        ]
        right_edge = [
            (flame_x + box_width_x) * 100,  # Convert back to cm for YT
            (flame_y + box_width_y) * 100,
            dataset.domain_right_edge[2].to_value()
        ]
        
        # Ensure bounds don't exceed domain
        domain_left = dataset.domain_left_edge.to_value()
        domain_right = dataset.domain_right_edge.to_value()
        
        left_edge[0] = max(left_edge[0], domain_left[0])
        left_edge[1] = max(left_edge[1], domain_left[1])
        right_edge[0] = min(right_edge[0], domain_right[0])
        right_edge[1] = min(right_edge[1], domain_right[1])
        
        # Calculate dimensions for covering grid (ensure at least 50x50 grid)
        dims_x = max(50, int((right_edge[0] - left_edge[0]) / dataset.index.get_smallest_dx().to_value()))
        dims_y = max(50, int((right_edge[1] - left_edge[1]) / dataset.index.get_smallest_dx().to_value()))
        dims = [dims_x, dims_y, 1]
        
        print(f"  Creating local covering grid: {dims_x}x{dims_y} around flame at ({flame_x:.6f}, {flame_y:.6f})m")
        
        try:
            # Create covering grid
            cg = dataset.covering_grid(
                level=max_level,
                left_edge=left_edge,
                dims=dims,
                fields=[('boxlib', 'x'), ('boxlib', 'y'), 'Temp']
            )
            
            # Extract grid data
            subgrid_x = cg['boxlib', 'x'].to_value().flatten() / 100  # Convert cm to m
            subgrid_y = cg['boxlib', 'y'].to_value().flatten() / 100  # Convert cm to m
            subgrid_temperatures = cg['Temp'].to_value().flatten()
            
            print(f"  Extracted {len(subgrid_x)} grid points from local covering grid")
            
            return subgrid_x, subgrid_y, subgrid_temperatures
            
        except Exception as e:
            print(f"  Warning: Covering grid extraction failed: {e}")
            print("  Falling back to original grid-based method...")
            
            # Fallback to original method
            max_level = dataset.index.max_level
            grids = [grid for grid in dataset.index.grids if grid.Level == max_level]

            subgrid_x, subgrid_y, subgrid_temperatures = [], [], []

            for temp_grid in grids:
                x = temp_grid["boxlib", "x"].to_value().flatten()
                y = temp_grid["boxlib", "y"].to_value().flatten()
                temp = temp_grid["Temp"].flatten()
                
                # Calculate the mean x value for the current grid
                current_mean_x = np.mean(x)
                # If the difference in mean x values is too large, skip the current grid
                if current_mean_x > flame_x * 100 + 1e-2:  # Convert flame_x to cm for comparison
                    continue

                # Collect the values from this grid
                subgrid_x.extend(x)
                subgrid_y.extend(y)
                subgrid_temperatures.extend(temp)

            # Convert to m for consistency
            subgrid_x = np.array(subgrid_x) / 100  # Convert cm to m
            subgrid_y = np.array(subgrid_y) / 100  # Convert cm to m
            subgrid_temperatures = np.array(subgrid_temperatures)

            return subgrid_x, subgrid_y, subgrid_temperatures

    def _create_subgrid(self, subgrid_x, subgrid_y, subgrid_temperatures, flame_x, flame_y, 
                       flame_x_arr, flame_y_arr, flame_x_arr_idx, flame_y_arr_idx):
        """Create subgrid using LinearNDInterpolator with covering grid data."""
        from scipy.interpolate import LinearNDInterpolator
        
        print(f"  Creating interpolator with {len(subgrid_x)} covering grid points around flame")
        
        # Step 1: Use the covering grid data directly with LinearNDInterpolator
        # Filter out any invalid temperature values
        valid_mask = np.isfinite(subgrid_temperatures)
        
        if np.sum(valid_mask) < 3:
            print(f"  Error: Insufficient valid temperature data ({np.sum(valid_mask)} points)")
            return None, None, None
        
        valid_points = np.column_stack((subgrid_x[valid_mask], subgrid_y[valid_mask]))
        valid_temps = subgrid_temperatures[valid_mask]
        
        # Step 2: Create LinearNDInterpolator directly with the covering grid data
        interpolator = LinearNDInterpolator(valid_points, valid_temps, fill_value=np.nan)
        
        # Step 3: Create region grid for analysis (local window around flame)
        # Determine analysis window size based on grid resolution
        dx = np.min(np.diff(np.unique(subgrid_x)))
        dy = np.min(np.diff(np.unique(subgrid_y))) 
        
        # Create analysis window (typically 21x21 points around flame)
        window_size = 10  # +/- 10 points around flame
        
        x_min = flame_x - window_size * dx
        x_max = flame_x + window_size * dx
        y_min = flame_y - window_size * dy
        y_max = flame_y + window_size * dy
        
        # Ensure bounds stay within available data
        x_min = max(x_min, np.min(subgrid_x))
        x_max = min(x_max, np.max(subgrid_x))
        y_min = max(y_min, np.min(subgrid_y))
        y_max = min(y_max, np.max(subgrid_y))
        
        # Create regular analysis grid
        analysis_x = np.linspace(x_min, x_max, 21)
        analysis_y = np.linspace(y_min, y_max, 21)
        
        region_grid = np.dstack(np.meshgrid(analysis_x, analysis_y)).reshape(-1, 2)
        
        # Step 4: Evaluate interpolator on analysis grid to create region_temperature
        region_temps_flat = interpolator(region_grid)
        region_temperature = region_temps_flat.reshape(21, 21)
        
        # Test interpolator at flame location
        test_result = interpolator(np.array([[flame_x, flame_y]]))
        
        if not np.isnan(test_result[0]):
            print(f"  LinearNDInterpolator created successfully with {np.sum(valid_mask)} valid points")
            print(f"  Temperature at flame location: {test_result[0]:.1f}K")
        else:
            print(f"  Warning: Interpolator returns NaN at flame location")
        
        return region_grid, region_temperature, interpolator

    def _calculate_contour_normal(self, contour_points: np.ndarray) -> np.ndarray:
        """Calculate contour normals using original method."""
        # Step 1: Compute the gradient of the contour points
        dx = np.gradient(contour_points[:, 0])
        dy = np.gradient(contour_points[:, 1])

        # Step 2: Compute the normals
        normals = np.zeros_like(contour_points)
        # Case 1: If the contour is aligned with the x-axis, the normal should be along the y-axis
        for i in range(len(dx)):
            if (dx[i] == 0):  # No change in x-coordinates, thus the normal is along y-axis
                normals[i, 0] = 0  # Normal along the y-axis (positive direction)
                normals[i, 1] = 1  # No change in y for normal direction

            # Case 2: If the contour is aligned with the y-axis, the normal should be along the x-axis
            elif (dy[i] == 0):  # No change in y-coordinates, normal should be along x-axis
                normals[i, 0] = 1  # No change in x for normal direction
                normals[i, 1] = 0  # Normal along the x-axis (positive direction)

            # General case: Calculate the normal by rotating the tangent 90 degrees
            else:
                normals[i, 0] = dy[i]  # Rotate by 90 degrees
                normals[i, 1] = -dx[i]  # Invert the x-component of the tangent

        # Step 3: Normalize the normal vectors
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        return normals

    def _calculate_normal_vector_line(self, normal_vector: np.ndarray, region_grid: np.ndarray):
        """Calculate normal vector line using original method."""
        # Step 1: Determine the spacing to be used for the normal vector
        dx = np.abs(np.unique(region_grid[:, 0])[1] - np.unique(region_grid[:, 0])[0])
        dy = np.abs(np.unique(region_grid[:, 1])[1] - np.unique(region_grid[:, 1])[0])
        t_step = min(dx, dy) / np.linalg.norm(normal_vector)  # Adjust step size for resolution

        # Center point of the array
        center_point = region_grid[region_grid.shape[0] // 2]

        # Step 2: Determine the bounds for the normal vector
        t_min_x = (np.min(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[0] != 0 else -np.inf
        t_max_x = (np.max(region_grid[:, 0]) - center_point[0]) / normal_vector[0] if normal_vector[0] != 0 else np.inf
        t_min_y = (np.min(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[1] != 0 else -np.inf
        t_max_y = (np.max(region_grid[:, 1]) - center_point[1]) / normal_vector[1] if normal_vector[1] != 0 else np.inf

        t_start = max(min(t_min_x, t_max_x), min(t_min_y, t_max_y))
        t_end = min(max(t_min_x, t_max_x), max(t_min_y, t_max_y))

        # Step 3: Generate t_range
        t_range = np.arange(t_start, t_end, t_step / 1e2, dtype=np.float32)

        # Step 4: Generate line points along the normal vector
        x_line_points = np.array(center_point[0] + t_range * normal_vector[0], dtype=np.float32)
        y_line_points = np.array(center_point[1] + t_range * normal_vector[1], dtype=np.float32)
        line_points = np.column_stack((x_line_points, y_line_points))

        # Step 5: Filter line points to ensure they remain within bounds
        min_x, max_x = np.min(region_grid[:, 0]), np.max(region_grid[:, 0])
        min_y, max_y = np.min(region_grid[:, 1]), np.max(region_grid[:, 1])
        line_points_filtered = line_points[
            (line_points[:, 0] >= min_x) & (line_points[:, 0] <= max_x) &
            (line_points[:, 1] >= min_y) & (line_points[:, 1] <= max_y)
            ]

        return line_points_filtered


def create_flame_analyzer(mechanism_file: str = None, thermo_calculator: Optional['ThermodynamicCalculator'] = None, **kwargs) -> FlameAnalyzer:
    """Factory for flame analyzers with configuration options.

    Args:
        flame_temperature: Temperature threshold for flame detection (K) - required for thickness/contour
        transport_species: Species to track for consumption rate calculations - required for consumption_rate
        mechanism_file: Path to Cantera mechanism file
        thermo_calculator: Optional thermodynamic calculator for sound speed calculations
        **kwargs: Additional configuration flags passed to PeleFlameAnalyzer

    Returns:
        Configured PeleFlameAnalyzer instance
    """
    return PeleFlameAnalyzer(mechanism_file, thermo_calculator, **kwargs)