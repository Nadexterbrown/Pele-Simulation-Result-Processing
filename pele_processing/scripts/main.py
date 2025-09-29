#!/usr/bin/env python3
"""
Main Pele Processing Script
A comprehensive script to process Pele simulation data using the pele_processing package.
Calculates flame properties, shock properties, thermodynamic states, and generates animations.

Based on the original 2D-Pele-Processing.py but using the new modular architecture.
Fixes thermodynamic extraction issues found in comprehensive_processing_script.py.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import yt

# Try to import from installed package first, fall back to local development path if needed
try:
    # This will work when the package is pip installed
    from pele_processing.base_processing import (
        # Core functionality
        create_data_loader, create_data_extractor, create_flame_analyzer, create_shock_analyzer,
        create_burned_gas_analyzer, create_unit_converter, setup_logging, load_dataset_paths,

        # Data structures
        FieldData, FlameProperties, ShockProperties, BurnedGasProperties, ThermodynamicState, Point2D,
        WaveType, Direction, ProcessingResult, ProcessingBatch, DatasetInfo, PressureWaveProperties,
        AnimationFrame,

        # Visualization
        StandardPlotter, LocalViewPlotter, ComparisonPlotter,
        create_formatter,

        # Configuration
        ProcessingMode, LogLevel,

        # Parallel processing
        MPICoordinator, SequentialCoordinator, create_processing_strategy,

        # Pressure wave analysis
        create_pressure_wave_analyzer, PelePressureWaveAnalyzer, DetectionMethod
    )

    # Animation imports
    from pele_processing.base_processing.visualization.animators import (
        FrameAnimator, BatchAnimator
    )
except ImportError:
    # Fall back to development path if not installed
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir.parent / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from pele_processing.base_processing import (
    # Core functionality
    create_data_loader, create_data_extractor, create_flame_analyzer, create_shock_analyzer,
    create_burned_gas_analyzer, create_unit_converter, setup_logging, load_dataset_paths,

    # Data structures
    FieldData, FlameProperties, ShockProperties, BurnedGasProperties, ThermodynamicState, Point2D,
    WaveType, Direction, ProcessingResult, ProcessingBatch, DatasetInfo, PressureWaveProperties,
    AnimationFrame,

    # Visualization
    StandardPlotter, LocalViewPlotter, ComparisonPlotter,
    create_formatter,

    # Configuration
    ProcessingMode, LogLevel,

    # Parallel processing
    MPICoordinator, SequentialCoordinator, create_processing_strategy,

    # Pressure wave analysis
    create_pressure_wave_analyzer, PelePressureWaveAnalyzer, DetectionMethod
)

# Animation imports
from pele_processing.base_processing.visualization.animators import (
    FrameAnimator, BatchAnimator
)

# MPI detection and setup
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    
    # Initialize MPI communicator
    comm = MPI.COMM_WORLD
    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()
    # Only consider it an MPI run if we have multiple processes
    IS_MPI_RUN = MPI_SIZE > 1
    
except ImportError:
    # Fallback for non-MPI environments
    MPI_AVAILABLE = False
    IS_MPI_RUN = False
    MPI_RANK = 0
    MPI_SIZE = 1
    comm = None

# Set YT log level to reduce noise
yt.set_log_level(0)

# Global configuration
VERSION = "3.0.0"
DEFAULT_FLAME_TEMP = 2500.0
DEFAULT_SHOCK_PRESSURE_RATIO = 1.01

########################################################################################################################
# Configuration and Setup
########################################################################################################################

class FlameConfig:
    """Configuration for flame analysis"""
    def __init__(self):
        self.enable_all = True  # Comprehensive analysis
        # Individual flags (overridden by enable_all if True)
        self.enable_thermodynamics = False
        self.enable_surface_length = False
        self.enable_thickness = False
        self.enable_consumption = False

class ProcessingConfig:
    """Configuration for processing parameters"""
    def __init__(self):
        # Data extraction parameters
        self.extraction_location = (0.0462731 + (8.7e-5 / 2)) / 100  # meters - y-location for 1D ray extraction along X
        self.flame_temperature = DEFAULT_FLAME_TEMP
        self.shock_pressure_ratio = DEFAULT_SHOCK_PRESSURE_RATIO
        self.transport_species = 'H2'
        self.chemical_mechanism = '../chemical_mechanisms/LiDryer.yaml'

        # Processing flags
        self.analyze_flame = True
        self.flame_config = FlameConfig()

        self.analyze_shock = True
        self.analyze_burned_gas = True
        self.analyze_pressure_wave = True  # Enable pressure wave analysis
        self.calculate_velocities = True

        # Pressure wave detection settings
        self.pressure_wave_detection_method = "max_value"  # 'max_value', 'max_gradient', or 'threshold'
        self.pressure_wave_threshold = None  # Only needed for threshold method
        self.pressure_wave_field = "pressure"  # Field to use for detection

        # Output options
        self.create_animations = True
        self.create_plots = True
        self.save_individual_results = False  # Use single combined file only
        self.save_combined_results = True

        # Animation parameters
        self.local_window_size = 1e-3  # 1mm window for local plots
        self.generate_mp4 = True  # Generate MP4 animations from frames
        self.animation_fps = 30.0  # Frames per second for animations
        self.animation_formats = ['mp4']  # Can also include 'gif', 'avi'

        # Thermodynamic calculation offsets (meters)
        self.flame_thermo_offset = 10e-6    # 10 microns ahead of flame
        self.burned_gas_offset = -10e-6     # 10 microns behind flame
        self.pre_shock_offset = 10e-6       # 10 microns ahead of shock
        self.post_shock_offset = -10e-6     # 10 microns behind shock

def setup_directories(output_base: Path) -> Dict[str, Path]:
    """Create organized output directory structure"""
    directories = {
        'base': output_base,
        'animations': output_base / 'Animation-Frames',
    }
    
    # Animation subdirectories
    anim_dirs = {
        'temperature': 'Temperature-Plt-Files',
        'temperature_local': 'Local-Temperature-Plt-Files',
        'pressure': 'Pressure-Plt-Files',
        'pressure_local': 'Local-Pressure-Plt-Files',
        'velocity': 'GasVelocity-Plt-Files',
        'velocity_local': 'Local-GasVelocity-Plt-Files',
        'heat_release': 'HeatReleaseRate-Plt-Files',
        'heat_release_local': 'Local-HeatReleaseRate-Plt-Files',
        'flame_geometry': 'Surface-Contour-Plt-Files',
        'flame_thickness': 'Flame-Thickness-Plt-Files',
        'collective': 'Pressure-Temperature-Plt-Files'
    }
    
    for key, subdir in anim_dirs.items():
        directories[f'anim_{key}'] = directories['animations'] / subdir
    
    # Create all directories
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
        
    return directories

def ensure_long_path_prefix(path: str) -> str:
    """Ensure Windows long path prefix for paths over 260 characters"""
    if os.name == 'nt' and len(path) > 200 and not path.startswith('\\\\?\\'):
        abs_path = os.path.abspath(path)
        if abs_path and abs_path != '\\\\':
            return f'\\\\?\\{abs_path}'
    return path

########################################################################################################################
# Core Processing Functions
########################################################################################################################

def extract_field_data(dataset, config: ProcessingConfig) -> FieldData:
    """Extract 1D field data using the data extractor"""
    print(f"  Extracting 1D field data along X at y = {config.extraction_location:.6f} m...")
    
    # Create unit converter and data extractor with Cantera mechanism
    unit_converter = create_unit_converter("pele")
    extractor = create_data_extractor("pele", unit_converter=unit_converter, mechanism_file=config.chemical_mechanism)
    
    # Extract 1D ray data along X direction at specific Y location
    field_data = extractor.extract_ray_data(dataset, config.extraction_location, Direction.X)
    
    print(f"    Data points extracted: {len(field_data.coordinates)}")
    print(f"    Temperature range: [{np.min(field_data.temperature):.1f}, {np.max(field_data.temperature):.1f}] K")
    print(f"    Pressure range: [{np.min(field_data.pressure):.0f}, {np.max(field_data.pressure):.0f}] Pa")
    
    return field_data, extractor

def analyze_flame_properties(dataset, field_data: FieldData, config: ProcessingConfig) -> Dict[str, Any]:
    """Comprehensive flame analysis"""
    print(f"  Analyzing flame properties...")
    results = {}
    
    try:
        # Create flame analyzer with mechanism file and configuration
        flame_analyzer = create_flame_analyzer(
            enable_all=config.flame_config.enable_all,
            mechanism_file=config.chemical_mechanism,
            flame_temperature=config.flame_temperature,
            transport_species=config.transport_species
        )

        # Perform comprehensive flame analysis with thermodynamic offset and contour fitting
        flame_properties = flame_analyzer.analyze_flame_properties(
            dataset, field_data, config.extraction_location, config.flame_thermo_offset
        )
        results['flame_properties'] = flame_properties
        
        # Extract individual properties with error handling
        flame_position = getattr(flame_properties, 'position', 0.0)
        flame_index = getattr(flame_properties, 'index', 0)
        
        results['flame_position'] = flame_position
        results['flame_index'] = flame_index
        
        print(f"    Flame position: {flame_position:.6f} m (index: {flame_index})")
        
        # Extract thermodynamic state at flame
        thermo_state = getattr(flame_properties, 'thermodynamic_state', None)
        if thermo_state:
            results['flame_temperature'] = getattr(thermo_state, 'temperature', 0.0)
            results['flame_pressure'] = getattr(thermo_state, 'pressure', 0.0)
            results['flame_density'] = getattr(thermo_state, 'density', 0.0)
            results['flame_sound_speed'] = getattr(thermo_state, 'sound_speed', 0.0)
            
            print(f"    Flame thermodynamics: T={results['flame_temperature']:.1f}K, "
                  f"P={results['flame_pressure']:.0f}Pa, rho={results['flame_density']:.3f}kg/m^3")
        else:
            print(f"    Warning: No thermodynamic state available from flame analysis")
            results.update({
                'flame_temperature': 0.0,
                'flame_pressure': 0.0, 
                'flame_density': 0.0,
                'flame_sound_speed': 0.0
            })
        
        # Extract geometry properties with nan handling
        flame_thickness = getattr(flame_properties, 'thickness', 0.0)
        results['flame_thickness'] = 0.0 if (flame_thickness is None or np.isnan(flame_thickness)) else flame_thickness
        
        surface_length = getattr(flame_properties, 'surface_length', 0.0)
        results['surface_length'] = 0.0 if (surface_length is None or np.isnan(surface_length)) else surface_length
        
        consumption_rate = getattr(flame_properties, 'consumption_rate', 0.0)
        results['consumption_rate'] = 0.0 if (consumption_rate is None or np.isnan(consumption_rate)) else consumption_rate
        
        burning_velocity = getattr(flame_properties, 'burning_velocity', 0.0)
        results['burning_velocity'] = 0.0 if (burning_velocity is None or np.isnan(burning_velocity)) else burning_velocity

        # Extract flame skirt position (flame at 95% domain height)
        skirt_position = getattr(flame_properties, 'skirt_pos', 0.0)
        results['flame_skirt_position'] = 0.0 if (skirt_position is None or np.isnan(skirt_position)) else skirt_position

        # Extract heat release rate at flame location
        if field_data.heat_release_rate is not None and flame_index < len(field_data.heat_release_rate):
            results['flame_hrr'] = field_data.heat_release_rate[flame_index]
        else:
            results['flame_hrr'] = 0.0
        
        if results['flame_thickness'] > 0:
            print(f"    Geometry: thickness={results['flame_thickness']*1e6:.1f}um, "
                  f"surface_length={results['surface_length']:.6f}m")
        else:
            print(f"    Geometry: thickness calculation failed, "
                  f"surface_length={results['surface_length']:.6f}m")
        print(f"    Consumption: rate={results['consumption_rate']:.3e}kg/s, "
              f"velocity={results['burning_velocity']:.1f}m/s")

    except Exception as e:
        print(f"    Error in flame analysis: {e}")
        # Fill with default values
        results.update({
            'flame_properties': None,
            'flame_position': 0.0,
            'flame_index': 0,
            'flame_temperature': 0.0,
            'flame_pressure': 0.0,
            'flame_density': 0.0,
            'flame_sound_speed': 0.0,
            'flame_thickness': 0.0,
            'surface_length': 0.0,
            'consumption_rate': 0.0,
            'burning_velocity': 0.0,
            'flame_hrr': 0.0,
            'flame_skirt_position': 0.0
        })
    
    return results

def analyze_shock_properties(dataset, field_data: FieldData, config: ProcessingConfig) -> Dict[str, Any]:
    """Comprehensive shock analysis"""
    print(f"  Analyzing shock properties...")
    results = {}
    
    try:
        # Create shock analyzer
        shock_analyzer = create_shock_analyzer(pressure_ratio_threshold=config.shock_pressure_ratio)
        
        # Perform shock analysis
        shock_properties = shock_analyzer.analyze_shock_properties(field_data)
        results['shock_properties'] = shock_properties
        
        # Extract individual properties
        shock_position = getattr(shock_properties, 'position', 0.0)
        shock_index = getattr(shock_properties, 'index', 0)
        
        results['shock_position'] = shock_position
        results['shock_index'] = shock_index
        
        print(f"    Shock position: {shock_position:.6f} m (index: {shock_index})")
        
        # Extract pre-shock and post-shock states
        pre_shock_state = getattr(shock_properties, 'pre_shock_state', None)
        post_shock_state = getattr(shock_properties, 'post_shock_state', None)
        
        if pre_shock_state:
            results.update({
                'pre_shock_temperature': getattr(pre_shock_state, 'temperature', 0.0),
                'pre_shock_pressure': getattr(pre_shock_state, 'pressure', 0.0),
                'pre_shock_density': getattr(pre_shock_state, 'density', 0.0),
                'pre_shock_sound_speed': getattr(pre_shock_state, 'sound_speed', 0.0)
            })
        else:
            results.update({
                'pre_shock_temperature': 0.0,
                'pre_shock_pressure': 0.0,
                'pre_shock_density': 0.0,
                'pre_shock_sound_speed': 0.0
            })
        
        if post_shock_state:
            results.update({
                'post_shock_temperature': getattr(post_shock_state, 'temperature', 0.0),
                'post_shock_pressure': getattr(post_shock_state, 'pressure', 0.0),
                'post_shock_density': getattr(post_shock_state, 'density', 0.0),
                'post_shock_sound_speed': getattr(post_shock_state, 'sound_speed', 0.0)
            })
        else:
            results.update({
                'post_shock_temperature': 0.0,
                'post_shock_pressure': 0.0,
                'post_shock_density': 0.0,
                'post_shock_sound_speed': 0.0
            })
        
    except Exception as e:
        print(f"    Error in shock analysis: {e}")
        # Fill with default values
        results.update({
            'shock_properties': None,
            'shock_position': 0.0,
            'shock_index': 0,
            'pre_shock_temperature': 0.0,
            'pre_shock_pressure': 0.0,
            'pre_shock_density': 0.0,
            'pre_shock_sound_speed': 0.0,
            'post_shock_temperature': 0.0,
            'post_shock_pressure': 0.0,
            'post_shock_density': 0.0,
            'post_shock_sound_speed': 0.0
        })
    
    return results

def analyze_burned_gas_properties(dataset, field_data: FieldData, burned_gas_analyzer,
                                flame_position: float, flame_index: int, config: ProcessingConfig) -> Dict[str, Any]:
    """Analyze burned gas properties behind the flame using the burned gas analyzer"""
    print(f"  Analyzing burned gas properties...")
    results = {}

    try:
        # Use the burned gas analyzer to get properties
        burned_gas_props = burned_gas_analyzer.analyze_burned_gas_properties(
            field_data, flame_position, flame_index
        )

        # Extract results
        results['burned_gas_velocity'] = burned_gas_props.gas_velocity or 0.0

        if burned_gas_props.thermodynamic_state:
            thermo_state = burned_gas_props.thermodynamic_state
            results.update({
                'burned_gas_temperature': thermo_state.temperature,
                'burned_gas_pressure': thermo_state.pressure,
                'burned_gas_density': thermo_state.density,
                'burned_gas_sound_speed': thermo_state.sound_speed
            })
        else:
            results.update({
                'burned_gas_temperature': 0.0,
                'burned_gas_pressure': 0.0,
                'burned_gas_density': 0.0,
                'burned_gas_sound_speed': 0.0
            })

        print(f"    Burned gas velocity: {results['burned_gas_velocity']:.1f} m/s")
        print(f"    Burned gas state: T={results['burned_gas_temperature']:.1f}K, "
              f"P={results['burned_gas_pressure']:.0f}Pa, "
              f"rho={results['burned_gas_density']:.3f}kg/m^3, "
              f"c={results['burned_gas_sound_speed']:.1f}m/s")

    except Exception as e:
        print(f"    Error in burned gas analysis: {e}")
        results.update({
            'burned_gas_velocity': 0.0,
            'burned_gas_temperature': 0.0,
            'burned_gas_pressure': 0.0,
            'burned_gas_density': 0.0,
            'burned_gas_sound_speed': 0.0
        })

    return results

def analyze_pressure_wave_properties(dataset, field_data: FieldData, config: ProcessingConfig,
                                    extractor=None) -> Dict[str, Any]:
    """Analyze maximum pressure wave properties"""
    print(f"  Analyzing pressure wave properties...")
    results = {}

    try:
        # Get thermodynamic calculator if available
        thermo_calc = None
        if hasattr(extractor, 'gas') and extractor.gas is not None:
            # Create a simple wrapper for Cantera gas object
            class CanteraThermodynamicCalculator:
                def __init__(self, gas):
                    self.gas = gas

                def calculate_state(self, temperature, pressure, composition):
                    self.gas.TPY = temperature, pressure, composition
                    return ThermodynamicState(
                        temperature=temperature,
                        pressure=pressure,
                        density=self.gas.density,
                        sound_speed=self.gas.sound_speed
                    )

            thermo_calc = CanteraThermodynamicCalculator(extractor.gas)

        # Create pressure wave analyzer with configuration
        pressure_wave_analyzer = create_pressure_wave_analyzer(
            detection_method=config.pressure_wave_detection_method,
            threshold_value=config.pressure_wave_threshold,
            field_name=config.pressure_wave_field,
            thermo_calculator=thermo_calc
        )

        # Perform pressure wave analysis
        pressure_wave_properties = pressure_wave_analyzer.analyze_pressure_wave_properties(
            dataset, field_data, config.extraction_location
        )
        results['pressure_wave_properties'] = pressure_wave_properties

        # Extract individual properties
        wave_position = getattr(pressure_wave_properties, 'position', 0.0)
        wave_index = getattr(pressure_wave_properties, 'index', 0)

        results['max_pressure_position'] = wave_position
        results['max_pressure_index'] = wave_index

        # Get the actual maximum pressure value
        max_pressure_value = np.max(field_data.pressure)
        results['max_pressure_value'] = max_pressure_value

        print(f"    Maximum pressure position: {wave_position:.6f} m (index: {wave_index})")
        print(f"    Maximum pressure value: {max_pressure_value:.0f} Pa")

        # Extract thermodynamic state at pressure wave
        thermo_state = getattr(pressure_wave_properties, 'thermodynamic_state', None)
        if thermo_state:
            results.update({
                'pressure_wave_temperature': getattr(thermo_state, 'temperature', 0.0),
                'pressure_wave_pressure': getattr(thermo_state, 'pressure', 0.0),
                'pressure_wave_density': getattr(thermo_state, 'density', 0.0),
                'pressure_wave_sound_speed': getattr(thermo_state, 'sound_speed', 0.0)
            })

            print(f"    Pressure wave state: T={results['pressure_wave_temperature']:.1f}K, "
                  f"rho={results['pressure_wave_density']:.3f}kg/m^3, "
                  f"c={results['pressure_wave_sound_speed']:.1f}m/s")
        else:
            results.update({
                'pressure_wave_temperature': 0.0,
                'pressure_wave_pressure': 0.0,
                'pressure_wave_density': 0.0,
                'pressure_wave_sound_speed': 0.0
            })

    except Exception as e:
        print(f"    Error in pressure wave analysis: {e}")
        import traceback
        print(f"    Traceback: {traceback.format_exc()}")
        results.update({
            'pressure_wave_properties': None,
            'max_pressure_position': 0.0,
            'max_pressure_index': 0,
            'max_pressure_value': 0.0,
            'pressure_wave_temperature': 0.0,
            'pressure_wave_pressure': 0.0,
            'pressure_wave_density': 0.0,
            'pressure_wave_sound_speed': 0.0
        })

    return results

def calculate_wave_velocities(all_results: List[Dict[str, Any]]) -> None:
    """Calculate flame and shock velocities from position time series"""
    print("Calculating wave velocities from time series...")
    
    if len(all_results) < 2:
        print("  Warning: Need at least 2 time points to calculate velocities")
        return
    
    # Filter out failed results first
    valid_results = [result for result in all_results if 'error' not in result and 'Flame' in result and 'Shock' in result]
    
    if len(valid_results) < 2:
        print(f"  Need at least 2 valid results to calculate velocities (have {len(valid_results)} valid out of {len(all_results)} total)")
        return
    
    # Extract time series data from valid results only
    times = np.array([result['Time'] for result in valid_results])
    flame_positions = np.array([result['Flame'].get('Position [m]', 0.0) for result in valid_results])
    shock_positions = np.array([result['Shock'].get('Position [m]', 0.0) for result in valid_results])
    skirt_positions = np.array([result['Flame'].get('Skirt Position [m]', 0.0) for result in valid_results])
    
    # Calculate flame velocities
    if len(flame_positions) > 1 and not np.all(flame_positions == 0):
        flame_velocities = np.gradient(flame_positions, times)
        for i, result in enumerate(valid_results):
            result['Flame']['Velocity [m / s]'] = flame_velocities[i]
            # Calculate relative velocity (flame velocity - gas velocity)
            gas_velocity = result['Flame'].get('Gas Velocity [m / s]', 0.0)
            result['Flame']['Relative Velocity [m / s]'] = flame_velocities[i] - gas_velocity
    else:
        for result in valid_results:
            result['Flame']['Velocity [m / s]'] = 0.0
            result['Flame']['Relative Velocity [m / s]'] = 0.0
    
    # Calculate shock velocities
    if len(shock_positions) > 1 and not np.all(shock_positions == 0):
        shock_velocities = np.gradient(shock_positions, times)
        for i, result in enumerate(valid_results):
            result['Shock']['Velocity [m / s]'] = shock_velocities[i]
    else:
        for result in valid_results:
            result['Shock']['Velocity [m / s]'] = 0.0

    # Calculate flame skirt velocities
    if len(skirt_positions) > 1 and not np.all(skirt_positions == 0):
        skirt_velocities = np.gradient(skirt_positions, times)
        for i, result in enumerate(valid_results):
            result['Flame']['Skirt Velocity [m / s]'] = skirt_velocities[i]
    else:
        for result in valid_results:
            result['Flame']['Skirt Velocity [m / s]'] = 0.0

    print(f"  Calculated velocities for {len(valid_results)} valid time points out of {len(all_results)} total")

########################################################################################################################
# Animation and Visualization Functions
########################################################################################################################

def generate_mp4_animations(directories: Dict[str, Path], config: ProcessingConfig) -> None:
    """Generate MP4 animations from saved PNG frames."""
    if not config.generate_mp4:
        return

    print(f"\n{'='*80}")
    print("GENERATING MP4 ANIMATIONS FROM FRAMES")
    print(f"{'='*80}")

    try:
        animator = FrameAnimator()

        # Map of animation directories to output names
        animation_dirs = {
            'Temperature': directories['anim_temperature'],
            'Local-Temperature': directories['anim_temperature_local'],
            'Pressure': directories['anim_pressure'],
            'Local-Pressure': directories['anim_pressure_local'],
            'GasVelocity': directories['anim_velocity'],
            'Local-GasVelocity': directories['anim_velocity_local'],
            'HeatReleaseRate': directories['anim_heat_release'],
            'Local-HeatReleaseRate': directories['anim_heat_release_local'],
            'Flame-Geometry': directories['anim_flame_geometry'],
            'Flame-Thickness': directories['anim_flame_thickness'],
            'Collective': directories['anim_collective'],
        }

        # Create output directory for videos
        video_output_dir = directories['base'] / 'Animation-Videos'
        video_output_dir.mkdir(exist_ok=True)

        # Process each animation directory
        for name, frame_dir in animation_dirs.items():
            if not frame_dir.exists():
                continue

            # Check if directory has PNG files
            png_files = list(frame_dir.glob('*.png'))
            if not png_files:
                print(f"  No frames found in {name} - skipping")
                continue

            print(f"  Processing {name}: {len(png_files)} frames")

            # Generate animations in requested formats
            for format in config.animation_formats:
                output_file = video_output_dir / f"{name}_animation.{format}"

                try:
                    print(f"    Creating {format.upper()} animation...")
                    animator.create_animation(
                        frame_directory=frame_dir,
                        output_path=output_file,
                        frame_rate=config.animation_fps,
                        format=format
                    )
                    print(f"    [SUCCESS] Saved to: {output_file.name}")

                except Exception as e:
                    print(f"    [FAILED] Failed to create {format}: {e}")

        print(f"\nAll animations saved to: {video_output_dir}")

    except Exception as e:
        print(f"Error generating animations: {e}")

def create_animation_frames(dataset, field_data: FieldData, results: Dict[str, Any], 
                           directories: Dict[str, Path], config: ProcessingConfig) -> None:
    """Create animation frames for various fields"""
    if not config.create_animations:
        return
        
    print(f"  Creating animation frames...")
    
    try:
        # Get flame position for local views
        flame_pos = results.get('flame_position', config.extraction_location)
        timestamp = dataset.current_time.to_value()
        basename = dataset.basename
        
        # Create plotters
        standard_plotter = StandardPlotter()
        local_plotter = LocalViewPlotter()
        
        # Temperature plots
        create_field_animation(
            standard_plotter, field_data.coordinates, field_data.temperature,
            directories['anim_temperature'], basename, "Temperature", timestamp
        )
        create_local_field_animation(
            local_plotter, field_data.coordinates, field_data.temperature,
            directories['anim_temperature_local'], basename, "Local Temperature",
            flame_pos, config.local_window_size, timestamp
        )
        
        # Pressure plots
        create_field_animation(
            standard_plotter, field_data.coordinates, field_data.pressure,
            directories['anim_pressure'], basename, "Pressure", timestamp
        )
        create_local_field_animation(
            local_plotter, field_data.coordinates, field_data.pressure,
            directories['anim_pressure_local'], basename, "Local Pressure",
            flame_pos, config.local_window_size, timestamp
        )
        
        # Velocity plots - both domain-wide and local
        if field_data.velocity_x is not None:
            # Domain-wide velocity plot
            create_field_animation(
                standard_plotter, field_data.coordinates, field_data.velocity_x,
                directories['anim_velocity'], basename, "Gas Velocity", timestamp
            )
            # Local velocity plot
            create_local_field_animation(
                local_plotter, field_data.coordinates, field_data.velocity_x,
                directories['anim_velocity_local'], basename, "Local Gas Velocity",
                flame_pos, config.local_window_size, timestamp
            )
        
        # Heat release rate plots - both domain-wide and local
        if field_data.heat_release_rate is not None:
            # Domain-wide heat release rate plot
            create_field_animation(
                standard_plotter, field_data.coordinates, field_data.heat_release_rate,
                directories['anim_heat_release'], basename, "Heat Release Rate", timestamp
            )
            # Local heat release rate plot
            create_local_field_animation(
                local_plotter, field_data.coordinates, field_data.heat_release_rate,
                directories['anim_heat_release_local'], basename, "Local Heat Release Rate",
                flame_pos, config.local_window_size, timestamp
            )
        
        # Flame surface contour plot
        if results.get('flame_properties') and (config.flame_config.enable_all or config.flame_config.enable_surface_length):
            create_flame_contour_animation(
                dataset, results['flame_properties'],
                directories['anim_flame_geometry'], basename, timestamp
            )

        # Flame thickness plot
        if results.get('flame_thickness', 0.0) > 0 and results.get('flame_properties') and (config.flame_config.enable_all or config.flame_config.enable_thickness):
            create_flame_thickness_animation(
                dataset, field_data, results['flame_properties'], results['flame_thickness'],
                directories['anim_flame_thickness'], basename, timestamp
            )
        
        # Collective pressure-temperature plot
        create_collective_animation(
            standard_plotter, field_data.coordinates, 
            {"Temperature": field_data.temperature, "Pressure": field_data.pressure},
            directories['anim_collective'], basename, timestamp
        )
        
        print(f"    Animation frames created successfully")
        
    except Exception as e:
        print(f"    Error creating animation frames: {e}")

def create_field_animation(plotter, x_data, y_data, output_dir: Path, basename: str,
                          field_name: str, timestamp: float) -> None:
    """Create single field animation frame"""
    try:
        output_path = output_dir / f"{basename}.png"
        frame = AnimationFrame(
            dataset_basename=basename,
            field_name=field_name,
            x_data=x_data,
            y_data=y_data,
            output_path=output_path,
            timestamp=timestamp
        )
        
        plotter.create_field_plot(frame)
        
    except Exception as e:
        print(f"      Error creating {field_name} frame: {e}")

def create_local_field_animation(plotter, x_data, y_data, output_dir: Path, basename: str,
                                field_name: str, center: float, window_size: float,
                                timestamp: float) -> None:
    """Create local view animation frame"""
    try:
        output_path = output_dir / f"{basename}.png"
        frame = AnimationFrame(
            dataset_basename=basename,
            field_name=field_name,
            x_data=x_data,
            y_data=y_data,
            output_path=output_path,
            timestamp=timestamp
        )
        
        plotter.create_local_view(frame, center, window_size)
        
    except Exception as e:
        print(f"      Error creating {field_name} local frame: {e}")

def create_collective_animation(plotter, x_data, field_dict: Dict[str, np.ndarray],
                               output_dir: Path, basename: str, timestamp: float) -> None:
    """Create collective multi-field animation frame"""
    try:
        output_path = output_dir / f"{basename}.png"
        plotter.create_multi_field_plot(x_data, field_dict, output_path,
                                       title=f"Collective Fields - {basename}")
        
    except Exception as e:
        print(f"      Error creating collective frame: {e}")

def create_flame_contour_animation(dataset, flame_properties, output_dir: Path, 
                                  basename: str, timestamp: float) -> None:
    """Create flame surface contour animation frame using only pre-calculated data"""
    try:
        # Get pre-calculated contour points only
        contour_points = getattr(flame_properties, 'contour_points', None)
        surface_length = getattr(flame_properties, 'surface_length', 0.0)
        
        if contour_points is None or len(contour_points) == 0:
            print(f"      Warning: No pre-calculated contour points available - skipping contour plot")
            return
        
        # Create flame contour plot using only pre-calculated data
        import matplotlib.pyplot as plt
        
        output_path = output_dir / f"{basename}.png"
        
        plt.figure(figsize=(8, 6))
        
        # Plot flame contour (assuming it's already sorted)
        plt.plot(contour_points[:, 0], contour_points[:, 1], 
               'r-', linewidth=2, label='Sorted Flame Contour')
        
        # Optionally show points as scatter
        plt.scatter(contour_points[:, 0], contour_points[:, 1], 
                   color='k', s=10, alpha=0.6, label='Contour Points')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'{timestamp:.6e} s\nSurface Length: {surface_length:.6f} m')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"      Created flame contour frame: {len(contour_points)} pre-calculated points")
        
    except Exception as e:
        print(f"      Error creating flame contour frame: {e}")

def create_flame_thickness_animation(dataset, field_data: FieldData, flame_properties, 
                                   flame_thickness: float, output_dir: Path, basename: str, 
                                   timestamp: float) -> None:
    """Create 2D temperature contour plot with flame contour and colored normal line overlay"""
    try:
        # Get flame position and pre-calculated contour points
        flame_pos = getattr(flame_properties, 'position', None)
        contour_points = getattr(flame_properties, 'contour_points', None)
        
        # Check for pre-stored plotting data
        region_grid = getattr(flame_properties, 'region_grid', None)
        region_temperature = getattr(flame_properties, 'region_temperature', None)
        normal_line = getattr(flame_properties, 'normal_line', None)
        interpolated_temperatures = getattr(flame_properties, 'interpolated_temperatures', None)
        
        if flame_pos is None:
            print(f"      Warning: No flame position for thickness plot")
            return
        
        if contour_points is None or len(contour_points) == 0:
            print(f"      Warning: No pre-calculated contour points for thickness plot")
            return
        
        # Only use pre-stored data - do not recalculate anything
        if (region_grid is None or region_temperature is None or 
            normal_line is None or interpolated_temperatures is None):
            print(f"      Warning: No pre-calculated thickness plotting data available - skipping thickness plot")
            return
            
        # Use pre-stored data only
        X, Y = np.meshgrid(np.unique(region_grid[:, 0]), np.unique(region_grid[:, 1]))
        temp_field = region_temperature
        center_location = flame_pos
        
        # Create plot matching original format exactly
        import matplotlib.pyplot as plt
        
        output_path = output_dir / f"{basename}.png"
        
        plt.figure(figsize=(8, 6))
        
        # Plot flame center (red marker)
        plt.scatter(region_grid[len(region_grid) // 2, 0], region_grid[len(region_grid) // 2, 1], 
                   marker='o', color='r', s=100,
                   label=f'Flame Center: ({region_grid[len(region_grid) // 2, 0]:.6f}, {region_grid[len(region_grid) // 2, 1]:.6f})')
        
        # Plot simulation grid with temperature coloring
        plt.scatter(X.flatten(), Y.flatten(), c=temp_field.flatten(), cmap='hot')
        
        # Plot normal line with temperature coloring
        plt.scatter(normal_line[:, 0], normal_line[:, 1], c=interpolated_temperatures, cmap='hot')
        
        # Plot flame contour
        plt.plot(contour_points[:, 0], contour_points[:, 1], label='Sorted Flame Contour')
        
        # Set limits to match grid bounds
        plt.xlim(min(X.flatten()), max(X.flatten()))
        plt.ylim(min(Y.flatten()), max(Y.flatten()))
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.colorbar()
        plt.title(f'Flame Normal: {timestamp:.6e} s')
        
        plt.savefig(output_path, format='png')
        plt.close()
        
        print(f"      Created flame thickness frame: {flame_thickness*1e6:.1f} um matching original format")
        
    except Exception as e:
        print(f"      Error creating flame thickness frame: {e}")

########################################################################################################################
# Results Output Functions
########################################################################################################################

def organize_results_for_output(results: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
    """Organize results in the format expected by the output files"""
    organized = {
        'Time': timestamp,
        'Flame': {
            'Index': results.get('flame_index', 0),
            'Position [m]': results.get('flame_position', 0.0),
            'Gas Velocity [m / s]': results.get('burned_gas_velocity', 0.0),
            'Thermodynamic Temperature [K]': results.get('flame_temperature', 0.0),
            'Thermodynamic Pressure [kg / m / s^2]': results.get('flame_pressure', 0.0),
            'Thermodynamic Density [kg / m^3]': results.get('flame_density', 0.0),
            'Thermodynamic Sound Speed': results.get('flame_sound_speed', 0.0),
            'HRR': results.get('flame_hrr', 0.0),
            'Flame Thickness': results.get('flame_thickness', 0.0),
            'Surface Length [m]': results.get('surface_length', 0.0),
            'Consumption Rate [kg / s]': results.get('consumption_rate', 0.0),
            'Burning Velocity [m / s]': results.get('burning_velocity', 0.0),
            'Velocity [m / s]': results.get('flame_velocity', 0.0),
            'Relative Velocity [m / s]': results.get('flame_relative_velocity', 0.0),
            'Skirt Position [m]': results.get('flame_skirt_position', 0.0),
            'Skirt Velocity [m / s]': results.get('flame_skirt_velocity', 0.0)
        },
        'Burned Gas': {
            'Gas Velocity [m / s]': results.get('burned_gas_velocity', 0.0),
            'Thermodynamic Temperature [K]': results.get('burned_gas_temperature', 0.0),
            'Thermodynamic Pressure [kg / m / s^2]': results.get('burned_gas_pressure', 0.0),
            'Thermodynamic Density [kg / m^3]': results.get('burned_gas_density', 0.0),
            'Thermodynamic Sound Speed': results.get('burned_gas_sound_speed', 0.0)
        },
        'Shock': {
            'Index': results.get('shock_index', 0),
            'Position [m]': results.get('shock_position', 0.0),
            'PreShockThermodynamicState Temperature [K]': results.get('pre_shock_temperature', 0.0),
            'PreShockThermodynamicState Pressure [kg / m / s^2]': results.get('pre_shock_pressure', 0.0),
            'PreShockThermodynamicState Density [kg / m^3]': results.get('pre_shock_density', 0.0),
            'PreShockThermodynamicState Sound Speed': results.get('pre_shock_sound_speed', 0.0),
            'PostShockThermodynamicState Temperature [K]': results.get('post_shock_temperature', 0.0),
            'PostShockThermodynamicState Pressure [kg / m / s^2]': results.get('post_shock_pressure', 0.0),
            'PostShockThermodynamicState Density [kg / m^3]': results.get('post_shock_density', 0.0),
            'PostShockThermodynamicState Sound Speed': results.get('post_shock_sound_speed', 0.0),
            'Velocity [m / s]': results.get('shock_velocity', 0.0)
        },
        'Pressure Wave': {
            'Index': results.get('max_pressure_index', 0),
            'Position [m]': results.get('max_pressure_position', 0.0),
            'Maximum Pressure [kg / m / s^2]': results.get('max_pressure_value', 0.0),
            'Thermodynamic Temperature [K]': results.get('pressure_wave_temperature', 0.0),
            'Thermodynamic Pressure [kg / m / s^2]': results.get('pressure_wave_pressure', 0.0),
            'Thermodynamic Density [kg / m^3]': results.get('pressure_wave_density', 0.0),
            'Thermodynamic Sound Speed': results.get('pressure_wave_sound_speed', 0.0)
        }
    }
    return organized

def save_combined_results(all_results: List[Dict[str, Any]], output_file: Path) -> None:
    """Save all results to a single combined file matching the original format"""
    print(f"Saving combined results to: {output_file}")
    
    if not all_results:
        print("  No results to save")
        return
    
    def flatten_dict(d, parent_key='', sep=' '):
        """Flatten nested dictionary with hierarchical keys"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Flatten first result to get all headers
    flattened_first = flatten_dict(all_results[0])
    time_value = flattened_first.pop('Time', 0.0)
    headers = ['Time'] + list(flattened_first.keys())
    
    # Field width for formatting (65 characters to match original)
    field_width = 65
    
    try:
        with open(ensure_long_path_prefix(str(output_file)), 'w') as f:
            # Write column indices (line 1) - all columns left-justified, 65 chars each
            index_line = "#"
            for i in range(1, len(headers) + 1):
                if i == 1:  # First column
                    index_line += f"  {i:<{field_width-2}}"  # 2 spaces + "1" + left-justified in 63 chars
                else:
                    index_line += f"{i:<{field_width}}"  # Left-justified in 65-char field
            f.write(index_line + '\n')
            
            # Write column headers (line 2) - all columns left-justified, 65 chars each
            header_line = "#"
            for i, header in enumerate(headers):
                if i == 0:  # First column (Time)
                    header_line += f"  {header:<{field_width-2}}"  # 2 spaces + Time + left-justified in 63 chars
                else:
                    header_line += f"{header:<{field_width}}"  # Left-justified in 65-char field
            f.write(header_line + '\n')
            
            # Write data - all columns left-justified, 65 chars each
            for result in all_results:
                flattened = flatten_dict(result)
                time_val = flattened.pop('Time', 0.0)
                values = [time_val] + list(flattened.values())
                
                data_line = ""
                for i, value in enumerate(values):
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        value = 0.0
                    
                    if i == 0:  # First column (Time)
                        data_line += f"   {float(value):<{field_width-3}.6e}"  # Three spaces + left-justified in 62 chars
                    else:
                        data_line += f"{float(value):<{field_width}.6e}"  # Left-justified in 65-char field
                f.write(data_line + '\n')
        
        print(f"  Combined results saved successfully ({len(all_results)} time steps)")
        
    except Exception as e:
        print(f"  Error saving combined results: {e}")

# Individual results saving removed - using single combined file only

########################################################################################################################
# Main Processing Function
########################################################################################################################

def process_single_dataset(dataset_path: str, config: ProcessingConfig, 
                          directories: Dict[str, Path]) -> ProcessingResult:
    """Process a single dataset completely, returning a ProcessingResult object"""
    
    print(f"\n{'='*80}")
    print(f"Processing: {Path(dataset_path).name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    results = {}
    
    try:
        # Load dataset
        print("Loading dataset...")
        data_loader = create_data_loader("yt")
        dataset = data_loader.load_dataset(dataset_path)
        
        timestamp = dataset.current_time.to_value()
        results['time'] = timestamp
        results['dataset_basename'] = dataset.basename
        
        print(f"  Dataset: {dataset.basename}")
        print(f"  Time: {timestamp:.6e} s")
        
        # Create dataset info
        dataset_info = DatasetInfo.from_path(dataset_path)
        dataset_info.timestamp = timestamp
        dataset_info.basename = dataset.basename
        
        # Extract field data
        field_data, extractor = extract_field_data(dataset, config)
        
        # Initialize result components
        flame_data = None
        shock_data = None
        
        # Perform flame analysis
        if config.analyze_flame:
            flame_results = analyze_flame_properties(dataset, field_data, config)
            results.update(flame_results)
            flame_data = flame_results.get('flame_properties')
        
        # Perform shock analysis  
        if config.analyze_shock:
            shock_results = analyze_shock_properties(dataset, field_data, config)
            results.update(shock_results)
            shock_data = shock_results.get('shock_properties')
        
        # Perform burned gas analysis
        if config.analyze_burned_gas and results.get('flame_position', 0.0) > 0:
            # Create burned gas analyzer
            burned_gas_analyzer = create_burned_gas_analyzer(offset=config.burned_gas_offset)
            burned_gas_results = analyze_burned_gas_properties(
                dataset, field_data, burned_gas_analyzer,
                results['flame_position'], results['flame_index'], config)
            results.update(burned_gas_results)

        # Perform pressure wave analysis
        if config.analyze_pressure_wave:
            pressure_wave_results = analyze_pressure_wave_properties(
                dataset, field_data, config, extractor)
            results.update(pressure_wave_results)

        # Create animations and plots
        if config.create_animations:
            create_animation_frames(dataset, field_data, results, directories, config)
        
        processing_time = time.time() - start_time
        print(f"  Processing completed in {processing_time:.2f}s")
        
        # Create ProcessingResult
        processing_result = ProcessingResult(
            dataset_info=dataset_info,
            flame_data=flame_data,
            shock_data=shock_data,
            success=True,
            processing_time=processing_time
        )
        
        # Store the results dict as an attribute for backward compatibility
        processing_result._legacy_results = results
        
        return processing_result
        
    except Exception as e:
        print(f"  Error processing dataset: {e}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        
        processing_time = time.time() - start_time
        dataset_info = DatasetInfo.from_path(dataset_path)
        
        return ProcessingResult(
            dataset_info=dataset_info,
            success=False,
            error_message=str(e),
            processing_time=processing_time
        )


def convert_processing_result_to_dict(result: ProcessingResult) -> Dict[str, Any]:
    """Convert ProcessingResult back to the dictionary format for output compatibility"""
    if not result.is_successful():
        return {'Time': 0.0, 'error': result.error_message}
    
    # Extract data from legacy results if available
    if hasattr(result, '_legacy_results'):
        metadata = result._legacy_results
        timestamp = result.dataset_info.timestamp if result.dataset_info else 0.0
        return organize_results_for_output(metadata, timestamp)
    else:
        # Fallback: construct basic result dict from ProcessingResult fields
        timestamp = result.dataset_info.timestamp if result.dataset_info else 0.0
        
        basic_result = {
            'time': timestamp,
            'dataset_basename': result.dataset_info.basename if result.dataset_info else 'unknown'
        }
        
        # Add flame data if available
        if result.flame_data:
            basic_result.update({
                'flame_position': result.flame_data.position,
                'flame_temperature': result.flame_data.temperature,
                'flame_pressure': result.flame_data.pressure,
                'flame_velocity': result.flame_data.velocity,
                'flame_thickness': getattr(result.flame_data, 'thickness', 0.0),
                'flame_surface_length': getattr(result.flame_data, 'surface_length', 0.0),
                'flame_hrr': getattr(result.flame_data, 'hrr', 0.0),
            })
        
        # Add shock data if available
        if result.shock_data:
            basic_result.update({
                'shock_position': result.shock_data.position,
                'shock_velocity': result.shock_data.velocity,
                'shock_pre_temperature': result.shock_data.pre_shock_temperature,
                'shock_pre_pressure': result.shock_data.pre_shock_pressure,
                'shock_post_temperature': result.shock_data.post_shock_temperature,
                'shock_post_pressure': result.shock_data.post_shock_pressure,
            })
        
        return organize_results_for_output(basic_result, timestamp)

########################################################################################################################
# Parallel Processing Functions
########################################################################################################################

def process_datasets_parallel(dataset_paths: List[str], config: ProcessingConfig, 
                              directories: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Process datasets using proper MPI coordination through parallel module"""
    
    if MPI_RANK == 0:
        print(f"\n{'='*100}")
        print(f"MPI PARALLEL PROCESSING (Size: {MPI_SIZE})")
        print(f"{'='*100}")
    
    # Create processing function that matches the interface expected by the coordinator
    def processor_function(dataset_path: str) -> ProcessingResult:
        try:
            return process_single_dataset(dataset_path, config, directories)
        except Exception as e:
            dataset_info = DatasetInfo.from_path(dataset_path)
            return ProcessingResult(
                dataset_info=dataset_info,
                success=False,
                error_message=str(e),
                processing_time=0.0
            )
    
    # Create logger (simplified for this context)
    class SimpleLogger:
        def log_info(self, message: str, **kwargs):
            rank = kwargs.get('rank', MPI_RANK)
            print(f"Rank {rank}: {message}")
        
        def log_warning(self, message: str, **kwargs):
            rank = kwargs.get('rank', MPI_RANK)
            print(f"Rank {rank} WARNING: {message}")
        
        def log_error(self, message: str, **kwargs):
            rank = kwargs.get('rank', MPI_RANK)
            print(f"Rank {rank} ERROR: {message}")
        
        def log_debug(self, message: str, **kwargs):
            rank = kwargs.get('rank', MPI_RANK)
            print(f"Rank {rank} DEBUG: {message}")
    
    logger = SimpleLogger()
    
    # Use the proper MPI coordinator
    try:
        coordinator = MPICoordinator(logger=logger)
        processing_batch = coordinator.coordinate_processing(dataset_paths, processor_function)
        
        # Convert ProcessingBatch to dictionary format for backward compatibility
        if MPI_RANK == 0:
            results = []
            for result in processing_batch.results:
                dict_result = convert_processing_result_to_dict(result)
                results.append(dict_result)
            
            # Sort by time
            results.sort(key=lambda x: x.get('Time', 0.0))
            
            print(f"\nMPI processing summary:")
            print(f"  Total results collected: {len(results)}")
            successful = len([r for r in results if 'error' not in r])
            failed = len(results) - successful
            print(f"  Successful: {successful}, Failed: {failed}")
            
            return results
        else:
            return []
            
    except Exception as e:
        print(f"Rank {MPI_RANK}: Error in MPI coordination: {e}")
        if MPI_RANK == 0:
            # Fallback to manual distribution if coordinator fails
            return process_datasets_parallel_fallback(dataset_paths, config, directories)
        else:
            return []


def process_datasets_parallel_fallback(dataset_paths: List[str], config: ProcessingConfig, 
                                      directories: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Fallback manual MPI implementation if coordinator fails"""
    
    print(f"Rank {MPI_RANK}: Using fallback MPI implementation")
    
    # Synchronize all processes before starting
    comm.barrier()
    
    # Manual work distribution - each rank processes subset of datasets
    my_datasets = []
    for i, dataset_path in enumerate(dataset_paths):
        if i % MPI_SIZE == MPI_RANK:
            my_datasets.append(dataset_path)
    
    # Each rank processes its assigned datasets
    my_results = []
    for dataset_path in my_datasets:
        print(f"Rank {MPI_RANK}: Processing {Path(dataset_path).name}")
        
        try:
            result = process_single_dataset(dataset_path, config, directories)
            dict_result = convert_processing_result_to_dict(result)
            if 'error' not in dict_result:
                my_results.append(dict_result)
            else:
                print(f"Rank {MPI_RANK}: Dataset failed: {dict_result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Rank {MPI_RANK}: Error processing {Path(dataset_path).name}: {e}")
    
    # Gather all results to rank 0
    try:
        all_results = comm.gather(my_results, root=0)
    except Exception as e:
        print(f"Rank {MPI_RANK}: Error gathering results: {e}")
        return [] if MPI_RANK == 0 else []
    
    if MPI_RANK == 0:
        # Flatten the gathered results
        final_results = []
        for rank_results in all_results or []:
            if rank_results:
                final_results.extend(rank_results)
        
        # Sort by time
        final_results.sort(key=lambda x: x.get('Time', 0.0))
        
        print(f"\nFallback MPI processing summary:")
        print(f"  Total results collected: {len(final_results)}")
        print(f"  Successful: {len(final_results)}, Failed: {len(dataset_paths) - len(final_results)}")
        
        return final_results
    else:
        return []

def process_datasets_sequential(dataset_paths: List[str], config: ProcessingConfig, 
                               directories: Dict[str, Path]) -> List[Dict[str, Any]]:
    """Process datasets sequentially (original implementation)"""
    
    print(f"\n{'='*100}")
    print("SEQUENTIAL PROCESSING")
    print(f"{'='*100}")
    
    all_results = []
    successful = 0
    failed = 0
    total_start_time = time.time()
    
    for i, dataset_path in enumerate(dataset_paths, 1):
        print(f"\nDataset {i}/{len(dataset_paths)}")
        
        try:
            result = process_single_dataset(str(dataset_path), config, directories)
            dict_result = convert_processing_result_to_dict(result)
            
            if 'error' not in dict_result:
                all_results.append(dict_result)
                successful += 1
            else:
                print(f"  Dataset failed: {dict_result.get('error', 'Unknown error')}")
                failed += 1
                
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user")
            break
        except Exception as e:
            print(f"  Unexpected error: {e}")
            failed += 1
            continue
    
    total_time = time.time() - total_start_time
    print(f"\nSequential processing summary:")
    print(f"  Successful: {successful}, Failed: {failed}")
    print(f"  Total time: {total_time:.2f}s")
    
    return all_results

########################################################################################################################
# Main Execution
########################################################################################################################


def main():
    """Main execution function"""


    # Only show header on rank 0 (master) for MPI runs
    if MPI_RANK == 0:
        print(f"\n{'='*100}")
        print(f"MAIN PELE PROCESSING SCRIPT v{VERSION}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if IS_MPI_RUN:
            print(f"MPI Mode: {MPI_SIZE} processes")
        else:
            print("Sequential Mode")
        print(f"{'='*100}")
    
    # Configuration
    config = ProcessingConfig()
    
    # Setup paths
    input_directory = Path('../../pele_data_2d')
    output_directory = Path(f"../../Processed-MPI-Global-Results-V{VERSION}")
    
    if MPI_RANK == 0:
        print(f"Input directory: {input_directory}")
        print(f"Output directory: {output_directory}")
        print(f"Extraction location: y = {config.extraction_location:.6f} m (extracting along X direction)")
    
    # Validate input directory (only rank 0 needs to check)
    if MPI_RANK == 0:
        if not input_directory.exists():
            print(f"\nERROR: Input directory does not exist: {input_directory}")
            if IS_MPI_RUN:
                comm.Abort(1)
            else:
                sys.exit(1)
    
    # Setup directories (only rank 0)
    if MPI_RANK == 0:
        directories = setup_directories(output_directory)
        print("Output directory structure created")
    else:
        directories = {}
    
    # Broadcast directories and config to all ranks for MPI
    if IS_MPI_RUN:
        directories = comm.bcast(directories, root=0)
        config = comm.bcast(config, root=0)
    
    # Setup logging on all ranks
    setup_logging("INFO")
    
    # Load dataset paths (only rank 0)
    dataset_paths = None
    if MPI_RANK == 0:
        try:
            print("\nScanning for datasets...")
            dataset_paths = load_dataset_paths(input_directory)
            
            if not dataset_paths:
                print(f"No datasets found in {input_directory}")
                if IS_MPI_RUN:
                    comm.Abort(1)
                else:
                    sys.exit(1)
                
            print(f"Found {len(dataset_paths)} datasets to process")
            
            
        except Exception as e:
            print(f"Error loading dataset paths: {e}")
            if IS_MPI_RUN:
                comm.Abort(1)
            else:
                sys.exit(1)
    
    # Broadcast dataset paths to all ranks for MPI
    if IS_MPI_RUN:
        dataset_paths = comm.bcast(dataset_paths, root=0)
    
    # Choose processing approach automatically
    total_start_time = time.time()
    
    if IS_MPI_RUN and MPI_AVAILABLE and MPI_SIZE > 1:
        # Use MPI parallel processing only when we have multiple processes
        all_results = process_datasets_parallel(dataset_paths, config, directories)
    else:
        # Use sequential processing for single process or when MPI unavailable
        all_results = process_datasets_sequential(dataset_paths, config, directories)
    
    # Only rank 0 handles post-processing and output for MPI runs
    if MPI_RANK == 0:
        # Calculate wave velocities if we have multiple time points
        if config.calculate_velocities and len(all_results) > 1:
            print(f"\n{'='*80}")
            print("POST-PROCESSING: CALCULATING WAVE VELOCITIES")
            print(f"{'='*80}")
            calculate_wave_velocities(all_results)
        
        # Save combined results - single file with all timesteps
        if config.save_combined_results and all_results:
            print(f"\n{'='*80}")
            print("SAVING COMBINED RESULTS")
            print(f"{'='*80}")

            # Sort results by time in ascending order
            all_results.sort(key=lambda x: x.get('Time', 0.0))

            # Create output filename
            combined_file = directories['base'] / f"Processed-MPI-Global-Results-V-{VERSION}.txt"
            save_combined_results(all_results, combined_file)

        # Generate MP4 animations from the saved frames
        if config.generate_mp4 and config.create_animations:
            generate_mp4_animations(directories, config)
        
        # Final summary
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*100}")
        print("PROCESSING SUMMARY")
        print(f"{'='*100}")
        print(f"Total datasets found: {len(dataset_paths)}")
        print(f"Successfully processed: {len(all_results)}")
        print(f"Failed: {len(dataset_paths) - len(all_results)}")
        print(f"Total processing time: {total_time:.2f}s")
        if len(all_results) > 0:
            print(f"Average time per dataset: {total_time/len(all_results):.2f}s")
        print(f"Results saved to: {output_directory}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if all_results:
            print(f"\nTime range processed: {all_results[0]['Time']:.6e} to {all_results[-1]['Time']:.6e} s")
            if len(all_results) > 1:
                dt = all_results[1]['Time'] - all_results[0]['Time'] 
                print(f"Time step: {dt:.6e} s")
        
        print(f"\n{'='*100}")

if __name__ == "__main__":
    main()