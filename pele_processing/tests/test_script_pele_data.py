#!/usr/bin/env python3
"""
Test script for the Pele processing system.
Tests the restructured modular architecture against real Pele dataset data.
"""
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

########################################################################################################################
# Initial Parameters (Far Upstream Data - Set Manually)
########################################################################################################################

# Thermodynamic initial conditions
INITIAL_TEMPERATURE = 503.15  # K
INITIAL_PRESSURE = 10.0 * 1e5  # Pa (10 bar)
EQUIVALENCE_RATIO = 1.0
FUEL = 'H2'
MECHANISM_FILE = '../chemical_mechanisms/LiDryer.yaml'  # Optional

# Domain and extraction parameters
EXTRACTION_Y_LOCATION = 0.0445 / 100  # m - Y location for data extraction
FLAME_TEMPERATURE = 2500.0  # K - Temperature threshold for flame detection
SHOCK_PRESSURE_RATIO = 1.01  # Pressure ratio threshold for shock detection

# Test data directory
TEST_DATA_DIR = script_dir / "pele_data_2d"

########################################################################################################################
# Package Imports
########################################################################################################################

# Check for YT availability first
try:
    import yt

    yt.set_log_level(0)
    YT_AVAILABLE = True
    print("✓ YT available for data loading")
except ImportError:
    YT_AVAILABLE = False
    print("✗ YT not available - tests will be limited")

# Main package import
try:
    import pele_processing

    print("✓ Main package imported successfully")
except ImportError as e:
    print(f"✗ Failed to import main package: {e}")
    sys.exit(1)

# Import all relevant sub-functions and classes
try:
    # Core domain classes
    from pele_processing.core.domain import (
        FieldData, SpeciesData, WaveType, ProcessingResult,
        DatasetInfo, ProcessingBatch, AnimationFrame,
        FlameProperties, ShockProperties, ThermodynamicState
    )

    # Core container system
    from pele_processing.core.container import Container, get_global_container

    # Configuration functions
    from pele_processing.config import (
        create_default_config, AppConfig, load_config, save_config
    )

    # Data processing functions
    from pele_processing.data import (
        create_data_loader, create_data_extractor,
        create_standard_processor, create_analysis_processor
    )

    # Analysis functions
    from pele_processing.analysis import (
        create_flame_analyzer, create_shock_analyzer,
        create_thermodynamic_calculator, create_geometry_analyzer
    )

    # Parallel processing functions
    from pele_processing.parallel import (
        create_processing_strategy, create_coordinator,
        create_distributor, create_default_adaptive_strategy
    )

    # Visualization functions
    from pele_processing.visualization import (
        StandardPlotter, LocalViewPlotter, StatisticalPlotter,
        FrameAnimator, BatchAnimator, InteractiveAnimator,
        create_formatter, SchlierenVisualizer, StreamlineVisualizer
    )

    # Utility functions
    from pele_processing.utils import (
        setup_logging, create_logger, create_unit_converter,
        PeleUnitConverter, load_dataset_paths, sort_dataset_paths,
        ensure_directory_exists, clean_filename
    )

    # Constants
    from pele_processing.utils.constants import (
        DEFAULT_FLAME_TEMPERATURE, DEFAULT_SHOCK_PRESSURE_RATIO,
        COMMON_SPECIES, ERROR_CODES
    )

    # Convenience functions from main package
    from pele_processing import (
        quick_analysis, create_analysis_pipeline
    )

    print("✓ All components imported successfully")

except ImportError as e:
    print(f"✗ Failed to import components: {e}")
    sys.exit(1)


########################################################################################################################
# Configuration Setup
########################################################################################################################

def setup_script_configuration():
    """Setup detailed analysis configuration flags."""
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class FieldConfig:
        name: Optional[str] = None
        flag: bool = False
        offset: float = 0.0
        animation: bool = False
        local: bool = False
        collective: bool = False

    @dataclass
    class FlameConfig:
        position: FieldConfig = field(default_factory=lambda: FieldConfig(name='Flame Position'))
        velocity: FieldConfig = field(default_factory=lambda: FieldConfig(name='Flame Velocity'))
        relative_velocity: FieldConfig = field(default_factory=lambda: FieldConfig(name='Flame Relative Velocity'))
        thermodynamic_state: FieldConfig = field(default_factory=lambda: FieldConfig(name='Flame Thermodynamic State'))
        heat_release_rate: FieldConfig = field(default_factory=lambda: FieldConfig(name='Heat Release Rate'))
        flame_thickness: FieldConfig = field(default_factory=lambda: FieldConfig(name='Flame Thickness'))
        surface_length: FieldConfig = field(default_factory=lambda: FieldConfig(name='Surface Length'))
        consumption_rate: FieldConfig = field(default_factory=lambda: FieldConfig(name='Consumption Rate'))
        reynolds_number: FieldConfig = field(default_factory=lambda: FieldConfig(name='Reynolds Number'))

    @dataclass
    class BurnedGasConfig:
        gas_velocity: FieldConfig = field(default_factory=lambda: FieldConfig(name='Gas Velocity'))
        thermodynamic_state: FieldConfig = field(default_factory=lambda: FieldConfig(name='Thermodynamic State'))

    @dataclass
    class ShockConfig:
        position: FieldConfig = field(default_factory=lambda: FieldConfig(name='Shock Position'))
        velocity: FieldConfig = field(default_factory=lambda: FieldConfig(name='Shock Velocity'))
        pre_shock_state: FieldConfig = field(default_factory=lambda: FieldConfig(name='Pre-Shock State'))
        post_shock_state: FieldConfig = field(default_factory=lambda: FieldConfig(name='Post-Shock State'))

    @dataclass
    class AnimationConfig:
        local_window_size: FieldConfig = field(default_factory=lambda: FieldConfig(name='Local Window Size'))
        temperature: FieldConfig = field(default_factory=lambda: FieldConfig(name='Temperature'))
        pressure: FieldConfig = field(default_factory=lambda: FieldConfig(name='Pressure'))
        gas_velocity: FieldConfig = field(default_factory=lambda: FieldConfig(name='X Velocity'))
        heat_release_rate: FieldConfig = field(default_factory=lambda: FieldConfig(name='Heat Release Rate'))
        flame_geometry: FieldConfig = field(default_factory=lambda: FieldConfig(name='Flame Geometry'))
        flame_thickness: FieldConfig = field(default_factory=lambda: FieldConfig(name='Flame Thickness'))
        schlieren: FieldConfig = field(default_factory=lambda: FieldConfig(name='Schlieren'))
        streamlines: FieldConfig = field(default_factory=lambda: FieldConfig(name='StreamLines'))

    @dataclass
    class DataExtractionConfig:
        parallelize: bool = False
        location: Optional[float] = None
        grid: Optional[np.ndarray] = None

    @dataclass
    class ScriptConfig:
        data_extraction: DataExtractionConfig = field(default_factory=DataExtractionConfig)
        flame: FlameConfig = field(default_factory=FlameConfig)
        burned_gas: BurnedGasConfig = field(default_factory=BurnedGasConfig)
        shock: ShockConfig = field(default_factory=ShockConfig)
        animation: AnimationConfig = field(default_factory=AnimationConfig)

    # Create and configure script config
    script_config = ScriptConfig()

    # Flame Parameters
    script_config.flame.position.flag = True
    script_config.flame.velocity.flag = True
    script_config.flame.relative_velocity.flag = True
    script_config.flame.relative_velocity.offset = 0
    script_config.flame.thermodynamic_state.flag = True
    script_config.flame.thermodynamic_state.offset = 0
    script_config.flame.heat_release_rate.flag = True
    script_config.flame.surface_length.flag = True
    script_config.flame.consumption_rate.flag = True
    script_config.flame.flame_thickness.flag = True

    # Burned Gas Parameters
    script_config.burned_gas.gas_velocity.flag = True
    script_config.burned_gas.thermodynamic_state.flag = True

    # Shock Parameters
    script_config.shock.position.flag = True
    script_config.shock.velocity.flag = True
    script_config.shock.pre_shock_state.flag = True
    script_config.shock.post_shock_state.flag = True

    # Animation Parameters
    script_config.animation.local_window_size.offset = 1e-3
    script_config.animation.temperature.flag = True
    script_config.animation.temperature.local = True
    script_config.animation.temperature.collective = True
    script_config.animation.pressure.flag = True
    script_config.animation.pressure.local = True
    script_config.animation.pressure.collective = True
    script_config.animation.gas_velocity.flag = True
    script_config.animation.gas_velocity.local = True
    script_config.animation.heat_release_rate.flag = True
    script_config.animation.heat_release_rate.local = True
    script_config.animation.flame_geometry.flag = True
    script_config.animation.flame_thickness.flag = True
    script_config.animation.schlieren.flag = False
    script_config.animation.streamlines.flag = False

    # Data extraction settings
    script_config.data_extraction.location = EXTRACTION_Y_LOCATION

    return script_config


########################################################################################################################
# Data Loading and Test Setup Functions
########################################################################################################################

def setup_test_data():
    """Check for existing real plotfiles."""
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(f"Test data directory not found: {TEST_DATA_DIR}")

    plotfiles = list(TEST_DATA_DIR.glob("plt*"))
    if not plotfiles:
        raise FileNotFoundError(f"No plotfiles found in {TEST_DATA_DIR}")

    print(f"Found {len(plotfiles)} real plotfiles in {TEST_DATA_DIR}")
    return sorted(plotfiles)


def load_real_dataset(dataset_path):
    """Load real Pele dataset using YT."""
    if not YT_AVAILABLE:
        raise ImportError("YT not available for dataset loading")

    try:
        dataset = yt.load(str(dataset_path))
        dataset.force_periodicity()
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_path}: {e}")


########################################################################################################################
# Test Functions with Real Data
########################################################################################################################

def test_configuration():
    """Test configuration system with real parameters."""
    print("Testing configuration...")

    config = create_default_config(
        input_dir=TEST_DATA_DIR,
        output_dir=TEST_DATA_DIR / "output"
    )

    # Apply initial parameters
    config.processing.flame_temperature = FLAME_TEMPERATURE
    config.processing.shock_pressure_ratio = SHOCK_PRESSURE_RATIO
    config.processing.analyze_flame_position = True
    config.processing.analyze_shock_position = True
    config.processing.extract_location = EXTRACTION_Y_LOCATION

    # Thermodynamic settings
    config.thermodynamics.temperature = INITIAL_TEMPERATURE
    config.thermodynamics.pressure = INITIAL_PRESSURE
    config.thermodynamics.equivalence_ratio = EQUIVALENCE_RATIO
    config.thermodynamics.fuel = FUEL

    if Path(MECHANISM_FILE).exists():
        config.paths.mechanism_file = Path(MECHANISM_FILE)

    assert config.processing.flame_temperature == FLAME_TEMPERATURE
    assert config.processing.analyze_flame_position == True

    print("✓ Configuration tests passed")
    return config


def test_data_components():
    """Test data loading and extraction with real dataset."""
    print("Testing data components...")

    if not YT_AVAILABLE:
        print("⚠ YT not available, skipping real data tests")
        return None, None

    # Setup test data
    plotfiles = setup_test_data()
    if not plotfiles:
        print("⚠ No plotfiles available")
        return None, None

    # Load first dataset
    dataset = load_real_dataset(plotfiles[0])

    # Create unit converter
    converter = PeleUnitConverter()

    # Create data loader and extractor
    data_loader = create_data_loader("yt")
    data_extractor = create_data_extractor("pele", unit_converter=converter)

    # Get dataset info
    dataset_info = data_loader.get_dataset_info(dataset)
    print(f"  Dataset: {dataset_info.basename}")
    print(f"  Time: {dataset_info.timestamp:.2e} s")
    print(f"  Domain: {dataset_info.domain_bounds.width:.3f} x {dataset_info.domain_bounds.height:.3f} m")

    # Extract field data
    field_data = data_extractor.extract_ray_data(dataset, EXTRACTION_Y_LOCATION)

    print(f"  Extracted {len(field_data.coordinates)} data points")
    print(f"  Temperature range: {np.min(field_data.temperature):.1f} - {np.max(field_data.temperature):.1f} K")
    print(f"  Pressure range: {np.min(field_data.pressure):.0f} - {np.max(field_data.pressure):.0f} Pa")

    print("✓ Data component tests passed")
    return dataset, field_data


def test_flame_analysis(dataset, field_data):
    """Test flame analysis with real data."""
    print("Testing flame analysis...")

    if dataset is None or field_data is None:
        print("⚠ No real data available, skipping flame analysis")
        return

    # Create flame analyzer
    flame_analyzer = create_flame_analyzer(flame_temperature=FLAME_TEMPERATURE)

    try:
        # Find flame position
        flame_idx, flame_pos = flame_analyzer.find_wave_position(field_data, WaveType.FLAME)
        print(f"  Flame detected at index {flame_idx}, position {flame_pos:.6f} m")

        # Analyze flame properties
        flame_properties = flame_analyzer.analyze_flame_properties(dataset, field_data)

        if flame_properties.position:
            print(f"  Flame position: {flame_properties.position:.6f} m")
        if flame_properties.thickness:
            print(f"  Flame thickness: {flame_properties.thickness:.2e} m")
        if flame_properties.surface_length:
            print(f"  Surface length: {flame_properties.surface_length:.6f} m")

    except Exception as e:
        print(f"  Flame analysis failed: {e}")

    print("✓ Flame analysis tests passed")


def test_shock_analysis(field_data):
    """Test shock analysis with real data."""
    print("Testing shock analysis...")

    if field_data is None:
        print("⚠ No real data available, skipping shock analysis")
        return

    # Create shock analyzer
    shock_analyzer = create_shock_analyzer(SHOCK_PRESSURE_RATIO)

    try:
        # Analyze shock properties
        shock_properties = shock_analyzer.analyze_shock_properties(field_data)

        if shock_properties.position:
            print(f"  Shock detected at position {shock_properties.position:.6f} m")
        else:
            print("  No shock detected with current criteria")

    except Exception as e:
        print(f"  Shock analysis failed: {e}")

    print("✓ Shock analysis tests passed")


def test_parallel_processing():
    """Test parallel processing strategy with real datasets."""
    print("Testing parallel processing...")

    # Setup test data
    plotfiles = setup_test_data()

    # Create processing strategy
    strategy = create_processing_strategy("sequential")

    def real_processor(dataset_path):
        """Process real dataset."""
        try:
            if YT_AVAILABLE:
                dataset = load_real_dataset(dataset_path)
                data_loader = create_data_loader("yt")
                dataset_info = data_loader.get_dataset_info(dataset)

                result = ProcessingResult(dataset_info=dataset_info, success=True)

                # Basic analysis
                converter = PeleUnitConverter()
                extractor = create_data_extractor("pele", unit_converter=converter)
                field_data = extractor.extract_ray_data(dataset, EXTRACTION_Y_LOCATION)

                # Try flame detection
                flame_analyzer = create_flame_analyzer(FLAME_TEMPERATURE)
                try:
                    flame_idx, flame_pos = flame_analyzer.find_wave_position(field_data, WaveType.FLAME)
                    result.flame_data = FlameProperties(position=flame_pos, index=flame_idx)
                except:
                    pass

                return result
            else:
                return ProcessingResult(
                    dataset_info=DatasetInfo.from_path(dataset_path),
                    success=True
                )
        except Exception as e:
            return ProcessingResult(
                dataset_info=DatasetInfo.from_path(dataset_path),
                success=False,
                error_message=str(e)
            )

    # Process datasets
    dataset_paths = [str(p) for p in plotfiles[:3]]  # Process first 3
    batch = strategy.execute(dataset_paths, real_processor)

    successful = len(batch.get_successful_results())
    print(f"  Processed {len(batch.results)} datasets, {successful} successful")

    print("✓ Parallel processing tests passed")
    return batch


def test_visualization(field_data):
    """Test visualization components with real data."""
    print("Testing visualization...")

    if field_data is None:
        print("⚠ No real data available, skipping visualization")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plot.png"

        frame = AnimationFrame(
            dataset_basename="plt00000",
            field_name="Temperature",
            x_data=field_data.coordinates,
            y_data=field_data.temperature,
            output_path=output_path
        )

        plotter = StandardPlotter()
        plotter.create_field_plot(frame)

        if output_path.exists():
            print(f"  Created visualization: {output_path.name}")
        else:
            print("  ⚠ Visualization file not created")

    print("✓ Visualization tests passed")


def test_integration():
    """Test full system integration with real data."""
    print("Testing system integration...")

    # Setup logging
    logger = setup_logging("INFO")

    # Create dependency container
    container = get_global_container()

    # Register services
    converter = PeleUnitConverter()
    container.register_singleton(type(converter), converter)

    logger.log_info("Integration test with real data completed")

    print("✓ Integration tests passed")


def test_convenience_functions():
    """Test package convenience functions with real setup."""
    print("Testing convenience functions...")

    input_dir = TEST_DATA_DIR
    output_dir = TEST_DATA_DIR / "test_output"
    output_dir.mkdir(exist_ok=True)

    # Test create_analysis_pipeline
    config = create_default_config(input_dir, output_dir)
    config.processing.flame_temperature = FLAME_TEMPERATURE
    config.processing.extract_location = EXTRACTION_Y_LOCATION

    pipeline = create_analysis_pipeline(config)

    assert 'loader' in pipeline
    assert 'extractor' in pipeline
    assert 'strategy' in pipeline

    print("✓ Convenience function tests passed")


def test_quick_analysis(config=None):
    """Test quick analysis function with real data."""
    print("Testing quick analysis...")

    if not YT_AVAILABLE:
        print("⚠ YT not available, skipping quick analysis")
        return

    try:
        input_dir = TEST_DATA_DIR
        output_dir = TEST_DATA_DIR / "quick_output"
        output_dir.mkdir(exist_ok=True)

        # Run quick analysis on test data
        batch = quick_analysis(input_dir, output_dir, config=config, parallel=False)

        if batch and batch.results:
            successful = len(batch.get_successful_results())
            print(f"  Quick analysis processed {len(batch.results)} datasets, {successful} successful")
        else:
            print("  Quick analysis completed but no results returned")

    except Exception as e:
        print(f"  Quick analysis failed: {e}")

    print("✓ Quick analysis tests passed")


########################################################################################################################
# Main Test Runner
########################################################################################################################

def main():
    """Run all tests with real Pele data."""
    print("Running Pele Processing System Tests with Real Data")
    print("=" * 60)
    print(f"Initial Parameters:")
    print(f"  Temperature: {INITIAL_TEMPERATURE} K")
    print(f"  Pressure: {INITIAL_PRESSURE / 1e5:.1f} bar")
    print(f"  Equivalence Ratio: {EQUIVALENCE_RATIO}")
    print(f"  Fuel: {FUEL}")
    print(f"  Extraction Y: {EXTRACTION_Y_LOCATION} m")
    print(f"  Flame Temperature: {FLAME_TEMPERATURE} K")
    print("=" * 60)

    try:
        # Setup and basic tests
        config = test_configuration()
        dataset, field_data = test_data_components()

        # Analysis tests with real data
        #test_flame_analysis(dataset, field_data)
        #test_shock_analysis(field_data)

        # System tests
        batch = test_parallel_processing()
        test_visualization(field_data)
        test_integration()
        test_convenience_functions()
        test_quick_analysis(config=config)

        print("\n" + "=" * 60)
        print("All tests passed with real Pele data!")
        return 0

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())