#!/usr/bin/env python3
"""
Test script for the Pele processing system.
Tests the restructured modular architecture against sample data.
"""
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

# Add src directory to path for imports
from ..src.pele_processing.core import Container, get_global_container
from ..src.pele_processing.config import create_default_config
from ..src.pele_processing.data import create_data_loader, create_data_extractor
from ..src.pele_processing.analysis import create_flame_analyzer, create_thermodynamic_calculator
from ..src.pele_processing.parallel import create_processing_strategy
from ..src.pele_processing.visualization import StandardPlotter
from ..src.pele_processing.utils import setup_logging, PeleUnitConverter


class MockDataset:
    """Mock YT dataset for testing."""

    def __init__(self):
        self.basename = "plt00100"
        self.filename = f"/test/data/{self.basename}"
        self.current_time = MockTime(1e-6)
        self.domain_left_edge = [MockValue(0), MockValue(0), MockValue(0)]
        self.domain_right_edge = [MockValue(0.1), MockValue(0.05), MockValue(0.001)]
        self.index = MockIndex()
        self.field_list = [
            ("boxlib", "x"), ("boxlib", "y"), ("boxlib", "Temp"),
            ("boxlib", "pressure"), ("boxlib", "density"),
            ("boxlib", "x_velocity"), ("boxlib", "Y(H2)"), ("boxlib", "Y(HO2)")
        ]

    def ortho_ray(self, axis, coord):
        return MockRay()

    def all_data(self):
        return MockAllData()

    def force_periodicity(self):
        pass


class MockTime:
    def __init__(self, value):
        self.value = value

    def to_value(self):
        return self.value


class MockValue:
    def __init__(self, value):
        self.value = value

    def to_value(self):
        return self.value


class MockIndex:
    def __init__(self):
        self.max_level = 2
        self.grids = [MockGrid()]

    def get_smallest_dx(self):
        return MockValue(1e-5)


class MockGrid:
    def __init__(self):
        self.Level = 2
        self.dds = [MockValue(1e-5), MockValue(1e-5), MockValue(1e-6)]

        # Generate test data
        n = 100
        x = np.linspace(0, 10, n)  # cm
        self.data = {
            ("boxlib", "x"): MockFieldData(x),
            ("boxlib", "y"): MockFieldData(np.full(n, 2.5)),
            ("boxlib", "Temp"): MockFieldData(300 + 2200 * np.exp(-(x - 5) ** 2)),  # Gaussian flame
            ("boxlib", "pressure"): MockFieldData(np.full(n, 1e6)),
            ("boxlib", "density"): MockFieldData(np.full(n, 1.0)),
            ("boxlib", "x_velocity"): MockFieldData(np.full(n, 100.0)),
            ("boxlib", "Y(H2)"): MockFieldData(np.maximum(0, 0.1 - 0.1 * np.exp(-(x - 5) ** 2))),
            ("boxlib", "Y(HO2)"): MockFieldData(0.01 * np.exp(-(x - 5) ** 2)),
        }

    def __getitem__(self, key):
        return self.data[key]


class MockFieldData:
    def __init__(self, data):
        self._data = np.array(data)

    def to_value(self):
        return self._data

    def flatten(self):
        return self._data


class MockRay:
    def __init__(self):
        n = 100
        x = np.linspace(0, 10, n)
        self.data = {
            ("boxlib", "x"): MockFieldData(x),
            ("boxlib", "Temp"): MockFieldData(300 + 2200 * np.exp(-(x - 5) ** 2)),
            ("boxlib", "pressure"): MockFieldData(np.full(n, 1e6)),
            ("boxlib", "x_velocity"): MockFieldData(np.full(n, 100.0)),
            ("boxlib", "Y(H2)"): MockFieldData(np.maximum(0, 0.1 - 0.1 * np.exp(-(x - 5) ** 2))),
            ("boxlib", "Y(HO2)"): MockFieldData(0.01 * np.exp(-(x - 5) ** 2)),
        }

    def __getitem__(self, key):
        return self.data[key]


class MockAllData:
    def extract_isocontours(self, field, value):
        # Return mock contour points (cm)
        return np.array([[4.5, 2.0], [5.0, 2.5], [5.5, 2.0]])


def test_configuration():
    """Test configuration system."""
    print("Testing configuration...")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = create_default_config(
            input_dir=tmpdir,
            output_dir=tmpdir
        )

        # Test configuration validation
        config.processing.flame_temperature = 2500.0
        config.processing.analyze_flame_position = True

        assert config.processing.flame_temperature == 2500.0
        assert config.processing.analyze_flame_position == True

    print("✓ Configuration tests passed")


def test_data_components():
    """Test data loading and extraction."""
    print("Testing data components...")

    # Create mock dataset
    mock_dataset = MockDataset()

    # Test unit converter
    converter = PeleUnitConverter()
    temp_si = converter.convert_from_cgs(300, 'Temperature')
    assert temp_si == 300  # Temperature unchanged

    # Test data extractor
    extractor = create_data_extractor("pele", unit_converter=converter)

    # Mock the extraction (would normally extract from real YT dataset)
    field_data = extractor.extract_ray_data(mock_dataset, 0.025)

    assert len(field_data.coordinates) == 100
    assert np.max(field_data.temperature) > 2000  # Has flame

    print("✓ Data component tests passed")


def test_flame_analysis():
    """Test flame analysis."""
    print("Testing flame analysis...")

    # Create test data
    n = 100
    x = np.linspace(0, 0.1, n)  # meters
    temp = 300 + 2200 * np.exp(-((x - 0.05) * 1000) ** 2)  # Flame at 5cm
    ho2 = 0.01 * np.exp(-((x - 0.05) * 1000) ** 2)

    from ..src.pele_processing.core.domain import FieldData, SpeciesData
    species_data = SpeciesData()
    species_data.mass_fractions['HO2'] = ho2

    field_data = FieldData(
        coordinates=x,
        temperature=temp,
        pressure=np.full(n, 1e6),
        density=np.full(n, 1.0),
        velocity_x=np.full(n, 100.0),
        species_data=species_data
    )

    # Test flame analyzer
    flame_analyzer = create_flame_analyzer(flame_temperature=2000.0)

    from ..src.pele_processing.core.domain import WaveType
    flame_idx, flame_pos = flame_analyzer.find_wave_position(field_data, WaveType.FLAME)

    # Should find flame near center (0.05m)
    assert abs(flame_pos - 0.05) < 0.01

    print("✓ Flame analysis tests passed")


def test_parallel_processing():
    """Test parallel processing strategy."""
    print("Testing parallel processing...")

    # Test sequential strategy
    strategy = create_processing_strategy("sequential")

    def mock_processor(dataset_path):
        from ..src.pele_processing.core.domain import ProcessingResult, DatasetInfo
        return ProcessingResult(
            dataset_info=DatasetInfo.from_path(dataset_path),
            success=True
        )

    # Process mock datasets
    mock_paths = ["/data/plt00100", "/data/plt00200"]
    batch = strategy.execute(mock_paths, mock_processor)

    assert len(batch.results) == 2
    assert all(r.success for r in batch.results)

    print("✓ Parallel processing tests passed")


def test_visualization():
    """Test visualization components."""
    print("Testing visualization...")

    # Create test data
    x_data = np.linspace(0, 0.1, 100)
    y_data = 300 + 2200 * np.exp(-((x_data - 0.05) * 1000) ** 2)

    from ..src.pele_processing.core.domain import AnimationFrame

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plot.png"

        frame = AnimationFrame(
            dataset_basename="plt00100",
            field_name="Temperature",
            x_data=x_data,
            y_data=y_data,
            output_path=output_path,
            flame_position=0.05
        )

        plotter = StandardPlotter()
        plotter.create_field_plot(frame)

        assert output_path.exists()

    print("✓ Visualization tests passed")


def test_integration():
    """Test full system integration."""
    print("Testing system integration...")

    # Setup logging
    logger = setup_logging("INFO")

    # Create dependency container
    container = get_global_container()

    # Register services
    converter = PeleUnitConverter()
    container.register_singleton(type(converter), converter)

    logger.log_info("Integration test completed")

    print("✓ Integration tests passed")


def main():
    """Run all tests."""
    print("Running Pele Processing System Tests")
    print("=" * 50)

    try:
        test_configuration()
        test_data_components()
        test_flame_analysis()
        test_parallel_processing()
        test_visualization()
        test_integration()

        print("\n" + "=" * 50)
        print(" All tests passed!")
        return 0

    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())