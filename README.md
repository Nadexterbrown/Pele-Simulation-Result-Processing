# Pele Simulation Result Processing System

A comprehensive parallel processing system for analyzing Pele combustion simulation data with support for flame analysis, shock detection, thermodynamic calculations, and advanced visualization.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/Development%20Status-Beta-yellow.svg)](https://pypi.org/classifiers/)

## Overview

This system provides modular, scalable analysis tools for Pele combustion simulations, featuring:

- **Flame Analysis**: Position tracking, thickness calculation, consumption rates, and burning velocity
- **Shock Wave Detection**: Automated shock front identification and Rankine-Hugoniot validation
- **Burned Gas Analysis**: Thermodynamic property extraction behind flame fronts
- **Parallel Processing**: MPI-based parallel execution for large datasets
- **Advanced Visualization**: Animation generation, specialized plotting, and data export
- **Modular Architecture**: Clean separation of concerns with dependency injection

## Key Features

### 🔥 Combustion Analysis
- **Flame tracking**: Automatic flame front detection using temperature gradients
- **Flame thickness**: Temperature gradient method for accurate thickness measurements
- **Burning velocity**: Species consumption-based burning velocity calculations
- **Heat release analysis**: Integrated heat release rate calculations

### 💥 Shock Wave Analysis  
- **Shock detection**: Pressure jump-based shock front identification
- **Rankine-Hugoniot relations**: Thermodynamic validation of shock properties
- **Pre/post-shock states**: Complete thermodynamic state extraction
- **Wave velocity calculations**: Time-series analysis for propagation speeds

### 🧪 Thermodynamic Calculations
- **Cantera integration**: Real gas thermodynamic property calculations
- **Chemical mechanisms**: Support for detailed reaction mechanisms
- **Transport properties**: Viscosity, conductivity, and diffusion coefficients
- **State validation**: Physical consistency checking

### ⚡ Parallel Processing
- **MPI support**: Efficient parallel processing across multiple nodes
- **Load balancing**: Adaptive work distribution strategies
- **Fault tolerance**: Graceful handling of failed processes
- **Scalability**: Linear scaling to hundreds of cores

### 📊 Visualization & Export
- **Animation generation**: Automated frame creation and animation assembly
- **Multi-format export**: CSV, JSON, and structured data output
- **Specialized plots**: Schlieren, streamlines, and contour visualizations
- **Interactive analysis**: Configurable plotting and analysis parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- MPI implementation (OpenMPI, MPICH, or Intel MPI) for parallel processing
- Conda environment (recommended)

### Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd pele-processing

# Create conda environment
conda create -n pele-processing python=3.9
conda activate pele-processing

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

#### Core Requirements
- `numpy>=1.20.0` - Numerical computing
- `scipy>=1.7.0` - Scientific algorithms  
- `matplotlib>=3.5.0` - Plotting and visualization
- `pandas>=1.3.0` - Data manipulation
- `pyyaml>=6.0` - Configuration files

#### Scientific Computing
- `yt>=4.0.0` - AMReX/Chombo data analysis
- `h5py>=3.0.0` - HDF5 file support
- `netcdf4>=1.5.0` - NetCDF file support

#### Parallel Processing
- `mpi4py>=3.0.0` - MPI Python bindings

#### Thermodynamics
- `cantera>=2.6.0` - Chemical kinetics and thermodynamics

#### Visualization
- `pillow>=8.0.0` - Image processing
- `imageio>=2.9.0` - Animation creation
- `ffmpeg-python>=0.2.0` - Video encoding

## Usage

### Quick Start

```python
from pele_processing import quick_analysis

# Analyze all plotfiles in a directory
results = quick_analysis(
    input_dir="path/to/plotfiles",
    output_dir="path/to/results",
    parallel=True
)
```

### Main Processing Script

The main processing script provides comprehensive analysis with full configurability:

```bash
# Serial execution
python scripts/main_processing_script.py

# Parallel execution (MPI)
mpiexec -n 4 python scripts/main_processing_script.py

# With custom parameters
python scripts/main_processing_script.py --extraction-y 0.0005 --flame-temp 2000 --no-animations
```

#### Command Line Arguments
- `--extraction-y`: Y-location for 1D ray extraction (default: 0.000446 m)
- `--flame-temp`: Flame detection temperature threshold (default: 2500 K)
- `--shock-ratio`: Shock detection pressure ratio (default: 1.01)
- `--no-animations`: Disable animation generation for faster processing

### Configuration

The system uses a hierarchical configuration system:

```python
from pele_processing import ProcessingConfig

config = ProcessingConfig()
config.extraction_location = 0.0005  # meters
config.flame_temperature = 2500.0    # Kelvin
config.shock_pressure_ratio = 1.01
config.transport_species = 'H2'
config.chemical_mechanism = 'path/to/mechanism.yaml'
```

### Advanced Usage

#### Custom Analysis Pipeline

```python
from pele_processing import (
    create_data_loader, create_data_extractor, 
    create_flame_analyzer, create_shock_analyzer, create_burned_gas_analyzer
)

# Create components
loader = create_data_loader("yt")
extractor = create_data_extractor("pele")
flame_analyzer = create_flame_analyzer(flame_temperature=2500)
shock_analyzer = create_shock_analyzer(pressure_ratio_threshold=1.01)
burned_gas_analyzer = create_burned_gas_analyzer(offset=-10e-6)

# Process dataset
dataset = loader.load_dataset("plt00000")
field_data = extractor.extract_ray_data(dataset, 0.0005)

# Perform analysis
flame_props = flame_analyzer.analyze_flame_properties(dataset, field_data)
shock_props = shock_analyzer.analyze_shock_properties(field_data)
burned_gas_props = burned_gas_analyzer.analyze_burned_gas_properties(
    field_data, flame_props.position, flame_props.index
)
```

#### Parallel Processing Setup

```python
from pele_processing import create_processing_strategy

# Create MPI strategy for parallel execution
strategy = create_processing_strategy("mpi", logger=logger)

# Process datasets in parallel
results = strategy.execute(dataset_paths, process_single_dataset)
```

## Architecture

### Core Components

The system follows a clean architecture with clear separation of concerns:

```
src/pele_processing/
├── core/           # Domain models, interfaces, exceptions
├── data/           # Data loading and extraction
├── analysis/       # Scientific analysis algorithms
├── visualization/  # Plotting and animation
├── parallel/       # MPI and parallel processing
├── utils/          # Utilities and constants
└── config/         # Configuration management
```

### Design Principles

- **Dependency Injection**: Modular components with clear interfaces
- **Domain-Driven Design**: Rich domain models representing physical concepts
- **SOLID Principles**: Single responsibility, open/closed, interface segregation
- **Error Handling**: Custom exceptions with detailed error context
- **Type Safety**: Comprehensive type hints and validation

### Analyzers

The system provides specialized analyzers for different physical phenomena:

- **FlameAnalyzer**: Comprehensive flame front analysis
- **ShockAnalyzer**: Shock wave detection and characterization  
- **BurnedGasAnalyzer**: Post-flame thermodynamic property extraction
- **ThermodynamicCalculator**: Real gas property calculations
- **GeometryAnalyzer**: Flame surface and geometric analysis

## Output

### Data Files

The system generates structured output files:

- **Combined Results**: `processing_results_combined.csv` - All time points in single file
- **Summary Statistics**: Mean, std dev, min/max for all quantities
- **Metadata**: Processing configuration and system information

### Visualizations

- **Temperature Animations**: Global and local temperature evolution
- **Pressure Animations**: Pressure field visualization  
- **Velocity Animations**: Flow field analysis
- **Heat Release Animations**: Reaction zone visualization
- **Flame Thickness Plots**: Detailed flame structure analysis

### Calculated Properties

#### Flame Properties
- Position and velocity
- Temperature, pressure, density
- Flame thickness and surface area
- Heat release rate and consumption rate
- Burning velocity and Reynolds number

#### Shock Properties  
- Position and velocity
- Pre/post-shock thermodynamic states
- Pressure, density, temperature ratios
- Mach number and shock strength

#### Burned Gas Properties
- Thermodynamic state behind flame
- Gas velocity and flow properties
- Species composition (if available)

## Performance

### Scaling

The system demonstrates excellent parallel scaling:
- **Linear scaling** up to 100+ cores for large datasets
- **Memory efficient** with streaming data processing
- **Load balanced** adaptive work distribution

### Optimization Features

- **Cached data loading** to avoid redundant I/O
- **Vectorized computations** using NumPy
- **Minimal memory footprint** with lazy evaluation
- **Efficient MPI communication** with optimized data structures

## Development

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd pele-processing

# Create development environment
conda env create -f environment-dev.yml
conda activate pele-processing-dev

# Install in development mode with all dependencies
pip install -e .[all,dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pele_processing --cov-report=html

# Run MPI tests (requires MPI)
mpiexec -n 2 pytest tests/test_mpi/

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run integration tests only
```

### Code Quality

The project maintains high code quality standards:

- **Black**: Code formatting (`black .`)
- **isort**: Import sorting (`isort .`)  
- **flake8**: Linting (`flake8 src/`)
- **mypy**: Type checking (`mypy src/`)
- **pytest**: Comprehensive test suite

## Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow **PEP 8** style guidelines
- Write **comprehensive tests** for new features
- Update **documentation** for API changes
- Use **type hints** for all public interfaces
- Follow **conventional commits** for commit messages

## Troubleshooting

### Common Issues

#### MPI Import Error
```bash
# Error: No module named 'mpi4py'
conda install mpi4py
# or
pip install mpi4py
```

#### YT Data Loading Issues  
```bash
# Error: Cannot load plotfile
# Ensure YT is properly installed and plotfiles are valid AMReX format
conda install yt
```

#### Cantera Thermodynamic Errors
```bash
# Error: Cantera mechanism file not found  
# Ensure chemical mechanism file path is correct
# Download mechanisms from: https://github.com/Cantera/cantera/tree/main/data
```

#### Unicode/Encoding Issues
```bash
# Error: 'charmap' codec can't encode character
# Set environment variables:
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
```

### Performance Issues

- **Large datasets**: Use MPI parallel processing
- **Memory limitations**: Reduce cache sizes in configuration
- **Slow visualization**: Disable animations with `--no-animations`
- **Network file systems**: Copy data locally before processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Pele Development Team** - Core simulation framework
- **YT Project** - Scientific data analysis tools
- **Cantera** - Chemical kinetics library
- **AMReX** - Adaptive mesh refinement framework

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pele_processing,
  title = {Pele Simulation Result Processing System},
  author = {Nolan Dexter-Brown},
  year = {2024},
  url = {https://github.com/pele-combustion/pele-processing},
  version = {3.0.0}
}
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/pele-combustion/pele-processing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pele-combustion/pele-processing/discussions)
- **Email**: support@pele-processing.org

---

For more detailed documentation, visit our [full documentation](https://pele-processing.readthedocs.io).
