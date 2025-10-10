#!/usr/bin/env python3
"""
Batch YT Field Plotting with Flame Tracking (MPI-Enabled)

This script processes multiple datasets from a folder, tracks the flame position
in each one, and creates localized field plots around the tracked flame location.

Usage:
    Sequential: python main_surface_plotting.py
    MPI Parallel: mpiexec -n 4 python main_surface_plotting.py

The script will:
1. Find all plt* files in the data directory
2. For each dataset, detect the flame position (MPI: distributed across ranks)
3. Create localized contour plots for specified fields around the flame (OPTIMIZED: extract once)
4. Organize outputs as: output_dir/Field-Plots/FieldName/plt00100_localized_contour.png
5. Generate MP4 animations from frames (rank 0 only)
"""
import sys
from pathlib import Path
from typing import List, Dict

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# MPI detection and setup
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    comm = MPI.COMM_WORLD
    MPI_RANK = comm.Get_rank()
    MPI_SIZE = comm.Get_size()
    IS_MPI_RUN = MPI_SIZE > 1
except ImportError:
    MPI_AVAILABLE = False
    IS_MPI_RUN = False
    MPI_RANK = 0
    MPI_SIZE = 1
    comm = None

from base_processing.data.loaders import YTDataLoader
from base_processing.visualization import YTFieldPlotter
from base_processing import create_data_extractor, create_flame_analyzer, create_unit_converter, Direction, WaveType

# ============================================================================
# CONFIGURATION - Edit these paths and parameters
# ============================================================================

# Data directory containing plt files
DATA_DIR = Path('../../pele_data_2d')

# Output directory for results
OUTPUT_DIR = Path("../../Processed-MPI-Global-Results-V3.0.0")

# Localized plotting bounds (meters)
FORWARD_BOUND = 0.001   # 1mm forward of flame
BACKWARD_BOUND = 0.01   # 10mm backward of flame

# Additional x-offset from detected flame position (meters)
X_OFFSET = 0.0

# Configuration for tracking and offsets (from main.py)
EXTRACTION_LOCATION = (0.0462731 + (8.7e-5 / 2)) / 100  # meters - y-location for extraction
FLAME_TEMPERATURE = 2500.0  # K - temperature threshold for flame detection

# Thermodynamic extraction offsets (meters)
FLAME_THERMO_OFFSET = 10e-6    # 10 microns ahead of flame
BURNED_GAS_OFFSET = -10e-6     # 10 microns behind flame

# Fields to plot
FIELDS_TO_PLOT = [
    {'field': 'Temperature', 'colormap': 'plasma', 'log_scale': False},
    {'field': 'Pressure', 'colormap': 'plasma', 'log_scale': False},
    {'field': 'Y(H2)', 'colormap': 'plasma', 'log_scale': False},
]

# ============================================================================
# END CONFIGURATION
# ============================================================================


def get_flame_tracking_position(dataset, extraction_location: float, flame_temperature: float):
    """
    Extract flame position using flame analyzer.

    Args:
        dataset: YT dataset
        extraction_location: Y-location for extraction (meters)
        flame_temperature: Temperature threshold for flame detection (K)

    Returns:
        Flame position in cm (YT units) or None if detection fails
    """
    try:
        # Extract 1D field data along X at specific Y location
        unit_converter = create_unit_converter("pele")
        extractor = create_data_extractor("pele", unit_converter=unit_converter)
        field_data = extractor.extract_ray_data(dataset, extraction_location, Direction.X)

        # Find flame position
        flame_analyzer = create_flame_analyzer(
            flame_temperature=flame_temperature,
            transport_species='H2'
        )

        # Ensure flame_temperature is set (needed for find_wave_position)
        if not hasattr(flame_analyzer, 'flame_temperature'):
            flame_analyzer.flame_temperature = flame_temperature

        # Simple flame detection
        flame_idx, flame_pos = flame_analyzer.find_wave_position(field_data, WaveType.FLAME)

        print(f"    Detected flame at: {flame_pos:.6f} m (index: {flame_idx})")
        return flame_pos

    except Exception as e:
        print(f"    Warning: Could not detect flame position: {e}")
        return None


def process_single_dataset(
    dataset,
    plt_file: Path,
    output_base: Path,
    fields_to_plot: List[Dict],
    extraction_location: float,
    flame_temperature: float,
    forward_bound: float,
    backward_bound: float,
    x_offset: float = 0.0
):
    """
    Process a single dataset - detect flame and create localized plots.

    Args:
        dataset: YT dataset
        plt_file: Path to plt file (for naming)
        output_base: Base output directory
        fields_to_plot: List of field configurations
        extraction_location: Y-location for extraction (meters)
        flame_temperature: Temperature threshold for flame detection (K)
        forward_bound: Distance forward of flame (meters)
        backward_bound: Distance backward of flame (meters)
        x_offset: Additional x-offset from detected flame position (meters)
    """
    print(f"\n  Processing: {plt_file.name}")
    print(f"    Time: {dataset.current_time}")

    # Get flame tracking position
    flame_x_m = get_flame_tracking_position(dataset, extraction_location, flame_temperature)

    if flame_x_m is None:
        print(f"    Skipping {plt_file.name} - flame detection failed")
        return

    # Convert to cm for YT
    flame_x_cm = flame_x_m * 100

    # Get domain bounds
    left_edge = dataset.domain_left_edge.to_ndarray()
    right_edge = dataset.domain_right_edge.to_ndarray()

    # Create center point at flame position and extraction location
    center_point = [
        flame_x_cm,  # x at flame position (in cm for YT)
        extraction_location * 100,  # y at extraction location (convert m to cm)
        (left_edge[2] + right_edge[2]) / 2  # z at domain center
    ]

    # Convert bounds from meters to cm for YT
    forward_bound_cm = forward_bound * 100
    backward_bound_cm = backward_bound * 100

    print(f"    Tracking position:")
    print(f"      Flame x: {flame_x_cm:.6f} cm ({flame_x_m:.6f} m)")
    print(f"      Extraction y: {extraction_location:.6f} m")
    print(f"      Forward bound: {forward_bound:.6f} m")
    print(f"      Backward bound: {backward_bound:.6f} m")

    # Create plotter
    plotter = YTFieldPlotter(figure_size=(10, 8), dpi=150)

    # Output directory for field plots
    field_plots_dir = output_base / "Field-Plots"

    # OPTIMIZED: Extract localized region ONCE for all fields
    print(f"    Creating {len(fields_to_plot)} field plots (optimized - single extraction)...")
    try:
        plotter.plot_multiple_localized_contours(
            dataset=dataset,
            fields=fields_to_plot,  # Pass all fields at once
            center_point=center_point,
            forward_bound=forward_bound_cm,
            backward_bound=backward_bound_cm,
            output_path=field_plots_dir,
            axis='z',  # Plot x-y plane
            normal_axis='x',  # Bounds along x-axis
            auto_organize=True
        )
    except Exception as e:
        print(f"      Error in optimized plotting: {e}")
        # Fallback to individual plotting if needed
        print(f"      Falling back to individual field plotting...")
        for field_config in fields_to_plot:
            field_name = field_config['field']
            try:
                plotter.plot_localized_contour(
                    dataset=dataset,
                    field=field_name,
                    center_point=center_point,
                    forward_bound=forward_bound_cm,
                    backward_bound=backward_bound_cm,
                    output_path=field_plots_dir,
                    axis='z',
                    normal_axis='x',
                    levels=50,
                    colormap=field_config['colormap'],
                    log_scale=field_config['log_scale']
                )
                print(f"      ✓ {field_name}")
            except Exception as e:
                print(f"      ✗ {field_name}: {e}")


def generate_mp4_animations(output_dir: Path, fields_to_plot: List[Dict]) -> None:
    """Generate MP4 animations from saved PNG frames for each field."""
    print(f"\n{'='*80}")
    print("GENERATING MP4 ANIMATIONS FROM FRAMES")
    print(f"{'='*80}")

    try:
        from base_processing.visualization.animators import FrameAnimator

        animator = FrameAnimator()

        # Create output directory for videos
        field_plots_dir = output_dir / 'Field-Plots'
        video_output_dir = output_dir / 'Animation-Videos'
        video_output_dir.mkdir(exist_ok=True)

        # Process each field
        for field_config in fields_to_plot:
            field_name = field_config['field']

            # Get pretty field name for folder
            from base_processing.visualization.yt_field_plotter import get_pretty_field_name, get_pele_field_name
            pele_field = get_pele_field_name(field_name)
            pretty_field = get_pretty_field_name(pele_field)

            frame_dir = field_plots_dir / pretty_field

            if not frame_dir.exists():
                print(f"  Skipping {field_name} - no frames directory found")
                continue

            # Check if directory has PNG files
            png_files = list(frame_dir.glob('*.png'))
            if not png_files:
                print(f"  Skipping {field_name} - no frames found")
                continue

            print(f"  Processing {field_name}: {len(png_files)} frames")

            # Generate MP4 animation
            output_file = video_output_dir / f"{pretty_field}_animation.mp4"

            try:
                print(f"    Creating MP4 animation...")
                animator.create_animation(
                    frame_directory=frame_dir,
                    output_path=output_file,
                    frame_rate=30.0,
                    format='mp4'
                )
                print(f"    [SUCCESS] Saved to: {output_file.name}")

            except Exception as e:
                print(f"    [FAILED] Failed to create MP4: {e}")

        print(f"\nAll animations saved to: {video_output_dir}")

    except Exception as e:
        print(f"Error generating animations: {e}")


def main():
    """Main function for batch processing with MPI support."""
    # Use configuration from top of file
    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR

    # Only rank 0 prints header and finds files initially
    if MPI_RANK == 0:
        print("="*80)
        print("Batch YT Field Plotting with Flame Tracking")
        if IS_MPI_RUN:
            print(f"MPI Mode: {MPI_SIZE} processes")
        else:
            print("Sequential Mode")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Data directory: {data_dir.absolute()}")
        print(f"  Output directory: {output_dir.absolute()}")
        print(f"  Extraction location: {EXTRACTION_LOCATION:.6f} m")
        print(f"  Flame temperature threshold: {FLAME_TEMPERATURE:.1f} K")
        print(f"  Forward bound: {FORWARD_BOUND:.6f} m")
        print(f"  Backward bound: {BACKWARD_BOUND:.6f} m")
        print(f"  X-offset: {X_OFFSET:.6f} m")
        print(f"\nFields to plot:")
        for field_config in FIELDS_TO_PLOT:
            print(f"  - {field_config['field']}")

        # Find all plt files
        plt_files = sorted(data_dir.glob("plt*"))

        if not plt_files:
            print(f"Error: No plt files found in {data_dir}")
            if IS_MPI_RUN:
                comm.Abort(1)
            else:
                sys.exit(1)

        print(f"  Number of datasets: {len(plt_files)}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        plt_files = None

    # Broadcast dataset paths to all ranks
    if IS_MPI_RUN:
        plt_files = comm.bcast(plt_files, root=0)
        comm.barrier()  # Synchronize before starting

    # ========================================================================
    # PARALLEL PHASE: Each MPI rank processes a subset of datasets
    # ========================================================================

    if MPI_RANK == 0:
        print(f"\n{'='*80}")
        if IS_MPI_RUN:
            print(f"PARALLEL PROCESSING: Distributing {len(plt_files)} datasets across {MPI_SIZE} ranks")
        else:
            print("SEQUENTIAL PROCESSING")
        print("="*80)

    # Create YT data loader
    loader = YTDataLoader()

    # Distribute datasets across ranks (round-robin)
    my_datasets = []
    for i, plt_file in enumerate(plt_files):
        if i % MPI_SIZE == MPI_RANK:
            my_datasets.append(plt_file)

    if IS_MPI_RUN:
        print(f"Rank {MPI_RANK}: Processing {len(my_datasets)} datasets")

    # Process assigned datasets
    success_count = 0
    fail_count = 0

    for plt_file in my_datasets:
        try:
            if IS_MPI_RUN:
                print(f"\nRank {MPI_RANK}: Processing {plt_file.name}")

            # Load dataset
            dataset = loader.load_dataset(plt_file)

            # Process dataset (creates PNG frames with optimized extraction)
            process_single_dataset(
                dataset=dataset,
                plt_file=plt_file,
                output_base=output_dir,
                fields_to_plot=FIELDS_TO_PLOT,
                extraction_location=EXTRACTION_LOCATION,
                flame_temperature=FLAME_TEMPERATURE,
                forward_bound=FORWARD_BOUND,
                backward_bound=BACKWARD_BOUND,
                x_offset=X_OFFSET
            )

            success_count += 1

        except Exception as e:
            if IS_MPI_RUN:
                print(f"\nRank {MPI_RANK}: Error processing {plt_file.name}: {e}")
            else:
                print(f"\n  Error processing {plt_file.name}: {e}")
            fail_count += 1

    # Synchronize all ranks before gathering results
    if IS_MPI_RUN:
        comm.barrier()

        # Gather statistics from all ranks
        all_success = comm.gather(success_count, root=0)
        all_fail = comm.gather(fail_count, root=0)

        if MPI_RANK == 0:
            total_success = sum(all_success)
            total_fail = sum(all_fail)
    else:
        total_success = success_count
        total_fail = fail_count

    # ========================================================================
    # SEQUENTIAL PHASE: Rank 0 generates MP4 animations from PNG frames
    # ========================================================================

    if MPI_RANK == 0:
        # Summary
        print(f"\n{'='*80}")
        print("Figure Generation Complete")
        print("="*80)
        print(f"  Successfully processed: {total_success}/{len(plt_files)} datasets")
        print(f"  Failed: {total_fail}/{len(plt_files)} datasets")
        print(f"\nOutput saved to: {output_dir.absolute()}")
        print(f"Field plots organized in: {(output_dir / 'Field-Plots').absolute()}")
        print("="*80)

        # Generate MP4 animations from frames (sequential, rank 0 only)
        generate_mp4_animations(output_dir, FIELDS_TO_PLOT)

        print(f"\n{'='*80}")
        print("All processing complete!")
        print("="*80)


if __name__ == "__main__":
    main()
