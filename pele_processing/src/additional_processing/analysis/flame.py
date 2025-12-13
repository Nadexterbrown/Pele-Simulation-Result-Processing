from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import logging
import tempfile
import os
from base_processing.analysis.flame import PeleFlameAnalyzer as BaseFlameAnalyzer
from additional_processing.core.domain import CanteraFreeFlameProperties
from .flame_baseline import FlameBaselineFitter, FlameBaselineResult

class CanteraLaminarFlame:
    """Calculate free flame properties using Cantera."""

    def __init__(self, mechanism_file: str = 'h2o2.yaml', restart_file: Optional[str] = None,
                 mpi_rank: Optional[int] = None):
        """
        Initialize Cantera free flame calculator.

        Args:
            mechanism_file: Path to Cantera mechanism file (e.g., 'h2o2.yaml', 'gri30.yaml')
            restart_file: Path to YAML file for saving/loading flame solutions (default: auto-generated)
            mpi_rank: MPI rank for per-process restart files. If None, will auto-detect MPI or use 0.
        """
        try:
            import cantera as ct
            self.ct = ct
            self.mechanism_file = mechanism_file
            self.gas = None
            self.flame = None

            # Auto-detect MPI rank if not provided
            if mpi_rank is None:
                mpi_rank = self._get_mpi_rank()
            self.mpi_rank = mpi_rank

            # Setup restart file path - use YAML format for Cantera 3.0+
            # Include MPI rank in filename to avoid conflicts between processes
            if restart_file is None:
                # Use a temporary file in the current working directory with rank suffix
                self.restart_file = os.path.join(
                    os.getcwd(),
                    f'cantera_flame_restart_rank{self.mpi_rank}.yaml'
                )
            else:
                # If user provided a restart file, add rank suffix before extension
                base, ext = os.path.splitext(restart_file)
                self.restart_file = f"{base}_rank{self.mpi_rank}{ext}"

            self.has_saved_solution = False
            self.last_successful_T = None
            self.last_successful_P = None

        except ImportError:
            raise ImportError("Cantera is required for CanteraLaminarFlame but is not installed.")

    @staticmethod
    def _get_mpi_rank() -> int:
        """Auto-detect MPI rank, return 0 if MPI not available or not running."""
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            return comm.Get_rank()
        except (ImportError, RuntimeError):
            return 0

    def setup(self, T: float, P: float, composition: Dict[str, float]) -> None:
        """
        Set initial conditions for the flame calculation.

        Args:
            T: Temperature in Kelvin
            P: Pressure in Pascals
            composition: Dict of species mole fractions (X_i)
        """
        # Create gas object
        self.gas = self.ct.Solution(self.mechanism_file)

        # Set thermodynamic state using mole fractions
        self.gas.TPX = T, P, composition

        # Store initial conditions
        self.initial_T = T
        self.initial_P = P
        self.initial_composition = composition

    def save_solution(self) -> None:
        """Save the current flame solution to YAML file for restart."""
        if self.flame is not None:
            try:
                # Delete existing file first to avoid Cantera's "bad conversion" error
                # when overwriting files with potentially incompatible format
                if os.path.exists(self.restart_file):
                    try:
                        os.remove(self.restart_file)
                        logging.debug(f"Removed existing restart file: {self.restart_file}")
                    except Exception as e:
                        logging.warning(f"Could not remove existing restart file: {e}")

                # Save to fresh file (no overwrite needed since file was deleted)
                self.flame.save(self.restart_file, name='solution',
                               description=f'T={self.initial_T:.1f}K, P={self.initial_P/1e5:.1f}bar')
                self.has_saved_solution = True
                self.last_successful_T = self.initial_T
                self.last_successful_P = self.initial_P
                logging.debug(f"Saved flame solution to {self.restart_file}")
            except Exception as e:
                logging.warning(f"Failed to save flame solution: {e}")

    def load_solution(self) -> bool:
        """
        Load a previously saved flame solution.

        Args:
            use_as_initial_guess: If True, use loaded solution as initial guess for new calculation

        Returns:
            True if solution was loaded successfully, False otherwise
        """
        if not self.has_saved_solution or not os.path.exists(self.restart_file):
            return False

        try:
            # Restore the flame from saved solution
            self.flame.restore(self.restart_file, name='solution', loglevel=1)

            self.flame.inlet.T = self.initial_T
            self.flame.inlet.X = self.initial_composition
            self.flame.P = self.initial_P

            # Set max grid points for better resolution if needed
            self.flame.set_max_grid_points(self.flame.flame, 2000)

            logging.debug(f"Loaded flame solution from {self.restart_file} and updated BCs: "
            f"T={self.initial_T:.1f}K, P={self.initial_P / 1e5:.1f}bar")

            logging.debug(f"Loaded flame solution from {self.restart_file}")
            return True
        except Exception as e:
            logging.warning(f"Failed to load flame solution: {e}")
            return False

    def run(self, domain_width: float = 1e-5, refine_criteria: Optional[Dict] = None,
            use_restart: bool = True, max_retries: int = 2) -> 'ct.FreeFlame':
        """
        Calculate 1D freely propagating flame with restart capability.

        Args:
            domain_width: Width of computational domain in meters
            refine_criteria: Optional refinement criteria dict
            use_restart: If True, use saved solution from previous timestep if available
            max_retries: Maximum number of retry attempts with restart

        Returns:
            Cantera FreeFlame object with solution
        """
        if self.gas is None:
            raise ValueError("Must call setup() first")

        for attempt in range(max_retries):
            try:
                # Create flame object
                if attempt == 0 or self.flame is None:
                    self.flame = self.ct.FreeFlame(self.gas, width=domain_width)

                # Only use restart on retry attempts (after first failure)
                restart_used = False
                if use_restart and attempt > 0 and self.has_saved_solution:
                    # Always use saved solution on retry - don't check compatibility
                    if self.load_solution():
                        restart_used = True
                        logging.info(f"Using restart from T={self.last_successful_T:.1f}K, "
                                       f"P={self.last_successful_P/1e5:.1f}bar for current "
                                       f"T={self.initial_T:.1f}K, P={self.initial_P/1e5:.1f}bar")

                if not restart_used:
                    # Set initial guess normally
                    self.flame.set_initial_guess()

                # Set refinement criteria
                if refine_criteria:
                    self.flame.set_refine_criteria(**refine_criteria)
                else:
                    # Adjust criteria based on pressure
                    P_bar = self.initial_P / 1e5
                    if P_bar > 100:  # High pressure conditions
                        self.flame.set_refine_criteria(ratio=4, slope=0.01, curve=0.01, prune=0.0)
                    else:
                        self.flame.set_refine_criteria(ratio=3, slope=0.02, curve=0.02)

                # Solve with auto-refinement
                if not restart_used:
                    self.flame.solve(loglevel=0, refine_grid=True, auto=True)
                else:
                    self.flame.solve(loglevel=0, refine_grid=True, auto=False) 

                # Save successful solution
                self.save_solution()

                return self.flame

            except Exception as e:
                error_msg = str(e)
                if "monotonically increasing" in error_msg:
                    logging.warning(f"Grid error on attempt {attempt+1}/{max_retries}: {error_msg}")
                    if attempt < max_retries - 1:
                        # Will retry with restart on next iteration
                        continue
                    else:
                        # Last attempt failed
                        if self.has_saved_solution:
                            logging.error(f"Failed after {max_retries} attempts. Last successful: "
                                        f"T={self.last_successful_T:.1f}K, P={self.last_successful_P/1e5:.1f}bar")
                        raise
                else:
                    # Other error types, raise immediately
                    raise

    def _check_restart_compatibility(self) -> bool:
        """
        Check if saved solution is compatible with current conditions.

        Returns:
            True if restart file can be used for current conditions
        """
        if not self.has_saved_solution:
            return False

        # Check if conditions are reasonably close (within 20% for T, 50% for P)
        if self.last_successful_T is not None and self.last_successful_P is not None:
            T_ratio = abs(self.initial_T - self.last_successful_T) / self.last_successful_T
            P_ratio = abs(self.initial_P - self.last_successful_P) / self.last_successful_P

            # More lenient for restart attempts
            return T_ratio < 0.2 and P_ratio < 0.5

        return False

    def cleanup(self) -> None:
        """Remove the restart file if it exists."""
        if os.path.exists(self.restart_file):
            try:
                os.remove(self.restart_file)
                logging.debug(f"Removed restart file: {self.restart_file}")
            except Exception as e:
                logging.warning(f"Failed to remove restart file: {e}")

    def get_flame_properties(self) -> CanteraFreeFlameProperties:
        """
        Extract flame properties from Cantera solution.

        Returns:
            CanteraFreeFlameProperties object with Cantera results
        """
        if self.flame is None:
            raise ValueError("Must call run() first")

        # Find flame position (max heat release rate)
        hrr = self.flame.heat_release_rate

        # Calculate flame thickness (thermal thickness definition)
        T_max = np.max(self.flame.T)
        T_min = np.min(self.flame.T)
        dT_dx_max = np.max(np.abs(np.gradient(self.flame.T, self.flame.grid)))
        thermal_thickness = (T_max - T_min) / dT_dx_max

        # Calculate density ratio
        rho_u = self.flame.density[0]  # Unburned density
        rho_b = self.flame.density[-1]  # Burned density
        density_ratio = rho_u / rho_b

        # Create CanteraFreeFlameProperties object
        props = CanteraFreeFlameProperties(
            initial_temperature=self.initial_T,
            initial_pressure=self.initial_P,
            initial_density=rho_u,
            max_temperature=T_max,
            product_temperature=self.flame.T[-1],
            product_density=rho_b,
            flame_speed=self.flame.velocity[0],
            flame_thickness=thermal_thickness,
            max_heat_release_rate=np.max(hrr),
            density_ratio=density_ratio,
        )

        return props

    def get_profiles(self) -> Dict[str, np.ndarray]:
        """
        Get spatial profiles of key quantities.

        Returns:
            Dict with arrays for position, temperature, velocity, density, etc.
        """
        if self.flame is None:
            raise ValueError("Must call calculate_free_flame() first")

        profiles = {
            'x': self.flame.grid,
            'T': self.flame.T,
            'u': self.flame.velocity,
            'rho': self.flame.density,
            'P': np.ones_like(self.flame.grid) * self.initial_P,
            'hrr': self.flame.heat_release_rate
        }

        # Add major species mass fractions
        for species in ['H2', 'O2', 'H2O', 'N2', 'CO', 'CO2', 'CH4']:
            if species in self.gas.species_names:
                idx = self.gas.species_index(species)
                profiles[f'Y_{species}'] = self.flame.Y[idx]

        return profiles

    def create_profile_figures(self, output_dir: str = None,
                              species_list: Optional[List[str]] = None,
                              save_data: bool = False) -> Dict[str, Any]:
        """
        Create and save figures of Cantera free flame profiles.

        Args:
            output_dir: Directory to save figures (creates if doesn't exist)
            species_list: List of species to plot (if None, uses default major species)
            save_data: Whether to save profile data to CSV files

        Returns:
            Dict with figure paths and profile data
        """
        if self.flame is None:
            raise ValueError("Must call run() first to calculate flame")

        import matplotlib.pyplot as plt

        # Setup output directory
        if output_dir is None:
            output_dir = Path.cwd() / "cantera_flame_profiles"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get profiles
        profiles = self.get_profiles()

        # Convert position to mm for better readability
        x_mm = profiles['x'] * 1000  # Convert m to mm

        results = {
            'output_dir': str(output_dir),
            'figures': {},
            'profiles': profiles
        }

        # 1. Temperature Profile
        fig_temp_path = self._plot_temperature_profile(x_mm, profiles['T'], output_dir)
        results['figures']['temperature'] = str(fig_temp_path)

        # 2. Velocity Profile
        fig_vel_path = self._plot_velocity_profile(x_mm, profiles['u'], output_dir)
        results['figures']['velocity'] = str(fig_vel_path)

        # 3. Species Profiles
        if species_list is None:
            # Default major species
            species_list = ['H2', 'O2', 'H2O', 'OH', 'H', 'O']
        fig_species_path = self._plot_species_profiles(x_mm, profiles, species_list, output_dir)
        results['figures']['species'] = str(fig_species_path)

        # 4. Heat Release Rate Profile
        fig_hrr_path = self._plot_hrr_profile(x_mm, profiles['hrr'], output_dir)
        results['figures']['heat_release_rate'] = str(fig_hrr_path)

        # 5. Combined Profile (Temperature + HRR)
        fig_combined_path = self._plot_combined_profiles(x_mm, profiles, output_dir)
        results['figures']['combined'] = str(fig_combined_path)

        # 6. Save data to CSV if requested
        if save_data:
            csv_path = output_dir / "flame_profiles.csv"
            self._save_profiles_to_csv(profiles, csv_path)
            results['data_file'] = str(csv_path)

        return results

    def _plot_temperature_profile(self, x_mm: np.ndarray, T: np.ndarray,
                                 output_dir: Path) -> Path:
        """Create temperature profile plot."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Plot with small points
        ax.plot(x_mm, T, 'ro', markersize=2, label='Temperature', alpha=0.7)
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Temperature (K)', fontsize=12)
        ax.set_title('Cantera Free Flame Temperature Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Add annotations
        T_max = np.max(T)
        T_min = np.min(T)
        ax.axhline(y=T_max, color='r', linestyle='--', alpha=0.3, label=f'T_max = {T_max:.0f} K')
        ax.axhline(y=T_min, color='b', linestyle='--', alpha=0.3, label=f'T_min = {T_min:.0f} K')

        plt.tight_layout()
        output_path = output_dir / "temperature_profile.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_velocity_profile(self, x_mm: np.ndarray, u: np.ndarray,
                              output_dir: Path) -> Path:
        """Create velocity profile plot."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Convert to cm/s for better readability
        u_cms = u * 100

        ax.plot(x_mm, u_cms, 'b-', linewidth=2, label='Velocity')
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Velocity (cm/s)', fontsize=12)
        ax.set_title('Cantera Free Flame Velocity Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add flame speed annotation
        flame_speed = u_cms[0]
        ax.axhline(y=flame_speed, color='g', linestyle='--', alpha=0.3)
        ax.text(x_mm[-1]*0.7, flame_speed*1.05, f'S_L = {flame_speed:.1f} cm/s',
                fontsize=10, color='g')

        plt.tight_layout()
        output_path = output_dir / "velocity_profile.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_species_profiles(self, x_mm: np.ndarray, profiles: Dict,
                              species_list: List[str], output_dir: Path) -> Path:
        """Create species mass fraction profiles plot."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

        # Color map for species
        colors = plt.cm.tab10(np.linspace(0, 1, len(species_list)))

        for i, species in enumerate(species_list):
            key = f'Y_{species}'
            if key in profiles:
                ax.plot(x_mm, profiles[key], linewidth=2,
                       label=species, color=colors[i])

        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Mass Fraction', fontsize=12)
        ax.set_title('Cantera Free Flame Species Profiles', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        output_path = output_dir / "species_profiles.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_hrr_profile(self, x_mm: np.ndarray, hrr: np.ndarray,
                         output_dir: Path) -> Path:
        """Create heat release rate profile plot."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Convert to MW/m^3 for readability
        hrr_MW = hrr / 1e6

        ax.plot(x_mm, hrr_MW, 'orange', linewidth=2, label='Heat Release Rate')
        ax.fill_between(x_mm, 0, hrr_MW, alpha=0.3, color='orange')
        ax.set_xlabel('Position (mm)', fontsize=12)
        ax.set_ylabel('Heat Release Rate (MW/m³)', fontsize=12)
        ax.set_title('Cantera Free Flame Heat Release Rate Profile', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark maximum HRR location
        max_hrr_idx = np.argmax(hrr_MW)
        ax.plot(x_mm[max_hrr_idx], hrr_MW[max_hrr_idx], 'ro', markersize=8)
        ax.annotate(f'Max HRR\n{hrr_MW[max_hrr_idx]:.1f} MW/m³',
                   xy=(x_mm[max_hrr_idx], hrr_MW[max_hrr_idx]),
                   xytext=(x_mm[max_hrr_idx]*1.2, hrr_MW[max_hrr_idx]*0.9),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

        plt.tight_layout()
        output_path = output_dir / "heat_release_rate_profile.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _plot_combined_profiles(self, x_mm: np.ndarray, profiles: Dict,
                               output_dir: Path) -> Path:
        """Create combined plot with temperature, HRR, and major species."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), dpi=150, sharex=True)

        # Temperature
        ax1.plot(x_mm, profiles['T'], 'r-', linewidth=2)
        ax1.set_ylabel('Temperature (K)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Cantera Free Flame Profiles', fontsize=14, fontweight='bold')

        # Heat Release Rate
        hrr_MW = profiles['hrr'] / 1e6
        ax2.plot(x_mm, hrr_MW, 'orange', linewidth=2)
        ax2.fill_between(x_mm, 0, hrr_MW, alpha=0.3, color='orange')
        ax2.set_ylabel('HRR (MW/m³)', fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Major Species
        species_to_plot = ['H2', 'O2', 'H2O']
        colors = ['blue', 'green', 'red']
        for species, color in zip(species_to_plot, colors):
            key = f'Y_{species}'
            if key in profiles:
                ax3.plot(x_mm, profiles[key], linewidth=2, label=species, color=color)

        ax3.set_xlabel('Position (mm)', fontsize=12)
        ax3.set_ylabel('Mass Fraction', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        ax3.set_ylim(bottom=0)

        plt.tight_layout()
        output_path = output_dir / "combined_profiles.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def _save_profiles_to_csv(self, profiles: Dict, csv_path: Path) -> None:
        """Save profile data to CSV file."""
        import pandas as pd

        # Create DataFrame from profiles
        df_data = {'x_m': profiles['x']}

        # Add all profile data
        for key, data in profiles.items():
            if key != 'x' and isinstance(data, np.ndarray):
                df_data[key] = data

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        print(f"Profile data saved to: {csv_path}")

    def _identify_fuel_species(self) -> Optional[str]:
        """
        Identify the fuel species in the mechanism.

        Returns:
            Name of fuel species or None if not found
        """
        common_fuels = ['H2', 'CH4', 'C2H4', 'C3H8', 'C7H16', 'C8H18']

        for fuel in common_fuels:
            if fuel in self.gas.species_names:
                return fuel

        return None




class CanteraBurningRate():

    def __init__(self):
        try:
            import cantera as ct
            self.ct = ct
        except ImportError:
            raise ImportError("Cantera is required for CanteraBurningRate but is not installed.")


class FlameContourAnalyzer:
    """Analyze flame contours with baseline fitting and wrinkledness metrics."""

    def __init__(self, flame_temperature: float = 2500.0,
                 smoothing_method: str = 'savgol',
                 smoothing_strength: float = 0.3):
        """
        Initialize flame contour analyzer.

        Args:
            flame_temperature: Temperature threshold for flame contour (K)
            smoothing_method: Method for smoothing ('savgol', 'gaussian', 'spline')
            smoothing_strength: Strength of smoothing (0-1)
        """
        self.flame_temperature = flame_temperature
        self.base_analyzer = BaseFlameAnalyzer(flame_temperature=flame_temperature)
        self.contour_points = None
        self.fitted_points = None

        # Initialize baseline fitter
        self.baseline_fitter = FlameBaselineFitter(
            smoothing_method=smoothing_method,
            smoothing_strength=smoothing_strength,
            use_ransac=False
        )

        # Parameters for different fit types will be stored here
        self.fit_params = {}
        self.baseline_result = None

    def extract_contour(self, dataset: Any, flame_pos: Optional[float] = None) -> np.ndarray:
        """
        Extract flame contour using base_processing functionality.

        Args:
            dataset: YT dataset object
            flame_pos: Optional flame position for focused extraction (m)

        Returns:
            Array of contour points (x, y) in meters
        """
        # Use base_processing flame analyzer to extract contour
        contour_points = self.base_analyzer.extract_flame_contour(dataset, flame_pos)

        # Sort contour points for better fitting
        if len(contour_points) > 0:
            sorted_points, segments, _ = self.base_analyzer.sort_contour_by_nearest_neighbors(
                contour_points, dataset
            )
            self.contour_points = sorted_points
        else:
            self.contour_points = contour_points

        return self.contour_points

    def fit_baseline(self, contour_points: Optional[np.ndarray] = None,
                    n_upper: Optional[float] = None,
                    n_lower: Optional[float] = None,
                    auto_optimize: bool = True,
                    use_ransac: bool = False) -> Dict[str, Any]:
        """
        Fit smooth baseline to flame contour using asymmetric power-law model.

        Args:
            contour_points: (N, 2) array of x,y coordinates (uses stored if None)
            n_upper: Shape parameter for upper half (None to auto-optimize)
            n_lower: Shape parameter for lower half (None to auto-optimize)
            auto_optimize: Whether to automatically find best shape parameters
            use_ransac: Whether to use RANSAC for robust fitting

        Returns:
            Dictionary with baseline fit results and wrinkledness metrics
        """
        if contour_points is None:
            contour_points = self.contour_points

        if contour_points is None or len(contour_points) < 10:
            return {
                'error': 'Insufficient contour points for fitting',
                'fitted_points': np.array([]),
                'parameters': {},
                'wrinkledness_metrics': {}
            }

        # Update RANSAC setting
        self.baseline_fitter.use_ransac = use_ransac

        # Perform fitting
        self.baseline_result = self.baseline_fitter.fit(
            contour_points,
            n_upper=n_upper,
            n_lower=n_lower,
            auto_optimize=auto_optimize
        )

        # Store fitted points
        self.fitted_points = self.baseline_result.fitted_points

        # Prepare return dictionary
        return {
            'fitted_points': self.baseline_result.fitted_points,
            'parameters': self.baseline_result.parameters,
            'upper_shape': {
                'n': self.baseline_result.upper_params['n'],
                'shape_type': self.baseline_result.parameters['shape_upper']
            },
            'lower_shape': {
                'n': self.baseline_result.lower_params['n'],
                'shape_type': self.baseline_result.parameters['shape_lower']
            },
            'fit_quality': {
                'r_squared': self.baseline_result.r_squared,
                'rmse': self.baseline_result.rmse
            },
            'wrinkledness_metrics': self.baseline_result.wrinkledness_metrics,
            'preprocessing': self.baseline_result.preprocessing_info
        }

    def get_wrinkledness_metrics(self) -> Optional[Dict[str, float]]:
        """
        Get wrinkledness metrics from last baseline fit.

        Returns:
            Dictionary with wrinkledness metrics or None if not fitted
        """
        if self.baseline_result is None:
            return None
        return self.baseline_result.wrinkledness_metrics

    def compare_lengths(self) -> Dict[str, Any]:
        """
        Compare contour length to baseline length.

        Returns:
            Dictionary with length comparison metrics
        """
        if self.contour_points is None or self.baseline_result is None:
            return {
                'contour_length': 0.0,
                'baseline_length': 0.0,
                'length_ratio': 1.0,
                'interpretation': 'No baseline fitted'
            }

        # Calculate arc lengths
        contour_length = self._calculate_arc_length(self.contour_points)
        baseline_length = self._calculate_arc_length(self.baseline_result.fitted_points)

        ratio = contour_length / baseline_length if baseline_length > 0 else 1.0

        return {
            'contour_length': contour_length,
            'baseline_length': baseline_length,
            'length_ratio': ratio,
            'percent_increase': (ratio - 1) * 100,
            'interpretation': self.baseline_result.wrinkledness_metrics.get('interpretation', 'Unknown')
        }

    def _calculate_arc_length(self, points: np.ndarray) -> float:
        """Calculate arc length of a curve."""
        if len(points) < 2:
            return 0.0
        segments = np.diff(points, axis=0)
        lengths = np.sqrt(np.sum(segments**2, axis=1))
        return np.sum(lengths)

    def fit_parabola(self, contour_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit a parabola to the flame contour.

        Args:
            contour_points: Optional contour points to fit (uses stored if None)

        Returns:
            Dictionary with fit results
        """
        if contour_points is None:
            contour_points = self.contour_points

        if contour_points is None or len(contour_points) < 3:
            return {
                'error': 'Insufficient points for parabola fitting',
                'fitted_points': np.array([]),
                'coefficients': {},
                'r_squared': 0.0
            }

        # Extract x and y coordinates
        x = contour_points[:, 0]
        y = contour_points[:, 1]

        # Fit parabola: x = a*y^2 + b*y + c
        coeffs = np.polyfit(y, x, 2)

        # Generate fitted points
        y_fit = np.linspace(y.min(), y.max(), 200)
        x_fit = np.polyval(coeffs, y_fit)
        fitted_points = np.column_stack((x_fit, y_fit))

        # Calculate R-squared
        x_pred = np.polyval(coeffs, y)
        ss_res = np.sum((x - x_pred)**2)
        ss_tot = np.sum((x - np.mean(x))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'fitted_points': fitted_points,
            'coefficients': {'a': coeffs[0], 'b': coeffs[1], 'c': coeffs[2]},
            'r_squared': r_squared,
            'rmse': np.sqrt(np.mean((x - x_pred)**2)),
            'fit_type': 'parabola'
        }

    def fit_ellipse(self, contour_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit an ellipse to the flame contour (leading edge only).

        Args:
            contour_points: Optional contour points to fit (uses stored if None)

        Returns:
            Dictionary with fit results
        """
        if contour_points is None:
            contour_points = self.contour_points

        if contour_points is None or len(contour_points) < 5:
            return {
                'error': 'Insufficient points for ellipse fitting',
                'fitted_points': np.array([]),
                'parameters': {},
                'r_squared': 0.0
            }

        x = contour_points[:, 0]
        y = contour_points[:, 1]

        # For flame contours, we want to fit a semi-ellipse to the leading edge
        # Extract leading edge first (minimum x for each y bin)
        y_bins = np.linspace(y.min(), y.max(), 30)
        leading_edge = []
        for i in range(len(y_bins)-1):
            mask = (y >= y_bins[i]) & (y < y_bins[i+1])
            if i == len(y_bins) - 2:  # Include max in last bin
                mask = (y >= y_bins[i]) & (y <= y_bins[i+1])
            if np.any(mask):
                x_bin = x[mask]
                y_bin = y[mask]
                idx_min = np.argmin(x_bin)
                leading_edge.append([x_bin[idx_min], y_bin[idx_min]])

        if len(leading_edge) < 5:
            return {
                'error': 'Insufficient leading edge points',
                'fitted_points': np.array([]),
                'parameters': {},
                'r_squared': 0.0
            }

        leading_edge = np.array(leading_edge)
        x_edge = leading_edge[:, 0]
        y_edge = leading_edge[:, 1]

        # Fit ellipse parameters using least squares
        # For a semi-ellipse: (x - cx)²/a² + (y - cy)²/b² = 1
        # We'll use a simplified approach for flame-like shapes

        # Center estimates
        x_min = np.min(x_edge)
        x_max = np.max(x_edge)
        y_center = np.mean([np.min(y_edge), np.max(y_edge)])

        # For flames, the ellipse center x should be at or behind the minimum x (flame base)
        x_center = x_min  # Place center at the flame base

        # Estimate semi-axes
        a = x_max - x_min  # Semi-major axis along x
        b = (np.max(y_edge) - np.min(y_edge)) / 2  # Semi-minor axis along y

        # Generate fitted ellipse points (only the forward-facing half)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        x_ellipse = x_center + a * np.cos(theta)  # Positive because we want the right/forward half
        y_ellipse = y_center + b * np.sin(theta)

        # Clip to the y-range of the contour
        mask = (y_ellipse >= np.min(y_edge)) & (y_ellipse <= np.max(y_edge))
        fitted_points = np.column_stack((x_ellipse[mask], y_ellipse[mask]))

        # Calculate R-squared using leading edge points
        if len(fitted_points) > 0:
            from scipy.interpolate import interp1d
            # Sort fitted points by y for interpolation
            sort_idx = np.argsort(fitted_points[:, 1])
            y_fitted_sorted = fitted_points[sort_idx, 1]
            x_fitted_sorted = fitted_points[sort_idx, 0]

            # Interpolate to get predicted x values at edge y positions
            interp = interp1d(y_fitted_sorted, x_fitted_sorted,
                            kind='linear', bounds_error=False, fill_value='extrapolate')
            x_pred = interp(y_edge)

            # Calculate R-squared on leading edge
            ss_res = np.sum((x_edge - x_pred)**2)
            ss_tot = np.sum((x_edge - np.mean(x_edge))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((x_edge - x_pred)**2))
        else:
            r_squared = 0
            rmse = np.inf

        return {
            'fitted_points': fitted_points,
            'parameters': {
                'center_x': x_center,
                'center_y': y_center,
                'semi_major': a,
                'semi_minor': b
            },
            'r_squared': r_squared,
            'rmse': rmse,
            'fit_type': 'ellipse'
        }

    def fit_generalized_ellipse(self, contour_points: Optional[np.ndarray] = None,
                                n: Optional[float] = None,
                                optimize_center: bool = True,
                                debug: bool = False) -> Dict[str, Any]:
        """
        Fit a generalized ellipse (superellipse) to the flame contour with adjustable tip sharpness
        and asymmetric tip positioning.

        The generalized ellipse equation: |x/a|^n + |y/b|^n = 1
        where n controls the shape:
        - n < 2: pointed/diamond-like shape
        - n = 2: standard ellipse
        - n > 2: rectangular/boxy shape

        Args:
            contour_points: Optional contour points to fit (uses stored if None)
            n: Shape parameter controlling tip sharpness (if None, will optimize)
               Typical values: 1.5 (pointed), 2.0 (ellipse), 2.5 (blunt)
            optimize_center: Whether to optimize the vertical center position for asymmetry

        Returns:
            Dictionary with fit results
        """
        if contour_points is None:
            contour_points = self.contour_points

        if contour_points is None or len(contour_points) < 5:
            return {
                'error': 'Insufficient points for generalized ellipse fitting',
                'fitted_points': np.array([]),
                'parameters': {},
                'r_squared': 0.0
            }

        x = contour_points[:, 0]
        y = contour_points[:, 1]

        # Extract leading edge
        y_bins = np.linspace(y.min(), y.max(), 30)
        leading_edge = []
        for i in range(len(y_bins)-1):
            mask = (y >= y_bins[i]) & (y < y_bins[i+1])
            if i == len(y_bins) - 2:
                mask = (y >= y_bins[i]) & (y <= y_bins[i+1])
            if np.any(mask):
                x_bin = x[mask]
                y_bin = y[mask]
                idx_min = np.argmin(x_bin)
                leading_edge.append([x_bin[idx_min], y_bin[idx_min]])

        if len(leading_edge) < 5:
            return {
                'error': 'Insufficient leading edge points',
                'fitted_points': np.array([]),
                'parameters': {},
                'r_squared': 0.0
            }

        leading_edge = np.array(leading_edge)
        x_edge = leading_edge[:, 0]
        y_edge = leading_edge[:, 1]

        # Initial parameter estimates
        x_min = np.min(x_edge)
        x_max = np.max(x_edge)
        y_min = np.min(y_edge)
        y_max = np.max(y_edge)

        # Initial center - can be asymmetric
        y_center_init = np.mean([y_min, y_max])

        # For the generalized ellipse, we want:
        # - Base at x_min (theta = ±π)
        # - Tip at x_max (theta = 0)
        # This means x_center = x_min and a = x_max - x_min
        x_center = x_min
        a = x_max - x_min  # Full flame length
        b = (y_max - y_min) / 2

        # Find the index of maximum x to determine where the tip should be
        idx_max_x = np.argmax(x_edge)
        y_at_max_x = y_edge[idx_max_x]

        # Optimize parameters
        if n is None or optimize_center:
            from scipy.optimize import minimize

            def objective(params):
                if n is None and optimize_center:
                    n_val, y_center = params
                elif n is None:
                    n_val = params[0]
                    y_center = y_center_init
                else:
                    n_val = n
                    y_center = params[0] if optimize_center else y_center_init

                # Broader range for n to allow more shapes
                if n_val <= 0.3 or n_val > 6:
                    return 1e10

                # Adjust semi-axes based on y_center for asymmetric shape
                b_upper = abs(y_max - y_center)
                b_lower = abs(y_center - y_min)

                # Generate asymmetric superellipse points
                # Upper half (0 to π/2)
                theta_up = np.linspace(0, np.pi/2, 50)
                x_up = x_center + a * np.power(np.cos(theta_up), 2/n_val)
                y_up = y_center + b_upper * np.power(np.sin(theta_up), 2/n_val)

                # Lower half (-π/2 to 0)
                theta_down = np.linspace(-np.pi/2, 0, 50)
                x_down = x_center + a * np.power(np.cos(theta_down), 2/n_val)
                y_down = y_center - b_lower * np.power(np.abs(np.sin(theta_down)), 2/n_val)

                # Combine (avoiding duplicate at theta=0)
                x_super = np.concatenate([x_down[:-1], x_up])
                y_super = np.concatenate([y_down[:-1], y_up])

                # Clip to y-range
                mask = (y_super >= y_min) & (y_super <= y_max)
                if not np.any(mask):
                    return 1e10

                x_super = x_super[mask]
                y_super = y_super[mask]

                # Interpolate to compare with edge
                from scipy.interpolate import interp1d
                if len(x_super) < 2:
                    return 1e10

                sort_idx = np.argsort(y_super)
                y_super_sorted = y_super[sort_idx]
                x_super_sorted = x_super[sort_idx]

                try:
                    interp = interp1d(y_super_sorted, x_super_sorted,
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
                    x_pred = interp(y_edge)
                    error = np.sum((x_edge - x_pred)**2)

                    # Add strong penalty if tip doesn't align with max x
                    tip_x = np.max(x_super)
                    tip_penalty = 100 * (tip_x - x_max)**2
                    error += tip_penalty

                    # No regularization - let data determine the best n value
                    # Only apply very weak regularization for extreme values
                    if n_val < 0.5 or n_val > 5.0:
                        regularization = 10 * ((n_val - 2.5)**2)
                        error += regularization

                    # Penalize extreme asymmetry
                    if b_lower > 0:
                        asymmetry_ratio = b_upper / b_lower
                        if asymmetry_ratio > 3 or asymmetry_ratio < 0.33:
                            error += 50 * (asymmetry_ratio - 1)**2

                except:
                    return 1e10

                return error

            # Set up optimization with multiple initial guesses to avoid local minima
            if n is None and optimize_center:
                # Try multiple initial guesses for n to find global minimum
                best_result = None
                best_error = np.inf
                n_guesses = [1.2, 1.5, 2.0, 2.5, 3.0]  # Range of shape parameters

                for n_init in n_guesses:
                    initial_guess = [n_init, y_center_init]
                    bounds = [(0.5, 4.5), (y_min - b*0.3, y_max + b*0.3)]
                    result = minimize(objective, initial_guess, bounds=bounds,
                                    method='L-BFGS-B', options={'maxiter': 100})
                    if result.fun < best_error:
                        best_error = result.fun
                        best_result = result

                n, y_center = best_result.x
            elif n is None:
                # Only optimize n - try multiple initial guesses
                from scipy.optimize import minimize_scalar
                best_n = 2.0
                best_error = np.inf

                # Try multiple starting points
                for n_init in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                    result = minimize_scalar(lambda n_val: objective([n_val]),
                                           bounds=(0.5, 4.5), method='bounded',
                                           options={'xatol': 1e-4})
                    error = objective([result.x])
                    if error < best_error:
                        best_error = error
                        best_n = result.x

                n = best_n
                y_center = y_center_init
            elif optimize_center:
                # Only optimize y_center
                result = minimize_scalar(lambda y_c: objective([y_c]),
                                        bounds=(y_min - b*0.2, y_max + b*0.2), method='bounded')
                y_center = result.x
            else:
                y_center = y_center_init
        else:
            y_center = y_center_init

        # Recalculate semi-axes with final y_center for asymmetric shape
        b_upper = abs(y_max - y_center)
        b_lower = abs(y_center - y_min)

        # Generate final asymmetric superellipse points with optimal parameters
        # Split into upper and lower halves for asymmetric handling
        theta_upper = np.linspace(0, np.pi/2, 100)  # Upper half
        theta_lower = np.linspace(-np.pi/2, 0, 100)  # Lower half

        # Upper half with its own semi-axis
        x_upper = x_center + a * np.power(np.cos(theta_upper), 2/n)
        y_upper = y_center + b_upper * np.power(np.sin(theta_upper), 2/n)

        # Lower half with its own semi-axis
        x_lower = x_center + a * np.power(np.cos(theta_lower), 2/n)
        y_lower = y_center - b_lower * np.power(np.abs(np.sin(theta_lower)), 2/n)

        # Combine upper and lower (avoid duplicate at theta=0)
        x_ellipse = np.concatenate([x_lower[:-1], x_upper])
        y_ellipse = np.concatenate([y_lower[:-1], y_upper])

        # Clip to y-range of contour
        mask = (y_ellipse >= y_min) & (y_ellipse <= y_max)
        fitted_points = np.column_stack((x_ellipse[mask], y_ellipse[mask]))

        # Calculate R-squared
        if len(fitted_points) > 0:
            from scipy.interpolate import interp1d
            sort_idx = np.argsort(fitted_points[:, 1])
            y_fitted_sorted = fitted_points[sort_idx, 1]
            x_fitted_sorted = fitted_points[sort_idx, 0]

            interp = interp1d(y_fitted_sorted, x_fitted_sorted,
                            kind='linear', bounds_error=False, fill_value='extrapolate')
            x_pred = interp(y_edge)

            ss_res = np.sum((x_edge - x_pred)**2)
            ss_tot = np.sum((x_edge - np.mean(x_edge))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((x_edge - x_pred)**2))
        else:
            r_squared = 0
            rmse = np.inf

        # Classify shape based on n with finer granularity
        if n < 1.0:
            shape_type = "hyperbolic"
        elif n < 1.5:
            shape_type = "very_pointed"
        elif n < 1.8:
            shape_type = "pointed"
        elif n < 2.2:
            shape_type = "elliptical"
        elif n < 2.5:
            shape_type = "slightly_blunt"
        elif n < 3.0:
            shape_type = "blunt"
        elif n < 4.0:
            shape_type = "rectangular"
        else:
            shape_type = "super_rectangular"

        return {
            'fitted_points': fitted_points,
            'parameters': {
                'center_x': x_center,
                'center_y': y_center,
                'semi_major': a,
                'semi_minor_upper': b_upper,
                'semi_minor_lower': b_lower,
                'n': n,
                'shape_type': shape_type,
                'tip_x': x_center + a,  # Should equal x_max
                'base_x': x_center,     # Should equal x_min
                'asymmetry_ratio': b_upper / b_lower if b_lower > 0 else np.inf
            },
            'r_squared': r_squared,
            'rmse': rmse,
            'fit_type': 'generalized_ellipse'
        }

    def fit_power_law(self, contour_points: Optional[np.ndarray] = None, n: Optional[float] = None) -> Dict[str, Any]:
        """
        Fit a power-law curve to the flame contour.

        Args:
            contour_points: Optional contour points to fit (uses stored if None)
            n: Power exponent (if None, will optimize)

        Returns:
            Dictionary with fit results
        """
        if contour_points is None:
            contour_points = self.contour_points

        if contour_points is None or len(contour_points) < 3:
            return {
                'error': 'Insufficient points for power-law fitting',
                'fitted_points': np.array([]),
                'parameters': {},
                'r_squared': 0.0
            }

        x = contour_points[:, 0]
        y = contour_points[:, 1]

        # Normalize coordinates
        x_base = np.min(x)
        L = np.max(x) - x_base
        y_center = np.mean([np.min(y), np.max(y)])
        y_half = (np.max(y) - np.min(y)) / 2

        if y_half == 0:
            return {
                'error': 'Zero height range',
                'fitted_points': np.array([]),
                'parameters': {},
                'r_squared': 0.0
            }

        y_norm = (y - y_center) / y_half

        # Optimize n if not provided
        if n is None:
            from scipy.optimize import minimize_scalar
            def objective(n_val):
                if n_val <= 0 or n_val > 5:
                    return 1e10
                x_pred = x_base + L * (1 - np.abs(y_norm)**n_val)
                return np.sum((x - x_pred)**2)

            result = minimize_scalar(objective, bounds=(0.5, 4.0), method='bounded')
            n = result.x

        # Generate fitted curve
        y_fit = np.linspace(y.min(), y.max(), 200)
        y_fit_norm = (y_fit - y_center) / y_half
        x_fit = x_base + L * (1 - np.abs(y_fit_norm)**n)
        fitted_points = np.column_stack((x_fit, y_fit))

        # Calculate R-squared
        x_pred = x_base + L * (1 - np.abs(y_norm)**n)
        ss_res = np.sum((x - x_pred)**2)
        ss_tot = np.sum((x - np.mean(x))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Classify shape
        if n < 1.3:
            shape_type = "pointed"
        elif n < 2.3:
            shape_type = "parabolic"
        elif n < 3.0:
            shape_type = "blunt"
        else:
            shape_type = "stubby"

        return {
            'fitted_points': fitted_points,
            'parameters': {
                'x_base': x_base,
                'flame_length': L,
                'y_center': y_center,
                'y_half': y_half,
                'n': n,
                'shape_type': shape_type
            },
            'r_squared': r_squared,
            'rmse': np.sqrt(np.mean((x - x_pred)**2)),
            'fit_type': 'power_law'
        }

    def fit_all_curves(self, contour_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Fit all available curve types to the contour and compare results.

        Args:
            contour_points: Optional contour points to fit (uses stored if None)

        Returns:
            Dictionary with all fit results and comparison
        """
        if contour_points is None:
            contour_points = self.contour_points

        if contour_points is None or len(contour_points) < 5:
            return {
                'error': 'Insufficient contour points for fitting',
                'fits': {},
                'best_fit': None
            }

        # Fit all curve types
        results = {}

        # 1. Baseline (asymmetric power-law)
        print("  Fitting baseline (asymmetric)...")
        baseline_result = self.fit_baseline(contour_points, auto_optimize=True)
        if 'error' not in baseline_result:
            results['baseline'] = baseline_result

        # 2. Parabola
        print("  Fitting parabola...")
        parabola_result = self.fit_parabola(contour_points)
        if 'error' not in parabola_result:
            results['parabola'] = parabola_result

        # 3. Ellipse
        print("  Fitting ellipse...")
        ellipse_result = self.fit_ellipse(contour_points)
        if 'error' not in ellipse_result:
            results['ellipse'] = ellipse_result

        # 4. Generalized Ellipse
        print("  Fitting generalized ellipse...")
        gen_ellipse_result = self.fit_generalized_ellipse(contour_points)
        if 'error' not in gen_ellipse_result:
            results['generalized_ellipse'] = gen_ellipse_result

        # 5. Power-law (symmetric)
        print("  Fitting power-law...")
        power_law_result = self.fit_power_law(contour_points)
        if 'error' not in power_law_result:
            results['power_law'] = power_law_result

        # Find best fit based on R-squared
        best_fit = None
        best_r2 = -np.inf
        for fit_type, fit_data in results.items():
            if 'fit_quality' in fit_data:
                r2 = fit_data['fit_quality']['r_squared']
            elif 'r_squared' in fit_data:
                r2 = fit_data['r_squared']
            else:
                r2 = 0

            if r2 > best_r2:
                best_r2 = r2
                best_fit = fit_type

        # Calculate comparison metrics
        comparison = {
            'best_fit': best_fit,
            'best_r_squared': best_r2,
            'ranking': sorted(
                [(k, v.get('fit_quality', {}).get('r_squared', v.get('r_squared', 0)))
                 for k, v in results.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }

        return {
            'fits': results,
            'comparison': comparison,
            'contour_points': contour_points
        }