#!/usr/bin/env python
"""
Test script for Chapman-Jouguet analyzer functionality.
Tests both deflagration and detonation calculations using the unified pele_processing namespace.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src directory to path for testing
src_dir = os.path.join(os.path.dirname(__file__), 'src')
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)
    print(f"Added {src_dir} to Python path for testing\n")

def test_cj_analyzer():
    """Test the Chapman-Jouguet analyzer with various equivalence ratios."""

    try:
        # Import from additional_processing submodule
        from pele_processing.additional_processing import (
            CJAnalyzer,
            create_chapman_jouguet_analyzer,
            CJProperties,
            CJType
        )
        print("[OK] Successfully imported CJ components from pele_processing.additional_processing")

        # Also test that flame components are available from base_processing
        from pele_processing.base_processing import (
            PeleFlameAnalyzer,
            create_flame_analyzer
        )
        print("[OK] Successfully imported flame components from pele_processing.base_processing")
        print()
    except ImportError as e:
        print(f"[FAILED] Failed to import components: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check if SDToolbox is available for comparison
    try:
        from sdtoolbox.postshock import CJspeed
        sdtoolbox_available = True
        print("[OK] SDToolbox is available for comparison")
        print()
    except ImportError:
        sdtoolbox_available = False
        print("[WARNING] SDToolbox not available - skipping comparison tests")
        print()

    # Set initial thermodynamic state
    T = 300.0  # Initial temperature in K
    P = 101325.0  # Initial pressure in Pa

    # Try to find mechanism file
    possible_paths = [
        '../../../chemical_mechanisms/LiDryer.yaml',
        'chemical_mechanisms/LiDryer.yaml',
        '../chemical_mechanisms/LiDryer.yaml',
        '../../chemical_mechanisms/LiDryer.yaml',
    ]

    mech = None
    for path in possible_paths:
        if os.path.exists(path):
            mech = path
            break

    if mech is None:
        print("WARNING: Could not find LiDryer.yaml mechanism file.")
        print("Using simplified hydrogen mechanism string instead.")
        # Use a simple H2/O2 mechanism string
        mech = """
        phases:
        - name: gas
          thermo: ideal-gas
          elements: [O, H, N]
          species: [H2, O2, H2O, N2]
          kinetics: gas
          reactions: all
          state: {T: 300.0, P: 101325.0}

        species:
        - name: H2
          composition: {H: 2}
          thermo:
            model: NASA7
            temperature-ranges: [200.0, 1000.0, 3500.0]
            data:
            - [2.34433112, 0.00798052075, -1.9478151e-05, 2.01572094e-08, -7.37611761e-12,
               -917.935173, 0.683010238]
            - [3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10, 2.00255376e-14,
               -950.158922, -3.20502331]
        - name: O2
          composition: {O: 2}
          thermo:
            model: NASA7
            temperature-ranges: [200.0, 1000.0, 3500.0]
            data:
            - [3.78245636, -0.00299673416, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12,
               -1063.94356, 3.65767573]
            - [3.28253784, 0.00148308754, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14,
               -1088.45772, 5.45323129]
        - name: H2O
          composition: {H: 2, O: 1}
          thermo:
            model: NASA7
            temperature-ranges: [200.0, 1000.0, 3500.0]
            data:
            - [4.19864056, -0.0020364341, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12,
               -30293.7267, -0.849032208]
            - [3.03399249, 0.00217691804, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14,
               -30004.2971, 4.9667701]
        - name: N2
          composition: {N: 2}
          thermo:
            model: NASA7
            temperature-ranges: [300.0, 1000.0, 5000.0]
            data:
            - [3.298677, 0.0014082404, -3.963222e-06, 5.641515e-09, -2.444854e-12,
               -1020.8999, 3.950372]
            - [2.92664, 0.0014879768, -5.68476e-07, 1.0097038e-10, -6.753351e-15,
               -922.7977, 5.980528]
        """

    print(f"Using mechanism: {mech if isinstance(mech, str) and os.path.exists(mech) else 'inline mechanism'}")
    print()

    # Define equivalence ratio array
    phi_array = np.linspace(0.2, 2.0, 20)  # Reduced number for faster testing

    print("="*60)
    print("Testing Chapman-Jouguet Analyzer")
    print("="*60)

    try:
        # Create analyzer using factory function
        print("Creating CJ analyzer...")
        analyzer = create_chapman_jouguet_analyzer(mech, solver_type="manual", verbose=False)
        print("✓ Analyzer created successfully")
        print(f"  Type: {type(analyzer).__name__}")
        print()
    except Exception as e:
        print(f"✗ Failed to create analyzer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Storage for results
    def_cj_results = []
    det_cj_results = []
    sdtoolbox_results = [] if sdtoolbox_available else None

    print("Calculating CJ solutions for various equivalence ratios...")
    print("-"*60)

    for i, phi in enumerate(phi_array):
        # Stoichiometric hydrogen-oxygen mixture with nitrogen
        q = {'H2': 2 * phi, 'O2': 1, 'N2': 3.76}  # Adjust for equivalence ratio

        print(f"\n[{i+1}/{len(phi_array)}] φ={phi:.2f}:")

        # Deflagration calculation
        if phi > 1:
            print("  ⚠ Rich mixture (φ>1) - CJ deflagration may not converge")

        try:
            def_result = analyzer.analyze_cj_deflagration(T, P, q)
            def_cj_results.append(def_result)
            print(f'  ✓ Deflagration: v={def_result.velocity:.1f} m/s, T={def_result.temperature:.1f} K, P={def_result.pressure/1e5:.2f} bar')

            # Verify result type
            assert isinstance(def_result, CJProperties), "Result should be CJProperties instance"
            assert def_result.cj_type == CJType.DEFLAGRATION, "Should be deflagration type"
        except Exception as e:
            print(f'  ✗ Deflagration failed: {e}')
            def_cj_results.append(None)

        # Detonation calculation
        try:
            det_result = analyzer.analyze_cj_detonation(T, P, q)
            det_cj_results.append(det_result)
            print(f'  [OK] Detonation: v={det_result.velocity:.1f} m/s, T={det_result.temperature:.1f} K, P={det_result.pressure/1e5:.2f} bar')

            # Verify result type
            assert isinstance(det_result, CJProperties), "Result should be CJProperties instance"
            assert det_result.cj_type == CJType.DETONATION, "Should be detonation type"

            # Compare with SDToolbox if available
            if sdtoolbox_available:
                try:
                    import cantera as ct
                    gas = ct.Solution(mech)
                    gas.TPX = T, P, q
                    sd_speed = CJspeed(P, T, q, mech)
                    sdtoolbox_results.append(sd_speed)

                    # Calculate difference
                    abs_diff = abs(det_result.velocity - sd_speed)
                    rel_diff = abs_diff / sd_speed * 100

                    print(f'  [SDToolbox] Detonation: v={sd_speed:.1f} m/s')
                    print(f'  [Comparison] Difference: {abs_diff:.1f} m/s ({rel_diff:.2f}%)')

                    if rel_diff > 1.0:
                        print(f'  [WARNING] Large difference detected (>1%)')
                except Exception as e:
                    print(f'  [SDToolbox] Comparison failed: {e}')
                    sdtoolbox_results.append(None)

        except Exception as e:
            print(f'  [FAILED] Detonation failed: {e}')
            det_cj_results.append(None)
            if sdtoolbox_available:
                sdtoolbox_results.append(None)

    # Calculate statistics
    successful_def = sum(1 for r in def_cj_results if r is not None)
    successful_det = sum(1 for r in det_cj_results if r is not None)

    print("\n" + "="*60)
    print("Test Summary:")
    print("="*60)
    print(f"Total calculations: {len(phi_array)}")
    print(f"Successful deflagrations: {successful_def}/{len(phi_array)} ({100*successful_def/len(phi_array):.1f}%)")
    print(f"Successful detonations: {successful_det}/{len(phi_array)} ({100*successful_det/len(phi_array):.1f}%)")

    # Extract velocities for plotting
    def_cj_vel = [r.velocity if r else np.nan for r in def_cj_results]
    det_cj_vel = [r.velocity if r else np.nan for r in det_cj_results]

    # Extract temperatures
    def_cj_temp = [r.temperature if r else np.nan for r in def_cj_results]
    det_cj_temp = [r.temperature if r else np.nan for r in det_cj_results]

    # Extract pressures (convert to bar)
    def_cj_pres = [r.pressure/1e5 if r else np.nan for r in def_cj_results]
    det_cj_pres = [r.pressure/1e5 if r else np.nan for r in det_cj_results]

    # Create plots
    print("\nGenerating plots...")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Chapman-Jouguet Solutions vs Equivalence Ratio φ', fontsize=14)

    # Detonation plots (left column)
    axes[0, 0].plot(phi_array, det_cj_vel, 'r-', label='CJ Detonation (Manual)', linewidth=2, marker='o', markersize=4)

    # Add SDToolbox comparison if available
    if sdtoolbox_available and sdtoolbox_results:
        sd_vel = [v if v else np.nan for v in sdtoolbox_results]
        axes[0, 0].plot(phi_array, sd_vel, 'g--', label='SDToolbox', linewidth=1.5, marker='x', markersize=4)

    axes[0, 0].set_ylabel('Velocity [m/s]')
    axes[0, 0].set_title('Detonation Velocity')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot(phi_array, det_cj_temp, 'r-', label='CJ Detonation', linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_ylabel('Temperature [K]')
    axes[1, 0].set_title('Detonation Temperature')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[2, 0].plot(phi_array, det_cj_pres, 'r-', label='CJ Detonation', linewidth=2, marker='o', markersize=4)
    axes[2, 0].set_xlabel('Equivalence Ratio φ [-]')
    axes[2, 0].set_ylabel('Pressure [bar]')
    axes[2, 0].set_title('Detonation Pressure')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    # Deflagration plots (right column)
    axes[0, 1].plot(phi_array, def_cj_vel, 'b-', label='CJ Deflagration', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].set_title('Deflagration Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 1].plot(phi_array, def_cj_temp, 'b-', label='CJ Deflagration', linewidth=2, marker='s', markersize=4)
    axes[1, 1].set_ylabel('Temperature [K]')
    axes[1, 1].set_title('Deflagration Temperature')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[2, 1].plot(phi_array, def_cj_pres, 'b-', label='CJ Deflagration', linewidth=2, marker='s', markersize=4)
    axes[2, 1].set_xlabel('Equivalence Ratio φ [-]')
    axes[2, 1].set_ylabel('Pressure [bar]')
    axes[2, 1].set_title('Deflagration Pressure')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    plt.tight_layout()

    # Save figure
    output_file = 'cj_test_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")

    plt.show()

    print("\n" + "="*60)
    print("✓ Chapman-Jouguet Analyzer Test Complete!")
    print("="*60)

    # Print some specific values for verification
    if successful_det > 0:
        stoich_idx = np.argmin(np.abs(phi_array - 1.0))  # Find closest to stoichiometric
        if det_cj_results[stoich_idx] is not None:
            print(f"\nStoichiometric (φ≈1.0) Detonation Properties:")
            r = det_cj_results[stoich_idx]
            print(f"  Velocity: {r.velocity:.1f} m/s")
            print(f"  Temperature: {r.temperature:.1f} K")
            print(f"  Pressure: {r.pressure/1e5:.2f} bar")
            print(f"  Density: {r.density:.3f} kg/m³")
            print(f"  Mach number: {r.mach_number:.2f}")

            # Expected values for H2-O2 at φ=1.0
            print("\nExpected values for H2-O2-N2 (φ=1.0):")
            print("  Velocity: ~1970 m/s")
            print("  Temperature: ~2950 K")
            print("  Pressure: ~15.6 bar")

if __name__ == "__main__":
    print("Chapman-Jouguet Analyzer Test Script")
    print("="*60)
    print()

    # Check if Cantera is available
    try:
        import cantera as ct
        print(f"✓ Cantera version {ct.__version__} is available")
    except ImportError:
        print("✗ Cantera is not installed. This test requires Cantera.")
        print("  Install with: conda install -c cantera cantera")
        sys.exit(1)

    print()
    test_cj_analyzer()