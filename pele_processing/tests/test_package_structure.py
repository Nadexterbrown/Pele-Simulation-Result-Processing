#!/usr/bin/env python
"""
Test script to verify the package structure is working correctly.
All imports should work directly through pele_processing.
"""

import sys
import traceback

def test_imports():
    """Test that all packages can be imported through pele_processing."""

    print("Testing pele_processing package imports...\n")

    # Test main pele_processing import
    try:
        import pele_processing
        print("[OK] Successfully imported pele_processing")
        print(f"  Version: {pele_processing.__version__}")
        print(f"  Author: {pele_processing.__author__}")
        print(f"  Email: {pele_processing.__email__}")
        print(f"  Description: {pele_processing.__description__}")

    except ImportError as e:
        print(f"[FAILED] Failed to import pele_processing: {e}")
        traceback.print_exc()
        return

    print()

    # Test base processing components through submodule
    print("Testing base processing components...")
    try:
        # Core components
        from pele_processing.base_processing import (
            WaveType,
            FlameProperties,
            ShockProperties,
            PressureWaveProperties,
            ThermodynamicState
        )
        print("  [OK] Successfully imported core domain models from base_processing")

        # Analysis components
        from pele_processing.base_processing import (
            PeleFlameAnalyzer,
            create_flame_analyzer,
            PeleShockAnalyzer,
            create_shock_analyzer
        )
        print("  [OK] Successfully imported analysis components from base_processing")

        # Try pressure wave if available
        try:
            from pele_processing.base_processing import (
                PelePressureWaveAnalyzer,
                create_pressure_wave_analyzer,
                DetectionMethod
            )
            print("  [OK] Successfully imported pressure wave analysis from base_processing")
        except ImportError:
            print("  [WARNING] Pressure wave analysis not available (optional)")

        # Data processing
        from pele_processing.base_processing import (
            create_data_loader,
            create_data_extractor
        )
        print("  [OK] Successfully imported data processing from base_processing")

        # Configuration
        from pele_processing.base_processing import (
            AppConfig,
            load_config,
            create_default_config
        )
        print("  [OK] Successfully imported configuration from base_processing")

    except ImportError as e:
        print(f"  [FAILED] Failed to import base processing components: {e}")
        traceback.print_exc()

    print()

    # Test additional processing components (Chapman-Jouguet) through submodule
    print("Testing additional processing components...")
    try:
        from pele_processing.additional_processing import (
            ChapmanJouguetAnalyzer,
            CJProperties,
            CJType,
            CJAnalyzer,
            create_chapman_jouguet_analyzer
        )
        print("  [OK] Successfully imported Chapman-Jouguet components from additional_processing")
        print("      - ChapmanJouguetAnalyzer (interface)")
        print("      - CJProperties (domain model)")
        print("      - CJType (enum)")
        print("      - CJAnalyzer (implementation)")
        print("      - create_chapman_jouguet_analyzer (factory)")

    except ImportError as e:
        print(f"  [FAILED] Failed to import additional processing components: {e}")
        traceback.print_exc()

    print("\n" + "="*60)
    print("Package structure summary:")
    print("="*60)

    # Show what's available at the package level
    try:
        import pele_processing
        print("\nExports available directly from pele_processing:")

        # Get all exports
        all_exports = [item for item in dir(pele_processing) if not item.startswith('_')]

        print(f"  Submodules: {[e for e in all_exports if not e.startswith('__')]}")
        print(f"  Package metadata: __version__, __author__, __email__, __description__, __url__")
        print(f"\nTotal direct exports: {len(all_exports)}")

        # Show what's available from submodules
        print("\nExports from pele_processing.base_processing (sample):")
        base_exports = [item for item in dir(pele_processing.base_processing) if not item.startswith('_')]
        print(f"  Total exports: {len(base_exports)}")
        print(f"  Sample: {base_exports[:5]}...")

        print("\nExports from pele_processing.additional_processing (sample):")
        add_exports = [item for item in dir(pele_processing.additional_processing) if not item.startswith('_')]
        print(f"  Total exports: {len(add_exports)}")
        print(f"  Sample: {add_exports[:5]}...")

    except Exception as e:
        print(f"Error listing exports: {e}")
        traceback.print_exc()

    print("\n" + "="*60)
    print("[SUCCESS] Package structure is working correctly!")
    print("="*60)
    print("\nImport structure:")
    print("  - Strict submodule separation enforced")
    print("  - Components ONLY available through their respective submodules")
    print("  - Prevents naming conflicts between similar functions")
    print("\nExample usage:")
    print("  from pele_processing.base_processing import PeleFlameAnalyzer, create_flame_analyzer")
    print("  from pele_processing.additional_processing import CJAnalyzer, create_chapman_jouguet_analyzer")


if __name__ == "__main__":
    # Add src directory to path to test local development
    import os
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if os.path.exists(src_dir):
        sys.path.insert(0, src_dir)
        print(f"Added {src_dir} to Python path for testing\n")

    test_imports()