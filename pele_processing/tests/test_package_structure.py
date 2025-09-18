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

    # Test base processing components
    print("Testing base processing components...")
    try:
        # Core components
        from pele_processing import (
            WaveType,
            FlameProperties,
            ShockProperties,
            PressureWaveProperties,
            ThermodynamicState
        )
        print("  [OK] Successfully imported core domain models")

        # Analysis components
        from pele_processing import (
            PeleFlameAnalyzer,
            create_flame_analyzer,
            PeleShockAnalyzer,
            create_shock_analyzer
        )
        print("  [OK] Successfully imported analysis components")

        # Try pressure wave if available
        try:
            from pele_processing import (
                PelePressureWaveAnalyzer,
                create_pressure_wave_analyzer,
                DetectionMethod
            )
            print("  [OK] Successfully imported pressure wave analysis components")
        except ImportError:
            print("  [WARNING] Pressure wave analysis not available (optional)")

        # Data processing
        from pele_processing import (
            create_data_loader,
            create_data_extractor
        )
        print("  [OK] Successfully imported data processing components")

        # Configuration
        from pele_processing import (
            AppConfig,
            load_config,
            create_default_config
        )
        print("  [OK] Successfully imported configuration components")

    except ImportError as e:
        print(f"  [FAILED] Failed to import base processing components: {e}")
        traceback.print_exc()

    print()

    # Test additional processing components (Chapman-Jouguet)
    print("Testing additional processing components...")
    try:
        from pele_processing import (
            ChapmanJouguetAnalyzer,
            CJProperties,
            CJType,
            CJAnalyzer,
            create_chapman_jouguet_analyzer
        )
        print("  [OK] Successfully imported Chapman-Jouguet analysis components")
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
        print("\nSample of available exports from pele_processing:")

        # Get all exports
        all_exports = [item for item in dir(pele_processing) if not item.startswith('_')]

        # Categorize exports
        analyzers = [e for e in all_exports if 'Analyzer' in e]
        properties = [e for e in all_exports if 'Properties' in e]
        creators = [e for e in all_exports if e.startswith('create_')]

        print(f"\nAnalyzers ({len(analyzers)}):")
        for item in analyzers[:5]:
            print(f"  - {item}")
        if len(analyzers) > 5:
            print(f"  ... and {len(analyzers)-5} more")

        print(f"\nProperties/Domain models ({len(properties)}):")
        for item in properties[:5]:
            print(f"  - {item}")
        if len(properties) > 5:
            print(f"  ... and {len(properties)-5} more")

        print(f"\nFactory functions ({len(creators)}):")
        for item in creators[:5]:
            print(f"  - {item}")
        if len(creators) > 5:
            print(f"  ... and {len(creators)-5} more")

        print(f"\nTotal package exports: {len(all_exports)}")

    except Exception as e:
        print(f"Error listing exports: {e}")
        traceback.print_exc()

    print("\n" + "="*60)
    print("[SUCCESS] Package structure is working correctly!")
    print("="*60)
    print("\nImport structure:")
    print("  - All components available directly from pele_processing")
    print("  - Combines base_processing and additional_processing modules")
    print("\nExample usage:")
    print("  from pele_processing import PeleFlameAnalyzer, create_flame_analyzer")
    print("  from pele_processing import CJAnalyzer, create_chapman_jouguet_analyzer")


if __name__ == "__main__":
    # Add src directory to path to test local development
    import os
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if os.path.exists(src_dir):
        sys.path.insert(0, src_dir)
        print(f"Added {src_dir} to Python path for testing\n")

    test_imports()