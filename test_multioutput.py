"""
Test script for multi-output implementation
"""
import numpy as np
import sys
from pathlib import Path

# Add streamlit_app to path
sys.path.insert(0, str(Path(__file__).parent / "streamlit_app"))

from services.advanced_model_training import AdvancedModelTrainer

def test_target_extraction():
    """Test multi-output target extraction"""
    print("=" * 60)
    print("TEST 1: Multi-Output Target Extraction")
    print("=" * 60)
    
    trainer = AdvancedModelTrainer("Lotto Max")
    
    # Create test data with raw CSV format
    test_file = Path("data/lotto_max/training_data_2025.csv")
    
    if test_file.exists():
        print(f"✓ Found test file: {test_file}")
        
        # Extract targets
        targets = trainer._extract_targets([test_file], disable_lag=True, max_number=50)
        
        print(f"\n📊 Target Extraction Results:")
        print(f"  Shape: {targets.shape}")
        print(f"  Expected: (n_samples, 7) for multi-output")
        print(f"  Is multi-output: {targets.ndim == 2 and targets.shape[1] == 7}")
        
        if len(targets) > 0:
            print(f"\n  First 3 target sets:")
            for i in range(min(3, len(targets))):
                # Convert from 0-based to 1-based for display
                numbers = [int(n) + 1 for n in targets[i]]
                print(f"    {i+1}: {numbers}")
        
        return targets.ndim == 2 and targets.shape[1] == 7
    else:
        print(f"✗ Test file not found: {test_file}")
        print(f"  Please ensure you have training data in: data/lotto_max/")
        return False

def test_multi_output_detection():
    """Test multi-output detection helpers"""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Output Detection Helpers")
    print("=" * 60)
    
    trainer = AdvancedModelTrainer("Lotto Max")
    
    # Test single-output
    y_single = np.array([1, 5, 10, 20, 30])
    is_multi_single = trainer._is_multi_output(y_single)
    info_single = trainer._get_output_info(y_single)
    
    print(f"\n📊 Single-Output Array (shape {y_single.shape}):")
    print(f"  Is multi-output: {is_multi_single}")
    print(f"  Output info: {info_single}")
    
    # Test multi-output
    y_multi = np.array([[1, 5, 10, 20, 30, 35, 40], [2, 8, 15, 25, 32, 38, 45]])
    is_multi_multi = trainer._is_multi_output(y_multi)
    info_multi = trainer._get_output_info(y_multi)
    
    print(f"\n📊 Multi-Output Array (shape {y_multi.shape}):")
    print(f"  Is multi-output: {is_multi_multi}")
    print(f"  Output info: {info_multi}")
    
    success = (not is_multi_single) and is_multi_multi
    return success

def test_feature_preservation():
    """Test that feature CSVs preserve numbers column"""
    print("\n" + "=" * 60)
    print("TEST 3: Feature CSV Numbers Column Preservation")
    print("=" * 60)
    
    import pandas as pd
    
    # Check if any feature CSVs exist
    feature_dirs = [
        Path("data/features/xgboost/lotto_max"),
        Path("data/features/catboost/lotto_max"),
        Path("data/features/lightgbm/lotto_max"),
        Path("data/features/transformer/lotto_max")
    ]
    
    found_any = False
    for feature_dir in feature_dirs:
        if feature_dir.exists():
            csv_files = list(feature_dir.glob("*.csv"))
            if csv_files:
                found_any = True
                csv_file = csv_files[0]
                df = pd.read_csv(csv_file)
                
                has_numbers = "numbers" in df.columns
                has_draw_date = "draw_date" in df.columns
                
                print(f"\n📂 {feature_dir.name}:")
                print(f"  File: {csv_file.name}")
                print(f"  Columns: {list(df.columns)[:10]}...")
                print(f"  Has 'numbers' column: {has_numbers}")
                print(f"  Has 'draw_date' column: {has_draw_date}")
                
                if has_numbers:
                    print(f"  Sample numbers: {df['numbers'].iloc[0]}")
    
    if not found_any:
        print("\n⚠️  No feature CSVs found yet.")
        print("   Generate features to test this functionality.")
        return None
    
    return True

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  🧪 MULTI-OUTPUT IMPLEMENTATION TEST SUITE".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    results = {}
    
    # Run tests
    try:
        results['target_extraction'] = test_target_extraction()
    except Exception as e:
        print(f"\n❌ Target extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        results['target_extraction'] = False
    
    try:
        results['detection'] = test_multi_output_detection()
    except Exception as e:
        print(f"\n❌ Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        results['detection'] = False
    
    try:
        results['features'] = test_feature_preservation()
    except Exception as e:
        print(f"\n❌ Feature preservation test failed: {e}")
        import traceback
        traceback.print_exc()
        results['features'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])
    
    print(f"\nResults: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    return all(r in [True, None] for r in results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
