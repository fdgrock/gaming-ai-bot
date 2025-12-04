#!/usr/bin/env python3
"""Test to verify prediction validation fix for both games."""

import numpy as np
from streamlit_app.pages.predictions import _validate_prediction_numbers
from streamlit_app.core.unified_utils import get_game_config, sanitize_game_name

def test_validation_fix():
    """Test that validation now works correctly for both games."""
    print("\n" + "="*80)
    print("PREDICTION VALIDATION FIX TEST")
    print("="*80)
    
    # Test Lotto 6/49
    print("\n🎮 LOTTO 6/49")
    config_649 = get_game_config("Lotto 6/49")
    max_num_649 = config_649['number_range'][1]
    main_nums_649 = config_649['main_numbers']
    
    print(f"   Max number: {max_num_649}, Main numbers: {main_nums_649}")
    
    # Valid Lotto 6/49 set
    valid_649 = [1, 10, 25, 35, 45, 49]
    result = _validate_prediction_numbers(valid_649, max_num_649, main_nums_649)
    print(f"   Valid set {valid_649}: {result} {'✅' if result else '❌'}")
    
    # Invalid - too many numbers
    invalid_649_count = [1, 2, 3, 4, 5, 6, 7]
    result = _validate_prediction_numbers(invalid_649_count, max_num_649, main_nums_649)
    print(f"   Too many numbers {invalid_649_count}: {not result} {'✅' if not result else '❌'}")
    
    # Invalid - too few numbers
    invalid_649_few = [1, 2, 3, 4, 5]
    result = _validate_prediction_numbers(invalid_649_few, max_num_649, main_nums_649)
    print(f"   Too few numbers {invalid_649_few}: {not result} {'✅' if not result else '❌'}")
    
    # Test Lotto Max
    print("\n🎮 LOTTO MAX")
    config_max = get_game_config("Lotto Max")
    max_num_max = config_max['number_range'][1]
    main_nums_max = config_max['main_numbers']
    
    print(f"   Max number: {max_num_max}, Main numbers: {main_nums_max}")
    
    # Valid Lotto Max set (7 numbers)
    valid_max = [1, 10, 25, 35, 45, 48, 50]
    result = _validate_prediction_numbers(valid_max, max_num_max, main_nums_max)
    print(f"   Valid set {valid_max}: {result} {'✅' if result else '❌'}")
    
    # Invalid - only 6 numbers (was working before, should fail now)
    invalid_max_6 = [1, 10, 25, 35, 45, 48]
    result = _validate_prediction_numbers(invalid_max_6, max_num_max, main_nums_max)
    print(f"   Only 6 numbers {invalid_max_6}: {not result} {'✅' if not result else '❌'}")
    
    # Invalid - 8 numbers
    invalid_max_8 = [1, 10, 25, 35, 45, 48, 50, 2]
    result = _validate_prediction_numbers(invalid_max_8, max_num_max, main_nums_max)
    print(f"   Too many (8) numbers {invalid_max_8}: {not result} {'✅' if not result else '❌'}")
    
    # Duplicate in Lotto Max set
    invalid_max_dup = [1, 10, 25, 35, 45, 48, 48]
    result = _validate_prediction_numbers(invalid_max_dup, max_num_max, main_nums_max)
    print(f"   With duplicate {invalid_max_dup}: {not result} {'✅' if not result else '❌'}")
    
    # Out of range
    invalid_max_range = [1, 10, 25, 35, 45, 48, 51]
    result = _validate_prediction_numbers(invalid_max_range, max_num_max, main_nums_max)
    print(f"   Out of range {invalid_max_range}: {not result} {'✅' if not result else '❌'}")
    
    print("\n" + "="*80)
    print("✅ All validation tests completed!")
    print("="*80)

if __name__ == "__main__":
    test_validation_fix()
