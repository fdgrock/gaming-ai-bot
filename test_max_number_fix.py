#!/usr/bin/env python3
"""Test that max_number is correctly extracted for each game."""

from streamlit_app.core.unified_utils import get_game_config

def test_max_number_extraction():
    """Test that max_number is properly extracted from game configs."""
    
    test_cases = [
        ("Lotto Max", 50),
        ("Lotto 6/49", 49),
        ("Daily Grand", 49),
        ("Powerball", 69),
        ("Mega Millions", 70),
        ("Euromillions", 50),
    ]
    
    print("Testing max_number extraction from game configs...\n")
    
    all_passed = True
    for game_name, expected_max in test_cases:
        config = get_game_config(game_name)
        
        # Extract max_number using the SAME logic as predictions.py
        number_range = config.get('number_range', (1, 49))
        max_number = number_range[1] if isinstance(number_range, (tuple, list)) else config.get('max_number', 49)
        
        passed = max_number == expected_max
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"{status}: {game_name}")
        print(f"  Config: {config}")
        print(f"  Expected max_number: {expected_max}")
        print(f"  Extracted max_number: {max_number}")
        print()
        
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✅ All tests passed! max_number extraction works correctly for all games.")
    else:
        print("❌ Some tests failed. Fix the configuration issues.")
    
    return all_passed

if __name__ == "__main__":
    success = test_max_number_extraction()
    exit(0 if success else 1)
