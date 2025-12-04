#!/usr/bin/env python3
"""Test that prediction validation works correctly with proper max_number for each game."""

from streamlit_app.core.unified_utils import get_game_config, sanitize_game_name

def validate_prediction_numbers(numbers, max_number=49):
    """Mimic the validation logic from predictions.py"""
    if not isinstance(numbers, (list, tuple)):
        return False
    
    if len(numbers) != 6:
        return False
    
    if len(numbers) != len(set(numbers)):  # Check for duplicates
        return False
    
    if not all(isinstance(n, (int, float)) and 1 <= n <= max_number for n in numbers):
        return False
    
    return True

def test_validation_for_games():
    """Test that validation works with game-specific max_number."""
    
    test_cases = [
        # (game_name, valid_numbers, invalid_numbers)
        ("Lotto Max", [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 51]),  # Lotto Max: max is 50
        ("Lotto 6/49", [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 50]),  # Lotto 6/49: max is 49
        ("Daily Grand", [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 50]),  # Daily Grand: max is 49
        ("Powerball", [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 70]),  # Powerball: max is 69
    ]
    
    print("Testing prediction validation with game-specific max_number...\n")
    
    all_passed = True
    for game_name, valid_nums, invalid_nums in test_cases:
        config = get_game_config(game_name)
        number_range = config.get('number_range', (1, 49))
        max_number = number_range[1] if isinstance(number_range, (tuple, list)) else config.get('max_number', 49)
        
        print(f"🎮 {game_name} (max_number={max_number})")
        
        # Test valid numbers
        valid_result = validate_prediction_numbers(valid_nums, max_number)
        valid_status = "✅" if valid_result else "❌"
        print(f"  {valid_status} Valid numbers {valid_nums}: {valid_result}")
        
        # Test invalid numbers
        invalid_result = validate_prediction_numbers(invalid_nums, max_number)
        invalid_status = "✅" if not invalid_result else "❌"
        print(f"  {invalid_status} Invalid numbers {invalid_nums} (exceeds max): {not invalid_result}")
        
        # Test duplicate detection
        dup_nums = [1, 2, 3, 4, 5, 5]  # Has duplicate
        dup_result = validate_prediction_numbers(dup_nums, max_number)
        dup_status = "✅" if not dup_result else "❌"
        print(f"  {dup_status} Duplicate numbers {dup_nums}: {not dup_result}")
        
        # Test wrong count
        short_nums = [1, 2, 3]  # Only 3 numbers
        short_result = validate_prediction_numbers(short_nums, max_number)
        short_status = "✅" if not short_result else "❌"
        print(f"  {short_status} Wrong count {short_nums}: {not short_result}")
        
        print()
        
        if not (valid_result and not invalid_result and not dup_result and not short_result):
            all_passed = False
    
    if all_passed:
        print("✅ All validation tests passed!")
    else:
        print("❌ Some validation tests failed.")
    
    return all_passed

if __name__ == "__main__":
    success = test_validation_for_games()
    exit(0 if success else 1)
