#!/usr/bin/env python3
"""Trace exactly what _generate_predictions returns for Lotto 6/49."""

from streamlit_app.core.unified_utils import get_game_config, sanitize_game_name
from streamlit_app.core import get_models_dir
import sys

def test_generate_for_game(game_name, num_sets=4):
    """Call the generation function and see what it returns."""
    
    print(f"\n{'='*80}")
    print(f"Testing {game_name}")
    print('='*80)
    
    config = get_game_config(game_name)
    max_num = config['number_range'][1]
    main_nums = config['main_numbers']
    
    print(f"Game config: max_number={max_num}, main_nums={main_nums}")
    
    # Try importing the generation function
    try:
        from streamlit_app.pages.predictions import _generate_predictions
        print(f"OK - Imported _generate_predictions")
    except Exception as e:
        print(f"ERROR - Failed to import: {e}")
        return
    
    # Try to generate predictions
    try:
        print(f"\nGenerating {num_sets} predictions using XGBoost model...")
        result = _generate_predictions(
            game=game_name,
            count=num_sets,
            mode="single_model",
            confidence_threshold=0.5,
            model_type="XGBoost",
            model_name="xgboost_model"
        )
        
        if result:
            if 'error' in result:
                print(f"ERROR - Generation error: {result['error']}")
                return
            
            print(f"\nOK - Generation succeeded!")
            print(f"Number of sets: {len(result.get('sets', []))}")
            print(f"Confidence scores: {result.get('confidence_scores', [])}")
            
            sets = result.get('sets', [])
            confidence_scores = result.get('confidence_scores', [])
            
            for i, (numbers, confidence) in enumerate(zip(sets, confidence_scores)):
                print(f"\n  Set {i+1}: {numbers}")
                print(f"    Confidence: {confidence:.1%}")
                
                # Validate
                if len(numbers) != main_nums:
                    print(f"    WARNING - Wrong count! Expected {main_nums}, got {len(numbers)}")
                if any(n < 1 or n > max_num for n in numbers):
                    print(f"    WARNING - Out of range! Some numbers not in 1-{max_num}")
                if len(numbers) != len(set(numbers)):
                    print(f"    WARNING - Duplicates found!")
                    
        else:
            print(f"ERROR - Generation returned None")
            
    except Exception as e:
        print(f"ERROR - Exception during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generate_for_game("Lotto 6/49", 4)
    test_generate_for_game("Lotto Max", 4)
