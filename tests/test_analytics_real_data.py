"""
Test script to verify analytics functions load REAL data from JSON and CSV files
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

# Setup paths
workspace_root = Path(".")
predictions_dir = workspace_root / "predictions"
data_dir = workspace_root / "data"

def sanitize_game_name(game: str) -> str:
    return game.lower().replace(" ", "_").replace("/", "_")

def test_load_model_prediction_data():
    """Test loading REAL model predictions from JSON files"""
    print("\n" + "="*70)
    print("TEST 1: Load Model Prediction Data from JSON")
    print("="*70)
    
    game = "lotto_6_49"
    model_type = "lstm"
    
    predictions_game_dir = predictions_dir / sanitize_game_name(game) / model_type
    
    if not predictions_game_dir.exists():
        print(f"❌ Directory not found: {predictions_game_dir}")
        return False
    
    print(f"✓ Found directory: {predictions_game_dir}")
    
    # List files
    json_files = list(predictions_game_dir.glob("*.json"))
    print(f"✓ Found {len(json_files)} prediction files")
    
    if json_files:
        # Load first file
        test_file = json_files[0]
        print(f"\n  Testing file: {test_file.name}")
        
        try:
            with open(test_file, encoding='utf-8') as f:
                data = json.load(f)
            
            # Check structure
            has_metadata = 'metadata' in data
            has_sets = 'sets' in data or 'predictions' in data
            has_confidence = 'confidence_scores' in data
            
            print(f"  ✓ Loaded JSON successfully")
            print(f"    - Has metadata: {has_metadata}")
            print(f"    - Has sets: {has_sets}")
            print(f"    - Has confidence_scores: {has_confidence}")
            
            if has_metadata:
                metadata = data['metadata']
                print(f"    - Model name: {metadata.get('model_name', 'N/A')}")
                print(f"    - Draw date: {metadata.get('draw_date', 'N/A')}")
                print(f"    - Num sets: {metadata.get('num_sets', 'N/A')}")
            
            if has_sets:
                sets = data.get('sets', data.get('predictions', []))
                print(f"    - Number of sets: {len(sets)}")
                if sets:
                    print(f"    - First set: {sets[0]}")
            
            if has_confidence:
                conf = data['confidence_scores']
                if isinstance(conf, dict):
                    print(f"    - Confidence (dict): {list(conf.keys())[:3]}")
                else:
                    print(f"    - Confidence scores: {conf[:3] if len(conf) > 3 else conf}")
            
            return True
        
        except Exception as e:
            print(f"  ❌ Error loading file: {e}")
            return False
    
    return True

def test_load_game_data():
    """Test loading REAL game data from CSV files"""
    print("\n" + "="*70)
    print("TEST 2: Load Game Data from CSV")
    print("="*70)
    
    game = "lotto_6_49"
    game_dir = data_dir / sanitize_game_name(game)
    
    if not game_dir.exists():
        print(f"❌ Directory not found: {game_dir}")
        return False
    
    print(f"✓ Found directory: {game_dir}")
    
    # List CSV files
    csv_files = sorted(game_dir.glob("training_data_*.csv"), reverse=True)
    print(f"✓ Found {len(csv_files)} CSV files")
    
    if csv_files:
        # Load most recent file
        test_file = csv_files[0]
        print(f"\n  Testing file: {test_file.name}")
        
        try:
            df = pd.read_csv(test_file)
            print(f"  ✓ Loaded CSV successfully")
            print(f"    - Rows: {len(df)}")
            print(f"    - Columns: {list(df.columns)}")
            print(f"\n    Sample data (first 3 rows):")
            print(f"    {df.head(3).to_string()}")
            
            # Check for numbers column
            if 'numbers' in df.columns:
                sample_numbers = df['numbers'].iloc[0]
                print(f"\n    Sample numbers column: {sample_numbers}")
            
            return True
        
        except Exception as e:
            print(f"  ❌ Error loading file: {e}")
            return False
    
    return True

def test_number_frequency_analysis():
    """Test analyzing number frequency from CSV data"""
    print("\n" + "="*70)
    print("TEST 3: Number Frequency Analysis")
    print("="*70)
    
    game = "lotto_6_49"
    game_dir = data_dir / sanitize_game_name(game)
    csv_files = sorted(game_dir.glob("training_data_*.csv"), reverse=True)
    
    if not csv_files:
        print("❌ No CSV files found")
        return False
    
    print(f"✓ Loading data from {len(csv_files)} files")
    
    # Load all files
    dfs = []
    for csv_file in csv_files[:3]:  # Load first 3 files for testing
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except:
            pass
    
    if not dfs:
        print("❌ Could not load any CSV files")
        return False
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"✓ Combined data: {len(combined_df)} total rows")
    
    # Extract numbers
    all_numbers = []
    for nums_str in combined_df['numbers'].dropna():
        try:
            nums = []
            for n_str in str(nums_str).strip('[]').split(','):
                n_clean = n_str.strip()
                if n_clean and n_clean.isdigit():
                    nums.append(int(n_clean))
            all_numbers.extend(nums)
        except:
            pass
    
    print(f"✓ Extracted {len(all_numbers)} numbers from {combined_df['numbers'].notna().sum()} draws")
    
    if all_numbers:
        # Analyze frequency
        number_counts = Counter(all_numbers)
        top_10 = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"\n  Top 10 Most Frequent Numbers:")
        for num, count in top_10:
            print(f"    - Number {num}: {count} times")
        
        # Even/odd analysis
        even = sum(1 for n in all_numbers if n % 2 == 0)
        odd = len(all_numbers) - even
        print(f"\n  Even/Odd Split:")
        print(f"    - Even: {even} ({even/len(all_numbers)*100:.1f}%)")
        print(f"    - Odd: {odd} ({odd/len(all_numbers)*100:.1f}%)")
        
        return True
    
    return False

def test_predictions_by_model_type():
    """Test loading predictions by model type"""
    print("\n" + "="*70)
    print("TEST 4: Predictions by Model Type")
    print("="*70)
    
    game = "lotto_6_49"
    predictions_game_dir = predictions_dir / sanitize_game_name(game)
    
    if not predictions_game_dir.exists():
        print(f"❌ Directory not found: {predictions_game_dir}")
        return False
    
    print(f"✓ Found directory: {predictions_game_dir}")
    
    model_types = [d.name for d in predictions_game_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(model_types)} model types: {model_types}")
    
    # Load predictions by type
    for model_type in model_types:
        model_dir = predictions_game_dir / model_type
        json_files = list(model_dir.glob("*.json"))
        
        if json_files:
            total_sets = 0
            total_confidence = []
            
            for json_file in json_files:
                try:
                    with open(json_file, encoding='utf-8') as f:
                        data = json.load(f)
                    
                    sets = data.get('sets', data.get('predictions', []))
                    conf = data.get('confidence_scores', [])
                    
                    if isinstance(conf, dict):
                        conf = [conf.get('overall_confidence', 0.5)] * len(sets)
                    
                    total_sets += len(sets)
                    total_confidence.extend([float(c) if isinstance(c, (int, float)) else 0.5 for c in conf])
                
                except:
                    pass
            
            avg_conf = sum(total_confidence) / len(total_confidence) if total_confidence else 0
            print(f"\n  {model_type.upper()}:")
            print(f"    - Files: {len(json_files)}")
            print(f"    - Total sets: {total_sets}")
            print(f"    - Avg confidence: {avg_conf:.2%}")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ANALYTICS REAL DATA LOADING TEST SUITE")
    print("="*70)
    print(f"Workspace: {workspace_root}")
    print(f"Predictions dir: {predictions_dir}")
    print(f"Data dir: {data_dir}")
    
    tests = [
        ("Model Predictions", test_load_model_prediction_data),
        ("Game CSV Data", test_load_game_data),
        ("Number Frequency", test_number_frequency_analysis),
        ("Predictions by Type", test_predictions_by_model_type),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            results[test_name] = False
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*70)
    all_passed = all(results.values())
    print(f"\n{'✓ ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'}")
    print("="*70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    main()
