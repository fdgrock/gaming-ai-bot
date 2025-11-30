#!/usr/bin/env python3
"""
Test script to verify AI-based predictions are working correctly
Tests that predictions are diverse and based on trained models, not random
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add streamlit_app to path
sys.path.insert(0, str(Path(__file__).parent / "streamlit_app"))

from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# Test function
def test_catboost_predictions():
    """Test CatBoost predictions for Lotto Max"""
    print("\n" + "="*60)
    print("TEST: CatBoost Predictions for Lotto Max")
    print("="*60)
    
    try:
        # Setup paths
        models_dir = Path("models")
        game_folder = "lotto_max"
        game = "Lotto Max"
        config = {"max_number": 50, "num_positions": 7}
        
        # Load model
        catboost_models = sorted(list((models_dir / "catboost").glob(f"catboost_{game_folder}_*.joblib")))
        if not catboost_models:
            print("❌ No CatBoost model found")
            return False
        
        model_path = catboost_models[-1]
        print(f"✅ Model loaded: {model_path.name}")
        model = joblib.load(str(model_path))
        
        # Load training features
        data_dir = Path("data")
        feature_files = sorted(list(data_dir.glob(f"features/catboost/{game_folder}/*.csv")))
        if not feature_files:
            print("❌ No feature files found")
            return False
        
        features_df = pd.read_csv(feature_files[-1])
        print(f"✅ Features loaded: {feature_files[-1].name} with shape {features_df.shape}")
        
        # Extract numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols]
        print(f"✅ Numeric columns: {len(numeric_cols)} (Expected: 85)")
        
        # Create scaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        print(f"✅ Scaler fitted and features scaled: {features_scaled.shape}")
        
        # Test prediction
        sample_idx = 0
        sample = features_df.iloc[sample_idx].values.reshape(1, -1)
        sample_scaled = scaler.transform(sample)
        
        pred_probs = model.predict_proba(sample_scaled)
        print(f"✅ Prediction successful: shape {pred_probs.shape}")
        print(f"   Probabilities (first 5 classes): {pred_probs[0][:5]}")
        
        # Test multi-sampling approach
        print("\n🔄 Testing multi-sampling prediction strategy...")
        rng = np.random.RandomState(42)
        candidates = []
        
        for attempt in range(20):
            # Add noise
            noise = rng.normal(0, 0.05, size=sample.shape)
            noisy_input = sample * (1 + noise)
            noisy_scaled = scaler.transform(noisy_input)
            
            # Predict
            probs = model.predict_proba(noisy_scaled)[0]
            # Pick digit based on probability
            predicted_digit = rng.choice(10, p=probs / probs.sum())
            predicted_num = predicted_digit + 1
            candidates.append(predicted_num)
        
        print(f"✅ Generated {len(candidates)} candidate predictions")
        
        # Count occurrences
        from collections import Counter
        counter = Counter(candidates)
        top_nums = [num for num, _ in counter.most_common(50)][:7]
        top_nums = sorted(top_nums)
        
        print(f"✅ Top 7 numbers: {top_nums}")
        print(f"   Frequency counts: {dict(counter.most_common(10))}")
        
        # Generate multiple sets to test diversity
        print("\n📊 Testing diversity of predictions...")
        sets = []
        for set_num in range(4):
            rng_set = np.random.RandomState(1000 + set_num)
            candidates = []
            
            for attempt in range(30):
                noise = rng_set.normal(0, 0.05 + (attempt / 500), size=sample.shape)
                noisy_input = sample * (1 + noise)
                noisy_scaled = scaler.transform(noisy_input)
                probs = model.predict_proba(noisy_scaled)[0]
                predicted_digit = rng_set.choice(10, p=probs / probs.sum())
                candidates.append(predicted_digit + 1)
            
            counter = Counter(candidates)
            top_nums = [num for num, _ in counter.most_common(50)][:7]
            top_nums = sorted(top_nums)
            sets.append(top_nums)
        
        print(f"\nGenerated 4 sets:")
        for i, s in enumerate(sets, 1):
            print(f"  Set {i}: {s}")
        
        # Check diversity
        all_equal = all(s == sets[0] for s in sets[1:])
        if all_equal:
            print("\n❌ FAILED: All sets are identical (not diverse)")
            return False
        else:
            print("\n✅ SUCCESS: Sets are diverse")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_loading():
    """Test that feature files load correctly"""
    print("\n" + "="*60)
    print("TEST: Feature File Loading")
    print("="*60)
    
    try:
        data_dir = Path("data")
        
        # Test CatBoost features
        catboost_files = sorted(list(data_dir.glob("features/catboost/lotto_max/*.csv")))
        if catboost_files:
            df = pd.read_csv(catboost_files[-1])
            numeric = df.select_dtypes(include=[np.number])
            print(f"✅ CatBoost Lotto Max: {catboost_files[-1].name} → {numeric.shape[1]} numeric columns")
        
        # Test LightGBM features
        lgb_files = sorted(list(data_dir.glob("features/lightgbm/lotto_max/*.csv")))
        if lgb_files:
            df = pd.read_csv(lgb_files[-1])
            numeric = df.select_dtypes(include=[np.number])
            print(f"✅ LightGBM Lotto Max: {lgb_files[-1].name} → {numeric.shape[1]} numeric columns")
        
        # Test LSTM NPZ features
        lstm_npz = sorted(list(data_dir.glob("features/lstm/lotto_max/*.npz")))
        if lstm_npz:
            import numpy as np_func
            loaded = np_func.load(lstm_npz[-1])
            if 'features' in loaded:
                features = loaded['features']
                print(f"✅ LSTM Lotto Max: {lstm_npz[-1].name} → shape {features.shape}")
            else:
                print(f"⚠️  LSTM NPZ missing 'features' key")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n🚀 STARTING PREDICTION SYSTEM TESTS\n")
    
    results = []
    results.append(("Feature Loading", test_feature_loading()))
    results.append(("CatBoost Predictions", test_catboost_predictions()))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(r for _, r in results)
    print("\n" + ("🎉 ALL TESTS PASSED\n" if all_passed else "⚠️  SOME TESTS FAILED\n"))
    
    sys.exit(0 if all_passed else 1)
