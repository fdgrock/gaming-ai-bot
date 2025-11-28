#!/usr/bin/env python3
"""Test script to verify CatBoost prediction fix works for both Lotto Max and Lotto 6/49."""

import sys
from pathlib import Path

# Add streamlit_app to path
sys.path.insert(0, str(Path(__file__).parent / "streamlit_app"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import joblib

def test_catboost_predictions():
    """Test CatBoost predictions for both games."""
    
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    features_dir = project_root / "data" / "features"
    
    games = ["lotto_max", "lotto_6_49"]
    results = {}
    
    for game in games:
        print(f"\n{'='*60}")
        print(f"Testing CatBoost for {game.replace('_', '/').upper()}")
        print(f"{'='*60}")
        
        try:
            # 1. Load CatBoost model
            model_path = models_dir / game / "catboost" / "model.pkl"
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                results[game] = "FAILED - Model not found"
                continue
            
            model = joblib.load(model_path)
            print(f"✓ Loaded model from {model_path}")
            
            # 2. Load features for scaling
            features_path = features_dir / "catboost" / game
            if not features_path.exists():
                print(f"❌ Features directory not found: {features_path}")
                results[game] = "FAILED - Features not found"
                continue
            
            csv_files = list(features_path.glob("*.csv"))
            if not csv_files:
                print(f"❌ No CSV features found in {features_path}")
                results[game] = "FAILED - No CSV features"
                continue
            
            X_features = pd.read_csv(csv_files[0])
            numeric_cols = X_features.select_dtypes(include=[np.number]).columns
            X_features = X_features[numeric_cols]
            feature_dim = X_features.shape[1]
            print(f"✓ Loaded {feature_dim} features from {csv_files[0].name}")
            
            # 3. Fit scaler
            scaler = StandardScaler()
            scaler.fit(X_features.values)
            print(f"✓ Fitted StandardScaler")
            
            # 4. Generate random prediction input
            random_input = np.random.randn(1, feature_dim)
            random_input_scaled = scaler.transform(random_input)
            print(f"✓ Created random input with shape {random_input_scaled.shape}")
            
            # 5. Make prediction
            try:
                predictions = model.predict_proba(random_input_scaled)[0]
                print(f"✓ Successfully generated predictions with shape {predictions.shape}")
                print(f"  Prediction scores: {predictions[:5]}...")
                results[game] = "SUCCESS"
            except Exception as pred_err:
                print(f"❌ Prediction failed: {pred_err}")
                results[game] = f"FAILED - {pred_err}"
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results[game] = f"FAILED - {e}"
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for game, result in results.items():
        status = "✓" if result == "SUCCESS" else "❌"
        print(f"{status} {game.replace('_', '/').upper()}: {result}")
    
    success_count = sum(1 for r in results.values() if r == "SUCCESS")
    print(f"\nPassed: {success_count}/{len(results)}")
    
    return all(r == "SUCCESS" for r in results.values())

if __name__ == "__main__":
    success = test_catboost_predictions()
    sys.exit(0 if success else 1)
