#!/usr/bin/env python3
"""Debug script to trace exact numbers being predicted for Lotto 6/49."""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from streamlit_app.core.unified_utils import get_game_config, sanitize_game_name
from streamlit_app.core import get_models_dir, get_data_dir

def test_lotto_649():
    """Test what Lotto 6/49 XGBoost model actually predicts."""
    print("\n" + "="*80)
    print("LOTTO 6/49 MODEL PREDICTION TEST")
    print("="*80)
    
    game = "Lotto 6/49"
    config = get_game_config(game)
    game_folder = sanitize_game_name(game)
    max_number = config['number_range'][1]
    main_nums = config['main_numbers']
    
    print(f"Game: {game}")
    print(f"Max number: {max_number}")
    print(f"Main numbers: {main_nums}")
    print(f"Config: {config}")
    
    # Load model
    models_dir = Path(get_models_dir()) / game_folder
    xgb_files = sorted(list((models_dir / "xgboost").glob("xgboost_*.joblib")))
    
    if not xgb_files:
        print("❌ No XGBoost model found!")
        return
    
    model_path = xgb_files[-1]
    print(f"\nLoading model: {model_path.name}")
    model = joblib.load(str(model_path))
    
    # Load feature data
    data_dir = Path(get_data_dir())
    feature_files = sorted(list((data_dir / "features" / "xgboost" / game_folder).glob("*.csv")))
    
    if not feature_files:
        print("❌ No feature files found!")
        return
    
    features_df = pd.read_csv(feature_files[-1])
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    print(f"Feature file: {feature_files[-1].name}")
    print(f"Features shape: {features_df.shape}")
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Get predictions for first 5 samples
    print(f"\n--- Model Output Analysis ---\n")
    
    for i in range(min(5, len(features_df))):
        sample = features_df.iloc[[i]][numeric_cols]
        print(f"\nSample {i+1}:")
        print(f"  Input shape: {sample.shape}")
        
        try:
            pred_proba = model.predict_proba(sample)
            print(f"  Prediction shape: {pred_proba.shape}")
            print(f"  Classes found: {len(pred_proba[0])}")
            
            # Get top 6 indices
            top_indices = np.argsort(pred_proba[0])[-main_nums:]
            top_numbers = sorted((top_indices + 1).tolist())
            print(f"  Top indices: {top_indices}")
            print(f"  Generated numbers (index + 1): {top_numbers}")
            
            # Check validation
            valid = all(1 <= n <= max_number for n in top_numbers)
            print(f"  Valid (all 1-{max_number})? {valid}")
            
        except Exception as e:
            print(f"  Error: {e}")

def test_lotto_max():
    """Test what Lotto Max XGBoost model actually predicts."""
    print("\n" + "="*80)
    print("LOTTO MAX MODEL PREDICTION TEST")
    print("="*80)
    
    game = "Lotto Max"
    config = get_game_config(game)
    game_folder = sanitize_game_name(game)
    max_number = config['number_range'][1]
    main_nums = config['main_numbers']
    
    print(f"Game: {game}")
    print(f"Max number: {max_number}")
    print(f"Main numbers: {main_nums}")
    print(f"Config: {config}")
    
    # Load model
    models_dir = Path(get_models_dir()) / game_folder
    xgb_files = sorted(list((models_dir / "xgboost").glob("xgboost_*.joblib")))
    
    if not xgb_files:
        print("❌ No XGBoost model found!")
        return
    
    model_path = xgb_files[-1]
    print(f"\nLoading model: {model_path.name}")
    model = joblib.load(str(model_path))
    
    # Load feature data
    data_dir = Path(get_data_dir())
    feature_files = sorted(list((data_dir / "features" / "xgboost" / game_folder).glob("*.csv")))
    
    if not feature_files:
        print("❌ No feature files found!")
        return
    
    features_df = pd.read_csv(feature_files[-1])
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    print(f"Feature file: {feature_files[-1].name}")
    print(f"Features shape: {features_df.shape}")
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Get predictions for first 5 samples
    print(f"\n--- Model Output Analysis ---\n")
    
    for i in range(min(5, len(features_df))):
        sample = features_df.iloc[[i]][numeric_cols]
        print(f"Sample {i+1}:")
        print(f"  Input shape: {sample.shape}")
        
        try:
            pred_proba = model.predict_proba(sample)
            print(f"  Prediction shape: {pred_proba.shape}")
            print(f"  Classes found: {len(pred_proba[0])}")
            
            # Get top 7 indices (for Lotto Max)
            top_indices = np.argsort(pred_proba[0])[-main_nums:]
            top_numbers = sorted((top_indices + 1).tolist())
            print(f"  Top indices: {top_indices}")
            print(f"  Generated numbers (index + 1): {top_numbers}")
            
            # Check validation
            valid = all(1 <= n <= max_number for n in top_numbers)
            print(f"  Valid (all 1-{max_number})? {valid}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_lotto_649()
    test_lotto_max()
