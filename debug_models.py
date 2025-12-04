#!/usr/bin/env python3
"""Debug script to check model outputs for Lotto 6/49 vs Lotto Max."""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_app.core.unified_utils import get_game_config, sanitize_game_name
from streamlit_app.core import get_models_dir, get_data_dir

def check_game_models(game_name):
    """Check what models exist and their output shapes for a game."""
    print(f"\n{'='*80}")
    print(f"Checking {game_name}")
    print('='*80)
    
    config = get_game_config(game_name)
    game_folder = sanitize_game_name(game_name)
    models_dir = Path(get_models_dir()) / game_folder
    data_dir = Path(get_data_dir())
    
    print(f"\nConfig: {config}")
    print(f"Models directory: {models_dir}")
    print(f"Models directory exists: {models_dir.exists()}")
    
    if not models_dir.exists():
        print(f"❌ Models directory does not exist!")
        return
    
    # Check XGBoost
    print(f"\n--- XGBoost ---")
    xgb_path = models_dir / "xgboost" / f"xgboost_{game_folder}_model.pkl"
    print(f"Path: {xgb_path}")
    print(f"Exists: {xgb_path.exists()}")
    if xgb_path.exists():
        try:
            import joblib
            model = joblib.load(str(xgb_path))
            print(f"Model type: {type(model)}")
            print(f"Model: {model}")
            
            # Check feature files
            feature_files = sorted(list((data_dir / "features" / "xgboost" / game_folder).glob("*.csv")))
            if feature_files:
                df = pd.read_csv(feature_files[-1])
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                print(f"Feature file shape: {df.shape}")
                print(f"Numeric columns: {len(numeric_cols)}")
                
                # Try prediction
                test_input = df.iloc[0:1][numeric_cols]
                try:
                    output = model.predict(test_input)
                    print(f"Prediction output shape: {output.shape}")
                    print(f"Prediction output sample: {output[0][:10]}")
                except Exception as e:
                    print(f"Prediction failed: {e}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Check CatBoost
    print(f"\n--- CatBoost ---")
    cb_path = models_dir / "catboost" / f"catboost_{game_folder}_model.pkl"
    print(f"Path: {cb_path}")
    print(f"Exists: {cb_path.exists()}")
    if cb_path.exists():
        try:
            import joblib
            model = joblib.load(str(cb_path))
            print(f"Model type: {type(model)}")
            
            # Check feature files
            feature_files = sorted(list((data_dir / "features" / "catboost" / game_folder).glob("*.csv")))
            if feature_files:
                df = pd.read_csv(feature_files[-1])
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                print(f"Feature file shape: {df.shape}")
                print(f"Numeric columns: {len(numeric_cols)}")
                
                # Try prediction
                test_input = df.iloc[0:1][numeric_cols]
                try:
                    output = model.predict_proba(test_input)
                    print(f"Prediction proba output shape: {output.shape}")
                    print(f"Prediction proba sample: {output[0][:10]}")
                except Exception as e:
                    print(f"Prediction failed: {e}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Check LightGBM
    print(f"\n--- LightGBM ---")
    lgb_path = models_dir / "lightgbm" / f"lightgbm_{game_folder}_model.pkl"
    print(f"Path: {lgb_path}")
    print(f"Exists: {lgb_path.exists()}")
    if lgb_path.exists():
        try:
            import joblib
            model = joblib.load(str(lgb_path))
            print(f"Model type: {type(model)}")
            
            # Check feature files
            feature_files = sorted(list((data_dir / "features" / "lightgbm" / game_folder).glob("*.csv")))
            if feature_files:
                df = pd.read_csv(feature_files[-1])
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                print(f"Feature file shape: {df.shape}")
                print(f"Numeric columns: {len(numeric_cols)}")
                
                # Try prediction
                test_input = df.iloc[0:1][numeric_cols]
                try:
                    output = model.predict_proba(test_input)
                    print(f"Prediction proba output shape: {output.shape}")
                    print(f"Prediction proba sample: {output[0][:10]}")
                except Exception as e:
                    print(f"Prediction failed: {e}")
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_game_models("Lotto 6/49")
    check_game_models("Lotto Max")
