"""
Debug script to understand what the models actually output and why predictions cluster around 1-10
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'streamlit_app'))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
import tensorflow as tf

# Setup paths
project_root = Path(__file__).parent
data_dir = project_root / 'data'
models_dir = project_root / 'models'

def analyze_model_output():
    """Analyze what different model types output"""
    
    # Test with Lotto Max as example
    game = 'Lotto Max'
    game_folder = 'lotto_max'
    max_number = 49
    main_nums = 7
    
    print("=" * 80)
    print(f"ANALYZING MODEL OUTPUTS FOR {game}")
    print("=" * 80)
    
    # 1. Check XGBoost model structure
    print("\n1. XGBoost Model Analysis")
    print("-" * 80)
    xgb_models = sorted(list((models_dir / 'xgboost').glob(f'xgboost_{game_folder}_*.joblib')))
    if xgb_models:
        xgb_model = joblib.load(str(xgb_models[-1]))
        print(f"   Loaded: {xgb_models[-1].name}")
        
        # Check model properties
        if hasattr(xgb_model, 'n_classes_'):
            print(f"   Number of classes: {xgb_model.n_classes_}")
        if hasattr(xgb_model, 'classes_'):
            print(f"   Classes: {xgb_model.classes_}")
        
        # Generate test input
        feature_dim = 128  # Typical feature dimension
        test_input = np.random.randn(1, feature_dim)
        
        # Get predictions
        try:
            proba = xgb_model.predict_proba(test_input)
            print(f"   predict_proba() output shape: {proba.shape}")
            print(f"   predict_proba() sample values: {proba[0][:10]}")
            print(f"   Max value: {np.max(proba)}, Min value: {np.min(proba)}")
            
            # What do top indices represent?
            top_indices = np.argsort(proba[0])[-main_nums:]
            print(f"   Top {main_nums} indices: {sorted(top_indices)}")
            print(f"   Top {main_nums} as numbers (index+1): {sorted(top_indices + 1)}")
            
        except Exception as e:
            print(f"   Error getting predictions: {e}")
    else:
        print("   No XGBoost model found")
    
    # 2. Check LSTM model structure
    print("\n2. LSTM Model Analysis")
    print("-" * 80)
    lstm_models = sorted(list((models_dir / 'lstm').glob(f'lstm_{game_folder}_*.keras')))
    if lstm_models:
        lstm_model = tf.keras.models.load_model(str(lstm_models[-1]))
        print(f"   Loaded: {lstm_models[-1].name}")
        print(f"   Model input shape: {lstm_model.input_shape}")
        print(f"   Model output shape: {lstm_model.output_shape}")
        
        # Generate test input (typically (1, 25, 45) for LSTM)
        input_shape = lstm_model.input_shape
        if len(input_shape) == 3:
            test_input = np.random.randn(input_shape[0], input_shape[1], input_shape[2])
        else:
            test_input = np.random.randn(*input_shape)
        
        # Get predictions
        pred = lstm_model.predict(test_input, verbose=0)
        print(f"   predict() output shape: {pred.shape}")
        print(f"   predict() sample values: {pred[0][:10]}")
        print(f"   Max value: {np.max(pred)}, Min value: {np.min(pred)}")
        
        # What do top indices represent?
        if len(pred.shape) > 1:
            top_indices = np.argsort(pred[0])[-main_nums:]
            print(f"   Top {main_nums} indices: {sorted(top_indices)}")
            print(f"   Top {main_nums} as numbers (index+1): {sorted(top_indices + 1)}")
    else:
        print("   No LSTM model found")
    
    # 3. Check training data to understand what model was trained on
    print("\n3. Training Data Analysis")
    print("-" * 80)
    
    # Load training features to understand dimensions
    feature_files = list((data_dir / 'features' / 'xgboost' / game_folder).glob('*.csv'))
    if feature_files:
        df = pd.read_csv(feature_files[0])
        print(f"   Sample XGBoost training features shape: {df.shape}")
        print(f"   Feature columns sample: {list(df.columns[:10])}")
        print(f"   Data sample:\n{df.head(3).to_string()}")
    else:
        print(f"   No training features found in {data_dir / 'features' / 'xgboost' / game_folder}")
    
    # Check historical draws to understand labels
    historical_files = list((data_dir / 'historical_draws').glob(f'*{game_folder}*'))
    if not historical_files:
        historical_files = list((data_dir / 'draws').glob(f'*{game_folder}*'))
    
    if historical_files:
        df_draws = pd.read_csv(historical_files[0])
        print(f"\n   Historical draws shape: {df_draws.shape}")
        print(f"   Draw sample:\n{df_draws.head(3).to_string()}")
        
        # Analyze what numbers appear where
        if 'Number 1' in df_draws.columns:
            # Multi-column format
            num_cols = [col for col in df_draws.columns if 'Number' in col]
            print(f"\n   Number distribution (if multi-column format):")
            for col in num_cols:
                print(f"     {col}: min={df_draws[col].min()}, max={df_draws[col].max()}, mean={df_draws[col].mean():.1f}")
        else:
            # Potentially single-column format
            print(f"\n   Column info: {list(df_draws.columns)}")
    else:
        print("   No historical draws found")
    
    # 4. Check what the prediction code is currently doing
    print("\n4. Current Prediction Code Logic")
    print("-" * 80)
    print("   The code does:")
    print("   1. Load model")
    print("   2. Generate random input (or sample from training data)")
    print("   3. Call model.predict() or model.predict_proba()")
    print("   4. Get top N indices: np.argsort(pred_probs)[-main_nums:]")
    print("   5. Convert to numbers: (indices + 1)")
    print("   6. THIS ASSUMES: indices 0-48 map to numbers 1-49")
    print("   7. BUT PROBLEM: If model outputs probability for EACH POSITION in set,")
    print("      then taking top 6-7 is WRONG - it's like averaging all positions together")

if __name__ == '__main__':
    analyze_model_output()
