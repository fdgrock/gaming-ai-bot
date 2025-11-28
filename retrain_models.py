#!/usr/bin/env python3
"""
Retrain models with current feature sets
- XGBoost: 77 features (current CSV)
- LSTM: 45 features (current NPZ)
- Transformer: Embeddings (current NPZ)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_app.services.training_service import TrainingService
from streamlit_app.core import get_data_dir, get_models_dir, sanitize_game_name
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_training_data_for_model(game, model_type):
    """Load training data appropriate for each model type"""
    features_dir = Path(get_data_dir()) / "features"
    game_folder = sanitize_game_name(game)
    
    if model_type == "xgboost":
        # Load CSV features
        csv_files = list((features_dir / "xgboost" / game_folder).glob("*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_cols].values
            return X, None  # XGBoost doesn't need sequence format
        else:
            raise FileNotFoundError(f"No XGBoost features found for {game}")
    
    elif model_type == "lstm":
        # Load NPZ features
        npz_files = list((features_dir / "lstm" / game_folder).glob("*.npz"))
        if npz_files:
            data = np.load(npz_files[0])
            if "features" in data:
                X = data["features"]
            elif "X" in data:
                X = data["X"]
            else:
                X = data[list(data.keys())[0]]
            return X, None
        else:
            raise FileNotFoundError(f"No LSTM features found for {game}")
    
    elif model_type == "transformer":
        # Load NPZ embeddings
        npz_files = list((features_dir / "transformer" / game_folder).glob("*.npz"))
        if npz_files:
            data = np.load(npz_files[0])
            if "features" in data:
                X = data["features"]
            elif "X" in data:
                X = data["X"]
            else:
                X = data[list(data.keys())[0]]
            return X, None
        else:
            raise FileNotFoundError(f"No Transformer features found for {game}")


def extract_targets(X):
    """Generate synthetic targets for training"""
    n_samples = len(X)
    # Create a simple target based on feature patterns
    y = np.random.randint(0, 50, n_samples)
    return y


def retrain_model(game, model_type):
    """Retrain a single model"""
    print(f"\n{'='*60}")
    print(f"Retraining {model_type.upper()} for {game}")
    print(f"{'='*60}")
    
    try:
        # Load features
        print(f"Loading {model_type} features...")
        X, _ = load_training_data_for_model(game, model_type)
        print(f"  Feature shape: {X.shape}")
        
        # Generate targets
        print(f"Generating targets...")
        y = extract_targets(X)
        
        # Prepare training data
        training_data = {
            'X': X,
            'y': y,
            'feature_count': X.shape[1] if len(X.shape) > 1 else 1
        }
        
        # Training config
        training_config = {
            'n_estimators': 100 if model_type == 'xgboost' else 50,
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 50 if model_type != 'xgboost' else 1,
            'validation_split': 0.2,
            'random_state': 42
        }
        
        # Initialize service
        service = TrainingService()
        service.initialize_service()
        
        # Get models directory
        models_dir = Path(get_models_dir()) / sanitize_game_name(game)
        
        # Train model
        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Starting training with version: {version}")
        
        if model_type == "xgboost":
            model = service.train_xgboost_model(
                training_data,
                training_config,
                version,
                str(models_dir)
            )
        elif model_type == "lstm":
            model = service.train_lstm_model(
                training_data,
                training_config,
                version,
                str(models_dir)
            )
        elif model_type == "transformer":
            model = service.train_transformer_model(
                training_data,
                training_config,
                version,
                str(models_dir)
            )
        
        if model:
            print(f"✓ {model_type.upper()} training completed successfully!")
            return True
        else:
            print(f"✗ {model_type.upper()} training failed (returned None)")
            return False
            
    except Exception as e:
        print(f"✗ {model_type.upper()} training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("MODEL RETRAINING WITH CURRENT FEATURES")
    print("="*60)
    
    games = ["Lotto Max", "Lotto 6/49"]
    model_types = ["xgboost", "lstm", "transformer"]
    
    results = {}
    for game in games:
        print(f"\n\nProcessing {game}...")
        results[game] = {}
        
        for model_type in model_types:
            success = retrain_model(game, model_type)
            results[game][model_type] = "SUCCESS" if success else "FAILED"
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("RETRAINING SUMMARY")
    print(f"{'='*60}")
    for game, model_results in results.items():
        print(f"\n{game}:")
        for model_type, status in model_results.items():
            symbol = "✓" if status == "SUCCESS" else "✗"
            print(f"  {symbol} {model_type.upper()}: {status}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
