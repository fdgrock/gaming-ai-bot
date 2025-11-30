"""
Diagnostic script to identify why predictions are showing 50% confidence.

This script helps identify the root cause:
1. Are models being loaded correctly?
2. Are training features being found and loaded?
3. What probability values do models output?
4. Is confidence calculation correct?
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "streamlit_app"))

from core.unified_utils import (
    get_models_dir, get_data_dir, get_predictions_dir,
    sanitize_game_name, get_game_config
)

def analyze_predictions_by_game():
    """Analyze actual prediction files to see what confidence values were saved."""
    print("\n" + "="*80)
    print("ANALYZING SAVED PREDICTIONS")
    print("="*80)
    
    predictions_dir = get_predictions_dir()
    
    if not predictions_dir.exists():
        print(f"❌ Predictions directory does not exist: {predictions_dir}")
        return
    
    # Find all prediction files
    pred_files = list(predictions_dir.rglob("*.json"))
    print(f"\n📁 Found {len(pred_files)} prediction files")
    
    if len(pred_files) == 0:
        print("⚠️  No prediction files found. Generate some predictions first.")
        return
    
    # Analyze most recent predictions by game
    games_data = {}
    
    for pred_file in pred_files[-20:]:  # Last 20 files
        try:
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            
            game = pred_data.get('game', 'unknown')
            conf_scores = pred_data.get('confidence_scores', [])
            model_type = pred_data.get('model_type', 'unknown')
            sets = pred_data.get('sets', [])
            
            if game not in games_data:
                games_data[game] = []
            
            games_data[game].append({
                'file': pred_file.name,
                'model_type': model_type,
                'num_sets': len(sets),
                'conf_scores': conf_scores,
                'avg_conf': np.mean(conf_scores) if conf_scores else None,
                'min_conf': np.min(conf_scores) if conf_scores else None,
                'max_conf': np.max(conf_scores) if conf_scores else None,
            })
        except Exception as e:
            print(f"⚠️  Could not read {pred_file.name}: {e}")
    
    # Display results
    for game, preds in games_data.items():
        print(f"\n🎮 Game: {game}")
        print(f"   Last {len(preds)} predictions:")
        for i, pred in enumerate(preds[-5:], 1):  # Show last 5
            conf_text = f"{pred['avg_conf']:.2%}" if pred['avg_conf'] is not None else "N/A"
            conf_range = f"[{pred['min_conf']:.2%} - {pred['max_conf']:.2%}]" if pred['avg_conf'] is not None else ""
            print(f"   {i}. {pred['model_type']:12} | {pred['num_sets']} sets | Avg Conf: {conf_text} {conf_range}")
        
        # Check if all are 50%
        avg_confs = [p['avg_conf'] for p in preds if p['avg_conf'] is not None]
        if avg_confs:
            all_same = all(abs(c - 0.5) < 0.01 for c in avg_confs)
            if all_same:
                print(f"   ⚠️  ALL PREDICTIONS showing ~50% confidence!")
            else:
                print(f"   ✓ Varied confidence values: {[f'{c:.1%}' for c in avg_confs[-5:]]}")

def check_training_features():
    """Check if training features exist for all models and games."""
    print("\n" + "="*80)
    print("CHECKING TRAINING FEATURES")
    print("="*80)
    
    data_dir = get_data_dir()
    features_dir = data_dir / "features"
    
    if not features_dir.exists():
        print(f"❌ Features directory does not exist: {features_dir}")
        return
    
    print(f"\n📁 Features directory: {features_dir}")
    
    # List model types and games with features
    model_types = [d.name for d in features_dir.iterdir() if d.is_dir()]
    print(f"\n📊 Model types with features: {model_types}")
    
    for model_type in sorted(model_types):
        model_dir = features_dir / model_type
        games = [d.name for d in model_dir.iterdir() if d.is_dir()]
        print(f"\n   {model_type.upper()}:")
        for game in sorted(games):
            game_dir = model_dir / game
            files = list(game_dir.glob("*"))
            if files:
                latest_file = sorted(files)[-1]
                file_size = latest_file.stat().st_size / 1024  # KB
                
                # Try to get shape if CSV
                if latest_file.suffix == '.csv':
                    try:
                        df = pd.read_csv(latest_file, nrows=100)
                        shape = f"shape={df.shape}"
                    except:
                        shape = f"size={file_size:.1f}KB"
                else:
                    shape = f"size={file_size:.1f}KB"
                
                print(f"      {game}: {latest_file.name} ({shape})")
            else:
                print(f"      {game}: ❌ NO FEATURES FOUND")

def check_models():
    """Check if models exist for each game and model type."""
    print("\n" + "="*80)
    print("CHECKING TRAINED MODELS")
    print("="*80)
    
    games = ["Lotto Max", "Lotto 6/49", "Super 7"]  # Example games
    model_types = ["CNN", "LSTM", "Transformer", "XGBoost", "CatBoost", "LightGBM"]
    
    for game in games:
        game_folder = sanitize_game_name(game)
        models_dir = get_models_dir(game)
        
        if not models_dir.exists():
            print(f"\n🎮 {game}: ❌ Models directory does not exist")
            continue
        
        print(f"\n🎮 {game}:")
        for model_type in model_types:
            model_subdir = models_dir / model_type.lower()
            if model_subdir.exists():
                model_files = sorted(list(model_subdir.glob(f"{model_type.lower()}_{game_folder}_*.{('keras' if model_type in ['CNN', 'LSTM', 'Transformer'] else 'joblib')}")))
                if model_files:
                    latest = model_files[-1]
                    size = latest.stat().st_size / (1024*1024)  # MB
                    print(f"   ✓ {model_type:12}: {latest.name} ({size:.1f}MB)")
                else:
                    print(f"   ❌ {model_type:12}: Directory exists but no model files found")
            else:
                print(f"   ❌ {model_type:12}: No directory")

def diagnose_issue():
    """Run all diagnostics."""
    print("\n" + "="*80)
    print("🔍 CONFIDENCE DIAGNOSTICS")
    print("="*80)
    
    check_models()
    check_training_features()
    analyze_predictions_by_game()
    
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print("""
    If ALL predictions show 50% confidence:
    
    POSSIBLE CAUSES (in order of likelihood):
    
    1. ⚠️  RANDOM INPUT FALLBACK
       - Training features not found
       - Models getting random noise as input
       - Random noise → uncertain predictions (~0.5)
       - CHECK: Look for "No training features found" warnings in logs
       - FIX: Generate training features (run training script first)
    
    2. ⚠️  MODELS POORLY TRAINED
       - Models output uncertain predictions even on real data
       - All probability distributions flat/uniform
       - CHECK: Look at model accuracy metadata
       - FIX: Retrain models with better data/parameters
    
    3. ⚠️  CONFIDENCE THRESHOLD TOO HIGH
       - Calculated confidence < 0.5 for all predictions
       - Clamped to minimum threshold of 0.5
       - CHECK: Try lowering confidence_threshold slider to 0.0
       - FIX: This is by design, models truly are uncertain
    
    4. ⚠️  BUG IN CONFIDENCE CALCULATION
       - Calculation has logic error
       - All confidence values hardcoded to 0.5
       - CHECK: Look at recent code changes to confidence calculation
       - FIX: Review and fix the confidence calculation logic
    
    NEXT STEPS:
    - Run the script above to generate the diagnostic output
    - Check logs for "No training features found" messages
    - Verify models can load and make predictions
    - Check if training features exist in data/features/
    - Manually test a single model prediction
    """)

if __name__ == "__main__":
    diagnose_issue()
