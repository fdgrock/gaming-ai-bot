"""
Diagnostic script to identify feature count mismatches between:
1. Generated features (CSV files)
2. Saved schemas (JSON files)
3. Trained models (joblib/keras files)
"""

import json
import joblib
import glob
import numpy as np
import pandas as pd
from pathlib import Path

def check_feature_csv_files():
    """Check actual feature count in CSV files"""
    print("\n" + "="*80)
    print("1. FEATURE CSV FILES")
    print("="*80)
    
    features_dir = Path("data/features")
    
    for model_type in ["xgboost", "catboost", "lightgbm", "lstm", "cnn", "transformer"]:
        model_dir = features_dir / model_type
        if not model_dir.exists():
            continue
        
        for game_dir in model_dir.iterdir():
            if game_dir.is_dir():
                game = game_dir.name.replace("_", " ").title()
                
                # Find latest CSV
                csv_files = sorted(glob.glob(str(game_dir / "*.csv")))
                if csv_files:
                    latest_csv = csv_files[-1]
                    df = pd.read_csv(latest_csv)
                    print(f"\n{model_type.upper()} - {game}")
                    print(f"  CSV File: {Path(latest_csv).name}")
                    print(f"  Shape: {df.shape}")
                    print(f"  Features: {df.shape[1]}")
                    if df.shape[1] <= 10:
                        print(f"  Columns: {list(df.columns)}")
                    else:
                        print(f"  First 5 columns: {list(df.columns[:5])}")
                        print(f"  Last 5 columns: {list(df.columns[-5:])}")

def check_feature_schemas():
    """Check feature count in schema JSON files"""
    print("\n" + "="*80)
    print("2. FEATURE SCHEMAS (JSON)")
    print("="*80)
    
    features_dir = Path("data/features")
    
    for model_type in ["xgboost", "catboost", "lightgbm", "lstm", "cnn", "transformer"]:
        model_dir = features_dir / model_type
        if not model_dir.exists():
            continue
        
        for game_dir in model_dir.iterdir():
            if game_dir.is_dir():
                game = game_dir.name.replace("_", " ").title()
                schema_file = game_dir / "feature_schema.json"
                
                if schema_file.exists():
                    with open(schema_file) as f:
                        schema = json.load(f)
                    
                    feature_count = schema.get("feature_count", "MISSING")
                    feature_names = schema.get("feature_names", [])
                    
                    print(f"\n{model_type.upper()} - {game}")
                    print(f"  feature_count field: {feature_count}")
                    print(f"  feature_names length: {len(feature_names)}")
                    
                    # Check for mismatch
                    if isinstance(feature_count, int) and len(feature_names) != feature_count:
                        print(f"  ⚠️ MISMATCH: Count={feature_count}, Names={len(feature_names)}")

def check_trained_models():
    """Check feature count in trained models"""
    print("\n" + "="*80)
    print("3. TRAINED MODELS")
    print("="*80)
    
    model_files = sorted(glob.glob("models/**/*.joblib", recursive=True))
    model_files += sorted(glob.glob("models/**/*.keras", recursive=True))
    
    for model_file in model_files[:12]:
        try:
            model = joblib.load(model_file)
            n_features = getattr(model, 'n_features_in_', None)
            
            path_parts = model_file.split("\\")[-1]
            print(f"\n{path_parts}")
            print(f"  Type: {type(model).__name__}")
            
            if n_features is not None:
                print(f"  Trained with: {n_features} features")
            else:
                print(f"  n_features_in_: Not found (likely Keras)")
        except Exception as e:
            print(f"\n{model_file}")
            print(f"  ERROR: {e}")

def compare_all():
    """Compare feature counts across all three sources"""
    print("\n" + "="*80)
    print("4. COMPARISON MATRIX")
    print("="*80)
    
    features_dir = Path("data/features")
    results = []
    
    for model_type in ["xgboost", "catboost", "lightgbm", "lstm", "cnn", "transformer"]:
        model_dir = features_dir / model_type
        if not model_dir.exists():
            continue
        
        for game_dir in model_dir.iterdir():
            if game_dir.is_dir():
                game = game_dir.name.replace("_", " ").title()
                
                # 1. CSV count
                csv_files = sorted(glob.glob(str(game_dir / "*.csv")))
                csv_count = pd.read_csv(csv_files[-1]).shape[1] if csv_files else 0
                
                # 2. Schema count
                schema_file = game_dir / "feature_schema.json"
                schema_count = 0
                if schema_file.exists():
                    with open(schema_file) as f:
                        schema = json.load(f)
                    schema_count = schema.get("feature_count", len(schema.get("feature_names", [])))
                
                # 3. Model count
                model_count = 0
                game_folder = "lotto_max" if "Max" in game else "lotto_6_49"
                model_files = glob.glob(f"models/{game_folder}/{model_type}/*.joblib")
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    model = joblib.load(latest_model)
                    model_count = getattr(model, 'n_features_in_', 0)
                
                results.append({
                    'Model': model_type.upper(),
                    'Game': game,
                    'CSV': csv_count,
                    'Schema': schema_count,
                    'Model': model_count,
                    'Status': '✅' if csv_count == schema_count == model_count and csv_count > 0 else '❌'
                })
    
    # Print table
    if results:
        print(f"\n{'Model':<12} {'Game':<15} {'CSV':<6} {'Schema':<8} {'Model':<8} {'Status':<6}")
        print("-" * 60)
        for r in results:
            print(f"{r['Model']:<12} {r['Game']:<15} {r['CSV']:<6} {r['Schema']:<8} {r['Model']:<8} {r['Status']:<6}")
        
        # Summary
        print("\n" + "-" * 60)
        passed = sum(1 for r in results if r['Status'] == '✅')
        total = len(results)
        print(f"OVERALL: {passed}/{total} models have matching feature counts")

if __name__ == "__main__":
    print("\n🔍 FEATURE COUNT DIAGNOSTIC REPORT")
    print(f"Generated: {pd.Timestamp.now()}")
    
    check_feature_csv_files()
    check_feature_schemas()
    check_trained_models()
    compare_all()
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print("""
1. **CSV files** = Actual features generated by AdvancedFeatureGenerator
2. **Schema JSON** = What schema says was generated
3. **Model files** = What features model was actually trained with

If all three match → System is working correctly
If they differ → There's a synchronization bug

MOST COMMON ISSUE: Schema says 85, Model trained with 93
This means the padded features were created AFTER schema was saved.
""")
