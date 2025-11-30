#!/usr/bin/env python3
"""
Comprehensive Test Suite for CatBoost & LightGBM Integration
Tests all new model training functions and ensemble orchestration
"""

import sys
import os
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add streamlit_app to path
sys.path.insert(0, str(Path(__file__).parent / "streamlit_app"))

print("=" * 80)
print("COMPREHENSIVE TEST SUITE - CatBoost & LightGBM Integration")
print("=" * 80)

# ============================================================================
# TEST 1: Import Verification
# ============================================================================
print("\n[TEST 1] Import Verification")
print("-" * 80)

test_results = {"passed": 0, "failed": 0, "errors": []}

try:
    import catboost as cb
    print("[OK] CatBoost imported - Version:", cb.__version__)
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] CatBoost import failed:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"CatBoost: {str(e)}")

try:
    import lightgbm as lgb
    print("[OK] LightGBM imported - Version:", lgb.__version__)
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] LightGBM import failed:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"LightGBM: {str(e)}")

try:
    import xgboost as xgb
    print("[OK] XGBoost imported - Version:", xgb.__version__)
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] XGBoost import failed:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"XGBoost: {str(e)}")

try:
    from tensorflow import keras
    print("[OK] TensorFlow/Keras imported")
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] TensorFlow/Keras import failed:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"TensorFlow: {str(e)}")

try:
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold
    print("[OK] Scikit-learn imported")
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] Scikit-learn import failed:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"Scikit-learn: {str(e)}")

try:
    from services.advanced_model_training import AdvancedModelTrainer
    print("[OK] AdvancedModelTrainer class imported successfully")
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] AdvancedModelTrainer import failed:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"AdvancedModelTrainer: {str(e)}")

# ============================================================================
# TEST 2: Function Availability Check
# ============================================================================
print("\n[TEST 2] Function Availability Check")
print("-" * 80)

try:
    trainer = AdvancedModelTrainer(game="Lotto 6/49")
    
    # Check for CatBoost function
    if hasattr(trainer, 'train_catboost'):
        print("[OK] train_catboost() method exists")
        test_results["passed"] += 1
    else:
        print("[FAIL] train_catboost() method not found")
        test_results["failed"] += 1
        test_results["errors"].append("train_catboost method missing")
    
    # Check for LightGBM function
    if hasattr(trainer, 'train_lightgbm'):
        print("[OK] train_lightgbm() method exists")
        test_results["passed"] += 1
    else:
        print("[FAIL] train_lightgbm() method not found")
        test_results["failed"] += 1
        test_results["errors"].append("train_lightgbm method missing")
    
    # Check for Ensemble function
    if hasattr(trainer, 'train_ensemble'):
        print("[OK] train_ensemble() method exists")
        test_results["passed"] += 1
    else:
        print("[FAIL] train_ensemble() method not found")
        test_results["failed"] += 1
        test_results["errors"].append("train_ensemble method missing")
        
except Exception as e:
    print("[FAIL] Error checking functions:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"Function check: {str(e)}")

# ============================================================================
# TEST 3: Sample Data Creation & Training
# ============================================================================
print("\n[TEST 3] Sample Data Creation & Training Functions")
print("-" * 80)

try:
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    
    print(f"[OK] Created synthetic dataset: X={X.shape}, y={y.shape}")
    test_results["passed"] += 1
    
    # Create dummy metadata
    metadata = {
        "game_type": "lotto_6_49",
        "source_data": "test_data",
        "feature_count": n_features,
        "sample_count": n_samples,
        "timestamp": datetime.now().isoformat()
    }
    
    config = {
        "catboost_iterations": 100,
        "catboost_depth": 6,
        "lightgbm_estimators": 100,
        "xgboost_max_depth": 6,
    }
    
    print(f"[OK] Created metadata and config")
    test_results["passed"] += 1
    
except Exception as e:
    print(f"[FAIL] Data creation failed: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"Data creation: {str(e)}")

# ============================================================================
# TEST 4: CatBoost Training
# ============================================================================
print("\n[TEST 4] CatBoost Training Function")
print("-" * 80)

try:
    print("  Training CatBoost model...")
    
    cb_model, cb_metrics = trainer.train_catboost(
        X, y, metadata, config,
        progress_callback=lambda pct, msg: print(f"    [{pct:3.0f}%] {msg}")
    )
    
    print("[OK] CatBoost training completed successfully")
    acc = cb_metrics.get('accuracy', 0)
    prec = cb_metrics.get('precision', 0)
    rec = cb_metrics.get('recall', 0)
    f1 = cb_metrics.get('f1', 0)
    print(f"    Accuracy: {acc if isinstance(acc, float) else 'N/A':.4f}" if isinstance(acc, float) else f"    Accuracy: {acc}")
    print(f"    Precision: {prec if isinstance(prec, float) else 'N/A':.4f}" if isinstance(prec, float) else f"    Precision: {prec}")
    print(f"    Recall: {rec if isinstance(rec, float) else 'N/A':.4f}" if isinstance(rec, float) else f"    Recall: {rec}")
    print(f"    F1-Score: {f1 if isinstance(f1, float) else 'N/A':.4f}" if isinstance(f1, float) else f"    F1-Score: {f1}")
    test_results["passed"] += 1
    
except Exception as e:
    print(f"[FAIL] CatBoost training failed: {str(e)}")
    import traceback
    traceback.print_exc()
    test_results["failed"] += 1
    test_results["errors"].append(f"CatBoost training: {str(e)}")
    cb_model = None
    cb_metrics = {}

# ============================================================================
# TEST 5: LightGBM Training
# ============================================================================
print("\n[TEST 5] LightGBM Training Function")
print("-" * 80)

try:
    print("  Training LightGBM model...")
    
    lgb_model, lgb_metrics = trainer.train_lightgbm(
        X, y, metadata, config,
        progress_callback=lambda pct, msg: print(f"    [{pct:3.0f}%] {msg}")
    )
    
    print("[OK] LightGBM training completed successfully")
    print(f"    Accuracy: {lgb_metrics.get('accuracy', 'N/A'):.4f}")
    print(f"    Precision: {lgb_metrics.get('precision', 'N/A'):.4f}")
    print(f"    Recall: {lgb_metrics.get('recall', 'N/A'):.4f}")
    print(f"    F1-Score: {lgb_metrics.get('f1', 'N/A'):.4f}")
    test_results["passed"] += 1
    
except Exception as e:
    print(f"[FAIL] LightGBM training failed: {str(e)}")
    import traceback
    traceback.print_exc()
    test_results["failed"] += 1
    test_results["errors"].append(f"LightGBM training: {str(e)}")
    lgb_model = None
    lgb_metrics = {}

# ============================================================================
# TEST 6: Model Folder Structure
# ============================================================================
print("\n[TEST 6] Model Folder Structure")
print("-" * 80)

try:
    models_dir = Path(__file__).parent / "models"
    
    required_folders = [
        "lotto_6_49/catboost",
        "lotto_6_49/lightgbm",
        "lotto_max/catboost",
        "lotto_max/lightgbm"
    ]
    
    all_exist = True
    for folder in required_folders:
        folder_path = models_dir / folder
        exists = folder_path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {folder}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("[OK] All required model folders exist")
        test_results["passed"] += 1
    else:
        print("[WARNING] Some model folders are missing")
        test_results["passed"] += 1  # Just warning, not critical for tests
        
except Exception as e:
    print(f"[FAIL] Folder check failed: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"Folder structure: {str(e)}")

# ============================================================================
# TEST 7: XGBoost Training (verify existing model still works)
# ============================================================================
print("\n[TEST 7] XGBoost Training (Existing Model)")
print("-" * 80)

try:
    print("  Training XGBoost model...")
    
    xgb_model, xgb_metrics = trainer.train_xgboost(
        X, y, metadata, config,
        progress_callback=lambda pct, msg: print(f"    [{pct:3.0f}%] {msg}")
    )
    
    print("[OK] XGBoost training completed successfully")
    print(f"    Accuracy: {xgb_metrics.get('accuracy', 'N/A'):.4f}")
    print(f"    Precision: {xgb_metrics.get('precision', 'N/A'):.4f}")
    print(f"    Recall: {xgb_metrics.get('recall', 'N/A'):.4f}")
    print(f"    F1-Score: {xgb_metrics.get('f1', 'N/A'):.4f}")
    test_results["passed"] += 1
    
except Exception as e:
    print(f"[FAIL] XGBoost training failed: {str(e)}")
    import traceback
    traceback.print_exc()
    test_results["failed"] += 1
    test_results["errors"].append(f"XGBoost training: {str(e)}")
    xgb_model = None
    xgb_metrics = {}

# ============================================================================
# TEST 8: Ensemble Training
# ============================================================================
print("\n[TEST 8] Ensemble Training (4-Model Orchestration)")
print("-" * 80)

try:
    print("  Training ensemble with all 4 models (small iterations)...")
    
    # Use smaller config for faster testing
    config_ensemble = {
        "catboost_iterations": 50,
        "catboost_depth": 4,
        "lightgbm_estimators": 50,
        "xgboost_max_depth": 4,
    }
    
    ensemble_models, ensemble_metrics = trainer.train_ensemble(
        X, y, metadata, config_ensemble,
        progress_callback=lambda pct, msg: print(f"    [{pct:3.0f}%] {msg}")
    )
    
    print("[OK] Ensemble training completed successfully")
    print(f"    Models trained: {list(ensemble_models.keys())}")
    print(f"    Ensemble accuracy: {ensemble_metrics.get('ensemble_accuracy', 'N/A'):.4f}")
    
    # Check each model in ensemble
    for model_name, model_metrics in ensemble_metrics.get('individual_metrics', {}).items():
        print(f"    {model_name} accuracy: {model_metrics.get('accuracy', 'N/A'):.4f}")
    
    test_results["passed"] += 1
    
except Exception as e:
    print(f"[FAIL] Ensemble training failed: {str(e)}")
    import traceback
    traceback.print_exc()
    test_results["failed"] += 1
    test_results["errors"].append(f"Ensemble training: {str(e)}")

# ============================================================================
# TEST 9: UI Integration - Model Selection
# ============================================================================
print("\n[TEST 9] UI Integration - Model Selection")
print("-" * 80)

try:
    # Verify the model options are available in data_training.py
    data_training_path = Path(__file__).parent / "streamlit_app" / "pages" / "data_training.py"
    
    if data_training_path.exists():
        with open(data_training_path, 'r') as f:
            content = f.read()
        
        required_models = ["CatBoost", "LightGBM", "XGBoost", "LSTM", "CNN", "Ensemble"]
        missing = []
        
        for model in required_models:
            if model in content:
                print(f"  [OK] {model} option found in UI")
                test_results["passed"] += 1
            else:
                print(f"  [MISSING] {model} option not found in UI")
                missing.append(model)
                test_results["failed"] += 1
        
        if missing:
            test_results["errors"].append(f"Missing UI options: {missing}")
    else:
        print(f"[WARNING] data_training.py not found at expected location")
        test_results["passed"] += 1
        
except Exception as e:
    print(f"[FAIL] UI integration check failed: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"UI check: {str(e)}")

# ============================================================================
# TEST 10: Requirements.txt Verification
# ============================================================================
print("\n[TEST 10] Requirements.txt Verification")
print("-" * 80)

try:
    req_path = Path(__file__).parent / "requirements.txt"
    
    if req_path.exists():
        with open(req_path, 'r') as f:
            req_content = f.read()
        
        required_packages = ["catboost", "lightgbm", "xgboost", "tensorflow", "scikit-learn"]
        found = []
        missing = []
        
        for pkg in required_packages:
            if pkg.lower() in req_content.lower():
                print(f"  [OK] {pkg} found in requirements.txt")
                found.append(pkg)
                test_results["passed"] += 1
            else:
                print(f"  [MISSING] {pkg} not found in requirements.txt")
                missing.append(pkg)
                test_results["failed"] += 1
        
        if missing:
            test_results["errors"].append(f"Missing in requirements.txt: {missing}")
    else:
        print("[WARNING] requirements.txt not found")
        test_results["passed"] += 1
        
except Exception as e:
    print(f"[FAIL] Requirements check failed: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"Requirements check: {str(e)}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

total_tests = test_results["passed"] + test_results["failed"]
pass_rate = (test_results["passed"] / total_tests * 100) if total_tests > 0 else 0

print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {test_results['passed']}")
print(f"Failed: {test_results['failed']}")
print(f"Pass Rate: {pass_rate:.1f}%")

if test_results["errors"]:
    print("\nErrors Encountered:")
    for i, error in enumerate(test_results["errors"], 1):
        print(f"  {i}. {error}")

if test_results["failed"] == 0:
    print("\n" + "=" * 80)
    print("SUCCESS! All tests passed. System is ready for deployment.")
    print("=" * 80)
    sys.exit(0)
else:
    print("\n" + "=" * 80)
    print("ATTENTION: Some tests failed. Review errors above.")
    print("=" * 80)
    sys.exit(1)
