#!/usr/bin/env python3
"""
Simplified Test Suite for CatBoost & LightGBM Integration
Focus: Verify new model training functions work correctly
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add streamlit_app to path
sys.path.insert(0, str(Path(__file__).parent / "streamlit_app"))

print("=" * 80)
print("TEST SUITE: CatBoost & LightGBM Integration")
print("=" * 80)

test_results = {"passed": 0, "failed": 0, "errors": []}

# ============================================================================
# TEST 1: Import Verification
# ============================================================================
print("\n[TEST 1] Import Verification")
print("-" * 80)

try:
    import catboost as cb
    print("[OK] CatBoost imported - Version:", cb.__version__)
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] CatBoost:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"CatBoost: {e}")

try:
    import lightgbm as lgb
    print("[OK] LightGBM imported - Version:", lgb.__version__)
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] LightGBM:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"LightGBM: {e}")

try:
    import xgboost as xgb
    print("[OK] XGBoost imported - Version:", xgb.__version__)
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] XGBoost:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"XGBoost: {e}")

try:
    from services.advanced_model_training import AdvancedModelTrainer
    print("[OK] AdvancedModelTrainer imported")
    test_results["passed"] += 1
except Exception as e:
    print("[FAIL] AdvancedModelTrainer:", str(e))
    test_results["failed"] += 1
    test_results["errors"].append(f"AdvancedModelTrainer: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Method Availability
# ============================================================================
print("\n[TEST 2] Method Availability")
print("-" * 80)

try:
    trainer = AdvancedModelTrainer(game="Lotto 6/49")
    
    methods = ['train_catboost', 'train_lightgbm', 'train_ensemble']
    for method in methods:
        if hasattr(trainer, method):
            print(f"[OK] {method}() exists")
            test_results["passed"] += 1
        else:
            print(f"[FAIL] {method}() not found")
            test_results["failed"] += 1
            test_results["errors"].append(f"Missing method: {method}")
            
except Exception as e:
    print(f"[FAIL] Error: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"Method check: {e}")

# ============================================================================
# TEST 3: Create Test Data
# ============================================================================
print("\n[TEST 3] Create Test Data")
print("-" * 80)

try:
    np.random.seed(42)
    n_samples = 300
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    metadata = {
        "game_type": "lotto_6_49",
        "feature_count": n_features,
        "sample_count": n_samples
    }
    
    config = {
        "catboost_iterations": 50,
        "catboost_depth": 5,
        "lightgbm_estimators": 50,
    }
    
    print(f"[OK] Created test data: X={X.shape}, y={y.shape}")
    test_results["passed"] += 1
    
except Exception as e:
    print(f"[FAIL] Data creation: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"Data creation: {e}")
    sys.exit(1)

# ============================================================================
# TEST 4: CatBoost Training
# ============================================================================
print("\n[TEST 4] CatBoost Training")
print("-" * 80)

try:
    print("  Training CatBoost model...")
    
    cb_model, cb_metrics = trainer.train_catboost(
        X, y, metadata, config,
        progress_callback=lambda pct, msg: None  # Silent callback
    )
    
    print("[OK] CatBoost training completed")
    print(f"    Accuracy: {cb_metrics.get('accuracy', 'N/A')}")
    print(f"    Precision: {cb_metrics.get('precision', 'N/A')}")
    test_results["passed"] += 1
    
except Exception as e:
    print(f"[FAIL] CatBoost training: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"CatBoost: {e}")

# ============================================================================
# TEST 5: LightGBM Training
# ============================================================================
print("\n[TEST 5] LightGBM Training")
print("-" * 80)

try:
    print("  Training LightGBM model...")
    
    lgb_model, lgb_metrics = trainer.train_lightgbm(
        X, y, metadata, config,
        progress_callback=lambda pct, msg: None  # Silent callback
    )
    
    print("[OK] LightGBM training completed")
    print(f"    Accuracy: {lgb_metrics.get('accuracy', 'N/A')}")
    print(f"    Precision: {lgb_metrics.get('precision', 'N/A')}")
    test_results["passed"] += 1
    
except Exception as e:
    print(f"[FAIL] LightGBM training: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"LightGBM: {e}")

# ============================================================================
# TEST 6: Model Folders
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
    
    for folder in required_folders:
        folder_path = models_dir / folder
        status = "[OK]" if folder_path.exists() else "[MISSING]"
        print(f"  {status} {folder}")
        if folder_path.exists():
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1
            test_results["errors"].append(f"Missing folder: {folder}")
    
except Exception as e:
    print(f"[FAIL] Folder check: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"Folder check: {e}")

# ============================================================================
# TEST 7: Requirements.txt
# ============================================================================
print("\n[TEST 7] Requirements.txt")
print("-" * 80)

try:
    req_path = Path(__file__).parent / "requirements.txt"
    with open(req_path, 'r') as f:
        req_content = f.read()
    
    packages = {
        "catboost": "catboost" in req_content.lower(),
        "lightgbm": "lightgbm" in req_content.lower(),
        "xgboost": "xgboost" in req_content.lower(),
    }
    
    for pkg, found in packages.items():
        status = "[OK]" if found else "[MISSING]"
        print(f"  {status} {pkg}")
        if found:
            test_results["passed"] += 1
        else:
            test_results["failed"] += 1
            test_results["errors"].append(f"Missing requirement: {pkg}")
    
except Exception as e:
    print(f"[FAIL] Requirements check: {str(e)}")
    test_results["failed"] += 1
    test_results["errors"].append(f"Requirements: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

total = test_results["passed"] + test_results["failed"]
pass_rate = (test_results["passed"] / total * 100) if total > 0 else 0

print(f"\nTotal Tests: {total}")
print(f"Passed: {test_results['passed']}")
print(f"Failed: {test_results['failed']}")
print(f"Pass Rate: {pass_rate:.1f}%")

if test_results["errors"]:
    print("\nErrors:")
    for error in test_results["errors"]:
        print(f"  - {error}")

if test_results["failed"] == 0:
    print("\n" + "=" * 80)
    print("SUCCESS! CatBoost and LightGBM integration is working correctly.")
    print("=" * 80)
    sys.exit(0)
else:
    print("\n" + "=" * 80)
    print("ATTENTION: Some tests failed. Review errors above.")
    print("=" * 80)
    sys.exit(1)
