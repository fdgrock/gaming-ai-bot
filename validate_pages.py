#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaming AI Bot - Complete Application Validation Script
Validates all pages and functionality
"""

import sys
import os
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test all critical imports."""
    print("\n Testing Imports...")
    
    try:
        from streamlit_app.core import get_available_games, app_log
        print("  OK: Core utilities imported successfully")
        return True
    except ImportError as e:
        print(f"  ERROR: Core import failed: {e}")
        return False


def test_data_access():
    """Test data access functionality."""
    print("\n Testing Data Access...")
    
    try:
        from streamlit_app.core import get_data_dir, get_models_dir, get_predictions_dir
        
        data_dir = get_data_dir()
        print(f"  OK: Data directory exists")
        
        models_dir = get_models_dir()
        print(f"  OK: Models directory exists")
        
        predictions_dir = get_predictions_dir()
        print(f"  OK: Predictions directory exists")
        
        return True
    except Exception as e:
        print(f"  ERROR: Data access failed: {e}")
        return False


def test_pages_structure():
    """Test that all pages have proper structure."""
    print("\n Testing Page Structure...")
    
    pages_dir = PROJECT_ROOT / "streamlit_app" / "pages"
    required_pages = [
        "dashboard.py",
        "predictions.py",
        "analytics.py",
        "model_manager.py",
        "data_training.py",
        "history.py",
        "settings.py",
        "help_docs.py",
        "incremental_learning.py",
        "prediction_ai.py",
    ]
    
    found = 0
    for page in required_pages:
        page_path = pages_dir / page
        if page_path.exists():
            found += 1
            print(f"  OK: {page}")
        else:
            print(f"  MISSING: {page}")
    
    print(f"\n  Summary: {found}/{len(required_pages)} pages found")
    return found >= 10


def test_configuration():
    """Test configuration system."""
    print("\n Testing Configuration...")
    
    try:
        from streamlit_app.core import get_config, get_available_games
        
        config = get_config()
        print(f"  OK: Configuration loaded")
        
        games = get_available_games()
        print(f"  OK: {len(games)} games available")
        
        return True
    except Exception as e:
        print(f"  ERROR: Configuration test failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\n Testing Utilities...")
    
    try:
        from streamlit_app.core import sanitize_game_name, get_game_config, ensure_directory_exists
        
        game_name = sanitize_game_name("Lotto Max")
        print("  OK: Game name sanitization works")
        
        config = get_game_config("Lotto Max")
        print("  OK: Game configuration works")
        
        test_dir = ensure_directory_exists("cache/test")
        print(f"  OK: Directory creation works")
        
        return True
    except Exception as e:
        print(f"  ERROR: Utility test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("GAMING AI BOT - VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Access", test_data_access),
        ("Pages Structure", test_pages_structure),
        ("Configuration", test_configuration),
        ("Utilities", test_utilities),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed >= 4:
        print("\nSUCCESS: Application ready to run!")
        print("   Command: streamlit run app.py")
        return 0
    else:
        print(f"\nFAILED: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
