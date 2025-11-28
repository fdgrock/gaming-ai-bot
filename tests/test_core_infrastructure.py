#!/usr/bin/env python3
"""
Integration test script for the enhanced core infrastructure.

This script tests the basic functionality of all core modules to ensure
they work correctly and don't have import issues or circular dependencies.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports from core modules."""
    print("🔄 Testing basic imports...")
    
    try:
        # Test config module
        from streamlit_app.core.config import AppConfig, get_config
        print("✅ Config module imported successfully")
        
        # Test logger module
        from streamlit_app.core.logger import AppLogger, app_log
        print("✅ Logger module imported successfully")
        
        # Test exceptions module
        from streamlit_app.core.exceptions import LottoAIError, DataError
        print("✅ Exceptions module imported successfully")
        
        # Test utils module
        from streamlit_app.core.utils import sanitize_game_name, get_est_now
        print("✅ Utils module imported successfully")
        
        # Test data_manager module
        from streamlit_app.core.data_manager import DataManager, get_data_manager
        print("✅ Data manager module imported successfully")
        
        # Test session_manager module
        from streamlit_app.core.session_manager import SessionManager, get_session_manager
        print("✅ Session manager module imported successfully")
        
        # Test core __init__ imports
        from streamlit_app.core import initialize_core_infrastructure, get_core_info
        print("✅ Core package imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_configuration():
    """Test configuration functionality."""
    print("\n🔄 Testing configuration...")
    
    try:
        from streamlit_app.core import get_config, is_feature_enabled, get_data_path
        
        # Get configuration
        config = get_config()
        print(f"✅ Config loaded: {config.app_name} v{config.app_version}")
        
        # Test feature flags
        is_ai_enabled = is_feature_enabled("prediction_ai")
        print(f"✅ Feature flag test: prediction_ai = {is_ai_enabled}")
        
        # Test path management
        lotto_max_path = get_data_path("Lotto Max")
        print(f"✅ Path management: {lotto_max_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_logging():
    """Test logging functionality."""
    print("\n🔄 Testing logging...")
    
    try:
        from streamlit_app.core import app_log, log_data_operation
        
        # Test basic logging
        app_log("Test log message", "info")
        print("✅ Basic logging works")
        
        # Test structured logging
        log_data_operation("test_operation", "Lotto Max", record_count=100, status="success")
        print("✅ Structured logging works")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\n🔄 Testing utilities...")
    
    try:
        from streamlit_app.core import (
            sanitize_game_name, get_est_now, format_numbers,
            validate_number_combination, safe_float_conversion
        )
        
        # Test game name sanitization
        sanitized = sanitize_game_name("Lotto 6/49")
        assert sanitized == "lotto_6_49", f"Expected 'lotto_6_49', got '{sanitized}'"
        print("✅ Game name sanitization works")
        
        # Test number formatting
        formatted = format_numbers([7, 15, 23, 42, 1, 50])
        print(f"✅ Number formatting works: {formatted}")
        
        # Test safe conversions
        safe_val = safe_float_conversion("85.5%", 0.0)
        print(f"✅ Safe float conversion: {safe_val}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False

def test_data_manager():
    """Test data manager functionality."""
    print("\n🔄 Testing data manager...")
    
    try:
        from streamlit_app.core import get_data_manager, get_available_games
        
        # Get data manager
        dm = get_data_manager()
        print("✅ Data manager instance created")
        
        # Test game listing
        games = get_available_games()
        print(f"✅ Available games: {games}")
        
        # Test cache operations
        dm.clear_all_cache()
        print("✅ Cache operations work")
        
        return True
        
    except Exception as e:
        print(f"❌ Data manager test failed: {e}")
        return False

def test_exceptions():
    """Test exception handling."""
    print("\n🔄 Testing exception handling...")
    
    try:
        from streamlit_app.core import (
            LottoAIError, GameNotSupportedError, safe_execute
        )
        
        # Test basic exception
        try:
            raise LottoAIError("Test error", "TEST_001", {"context": "testing"})
        except LottoAIError as e:
            print(f"✅ Basic exception works: {e.error_code}")
        
        # Test specific exception
        try:
            raise GameNotSupportedError("Invalid Game", ["Lotto Max", "Lotto 6/49"])
        except GameNotSupportedError as e:
            print(f"✅ Specific exception works: {e}")
        
        # Test safe execution
        def failing_function():
            raise ValueError("Test failure")
        
        result = safe_execute(failing_function, default_return="fallback", log_errors=False)
        print("✅ Safe execution works")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception handling test failed: {e}")
        return False

def test_core_initialization():
    """Test core infrastructure initialization."""
    print("\n🔄 Testing core initialization...")
    
    try:
        from streamlit_app.core import initialize_core_infrastructure, get_core_info
        
        # Initialize core
        success = initialize_core_infrastructure()
        print(f"✅ Core initialization: {success}")
        
        # Get core info
        info = get_core_info()
        print(f"✅ Core info: {info.get('app_name', 'Unknown')} - {info.get('features_enabled', 0)} features enabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Core initialization test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting Core Infrastructure Integration Tests\n")
    
    tests = [
        test_basic_imports,
        test_configuration,
        test_logging,
        test_utilities,
        test_data_manager,
        test_exceptions,
        test_core_initialization
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Core infrastructure is ready.")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)