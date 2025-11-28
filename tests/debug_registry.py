#!/usr/bin/env python3
"""
Debug script to isolate registry initialization issues
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_basic_imports():
    """Test basic module imports"""
    print("🧪 Testing basic imports...")
    
    try:
        # Test config import
        print("  📦 Importing configs...")
        from streamlit_app.configs import get_config
        print("  ✅ Configs imported successfully")
        
        # Test registry imports
        print("  📦 Importing registry modules...")
        from streamlit_app.registry import (
            EnhancedPageRegistry,
            ServicesRegistry,
            ComponentsRegistry,
            AIEnginesRegistry,
            NavigationContext
        )
        print("  ✅ Registry modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing configuration loading...")
    
    try:
        from streamlit_app.configs import get_config
        config = get_config()
        print(f"  ✅ Configuration loaded: {config.environment.value}")
        return True
        
    except Exception as e:
        print(f"  ❌ Config loading error: {e}")
        return False

def test_registry_creation():
    """Test registry object creation"""
    print("\n🧪 Testing registry object creation...")
    
    try:
        from streamlit_app.registry import (
            EnhancedPageRegistry,
            ServicesRegistry,
            ComponentsRegistry,
            AIEnginesRegistry
        )
        
        # Test each registry
        print("  🔧 Creating ServicesRegistry...")
        services_registry = ServicesRegistry()
        print("  ✅ ServicesRegistry created")
        
        print("  🔧 Creating ComponentsRegistry...")
        components_registry = ComponentsRegistry()
        print("  ✅ ComponentsRegistry created")
        
        print("  🔧 Creating AIEnginesRegistry...")
        ai_engines_registry = AIEnginesRegistry()
        print("  ✅ AIEnginesRegistry created")
        
        print("  🔧 Creating EnhancedPageRegistry...")
        page_registry = EnhancedPageRegistry()
        print("  ✅ EnhancedPageRegistry created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Registry creation error: {e}")
        import traceback
        print(f"  📋 Traceback:\n{traceback.format_exc()}")
        return False

def test_app_class_creation():
    """Test EnhancedGamingAIApp creation without Streamlit"""
    print("\n🧪 Testing app class creation...")
    
    try:
        # Import without Streamlit context
        import app
        
        print("  🔧 App module imported successfully")
        
        # Note: We can't actually create the app class without Streamlit context
        # But we can verify the module loads correctly
        print("  ✅ App module loads without errors")
        return True
        
    except Exception as e:
        print(f"  ❌ App class error: {e}")
        import traceback
        print(f"  📋 Traceback:\n{traceback.format_exc()}")
        return False

def main():
    """Run all debug tests"""
    print("🔍 Registry Debug Analysis")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_config_loading,
        test_registry_creation,
        test_app_class_creation
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Debug Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All debug tests passed! Registry system should be working.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()