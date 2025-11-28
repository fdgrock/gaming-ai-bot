#!/usr/bin/env python3
"""
Page Import Test - Check which pages have import issues
"""
import sys
from pathlib import Path
import importlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_page_imports():
    """Test importing all page modules and check for render_page functions"""
    
    pages = [
        'analytics',
        'dashboard', 
        'data_training',
        'help_docs',
        'history',
        'incremental_learning',
        'model_manager',
        'predictions',
        'prediction_ai', 
        'prediction_engine',
        'settings'
    ]
    
    print("🧪 Testing Page Imports and Functions")
    print("=" * 60)
    
    results = {
        'import_success': [],
        'import_failed': [],
        'has_render_page': [],
        'missing_render_page': [],
        'errors': {}
    }
    
    for page_name in pages:
        print(f"\n📄 Testing: {page_name}.py")
        
        try:
            # Test import
            module_name = f"streamlit_app.pages.{page_name}"
            module = importlib.import_module(module_name)
            print(f"  ✅ Import successful")
            results['import_success'].append(page_name)
            
            # Check for render_page function
            if hasattr(module, 'render_page'):
                print(f"  ✅ render_page() function found")
                results['has_render_page'].append(page_name)
                
                # Check function signature
                import inspect
                sig = inspect.signature(module.render_page)
                params = list(sig.parameters.keys())
                print(f"  📋 Function signature: render_page({', '.join(params)})")
                
            else:
                print(f"  ⚠️ render_page() function missing")
                results['missing_render_page'].append(page_name)
                
                # Check for alternative main functions
                alt_functions = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if callable(attr) and not attr_name.startswith('_'):
                        if attr_name in ['main', 'show', 'display', 'run']:
                            alt_functions.append(attr_name)
                
                if alt_functions:
                    print(f"  🔍 Alternative functions found: {', '.join(alt_functions)}")
        
        except ImportError as e:
            print(f"  ❌ Import failed: {e}")
            results['import_failed'].append(page_name)
            results['errors'][page_name] = str(e)
            
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            results['import_failed'].append(page_name)
            results['errors'][page_name] = str(e)
    
    return results

def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 60)
    print("📊 PAGE IMPORT TEST SUMMARY")
    print("=" * 60)
    
    total_pages = len(results['import_success']) + len(results['import_failed'])
    
    print(f"📄 Total pages tested: {total_pages}")
    print(f"✅ Successfully imported: {len(results['import_success'])}")
    print(f"❌ Import failures: {len(results['import_failed'])}")
    print(f"🎯 Pages with render_page(): {len(results['has_render_page'])}")
    print(f"⚠️ Pages missing render_page(): {len(results['missing_render_page'])}")
    
    if results['import_failed']:
        print(f"\n❌ FAILED IMPORTS:")
        for page in results['import_failed']:
            print(f"  • {page}: {results['errors'].get(page, 'Unknown error')}")
    
    if results['missing_render_page']:
        print(f"\n⚠️ MISSING render_page() FUNCTION:")
        for page in results['missing_render_page']:
            print(f"  • {page}")
    
    if results['has_render_page']:
        print(f"\n✅ PAGES WITH render_page():")
        for page in results['has_render_page']:
            print(f"  • {page}")
    
    # Calculate standardization progress
    if results['import_success']:
        standardization_rate = len(results['has_render_page']) / len(results['import_success']) * 100
        print(f"\n📊 Page Standardization Progress: {standardization_rate:.1f}%")

def main():
    """Run page import tests"""
    results = test_page_imports()
    print_summary(results)

if __name__ == "__main__":
    main()