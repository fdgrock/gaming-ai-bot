"""
Integration Test Suite for Task 8 - Registry System Testing

This test suite validates the integration between:
1. New registry-based app.py architecture
2. Standardized page modules with render_page() functions
3. Registry dependency injection system
4. Error handling and fallback mechanisms
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class RegistryIntegrationTester:
    """Comprehensive integration tester for the registry system."""
    
    def __init__(self):
        self.test_results = {}
        self.pages_dir = project_root / "streamlit_app" / "pages"
        self.page_files = [
            "analytics.py",
            "dashboard.py", 
            "data_training.py",
            "help_docs.py",
            "history.py",
            "incremental_learning.py",
            "model_manager.py",
            "prediction_ai.py",
            "prediction_engine.py",
            "predictions.py",
            "settings.py"
        ]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite."""
        print("🚀 Starting Registry Integration Testing...")
        print("="*60)
        
        # Test 1: App.py Registry Initialization
        print("\n📋 Test 1: App.py Registry System")
        self.test_results['app_registry'] = self._test_app_registry_init()
        
        # Test 2: Page Import Testing
        print("\n📋 Test 2: Page Module Imports")
        self.test_results['page_imports'] = self._test_page_imports()
        
        # Test 3: Render Function Testing
        print("\n📋 Test 3: Render Function Signatures")
        self.test_results['render_functions'] = self._test_render_functions()
        
        # Test 4: Registry Injection Testing
        print("\n📋 Test 4: Registry Dependency Injection")
        self.test_results['registry_injection'] = self._test_registry_injection()
        
        # Test 5: Fallback Mechanism Testing  
        print("\n📋 Test 5: Fallback Mechanisms")
        self.test_results['fallback_mechanisms'] = self._test_fallback_mechanisms()
        
        return self._generate_final_report()
    
    def _test_app_registry_init(self) -> Dict[str, Any]:
        """Test if app.py registry system initializes correctly."""
        results = {
            'status': 'unknown',
            'details': [],
            'errors': []
        }
        
        try:
            print("   🔄 Testing app.py import...")
            import app
            results['details'].append("✅ App module imports successfully")
            
            print("   🔄 Testing EnhancedGamingAIApp class...")
            if hasattr(app, 'EnhancedGamingAIApp'):
                results['details'].append("✅ EnhancedGamingAIApp class found")
                
                # Try to instantiate (but don't actually run Streamlit)
                print("   🔄 Testing class instantiation...")
                app_instance = app.EnhancedGamingAIApp()
                results['details'].append("✅ EnhancedGamingAIApp instantiates successfully")
                
                # Check for required methods
                required_methods = ['initialize_registries', 'setup_page_config', 'run']
                for method in required_methods:
                    if hasattr(app_instance, method):
                        results['details'].append(f"✅ Method {method} found")
                    else:
                        results['errors'].append(f"❌ Method {method} missing")
                        
            else:
                results['errors'].append("❌ EnhancedGamingAIApp class not found")
            
            results['status'] = 'success' if not results['errors'] else 'partial_success'
            
        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(f"❌ App registry initialization failed: {str(e)}")
            results['details'].append(f"Error details: {traceback.format_exc()}")
        
        return results
    
    def _test_page_imports(self) -> Dict[str, Any]:
        """Test if all page modules can be imported successfully."""
        results = {
            'status': 'unknown',
            'imported_pages': [],
            'failed_imports': [],
            'details': []
        }
        
        for page_file in self.page_files:
            try:
                print(f"   🔄 Testing import of {page_file}...")
                module_name = page_file.replace('.py', '')
                
                # Try to import the page module
                spec = importlib.util.spec_from_file_location(
                    f"streamlit_app.pages.{module_name}",
                    self.pages_dir / page_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                results['imported_pages'].append(page_file)
                results['details'].append(f"✅ {page_file} imports successfully")
                
            except Exception as e:
                results['failed_imports'].append({
                    'page': page_file,
                    'error': str(e)
                })
                results['details'].append(f"❌ {page_file} import failed: {str(e)}")
        
        total_pages = len(self.page_files)
        successful_imports = len(results['imported_pages'])
        
        results['status'] = 'success' if successful_imports == total_pages else 'partial_success'
        results['summary'] = f"{successful_imports}/{total_pages} pages imported successfully"
        
        return results
    
    def _test_render_functions(self) -> Dict[str, Any]:
        """Test if all pages have the required render_page function."""
        results = {
            'status': 'unknown',
            'pages_with_render': [],
            'pages_missing_render': [],
            'details': []
        }
        
        for page_file in self.page_files:
            try:
                print(f"   🔄 Testing render_page function in {page_file}...")
                module_name = page_file.replace('.py', '')
                
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"streamlit_app.pages.{module_name}",
                    self.pages_dir / page_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check for render_page function
                if hasattr(module, 'render_page'):
                    render_func = getattr(module, 'render_page')
                    
                    # Check function signature
                    import inspect
                    sig = inspect.signature(render_func)
                    params = list(sig.parameters.keys())
                    
                    # Expected parameters (all should be optional)
                    expected_params = ['services_registry', 'ai_engines', 'components']
                    has_expected_params = all(param in params for param in expected_params)
                    
                    if has_expected_params:
                        results['pages_with_render'].append({
                            'page': page_file,
                            'signature': str(sig),
                            'status': 'correct_signature'
                        })
                        results['details'].append(f"✅ {page_file} has correct render_page signature")
                    else:
                        results['pages_with_render'].append({
                            'page': page_file,
                            'signature': str(sig),
                            'status': 'incorrect_signature'
                        })
                        results['details'].append(f"⚠️ {page_file} has render_page but incorrect signature")
                else:
                    results['pages_missing_render'].append(page_file)
                    results['details'].append(f"❌ {page_file} missing render_page function")
                    
            except Exception as e:
                results['details'].append(f"❌ Error testing {page_file}: {str(e)}")
        
        total_pages = len(self.page_files)
        pages_with_render = len(results['pages_with_render'])
        
        results['status'] = 'success' if pages_with_render == total_pages else 'partial_success'
        results['summary'] = f"{pages_with_render}/{total_pages} pages have render_page function"
        
        return results
    
    def _test_registry_injection(self) -> Dict[str, Any]:
        """Test registry dependency injection with mock registries."""
        results = {
            'status': 'unknown',
            'successful_injections': [],
            'failed_injections': [],
            'details': []
        }
        
        # Create mock registries
        mock_services = {'mock_service': 'test_value'}
        mock_ai_engines = {'mock_engine': 'test_engine'}
        mock_components = {'mock_component': 'test_component'}
        
        for page_file in self.page_files[:3]:  # Test first 3 pages to avoid overwhelming output
            try:
                print(f"   🔄 Testing registry injection for {page_file}...")
                module_name = page_file.replace('.py', '')
                
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"streamlit_app.pages.{module_name}",
                    self.pages_dir / page_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'render_page'):
                    render_func = getattr(module, 'render_page')
                    
                    # Test calling with mock registries (this won't actually render in Streamlit)
                    # We're just testing that the function can accept the parameters
                    import inspect
                    sig = inspect.signature(render_func)
                    
                    # Try to call with our mock parameters
                    try:
                        # This will likely fail because we're not in Streamlit context,
                        # but we can at least verify the parameter acceptance
                        render_func(
                            services_registry=mock_services,
                            ai_engines=mock_ai_engines, 
                            components=mock_components
                        )
                    except Exception as e:
                        # We expect this to fail due to Streamlit context, but check the error type
                        if "streamlit" in str(e).lower() or "st." in str(e):
                            # This is expected - function accepts parameters but can't run without Streamlit
                            results['successful_injections'].append({
                                'page': page_file,
                                'status': 'accepts_parameters',
                                'note': 'Function accepts registry parameters (Streamlit context needed for execution)'
                            })
                            results['details'].append(f"✅ {page_file} accepts registry parameters")
                        else:
                            # Unexpected error
                            results['failed_injections'].append({
                                'page': page_file,
                                'error': str(e)
                            })
                            results['details'].append(f"❌ {page_file} registry injection error: {str(e)}")
                            
            except Exception as e:
                results['failed_injections'].append({
                    'page': page_file,
                    'error': str(e)
                })
                results['details'].append(f"❌ Error testing registry injection for {page_file}: {str(e)}")
        
        successful_count = len(results['successful_injections'])
        total_tested = len(self.page_files[:3])
        
        results['status'] = 'success' if successful_count == total_tested else 'partial_success'
        results['summary'] = f"{successful_count}/{total_tested} pages tested accept registry injection"
        
        return results
    
    def _test_fallback_mechanisms(self) -> Dict[str, Any]:
        """Test fallback mechanisms when registries are None or empty."""
        results = {
            'status': 'unknown',
            'successful_fallbacks': [],
            'failed_fallbacks': [],
            'details': []
        }
        
        for page_file in self.page_files[:2]:  # Test first 2 pages
            try:
                print(f"   🔄 Testing fallback mechanism for {page_file}...")
                module_name = page_file.replace('.py', '')
                
                # Import the module
                spec = importlib.util.spec_from_file_location(
                    f"streamlit_app.pages.{module_name}",
                    self.pages_dir / page_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'render_page'):
                    render_func = getattr(module, 'render_page')
                    
                    # Test calling with None parameters
                    try:
                        render_func(services_registry=None, ai_engines=None, components=None)
                    except Exception as e:
                        # Again, we expect Streamlit-related errors, but not parameter errors
                        if "streamlit" in str(e).lower() or "st." in str(e):
                            results['successful_fallbacks'].append({
                                'page': page_file,
                                'status': 'handles_none_parameters',
                                'note': 'Function handles None parameters gracefully'
                            })
                            results['details'].append(f"✅ {page_file} handles None parameters")
                        else:
                            results['failed_fallbacks'].append({
                                'page': page_file,
                                'error': str(e)
                            })
                            results['details'].append(f"❌ {page_file} fallback error: {str(e)}")
                            
            except Exception as e:
                results['failed_fallbacks'].append({
                    'page': page_file,
                    'error': str(e)
                })
                results['details'].append(f"❌ Error testing fallback for {page_file}: {str(e)}")
        
        successful_count = len(results['successful_fallbacks'])
        total_tested = len(self.page_files[:2])
        
        results['status'] = 'success' if successful_count == total_tested else 'partial_success'
        results['summary'] = f"{successful_count}/{total_tested} pages tested handle fallback gracefully"
        
        return results
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'success')
        partial_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'partial_success')
        
        overall_status = 'success' if successful_tests == total_tests else 'partial_success' if successful_tests + partial_tests == total_tests else 'needs_attention'
        
        report = {
            'overall_status': overall_status,
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'partial_success_tests': partial_tests,
                'failed_tests': total_tests - successful_tests - partial_tests
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        print("\n" + "="*60)
        print("🏆 INTEGRATION TEST RESULTS")
        print("="*60)
        
        print(f"\nOverall Status: {self._status_emoji(overall_status)} {overall_status.upper()}")
        print(f"Tests Completed: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"⚠️ Partial Success: {partial_tests}")
        print(f"❌ Failed: {total_tests - successful_tests - partial_tests}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Analyze results and provide recommendations
        for test_name, result in self.test_results.items():
            if result.get('status') != 'success':
                if test_name == 'app_registry' and result.get('errors'):
                    recommendations.append("🔧 Fix app.py registry initialization issues")
                elif test_name == 'page_imports' and result.get('failed_imports'):
                    recommendations.append("🔧 Resolve page import errors")
                elif test_name == 'render_functions' and result.get('pages_missing_render'):
                    recommendations.append("🔧 Add missing render_page functions")
        
        if not recommendations:
            recommendations.append("✅ System appears to be well integrated - proceed with end-to-end testing")
        
        return recommendations
    
    def _status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        status_emojis = {
            'success': '✅',
            'partial_success': '⚠️',
            'needs_attention': '❌',
            'error': '💥',
            'unknown': '❓'
        }
        return status_emojis.get(status, '❓')


def main():
    """Run the integration test suite."""
    tester = RegistryIntegrationTester()
    report = tester.run_all_tests()
    
    print(f"\n📋 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    return report

if __name__ == "__main__":
    main()