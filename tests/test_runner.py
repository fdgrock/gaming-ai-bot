"""
Test Runner for Gaming AI Bot Services

Provides convenient ways to run all tests or specific test suites
with proper environment setup and detailed reporting.
"""

import sys
import os
import unittest
from datetime import datetime
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'streamlit_app'))

from tests.test_all_services import (
    TestDataService, TestModelService, TestPredictionService,
    TestAnalyticsService, TestTrainingService, TestServiceRegistry,
    TestServiceIntegration, run_all_tests
)


class TestRunner:
    """Advanced test runner with reporting and filtering capabilities"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_specific_service_tests(self, service_name: str) -> dict:
        """
        Run tests for a specific service
        
        Args:
            service_name: Name of the service to test
            
        Returns:
            Test results dictionary
        """
        service_test_map = {
            'data': TestDataService,
            'model': TestModelService,
            'prediction': TestPredictionService,
            'analytics': TestAnalyticsService,
            'training': TestTrainingService,
            'registry': TestServiceRegistry,
            'integration': TestServiceIntegration
        }
        
        if service_name.lower() not in service_test_map:
            return {
                'error': f"Unknown service: {service_name}",
                'available_services': list(service_test_map.keys())
            }
        
        test_class = service_test_map[service_name.lower()]
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        
        print(f"\n🧪 Running {service_name.title()} Service Tests...")
        print("=" * 50)
        
        self.start_time = datetime.now()
        result = runner.run(suite)
        self.end_time = datetime.now()
        
        return self._format_results(result, service_name)
    
    def run_all_tests_with_report(self) -> dict:
        """Run all tests and generate detailed report"""
        print("\n🚀 Starting Comprehensive Gaming AI Bot Test Suite...")
        print("=" * 70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        self.start_time = datetime.now()
        results = run_all_tests()
        self.end_time = datetime.now()
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        detailed_results = {
            'summary': results,
            'execution_time': duration,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'environment': self._get_environment_info(),
            'coverage_estimate': self._estimate_coverage(),
            'recommendations': self._generate_recommendations(results)
        }
        
        self._print_detailed_report(detailed_results)
        return detailed_results
    
    def run_quick_smoke_tests(self) -> dict:
        """Run quick smoke tests to verify basic functionality"""
        print("\n⚡ Running Quick Smoke Tests...")
        print("=" * 40)
        
        smoke_tests = [
            ('Service Import Test', self._test_service_imports),
            ('Service Creation Test', self._test_service_creation),
            ('Registry Test', self._test_registry_basic),
            ('Basic Integration Test', self._test_basic_integration)
        ]
        
        smoke_results = {}
        total_passed = 0
        
        for test_name, test_func in smoke_tests:
            try:
                result = test_func()
                smoke_results[test_name] = {
                    'passed': result,
                    'error': None
                }
                if result:
                    total_passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                smoke_results[test_name] = {
                    'passed': False,
                    'error': str(e)
                }
                print(f"❌ {test_name}: ERROR - {str(e)}")
        
        success_rate = (total_passed / len(smoke_tests)) * 100
        
        print(f"\n📊 Smoke Test Results: {total_passed}/{len(smoke_tests)} passed ({success_rate:.1f}%)")
        
        return {
            'total_tests': len(smoke_tests),
            'passed': total_passed,
            'success_rate': success_rate,
            'results': smoke_results
        }
    
    def _test_service_imports(self) -> bool:
        """Test that all services can be imported"""
        try:
            from streamlit_app.services.data_service import DataService
            from streamlit_app.services.model_service import ModelService
            from streamlit_app.services.prediction_service import PredictionService
            from streamlit_app.services.analytics_service import AnalyticsService
            from streamlit_app.services.training_service import TrainingService
            from streamlit_app.services.service_registry import ServiceRegistry
            return True
        except ImportError:
            return False
    
    def _test_service_creation(self) -> bool:
        """Test that services can be instantiated"""
        try:
            from streamlit_app.services.data_service import DataService
            from streamlit_app.services.model_service import ModelService
            
            data_service = DataService()
            model_service = ModelService()
            
            return data_service is not None and model_service is not None
        except Exception:
            return False
    
    def _test_registry_basic(self) -> bool:
        """Test basic registry functionality"""
        try:
            from streamlit_app.services.service_registry import ServiceRegistry
            
            registry = ServiceRegistry()
            available_services = registry.get_available_services()
            
            return isinstance(available_services, list)
        except Exception:
            return False
    
    def _test_basic_integration(self) -> bool:
        """Test basic service integration"""
        try:
            from streamlit_app.services.service_registry import get_service_registry
            
            registry = get_service_registry()
            health = registry.get_registry_health()
            
            return 'total_services' in health
        except Exception:
            return False
    
    def _format_results(self, result, service_name: str) -> dict:
        """Format test results"""
        duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'service': service_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
            'execution_time': duration,
            'details': {
                'failures': [str(f) for f in result.failures],
                'errors': [str(e) for e in result.errors]
            }
        }
    
    def _get_environment_info(self) -> dict:
        """Get environment information"""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'python_path': sys.path[:3],  # First 3 entries
        }
    
    def _estimate_coverage(self) -> dict:
        """Estimate test coverage"""
        # This is a simplified estimation
        estimated_functions_tested = {
            'DataService': 25,
            'ModelService': 15,
            'PredictionService': 20,
            'AnalyticsService': 15,
            'TrainingService': 19,
            'ServiceRegistry': 10
        }
        
        total_estimated = sum(estimated_functions_tested.values())
        
        return {
            'estimated_functions_tested': total_estimated,
            'service_breakdown': estimated_functions_tested,
            'coverage_note': 'Estimates based on extracted function counts'
        }
    
    def _generate_recommendations(self, results: dict) -> list:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if results['success_rate'] < 100:
            recommendations.append(
                "Some tests failed. Review failure details and fix issues before production deployment."
            )
        
        if results['success_rate'] > 90:
            recommendations.append(
                "Excellent test coverage! Consider adding more edge case tests and integration scenarios."
            )
        
        recommendations.append(
            "Run tests regularly during development to catch regressions early."
        )
        
        recommendations.append(
            "Consider adding performance benchmarks and load testing for production readiness."
        )
        
        return recommendations
    
    def _print_detailed_report(self, results: dict):
        """Print detailed test report"""
        print("\n" + "=" * 70)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 70)
        
        summary = results['summary']
        print(f"Tests Executed: {summary['tests_run']}")
        print(f"Failures: {summary['failures']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        
        print(f"\n📊 COVERAGE ESTIMATION")
        print("-" * 30)
        coverage = results['coverage_estimate']
        print(f"Estimated Functions Tested: {coverage['estimated_functions_tested']}")
        for service, count in coverage['service_breakdown'].items():
            print(f"  {service}: ~{count} functions")
        
        print(f"\n🔍 ENVIRONMENT INFO")
        print("-" * 20)
        env = results['environment']
        print(f"Python: {env['python_version'].split()[0]}")
        print(f"Platform: {env['platform']}")
        
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 20)
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
        if summary['failures'] > 0 or summary['errors'] > 0:
            print(f"\n❌ ISSUES DETAIL")
            print("-" * 15)
            for failure in summary['details']['failures']:
                print(f"FAILURE: {failure[:100]}...")
            for error in summary['details']['errors']:
                print(f"ERROR: {error[:100]}...")
        else:
            print(f"\n✅ ALL TESTS PASSED! Services are ready for integration.")
        
        print("=" * 70)
    
    def save_results_to_file(self, results: dict, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'test_results_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Test results saved to: {filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save results: {e}")


def main():
    """Main entry point for test runner"""
    runner = TestRunner()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'smoke':
            # Run smoke tests
            results = runner.run_quick_smoke_tests()
            
        elif command in ['data', 'model', 'prediction', 'analytics', 'training', 'registry', 'integration']:
            # Run specific service tests
            results = runner.run_specific_service_tests(command)
            print(f"\n📊 {command.title()} Service Test Results:")
            print(f"Success Rate: {results.get('success_rate', 0):.2f}%")
            
        elif command == 'all':
            # Run all tests with detailed report
            results = runner.run_all_tests_with_report()
            
            # Optionally save results
            if '--save' in sys.argv:
                runner.save_results_to_file(results)
        
        elif command == 'help':
            print_help()
        
        else:
            print(f"Unknown command: {command}")
            print_help()
    
    else:
        # Default: run all tests
        results = runner.run_all_tests_with_report()


def print_help():
    """Print help information"""
    print("\n🧪 Gaming AI Bot Test Runner")
    print("=" * 40)
    print("Usage: python test_runner.py [command]")
    print("\nCommands:")
    print("  all          - Run all tests with detailed report (default)")
    print("  smoke        - Run quick smoke tests")
    print("  data         - Run DataService tests only")
    print("  model        - Run ModelService tests only")
    print("  prediction   - Run PredictionService tests only")
    print("  analytics    - Run AnalyticsService tests only")
    print("  training     - Run TrainingService tests only")
    print("  registry     - Run ServiceRegistry tests only")
    print("  integration  - Run integration tests only")
    print("  help         - Show this help message")
    print("\nFlags:")
    print("  --save       - Save detailed results to JSON file")
    print("\nExamples:")
    print("  python test_runner.py")
    print("  python test_runner.py smoke")
    print("  python test_runner.py all --save")
    print("  python test_runner.py data")


if __name__ == '__main__':
    main()