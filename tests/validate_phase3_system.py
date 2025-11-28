"""
System Validation Script for Phase 3 AI Engine Modularization

This script validates the current AI system capabilities and tests
the existing components for functionality.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_system_components():
    """Validate all system components"""
    logger.info("🔍 Starting Phase 3 AI Engine Modularization system validation")
    
    validation_results = {
        'ai_engines': {},
        'orchestration': {},
        'model_interface': {},
        'visualization': {},
        'registry': {},
        'overall_status': 'unknown'
    }
    
    # Test AI Engines
    logger.info("📊 Testing AI Engines...")
    try:
        from streamlit_app.ai_engines.phase1_mathematical import MathematicalEngine
        math_engine = MathematicalEngine()
        validation_results['ai_engines']['phase1_mathematical'] = '✅ Available'
        logger.info("✅ Phase 1 Mathematical Engine - Available")
        
        # Test basic functionality
        sample_data = [[1, 2, 3, 4, 5, 6] for _ in range(10)]
        insights = math_engine.get_mathematical_insights(sample_data)
        if 'high_confidence_numbers' in insights:
            validation_results['ai_engines']['phase1_mathematical'] += ' & Functional'
            logger.info("✅ Phase 1 Mathematical Engine - Functional")
        
    except Exception as e:
        validation_results['ai_engines']['phase1_mathematical'] = f'❌ Error: {e}'
        logger.error(f"❌ Phase 1 Mathematical Engine - Error: {e}")
    
    try:
        from streamlit_app.ai_engines.phase2_expert_ensemble import ExpertEnsembleEngine
        expert_engine = ExpertEnsembleEngine()
        validation_results['ai_engines']['phase2_expert_ensemble'] = '✅ Available'
        logger.info("✅ Phase 2 Expert Ensemble Engine - Available")
        
    except Exception as e:
        validation_results['ai_engines']['phase2_expert_ensemble'] = f'❌ Error: {e}'
        logger.error(f"❌ Phase 2 Expert Ensemble Engine - Error: {e}")
    
    try:
        from streamlit_app.ai_engines.phase3_set_optimizer import SetOptimizationEngine
        set_engine = SetOptimizationEngine()
        validation_results['ai_engines']['phase3_set_optimizer'] = '✅ Available'
        logger.info("✅ Phase 3 Set Optimization Engine - Available")
        
    except Exception as e:
        validation_results['ai_engines']['phase3_set_optimizer'] = f'❌ Error: {e}'
        logger.error(f"❌ Phase 3 Set Optimization Engine - Error: {e}")
    
    try:
        from streamlit_app.ai_engines.phase4_temporal import TemporalEngine
        temporal_engine = TemporalEngine()
        validation_results['ai_engines']['phase4_temporal'] = '✅ Available'
        logger.info("✅ Phase 4 Temporal Engine - Available")
        
    except Exception as e:
        validation_results['ai_engines']['phase4_temporal'] = f'❌ Error: {e}'
        logger.error(f"❌ Phase 4 Temporal Engine - Error: {e}")
    
    # Test Orchestration System
    logger.info("🎯 Testing Orchestration System...")
    try:
        from streamlit_app.ai_engines.prediction_orchestrator import PredictionOrchestrator
        orchestrator = PredictionOrchestrator()
        validation_results['orchestration']['prediction_orchestrator'] = '✅ Available'
        logger.info("✅ Prediction Orchestrator - Available")
        
    except Exception as e:
        validation_results['orchestration']['prediction_orchestrator'] = f'❌ Error: {e}'
        logger.error(f"❌ Prediction Orchestrator - Error: {e}")
    
    # Test Model Interface
    logger.info("🔧 Testing Model Interface...")
    try:
        from streamlit_app.ai_engines.model_interface import BaseModel
        model_interface = BaseModel()
        validation_results['model_interface']['base_model'] = '✅ Available'
        logger.info("✅ Base Model Interface - Available")
        
    except Exception as e:
        validation_results['model_interface']['base_model'] = f'❌ Error: {e}'
        logger.error(f"❌ Base Model Interface - Error: {e}")
    
    # Test Visualization Components
    logger.info("📊 Testing Visualization Components...")
    try:
        from streamlit_app.components.data_visualizations import create_performance_dashboard
        validation_results['visualization']['performance_dashboard'] = '✅ Available'
        logger.info("✅ Performance Dashboard - Available")
        
        # Test function with sample data
        sample_metrics = {
            'accuracy': [0.75, 0.80, 0.78, 0.82],
            'confidence': [0.70, 0.75, 0.73, 0.77]
        }
        dashboard = create_performance_dashboard(sample_metrics)
        if dashboard:
            validation_results['visualization']['performance_dashboard'] += ' & Functional'
            logger.info("✅ Performance Dashboard - Functional")
        
    except Exception as e:
        validation_results['visualization']['performance_dashboard'] = f'❌ Error: {e}'
        logger.error(f"❌ Performance Dashboard - Error: {e}")
    
    # Test Engine Registry
    logger.info("📝 Testing Engine Registry...")
    try:
        from streamlit_app.ai_engines.engine_registry import get_engine_registry
        registry = get_engine_registry()
        validation_results['registry']['engine_registry'] = '✅ Available'
        logger.info("✅ Engine Registry - Available")
        
        # Test registry functionality
        status = registry.get_registry_status()
        if isinstance(status, dict):
            validation_results['registry']['engine_registry'] += ' & Functional'
            logger.info("✅ Engine Registry - Functional")
        
    except Exception as e:
        validation_results['registry']['engine_registry'] = f'❌ Error: {e}'
        logger.error(f"❌ Engine Registry - Error: {e}")
    
    # Calculate overall status
    successful_components = 0
    total_components = 0
    
    for category in validation_results:
        if category == 'overall_status':
            continue
        for component, status in validation_results[category].items():
            total_components += 1
            if '✅' in status:
                successful_components += 1
    
    success_rate = (successful_components / total_components) * 100 if total_components > 0 else 0
    
    if success_rate >= 80:
        validation_results['overall_status'] = '🎉 Excellent'
    elif success_rate >= 60:
        validation_results['overall_status'] = '✅ Good'
    elif success_rate >= 40:
        validation_results['overall_status'] = '⚠️ Moderate'
    else:
        validation_results['overall_status'] = '❌ Poor'
    
    # Print summary
    logger.info("="*60)
    logger.info("📋 PHASE 3 AI ENGINE MODULARIZATION VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"🏆 Overall Status: {validation_results['overall_status']}")
    logger.info(f"📊 Success Rate: {success_rate:.1f}% ({successful_components}/{total_components} components)")
    logger.info("")
    
    logger.info("🤖 AI Engines:")
    for engine, status in validation_results['ai_engines'].items():
        logger.info(f"   {engine}: {status}")
    
    logger.info("🎯 Orchestration:")
    for component, status in validation_results['orchestration'].items():
        logger.info(f"   {component}: {status}")
    
    logger.info("🔧 Model Interface:")
    for component, status in validation_results['model_interface'].items():
        logger.info(f"   {component}: {status}")
    
    logger.info("📊 Visualization:")
    for component, status in validation_results['visualization'].items():
        logger.info(f"   {component}: {status}")
    
    logger.info("📝 Registry:")
    for component, status in validation_results['registry'].items():
        logger.info(f"   {component}: {status}")
    
    logger.info("="*60)
    
    return validation_results

def test_integration_workflow():
    """Test integration workflow between components"""
    logger.info("🔗 Testing integration workflow...")
    
    try:
        # Test orchestrated prediction workflow
        logger.info("🎯 Testing orchestrated prediction workflow...")
        
        from streamlit_app.ai_engines.prediction_orchestrator import PredictionOrchestrator
        from streamlit_app.ai_engines.phase1_mathematical import MathematicalEngine
        
        orchestrator = PredictionOrchestrator()
        math_engine = MathematicalEngine()
        
        # Test with sample data
        sample_data = [[i, i+1, i+2, i+3, i+4, i+5] for i in range(1, 11)]
        
        # Get predictions from individual engine
        math_insights = math_engine.get_mathematical_insights(sample_data)
        
        if 'high_confidence_numbers' in math_insights:
            logger.info("✅ Individual engine prediction successful")
            
            # Test orchestrator (basic functionality)
            try:
                orchestrator_result = orchestrator.orchestrate_predictions(
                    engines={'mathematical': math_engine},
                    historical_data=sample_data
                )
                
                if orchestrator_result and 'predictions' in orchestrator_result:
                    logger.info("✅ Orchestrated prediction successful")
                    return True
                else:
                    logger.warning("⚠️ Orchestrator returned empty or invalid results")
                    return False
                    
            except Exception as e:
                logger.warning(f"⚠️ Orchestrator test failed: {e}")
                return False
        else:
            logger.warning("⚠️ Individual engine prediction failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Integration workflow test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Phase 3 AI Engine Modularization System Validation")
    print("=" * 70)
    
    # Run system validation
    validation_results = validate_system_components()
    
    # Run integration test
    integration_success = test_integration_workflow()
    
    # Final summary
    print("\n🏁 FINAL VALIDATION SUMMARY")
    print("=" * 40)
    print(f"System Status: {validation_results['overall_status']}")
    print(f"Integration Test: {'✅ Passed' if integration_success else '❌ Failed'}")
    
    # Determine overall success
    overall_success = (
        validation_results['overall_status'] in ['🎉 Excellent', '✅ Good'] and
        integration_success
    )
    
    if overall_success:
        print("🎉 Phase 3 AI Engine Modularization validation SUCCESSFUL!")
        return True
    else:
        print("⚠️ Phase 3 AI Engine Modularization validation completed with issues.")
        print("💡 Recommendation: Review failed components and enhance system integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)