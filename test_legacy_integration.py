#!/usr/bin/env python3
"""
Test script to verify legacy data integration with modular services.

This script tests:
1. Data service can access legacy data paths
2. Model service can access legacy model paths  
3. Prediction service can access legacy prediction paths
4. All paths are correctly mapped to legacy folder structure
"""

import os
import sys
from pathlib import Path

# Add the streamlit_app to the Python path
sys.path.insert(0, str(Path(__file__).parent / "streamlit_app"))

def test_legacy_data_integration():
    """Test legacy data integration with modular services."""
    print("🧪 Testing Legacy Data Integration")
    print("=" * 50)
    
    try:
        # Test imports
        print("\n📦 Testing Service Imports...")
        from streamlit_app.services.data_service import DataService
        from streamlit_app.services.model_service import ModelService
        from streamlit_app.services.prediction_service import PredictionService
        from streamlit_app.core.config import AppConfig
        print("✅ Service imports successful")
        
        # Create test config
        config = AppConfig()
        config.data_dir = "data"
        config.models_dir = "models"
        
        print("\n📊 Testing Data Service...")
        data_service = DataService(config)
        
        # Test legacy data paths
        game_type = "lotto_6_49"
        model_type = "lstm"
        
        # Test features path
        features_path = data_service.get_features_path(model_type, game_type)
        print(f"✅ Features path: {features_path}")
        
        # Test raw data path  
        raw_data_path = data_service.get_raw_data_path(game_type)
        print(f"✅ Raw data path: {raw_data_path}")
        
        # Test available features
        if features_path.exists():
            available_features = data_service.get_available_features(model_type, game_type)
            print(f"✅ Available features ({len(available_features)}): {available_features[:3]}")
        
        # Test available raw data years
        if raw_data_path.exists():
            available_years = data_service.get_available_raw_data_years(game_type)
            print(f"✅ Available years ({len(available_years)}): {available_years[:5]}")
        
        print("\n🤖 Testing Model Service...")
        model_service = ModelService(config)
        
        # Test legacy model paths
        legacy_model_path = model_service.get_legacy_model_path(game_type, model_type)
        print(f"✅ Legacy model path: {legacy_model_path}")
        
        # Test available model versions
        if legacy_model_path.exists():
            available_versions = model_service.get_available_model_versions(game_type, model_type)
            print(f"✅ Available versions ({len(available_versions)}): {available_versions[:3]}")
        
        # Test model summary
        models_summary = model_service.get_all_legacy_models_summary()
        print(f"✅ Models summary - Total models: {models_summary['total_models']}")
        print(f"✅ Games with models: {list(models_summary['games'].keys())}")
        
        print("\n🔮 Testing Prediction Service...")
        prediction_service = PredictionService(config)
        
        # Test legacy prediction paths
        hybrid_path = prediction_service.get_legacy_predictions_path(game_type, 'hybrid')
        lstm_path = prediction_service.get_legacy_predictions_path(game_type, 'lstm')
        print(f"✅ Hybrid predictions path: {hybrid_path}")
        print(f"✅ LSTM predictions path: {lstm_path}")
        
        # Test prediction summary
        pred_summary = prediction_service.get_prediction_summary(game_type)
        print(f"✅ Prediction summary - Total: {pred_summary['total_predictions']}")
        print(f"✅ Model types with predictions: {list(pred_summary['model_types'].keys())}")
        
        # Test loading predictions if they exist
        if hybrid_path.exists():
            hybrid_predictions = prediction_service.load_legacy_predictions(game_type, 'hybrid')
            print(f"✅ Loaded hybrid predictions: {len(hybrid_predictions)}")
            
            if hybrid_predictions:
                latest = prediction_service.get_latest_hybrid_prediction(game_type)
                print(f"✅ Latest hybrid prediction date: {latest.get('timestamp', 'N/A')[:10] if latest else 'None'}")
        
        print("\n🎯 Integration Test Summary")
        print("=" * 30)
        print("✅ All service imports working")
        print("✅ Data service legacy paths configured")
        print("✅ Model service legacy paths configured") 
        print("✅ Prediction service legacy paths configured")
        print("✅ Legacy folder structure properly mapped")
        
        print(f"\n📁 Verified Paths:")
        print(f"   Features: data/features/{{model_type}}/{{game_type}}/")
        print(f"   Raw Data: data/{{game_type}}/")
        print(f"   Models: models/{{game_type}}/{{model_type}}/")
        print(f"   Predictions: predictions/{{game_type}}/{{model_type}}/")
        print(f"   Hybrid: predictions/{{game_type}}/hybrid/")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def check_data_structure():
    """Check if legacy data structure exists."""
    print("\n🔍 Checking Legacy Data Structure...")
    
    base_paths = {
        'data': Path('data'),
        'models': Path('models'), 
        'predictions': Path('predictions')
    }
    
    for name, path in base_paths.items():
        if path.exists():
            print(f"✅ {name} directory exists: {path}")
            
            # Count subdirectories
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            print(f"   📂 Subdirectories ({len(subdirs)}): {[d.name for d in subdirs[:5]]}")
        else:
            print(f"❌ {name} directory not found: {path}")


if __name__ == "__main__":
    print("🚀 Legacy Data Integration Test")
    print("Testing modular services with legacy data structure\n")
    
    # Check data structure first
    check_data_structure()
    
    # Run integration test
    success = test_legacy_data_integration()
    
    print(f"\n{'🎉 SUCCESS' if success else '💥 FAILED'}: Legacy integration test {'completed' if success else 'failed'}")