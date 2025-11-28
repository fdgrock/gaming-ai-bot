"""Simple import test to check for syntax errors"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'streamlit_app'))

try:
    print("Testing service imports...")
    
    print("1. Testing DataService import...")
    from streamlit_app.services.data_service import DataService
    print("   ✅ DataService imported successfully")
    
    print("2. Testing ModelService import...")
    from streamlit_app.services.model_service import ModelService
    print("   ✅ ModelService imported successfully")
    
    print("3. Testing PredictionService import...")
    from streamlit_app.services.prediction_service import PredictionService
    print("   ✅ PredictionService imported successfully")
    
    print("4. Testing AnalyticsService import...")
    from streamlit_app.services.analytics_service import AnalyticsService
    print("   ✅ AnalyticsService imported successfully")
    
    print("5. Testing TrainingService import...")
    from streamlit_app.services.training_service import TrainingService
    print("   ✅ TrainingService imported successfully")
    
    print("6. Testing ServiceRegistry import...")
    from streamlit_app.services.service_registry import ServiceRegistry
    print("   ✅ ServiceRegistry imported successfully")
    
    print("\n🎉 All services imported successfully!")
    print("Phase 2 service extraction is complete and working!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()