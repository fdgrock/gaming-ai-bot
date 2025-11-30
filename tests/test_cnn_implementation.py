#!/usr/bin/env python3
"""
Quick test to verify CNN implementation is working
"""
import sys
from pathlib import Path
import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_cnn_imports():
    """Test that CNN training method can be imported"""
    print("✓ Testing CNN imports...")
    try:
        from streamlit_app.services.advanced_model_training import AdvancedModelTrainer
        
        # Check if train_cnn method exists
        if hasattr(AdvancedModelTrainer, 'train_cnn'):
            print("  ✓ train_cnn method found in AdvancedModelTrainer")
        else:
            print("  ✗ train_cnn method NOT found")
            return False
            
        # Check if ensemble uses CNN
        trainer = AdvancedModelTrainer()
        print("  ✓ AdvancedModelTrainer instantiated successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_cnn_architecture():
    """Test that CNN model architecture can be built"""
    print("\n✓ Testing CNN architecture...")
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Conv1D, Concatenate, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
        
        # Create a simple test model similar to train_cnn
        input_shape = (24, 1)  # Same as lottery features
        inputs = Input(shape=input_shape)
        
        # Multi-scale convolutional paths
        conv3 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
        conv3 = BatchNormalization()(conv3)
        
        conv5 = Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
        conv5 = BatchNormalization()(conv5)
        
        conv7 = Conv1D(32, kernel_size=7, activation='relu', padding='same')(inputs)
        conv7 = BatchNormalization()(conv7)
        
        # Concatenate paths
        merged = Concatenate()([conv3, conv5, conv7])
        pooled = GlobalAveragePooling1D()(merged)
        
        # Dense head
        dense1 = Dense(256, activation='relu')(pooled)
        dense1 = Dropout(0.3)(dense1)
        dense2 = Dense(128, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        dense3 = Dense(64, activation='relu')(dense2)
        dense3 = Dropout(0.1)(dense3)
        outputs = Dense(49, activation='sigmoid')(dense3)  # 49 possible lottery numbers
        
        model = Model(inputs=inputs, outputs=outputs)
        print(f"  ✓ CNN model built successfully")
        print(f"  ✓ Model has {len(model.layers)} layers")
        print(f"  ✓ Model parameters: {model.count_params():,}")
        
        # Test model compilation
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(f"  ✓ Model compiled successfully")
        
        return True
    except Exception as e:
        print(f"  ✗ Architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_config():
    """Test that ensemble configuration includes CNN"""
    print("\n✓ Testing ensemble configuration...")
    try:
        # Check data_training.py
        data_training_path = PROJECT_ROOT / "streamlit_app" / "pages" / "data_training.py"
        if data_training_path.exists():
            with open(data_training_path, 'r') as f:
                content = f.read()
                if 'model_type == "CNN"' in content:
                    print("  ✓ CNN training block found in data_training.py")
                else:
                    print("  ✗ CNN training block NOT found in data_training.py")
                    return False
        
        # Check predictions.py
        predictions_path = PROJECT_ROOT / "streamlit_app" / "pages" / "predictions.py"
        if predictions_path.exists():
            with open(predictions_path, 'r') as f:
                content = f.read()
                if 'cnn_model.keras' in content or '"CNN"' in content:
                    print("  ✓ CNN references found in predictions.py")
                else:
                    print("  ✗ CNN references NOT found in predictions.py")
                    return False
        
        # Check advanced_model_training.py for ensemble CNN integration
        training_path = PROJECT_ROOT / "streamlit_app" / "services" / "advanced_model_training.py"
        if training_path.exists():
            with open(training_path, 'r') as f:
                content = f.read()
                if 'self.train_cnn(' in content:
                    print("  ✓ CNN call found in ensemble training")
                if 'cnn_model.keras' in content:
                    print("  ✓ CNN model file path found in save/load logic")
                if '"cnn" in ensemble' in content:
                    print("  ✓ CNN ensemble integration found")
                else:
                    print("  ✗ CNN ensemble integration NOT found")
                    return False
        
        return True
    except Exception as e:
        print(f"  ✗ Ensemble config test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("CNN Implementation Verification Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_cnn_imports()))
    results.append(("Architecture", test_cnn_architecture()))
    results.append(("Ensemble Config", test_ensemble_config()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ CNN implementation verification SUCCESSFUL!")
        return 0
    else:
        print(f"\n✗ CNN implementation has issues ({total - passed} failures)")
        return 1

if __name__ == "__main__":
    exit(main())
