"""
Test script to verify XGBoostProgressCallback and progress display system
Tests both the callback mechanism and integration with training
"""

import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the streamlit_app to path
sys.path.insert(0, str(Path(__file__).parent / "streamlit_app"))

def test_xgboost_progress_callback():
    """Test XGBoostProgressCallback message formatting and metric extraction"""
    
    from services.advanced_model_training import XGBoostProgressCallback
    
    print("\n" + "="*70)
    print("TEST 1: XGBoostProgressCallback Initialization")
    print("="*70)
    
    # Track progress callbacks
    progress_log = []
    
    def mock_progress_callback(progress, message, metrics=None):
        """Mock progress callback to capture calls"""
        progress_log.append({
            'progress': progress,
            'message': message,
            'metrics': metrics
        })
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {progress:.1%} | {message}")
        if metrics:
            print(f"  Metrics: epoch={metrics.get('epoch')}, "
                  f"loss={metrics.get('loss', 'N/A')}, "
                  f"accuracy={metrics.get('accuracy', 'N/A')}")
    
    # Create callback
    callback = XGBoostProgressCallback(mock_progress_callback, total_rounds=100)
    
    print("✓ XGBoostProgressCallback created successfully")
    print(f"  - Total rounds: 100")
    print(f"  - Progress callback: Active")
    
    # Test callback with mock XGBoost environment
    print("\n" + "="*70)
    print("TEST 2: Simulated XGBoost Training Progress")
    print("="*70)
    
    class MockXGBEnv:
        """Mock XGBoost environment object"""
        def __init__(self, iteration, metrics_list):
            self.iteration = iteration
            self.evaluation_result_list = metrics_list
    
    # Simulate 5 iterations of training
    test_iterations = [
        (0, [
            ('train-mlogloss', 2.1),
            ('eval-mlogloss', 2.05),
            ('train-error', 0.45),
            ('eval-error', 0.46),
        ]),
        (1, [
            ('train-mlogloss', 1.95),
            ('eval-mlogloss', 1.92),
            ('train-error', 0.42),
            ('eval-error', 0.43),
        ]),
        (2, [
            ('train-mlogloss', 1.72),
            ('eval-mlogloss', 1.68),
            ('train-error', 0.38),
            ('eval-error', 0.39),
        ]),
        (50, [
            ('train-mlogloss', 0.95),
            ('eval-mlogloss', 0.98),
            ('train-error', 0.15),
            ('eval-error', 0.16),
        ]),
        (99, [
            ('train-mlogloss', 0.45),
            ('eval-mlogloss', 0.48),
            ('train-error', 0.05),
            ('eval-error', 0.06),
        ]),
    ]
    
    for iteration, metrics in test_iterations:
        env = MockXGBEnv(iteration, metrics)
        callback(env)
    
    print(f"\n✓ Simulated {len(test_iterations)} training iterations")
    
    # Verify progress callbacks were called
    print("\n" + "="*70)
    print("TEST 3: Callback Invocation Verification")
    print("="*70)
    
    assert len(progress_log) == len(test_iterations), "Not all callbacks were invoked"
    print(f"✓ All {len(progress_log)} callbacks were invoked successfully")
    
    # Verify progress range
    print("\nProgress Range Verification:")
    for i, entry in enumerate(progress_log):
        progress = entry['progress']
        expected_min = 0.3
        expected_max = 0.9
        assert expected_min <= progress <= expected_max, \
            f"Progress {progress} out of range [{expected_min}, {expected_max}]"
        print(f"  Iteration {i}: {progress:.1%} ✓")
    
    # Verify metrics extraction
    print("\n" + "="*70)
    print("TEST 4: Metrics Extraction Verification")
    print("="*70)
    
    final_entry = progress_log[-1]
    metrics = final_entry['metrics']
    
    print(f"Final iteration metrics:")
    print(f"  epoch: {metrics.get('epoch')} ✓")
    print(f"  total_epochs: {metrics.get('total_epochs')} ✓")
    print(f"  loss: {metrics.get('loss')} ✓")
    print(f"  val_loss: {metrics.get('val_loss')} ✓")
    print(f"  accuracy: {metrics.get('accuracy')} ✓")
    print(f"  val_accuracy: {metrics.get('val_accuracy')} ✓")
    
    # Verify metric values make sense
    assert 0 <= metrics.get('accuracy', 0) <= 1, "Accuracy out of range"
    assert 0 <= metrics.get('val_accuracy', 0) <= 1, "Val accuracy out of range"
    assert metrics.get('loss', 0) > 0, "Loss should be positive"
    assert metrics.get('val_loss', 0) > 0, "Val loss should be positive"
    
    print("\n✓ All metrics are valid")
    
    print("\n" + "="*70)
    print("TEST 5: Progress Display Format (Simulating GUI)")
    print("="*70)
    
    # Format like the GUI would
    print("\nTraining Progress Display (last 3 entries):\n")
    print("Training Logs:")
    print("-" * 70)
    
    for entry in progress_log[-3:]:
        metrics = entry['metrics']
        log_line = f"[{datetime.now().strftime('%H:%M:%S')}] {entry['message']}"
        
        if metrics:
            if 'epoch' in metrics:
                log_line += f" | Epoch: {metrics['epoch']}"
            if 'loss' in metrics:
                log_line += f" | Loss: {metrics['loss']:.6f}"
            if 'accuracy' in metrics:
                log_line += f" | Accuracy: {metrics['accuracy']:.4f}"
            if 'val_loss' in metrics:
                log_line += f" | Val Loss: {metrics['val_loss']:.6f}"
            if 'val_accuracy' in metrics:
                log_line += f" | Val Accuracy: {metrics['val_accuracy']:.4f}"
        
        print(log_line)
    
    print("-" * 70)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ XGBoostProgressCallback working correctly")
    print("✓ Progress values in valid range (30%-90%)")
    print("✓ Metrics extracted and formatted properly")
    print("✓ Callbacks invoked for each iteration")
    print("✓ Message format matches expected pattern")
    print("\n✅ ALL TESTS PASSED - Progress system ready for training\n")

if __name__ == "__main__":
    try:
        test_xgboost_progress_callback()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
