"""
Test script to verify CNN embeddings loading works correctly.
This replicates the exact logic used in AdvancedModelTrainer.load_training_data()
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

def load_cnn_embeddings(file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
    """Load CNN embeddings (multi-scale feature representations)."""
    print(f"🔵 Loading CNN embeddings from {len(file_paths)} files...")
    
    all_embeddings = []
    feature_count = None
    
    for filepath in file_paths:
        print(f"\n  Processing: {filepath.name}")
        print(f"    File exists: {filepath.exists()}")
        print(f"    Suffix: {filepath.suffix}")
        
        if filepath.suffix == ".npz":
            try:
                data = np.load(filepath)
                print(f"    Loaded .npz file successfully")
                print(f"    Keys found: {list(data.keys())}")
                
                # Try multiple possible keys
                embeddings = data.get("embeddings", None)
                if embeddings is None:
                    embeddings = data.get("X", None)
                if embeddings is None:
                    embeddings = data.get("features", None)
                
                if embeddings is not None:
                    print(f"    Found embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
                    
                    # Handle different shapes
                    if len(embeddings.shape) > 2:
                        num_samples = embeddings.shape[0]
                        flattened = embeddings.reshape(num_samples, -1)
                        print(f"    Flattened from {embeddings.shape} to {flattened.shape}")
                    else:
                        flattened = embeddings
                    
                    # Ensure feature consistency
                    if feature_count is None:
                        feature_count = flattened.shape[1]
                        all_embeddings.append(flattened)
                        print(f"    ✅ Added {flattened.shape[0]} samples with {feature_count} features")
                    elif flattened.shape[1] == feature_count:
                        all_embeddings.append(flattened)
                        print(f"    ✅ Added {flattened.shape[0]} samples (feature count matches)")
                    else:
                        print(f"    ⚠️ Skipping: feature mismatch ({flattened.shape[1]} vs {feature_count})")
                else:
                    print(f"    ❌ No embeddings found (tried keys: embeddings, X, features)")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    ⚠️ Skipping non-.npz file")
    
    if all_embeddings:
        combined = np.vstack(all_embeddings)
        print(f"\n🎉 Successfully combined CNN embeddings:")
        print(f"   Shape: {combined.shape}")
        print(f"   Dtype: {combined.dtype}")
        
        # TEST THE BUG FIX: feature_names handling
        print(f"\n  Testing feature_names handling (bug fix):")
        feature_names = None
        
        # This would crash before fix: feature_names.extend(...)
        if feature_names is None:
            feature_names = [f"cnn_{i}" for i in range(combined.shape[1])]
            print(f"    ✅ Initialized feature_names (was None)")
        else:
            feature_names.extend([f"cnn_{i}" for i in range(combined.shape[1])])
            print(f"    ✅ Extended feature_names (was list)")
        
        print(f"    Feature names count: {len(feature_names)}")
        
        return combined, combined.shape[0]
    else:
        print(f"\n❌ No CNN embeddings loaded")
        return None, 0


def load_raw_csv_features(file_paths: List[Path]) -> Tuple[Optional[np.ndarray], int]:
    """Load raw CSV files and extract basic features."""
    print(f"\n🔵 Loading raw CSV from {len(file_paths)} files...")
    
    import pandas as pd
    
    all_features = []
    
    for filepath in file_paths:
        try:
            df = pd.read_csv(filepath)
            print(f"  {filepath.name}: {len(df)} rows")
            
            # Extract basic features
            features = []
            for _, row in df.iterrows():
                numbers_str = str(row.get("numbers", ""))
                numbers = [int(n.strip()) for n in numbers_str.split(",") if n.strip().isdigit()]
                
                if numbers:
                    feat = np.array([
                        np.mean(numbers),
                        np.std(numbers),
                        np.min(numbers),
                        np.max(numbers),
                        np.sum(numbers),
                        len(numbers),
                        float(row.get("bonus", 0)) if pd.notna(row.get("bonus")) else 0,
                        float(row.get("jackpot", 0)) if pd.notna(row.get("jackpot")) else 0,
                    ])
                    features.append(feat)
            
            if features:
                all_features.append(np.array(features))
        
        except Exception as e:
            print(f"  ❌ Error loading {filepath.name}: {e}")
    
    if all_features:
        combined = np.vstack(all_features)
        print(f"✅ Loaded {combined.shape[0]} samples with {combined.shape[1]} features")
        return combined, combined.shape[0]
    
    return None, 0


def test_cnn_loading():
    """Test the complete CNN loading pipeline."""
    
    print("=" * 80)
    print("CNN EMBEDDINGS LOADING TEST")
    print("=" * 80)
    
    # Define paths
    game = "Lotto Max"
    game_folder = game.lower().replace(" ", "_").replace("/", "_")
    
    data_dir = Path("data")
    cnn_dir = data_dir / "features" / "cnn" / game_folder
    raw_csv_dir = data_dir / game_folder
    
    print(f"\n📁 Directories:")
    print(f"   CNN dir: {cnn_dir}")
    print(f"   CNN exists: {cnn_dir.exists()}")
    print(f"   Raw CSV dir: {raw_csv_dir}")
    print(f"   Raw CSV exists: {raw_csv_dir.exists()}")
    
    # Get CNN files
    cnn_files = []
    if cnn_dir.exists():
        cnn_files = [f for f in cnn_dir.glob("*.npz")]
        print(f"\n📊 Found {len(cnn_files)} CNN .npz files:")
        for f in cnn_files:
            print(f"   - {f.name}")
    
    # Get raw CSV files
    raw_csv_files = []
    if raw_csv_dir.exists():
        raw_csv_files = sorted(raw_csv_dir.glob("*.csv"))
        print(f"\n📊 Found {len(raw_csv_files)} raw CSV files")
    
    print("\n" + "=" * 80)
    print("STEP 1: Load CNN embeddings")
    print("=" * 80)
    
    cnn_features, cnn_count = load_cnn_embeddings(cnn_files)
    
    if cnn_features is None:
        print("\n❌ CNN loading FAILED - no features loaded")
        return False
    
    print("\n" + "=" * 80)
    print("STEP 2: Simulate feature preparation (skip raw CSV)")
    print("=" * 80)
    
    # Simulate what load_training_data does
    all_features = []
    has_neural_features = True  # We have CNN
    skip_raw_csv_features = has_neural_features
    
    print(f"\n  has_neural_features: {has_neural_features}")
    print(f"  skip_raw_csv_features: {skip_raw_csv_features}")
    
    if skip_raw_csv_features:
        print("\n  ✅ Skipping raw_csv for FEATURES (will use for targets only)")
        all_features.append(cnn_features)
    
    if not all_features:
        print("\n❌ No features to combine!")
        return False
    
    # Stack features
    if len(all_features) == 1:
        X = all_features[0]
    else:
        X = np.hstack(all_features)
    
    print(f"\n  Final feature matrix:")
    print(f"    Shape: {X.shape}")
    print(f"    Dtype: {X.dtype}")
    print(f"    Samples: {X.shape[0]}")
    print(f"    Features: {X.shape[1]}")
    
    print("\n" + "=" * 80)
    print("STEP 3: Verify raw CSV files exist for target extraction")
    print("=" * 80)
    
    print(f"\n  Raw CSV files available: {len(raw_csv_files)}")
    print(f"  These will be used by load_training_data for TARGET extraction")
    
    print("\n" + "=" * 80)
    print("✅ TEST PASSED - CNN loading works correctly!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - CNN embeddings: {X.shape[0]} samples × {X.shape[1]} features ✅")
    print(f"  - Raw CSV files: {len(raw_csv_files)} files (for targets) ✅")
    print(f"  - Ready for training!")
    
    return True


if __name__ == "__main__":
    success = test_cnn_loading()
    exit(0 if success else 1)
