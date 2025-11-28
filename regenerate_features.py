#!/usr/bin/env python3
"""Regenerate CatBoost and LightGBM features with correct 85-feature dimension."""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add streamlit_app to path
sys.path.insert(0, str(Path(__file__).parent / "streamlit_app"))

from services.advanced_feature_generator import AdvancedFeatureGenerator

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)

def main():
    project_root = Path(__file__).parent
    
    for game in ["lotto_max", "lotto_6_49"]:
        print(f"\n{'='*60}")
        print(f"Regenerating features for {game.upper()}")
        print(f"{'='*60}")
        
        try:
            # Load raw data
            raw_data_path = project_root / "data" / "raw" / f"{game}.csv"
            if not raw_data_path.exists():
                print(f"ERROR: Raw data not found: {raw_data_path}")
                continue
            
            raw_data = pd.read_csv(raw_data_path)
            print(f"Loaded {len(raw_data)} draws from {raw_data_path.name}")
            
            # Create feature generator
            generator = AdvancedFeatureGenerator(game, logger=app_logger)
            
            # Generate CatBoost features
            print("\nGenerating CatBoost features...")
            cb_features, cb_metadata = generator.generate_catboost_features(raw_data)
            print(f"  Generated: {len(cb_features)} rows x {len(cb_features.columns)-1} numeric features")
            
            if generator.save_catboost_features(cb_features, cb_metadata):
                print(f"  [OK] Saved CatBoost features")
            else:
                print(f"  [FAILED] Could not save CatBoost features")
            
            # Generate LightGBM features
            print("Generating LightGBM features...")
            lgb_features, lgb_metadata = generator.generate_lightgbm_features(raw_data)
            print(f"  Generated: {len(lgb_features)} rows x {len(lgb_features.columns)-1} numeric features")
            
            if generator.save_lightgbm_features(lgb_features, lgb_metadata):
                print(f"  [OK] Saved LightGBM features")
            else:
                print(f"  [FAILED] Could not save LightGBM features")
            
        except Exception as e:
            print(f"ERROR: {e}")
            app_logger.error(f"Error for {game}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Feature regeneration complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
