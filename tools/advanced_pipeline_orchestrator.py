"""
Advanced Model Training Pipeline - Main Orchestrator
Coordinates feature engineering, data preparation, and model training for both games
"""

import sys
from pathlib import Path
import logging
from typing import Dict
import json
from datetime import datetime

import numpy as np
import pandas as pd

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent))

from advanced_feature_engineering import (
    AdvancedFeatureEngineering,
    GameConfig,
    create_combined_features
)
from advanced_data_loader import prepare_game_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedPipelineOrchestrator:
    """
    Main orchestrator for the advanced ML pipeline
    Handles feature engineering, data splitting, and preparation for all model types
    """
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else Path('.').resolve()
        self.data_dir = self.base_dir / 'data'
        self.features_dir = self.data_dir / 'features' / 'advanced'
        self.models_dir = self.base_dir / 'models'
        
        logger.info(f"Orchestrator initialized with base_dir: {self.base_dir}")
        logger.info(f"Data dir: {self.data_dir}")
        
        # Game configurations
        self.game_configs = {
            'lotto_6_49': GameConfig(
                game_name='lotto_6_49',
                num_balls=6,
                num_numbers=49
            ),
            'lotto_max': GameConfig(
                game_name='lotto_max',
                num_balls=7,
                num_numbers=50
            )
        }
    
    def run_phase_1_feature_engineering(self, game_name: str) -> Dict:
        """
        Execute Phase 1: Advanced Data Representation & Feature Engineering
        
        Args:
            game_name: 'lotto_6_49' or 'lotto_max'
            
        Returns:
            Dictionary with feature engineering results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 1: Feature Engineering for {game_name}")
        logger.info(f"{'='*80}\n")
        
        # Get game config
        config = self.game_configs[game_name]
        fe = AdvancedFeatureEngineering(config)
        
        # Load and prepare data
        logger.info(f"Loading historical data for {game_name}...")
        logger.info(f"Using data_dir: {self.data_dir}")
        draws_df = prepare_game_dataset(
            game_name,
            data_dir=self.data_dir
        )
        logger.info(f"✓ Loaded {len(draws_df)} draws")
        
        # Generate temporal features
        logger.info(f"\nGenerating temporal features for {len(draws_df)} draws...")
        temporal_features = fe.generate_temporal_features(draws_df)
        logger.info(f"✓ Generated {len(temporal_features)} temporal feature records")
        logger.info(f"  Feature columns: {temporal_features.columns.tolist()}")
        
        # Generate global draw features
        logger.info(f"\nGenerating global draw features...")
        global_features = fe.generate_global_draw_features(draws_df)
        logger.info(f"✓ Generated {len(global_features)} global feature records")
        logger.info(f"  Feature columns: {global_features.columns.tolist()}")
        
        # Generate Skip-Gram targets
        logger.info(f"\nGenerating Skip-Gram co-occurrence targets...")
        skipgram_targets = fe.generate_skipgram_targets(draws_df)
        logger.info(f"✓ Generated {len(skipgram_targets)} Skip-Gram targets")
        
        # Generate Distribution Forecasting targets
        logger.info(f"\nGenerating Distribution Forecasting targets...")
        distribution_targets = fe.generate_distribution_targets(draws_df)
        logger.info(f"✓ Generated {len(distribution_targets)} distribution targets")
        
        # Create combined features
        logger.info(f"\nCombining temporal and global features...")
        combined_features = create_combined_features(temporal_features, global_features)
        logger.info(f"✓ Combined features shape: {combined_features.shape}")
        
        # Data splitting with temporal integrity
        logger.info(f"\nApplying temporal integrity preserving split...")
        train_split, val_split, test_split = fe.create_train_val_test_split(
            n_samples=len(combined_features),
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )
        logger.info(f"✓ Train split: indices {train_split[0]}-{train_split[1]} ({train_split[1]-train_split[0]} samples)")
        logger.info(f"✓ Val split:   indices {val_split[0]}-{val_split[1]} ({val_split[1]-val_split[0]} samples)")
        logger.info(f"✓ Test split:  indices {test_split[0]}-{test_split[1]} ({test_split[1]-test_split[0]} samples)")
        
        # Save engineered dataset
        output_dir = self.features_dir / game_name
        logger.info(f"\nSaving engineered features to {output_dir}...")
        fe.save_engineered_dataset(
            temporal_features,
            global_features,
            skipgram_targets,
            distribution_targets,
            output_dir
        )
        logger.info(f"✓ Features saved successfully")
        
        # Create summary
        summary = {
            'game_name': game_name,
            'num_balls': config.num_balls,
            'num_numbers': config.num_numbers,
            'total_draws': len(draws_df),
            'temporal_features': {
                'num_records': len(temporal_features),
                'columns': temporal_features.columns.tolist()
            },
            'global_features': {
                'num_records': len(global_features),
                'columns': global_features.columns.tolist()
            },
            'combined_features': {
                'shape': list(combined_features.shape),
                'columns': combined_features.columns.tolist()
            },
            'skipgram_targets': len(skipgram_targets),
            'distribution_targets': len(distribution_targets),
            'data_split': {
                'train': {'start': train_split[0], 'end': train_split[1], 'size': train_split[1]-train_split[0]},
                'val': {'start': val_split[0], 'end': val_split[1], 'size': val_split[1]-val_split[0]},
                'test': {'start': test_split[0], 'end': test_split[1], 'size': test_split[1]-test_split[0]}
            },
            'output_directory': str(output_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = output_dir / 'phase1_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\n✓ Phase 1 complete! Summary saved to {summary_path}")
        
        return summary
    
    def run_complete_pipeline(self):
        """
        Run complete Phase 1 for both games
        """
        logger.info("\n" + "="*80)
        logger.info("ADVANCED ML PIPELINE - PHASE 1 INITIALIZATION")
        logger.info("="*80)
        
        results = {}
        
        # Process Lotto 649
        try:
            results['lotto_6_49'] = self.run_phase_1_feature_engineering('lotto_6_49')
        except Exception as e:
            logger.error(f"Error processing Lotto 649: {e}")
            results['lotto_6_49'] = {'error': str(e)}
        
        # Process Lotto Max
        try:
            results['lotto_max'] = self.run_phase_1_feature_engineering('lotto_max')
        except Exception as e:
            logger.error(f"Error processing Lotto Max: {e}")
            results['lotto_max'] = {'error': str(e)}
        
        # Save overall results
        results_path = self.base_dir / 'phase1_pipeline_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 1 PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults saved to {results_path}")
        logger.info("\nNext Steps:")
        logger.info("  1. Review feature distributions")
        logger.info("  2. Proceed to Phase 2: Model Architecture Optimization")
        logger.info("  3. Train tree-based models (XGBoost, LightGBM, CatBoost)")
        logger.info("  4. Train neural network models (LSTM, Transformer, CNN)")
        
        return results


if __name__ == '__main__':
    # Find project root (tools is 1 level down from root)
    project_root = Path(__file__).parent.parent.resolve()
    
    # Run pipeline
    orchestrator = AdvancedPipelineOrchestrator(base_dir=project_root)
    results = orchestrator.run_complete_pipeline()
    
    # Print summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    for game, result in results.items():
        if 'error' not in result:
            print(f"\n{game.upper()}:")
            print(f"  OK Total draws: {result['total_draws']}")
            print(f"  OK Temporal features: {result['temporal_features']['num_records']}")
            print(f"  OK Global features: {result['global_features']['num_records']}")
            print(f"  OK Combined feature shape: {result['combined_features']['shape']}")
            print(f"  OK Output directory: {result['output_directory']}")
        else:
            print(f"\n{game.upper()}: ERROR - {result['error']}")
