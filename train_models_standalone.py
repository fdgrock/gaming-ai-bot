#!/usr/bin/env python3
"""
Standalone Model Training Script
Trains all 4 models (XGBoost, LSTM, Transformer, Ensemble) without GUI
Prevents Streamlit crashes during long training operations
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit_app.services.advanced_feature_generator import AdvancedFeatureGenerator
from streamlit_app.services.advanced_model_training import AdvancedModelTrainer
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandaloneTrainer:
    """Standalone trainer that runs without Streamlit GUI"""
    
    def __init__(self, game: str = "lotto_max"):
        self.game = game
        self.trainer = AdvancedModelTrainer(game=game)
        self.feature_gen = AdvancedFeatureGenerator(game=game)
        logger.info(f"Initialized trainer for {game}")
    
    def train_single_model(self, model_type: str, X=None, y=None, metadata=None, config: dict = None) -> dict:
        """
        Train a single model
        
        Args:
            model_type: "xgboost", "lstm", "transformer"
            X: Feature matrix (if None, will be generated)
            y: Target array (if None, will be generated)
            metadata: Training metadata (if None, will be generated)
            config: Training configuration
        
        Returns:
            dict: Training results
        """
        if config is None:
            config = {
                "epochs": 150,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2
            }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {model_type.upper()} Model")
        logger.info(f"{'='*70}")
        logger.info(f"Game: {self.game}")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        
        try:
            # Generate features if not provided
            if X is None or y is None:
                logger.info("\n[1/3] Generating advanced features...")
                from pathlib import Path
                game_data_dir = Path("data") / self.game
                if not game_data_dir.exists():
                    logger.error(f"Data directory not found: {game_data_dir}")
                    raise FileNotFoundError(f"Data directory not found: {game_data_dir}")
                
                # Find CSV files
                data_files = list(game_data_dir.glob("*.csv"))
                if not data_files:
                    logger.error(f"No CSV files found in {game_data_dir}")
                    raise FileNotFoundError(f"No CSV files found in {game_data_dir}")
                
                data_sources = {self.game: data_files}
                X, y, metadata = self.trainer.load_training_data(data_sources)
                logger.info(f"✓ Generated {X.shape[0]} samples with {X.shape[1]} features")
            
            # Train model
            logger.info(f"\n[2/3] Training {model_type.upper()} model...")
            start_time = datetime.now()
            
            if model_type.lower() == "xgboost":
                model, metrics = self.trainer.train_xgboost(
                    X, y, metadata, config, 
                    progress_callback=self._progress_callback
                )
            elif model_type.lower() == "lstm":
                model, metrics = self.trainer.train_lstm(
                    X, y, metadata, config,
                    progress_callback=self._progress_callback
                )
            elif model_type.lower() == "transformer":
                model, metrics = self.trainer.train_transformer(
                    X, y, metadata, config,
                    progress_callback=self._progress_callback
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Save model
            logger.info(f"\n[3/3] Saving {model_type.upper()} model...")
            model_path = self.trainer.save_model(model, model_type, metrics)
            logger.info(f"✓ Model saved to: {model_path}")
            
            # Display results
            logger.info(f"\n{'='*70}")
            logger.info(f"Training Complete - {model_type.upper()}")
            logger.info(f"{'='*70}")
            logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            logger.info(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
            logger.info(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
            logger.info(f"F1 Score:  {metrics['f1']:.4f}")
            logger.info(f"Training Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            logger.info(f"Model Path: {model_path}\n")
            
            return {
                "model_type": model_type,
                "status": "success",
                "metrics": metrics,
                "model_path": str(model_path),
                "training_time": elapsed
            }
            
        except Exception as e:
            logger.error(f"✗ Training failed: {e}", exc_info=True)
            return {
                "model_type": model_type,
                "status": "failed",
                "error": str(e)
            }
    
    def train_ensemble(self, config: dict = None) -> dict:
        """
        Train all three models as ensemble
        
        Args:
            config: Training configuration
        
        Returns:
            dict: Training results for all models
        """
        if config is None:
            config = {
                "epochs": 150,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2
            }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Training ENSEMBLE (XGBoost + LSTM + Transformer)")
        logger.info(f"{'='*70}")
        logger.info(f"Game: {self.game}")
        logger.info(f"Config: {json.dumps(config, indent=2)}\n")
        
        # Generate features once for all models
        logger.info("Generating advanced features...")
        from pathlib import Path
        game_data_dir = Path("data") / self.game
        if not game_data_dir.exists():
            logger.error(f"Data directory not found: {game_data_dir}")
            raise FileNotFoundError(f"Data directory not found: {game_data_dir}")
        
        # Find CSV files
        data_files = list(game_data_dir.glob("*.csv"))
        if not data_files:
            logger.error(f"No CSV files found in {game_data_dir}")
            raise FileNotFoundError(f"No CSV files found in {game_data_dir}")
        
        data_sources = {self.game: data_files}
        X, y, metadata = self.trainer.load_training_data(data_sources)
        logger.info(f"✓ Generated {X.shape[0]} samples with {X.shape[1]} features\n")
        
        start_time = datetime.now()
        results = []
        
        # Train XGBoost
        logger.info("Training XGBoost component (500+ trees)...")
        xgb_result = self.train_single_model("xgboost", X=X, y=y, metadata=metadata, config=config)
        results.append(xgb_result)
        
        # Train LSTM
        logger.info("\nTraining LSTM component (4-layer bidirectional)...")
        lstm_result = self.train_single_model("lstm", X=X, y=y, metadata=metadata, config=config)
        results.append(lstm_result)
        
        # Train Transformer
        logger.info("\nTraining Transformer component (4-block attention)...")
        trans_result = self.train_single_model("transformer", X=X, y=y, metadata=metadata, config=config)
        results.append(trans_result)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"ENSEMBLE TRAINING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total Training Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"\nComponent Results:")
        
        for result in results:
            if result["status"] == "success":
                metrics = result["metrics"]
                logger.info(
                    f"  {result['model_type'].upper():12} - "
                    f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%) - "
                    f"Time: {result['training_time']:.1f}s"
                )
            else:
                logger.info(f"  {result['model_type'].upper():12} - FAILED: {result['error']}")
        
        # Calculate ensemble accuracy
        successful = [r for r in results if r["status"] == "success"]
        if successful:
            ensemble_accuracy = sum([r["metrics"]["accuracy"] for r in successful]) / len(successful)
            logger.info(f"\nEnsemble Average Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        
        logger.info(f"{'='*70}\n")
        
        return {
            "status": "complete",
            "game": self.game,
            "components": results,
            "total_time": elapsed
        }
    
    def _progress_callback(self, progress: float, message: str):
        """Progress callback for training"""
        percent = int(progress * 100)
        logger.info(f"  [{percent:3d}%] {message}")

def main():
    parser = argparse.ArgumentParser(
        description="Train gaming AI models without GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models_standalone.py --model lstm --game lotto_max
  python train_models_standalone.py --model ensemble --game lotto_6_49
  python train_models_standalone.py --all --game lotto_max
        """
    )
    
    parser.add_argument(
        "--model",
        choices=["xgboost", "lstm", "transformer", "ensemble"],
        default="ensemble",
        help="Model to train (default: ensemble)"
    )
    
    parser.add_argument(
        "--game",
        choices=["lotto_6_49", "lotto_max"],
        default="lotto_max",
        help="Lottery game to train on (default: lotto_max)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = StandaloneTrainer(game=args.game)
    
    # Build config
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_split": 0.2
    }
    
    # Train
    if args.model == "ensemble":
        results = trainer.train_ensemble(config)
    else:
        results = trainer.train_single_model(args.model, config)
    
    return 0 if results.get("status") in ["success", "complete"] else 1

if __name__ == "__main__":
    sys.exit(main())
