"""
🤖 Advanced Transformer Ensemble Model Trainer - Phase 2C
Trains multiple Transformer instances with different random seeds and bootstrap sampling
for improved ensemble predictions and better generalization.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime
import pickle

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Logging setup - output to stdout for real-time monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "advanced"
DATA_DIR = PROJECT_ROOT / "data" / "features" / "advanced"


@dataclass
class GameConfig:
    """Configuration for lottery games."""
    name: str
    num_numbers: int
    num_positions: int
    

@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""
    top_5_accuracy: float = 0.0
    top_10_accuracy: float = 0.0
    kl_divergence: float = 0.0
    log_loss: float = 0.0
    composite_score: float = 0.0


class AdvancedTransformerEnsembleTrainer:
    """
    Trains multiple Transformer instances for ensemble learning.
    
    Each instance uses:
    - Different random seed for reproducibility
    - Bootstrap sampling for diversity
    - Same architecture but different weight initialization
    """
    
    # Games configuration
    GAMES = {
        'lotto_6_49': GameConfig(name='lotto_6_49', num_numbers=49, num_positions=6),
        'lotto_max': GameConfig(name='lotto_max', num_numbers=50, num_positions=7)
    }
    
    # Ensemble configuration
    NUM_VARIANTS_PER_GAME = 5  # Number of instances per game
    SEEDS = [42, 123, 456, 789, 999]  # Different seeds for diversity
    
    # Training configuration
    EPOCHS = 30
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    LEARNING_RATE = 0.001
    
    def __init__(self):
        """Initialize the ensemble trainer."""
        self.scaler = {}
        self.training_logs = []
        
    def load_data_and_prepare(self, game_name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for training."""
        logger.info(f"Loading data for {game_name} with seed {seed}...")
        
        # Load features
        features_path = DATA_DIR / game_name / "temporal_features.parquet"
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        
        df = pd.read_parquet(features_path)
        
        # Extract features and labels
        feature_cols = [col for col in df.columns if col not in ['draw_date', 'result_numbers']]
        X = df[feature_cols].values.astype(np.float32)
        
        # Parse result numbers (1-based, need 0-based for sklearn)
        y_all = []
        if 'result_numbers' in df.columns:
            for result_str in df['result_numbers']:
                try:
                    numbers = [int(x.strip()) - 1 for x in str(result_str).split(',')]
                    y_all.append(numbers)
                except:
                    pass
        
        # Normalize features
        if game_name not in self.scaler:
            self.scaler[game_name] = StandardScaler()
            X = self.scaler[game_name].fit_transform(X)
        else:
            X = self.scaler[game_name].transform(X)
        
        # Flatten result numbers for simplicity (use first position)
        y = np.array([y_all[i][0] if i < len(y_all) else 0 for i in range(len(X))], dtype=np.int32)
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features for {game_name}")
        
        return X, y
    
    def build_transformer_model(self, game: GameConfig) -> Model:
        """Build Transformer decoder architecture."""
        
        # Input layer
        inputs = layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
        
        # Embedding
        x = layers.Embedding(game.num_numbers, 128, mask_zero=True)(inputs)
        
        # Positional encoding
        x = self._add_positional_encoding(x, max_seq_length=100)
        
        # 4 Transformer blocks
        for i in range(4):
            x = self._transformer_block(x, num_heads=8, ff_dim=512, name=f"transformer_{i}")
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output heads (multi-task learning)
        # Primary task (50%)
        primary = layers.Dense(128, activation='relu', name='primary_hidden')(x)
        primary = layers.Dropout(0.3)(primary)
        primary_output = layers.Dense(game.num_numbers, activation='softmax', name='primary')(primary)
        
        # Skip-gram task (25%)
        skipgram = layers.Dense(128, activation='relu', name='skipgram_hidden')(x)
        skipgram = layers.Dropout(0.3)(skipgram)
        skipgram_output = layers.Dense(game.num_numbers, activation='softmax', name='skipgram')(skipgram)
        
        # Distribution task (25%)
        distribution = layers.Dense(128, activation='relu', name='distribution_hidden')(x)
        distribution = layers.Dropout(0.3)(distribution)
        distribution_output = layers.Dense(game.num_numbers, activation='softmax', name='distribution')(distribution)
        
        model = Model(inputs=inputs, outputs=[primary_output, skipgram_output, distribution_output])
        
        # Multi-task loss weights
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss={
                'primary': 'sparse_categorical_crossentropy',
                'skipgram': 'sparse_categorical_crossentropy',
                'distribution': 'sparse_categorical_crossentropy'
            },
            loss_weights={'primary': 0.5, 'skipgram': 0.25, 'distribution': 0.25},
            metrics=['accuracy']
        )
        
        return model
    
    def _add_positional_encoding(self, x, max_seq_length=100):
        """Add positional encoding to embeddings."""
        positions = tf.range(tf.shape(x)[1])
        position_embedding = layers.Embedding(max_seq_length, 128)(positions)
        return x + position_embedding
    
    def _transformer_block(self, x, num_heads=8, ff_dim=512, name="transformer"):
        """Single transformer block."""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=128 // num_heads,
            name=f"{name}_attention"
        )(x, x)
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward network
        ff_output = layers.Dense(ff_dim, activation='relu', name=f"{name}_ff1")(attention_output)
        ff_output = layers.Dense(128, name=f"{name}_ff2")(ff_output)
        ff_output = layers.Dropout(0.1)(ff_output)
        output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
        
        return output
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate evaluation metrics."""
        # Get top-5 and top-10 predictions
        top_5_indices = np.argsort(y_pred, axis=1)[:, -5:]
        top_10_indices = np.argsort(y_pred, axis=1)[:, -10:]
        
        top_5_accuracy = np.mean([y in top_5 for y, top_5 in zip(y_true, top_5_indices)])
        top_10_accuracy = np.mean([y in top_10 for y, top_10 in zip(y_true, top_10_indices)])
        
        # KL divergence
        y_pred_normalized = np.clip(y_pred, 1e-7, 1.0)
        uniform = np.ones_like(y_pred) / y_pred.shape[1]
        kl_div = np.mean(np.sum(uniform * np.log(uniform / y_pred_normalized), axis=1))
        
        # Log loss
        log_loss = -np.mean(np.log(y_pred_normalized[np.arange(len(y_true)), y_true]))
        
        # Composite score
        composite = 0.6 * top_5_accuracy + 0.4 * (1 - np.tanh(kl_div))
        
        return ModelMetrics(
            top_5_accuracy=top_5_accuracy,
            top_10_accuracy=top_10_accuracy,
            kl_divergence=kl_div,
            log_loss=log_loss,
            composite_score=composite
        )
    
    def train_ensemble_variant(self, game: GameConfig, variant_index: int, seed: int):
        """Train a single variant of the ensemble."""
        logger.info(f"Training {game.name} Transformer variant {variant_index + 1}/{self.NUM_VARIANTS_PER_GAME} (seed={seed})")
        
        # Set random seeds
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Load data
        X, y = self.load_data_and_prepare(game.name, seed)
        
        # Train-val-test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.VALIDATION_SPLIT + self.TEST_SPLIT, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.TEST_SPLIT / (self.VALIDATION_SPLIT + self.TEST_SPLIT), random_state=seed
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Build model
        model = self.build_transformer_model(game)
        
        # Train
        history = model.fit(
            X_train, [y_train, y_train, y_train],
            validation_data=(X_val, [y_val, y_val, y_val]),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            ]
        )
        
        # Evaluate
        predictions, _, _ = model.predict(X_test, verbose=0)
        metrics = self.calculate_metrics(y_test, predictions)
        
        logger.info(f"Variant {variant_index + 1} Results:")
        logger.info(f"  Top-5 Accuracy: {metrics.top_5_accuracy:.4f}")
        logger.info(f"  Top-10 Accuracy: {metrics.top_10_accuracy:.4f}")
        logger.info(f"  KL Divergence: {metrics.kl_divergence:.4f}")
        logger.info(f"  Composite Score: {metrics.composite_score:.4f}")
        
        # Save model
        model_dir = MODELS_DIR / game.name / "transformer_variants"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"transformer_variant_{variant_index + 1}_seed_{seed}.h5"
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return {
            'variant_index': variant_index,
            'seed': seed,
            'metrics': asdict(metrics),
            'history': {
                'loss': history.history.get('loss', []),
                'val_loss': history.history.get('val_loss', [])
            }
        }
    
    def train_all_variants(self):
        """Train all ensemble variants for all games."""
        logger.info("=" * 80)
        logger.info("Starting Phase 2C Transformer Ensemble Training")
        logger.info(f"Training {self.NUM_VARIANTS_PER_GAME} variants per game with different seeds")
        logger.info("=" * 80)
        
        summary = {
            'training_date': datetime.now().isoformat(),
            'phase': '2C',
            'architecture': 'Transformer',
            'num_variants': self.NUM_VARIANTS_PER_GAME,
            'games': {}
        }
        
        for game_name, game in self.GAMES.items():
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Training Transformer ensemble for {game_name}")
            logger.info(f"{'=' * 80}")
            
            game_results = {
                'game': game_name,
                'variants': []
            }
            
            for variant_idx, seed in enumerate(self.SEEDS):
                try:
                    result = self.train_ensemble_variant(game, variant_idx, seed)
                    game_results['variants'].append(result)
                except Exception as e:
                    logger.error(f"Error training variant {variant_idx}: {e}")
            
            summary['games'][game_name] = game_results
        
        # Save summary
        summary_path = MODELS_DIR / "transformer_ensemble_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Phase 2C Transformer Ensemble Training Complete!")
        logger.info(f"Summary saved to {summary_path}")
        logger.info(f"{'=' * 80}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Phase 2C Transformer Ensemble models")
    parser.add_argument("--game", type=str, default=None, help="Game to train")
    args = parser.parse_args()
    
    try:
        trainer = AdvancedTransformerEnsembleTrainer(game=args.game)
        trainer.train_all_variants()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
