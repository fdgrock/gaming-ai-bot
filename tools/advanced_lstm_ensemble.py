"""
🔗 Advanced LSTM Ensemble Model Trainer - Phase 2C
Trains multiple LSTM instances with different random seeds and bootstrap sampling
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


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and float32 types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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


class AttentionLayer(layers.Layer):
    """Luong-style attention mechanism."""
    
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # Attention weights
        self.W = self.add_weight(
            name='W',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.U = self.add_weight(
            name='U',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.v = self.add_weight(
            name='v',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: [batch, time_steps, units]
        decoder_hidden: [batch, units]
        """
        # Expand decoder hidden for broadcasting
        decoder_hidden_expanded = tf.expand_dims(decoder_hidden, 1)
        
        # Calculate attention scores
        # score = v^T * tanh(W * encoder_outputs + U * decoder_hidden)
        score = tf.nn.tanh(
            tf.matmul(encoder_outputs, self.W) + 
            tf.matmul(decoder_hidden_expanded, self.U)
        )
        score = tf.matmul(score, self.v)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context = tf.reduce_sum(
            encoder_outputs * attention_weights,
            axis=1
        )
        
        return context, attention_weights


class AdvancedLSTMEnsembleTrainer:
    """
    Trains multiple LSTM instances for ensemble learning.
    
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
    NUM_VARIANTS_PER_GAME = 3  # Number of instances per game
    SEEDS = [42, 123, 456]  # Different seeds for diversity
    
    # Training configuration
    EPOCHS = 30
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    LEARNING_RATE = 0.001
    LOOKBACK = 100  # 100-draw lookback
    
    def __init__(self, game: str = None):
        """
        Initialize the ensemble trainer.
        
        Args:
            game: Optional game name to filter training. If None, trains all games.
        """
        self.scaler = {}
        self.training_logs = []
        self.game_filter = game  # Optional: can filter to specific game
    
    def create_sequences(self, X: np.ndarray, lookback: int = 100, stride: int = 10) -> np.ndarray:
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(0, len(X) - lookback, stride):
            sequences.append(X[i:i + lookback])
        return np.array(sequences)
    
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
    
    def build_lstm_model(self, game: GameConfig, n_features: int) -> Model:
        """Build LSTM encoder-decoder architecture."""
        
        # Input: 3D tensor [batch, 100, n_features]
        inputs = layers.Input(shape=(self.LOOKBACK, n_features), name='input_sequence')
        
        # Encoder: Bidirectional LSTM with return_sequences for attention
        encoder = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, return_state=True, name='encoder_lstm'),
            name='encoder_bidirectional'
        )
        encoder_result = encoder(inputs)
        # Bidirectional with return_sequences=True returns: (sequences, fwd_h, fwd_c, bwd_h, bwd_c)
        encoder_sequences = encoder_result[0]  # [batch, time, 256]
        forward_h, forward_c, backward_h, backward_c = encoder_result[1], encoder_result[2], encoder_result[3], encoder_result[4]
        
        # Combine bidirectional states for decoder initial state
        state_h = layers.Concatenate()([forward_h, backward_h])
        state_c = layers.Concatenate()([forward_c, backward_c])
        
        # Attention layer - takes full sequence and decoder hidden state
        attention_output, attention_weights = AttentionLayer(256, name='attention')(
            encoder_sequences, state_h
        )
        
        # Decoder: LSTM with attention context (2D) - need to expand for LSTM input
        attention_expanded = layers.Reshape((1, 256))(attention_output)  # [batch, 1, 256]
        decoder = layers.LSTM(256, name='decoder_lstm')
        decoder_output = decoder(attention_expanded)
        
        # Output heads (multi-task learning)
        # Primary task (50%)
        primary = layers.Dense(128, activation='relu', name='primary_hidden')(decoder_output)
        primary = layers.Dropout(0.3)(primary)
        primary_output = layers.Dense(game.num_numbers, activation='softmax', name='primary')(primary)
        
        # Skip-gram task (25%)
        skipgram = layers.Dense(128, activation='relu', name='skipgram_hidden')(decoder_output)
        skipgram = layers.Dropout(0.3)(skipgram)
        skipgram_output = layers.Dense(game.num_numbers, activation='softmax', name='skipgram')(skipgram)
        
        # Distribution task (25%)
        distribution = layers.Dense(128, activation='relu', name='distribution_hidden')(decoder_output)
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
            metrics={
                'primary': 'accuracy',
                'skipgram': 'accuracy',
                'distribution': 'accuracy'
            }
        )
        
        return model
    
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
        logger.info(f"Training {game.name} LSTM variant {variant_index + 1}/{self.NUM_VARIANTS_PER_GAME} (seed={seed})")
        
        # Set random seeds
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Load data
        X, y = self.load_data_and_prepare(game.name, seed)
        
        # Create sequences with stride=10
        stride = 10
        X_sequences = self.create_sequences(X, self.LOOKBACK, stride=stride)
        # Apply same stride to targets: take every 10th element starting at LOOKBACK
        y_sequences = y[self.LOOKBACK::stride]
        
        # Ensure alignment - trim to match shortest
        min_len = min(len(X_sequences), len(y_sequences))
        X_sequences = X_sequences[:min_len]
        y_sequences = y_sequences[:min_len]
        
        logger.info(f"Sequences created: X={X_sequences.shape}, y={y_sequences.shape}")
        
        # Train-val-test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_sequences, y_sequences, test_size=self.VALIDATION_SPLIT + self.TEST_SPLIT, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.TEST_SPLIT / (self.VALIDATION_SPLIT + self.TEST_SPLIT), random_state=seed
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Build model
        model = self.build_lstm_model(game, X.shape[1])
        
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
        model_dir = MODELS_DIR / game.name / "lstm_variants"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"lstm_variant_{variant_index + 1}_seed_{seed}.h5"
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
        """Train all ensemble variants for all games (or filtered game)."""
        logger.info("=" * 80)
        logger.info("Starting Phase 2C LSTM Ensemble Training")
        logger.info(f"Training {self.NUM_VARIANTS_PER_GAME} variants per game with different seeds")
        if self.game_filter:
            logger.info(f"Game filter: {self.game_filter}")
        logger.info("=" * 80)
        
        summary = {
            'training_date': datetime.now().isoformat(),
            'phase': '2C',
            'architecture': 'LSTM',
            'num_variants': self.NUM_VARIANTS_PER_GAME,
            'games': {}
        }
        
        # Filter games if game_filter is specified
        games_to_train = self.GAMES
        if self.game_filter:
            # Normalize game name - convert "Lotto 6/49" to "lotto_6_49"
            game_key = self.game_filter.lower().replace(" ", "_").replace("/", "_")
            if game_key in self.GAMES:
                games_to_train = {game_key: self.GAMES[game_key]}
            else:
                logger.warning(f"Game {self.game_filter} not found. Available games: {list(self.GAMES.keys())}")
                return
        
        for game_name, game in games_to_train.items():
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Training LSTM ensemble for {game_name}")
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
            
            # Save metadata for this game's variants in the variant folder
            game_variant_dir = MODELS_DIR / game_name / "lstm_variants"
            game_variant_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = game_variant_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(game_results, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Metadata saved to {metadata_path}")
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Phase 2C LSTM Ensemble Training Complete!")
        logger.info(f"{'=' * 80}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train Phase 2C LSTM Ensemble models")
    parser.add_argument("--game", type=str, default=None, help="Game to train")
    args = parser.parse_args()
    
    try:
        trainer = AdvancedLSTMEnsembleTrainer(game=args.game)
        trainer.train_all_variants()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
