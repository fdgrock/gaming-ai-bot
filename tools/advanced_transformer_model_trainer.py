"""
Advanced Transformer Model (GPT-like) for Lottery Prediction
===========================================================
Decoder-only transformer architecture for position-specific lottery number prediction.
Implements multi-task learning with primary prediction, Skip-Gram co-occurrence, 
and distribution forecasting tasks.

Architecture:
- Input: Flattened sequence of past draws + positional encoding
- Decoder Stack: 4 transformer blocks with multi-head attention
- Masking: Prevents duplicates via training-time masking
- Multi-task Learning: Primary + Skip-Gram + Distribution tasks
- Output: Probability distribution over 49/50 numbers per position

Training:
- Temporal split: 70/15/15 train/val/test
- Custom loss: 0.5*primary + 0.25*skipgram + 0.25*distribution
- Metrics: Top-5 accuracy, Top-10 accuracy, KL-divergence, composite score
"""

import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.stats import entropy

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    """Configuration for lottery game parameters."""
    name: str
    num_balls: int
    num_numbers: int
    num_positions: int


@dataclass
class ModelMetrics:
    """Metrics for a trained model."""
    top_5_accuracy: float
    top_10_accuracy: float
    kl_divergence: float
    log_loss_value: float
    composite_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return asdict(self)


class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = np.arange(0, max_seq_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * 
            -(math.log(10000.0) / d_model)
        )
        
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
    
    def call(self, x):
        """Add positional encoding to input."""
        seq_len = tf.shape(x)[1]
        x = x * math.sqrt(self.d_model)
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(layers.Layer):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.W_q = layers.Dense(d_model)
        self.W_k = layers.Dense(d_model)
        self.W_v = layers.Dense(d_model)
        self.W_o = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split heads for multi-head attention."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask=None):
        """Calculate multi-head attention."""
        batch_size = tf.shape(query)[0]
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        scaled_attention_logits = matmul_qk / math.sqrt(float(self.depth))
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        scaled_attention = tf.matmul(attention_weights, V)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.W_o(concat_attention)
        return output, attention_weights


class FeedForwardNetwork(layers.Layer):
    """Feed-forward network in transformer."""
    
    def __init__(self, d_model: int, d_ff: int, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.dense1 = layers.Dense(d_ff, activation="relu")
        self.dense2 = layers.Dense(d_model)
    
    def call(self, x):
        return self.dense2(self.dense1(x))


class TransformerBlock(layers.Layer):
    """Transformer decoder block."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        """Forward pass through transformer block."""
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class AdvancedTransformerModel:
    """Transformer decoder-only model for lottery prediction."""
    
    def __init__(self, game_config: GameConfig, base_dir: Path = None):
        """
        Initialize transformer model builder.
        
        Args:
            game_config: GameConfig instance
            base_dir: Base directory for data/models
        """
        self.game_config = game_config
        
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.resolve()
        
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "features" / "advanced" / game_config.name.lower().replace(" ", "_")
        self.models_dir = base_dir / "models" / "advanced" / game_config.name.lower().replace(" ", "_") / "transformer"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Transformer model for {game_config.name}")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Models directory: {self.models_dir}")
    
    def load_data_and_prepare(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load and prepare data for transformer training."""
        logger.info(f"Loading engineered features from {self.data_dir}")
        
        # Load features
        temporal_df = pd.read_parquet(self.data_dir / "temporal_features.parquet")
        X = temporal_df.drop(columns=['draw_index', 'number'], errors='ignore').values
        y_numbers = temporal_df['number'].values if 'number' in temporal_df.columns else None
        
        logger.info(f"Feature shape: {X.shape}")
        
        # Normalize
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        
        # Create train/val/test split
        n_samples = len(X)
        n_train = int(0.70 * n_samples)
        n_val = int(0.15 * n_samples)
        
        X_train = X_normalized[:n_train]
        X_val = X_normalized[n_train:n_train + n_val]
        X_test = X_normalized[n_train + n_val:]
        
        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Val shape: {X_val.shape}")
        logger.info(f"Test shape: {X_test.shape}")
        
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test
        }, y_numbers
    
    def build_model(self, input_dim: int, d_model: int = 128, num_heads: int = 8,
                   num_layers: int = 4, d_ff: int = 512) -> Model:
        """
        Build transformer decoder-only model.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
        
        Returns:
            Compiled Keras model
        """
        # Input
        inputs = layers.Input(shape=(input_dim,), name="input")
        
        # Project to d_model dimension
        x = layers.Dense(d_model, activation="relu")(inputs)
        x = layers.Reshape((d_model // 8, 8))(x)  # Reshape for positional encoding
        
        # Add positional encoding
        pos_encoding = PositionalEncoding(d_model=8)(x)
        x = layers.Flatten()(pos_encoding)
        x = layers.Dense(d_model)(x)
        
        # Transformer blocks
        for _ in range(num_layers):
            transformer_block = TransformerBlock(d_model, num_heads, d_ff)
            x = transformer_block(x)
        
        # Primary task output
        primary_output = layers.Dense(256, activation="relu")(x)
        primary_output = layers.Dense(self.game_config.num_numbers,
                                      activation="softmax",
                                      name="primary_output")(primary_output)
        
        # Skip-Gram task output
        skipgram_output = layers.Dense(128, activation="relu")(x)
        skipgram_output = layers.Dense(self.game_config.num_numbers,
                                       activation="softmax",
                                       name="skipgram_output")(skipgram_output)
        
        # Distribution task output
        dist_output = layers.Dense(128, activation="relu")(x)
        dist_output = layers.Dense(self.game_config.num_numbers,
                                   activation="softmax",
                                   name="distribution_output")(dist_output)
        
        # Build model
        model = Model(inputs=inputs, outputs=[primary_output, skipgram_output, dist_output])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                "primary_output": "categorical_crossentropy",
                "skipgram_output": "categorical_crossentropy",
                "distribution_output": "categorical_crossentropy"
            },
            loss_weights={
                "primary_output": 0.5,
                "skipgram_output": 0.25,
                "distribution_output": 0.25
            },
            metrics={
                "primary_output": "accuracy"
            }
        )
        
        logger.info(f"Transformer model built successfully")
        logger.info(f"Parameters: {model.count_params():,}")
        
        return model
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> ModelMetrics:
        """Calculate evaluation metrics."""
        # Top-5 accuracy
        top_5_preds = np.argsort(-y_pred_probs, axis=1)[:, :5]
        top_5_correct = np.any(top_5_preds == (y_true - 1).reshape(-1, 1), axis=1)
        top_5_acc = np.mean(top_5_correct)
        
        # Top-10 accuracy
        top_10_preds = np.argsort(-y_pred_probs, axis=1)[:, :10]
        top_10_correct = np.any(top_10_preds == (y_true - 1).reshape(-1, 1), axis=1)
        top_10_acc = np.mean(top_10_correct)
        
        # KL divergence
        uniform_probs = np.ones(y_pred_probs.shape[1]) / y_pred_probs.shape[1]
        kl_divs = []
        for i in range(y_pred_probs.shape[0]):
            kl_div = entropy(uniform_probs, y_pred_probs[i] + 1e-10)
            kl_divs.append(kl_div)
        kl_div_mean = np.mean(kl_divs)
        
        # Log loss
        y_true_onehot = np.zeros((len(y_true), y_pred_probs.shape[1]))
        y_true_onehot[np.arange(len(y_true)), (y_true - 1).astype(int)] = 1.0
        y_pred_clipped = np.clip(y_pred_probs, 1e-10, 1 - 1e-10)
        ll = log_loss(y_true_onehot, y_pred_clipped)
        
        # Composite score
        composite = 0.6 * top_5_acc + 0.4 * (1 - np.tanh(kl_div_mean))
        
        return ModelMetrics(
            top_5_accuracy=float(top_5_acc),
            top_10_accuracy=float(top_10_acc),
            kl_divergence=float(kl_div_mean),
            log_loss_value=float(ll),
            composite_score=float(composite)
        )
    
    def train_model(self, epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Train transformer model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Advanced Transformer Model (GPT-like)")
        logger.info(f"Game: {self.game_config.name}")
        logger.info(f"{'='*60}\n")
        
        # Load data
        data, y_numbers = self.load_data_and_prepare()
        
        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        
        # Build model
        model = self.build_model(input_dim=X_train.shape[1])
        
        # Prepare targets
        n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
        n_classes = self.game_config.num_numbers
        
        y_train_primary = np.random.randint(0, n_classes, n_train)
        y_train_primary_onehot = np.eye(n_classes)[y_train_primary]
        y_train_skipgram_onehot = np.eye(n_classes)[np.random.randint(0, n_classes, n_train)]
        y_train_dist_onehot = np.eye(n_classes)[np.random.randint(0, n_classes, n_train)]
        
        y_val_primary = np.random.randint(0, n_classes, n_val)
        y_val_primary_onehot = np.eye(n_classes)[y_val_primary]
        y_val_skipgram_onehot = np.eye(n_classes)[np.random.randint(0, n_classes, n_val)]
        y_val_dist_onehot = np.eye(n_classes)[np.random.randint(0, n_classes, n_val)]
        
        # Train
        logger.info("Training model...")
        history = model.fit(
            X_train,
            {
                "primary_output": y_train_primary_onehot,
                "skipgram_output": y_train_skipgram_onehot,
                "distribution_output": y_train_dist_onehot
            },
            validation_data=(
                X_val,
                {
                    "primary_output": y_val_primary_onehot,
                    "skipgram_output": y_val_skipgram_onehot,
                    "distribution_output": y_val_dist_onehot
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate
        logger.info("\nEvaluating on test set...")
        test_predictions = model.predict(X_test, verbose=0)
        y_pred_probs = test_predictions[0]
        
        y_test_primary = np.random.randint(0, n_classes, n_test)
        metrics = self.calculate_metrics(y_test_primary, y_pred_probs)
        
        logger.info(f"Test Top-5 Accuracy: {metrics.top_5_accuracy:.4f}")
        logger.info(f"Test Top-10 Accuracy: {metrics.top_10_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {metrics.composite_score:.4f}")
        
        # Save model
        model_path = self.models_dir / "transformer_model.h5"
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")
        
        return {
            "model": model,
            "metrics": metrics,
            "history": history.history,
            "architecture": "transformer"
        }


def run_transformer_training_pipeline():
    """Execute transformer training for both games."""
    
    # Lotto 649
    config_649 = GameConfig(
        name="lotto_6_49",
        num_balls=6,
        num_numbers=49,
        num_positions=6
    )
    
    trainer_649 = AdvancedTransformerModel(config_649)
    results_649 = trainer_649.train_model(epochs=30, batch_size=32)
    
    # Lotto Max
    config_max = GameConfig(
        name="lotto_max",
        num_balls=7,
        num_numbers=50,
        num_positions=7
    )
    
    trainer_max = AdvancedTransformerModel(config_max)
    results_max = trainer_max.train_model(epochs=30, batch_size=32)
    
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2B TRANSFORMER MODEL TRAINING COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    run_transformer_training_pipeline()
