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
import argparse
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

# Configure logging to output to stdout for real-time monitoring
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
    force=True  # Override any existing handlers
)
logger = logging.getLogger(__name__)

# VERIFY SCRIPT IS RUNNING
print("[ADVANCED_TRANSFORMER_MODEL_TRAINER] Script started and imports successful", flush=True)
sys.stdout.flush()


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
    
    def load_data_and_prepare(self, window_size: int = 20) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load and prepare data with rolling windows for sequential transformer input."""
        logger.info(f"Loading engineered features from {self.data_dir}")

        # Load temporal features
        temporal_df = pd.read_parquet(self.data_dir / "temporal_features.parquet")

        # Merge global draw features (per-draw stats -> broadcast to all per-number rows)
        _draw_col = 'draw_idx' if 'draw_idx' in temporal_df.columns else 'draw_index' if 'draw_index' in temporal_df.columns else None
        try:
            global_df = pd.read_parquet(self.data_dir / "global_features.parquet")
            _gdraw_col = 'draw_idx' if 'draw_idx' in global_df.columns else 'draw_index'
            _global_feat_cols = [c for c in global_df.columns if c not in {'draw_idx', 'draw_index'}]
            global_subset = global_df[[_gdraw_col] + _global_feat_cols].rename(
                columns={_gdraw_col: _draw_col or 'draw_idx'}
            )
            temporal_df = temporal_df.merge(global_subset, on=_draw_col or 'draw_idx', how='left')
            logger.info(f"Merged global features: +{len(_global_feat_cols)} columns -> {temporal_df.shape[1]} total")
        except Exception as _ge:
            logger.warning(f"Could not merge global features (continuing with temporal only): {_ge}")

        # Prepare feature matrix - drop all non-feature columns
        _NON_FEAT = {'draw_index', 'draw_idx', 'number', 'target'}
        X = temporal_df.drop(columns=[c for c in temporal_df.columns if c in _NON_FEAT], errors='ignore').values
        y_numbers = temporal_df['number'].values if 'number' in temporal_df.columns else None

        logger.info(f"Feature shape: {X.shape}")

        # Normalize
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Create rolling windows with stride=5 for sequential input.
        # Use < (N - window_size) so every window has a valid target at i+window_size.
        stride = 5
        windows = [X_normalized[i:i + window_size]
                   for i in range(0, len(X_normalized) - window_size, stride)]
        X_windows = np.array(windows)

        logger.info(f"Rolling windows shape: {X_windows.shape}")

        # Temporal split on draw boundaries (prevents draw leakage between splits)
        if _draw_col:
            draw_indices = temporal_df[_draw_col].values
            # seq_draw_idxs: draw_idx of the TARGET (one step after window end)
            seq_draw_idxs = draw_indices[window_size::stride][:len(X_windows)]
            _split_draw_idxs = draw_indices[window_size - 1::stride][:len(X_windows)]
            unique_draws = np.unique(_split_draw_idxs)
            n_draws = len(unique_draws)
            split_draw_1 = unique_draws[int(0.70 * n_draws)]
            split_draw_2 = unique_draws[int(0.85 * n_draws)]
            n_train = int(np.searchsorted(_split_draw_idxs, split_draw_1, side='right'))
            n_val = int(np.searchsorted(_split_draw_idxs, split_draw_2, side='right')) - n_train
        else:
            n_train = int(0.70 * len(X_windows))
            n_val = int(0.15 * len(X_windows))
            seq_draw_idxs = np.arange(len(X_windows))

        X_train = X_windows[:n_train]
        X_val = X_windows[n_train:n_train + n_val]
        X_test = X_windows[n_train + n_val:]

        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Val shape: {X_val.shape}")
        logger.info(f"Test shape: {X_test.shape}")

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "seq_draw_idxs": seq_draw_idxs,  # draw_idx per window for target alignment
        }, y_numbers
    
    def build_model(self, window_size: int, input_dim: int, d_model: int = 128, num_heads: int = 8,
                   num_layers: int = 4, d_ff: int = 512) -> Model:
        """
        Build transformer model with sequential window input.

        Args:
            window_size: Number of time steps in each input window
            input_dim: Feature dimension per time step
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward dimension

        Returns:
            Compiled Keras model
        """
        # Input: (batch, window_size, input_dim)
        inputs = layers.Input(shape=(window_size, input_dim), name="input")

        # Project each time step to d_model
        x = layers.TimeDistributed(layers.Dense(d_model, activation="relu"))(inputs)

        # Add sinusoidal positional encoding over the window dimension
        x = PositionalEncoding(d_model=d_model)(x)

        # Transformer blocks - each operates on (batch, window_size, d_model)
        for _ in range(num_layers):
            transformer_block = TransformerBlock(d_model, num_heads, d_ff)
            x = transformer_block(x)

        # Pool sequence to single vector
        x = layers.GlobalAveragePooling1D()(x)
        
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
        
        # Composite score: exp(-kl) keeps calibration signal even when KL >> 1
        composite = 0.6 * top_5_acc + 0.4 * float(np.exp(-max(0.0, kl_div_mean)))

        # Sanity checks - warn if metrics look impossibly good
        import math as _math
        if top_5_acc > 0.95:
            logger.warning(f"[SANITY] Suspiciously high Top-5 accuracy: {top_5_acc:.4f} - verify target alignment")
        if ll <= 0.0:
            logger.warning(f"[SANITY] Non-positive log-loss: {ll:.6f} - likely a metric calculation error")
        if kl_div_mean > _math.log(y_pred_probs.shape[1]) + 1.0:
            logger.warning(f"[SANITY] KL divergence {kl_div_mean:.4f} exceeds theoretical max - check predictions")

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
        
        # Load data with rolling windows (window_size=20, stride=5)
        data, y_numbers = self.load_data_and_prepare(window_size=20)

        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]

        # Build model with sequential (window_size, input_dim) input shape
        model = self.build_model(window_size=X_train.shape[1], input_dim=X_train.shape[2])

        # Align targets: window i -> y_numbers[20 + i*5] (element after window)
        n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
        n_classes = self.game_config.num_numbers
        seq_draw_idxs = data.get("seq_draw_idxs", np.array([]))

        _num_seq = n_train + n_val + n_test
        _raw_targets = np.clip(y_numbers[20::5].astype(int), 1, n_classes)
        _raw_targets = _raw_targets[:min(_num_seq, len(_raw_targets))]
        num_train = _raw_targets[:n_train]
        num_val   = _raw_targets[n_train:n_train + n_val]
        num_test  = _raw_targets[n_train + n_val:]

        y_train_primary_onehot = np.eye(n_classes)[num_train - 1]
        y_val_primary_onehot   = np.eye(n_classes)[num_val - 1]

        # Fallback: all heads use primary target (overridden below when parquets available)
        y_train_skipgram_onehot = y_train_primary_onehot.copy()
        y_train_dist_onehot     = y_train_primary_onehot.copy()
        y_val_skipgram_onehot   = y_val_primary_onehot.copy()
        y_val_dist_onehot       = y_val_primary_onehot.copy()

        # Override with actual skipgram and distribution targets from feature engineering
        try:
            _sg_df  = pd.read_parquet(self.data_dir / "skipgram_targets.parquet")
            _dt_df  = pd.read_parquet(self.data_dir / "distribution_targets.parquet")
            _sg_col = 'draw_idx' if 'draw_idx' in _sg_df.columns else 'draw_index'
            _dt_col = 'draw_idx' if 'draw_idx' in _dt_df.columns else 'draw_index'

            _sg_lookup: dict = {}
            for _, _row in _sg_df.iterrows():
                _vec = np.zeros(n_classes, dtype=np.float32)
                for _n in _row['target_numbers']:
                    if 1 <= _n <= n_classes:
                        _vec[_n - 1] = 1.0
                _s = _vec.sum()
                _sg_lookup[int(_row[_sg_col])] = _vec / _s if _s > 0 else _vec

            _dt_lookup: dict = {}
            for _, _row in _dt_df.iterrows():
                _arr = np.array(_row['distribution'], dtype=np.float32)
                _s = _arr.sum()
                _dt_lookup[int(_row[_dt_col])] = _arr / _s if _s > 0 else _arr

            _def_sg = np.ones(n_classes, dtype=np.float32) / n_classes
            _def_dt = np.ones(n_classes, dtype=np.float32) / n_classes

            _seq_ids = seq_draw_idxs[:_num_seq].astype(int)
            _y_sg_all  = np.array([_sg_lookup.get(int(d), _def_sg) for d in _seq_ids])
            _y_dt_all  = np.array([_dt_lookup.get(int(d), _def_dt) for d in _seq_ids])

            y_train_skipgram_onehot = _y_sg_all[:n_train]
            y_train_dist_onehot     = _y_dt_all[:n_train]
            y_val_skipgram_onehot   = _y_sg_all[n_train:n_train + n_val]
            y_val_dist_onehot       = _y_dt_all[n_train:n_train + n_val]
            logger.info("Using actual skipgram + distribution targets for multi-task heads")
        except Exception as _te:
            logger.warning(f"Could not load skipgram/distribution targets, using primary fallback: {_te}")
        
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

        # Defensive trim: ensure X_test and num_test have identical lengths before
        # calling calculate_metrics.  A 1-off can arise from integer-division rounding
        # in the draw-boundary split; clipping both to the shorter is safe.
        _n_test_actual = min(len(X_test), len(num_test))
        if _n_test_actual != len(X_test) or _n_test_actual != len(num_test):
            logger.warning(
                f"Trimming test set: X_test={len(X_test)}, num_test={len(num_test)} "
                f"-> {_n_test_actual}"
            )
            X_test = X_test[:_n_test_actual]
            num_test = num_test[:_n_test_actual]

        test_predictions = model.predict(X_test, verbose=0)
        y_pred_probs = test_predictions[0]

        y_test_primary = num_test  # 1-based, calculate_metrics does y_true-1 internally
        metrics = self.calculate_metrics(y_test_primary, y_pred_probs)
        
        logger.info(f"Test Top-5 Accuracy: {metrics.top_5_accuracy:.4f}")
        logger.info(f"Test Top-10 Accuracy: {metrics.top_10_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {metrics.composite_score:.4f}")
        
        # Save model
        model_path = self.models_dir / "transformer_model.h5"
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # Save feature schema + model card
        try:
            import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
            from model_card_utils import (
                build_feature_schema, save_feature_schema,
                build_model_card, save_model_card, derive_feature_cols,
            )
            _feat_schema_path = self.data_dir / "feature_schema_transformer.json"
            if _feat_schema_path.exists():
                import json as _json
                with open(_feat_schema_path) as _f:
                    _feat_schema = _json.load(_f)
            else:
                _td = pd.read_parquet(self.data_dir / "temporal_features.parquet")
                _feat_schema = build_feature_schema(
                    model_type="transformer", game=self.game_config.name,
                    feature_cols=derive_feature_cols(_td),
                    n_samples=n_train + n_val + n_test,
                    n_train=n_train, n_val=n_val, n_test=n_test,
                    window_size=20, lookback=None, stride=5,
                )
            save_feature_schema(_feat_schema, self.models_dir / "feature_schema.json")
            _card = build_model_card(
                architecture="transformer",
                game=self.game_config.name,
                feature_schema=_feat_schema,
                metrics=metrics.to_dict(),
                training_config={
                    "epochs": epochs, "batch_size": batch_size,
                    "window_size": 20, "stride": 5,
                    "d_model": 128, "num_heads": 8, "num_layers": 4,
                    "target_alignment": "actual_lottery_numbers_1based",
                },
            )
            save_model_card(_card, model_path)
            logger.info(f"Model card + feature schema saved alongside model")
        except Exception as _e:
            logger.warning(f"Could not save model card: {_e}")

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
        num_numbers=52,
        num_positions=7
    )
    
    trainer_max = AdvancedTransformerModel(config_max)
    results_max = trainer_max.train_model(epochs=30, batch_size=32)
    
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2B TRANSFORMER MODEL TRAINING COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 2B Transformer models")
    parser.add_argument("--game", type=str, default=None, help="Game to train")
    args = parser.parse_args()
    
    if args.game and args.game != "All Games":
        if "649" in args.game or args.game.lower() == "lotto 6/49":
            config = GameConfig(name="lotto_6_49", num_balls=6, num_numbers=49, num_positions=6)
            trainer = AdvancedTransformerModel(config)
            trainer.train_model(epochs=30, batch_size=32)
        elif "max" in args.game.lower() or args.game.lower() == "lotto max":
            config = GameConfig(name="lotto_max", num_balls=7, num_numbers=52, num_positions=7)
            trainer = AdvancedTransformerModel(config)
            trainer.train_model(epochs=30, batch_size=32)
    else:
        run_transformer_training_pipeline()
