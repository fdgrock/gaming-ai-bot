"""
Advanced CNN Model (1D Convolutions) for Lottery Prediction
==========================================================
1D convolutional neural network for detecting local patterns in lottery draw sequences.
Uses temporal convolutions with global max pooling to extract salient features.
Implements multi-task learning for comprehensive pattern recognition.

Architecture:
- Input: Draw sequences with 10-draw rolling windows
- Conv Layers: Multiple 1D convolutions with increasing receptive fields
- Pooling: Global max pooling to extract features
- Dense: Fully connected layers for final predictions
- Multi-task Learning: Primary + Skip-Gram + Distribution tasks

Training:
- Temporal split: 70/15/15 train/val/test
- Custom loss: 0.5*primary + 0.25*skipgram + 0.25*distribution
- Metrics: Top-5 accuracy, Top-10 accuracy, KL-divergence, composite score
"""

import json
import logging
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
print("[ADVANCED_CNN_MODEL_TRAINER] Script started and imports successful", flush=True)
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


class AdvancedCNNModel:
    """1D CNN model for lottery prediction."""
    
    def __init__(self, game_config: GameConfig, base_dir: Path = None):
        """
        Initialize CNN model builder.
        
        Args:
            game_config: GameConfig instance
            base_dir: Base directory for data/models
        """
        self.game_config = game_config
        
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.resolve()
        
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "features" / "advanced" / game_config.name.lower().replace(" ", "_")
        self.models_dir = base_dir / "models" / "advanced" / game_config.name.lower().replace(" ", "_") / "cnn"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized CNN model for {game_config.name}")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Models directory: {self.models_dir}")
    
    def create_rolling_windows(self, X: np.ndarray, window_size: int = 10,
                              stride: int = 1) -> np.ndarray:
        """
        Create rolling windows from features.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            window_size: Size of rolling window
            stride: Step size for windows
        
        Returns:
            Windows [n_windows, window_size, n_features]
        """
        # Use < (N - window_size) so every window has a valid target at i+window_size.
        windows = []
        for i in range(0, len(X) - window_size, stride):
            windows.append(X[i:i + window_size])
        
        return np.array(windows)
    
    def load_data_and_prepare(self, window_size: int = 50) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load and prepare data for CNN training."""
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

        # Create rolling windows
        stride = 5
        X_windows = self.create_rolling_windows(X, window_size=window_size, stride=stride)

        logger.info(f"Rolling windows shape: {X_windows.shape}")

        # Normalize
        scaler = StandardScaler()
        X_reshaped = X_windows.reshape(X_windows.shape[0], -1)
        X_normalized = scaler.fit_transform(X_reshaped)
        X_windows = X_normalized.reshape(X_windows.shape)

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
    
    def build_model(self, input_shape: Tuple[int, ...]) -> Model:
        """
        Build 1D CNN model.
        
        Args:
            input_shape: Shape of input (window_size, n_features)
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=input_shape, name="input")
        
        # Conv block 1: Extract local patterns (3 time steps)
        x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
        
        # Conv block 2: Larger receptive field (5 time steps)
        x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
        
        # Conv block 3: Even larger patterns (7 time steps)
        x = layers.Conv1D(256, kernel_size=7, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        
        # Global max pooling
        x = layers.GlobalMaxPooling1D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        
        # Primary task output
        primary_output = layers.Dense(self.game_config.num_numbers,
                                      activation="softmax",
                                      name="primary_output")(x)
        
        # Skip-Gram task output
        skipgram_branch = layers.Dense(64, activation="relu")(x)
        skipgram_output = layers.Dense(self.game_config.num_numbers,
                                       activation="softmax",
                                       name="skipgram_output")(skipgram_branch)
        
        # Distribution task output
        dist_branch = layers.Dense(64, activation="relu")(x)
        dist_output = layers.Dense(self.game_config.num_numbers,
                                   activation="softmax",
                                   name="distribution_output")(dist_branch)
        
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
        
        logger.info(f"CNN model built successfully")
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
        """Train CNN model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Advanced CNN Model (1D Convolutions)")
        logger.info(f"Game: {self.game_config.name}")
        logger.info(f"{'='*60}\n")
        
        # Load data
        data, y_numbers = self.load_data_and_prepare(window_size=50)
        
        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        
        # Build model
        model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Align targets to actual lottery numbers (window i -> y_numbers[50 + i*5])
        n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
        n_classes = self.game_config.num_numbers
        seq_draw_idxs = data.get("seq_draw_idxs", np.array([]))

        _num_seq = n_train + n_val + n_test
        _raw_targets = np.clip(y_numbers[50::5].astype(int), 1, n_classes)
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

        # Defensive trim: ensure X_test and num_test have identical lengths.
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

        y_test_primary = num_test  # 1-based lottery numbers (calculate_metrics does y_true-1)
        metrics = self.calculate_metrics(y_test_primary, y_pred_probs)
        
        logger.info(f"Test Top-5 Accuracy: {metrics.top_5_accuracy:.4f}")
        logger.info(f"Test Top-10 Accuracy: {metrics.top_10_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {metrics.composite_score:.4f}")
        
        # Save model
        model_path = self.models_dir / "cnn_model.h5"
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # Save feature schema + model card
        try:
            import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
            from model_card_utils import (
                build_feature_schema, save_feature_schema,
                build_model_card, save_model_card, derive_feature_cols,
            )
            _feat_schema_path = self.data_dir / "feature_schema_cnn.json"
            if _feat_schema_path.exists():
                import json as _json
                with open(_feat_schema_path) as _f:
                    _feat_schema = _json.load(_f)
            else:
                _td = pd.read_parquet(self.data_dir / "temporal_features.parquet")
                _feat_schema = build_feature_schema(
                    model_type="cnn", game=self.game_config.name,
                    feature_cols=derive_feature_cols(_td),
                    n_samples=n_train + n_val + n_test,
                    n_train=n_train, n_val=n_val, n_test=n_test,
                    window_size=50, lookback=None, stride=5,
                )
            save_feature_schema(_feat_schema, self.models_dir / "feature_schema.json")
            _card = build_model_card(
                architecture="cnn",
                game=self.game_config.name,
                feature_schema=_feat_schema,
                metrics=metrics.to_dict(),
                training_config={
                    "epochs": epochs, "batch_size": batch_size,
                    "window_size": 50, "stride": 5,
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
            "architecture": "cnn"
        }


def run_cnn_training_pipeline():
    """Execute CNN training for both games."""
    
    # Lotto 649
    config_649 = GameConfig(
        name="lotto_6_49",
        num_balls=6,
        num_numbers=49,
        num_positions=6
    )
    
    trainer_649 = AdvancedCNNModel(config_649)
    results_649 = trainer_649.train_model(epochs=30, batch_size=32)
    
    # Lotto Max
    config_max = GameConfig(
        name="lotto_max",
        num_balls=7,
        num_numbers=52,
        num_positions=7
    )
    
    trainer_max = AdvancedCNNModel(config_max)
    results_max = trainer_max.train_model(epochs=30, batch_size=32)
    
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2B CNN MODEL TRAINING COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 2B CNN models")
    parser.add_argument("--game", type=str, default=None, help="Game to train")
    args = parser.parse_args()
    
    if args.game and args.game != "All Games":
        if "649" in args.game or args.game.lower() == "lotto 6/49":
            config = GameConfig(name="lotto_6_49", num_balls=6, num_numbers=49, num_positions=6)
            trainer = AdvancedCNNModel(config)
            trainer.train_model(epochs=30, batch_size=32)
        elif "max" in args.game.lower() or args.game.lower() == "lotto max":
            config = GameConfig(name="lotto_max", num_balls=7, num_numbers=52, num_positions=7)
            trainer = AdvancedCNNModel(config)
            trainer.train_model(epochs=30, batch_size=32)
    else:
        run_cnn_training_pipeline()
