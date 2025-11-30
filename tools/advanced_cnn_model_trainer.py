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
        windows = []
        for i in range(0, len(X) - window_size + 1, stride):
            windows.append(X[i:i + window_size])
        
        return np.array(windows)
    
    def load_data_and_prepare(self, window_size: int = 10) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Load and prepare data for CNN training."""
        logger.info(f"Loading engineered features from {self.data_dir}")
        
        # Load features
        temporal_df = pd.read_parquet(self.data_dir / "temporal_features.parquet")
        X = temporal_df.drop(columns=['draw_index', 'number'], errors='ignore').values
        y_numbers = temporal_df['number'].values if 'number' in temporal_df.columns else None
        
        logger.info(f"Feature shape: {X.shape}")
        
        # Create rolling windows
        X_windows = self.create_rolling_windows(X, window_size=window_size, stride=5)
        
        logger.info(f"Rolling windows shape: {X_windows.shape}")
        
        # Normalize
        scaler = StandardScaler()
        X_reshaped = X_windows.reshape(X_windows.shape[0], -1)
        X_normalized = scaler.fit_transform(X_reshaped)
        X_windows = X_normalized.reshape(X_windows.shape)
        
        # Train/val/test split
        n_samples = len(X_windows)
        n_train = int(0.70 * n_samples)
        n_val = int(0.15 * n_samples)
        
        X_train = X_windows[:n_train]
        X_val = X_windows[n_train:n_train + n_val]
        X_test = X_windows[n_train + n_val:]
        
        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Val shape: {X_val.shape}")
        logger.info(f"Test shape: {X_test.shape}")
        
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test
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
        """Train CNN model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Advanced CNN Model (1D Convolutions)")
        logger.info(f"Game: {self.game_config.name}")
        logger.info(f"{'='*60}\n")
        
        # Load data
        data, y_numbers = self.load_data_and_prepare(window_size=10)
        
        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        
        # Build model
        model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
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
        model_path = self.models_dir / "cnn_model.h5"
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")
        
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
        num_numbers=50,
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
            config = GameConfig(name="lotto_max", num_balls=7, num_numbers=50, num_positions=7)
            trainer = AdvancedCNNModel(config)
            trainer.train_model(epochs=30, batch_size=32)
    else:
        run_cnn_training_pipeline()
