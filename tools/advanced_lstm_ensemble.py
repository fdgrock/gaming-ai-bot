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
        'lotto_max': GameConfig(name='lotto_max', num_numbers=52, num_positions=7)
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
    LOOKBACK_DRAWS = 15  # look back 15 COMPLETE draws (no partial-draw leakage)

    def __init__(self, game: str = None):
        """
        Initialize the ensemble trainer.

        Args:
            game: Optional game name to filter training. If None, trains all games.
        """
        self.scaler = {}
        self.training_logs = []
        self.game_filter = game  # Optional: can filter to specific game

    def create_draw_aligned_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        draw_indices: np.ndarray,
        num_positions: int,
        target_labels: np.ndarray = None,
    ):
        """Create draw-aligned LSTM sequences that prevent within-draw leakage.

        Each sequence contains exactly LOOKBACK_DRAWS complete draws.
        The target is the SMALLEST DRAWN number of the draw immediately after the
        window. target_labels (1=drawn, 0=not drawn) is used to identify which
        rows belong to drawn numbers; falls back to all rows if not supplied.

        Returns:
            X_seq:         (n, LOOKBACK_DRAWS * num_positions, n_features)
            y_seq:         (n,)         -- 0-based class of first drawn number
            t_draw:        (n,)         -- draw_idx of the target draw
            all_drawn:     list[list]   -- all 0-based drawn class indices per sequence
        """
        seq_len = self.LOOKBACK_DRAWS * num_positions

        # Group rows by draw in chronological order
        unique_draws = sorted(np.unique(draw_indices))
        draw_to_rows: dict = {}
        for row_i, d in enumerate(draw_indices):
            draw_to_rows.setdefault(int(d), []).append(row_i)

        X_seq_list, y_list, t_draw_list, all_drawn_list = [], [], [], []

        # Slide draw-by-draw (stride = 1 draw)
        for di in range(self.LOOKBACK_DRAWS, len(unique_draws)):
            target_draw = unique_draws[di]
            input_draws = unique_draws[di - self.LOOKBACK_DRAWS: di]

            # Collect input rows from the LOOKBACK_DRAWS draws before the target
            input_rows = []
            for d in input_draws:
                input_rows.extend(draw_to_rows.get(int(d), []))

            target_rows = draw_to_rows.get(int(target_draw), [])

            # Identify rows that correspond to actually-drawn numbers (target == 1).
            # When target_labels is available (all 52 numbers per draw in the feature
            # matrix), only drawn rows have target==1.  Without labels fall back to
            # all rows so the function remains backward-compatible.
            if target_labels is not None:
                drawn_rows = [r for r in target_rows if target_labels[r] == 1]
            else:
                drawn_rows = target_rows

            if not input_rows or not drawn_rows:
                continue

            X_win = X[input_rows]  # shape: (LOOKBACK_DRAWS * num_pos, n_features) ideally
            # Pad at front or trim at end if draw sizes are uneven
            if len(X_win) < seq_len:
                pad = np.zeros((seq_len - len(X_win), X_win.shape[1]), dtype=np.float32)
                X_win = np.vstack([pad, X_win])
            elif len(X_win) > seq_len:
                X_win = X_win[-seq_len:]

            X_seq_list.append(X_win)
            # Use first drawn number (sorted by class = sorted by lottery number) as primary target
            drawn_classes = sorted([int(y[r]) for r in drawn_rows])
            y_list.append(drawn_classes[0])
            all_drawn_list.append(drawn_classes)
            t_draw_list.append(int(target_draw))

        return (
            np.array(X_seq_list, dtype=np.float32),
            np.array(y_list, dtype=np.int32),
            np.array(t_draw_list, dtype=int),
            all_drawn_list,
        )
    
    def load_data_and_prepare(self, game_name: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare data for training.

        Returns:
            (X, y, draw_indices, target_labels)
            X             -- normalized feature matrix (n_rows, n_features)
            y             -- 0-based lottery number labels per row
            draw_indices  -- draw_idx per row (for skipgram/distribution lookup)
            target_labels -- 1 if the row's number was drawn in that draw, 0 otherwise
        """
        logger.info(f"Loading data for {game_name} with seed {seed}...")

        features_path = DATA_DIR / game_name / "temporal_features.parquet"
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")

        df = pd.read_parquet(features_path)

        # Merge global draw features (per-draw stats -> broadcast to all per-number rows)
        _draw_col = 'draw_idx' if 'draw_idx' in df.columns else 'draw_index' if 'draw_index' in df.columns else None
        try:
            global_df = pd.read_parquet(DATA_DIR / game_name / "global_features.parquet")
            _gdraw_col = 'draw_idx' if 'draw_idx' in global_df.columns else 'draw_index'
            _global_feat_cols = [c for c in global_df.columns if c not in {'draw_idx', 'draw_index'}]
            global_subset = global_df[[_gdraw_col] + _global_feat_cols].rename(
                columns={_gdraw_col: _draw_col or 'draw_idx'}
            )
            df = df.merge(global_subset, on=_draw_col or 'draw_idx', how='left')
            logger.info(f"Merged global features: +{len(_global_feat_cols)} columns -> {df.shape[1]} total")
        except Exception as _ge:
            logger.warning(f"Could not merge global features (continuing with temporal only): {_ge}")

        # Capture draw_indices for target alignment (before dropping the column)
        if _draw_col and _draw_col in df.columns:
            draw_indices = df[_draw_col].values.astype(int)
        else:
            draw_indices = np.arange(len(df), dtype=int)

        # Capture target labels (1 = this number was drawn in this draw, 0 = not drawn)
        # The feature matrix includes ALL candidate numbers per draw; target identifies drawn ones.
        target_labels = df['target'].values.astype(np.int32) if 'target' in df.columns else np.zeros(len(df), dtype=np.int32)

        # Extract features - drop ALL non-feature columns to prevent data leakage
        _NON_FEAT = {'draw_index', 'draw_idx', 'number', 'target', 'draw_date', 'result_numbers'}
        feature_cols = [col for col in df.columns if col not in _NON_FEAT]
        X = df[feature_cols].values.astype(np.float32)

        # Extract labels (1-based lottery numbers -> 0-based for sparse_categorical_crossentropy)
        if 'number' in df.columns:
            y = (df['number'].values.astype(int) - 1).astype(np.int32)
        else:
            y = np.zeros(len(X), dtype=np.int32)

        # Normalize features
        if game_name not in self.scaler:
            self.scaler[game_name] = StandardScaler()
            X = self.scaler[game_name].fit_transform(X)
        else:
            X = self.scaler[game_name].transform(X)

        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features for {game_name}")
        drawn_count = int(target_labels.sum()) if target_labels is not None else 0
        logger.info(f"Target labels: {drawn_count} drawn rows out of {len(target_labels)} total")

        return X, y, draw_indices, target_labels

    def build_lstm_model(self, game: GameConfig, n_features: int, seq_len: int = None) -> Model:
        """Build LSTM encoder-decoder architecture.

        Args:
            seq_len: Sequence length in rows (LOOKBACK_DRAWS * num_positions).
                     Derived from game config if not supplied.
        """
        if seq_len is None:
            seq_len = self.LOOKBACK_DRAWS * game.num_positions

        # Input: 3D tensor [batch, seq_len, n_features]
        inputs = layers.Input(shape=(seq_len, n_features), name='input_sequence')
        
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
        
        # Decoder: LSTM with attention context, initialized with encoder states
        attention_expanded = layers.Reshape((1, 256))(attention_output)  # [batch, 1, 256]
        decoder = layers.LSTM(256, name='decoder_lstm')
        decoder_output = decoder(attention_expanded, initial_state=[state_h, state_c])
        
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

        # Multi-task loss weights:
        # - primary: sparse_categorical_crossentropy (0-based integer labels)
        # - skipgram/distribution: categorical_crossentropy (normalized multi-hot vectors)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss={
                'primary': 'sparse_categorical_crossentropy',
                'skipgram': 'categorical_crossentropy',
                'distribution': 'categorical_crossentropy'
            },
            loss_weights={'primary': 0.5, 'skipgram': 0.25, 'distribution': 0.25},
            metrics={
                'primary': 'accuracy',
            }
        )

        return model
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          all_drawn_classes: list = None) -> ModelMetrics:
        """Calculate evaluation metrics.

        Args:
            y_true:           (n,) array of 0-based primary target (smallest drawn number).
            y_pred:           (n, num_classes) probability array from the primary head.
            all_drawn_classes: list of lists — all 0-based drawn class indices per sequence.
                               When provided, multi-ball hit-rate is used instead of
                               single-label accuracy, giving a realistic lottery metric.
        """
        top_5_indices = np.argsort(y_pred, axis=1)[:, -5:]
        top_10_indices = np.argsort(y_pred, axis=1)[:, -10:]

        if all_drawn_classes is not None and len(all_drawn_classes) == len(y_pred):
            # Multi-ball hit rate: fraction of drawn numbers that appear in top-K predictions
            top_5_sets  = [set(row) for row in top_5_indices]
            top_10_sets = [set(row) for row in top_10_indices]
            top_5_accuracy = float(np.mean([
                len(set(drawn) & t5) / max(len(drawn), 1)
                for drawn, t5 in zip(all_drawn_classes, top_5_sets)
            ]))
            top_10_accuracy = float(np.mean([
                len(set(drawn) & t10) / max(len(drawn), 1)
                for drawn, t10 in zip(all_drawn_classes, top_10_sets)
            ]))
        else:
            # Fallback: single-label accuracy against the primary target
            top_5_accuracy  = float(np.mean([y in t5  for y, t5  in zip(y_true, top_5_indices)]))
            top_10_accuracy = float(np.mean([y in t10 for y, t10 in zip(y_true, top_10_indices)]))
        
        # KL divergence
        y_pred_normalized = np.clip(y_pred, 1e-7, 1.0)
        uniform = np.ones_like(y_pred) / y_pred.shape[1]
        kl_div = np.mean(np.sum(uniform * np.log(uniform / y_pred_normalized), axis=1))

        # Log loss
        log_loss_val = -np.mean(np.log(y_pred_normalized[np.arange(len(y_true)), y_true]))

        # Composite score: exp(-kl) keeps calibration signal even when KL >> 1
        composite = 0.6 * top_5_accuracy + 0.4 * float(np.exp(-max(0.0, kl_div)))

        # Sanity checks - warn if metrics look impossibly good
        import math as _math
        if top_5_accuracy > 0.95:
            logger.warning(f"[SANITY] Suspiciously high Top-5 accuracy: {top_5_accuracy:.4f} - verify target alignment")
        if log_loss_val <= 0.0:
            logger.warning(f"[SANITY] Non-positive log-loss: {log_loss_val:.6f} - likely a metric calculation error")
        if kl_div > _math.log(y_pred.shape[1]) + 1.0:
            logger.warning(f"[SANITY] KL divergence {kl_div:.4f} exceeds theoretical max - check predictions")

        return ModelMetrics(
            top_5_accuracy=top_5_accuracy,
            top_10_accuracy=top_10_accuracy,
            kl_divergence=kl_div,
            log_loss=log_loss_val,
            composite_score=composite
        )
    
    def train_ensemble_variant(self, game: GameConfig, variant_index: int, seed: int):
        """Train a single variant of the ensemble."""
        logger.info(f"Training {game.name} LSTM variant {variant_index + 1}/{self.NUM_VARIANTS_PER_GAME} (seed={seed})")
        
        # Set random seeds
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Load data (X, y, draw_indices, target_labels)
        X, y, draw_indices, target_labels = self.load_data_and_prepare(game.name, seed)

        # Draw-aligned sequences: each window = LOOKBACK_DRAWS complete draws,
        # target = smallest DRAWN number of the NEXT draw.
        # target_labels identifies which rows correspond to drawn numbers so that
        # we never use the "row order == class 0" fallback that produced degenerate metrics.
        X_sequences, y_sequences, target_draw_idxs, all_drawn_classes = self.create_draw_aligned_sequences(
            X, y, draw_indices, game.num_positions, target_labels
        )
        seq_len = X_sequences.shape[1]  # LOOKBACK_DRAWS * num_positions

        logger.info(f"Draw-aligned sequences: X={X_sequences.shape}, y={y_sequences.shape}, "
                    f"seq_len={seq_len} ({self.LOOKBACK_DRAWS} draws x {game.num_positions} balls)")

        # Build skipgram (multi-hot, normalized) and distribution targets aligned to sequences
        n_classes = game.num_numbers
        _def_sg = np.ones(n_classes, dtype=np.float32) / n_classes
        _def_dt = np.ones(n_classes, dtype=np.float32) / n_classes

        try:
            _sg_df  = pd.read_parquet(DATA_DIR / game.name / "skipgram_targets.parquet")
            _dt_df  = pd.read_parquet(DATA_DIR / game.name / "distribution_targets.parquet")
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

            y_skipgram = np.array([_sg_lookup.get(int(d), _def_sg) for d in target_draw_idxs])
            y_dist     = np.array([_dt_lookup.get(int(d), _def_dt) for d in target_draw_idxs])
            logger.info("Using actual skipgram + distribution targets for multi-task heads")
        except Exception as _te:
            logger.warning(f"Could not load skipgram/distribution targets, using uniform fallback: {_te}")
            y_skipgram = np.tile(_def_sg, (len(X_sequences), 1))
            y_dist     = np.tile(_def_dt, (len(X_sequences), 1))

        # Temporal split - preserve chronological order (draw-aligned sequences are already ordered)
        n = len(X_sequences)
        train_end = int(n * (1.0 - self.VALIDATION_SPLIT - self.TEST_SPLIT))
        val_end   = int(n * (1.0 - self.TEST_SPLIT))

        X_train, X_val, X_test = X_sequences[:train_end], X_sequences[train_end:val_end], X_sequences[val_end:]
        y_train, y_val, y_test = y_sequences[:train_end], y_sequences[train_end:val_end], y_sequences[val_end:]
        sg_train, sg_val, sg_test = y_skipgram[:train_end], y_skipgram[train_end:val_end], y_skipgram[val_end:]
        dt_train, dt_val, dt_test = y_dist[:train_end], y_dist[train_end:val_end], y_dist[val_end:]
        all_drawn_test = all_drawn_classes[val_end:]  # drawn numbers for multi-ball metric

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Build model — pass seq_len so the Input layer matches the draw-aligned window size
        model = self.build_lstm_model(game, X.shape[1], seq_len=seq_len)

        # Train with distinct targets for each head
        history = model.fit(
            X_train, {'primary': y_train, 'skipgram': sg_train, 'distribution': dt_train},
            validation_data=(X_val, {'primary': y_val, 'skipgram': sg_val, 'distribution': dt_val}),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
                )
            ]
        )

        # Evaluate — multi-ball hit rate against all drawn numbers in each test draw
        predictions, _, _ = model.predict(X_test, verbose=0)
        metrics = self.calculate_metrics(y_test, predictions, all_drawn_classes=all_drawn_test)

        logger.info(f"Variant {variant_index + 1} Results:")
        logger.info(f"  Top-5 Multi-ball Hit Rate: {metrics.top_5_accuracy:.4f}")
        logger.info(f"  Top-10 Multi-ball Hit Rate: {metrics.top_10_accuracy:.4f}")
        logger.info(f"  KL Divergence: {metrics.kl_divergence:.4f}")
        logger.info(f"  Composite Score: {metrics.composite_score:.4f}")

        # Save model (.keras format handles custom layers without extra registry)
        model_dir = MODELS_DIR / game.name / "lstm_variants"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"lstm_variant_{variant_index + 1}_seed_{seed}.keras"
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
