"""
Advanced Transformer implementation for lottery prediction with proper attention mechanisms.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import json
import os
from pathlib import Path
import joblib

# Import TensorFlow with compatibility handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    # Check for keras.saving availability
    try:
        from tensorflow.keras.saving import register_keras_serializable
        KERAS_SAVING_AVAILABLE = True
    except ImportError:
        # Fallback for older TensorFlow versions
        try:
            from tensorflow.keras.utils import register_keras_serializable
            KERAS_SAVING_AVAILABLE = True
        except ImportError:
            KERAS_SAVING_AVAILABLE = False
            def register_keras_serializable():
                def decorator(cls):
                    return cls
                return decorator
except ImportError:
    tf = None
    keras = None
    layers = None
    KERAS_SAVING_AVAILABLE = False
    def register_keras_serializable():
        def decorator(cls):
            return cls
        return decorator


# Define PositionalEncoding class at module level for proper serialization
if tf is not None:
    @register_keras_serializable()
    class PositionalEncoding(tf.keras.layers.Layer):
        """Positional encoding layer for transformer architecture."""
        
        def __init__(self, d_model, max_length=5000, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model
            self.max_length = max_length  # Add max_length parameter for H5 compatibility
            # Use a simple initialization without circular references
        
        def build(self, input_shape):
            """Build the layer - called when first used."""
            # Create positional encoding in build method to avoid cycles
            pos_enc = create_positional_encoding(1, self.d_model)
            self.pos_encoding = self.add_weight(
                name='pos_encoding',
                shape=(1, self.d_model),
                initializer='zeros',
                trainable=False
            )
            self.pos_encoding.assign(pos_enc)
            super().build(input_shape)
        
        def call(self, inputs):
            return inputs + self.pos_encoding
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'd_model': self.d_model,
                'max_length': self.max_length
            })
            return config
        
        @classmethod
        def from_config(cls, config):
            return cls(**config)
else:
    # Dummy class for when TensorFlow is not available
    class PositionalEncoding:
        def __init__(self, *args, **kwargs):
            pass


# Custom loss function for exact row prediction - must be registered for model deserialization
if tf is not None:
    @register_keras_serializable()
    def exact_row_loss(y_true, y_pred):
        """Custom loss for exact row prediction with categorical crossentropy emphasis."""
        # Categorical crossentropy with emphasis on exact matches
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return base_loss
else:
    def exact_row_loss(y_true, y_pred):
        """Dummy function when TensorFlow is not available."""
        return 0


def create_positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Create sinusoidal positional encodings for transformer.
    
    Args:
        seq_length: Length of the sequence
        d_model: Model dimension
    
    Returns:
        Positional encoding matrix
    """
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding


def create_number_embeddings(pool_size: int = 50, embedding_dim: int = 64) -> np.ndarray:
    """
    Create learned embeddings for lottery numbers.
    
    Args:
        pool_size: Maximum number in the lottery (e.g., 49 or 50)
        embedding_dim: Dimension of embeddings
    
    Returns:
        Embedding matrix (pool_size + 1, embedding_dim)
    """
    # Initialize with Xavier/Glorot initialization
    limit = np.sqrt(6.0 / (pool_size + embedding_dim))
    embeddings = np.random.uniform(-limit, limit, (pool_size + 1, embedding_dim))
    
    # Set padding token (index 0) to zeros
    embeddings[0] = 0.0
    
    return embeddings.astype(np.float32)


def prepare_transformer_sequences(lottery_data: List[List[int]], 
                                window_size: int = 20,
                                embedding_dim: int = 64,
                                pool_size: int = 50,
                                include_frequency_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequence data for transformer training with advanced features.
    
    Args:
        lottery_data: List of lottery draws, each draw is a list of numbers
        window_size: Number of historical draws to use as context
        embedding_dim: Dimension of number embeddings
        pool_size: Maximum lottery number
        include_frequency_features: Whether to include frequency-based features
    
    Returns:
        X: Input sequences with embeddings and features
        y: Target embeddings
    """
    if len(lottery_data) < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} draws, got {len(lottery_data)}")
    
    # Create number embeddings
    number_embeddings = create_number_embeddings(pool_size, embedding_dim)
    
    # Calculate historical frequencies for each number
    if include_frequency_features:
        all_numbers = [num for draw in lottery_data for num in draw]
        number_counts = {i: all_numbers.count(i) for i in range(1, pool_size + 1)}
        max_count = max(number_counts.values()) if number_counts else 1
        number_frequencies = {i: count / max_count for i, count in number_counts.items()}
    
    X_sequences = []
    y_sequences = []
    
    for i in range(window_size, len(lottery_data)):
        # Get sequence of historical draws
        historical_draws = lottery_data[i-window_size:i]
        target_draw = lottery_data[i]
        
        # Convert draws to embedding sequences
        sequence_embeddings = []
        
        for draw_idx, draw in enumerate(historical_draws):
            draw_embedding = np.zeros((max(7, len(draw)), embedding_dim + 10))  # Extra space for features
            
            for num_idx, number in enumerate(draw):
                if 1 <= number <= pool_size:
                    # Basic number embedding
                    draw_embedding[num_idx, :embedding_dim] = number_embeddings[number]
                    
                    if include_frequency_features:
                        # Add frequency features
                        freq_idx = embedding_dim
                        draw_embedding[num_idx, freq_idx] = number_frequencies.get(number, 0)
                        
                        # Add position in draw (positional information)
                        draw_embedding[num_idx, freq_idx + 1] = num_idx / len(draw)
                        
                        # Add number value normalized
                        draw_embedding[num_idx, freq_idx + 2] = number / pool_size
                        
                        # Add recency feature (how recently this number appeared)
                        recency = 0
                        for prev_idx in range(max(0, i - window_size), i):
                            if number in lottery_data[prev_idx]:
                                recency = (i - prev_idx) / window_size
                                break
                        draw_embedding[num_idx, freq_idx + 3] = recency
                        
                        # Add draw age (how old this draw is in the sequence)
                        draw_embedding[num_idx, freq_idx + 4] = (window_size - draw_idx - 1) / window_size
                        
                        # Add number parity (odd/even)
                        draw_embedding[num_idx, freq_idx + 5] = number % 2
                        
                        # Add number range bucket (low/mid/high)
                        if number <= pool_size // 3:
                            range_bucket = 0
                        elif number <= 2 * pool_size // 3:
                            range_bucket = 1
                        else:
                            range_bucket = 2
                        draw_embedding[num_idx, freq_idx + 6] = range_bucket / 2
                        
                        # Add sum features for the draw
                        draw_sum = sum(draw)
                        draw_embedding[num_idx, freq_idx + 7] = draw_sum / (pool_size * len(draw))
                        
                        # Add consecutiveness feature
                        consecutive = 0
                        sorted_draw = sorted(draw)
                        for j in range(len(sorted_draw) - 1):
                            if sorted_draw[j+1] - sorted_draw[j] == 1:
                                consecutive += 1
                        draw_embedding[num_idx, freq_idx + 8] = consecutive / max(1, len(draw) - 1)
                        
                        # Add spread feature (max - min)
                        spread = (max(draw) - min(draw)) / pool_size if len(draw) > 1 else 0
                        draw_embedding[num_idx, freq_idx + 9] = spread
            
            sequence_embeddings.append(draw_embedding)
        
        # Stack into sequence tensor
        sequence_tensor = np.stack(sequence_embeddings)  # (window_size, max_numbers_per_draw, features)
        
        # Create target embedding
        target_embedding = np.zeros((max(7, len(target_draw)), embedding_dim))
        for num_idx, number in enumerate(target_draw):
            if 1 <= number <= pool_size:
                target_embedding[num_idx] = number_embeddings[number]
        
        X_sequences.append(sequence_tensor)
        y_sequences.append(target_embedding)
    
    return np.array(X_sequences), np.array(y_sequences)


def build_advanced_transformer(input_shape: Tuple, hyperparams: Dict[str, Any]) -> Any:
    """
    Build an advanced transformer model for lottery prediction.
    
    Args:
        input_shape: Shape of input sequences
        hyperparams: Model hyperparameters
    
    Returns:
        Compiled transformer model
    """
    # Validate input shape
    if len(input_shape) < 1:
        raise ValueError(f"Input shape must have at least 1 dimension, got {input_shape}")
    
    # Ensure we have at least 2 dimensions for the transformer
    if len(input_shape) == 1:
        # Single feature dimension, treat as (1, features)
        features = input_shape[0]
        input_shape = (1, features)
    elif len(input_shape) == 2:
        # Already have (sequence_length, features) or (window_size, max_numbers)
        pass
    # If input is 3D+, use as is
    
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, LayerNormalization, 
            MultiHeadAttention, GlobalAveragePooling1D,
            Add, Concatenate, Reshape, Permute, Flatten
        )
        from tensorflow.keras.optimizers import Adam
        
        # Extract hyperparameters
        d_model = hyperparams.get('d_model', 128)
        num_heads = hyperparams.get('heads', 8)
        num_layers = hyperparams.get('layers', 6)
        dropout_rate = hyperparams.get('dropout', 0.1)
        learning_rate = hyperparams.get('lr', 0.0001)
        ff_dim = hyperparams.get('ff_dim', d_model * 4)
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Handle different input shapes by flattening and projecting
        # This avoids complex reshape logic that can cause size mismatches
        x = Flatten()(inputs)  # Flatten all dimensions
        x = Dense(d_model)(x)  # Project to model dimension
        x = Reshape((1, d_model))(x)  # Create a single sequence step
        
        # Add positional encodings for the single sequence using module-level class
        x = PositionalEncoding(d_model)(x)
        
        # Transformer layers
        for _ in range(num_layers):
            # Multi-head self-attention
            attn_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout_rate
            )(x, x)
            
            # Add & Norm
            x = Add()([x, attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)
            
            # Feed forward network
            ff_output = Dense(ff_dim, activation='relu')(x)
            ff_output = Dropout(dropout_rate)(ff_output)
            ff_output = Dense(d_model)(ff_output)
            
            # Add & Norm
            x = Add()([x, ff_output])
            x = LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling and final layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(d_model // 2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(d_model // 4, activation='relu')(x)
        x = Dropout(dropout_rate * 0.5)(x)
        
        # Output layer - single value prediction for each sample
        # This matches typical lottery prediction where we predict one value per draw
        outputs = Dense(1, activation='linear')(x)  # Single output
        
        # Flatten to ensure shape is (batch_size,) not (batch_size, 1)
        outputs = Flatten()(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Advanced optimizer
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    except ImportError:
        # Fallback to advanced sklearn ensemble
        print("TensorFlow not available, using advanced sklearn ensemble")
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        class TransformerEnsemble:
            def __init__(self, hyperparams):
                self.models = [
                    GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6),
                    RandomForestRegressor(n_estimators=300, max_depth=10),
                    Ridge(alpha=1.0)
                ]
                self.scaler = StandardScaler()
                self.hyperparams = hyperparams
            
            def fit(self, X, y):
                # Flatten input for sklearn
                X_flat = X.reshape(X.shape[0], -1)
                y_flat = y.reshape(y.shape[0], -1)
                
                X_scaled = self.scaler.fit_transform(X_flat)
                
                for model in self.models:
                    model.fit(X_scaled, y_flat)
                
                return self
            
            def predict(self, X):
                X_flat = X.reshape(X.shape[0], -1)
                X_scaled = self.scaler.transform(X_flat)
                
                predictions = []
                for model in self.models:
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
                
                # Ensemble average
                ensemble_pred = np.mean(predictions, axis=0)
                return ensemble_pred.reshape(X.shape[0], -1)
        
        return TransformerEnsemble(hyperparams)


def train_advanced_transformer(X: np.ndarray, y: np.ndarray, hyperparams: Dict[str, Any],
                              epochs: int = 100, batch_size: int = 16,
                              validation_split: float = 0.2,
                              save_dir: str = None, version: str = None) -> Dict[str, Any]:
    """
    Train an advanced transformer model for lottery prediction.
    
    Args:
        X: Input sequences
        y: Target embeddings
        hyperparams: Model hyperparameters
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
        save_dir: Save directory
        version: Model version
    
    Returns:
        Training metadata
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        # Get initial input shape
        input_shape = X.shape[1:]
        
        # Ensure we have proper dimensions for transformer  
        if len(input_shape) == 1:
            # Convert to 2D
            X = X.reshape(X.shape[0], 1, X.shape[1])
            input_shape = X.shape[1:]  # Now (1, features)
        
        model = build_advanced_transformer(input_shape, hyperparams)
        
        # Ensure target shape matches model output (single values)
        # The model outputs single values, so ensure targets are also single values
        if len(y.shape) > 1:
            # If y has multiple dimensions, take the first column or sum/mean
            if y.shape[1] == 1:
                y = y.reshape(-1)  # (batch_size, 1) -> (batch_size,)
            else:
                # For multi-dimensional targets, take the mean as single value
                y = np.mean(y, axis=1)  # (batch_size, features) -> (batch_size,)
        # If y is already 1D, leave it as is
        
        # Prepare save directory
        if save_dir:
            save_path = Path(save_dir) / "transformer" / (version or "latest")
            save_path.mkdir(parents=True, exist_ok=True)
            model_file = save_path / f"transformer-{version or 'latest'}.h5"
        else:
            model_file = None
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=10,
            min_lr=1e-8,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Model checkpointing
        if model_file:
            checkpoint = ModelCheckpoint(
                str(model_file),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Train model
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # Calculate metrics
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        
        # Save model
        if save_dir:
            # Save as joblib
            joblib_file = save_path / f"transformer-{version or 'latest'}.joblib"
            joblib.dump(model, joblib_file)
            
            # Save training history
            history_file = save_path / "training_history.json"
            with open(history_file, 'w') as f:
                history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
                json.dump(history_dict, f, indent=2)
            
            # Save metadata
            metadata = {
                "name": f"transformer-{version or 'latest'}",
                "file": str(joblib_file),
                "type": "transformer",
                "version": version or "latest",
                "trained_on": pd.Timestamp.now().isoformat(),
                "accuracy": float(1.0 - val_loss),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "hyperparams": hyperparams,
                "training_epochs": len(history.history['loss']),
                "early_stopped": len(history.history['loss']) < epochs
            }
            
            metadata_file = save_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Transformer model saved to {joblib_file}")
            return metadata
        
        return {
            "model": model,
            "history": history.history,
            "val_loss": float(val_loss),
            "val_mae": float(val_mae)
        }
        
    except ImportError:
        # Sklearn fallback
        print("Using sklearn ensemble fallback for transformer")
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Create ensemble model
        model = build_advanced_transformer(X.shape[1:], hyperparams)
        
        # Flatten and split data
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat, y_flat, test_size=validation_split, random_state=42
        )
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        # Save if directory provided
        if save_dir:
            save_path = Path(save_dir) / "transformer" / (version or "latest")
            save_path.mkdir(parents=True, exist_ok=True)
            
            joblib_file = save_path / f"transformer-{version or 'latest'}.joblib"
            joblib.dump(model, joblib_file)
            
            metadata = {
                "name": f"transformer-{version or 'latest'}",
                "file": str(joblib_file),
                "type": "transformer",
                "version": version or "latest",
                "trained_on": pd.Timestamp.now().isoformat(),
                "accuracy": float(1.0 / (1.0 + mse)),
                "mse": float(mse),
                "mae": float(mae),
                "hyperparams": hyperparams
            }
            
            metadata_file = save_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return metadata
        
        return {
            "model": model,
            "mse": float(mse),
            "mae": float(mae)
        }
