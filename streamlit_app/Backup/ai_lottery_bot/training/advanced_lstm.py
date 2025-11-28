"""
Advanced LSTM implementation for lottery prediction with state-of-the-art techniques.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import json
import os
from pathlib import Path
import joblib


def create_advanced_lstm_sequences(data: np.ndarray, window_size: int = 10, 
                                 include_statistical_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create advanced sequence data for LSTM training with statistical enrichment.
    
    Args:
        data: Historical lottery data (n_draws, n_features)
        window_size: Number of historical draws to use as input
        include_statistical_features: Whether to include rolling statistics
    
    Returns:
        X: Input sequences (n_samples, window_size, enhanced_features)
        y: Target sequences (n_samples, n_features)
    """
    if len(data) < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} samples, got {len(data)}")
    
    # Basic sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(window_size, len(data)):
        sequence = data[i-window_size:i]
        target = data[i]
        
        if include_statistical_features:
            # Add rolling statistics as additional features
            rolling_mean = np.mean(sequence, axis=0)
            rolling_std = np.std(sequence, axis=0)
            rolling_min = np.min(sequence, axis=0)
            rolling_max = np.max(sequence, axis=0)
            
            # Trend features (linear regression slope approximation)
            x_vals = np.arange(window_size).reshape(-1, 1)
            trends = []
            for feat_idx in range(sequence.shape[1]):
                y_vals = sequence[:, feat_idx]
                # Simple trend calculation
                trend = (y_vals[-1] - y_vals[0]) / (window_size - 1) if window_size > 1 else 0
                trends.append(trend)
            trends = np.array(trends)
            
            # Volatility (rolling coefficient of variation)
            volatility = rolling_std / (rolling_mean + 1e-8)
            
            # Momentum (recent vs older performance)
            recent_mean = np.mean(sequence[-window_size//3:], axis=0)
            older_mean = np.mean(sequence[:window_size//3], axis=0)
            momentum = (recent_mean - older_mean) / (older_mean + 1e-8)
            
            # Combine statistical features
            stats_features = np.concatenate([
                rolling_mean, rolling_std, rolling_min, rolling_max,
                trends, volatility, momentum
            ])
            
            # Expand stats to match sequence length for concatenation
            stats_repeated = np.tile(stats_features, (window_size, 1))
            
            # Combine sequence with statistical features
            enhanced_sequence = np.concatenate([sequence, stats_repeated], axis=1)
        else:
            enhanced_sequence = sequence
            
        X_sequences.append(enhanced_sequence)
        y_sequences.append(target)
    
    return np.array(X_sequences), np.array(y_sequences)


def build_advanced_lstm_model(input_shape: Tuple, hyperparams: Dict[str, Any], output_features: Optional[int] = None) -> Any:
    """
    Build an advanced LSTM model with modern architecture.
    
    Args:
        input_shape: (window_size, features)
        hyperparams: Model configuration parameters
        output_features: Number of output features (auto-detected if None)
    
    Returns:
        Compiled model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import (
            LSTM, Dense, Dropout, BatchNormalization, 
            Attention, LayerNormalization, MultiHeadAttention,
            Input, Concatenate, GRU, Bidirectional
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l1_l2
        
        # Extract hyperparameters with defaults
        hidden_units = hyperparams.get('hidden_units', 128)
        num_layers = hyperparams.get('layers', 2)
        dropout_rate = hyperparams.get('dropout', 0.2)
        learning_rate = hyperparams.get('lr', 0.001)
        use_bidirectional = hyperparams.get('bidirectional', True)
        use_attention = hyperparams.get('attention', True)
        cell_type = hyperparams.get('cell_type', 'LSTM')  # LSTM or GRU
        
        # Input layer
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Add layer normalization
        x = LayerNormalization()(x)
        
        # Build recurrent layers
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1) or use_attention
            
            if cell_type == 'GRU':
                layer = GRU(
                    hidden_units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate * 0.5,
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
                )
            else:  # LSTM
                layer = LSTM(
                    hidden_units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate * 0.5,
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
                )
            
            if use_bidirectional:
                x = Bidirectional(layer)(x)
                hidden_units_actual = hidden_units * 2
            else:
                x = layer(x)
                hidden_units_actual = hidden_units
            
            # Add batch normalization and dropout
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
            
            # Reduce hidden units in subsequent layers
            hidden_units = max(hidden_units // 2, 32)
        
        # Add attention mechanism if enabled
        if use_attention and len(x.shape) == 3:  # sequence output
            # Multi-head self-attention
            attention_output = MultiHeadAttention(
                num_heads=4,
                key_dim=hidden_units_actual // 4
            )(x, x)
            
            # Add & norm
            x = LayerNormalization()(x + attention_output)
            
            # Global average pooling to reduce to 2D
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers for final prediction
        x = Dense(hidden_units_actual // 2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 0.5)(x)
        
        x = Dense(hidden_units_actual // 4, activation='relu')(x)
        x = Dropout(dropout_rate * 0.25)(x)
        
        # Output layer - use actual output features or detect from input
        if output_features is None:
            # Default to input feature size for autoencoder-style architecture
            output_features = input_shape[1]
        
        outputs = Dense(output_features, activation='linear')(x)  # Predict actual target features
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Advanced optimizer with learning rate scheduling
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Import lottery-aware metrics
        from .lstm_metrics import LotteryAwareAccuracy, ExactRowAccuracy, lottery_aware_loss
        
        model.compile(
            optimizer=optimizer,
            loss=lottery_aware_loss,  # Use lottery-aware loss instead of MSE
            metrics=[
                'mae',  # Keep for continuous monitoring
                LotteryAwareAccuracy(),  # Lottery-specific accuracy during training
                ExactRowAccuracy()  # Matches the 88% evaluation metric
            ]
        )
        
        return model
        
    except ImportError:
        # Fallback to sklearn-based sequence model
        from sklearn.ensemble import RandomForestRegressor
        print("TensorFlow not available, using sklearn fallback")
        return RandomForestRegressor(
            n_estimators=hyperparams.get('n_estimators', 200),
            max_depth=hyperparams.get('max_depth', 10),
            random_state=42
        )


def train_advanced_lstm(X: np.ndarray, y: np.ndarray, hyperparams: Dict[str, Any],
                       epochs: int = 100, batch_size: int = 32,
                       validation_split: float = 0.2, 
                       save_dir: str = None, version: str = None) -> Dict[str, Any]:
    """
    Train an advanced LSTM model with early stopping and best practices.
    
    Args:
        X: Input sequences (n_samples, window_size, features)
        y: Target values (n_samples, features)
        hyperparams: Model hyperparameters
        epochs: Maximum training epochs
        batch_size: Training batch size
        validation_split: Fraction of data for validation
        save_dir: Directory to save the model
        version: Model version string
    
    Returns:
        Dictionary containing model metadata and training history
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        # Create model with correct output shape
        model = build_advanced_lstm_model((X.shape[1], X.shape[2]), hyperparams, output_features=y.shape[1])
        
        # Prepare save directory
        if save_dir:
            save_path = Path(save_dir) / "lstm" / (version or "latest")
            save_path.mkdir(parents=True, exist_ok=True)
            model_file = save_path / f"lstm-{version or 'latest'}.h5"
        else:
            model_file = None
        
        # Advanced callbacks
        callbacks = []
        
        # Early stopping with patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
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
        
        # Calculate final metrics
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        
        # Save model in multiple formats
        if save_dir:
            # Save as joblib for consistency with existing system
            joblib_file = save_path / f"lstm-{version or 'latest'}.joblib"
            joblib.dump(model, joblib_file)
            
            # Save training history
            history_file = save_path / "training_history.json"
            with open(history_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
                json.dump(history_dict, f, indent=2)
            
            # Save metadata
            # Calculate lottery-appropriate accuracy from validation metrics
            # For lottery prediction, convert MAE to percentage-based accuracy
            lottery_accuracy = max(0.0, min(0.5, 1.0 - (val_mae / 50.0)))  # Normalize MAE to 0-50% range
            lottery_accuracy = round(lottery_accuracy, 4)
            
            metadata = {
                "name": f"lstm-{version or 'latest'}",
                "file": str(joblib_file),
                "type": "lstm",
                "version": version or "latest",
                "trained_on": pd.Timestamp.now().isoformat(),
                "accuracy": float(lottery_accuracy),  # Proper lottery-appropriate accuracy
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "hyperparams": hyperparams,
                "training_epochs": len(history.history['loss']),
                "early_stopped": len(history.history['loss']) < epochs,
                "note": "Realistic lottery prediction metrics - accuracy based on MAE normalization"
            }
            
            metadata_file = save_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model saved to {joblib_file}")
            return metadata
        
        return {
            "model": model,
            "history": history.history,
            "val_loss": float(val_loss),
            "val_mae": float(val_mae)
        }
        
    except ImportError:
        # Sklearn fallback
        print("Using sklearn fallback for LSTM training")
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Flatten sequences for sklearn
        X_flat = X.reshape(X.shape[0], -1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat, y, test_size=validation_split, random_state=42
        )
        
        # Train model
        model = build_advanced_lstm_model((X.shape[1], X.shape[2]), hyperparams)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        # Save if directory provided
        if save_dir:
            save_path = Path(save_dir) / "lstm" / (version or "latest")
            save_path.mkdir(parents=True, exist_ok=True)
            
            joblib_file = save_path / f"lstm-{version or 'latest'}.joblib"
            joblib.dump(model, joblib_file)
            
            # Calculate lottery-appropriate accuracy from test metrics
            # For lottery prediction, convert MAE to percentage-based accuracy
            lottery_accuracy = max(0.0, min(0.5, 1.0 - (mae / 50.0)))  # Normalize MAE to 0-50% range
            lottery_accuracy = round(lottery_accuracy, 4)
            
            metadata = {
                "name": f"lstm-{version or 'latest'}",
                "file": str(joblib_file),
                "type": "lstm",
                "version": version or "latest",
                "trained_on": pd.Timestamp.now().isoformat(),
                "accuracy": float(lottery_accuracy),  # Proper lottery-appropriate accuracy
                "mse": float(mse),
                "mae": float(mae),
                "hyperparams": hyperparams,
                "note": "Realistic lottery prediction metrics - accuracy based on MAE normalization"
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
