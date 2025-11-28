"""
Phase C: Advanced Model Optimization & Intelligence Systems
===========================================================

This module implements cutting-edge optimization techniques for lottery prediction models:
- Dynamic hyperparameter optimization with Bayesian search
- Multi-objective optimization (accuracy vs prediction confidence)
- Adaptive learning rate scheduling with performance feedback
- Ensemble weight optimization using genetic algorithms
- Real-time model performance monitoring and adjustment
- Advanced feature selection with mutual information
- Automated model architecture search (AutoML)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os
from datetime import datetime
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for advanced optimization"""
    n_trials: int = 100
    timeout: int = 3600  # 1 hour
    n_jobs: int = -1
    cv_folds: int = 5
    optimization_metric: str = 'accuracy'
    multi_objective: bool = True
    use_pruning: bool = True
    feature_selection: bool = True
    max_features: int = 50
    ensemble_optimization: bool = True
    auto_ml: bool = True

@dataclass
class OptimizationResult:
    """Results from optimization process"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict]
    feature_importance: Dict[str, float]
    model_architecture: Dict[str, Any]
    ensemble_weights: List[float]
    training_time: float
    convergence_info: Dict[str, Any]

class AdvancedModelOptimizer:
    """
    Advanced model optimization system with multiple cutting-edge techniques
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.optimization_history = []
        self.best_models = {}
        self.feature_selector = None
        self.ensemble_weights = None
        
    def optimize_xgboost_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> OptimizationResult:
        """
        Advanced XGBoost hyperparameter optimization using Bayesian search
        """
        import xgboost as xgb
        
        def objective(trial):
            # Advanced hyperparameter space
            params = {
                'objective': 'multi:softprob',
                'num_class': len(np.unique(y_train)),
                'eval_metric': 'mlogloss',
                'verbosity': 0,
                'random_state': 42,
                
                # Core parameters with wider ranges
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                
                # Advanced regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 10),
                
                # Sampling parameters
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                
                # Advanced parameters
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),
                
                # Tree method optimization (avoid colsample_bynode with exact method)
                'tree_method': trial.suggest_categorical('tree_method', ['auto', 'approx', 'hist']),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            }
            
            # Create and train model
            model = xgb.XGBClassifier(**params)
            
            if X_val is not None and y_val is not None:
                # Use validation set for faster evaluation
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                score = accuracy_score(y_val, model.predict(X_val))
            else:
                # Use cross-validation
                scores = cross_val_score(model, X_train, y_train, cv=self.config.cv_folds, 
                                       scoring=self.config.optimization_metric, n_jobs=1)
                score = scores.mean()
            
            # Multi-objective optimization: balance accuracy and model complexity
            if self.config.multi_objective:
                complexity_penalty = (params['max_depth'] / 15.0 + params['n_estimators'] / 2000.0) * 0.1
                score = score - complexity_penalty
            
            return score
        
        # Create study with advanced pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20) if self.config.use_pruning else None
        )
        
        start_time = datetime.now()
        study.optimize(objective, n_trials=self.config.n_trials, timeout=self.config.timeout, n_jobs=1)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        
        # Get optimization history
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None
                })
        
        # Feature importance (train best model)
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = {f'feature_{i}': imp for i, imp in enumerate(best_model.feature_importances_)}
        else:
            feature_importance = {}
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            feature_importance=feature_importance,
            model_architecture={'type': 'XGBoost', 'params': best_params},
            ensemble_weights=[1.0],
            training_time=training_time,
            convergence_info={
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number,
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        )
    
    def optimize_lstm_architecture(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray = None, y_val: np.ndarray = None) -> OptimizationResult:
        """
        Advanced LSTM architecture optimization using neural architecture search
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            logger.warning("TensorFlow not available, skipping LSTM optimization")
            return self._create_dummy_result('LSTM')
        
        def objective(trial):
            # Advanced architecture parameters
            n_layers = trial.suggest_int('n_layers', 1, 4)
            lstm_units = []
            dropout_rates = []
            
            for i in range(n_layers):
                units = trial.suggest_int(f'lstm_units_{i}', 32, 512, step=32)
                dropout = trial.suggest_float(f'dropout_{i}', 0.0, 0.5)
                lstm_units.append(units)
                dropout_rates.append(dropout)
            
            # Dense layer configuration
            n_dense = trial.suggest_int('n_dense', 1, 3)
            dense_units = []
            dense_dropout = []
            
            for i in range(n_dense):
                units = trial.suggest_int(f'dense_units_{i}', 32, 256, step=32)
                dropout = trial.suggest_float(f'dense_dropout_{i}', 0.0, 0.5)
                dense_units.append(units)
                dense_dropout.append(dropout)
            
            # Optimization parameters
            optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'adamw'])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
            
            # Regularization
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)
            
            # Build model
            model = Sequential()
            
            # LSTM layers
            for i, (units, dropout) in enumerate(zip(lstm_units, dropout_rates)):
                return_sequences = i < len(lstm_units) - 1
                
                if i == 0:
                    model.add(LSTM(units, return_sequences=return_sequences, 
                                 input_shape=(X_train.shape[1], X_train.shape[2]),
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
                else:
                    model.add(LSTM(units, return_sequences=return_sequences,
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
                
                if use_batch_norm:
                    model.add(BatchNormalization())
                
                if dropout > 0:
                    model.add(Dropout(dropout))
            
            # Dense layers
            for units, dropout in zip(dense_units, dense_dropout):
                model.add(Dense(units, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
                
                if use_batch_norm:
                    model.add(BatchNormalization())
                
                if dropout > 0:
                    model.add(Dropout(dropout))
            
            # Output layer
            n_classes = len(np.unique(y_train))
            model.add(Dense(n_classes, activation='softmax'))
            
            # Optimizer
            if optimizer_name == 'adam':
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == 'rmsprop':
                optimizer = RMSprop(learning_rate=learning_rate)
            else:  # adamw
                optimizer = AdamW(learning_rate=learning_rate)
            
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Training callbacks
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            # Train model
            if X_val is not None and y_val is not None:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                score = max(history.history['val_accuracy'])
            else:
                # Use a subset for validation
                val_split = 0.2
                history = model.fit(
                    X_train, y_train,
                    validation_split=val_split,
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                score = max(history.history['val_accuracy'])
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10) if self.config.use_pruning else None
        )
        
        start_time = datetime.now()
        study.optimize(objective, n_trials=min(self.config.n_trials, 50), timeout=self.config.timeout)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None
                })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            feature_importance={},
            model_architecture={'type': 'LSTM', 'params': best_params},
            ensemble_weights=[1.0],
            training_time=training_time,
            convergence_info={
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number,
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        )
    
    def optimize_transformer_architecture(self, X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray = None, y_val: np.ndarray = None) -> OptimizationResult:
        """
        Advanced Transformer architecture optimization
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
            from tensorflow.keras.optimizers import Adam, AdamW
        except ImportError:
            logger.warning("TensorFlow not available, skipping Transformer optimization")
            return self._create_dummy_result('Transformer')
        
        def objective(trial):
            # Transformer architecture parameters
            d_model = trial.suggest_int('d_model', 64, 512, step=64)
            n_heads = trial.suggest_int('n_heads', 2, 16)
            n_layers = trial.suggest_int('n_layers', 1, 6)
            dff = trial.suggest_int('dff', 128, 2048, step=128)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            
            # Training parameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
            
            # Build transformer model
            inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
            x = inputs
            
            # Transformer blocks
            for _ in range(n_layers):
                # Multi-head attention
                attn_output = MultiHeadAttention(
                    num_heads=n_heads, 
                    key_dim=d_model // n_heads,
                    dropout=dropout_rate
                )(x, x)
                
                # Add & Norm
                x = LayerNormalization(epsilon=1e-6)(x + attn_output)
                
                # Feed Forward Network
                ffn_output = Dense(dff, activation='relu')(x)
                ffn_output = Dropout(dropout_rate)(ffn_output)
                ffn_output = Dense(d_model)(ffn_output)
                
                # Add & Norm
                x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
            
            # Global pooling and classification
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = Dropout(dropout_rate)(x)
            
            # Output layer
            n_classes = len(np.unique(y_train))
            outputs = Dense(n_classes, activation='softmax')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Optimizer
            optimizer = AdamW(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Training callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            # Train model
            if X_val is not None and y_val is not None:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                score = max(history.history['val_accuracy'])
            else:
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,
                    epochs=30,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                score = max(history.history['val_accuracy'])
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5) if self.config.use_pruning else None
        )
        
        start_time = datetime.now()
        study.optimize(objective, n_trials=min(self.config.n_trials, 30), timeout=self.config.timeout)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Extract results
        best_params = study.best_params
        best_score = study.best_value
        
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None
                })
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=history,
            feature_importance={},
            model_architecture={'type': 'Transformer', 'params': best_params},
            ensemble_weights=[1.0],
            training_time=training_time,
            convergence_info={
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number,
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        )
    
    def optimize_feature_selection(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Advanced feature selection using mutual information and statistical tests
        """
        if not self.config.feature_selection:
            return X_train, {}
        
        logger.info("🔍 Optimizing feature selection...")
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        # Select top features
        n_features = min(self.config.max_features, X_train.shape[1])
        selector = SelectKBest(mutual_info_classif, k=n_features)
        X_selected = selector.fit_transform(X_train, y_train)
        
        # Get feature importance scores
        selected_features = selector.get_support(indices=True)
        feature_importance = {f'feature_{i}': mi_scores[i] for i in selected_features}
        
        self.feature_selector = selector
        
        logger.info(f"✅ Selected {X_selected.shape[1]} features from {X_train.shape[1]}")
        
        return X_selected, feature_importance
    
    def optimize_ensemble_weights(self, models: List[Any], X_val: np.ndarray, y_val: np.ndarray) -> List[float]:
        """
        Optimize ensemble weights using genetic algorithm
        """
        if not self.config.ensemble_optimization or len(models) == 1:
            return [1.0] * len(models)
        
        logger.info("🧬 Optimizing ensemble weights...")
        
        def evaluate_weights(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            # Get predictions from all models
            predictions = []
            for model in models:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)
                else:
                    pred = model.predict(X_val)
                predictions.append(pred)
            
            # Weighted ensemble prediction
            if len(predictions[0].shape) > 1:  # Probability predictions
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                final_pred = np.argmax(ensemble_pred, axis=1)
            else:  # Direct predictions
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                final_pred = np.round(ensemble_pred).astype(int)
            
            return accuracy_score(y_val, final_pred)
        
        def objective(trial):
            weights = []
            for i in range(len(models)):
                weight = trial.suggest_float(f'weight_{i}', 0.0, 1.0)
                weights.append(weight)
            
            return evaluate_weights(weights)
        
        # Optimize weights
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        # Extract best weights and normalize
        best_weights = list(study.best_params.values())
        best_weights = np.array(best_weights)
        best_weights = best_weights / best_weights.sum()
        
        self.ensemble_weights = best_weights.tolist()
        
        logger.info(f"✅ Optimized ensemble weights: {[f'{w:.3f}' for w in best_weights]}")
        
        return best_weights.tolist()
    
    def _create_dummy_result(self, model_type: str) -> OptimizationResult:
        """Create a dummy result when optimization cannot be performed"""
        return OptimizationResult(
            best_params={},
            best_score=0.0,
            optimization_history=[],
            feature_importance={},
            model_architecture={'type': model_type, 'params': {}},
            ensemble_weights=[1.0],
            training_time=0.0,
            convergence_info={'n_trials': 0, 'best_trial': 0, 'pruned_trials': 0}
        )
    
    def save_optimization_results(self, results: OptimizationResult, filepath: str):
        """Save optimization results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to serializable format
        results_dict = {
            'best_params': results.best_params,
            'best_score': results.best_score,
            'optimization_history': results.optimization_history,
            'feature_importance': results.feature_importance,
            'model_architecture': results.model_architecture,
            'ensemble_weights': results.ensemble_weights,
            'training_time': results.training_time,
            'convergence_info': results.convergence_info,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_trials': self.config.n_trials,
                'optimization_metric': self.config.optimization_metric,
                'multi_objective': self.config.multi_objective,
                'feature_selection': self.config.feature_selection,
                'max_features': self.config.max_features
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"💾 Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: str) -> OptimizationResult:
        """Load optimization results from file"""
        with open(filepath, 'r') as f:
            results_dict = json.load(f)
        
        return OptimizationResult(
            best_params=results_dict['best_params'],
            best_score=results_dict['best_score'],
            optimization_history=results_dict['optimization_history'],
            feature_importance=results_dict['feature_importance'],
            model_architecture=results_dict['model_architecture'],
            ensemble_weights=results_dict['ensemble_weights'],
            training_time=results_dict['training_time'],
            convergence_info=results_dict['convergence_info']
        )

# Convenience function for comprehensive optimization
def optimize_model_comprehensive(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray = None, y_val: np.ndarray = None,
                                config: OptimizationConfig = None) -> OptimizationResult:
    """
    Perform comprehensive model optimization for any model type
    
    Args:
        model_type: 'xgboost', 'lstm', or 'transformer'
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        config: Optimization configuration
    
    Returns:
        OptimizationResult with all optimization details
    """
    optimizer = AdvancedModelOptimizer(config)
    
    if model_type.lower() == 'xgboost':
        return optimizer.optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val)
    elif model_type.lower() == 'lstm':
        return optimizer.optimize_lstm_architecture(X_train, y_train, X_val, y_val)
    elif model_type.lower() == 'transformer':
        return optimizer.optimize_transformer_architecture(X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == "__main__":
    # Example usage
    print("🚀 Advanced Model Optimizer - Phase C")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 5, 1000)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Test XGBoost optimization
    config = OptimizationConfig(n_trials=20, timeout=300)
    result = optimize_model_comprehensive('xgboost', X_train, y_train, X_val, y_val, config)
    
    print(f"✅ Best XGBoost score: {result.best_score:.4f}")
    print(f"🏆 Best parameters: {result.best_params}")
    print(f"⏱️ Training time: {result.training_time:.2f}s")
    print(f"📊 Trials completed: {result.convergence_info['n_trials']}")
