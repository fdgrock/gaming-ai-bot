import joblib
import json
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Phase C Optimization imports
try:
    from ..optimization.phase_c_integration import (
        create_phase_c_optimizer, PhaseC_IntegratedOptimizer, OptimizationConfig
    )
    PHASE_C_AVAILABLE = True
except ImportError:
    PHASE_C_AVAILABLE = False

# Enhanced features import
try:
    from ..features.advanced_features import create_ultra_high_accuracy_features
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False


MODEL_DIR = Path("models")


def enhance_features_for_phase_c(features: np.ndarray, draws: List[List[int]], 
                               draw_dates: List[str], game_type: str) -> np.ndarray:
    """
    Enhance features specifically for Phase C optimization
    
    Args:
        features: Base feature matrix
        draws: Original draw data
        draw_dates: Draw dates for temporal analysis
        game_type: Game type for contextual features
    
    Returns:
        Enhanced feature matrix optimized for Phase C
    """
    enhanced_features = features.copy()
    
    # Add Phase C specific features
    if len(draws) > 10:
        # Volatility features
        recent_sums = [sum(draw) for draw in draws[-10:]]
        volatility = np.std(recent_sums) if len(recent_sums) > 1 else 0
        
        # Trend features
        sum_trend = np.polyfit(range(len(recent_sums)), recent_sums, 1)[0] if len(recent_sums) > 2 else 0
        
        # Add these features to each row
        volatility_col = np.full((enhanced_features.shape[0], 1), volatility)
        trend_col = np.full((enhanced_features.shape[0], 1), sum_trend)
        
        enhanced_features = np.hstack([enhanced_features, volatility_col, trend_col])
    
    # Add game-specific optimization features
    if game_type == 'lotto_649':
        bonus_feature = np.full((enhanced_features.shape[0], 1), 1.0)
    else:
        bonus_feature = np.full((enhanced_features.shape[0], 1), 0.0)
    
    enhanced_features = np.hstack([enhanced_features, bonus_feature])
    
    return enhanced_features


def save_model(model, name: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    path = MODEL_DIR / f"{name}.joblib"
    return joblib.load(path)


def extract_attention_from_model(model: Any) -> Any:
    """Best-effort: extract an attention map from a transformer-like model object.

    This is a light helper which looks for common attributes that might expose
    attention weights (e.g., model.attention, model.get_attention(), or
    model.encoder_attentions). Returns None if nothing found.
    """
    # attribute heuristics
    candidates = ['attention', 'attention_map', 'get_attention', 'encoder_attentions']
    for c in candidates:
        if hasattr(model, c):
            try:
                attr = getattr(model, c)
                if callable(attr):
                    return attr()
                return attr
            except Exception:
                continue
    return None


class ModelManager:
    def compare_models(self, models: List[Any]) -> Dict[str, float]:
        """Compare models and return their scores."""
        # TODO: Implement comparison logic
        pass

    def create_ensemble(self, models: List[Any], weights: List[float]) -> Any:
        """Create an ensemble model using weighted-average blending."""
        # TODO: Implement ensemble logic
        pass


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _write_registry(entry: Dict[str, Any]):
    reg_path = Path('model') / 'registry.json'
    try:
        if reg_path.exists():
            regs = json.loads(reg_path.read_text(encoding='utf-8'))
        else:
            regs = []
    except Exception:
        regs = []
    regs.append(entry)
    _ensure_dir(reg_path.parent)
    reg_path.write_text(json.dumps(regs, indent=2), encoding='utf-8')


def prepare_lottery_data_for_training(raw_data: List[Dict], 
                                     model_type: str = 'xgboost',
                                     use_4phase: bool = False,
                                     use_phase_c: bool = False,
                                     game_type: str = 'lotto_649') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare lottery data for different model types with advanced feature engineering.
    Enhanced to support 4-Phase Ultra-High Accuracy features and Phase C optimization.
    
    Args:
        raw_data: List of draw dictionaries with 'numbers', 'draw_date', etc.
        model_type: 'xgboost', 'lstm', or 'transformer'
        use_4phase: Enable 4-Phase Ultra-High Accuracy feature engineering
        use_phase_c: Enable Phase C optimization features
        game_type: Game type ('lotto_649' or 'lotto_max') for 4-phase analysis
    
    Returns:
        X: Feature matrix or sequences
        y: Target values
    """
    # Extract draws as lists of numbers and dates
    draws = []
    draw_dates = []
    
    for entry in raw_data:
        numbers = entry.get('numbers', [])
        if isinstance(numbers, str):
            numbers = [int(x.strip()) for x in numbers.split(',') if x.strip().isdigit()]
        draws.append(numbers)
        
        # Extract dates for 4-phase temporal analysis
        draw_date = entry.get('draw_date')
        if draw_date:
            draw_dates.append(draw_date)
        elif entry.get('date'):
            draw_dates.append(entry.get('date'))
    
    # Determine pool size from game type
    pool_size = 49 if game_type == 'lotto_649' else 50
    
    if model_type == 'xgboost':
        # Use advanced tabular features with optional 4-phase enhancement
        from ai_lottery_bot.features.advanced_features import create_advanced_lottery_features
        
        features = create_advanced_lottery_features(
            draws=draws,
            pool_size=pool_size,
            draw_dates=draw_dates if draw_dates else None,
            game_type=game_type,
            use_4phase=use_4phase
        )
        
        # Apply Phase C feature optimization if enabled
        if use_phase_c and PHASE_C_AVAILABLE:
            print("🚀 Applying Phase C feature optimization...")
            # Additional feature engineering for Phase C optimization
            features = enhance_features_for_phase_c(features, draws, draw_dates, game_type)
        
        # Create targets (predict next draw characteristics)
        X = features[:-1]  # All but last
        y = features[1:]   # All but first (next draw features)
        
        # For regression, predict sum of next draw as a simple target
        y_simple = np.array([sum(draw) for draw in draws[1:]])
        
        optimization_status = ""
        if use_4phase:
            optimization_status += "with 4-phase "
        if use_phase_c:
            optimization_status += "with Phase C "
        if not optimization_status:
            optimization_status = "traditional "
        
        print(f"Enhanced XGBoost features prepared: {X.shape} {optimization_status.strip()}")
        return X, y_simple
    
    elif model_type == 'lstm':
        # LSTM sequence preparation with optional 4-phase enhancement
        from ai_lottery_bot.features.advanced_features import create_sequence_features_for_lstm
        
        X, y = create_sequence_features_for_lstm(
            draws=draws,
            pool_size=pool_size,
            sequence_length=10,
            draw_dates=draw_dates if draw_dates else None,
            game_type=game_type,
            use_4phase=use_4phase
        )
        
        # Apply Phase C optimization if enabled
        if use_phase_c and PHASE_C_AVAILABLE:
            print("🚀 Applying Phase C sequence optimization...")
            # Additional sequence optimization for Phase C
            X = optimize_sequences_for_phase_c(X, draws, game_type)
        
        optimization_status = ""
        if use_4phase:
            optimization_status += "with 4-phase "
        if use_phase_c:
            optimization_status += "with Phase C "
        if not optimization_status:
            optimization_status = "traditional "
        
        print(f"Enhanced LSTM sequences prepared: {X.shape} {optimization_status.strip()}")
        return X, y
    
    elif model_type == 'transformer':
        # Transformer sequence preparation with optional 4-phase enhancement
        from ai_lottery_bot.features.advanced_features import create_sequence_features_for_transformer
        
        X, y = create_sequence_features_for_transformer(
            draws=draws,
            pool_size=pool_size,
            sequence_length=15,
            draw_dates=draw_dates if draw_dates else None,
            game_type=game_type,
            use_4phase=use_4phase
        )
        
        # Apply Phase C optimization if enabled
        if use_phase_c and PHASE_C_AVAILABLE:
            print("🚀 Applying Phase C transformer optimization...")
            # Additional transformer optimization for Phase C
            X = optimize_sequences_for_phase_c(X, draws, game_type)
        
        optimization_status = ""
        if use_4phase:
            optimization_status += "with 4-phase "
        if use_phase_c:
            optimization_status += "with Phase C "
        if not optimization_status:
            optimization_status = "traditional "
        
        print(f"Enhanced Transformer sequences prepared: {X.shape} {optimization_status.strip()}")
        return X, y
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def optimize_sequences_for_phase_c(sequences: np.ndarray, draws: List[List[int]], 
                                 game_type: str) -> np.ndarray:
    """
    Optimize sequences specifically for Phase C
    
    Args:
        sequences: Base sequence data
        draws: Original draw data
        game_type: Game type for optimization
    
    Returns:
        Optimized sequences for Phase C
    """
    # Add sequence-level optimization features
    optimized_sequences = sequences.copy()
    
    # Add temporal smoothing
    for i in range(optimized_sequences.shape[0]):
        for j in range(optimized_sequences.shape[2]):
            # Apply smoothing to each feature channel
            if optimized_sequences.shape[1] > 3:
                smoothed = np.convolve(optimized_sequences[i, :, j], 
                                     np.ones(3)/3, mode='same')
                optimized_sequences[i, :, j] = smoothed
    
    return optimized_sequences


def train_model_with_phase_c(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray = None, y_val: np.ndarray = None,
                           optimization_config: OptimizationConfig = None,
                           model_id: str = None) -> Tuple[Any, Dict]:
    """
    Train a model with Phase C optimization
    
    Args:
        model_type: Type of model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        optimization_config: Configuration for Phase C optimization
        model_id: Unique identifier for the model
    
    Returns:
        Tuple of (trained_model, optimization_results)
    """
    if not PHASE_C_AVAILABLE:
        raise ImportError("Phase C optimization not available")
    
    # Create Phase C optimizer
    phase_c = create_phase_c_optimizer(optimization_config)
    
    # Perform optimization and training
    model, optimization_result = phase_c.optimize_and_train_model(
        model_type, X_train, y_train, X_val, y_val, model_id
    )
    
    # Get comprehensive results
    results = {
        'best_score': optimization_result.best_score,
        'best_params': optimization_result.best_params,
        'training_time': optimization_result.training_time,
        'feature_importance': optimization_result.feature_importance,
        'optimization_history': optimization_result.optimization_history,
        'convergence_info': optimization_result.convergence_info
    }
    
    return model, results


def predict_with_phase_c_enhancement(phase_c_optimizer: PhaseC_IntegratedOptimizer,
                                    model_ids: List[str], X: np.ndarray,
                                    return_detailed: bool = True):
    """
    Make predictions with Phase C enhancement
    
    Args:
        phase_c_optimizer: Phase C optimizer instance
        model_ids: List of model IDs to use
        X: Input features
        return_detailed: Whether to return detailed results
    
    Returns:
        Enhanced predictions with comprehensive metadata
    """
    if not PHASE_C_AVAILABLE:
        raise ImportError("Phase C optimization not available")
    
    return phase_c_optimizer.predict_with_enhancement(model_ids, X, return_detailed)


def train_xgboost(features, labels, name: str = None, version: str = None, save_dir: str = None, hyperparams: dict = None):
    """Train an advanced XGBoost model and save metrics + artifact.

    features: pandas.DataFrame or numpy array
    labels: array-like
    Returns metadata dict describing the trained model.
    """
    import time
    
    # Try to use the advanced XGBoost trainer
    try:
        from ai_lottery_bot.training.advanced_xgboost import train_advanced_xgboost
        import numpy as np
        
        # Convert to numpy arrays if needed
        X = np.array(features) if not isinstance(features, np.ndarray) else features
        y = np.array(labels) if not isinstance(labels, np.ndarray) else labels
        
        # Set up parameters
        ts = int(time.time())
        version = version or f"v{ts}"
        name = name or f"advanced_xgboost_{version}"
        
        # Train using advanced method with comprehensive features
        metadata = train_advanced_xgboost(
            X=X,
            y=y,
            hyperparams=hyperparams,
            optimize_params=False,  # Can be made configurable
            validation_split=0.2,
            cv_folds=5,
            save_dir=save_dir,
            model_name=name,
            version=version
        )
        
        _write_registry(metadata)
        return metadata
        
    except Exception as e:
        import logging
        logging.warning(f"Advanced XGBoost training failed: {e}")
        logging.info("Falling back to basic XGBoost training...")
        
        # Fallback to basic implementation
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

        ts = int(time.time())
        version = version or f"v{ts}"
        name = name or f"xgboost-{version}"
        save_dir = Path(save_dir or MODEL_DIR) / 'xgboost' / version
        _ensure_dir(save_dir)

        # Enhanced hyperparameters with lottery-optimized defaults
        default_hyperparams = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'gamma': 0.1,
            'objective': 'multi:softprob',
            'random_state': 42,
            'n_jobs': -1
        }
        
        if hyperparams:
            default_hyperparams.update(hyperparams)

        # safe import: prefer xgboost, fallback to RandomForestRegressor
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(**default_hyperparams)
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            # Map XGBoost params to RandomForest
            rf_params = {
                'n_estimators': default_hyperparams.get('n_estimators', 200),
                'max_depth': default_hyperparams.get('max_depth', 8),
                'random_state': 42
            }
            model = RandomForestRegressor(**rf_params)

        X = features
        y = labels
        # Enhanced training with validation
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception:
            X_train, X_test, y_train, y_test = X, X, y, y

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Calculate regression metrics
        mse = float(mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mae = float(np.mean(np.abs(y_test - preds)))

        # feature importances if available
        feat_imp = None
        try:
            if hasattr(model, 'feature_importances_'):
                feat_imp = model.feature_importances_.tolist()
        except Exception:
            pass

        # Save model
        model_path = save_dir / f"{name}.joblib"
        joblib.dump(model, model_path)

        # Save comprehensive metadata
        meta = {
            'name': name,
            'file': str(model_path.relative_to(Path.cwd())),
            'type': 'xgboost',
            'version': version,
            'trained_on': str(pd.Timestamp.now()),
            'mse': mse,
            'r2_score': r2,
            'mae': mae,
            'accuracy': max(0, r2),  # Use R² as accuracy proxy
            'feature_importances': feat_imp,
            'hyperparams': default_hyperparams,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        # Save metadata
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        # Save metrics
        metrics = {
            'mse': mse,
            'r2_score': r2,
            'mae': mae,
            'feature_importances': feat_imp
        }
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        _write_registry(meta)
        return meta



def train_lstm(sequences, targets, name: str = None, version: str = None, save_dir: str = None, 
               epochs: int = 100, hyperparams: dict = None):
    """Train an advanced LSTM model for lottery prediction."""
    
    # Use the new advanced LSTM trainer
    try:
        from ai_lottery_bot.training.advanced_lstm import train_advanced_lstm
        
        # Set up hyperparameters
        default_hyperparams = {
            'hidden_units': 128,
            'layers': 2,
            'dropout': 0.2,
            'lr': 0.001,
            'bidirectional': True,
            'attention': True,
            'cell_type': 'LSTM'
        }
        
        if hyperparams:
            default_hyperparams.update(hyperparams)
        
        # Train using advanced method
        metadata = train_advanced_lstm(
            X=sequences,
            y=targets,
            hyperparams=default_hyperparams,
            epochs=epochs,
            save_dir=save_dir,
            version=version
        )
        
        _write_registry(metadata)
        return metadata
        
    except ImportError:
        # Fallback to basic LSTM
        from ai_lottery_bot.training.train_lstm import train_lstm as basic_train_lstm
        
        model = basic_train_lstm(sequences, targets, epochs=epochs)
        
        # Create basic metadata
        ts = int(time.time())
        version = version or f"v{ts}"
        name = name or f"lstm-{version}"
        save_dir = Path(save_dir or MODEL_DIR) / 'lstm' / version
        _ensure_dir(save_dir)
        
        model_path = save_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        
        meta = {
            'name': name,
            'file': str(model_path.relative_to(Path.cwd())),
            'type': 'lstm',
            'version': version,
            'trained_on': str(pd.Timestamp.now()),
            'accuracy': None,
            'hyperparams': hyperparams or {}
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        _write_registry(meta)
        return meta


def train_transformer(sequences, targets, name: str = None, version: str = None, save_dir: str = None,
                     epochs: int = 100, hyperparams: dict = None):
    """Train an advanced transformer model for lottery prediction."""
    
    # Use the new advanced transformer trainer
    try:
        from ai_lottery_bot.training.advanced_transformer import train_advanced_transformer
        
        # Set up hyperparameters
        default_hyperparams = {
            'd_model': 128,
            'heads': 8,
            'layers': 6,
            'dropout': 0.1,
            'lr': 0.0001,
            'ff_dim': 512
        }
        
        if hyperparams:
            default_hyperparams.update(hyperparams)
        
        # Train using advanced method
        metadata = train_advanced_transformer(
            X=sequences,
            y=targets,
            hyperparams=default_hyperparams,
            epochs=epochs,
            save_dir=save_dir,
            version=version
        )
        
        _write_registry(metadata)
        return metadata
        
    except ImportError:
        # Fallback to basic transformer
        from ai_lottery_bot.training.train_transformer import train_transformer as basic_train_transformer
        
        model = basic_train_transformer(sequences, targets, epochs=epochs)
        
        # Create basic metadata
        ts = int(time.time())
        version = version or f"v{ts}"
        name = name or f"transformer-{version}"
        save_dir = Path(save_dir or MODEL_DIR) / 'transformer' / version
        _ensure_dir(save_dir)
        
        model_path = save_dir / f"{name}.joblib"
        joblib.dump(model, model_path)
        
        meta = {
            'name': name,
            'file': str(model_path.relative_to(Path.cwd())),
            'type': 'transformer',
            'version': version,
            'trained_on': str(pd.Timestamp.now()),
            'accuracy': None,
            'hyperparams': hyperparams or {}
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        _write_registry(meta)
        return meta


def train_transformer(sequences, targets, name: str = None, version: str = None, save_dir: str = None,
                     epochs: int = 100, hyperparams: dict = None):
    """Train an advanced transformer model for lottery prediction."""
    
    # Use the new advanced transformer trainer
    try:
        from ai_lottery_bot.training.advanced_transformer import train_advanced_transformer
        
        # Set up hyperparameters
        default_hyperparams = {
            'd_model': 128,
            'heads': 8,
            'layers': 6,
            'dropout': 0.1,
            'lr': 0.0001,
            'ff_dim': 512
        }
        
        if hyperparams:
            default_hyperparams.update(hyperparams)
        
        # Train using advanced method
        metadata = train_advanced_transformer(
            X=sequences,
            y=targets,
            hyperparams=default_hyperparams,
            epochs=epochs,
            save_dir=save_dir,
            version=version
        )
        
        _write_registry(metadata)
        return metadata
        
    except ImportError:
        # Fallback to basic implementation
        import time
        ts = int(time.time())
        version = version or f"v{ts}"
        name = name or f"transformer-{version}"
        save_dir = Path(save_dir or MODEL_DIR) / 'transformer' / version
        _ensure_dir(save_dir)

        model_path = save_dir / f"{name}.joblib"
        
        try:
            # Simple fallback using sklearn
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Flatten sequences for sklearn
            X = sequences.reshape((sequences.shape[0], -1)) if hasattr(sequences, 'shape') else sequences
            model.fit(X, targets)
            joblib.dump(model, model_path)
            
        except Exception:
            # Final fallback: create empty file
            with open(model_path, 'wb') as f:
                pass

        try:
            file_ref = str(model_path.relative_to(Path.cwd()))
        except Exception:
            file_ref = str(model_path.resolve())
            
        meta = {
            'name': name,
            'file': file_ref,
            'type': 'transformer',
            'version': version,
            'trained_on': str(pd.Timestamp.now()),
            'accuracy': None,
            'hyperparams': hyperparams or {}
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        _write_registry(meta)
        return meta
