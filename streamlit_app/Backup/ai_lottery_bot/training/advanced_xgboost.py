# ai_lottery_bot/training/advanced_xgboost.py

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_advanced_xgboost_features(draws: List[List[int]], 
                                   include_statistical: bool = True,
                                   include_frequency: bool = True,
                                   include_patterns: bool = True,
                                   lookback_windows: List[int] = [5, 10, 20]) -> np.ndarray:
    """
    Create advanced feature matrix for XGBoost training.
    
    Args:
        draws: List of lottery draws (each draw is a list of numbers)
        include_statistical: Include statistical features
        include_frequency: Include frequency-based features
        include_patterns: Include pattern-based features
        lookback_windows: Different window sizes for rolling features
    
    Returns:
        Feature matrix with shape (n_samples, n_features)
    """
    if len(draws) < max(lookback_windows) + 1:
        raise ValueError(f"Need at least {max(lookback_windows) + 1} draws for feature creation")
    
    # Use the advanced feature engineering module
    try:
        from ai_lottery_bot.features.advanced_features import create_advanced_lottery_features
        features = create_advanced_lottery_features(
            draws,
            lookback_windows=lookback_windows,
            include_statistical=include_statistical,
            include_frequency=include_frequency,
            include_patterns=include_patterns
        )
        return features
    except ImportError:
        logger.warning("Advanced features module not available, using basic features")
        return _create_basic_features(draws, lookback_windows)


def _create_basic_features(draws: List[List[int]], lookback_windows: List[int]) -> np.ndarray:
    """Fallback basic feature creation if advanced module is not available."""
    features = []
    
    for i in range(max(lookback_windows), len(draws)):
        draw_features = []
        current_draw = draws[i]
        
        # Basic draw statistics
        draw_features.extend([
            sum(current_draw),
            np.mean(current_draw),
            np.std(current_draw),
            max(current_draw) - min(current_draw),
            len([x for x in current_draw if x % 2 == 0]),  # even count
            len([x for x in current_draw if x % 2 == 1]),  # odd count
        ])
        
        # Historical features for different windows
        for window in lookback_windows:
            if i >= window:
                recent_draws = draws[i-window:i]
                all_numbers = [num for draw in recent_draws for num in draw]
                
                # Frequency features
                unique_nums = set(all_numbers)
                draw_features.extend([
                    len(unique_nums),  # unique numbers in window
                    len(all_numbers) / len(unique_nums) if unique_nums else 0,  # repetition rate
                    np.mean([sum(draw) for draw in recent_draws]),  # avg sum
                    np.std([sum(draw) for draw in recent_draws]),   # sum std
                ])
        
        features.append(draw_features)
    
    return np.array(features)


def prepare_xgboost_targets(draws: List[List[int]], 
                          target_type: str = 'next_sum',
                          classification: bool = False) -> np.ndarray:
    """
    Prepare target variables for XGBoost training.
    
    Args:
        draws: List of lottery draws
        target_type: Type of target ('next_sum', 'next_numbers', 'next_patterns')
        classification: Whether to treat as classification problem
    
    Returns:
        Target array
    """
    if target_type == 'next_sum':
        # Predict the sum of the next draw
        targets = [sum(draws[i+1]) for i in range(len(draws)-1)]
        
        if classification:
            # Convert to bins for classification
            targets = np.array(targets)
            percentiles = np.percentile(targets, [20, 40, 60, 80])
            binned_targets = np.digitize(targets, percentiles)
            return binned_targets
        else:
            return np.array(targets)
    
    elif target_type == 'next_numbers':
        # Predict individual numbers (multi-target regression)
        max_numbers = max(len(draw) for draw in draws)
        targets = []
        for i in range(len(draws)-1):
            next_draw = draws[i+1]
            # Pad with zeros if needed
            padded = next_draw + [0] * (max_numbers - len(next_draw))
            targets.append(padded[:max_numbers])
        return np.array(targets)
    
    elif target_type == 'next_patterns':
        # Predict pattern characteristics
        targets = []
        for i in range(len(draws)-1):
            next_draw = draws[i+1]
            pattern_features = [
                sum(next_draw),
                len([x for x in next_draw if x % 2 == 0]),  # even count
                max(next_draw) - min(next_draw),            # range
                len(set(next_draw)),                        # unique count
            ]
            targets.append(pattern_features)
        return np.array(targets)
    
    else:
        raise ValueError(f"Unknown target type: {target_type}")


def build_advanced_xgboost_model(hyperparams: Dict[str, Any]) -> Any:
    """
    Build an advanced XGBoost model with optimized hyperparameters.
    
    Args:
        hyperparams: Dictionary of hyperparameters
    
    Returns:
        Configured XGBoost model
    """
    # Default advanced hyperparameters
    default_params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,      # L1 regularization
        'reg_lambda': 1.0,     # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',  # Faster training
        'enable_categorical': True
    }
    
    # Update with provided hyperparameters
    default_params.update(hyperparams)
    
    # Try to import XGBoost, fallback to alternatives
    try:
        import xgboost as xgb
        
        # Determine if classification or regression
        objective = default_params.get('objective', 'reg:squarederror')
        
        if 'multi:softmax' in objective or 'binary:logistic' in objective:
            model = xgb.XGBClassifier(**{k: v for k, v in default_params.items() 
                                       if k not in ['eval_metric', 'early_stopping_rounds', 'optimize_hyperparams', 'use_cv', 'include_statistical', 'include_frequency', 'include_patterns', 'auto_tune']})
        else:
            model = xgb.XGBRegressor(**{k: v for k, v in default_params.items() 
                                      if k not in ['eval_metric', 'early_stopping_rounds', 'optimize_hyperparams', 'use_cv', 'include_statistical', 'include_frequency', 'include_patterns', 'auto_tune']})
        
        logger.info(f"Created XGBoost model with parameters: {dict(default_params)}")
        return model
        
    except ImportError:
        logger.warning("XGBoost not available, trying LightGBM")
        
        try:
            import lightgbm as lgb
            
            # Convert XGBoost params to LightGBM params
            lgb_params = {
                'n_estimators': default_params.get('n_estimators', 500),
                'max_depth': default_params.get('max_depth', 8),
                'learning_rate': default_params.get('learning_rate', 0.1),
                'subsample': default_params.get('subsample', 0.8),
                'colsample_bytree': default_params.get('colsample_bytree', 0.8),
                'min_child_weight': default_params.get('min_child_weight', 3),
                'reg_alpha': default_params.get('reg_alpha', 0.1),
                'reg_lambda': default_params.get('reg_lambda', 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**lgb_params)
            logger.info(f"Created LightGBM model with parameters: {lgb_params}")
            return model
            
        except ImportError:
            logger.warning("LightGBM not available, falling back to Random Forest")
            
            from sklearn.ensemble import RandomForestRegressor
            
            # Convert to Random Forest params
            rf_params = {
                'n_estimators': min(default_params.get('n_estimators', 500), 200),
                'max_depth': default_params.get('max_depth', 8),
                'min_samples_split': max(2, default_params.get('min_child_weight', 3)),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**rf_params)
            logger.info(f"Created Random Forest model with parameters: {rf_params}")
            return model


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5,
                           n_iter: int = 50,
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using randomized search.
    
    Args:
        X: Feature matrix
        y: Target values
        cv_folds: Number of CV folds
        n_iter: Number of iterations for random search
        random_state: Random seed
    
    Returns:
        Best hyperparameters found
    """
    logger.info("Starting hyperparameter optimization...")
    
    # Define parameter search space
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0, 0.05, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0]
    }
    
    try:
        import xgboost as xgb
        base_model = xgb.XGBRegressor(
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist'
        )
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        # Adjust param space for Random Forest
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 8, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Perform randomized search
    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X, y)
    
    logger.info(f"Best parameters found: {search.best_params_}")
    logger.info(f"Best CV score: {-search.best_score_:.4f}")
    
    return search.best_params_


def train_advanced_xgboost(X: np.ndarray, y: np.ndarray,
                         hyperparams: Optional[Dict[str, Any]] = None,
                         optimize_params: bool = False,
                         validation_split: float = 0.2,
                         cv_folds: int = 5,
                         save_dir: Optional[str] = None,
                         model_name: Optional[str] = None,
                         version: Optional[str] = None) -> Dict[str, Any]:
    """
    Train an advanced XGBoost model with comprehensive evaluation.
    
    Args:
        X: Feature matrix
        y: Target values
        hyperparams: Model hyperparameters
        optimize_params: Whether to optimize hyperparameters
        validation_split: Fraction for validation set
        cv_folds: Number of cross-validation folds
        save_dir: Directory to save model
        model_name: Name for the model
        version: Version string
    
    Returns:
        Dictionary with training metadata and results
    """
    logger.info("Starting advanced XGBoost training...")
    
    # Set up directories and naming
    timestamp = int(time.time())
    version = version or f"v{timestamp}"
    model_name = model_name or f"advanced_xgboost_{version}"
    save_dir = Path(save_dir or "models") / "xgboost" / version
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Validation set size: {X_val.shape[0]}")
    logger.info(f"Feature dimensions: {X_train.shape[1]}")
    
    # Optimize hyperparameters if requested
    if optimize_params:
        logger.info("Optimizing hyperparameters...")
        best_params = optimize_hyperparameters(X_train, y_train, cv_folds=cv_folds)
        if hyperparams:
            best_params.update(hyperparams)
        hyperparams = best_params
    
    # Build model
    model = build_advanced_xgboost_model(hyperparams or {})
    
    # Train model with validation
    try:
        if hasattr(model, 'fit') and 'XGB' in str(type(model)):
            # XGBoost with early stopping
            if X_val is not None and len(X_val) > 0:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                # No validation set, train without early stopping
                model.fit(X_train, y_train, verbose=False)
        else:
            # Standard scikit-learn interface
            model.fit(X_train, y_train)
    except Exception as e:
        logger.warning(f"Error in advanced training: {e}")
        model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_preds)
    val_mse = mean_squared_error(y_val, val_preds)
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    train_mae = mean_absolute_error(y_train, train_preds)
    val_mae = mean_absolute_error(y_val, val_preds)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    cv_rmse_mean = np.sqrt(-cv_scores.mean())
    cv_rmse_std = np.sqrt(cv_scores.std())
    
    # Feature importance
    feature_importance = None
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_.tolist()
    except Exception:
        pass
    
    # Model complexity metrics
    n_estimators_used = getattr(model, 'n_estimators', 'N/A')
    if hasattr(model, 'best_iteration'):
        n_estimators_used = model.best_iteration
    
    # Save model
    model_path = save_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    
    # Create comprehensive metadata
    metadata = {
        'name': model_name,
        'version': version,
        'type': 'xgboost_advanced',
        'file': str(model_path),
        'trained_on': str(pd.Timestamp.now()),
        
        # Performance metrics
        'train_mse': float(train_mse),
        'val_mse': float(val_mse),
        'train_r2': float(train_r2),
        'val_r2': float(val_r2),
        'train_mae': float(train_mae),
        'val_mae': float(val_mae),
        'cv_rmse_mean': float(cv_rmse_mean),
        'cv_rmse_std': float(cv_rmse_std),
        
        # Calculate lottery-appropriate accuracy from validation MAE
        # For lottery prediction, convert MAE to percentage-based accuracy
        'accuracy': float(max(0.0, min(0.4, 1.0 - (val_mae / 50.0)))),  # Proper lottery accuracy
        
        # Model information
        'hyperparams': hyperparams or {},
        'n_estimators_used': n_estimators_used,
        'feature_importance': feature_importance,
        'training_samples': int(X_train.shape[0]),
        'validation_samples': int(X_val.shape[0]),
        'n_features': int(X_train.shape[1]),
        
        # Additional metadata
        'note': 'Realistic lottery prediction metrics - accuracy based on MAE normalization',
        
        # Training configuration
        'validation_split': validation_split,
        'cv_folds': cv_folds,
        'hyperparameter_optimization': optimize_params
    }
    
    # Save metadata
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save detailed metrics
    detailed_metrics = {
        'training_metrics': {
            'mse': float(train_mse),
            'r2': float(train_r2),
            'mae': float(train_mae),
            'rmse': float(np.sqrt(train_mse))
        },
        'validation_metrics': {
            'mse': float(val_mse),
            'r2': float(val_r2),
            'mae': float(val_mae),
            'rmse': float(np.sqrt(val_mse))
        },
        'cross_validation': {
            'rmse_mean': float(cv_rmse_mean),
            'rmse_std': float(cv_rmse_std),
            'scores': cv_scores.tolist()
        },
        'feature_importance': feature_importance,
        'predictions_sample': {
            'train_actual': y_train[:10].tolist(),
            'train_predicted': train_preds[:10].tolist(),
            'val_actual': y_val[:10].tolist(),
            'val_predicted': val_preds[:10].tolist()
        }
    }
    
    with open(save_dir / 'detailed_metrics.json', 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Validation R²: {val_r2:.4f}")
    logger.info(f"Validation RMSE: {np.sqrt(val_mse):.4f}")
    logger.info(f"CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
    logger.info(f"Model saved to: {model_path}")
    
    return metadata


def predict_with_advanced_xgboost(model, features: np.ndarray, 
                                 return_probabilities: bool = False) -> np.ndarray:
    """
    Make predictions using the trained XGBoost model.
    
    Args:
        model: Trained model
        features: Feature matrix for prediction
        return_probabilities: Whether to return probabilities (for classification)
    
    Returns:
        Predictions array
    """
    try:
        if return_probabilities and hasattr(model, 'predict_proba'):
            return model.predict_proba(features)
        else:
            return model.predict(features)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return default predictions
        return np.zeros(features.shape[0])


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Generate sample lottery draws
    sample_draws = []
    for i in range(100):
        draw = sorted(np.random.choice(49, 6, replace=False) + 1)
        sample_draws.append(draw)
    
    print("Testing advanced XGBoost training...")
    
    # Create features
    X = create_advanced_xgboost_features(sample_draws)
    y = prepare_xgboost_targets(sample_draws[:-len(X):], target_type='next_sum')
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Train model
    hyperparams = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
    
    metadata = train_advanced_xgboost(
        X, y,
        hyperparams=hyperparams,
        save_dir="test_models/xgboost"
    )
    
    print("Training metadata:")
    for key, value in metadata.items():
        if key not in ['feature_importance']:  # Skip long arrays
            print(f"  {key}: {value}")
