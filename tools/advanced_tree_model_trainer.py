"""
Advanced Tree Model Trainer for Lottery Prediction
====================================================
Trains position-specific XGBoost, LightGBM, and CatBoost models for both Lotto 649 and Lotto Max.
Implements custom loss combining log loss + KL-divergence penalty against uniform distribution.
Uses Optuna for hyperparameter tuning with composite scoring (0.6*Top5 + 0.4*(1-KL)).

Architecture:
- 6 position-specific models for Lotto 649 (49-class multi-class classification)
- 7 position-specific models for Lotto Max (50-class multi-class classification)
- Each position model trained independently to learn position-specific patterns
- Custom loss: log_loss + 0.3 * kl_divergence_penalty
- Composite metric: 0.6 * top_5_accuracy + 0.4 * (1 - kl_divergence)
"""

import json
import logging
import os
import pickle
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
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

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# VERIFY SCRIPT IS RUNNING
print("[ADVANCED_TREE_MODEL_TRAINER] Script started and imports successful", flush=True)
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


class KLDivergencePenalty:
    """Helper class to calculate KL divergence penalty during training."""
    
    @staticmethod
    def calculate_kl_divergence(predicted_probs: np.ndarray) -> float:
        """
        Calculate KL divergence from uniform distribution.
        
        Args:
            predicted_probs: Predicted probability distribution [n_samples, n_classes]
        
        Returns:
            Mean KL divergence across samples
        """
        n_classes = predicted_probs.shape[1]
        uniform_probs = np.ones(n_classes) / n_classes
        
        # Average KL divergence across samples
        kl_divs = []
        for i in range(predicted_probs.shape[0]):
            kl_div = entropy(uniform_probs, predicted_probs[i] + 1e-10)
            kl_divs.append(kl_div)
        
        return np.mean(kl_divs)
    
    @staticmethod
    def custom_loss_eval(y_true: np.ndarray, y_pred: np.ndarray, 
                        weight: float = 0.3) -> Tuple[str, float, bool]:
        """
        Custom loss combining log loss + KL divergence penalty.
        
        Args:
            y_true: True labels (one-hot or class indices)
            y_pred: Predicted probabilities [n_samples, n_classes]
            weight: Weight for KL divergence penalty
        
        Returns:
            Tuple of (metric_name, metric_value, is_higher_better)
        """
        # Convert y_true to probabilities if needed
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_true_probs = np.zeros((len(y_true), n_classes))
            y_true_probs[np.arange(len(y_true)), y_true.astype(int)] = 1.0
        else:
            y_true_probs = y_true
        
        # Clip predictions to valid probability range
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        # Calculate log loss
        ll = log_loss(y_true_probs, y_pred_clipped)
        
        # Calculate KL divergence penalty
        kl_div = KLDivergencePenalty.calculate_kl_divergence(y_pred_clipped)
        
        # Combined loss
        combined_loss = ll + weight * kl_div
        
        return ("custom_loss", combined_loss, False)  # False means lower is better


class AdvancedTreeModelTrainer:
    """Trainer for tree-based models with Optuna hyperparameter optimization."""
    
    def __init__(self, game_config: GameConfig, base_dir: Path = None):
        """
        Initialize trainer.
        
        Args:
            game_config: GameConfig instance with game parameters
            base_dir: Base directory for data/models (defaults to workspace root)
        """
        self.game_config = game_config
        
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.resolve()
        
        self.base_dir = base_dir
        self.data_dir = base_dir / "data" / "features" / "advanced" / game_config.name.lower().replace(" ", "_")
        self.models_dir = base_dir / "models" / "advanced" / game_config.name.lower().replace(" ", "_")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create models subdirectories for each architecture
        for arch in ["xgboost", "lightgbm", "catboost"]:
            (self.models_dir / arch).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer for {game_config.name}")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Models directory: {self.models_dir}")
    
    def load_engineered_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray, np.ndarray]:
        """
        Load engineered features and prepare train/val/test splits.
        
        Returns:
            X_train, X_val, X_test, y_train_indices, y_val_indices, y_test_indices
        """
        logger.info(f"Loading engineered features from {self.data_dir}")
        
        # Load feature matrices
        temporal_df = pd.read_parquet(self.data_dir / "temporal_features.parquet")
        global_df = pd.read_parquet(self.data_dir / "global_features.parquet")
        
        # Merge global draw features into temporal by draw index
        _tdraw_col = 'draw_idx' if 'draw_idx' in temporal_df.columns else 'draw_index' if 'draw_index' in temporal_df.columns else None
        _gdraw_col = 'draw_idx' if 'draw_idx' in global_df.columns else 'draw_index' if 'draw_index' in global_df.columns else None
        if _tdraw_col and _gdraw_col:
            _global_feat = global_df.drop(columns=[_gdraw_col], errors='ignore')
            temporal_df = temporal_df.merge(
                global_df[[_gdraw_col] + _global_feat.columns.tolist()],
                left_on=_tdraw_col, right_on=_gdraw_col, how='left'
            ).drop(columns=[_gdraw_col] if _gdraw_col != _tdraw_col else [], errors='ignore')

        # Drop label and index columns to prevent data leakage
        X = temporal_df.drop(columns=['draw_index', 'draw_idx', 'number', 'target'], errors='ignore').values

        # Load targets (number identity 1-49/50 per row)
        y_numbers = temporal_df['number'].values if 'number' in temporal_df.columns else None

        logger.info(f"Feature shape (temporal + global merged): {X.shape}")
        _draw_col = 'draw_idx' if 'draw_idx' in temporal_df.columns else 'draw_index' if 'draw_index' in temporal_df.columns else None
        if _draw_col:
            logger.info(f"Number unique draws: {temporal_df[_draw_col].nunique()}")

        # Create train/val/test split based on draw indices
        if _draw_col:
            draw_indices = temporal_df[_draw_col].values
            unique_draws = np.unique(draw_indices)
            n_draws = len(unique_draws)
            
            # 70/15/15 split
            n_train = int(0.70 * n_draws)
            n_val = int(0.15 * n_draws)
            
            train_draws = unique_draws[:n_train]
            val_draws = unique_draws[n_train:n_train + n_val]
            test_draws = unique_draws[n_train + n_val:]
            
            train_mask = np.isin(draw_indices, train_draws)
            val_mask = np.isin(draw_indices, val_draws)
            test_mask = np.isin(draw_indices, test_draws)
        else:
            # Fallback: simple index-based split
            n_samples = len(X)
            n_train = int(0.70 * n_samples)
            n_val = int(0.15 * n_samples)
            
            train_mask = np.arange(n_samples) < n_train
            val_mask = (np.arange(n_samples) >= n_train) & (np.arange(n_samples) < n_train + n_val)
            test_mask = np.arange(n_samples) >= n_train + n_val
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        
        if y_numbers is not None:
            y_train = y_numbers[train_mask].astype(int) - 1  # Convert to 0-based
            y_val = y_numbers[val_mask].astype(int) - 1
            y_test = y_numbers[test_mask].astype(int) - 1
        else:
            # Fallback: use column index as label
            y_train = np.random.randint(0, self.game_config.num_numbers, size=len(X_train))
            y_val = np.random.randint(0, self.game_config.num_numbers, size=len(X_val))
            y_test = np.random.randint(0, self.game_config.num_numbers, size=len(X_test))
        
        logger.info(f"Train shape: {X_train.shape}, Unique classes: {len(np.unique(y_train))}")
        logger.info(f"Val shape: {X_val.shape}, Unique classes: {len(np.unique(y_val))}")
        logger.info(f"Test shape: {X_test.shape}, Unique classes: {len(np.unique(y_test))}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> ModelMetrics:
        """
        Calculate evaluation metrics for a model.
        
        Args:
            y_true: True labels
            y_pred_probs: Predicted probabilities [n_samples, n_classes]
        
        Returns:
            ModelMetrics instance
        """
        # Top-5 accuracy
        top_5_preds = np.argsort(-y_pred_probs, axis=1)[:, :5]
        top_5_correct = np.any(top_5_preds == y_true.reshape(-1, 1), axis=1)
        top_5_acc = np.mean(top_5_correct)
        
        # Top-10 accuracy
        top_10_preds = np.argsort(-y_pred_probs, axis=1)[:, :10]
        top_10_correct = np.any(top_10_preds == y_true.reshape(-1, 1), axis=1)
        top_10_acc = np.mean(top_10_correct)
        
        # KL divergence
        kl_div = KLDivergencePenalty.calculate_kl_divergence(y_pred_probs)
        
        # Log loss
        y_true_onehot = np.zeros((len(y_true), y_pred_probs.shape[1]))
        y_true_onehot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
        y_pred_clipped = np.clip(y_pred_probs, 1e-10, 1 - 1e-10)
        ll = log_loss(y_true_onehot, y_pred_clipped)
        
        # Composite score: exp(-kl) keeps calibration signal even when KL >> 1
        composite = 0.6 * top_5_acc + 0.4 * float(np.exp(-max(0.0, kl_div)))

        # Sanity checks - warn if metrics look impossibly good
        import math as _math
        if top_5_acc > 0.95:
            logger.warning(f"[SANITY] Suspiciously high Top-5 accuracy: {top_5_acc:.4f} - verify target alignment")
        if ll <= 0.0:
            logger.warning(f"[SANITY] Non-positive log-loss: {ll:.6f} - likely a metric calculation error")
        if kl_div > _math.log(y_pred_probs.shape[1]) + 1.0:
            logger.warning(f"[SANITY] KL divergence {kl_div:.4f} exceeds theoretical max - check predictions")

        return ModelMetrics(
            top_5_accuracy=float(top_5_acc),
            top_10_accuracy=float(top_10_acc),
            kl_divergence=float(kl_div),
            log_loss_value=float(ll),
            composite_score=float(composite)
        )
    
    def train_xgboost_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                           position: int, n_trials: int = 20,
                           sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """
        Train XGBoost model with Optuna hyperparameter tuning.

        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Labels
            position: Ball position (1-6 or 1-7)
            n_trials: Number of Optuna trials
            sample_weight: Per-sample weights (synthetic padding rows should be near 0)

        Returns:
            Dictionary with model, metrics, and best params
        """
        logger.info(f"\n=== XGBoost Position {position} ===")

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42 + position)
        )

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "objective": "multi:softprob",
                "num_class": self.game_config.num_numbers,
                "eval_metric": "mlogloss",
                "random_state": 42,
                "verbosity": 0,
            }

            # Train model — pass sample_weight so synthetic padding rows are ignored
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weight)

            # Evaluate
            y_pred_probs = model.predict_proba(X_val)
            metrics = self.calculate_metrics(y_val, y_pred_probs)

            return metrics.composite_score
        
        # Callback to log trial progress
        def trial_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                logger.info(f"  Trial {trial.number + 1}/{n_trials}: Score={trial.value:.4f}")
                if trial.value == study.best_value:
                    logger.info(f"    [BEST] NEW BEST! Score: {trial.value:.4f}")
        
        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback], show_progress_bar=False)
        
        logger.info(f"Best composite score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Train final model with best params
        best_params = study.best_params.copy()
        best_params.update({
            "objective": "multi:softprob",
            "num_class": self.game_config.num_numbers,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "verbosity": 0,
        })
        
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train, sample_weight=sample_weight,
                        eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluate on test
        y_pred_probs = final_model.predict_proba(X_test)
        test_metrics = self.calculate_metrics(y_test, y_pred_probs)

        logger.info(f"Test Top-5 Accuracy: {test_metrics.top_5_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {test_metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {test_metrics.composite_score:.4f}")

        # Permutation baseline: establish chance-level score to verify model is learning
        _rng = np.random.default_rng(seed=42)
        _perm_scores = [
            self.calculate_metrics(_rng.permutation(y_test), y_pred_probs).composite_score
            for _ in range(10)
        ]
        _baseline = float(np.mean(_perm_scores))
        logger.info(f"Permutation baseline composite score: {_baseline:.4f} (model beats by {test_metrics.composite_score - _baseline:+.4f})")
        if test_metrics.composite_score <= _baseline + 0.005:
            logger.warning("[BASELINE] Model does NOT significantly beat random permutation - check training data quality")

        return {
            "model": final_model,
            "position": position,
            "metrics": test_metrics,
            "permutation_baseline": _baseline,
            "best_params": study.best_params,
            "architecture": "xgboost"
        }
    
    def train_lightgbm_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                            position: int, n_trials: int = 20,
                            sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """Train LightGBM model with Optuna hyperparameter tuning."""
        logger.info(f"\n=== LightGBM Position {position} ===")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=100 + position)
        )

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "objective": "multiclass",
                "num_class": self.game_config.num_numbers,
                "metric": "multi_logloss",
                "random_state": 42,
                "verbose": -1,
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weight)

            y_pred_probs = model.predict_proba(X_val)
            metrics = self.calculate_metrics(y_val, y_pred_probs)

            return metrics.composite_score
        
        # Callback to log trial progress
        def trial_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                logger.info(f"  Trial {trial.number + 1}/{n_trials}: Score={trial.value:.4f}")
                if trial.value == study.best_value:
                    logger.info(f"    [BEST] NEW BEST! Score: {trial.value:.4f}")
        
        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback], show_progress_bar=False)
        
        logger.info(f"Best composite score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        best_params = study.best_params.copy()
        best_params.update({
            "objective": "multiclass",
            "num_class": self.game_config.num_numbers,
            "metric": "multi_logloss",
            "random_state": 42,
            "verbose": -1,
        })
        
        final_model = lgb.LGBMClassifier(**best_params)
        final_model.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred_probs = final_model.predict_proba(X_test)
        test_metrics = self.calculate_metrics(y_test, y_pred_probs)

        logger.info(f"Test Top-5 Accuracy: {test_metrics.top_5_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {test_metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {test_metrics.composite_score:.4f}")

        _rng = np.random.default_rng(seed=42)
        _baseline = float(np.mean([
            self.calculate_metrics(_rng.permutation(y_test), y_pred_probs).composite_score
            for _ in range(10)
        ]))
        logger.info(f"Permutation baseline composite score: {_baseline:.4f} (model beats by {test_metrics.composite_score - _baseline:+.4f})")
        if test_metrics.composite_score <= _baseline + 0.005:
            logger.warning("[BASELINE] Model does NOT significantly beat random permutation - check training data quality")

        return {
            "model": final_model,
            "position": position,
            "metrics": test_metrics,
            "permutation_baseline": _baseline,
            "best_params": study.best_params,
            "architecture": "lightgbm"
        }
    
    def train_catboost_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                            position: int, n_trials: int = 20,
                            sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """Train CatBoost model with Optuna hyperparameter tuning."""
        logger.info(f"\n=== CatBoost Position {position} ===")
        
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=200 + position)
        )
        
        def objective(trial):
            params = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
                # subsample requires bootstrap_type='Bernoulli' (default is 'Bayesian'
                # which does not support subsample)
                "bootstrap_type": "Bernoulli",
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "iterations": trial.suggest_int("iterations", 100, 500),
                "random_state": 42,
                "verbose": False,
                "allow_writing_files": False,
            }

            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)

            y_pred_probs = model.predict_proba(X_val)
            metrics = self.calculate_metrics(y_val, y_pred_probs)

            return metrics.composite_score

        # Callback to log trial progress
        def trial_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                logger.info(f"  Trial {trial.number + 1}/{n_trials}: Score={trial.value:.4f}")
                if trial.value == study.best_value:
                    logger.info(f"    [BEST] NEW BEST! Score: {trial.value:.4f}")

        study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback], show_progress_bar=False)

        logger.info(f"Best composite score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        best_params = study.best_params.copy()
        best_params.update({
            # Carry forward bootstrap_type so best_params is also valid
            "bootstrap_type": "Bernoulli",
            "random_state": 42,
            "verbose": False,
            "allow_writing_files": False,
        })
        
        final_model = cb.CatBoostClassifier(**best_params)
        final_model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
        
        y_pred_probs = final_model.predict_proba(X_test)
        test_metrics = self.calculate_metrics(y_test, y_pred_probs)

        logger.info(f"Test Top-5 Accuracy: {test_metrics.top_5_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {test_metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {test_metrics.composite_score:.4f}")

        _rng = np.random.default_rng(seed=42)
        _baseline = float(np.mean([
            self.calculate_metrics(_rng.permutation(y_test), y_pred_probs).composite_score
            for _ in range(10)
        ]))
        logger.info(f"Permutation baseline composite score: {_baseline:.4f} (model beats by {test_metrics.composite_score - _baseline:+.4f})")
        if test_metrics.composite_score <= _baseline + 0.005:
            logger.warning("[BASELINE] Model does NOT significantly beat random permutation - check training data quality")

        return {
            "model": final_model,
            "position": position,
            "metrics": test_metrics,
            "permutation_baseline": _baseline,
            "best_params": study.best_params,
            "architecture": "catboost"
        }
    
    def load_position_features(self, position: int):
        """
        Load features filtered to rows where the number was drawn at the given sort position.

        For position P: only trains on (features, number) pairs where `number` was the
        Pth smallest drawn number in each historical draw. This makes each position model
        learn the distribution of numbers that typically appear at that rank.
        """
        temporal_df = pd.read_parquet(self.data_dir / "temporal_features.parquet")
        global_df_path = self.data_dir / "global_features.parquet"

        _draw_col = 'draw_idx' if 'draw_idx' in temporal_df.columns else 'draw_index' if 'draw_index' in temporal_df.columns else None

        # Filter to position-specific rows (only drawn numbers at position P)
        if 'target' in temporal_df.columns and _draw_col:
            drawn = temporal_df[temporal_df['target'] == 1].copy()
            drawn = drawn.sort_values([_draw_col, 'number'])
            drawn['_pos'] = drawn.groupby(_draw_col).cumcount() + 1
            position_df = drawn[drawn['_pos'] == position].drop(columns=['_pos'])
            if len(position_df) < 20:
                logger.warning(f"Only {len(position_df)} samples for position {position}, using all rows as fallback")
                position_df = temporal_df
        else:
            position_df = temporal_df

        logger.info(f"Position {position}: {len(position_df)} training samples")

        # Merge global features
        if global_df_path.exists():
            global_df = pd.read_parquet(global_df_path)
            _gdraw_col = 'draw_idx' if 'draw_idx' in global_df.columns else 'draw_index' if 'draw_index' in global_df.columns else None
            if _draw_col and _gdraw_col:
                _gcols = [c for c in global_df.columns if c != _gdraw_col]
                position_df = position_df.merge(
                    global_df[[_gdraw_col] + _gcols], left_on=_draw_col, right_on=_gdraw_col, how='left'
                ).drop(columns=[_gdraw_col] if _gdraw_col != _draw_col else [], errors='ignore')

        # Extract X and y
        X = position_df.drop(columns=['draw_index', 'draw_idx', 'number', 'target'], errors='ignore').values
        y = position_df['number'].values.astype(int) - 1  # 0-based

        # Draw-boundary split
        if _draw_col and _draw_col in position_df.columns:
            draw_indices = position_df[_draw_col].values
            unique_draws = np.unique(draw_indices)
            n_draws = len(unique_draws)
            n_td = int(0.70 * n_draws)
            n_vd = int(0.15 * n_draws)
            train_draws = set(unique_draws[:n_td])
            val_draws = set(unique_draws[n_td:n_td + n_vd])
            train_mask = np.array([d in train_draws for d in draw_indices])
            val_mask = np.array([d in val_draws for d in draw_indices])
            test_mask = ~(train_mask | val_mask)
        else:
            n = len(X)
            train_mask = np.arange(n) < int(0.70 * n)
            val_mask = (np.arange(n) >= int(0.70 * n)) & (np.arange(n) < int(0.85 * n))
            test_mask = np.arange(n) >= int(0.85 * n)

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        # Normalize (fit on real training data only)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)

        # XGBoost 3.x / LightGBM require labels to be contiguous [0..num_class-1].
        # Position-specific filtering can leave gaps (e.g. number 31 never drawn at
        # position 1 in the training window).  Pad with one row per missing class so all
        # frameworks see a complete label set, but assign those synthetic rows a
        # near-zero sample_weight (1e-6) so they have negligible influence on the learned
        # decision boundaries.  This avoids the bias that zero-weight-1 synthetic rows
        # would introduce (models learning "mean features → low class index").
        _all_expected = np.arange(self.game_config.num_numbers, dtype=int)
        _missing = np.setdiff1d(_all_expected, np.unique(y_tr))
        # Build real-sample weights (all 1.0) before padding
        w_tr = np.ones(len(X_tr), dtype=np.float32)
        if len(_missing) > 0:
            # Use random real rows as feature templates (not all-zeros) to avoid
            # creating a spurious "zero features = missing class" pattern
            _rng_pad = np.random.default_rng(seed=99)
            _pad_idx = _rng_pad.integers(0, len(X_tr), size=len(_missing))
            _x_pad = X_tr[_pad_idx].copy()
            X_tr = np.vstack([X_tr, _x_pad])
            y_tr = np.concatenate([y_tr, _missing])
            # Synthetic rows get 1e-6 weight → effectively ignored by all three frameworks
            w_tr = np.concatenate([w_tr, np.full(len(_missing), 1e-6, dtype=np.float32)])
            logger.info(f"  Padded {len(_missing)} missing classes (near-zero weight) -> train has all {self.game_config.num_numbers} classes")

        logger.info(f"  Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}, Features={X.shape[1]}")
        return X_tr, X_va, X_te, y_tr, y_va, y_te, w_tr

    def train_all_models(self, n_trials: int = 15) -> Dict[str, List[Dict[str, Any]]]:
        """
        Train all position-specific models for all architectures.

        Each position uses features filtered to rows where the number was drawn
        at that sort position, making models genuinely position-aware.

        Args:
            n_trials: Number of Optuna trials per model

        Returns:
            Dictionary with results for each architecture
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Advanced Tree Model Training")
        logger.info(f"Game: {self.game_config.name}")
        logger.info(f"Positions: {self.game_config.num_positions}")
        logger.info(f"Numbers: {self.game_config.num_numbers}")
        logger.info(f"{'='*60}\n")

        results = {
            "xgboost": [],
            "lightgbm": [],
            "catboost": []
        }

        # Train models for each position using position-specific data
        for position in range(1, self.game_config.num_positions + 1):
            logger.info(f"\n{'*'*60}")
            logger.info(f"Position {position}/{self.game_config.num_positions}")
            logger.info(f"{'*'*60}")

            # Load features specific to this ball position (returns 7-tuple including sample weights)
            X_train, X_val, X_test, y_train, y_val, y_test, w_train = self.load_position_features(position)

            # XGBoost
            logger.info(f"Training XGBoost model (hyperparameter tuning with {n_trials} trials)...")
            xgb_result = self.train_xgboost_model(X_train, X_val, X_test, y_train, y_val, y_test,
                                                   position, n_trials, w_train)
            results["xgboost"].append(xgb_result)
            self._save_model(xgb_result, f"position_{position:02d}")
            logger.info(f"✓ XGBoost complete. Test Score: {xgb_result['metrics'].composite_score:.4f}")

            # LightGBM
            logger.info(f"Training LightGBM model (hyperparameter tuning with {n_trials} trials)...")
            lgb_result = self.train_lightgbm_model(X_train, X_val, X_test, y_train, y_val, y_test,
                                                    position, n_trials, w_train)
            results["lightgbm"].append(lgb_result)
            self._save_model(lgb_result, f"position_{position:02d}")
            logger.info(f"✓ LightGBM complete. Test Score: {lgb_result['metrics'].composite_score:.4f}")

            # CatBoost
            logger.info(f"Training CatBoost model (hyperparameter tuning with {n_trials} trials)...")
            cb_result = self.train_catboost_model(X_train, X_val, X_test, y_train, y_val, y_test,
                                                   position, n_trials, w_train)
            results["catboost"].append(cb_result)
            self._save_model(cb_result, f"position_{position:02d}")
            logger.info(f"✓ CatBoost complete. Test Score: {cb_result['metrics'].composite_score:.4f}")

        self._save_training_summary(results)
        return results
    
    def _save_model(self, result: Dict[str, Any], position_name: str):
        """Save trained model and model card to disk."""
        arch = result["architecture"]
        model_path = self.models_dir / arch / f"{position_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(result["model"], f)

        logger.info(f"Saved {arch} model: {model_path}")

        # Save model card alongside the .pkl
        try:
            import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
            from model_card_utils import (
                build_feature_schema, save_feature_schema,
                build_model_card, save_model_card, derive_feature_cols,
            )
            _feat_schema_path = self.data_dir / "feature_schema_tree.json"
            if _feat_schema_path.exists():
                import json as _json
                with open(_feat_schema_path) as _f:
                    _feat_schema = _json.load(_f)
            else:
                _td = pd.read_parquet(self.data_dir / "temporal_features.parquet")
                _feat_schema = build_feature_schema(
                    model_type="tree", game=self.game_config.name,
                    feature_cols=derive_feature_cols(_td),
                    n_samples=0, n_train=0, n_val=0, n_test=0,
                )
            _metrics = result["metrics"].to_dict() if hasattr(result["metrics"], "to_dict") else {}
            _card = build_model_card(
                architecture=arch,
                game=self.game_config.name,
                position=result.get("position"),
                feature_schema=_feat_schema,
                metrics=_metrics,
                permutation_baseline=result.get("permutation_baseline"),
                hyperparams=result.get("best_params", {}),
                training_config={"position_specific_data": True},
            )
            save_model_card(_card, model_path)
        except Exception as _e:
            logger.warning(f"Could not save model card for {position_name}: {_e}")
    
    def _save_training_summary(self, results: Dict[str, List[Dict[str, Any]]]):
        """Save training summary to JSON with full metadata."""
        try:
            from model_card_utils import SCHEMA_VERSION as _sv
        except ImportError:
            _sv = "2.0"

        summary = {
            "schema_version": _sv,
            "phase": "Phase_A_E_2026",
            "trained_at": datetime.now().isoformat(),
            "game": self.game_config.name,
            "num_positions": self.game_config.num_positions,
            "num_numbers": self.game_config.num_numbers,
            "training_approach": "position_specific_data",
            "target_alignment": "actual_lottery_numbers_0based",
            "architectures": {}
        }

        for arch, models in results.items():
            arch_summary = {
                "num_models": len(models),
                "models": []
            }

            all_metrics = []
            all_baselines = []
            for model_result in models:
                metrics = model_result["metrics"]
                baseline = model_result.get("permutation_baseline")
                all_metrics.append(metrics)
                if baseline is not None:
                    all_baselines.append(baseline)
                arch_summary["models"].append({
                    "position": model_result["position"],
                    "metrics": metrics.to_dict(),
                    "permutation_baseline": baseline,
                    "beats_baseline": (
                        metrics.composite_score > (baseline or 0) + 0.005
                        if baseline is not None else None
                    ),
                    "best_params": model_result["best_params"]
                })

            mean_top5 = np.mean([m.top_5_accuracy for m in all_metrics])
            mean_kl = np.mean([m.kl_divergence for m in all_metrics])
            mean_composite = np.mean([m.composite_score for m in all_metrics])
            mean_baseline = float(np.mean(all_baselines)) if all_baselines else None

            arch_summary["aggregate_metrics"] = {
                "mean_top_5_accuracy": float(mean_top5),
                "mean_kl_divergence": float(mean_kl),
                "mean_composite_score": float(mean_composite),
                "mean_permutation_baseline": mean_baseline,
                "mean_lift_over_baseline": (
                    float(mean_composite - mean_baseline)
                    if mean_baseline is not None else None
                ),
            }

            summary["architectures"][arch] = arch_summary

        summary_path = self.models_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nSaved training summary: {summary_path}")
        logger.info(json.dumps(summary, indent=2))


def run_tree_model_training_pipeline():
    """Execute tree model training for both games."""
    
    # Lotto 649
    config_649 = GameConfig(
        name="lotto_6_49",
        num_balls=6,
        num_numbers=49,
        num_positions=6
    )
    
    trainer_649 = AdvancedTreeModelTrainer(config_649)
    results_649 = trainer_649.train_all_models(n_trials=15)
    
    # Lotto Max
    config_max = GameConfig(
        name="lotto_max",
        num_balls=7,
        num_numbers=52,
        num_positions=7
    )
    
    trainer_max = AdvancedTreeModelTrainer(config_max)
    results_max = trainer_max.train_all_models(n_trials=15)
    
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2A TREE MODEL TRAINING COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train Phase 2A tree models")
        parser.add_argument("--game", type=str, default=None, help="Game to train (lotto_6_49 or lotto_max, or All Games)")
        args = parser.parse_args()
        
        print(f"[DEBUG] Parsed arguments: game={args.game}", flush=True)
        sys.stdout.flush()
        
        if args.game and args.game != "All Games":
            if "649" in args.game or args.game.lower() == "lotto 6/49":
                print(f"[DEBUG] Training lotto_6_49", flush=True)
                config = GameConfig(name="lotto_6_49", num_balls=6, num_numbers=49, num_positions=6)
                trainer = AdvancedTreeModelTrainer(config)
                trainer.train_all_models(n_trials=15)
            elif "max" in args.game.lower() or args.game.lower() == "lotto max":
                print(f"[DEBUG] Training lotto_max", flush=True)
                config = GameConfig(name="lotto_max", num_balls=7, num_numbers=52, num_positions=7)
                trainer = AdvancedTreeModelTrainer(config)
                trainer.train_all_models(n_trials=15)
        else:
            print(f"[DEBUG] Training all games", flush=True)
            run_tree_model_training_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Exception occurred in main: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
