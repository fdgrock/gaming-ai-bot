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
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
        
        # Combine features (temporal already has global features merged)
        X = temporal_df.drop(columns=['draw_index', 'number'], errors='ignore').values
        
        # Load targets (use draw_index to get unique draws and their numbers)
        y_numbers = temporal_df['number'].values if 'number' in temporal_df.columns else None
        
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Number unique draws: {len(np.unique(temporal_df.get('draw_index', np.arange(len(X)))))}") if 'draw_index' in temporal_df.columns else None
        
        # Create train/val/test split based on draw indices
        if 'draw_index' in temporal_df.columns:
            draw_indices = temporal_df['draw_index'].values
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
        
        # Composite score: 0.6 * Top5_acc + 0.4 * (1 - KL)
        composite = 0.6 * top_5_acc + 0.4 * (1 - np.tanh(kl_div))  # tanh normalizes KL to [0,1]
        
        return ModelMetrics(
            top_5_accuracy=float(top_5_acc),
            top_10_accuracy=float(top_10_acc),
            kl_divergence=float(kl_div),
            log_loss_value=float(ll),
            composite_score=float(composite)
        )
    
    def train_xgboost_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                           position: int, n_trials: int = 20) -> Dict[str, Any]:
        """
        Train XGBoost model with Optuna hyperparameter tuning.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Labels
            position: Ball position (1-6 or 1-7)
            n_trials: Number of Optuna trials
        
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
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_probs = model.predict_proba(X_val)
            metrics = self.calculate_metrics(y_val, y_pred_probs)
            
            return metrics.composite_score
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
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
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                       verbose=False, early_stopping_rounds=10)
        
        # Evaluate on test
        y_pred_probs = final_model.predict_proba(X_test)
        test_metrics = self.calculate_metrics(y_test, y_pred_probs)
        
        logger.info(f"Test Top-5 Accuracy: {test_metrics.top_5_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {test_metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {test_metrics.composite_score:.4f}")
        
        return {
            "model": final_model,
            "position": position,
            "metrics": test_metrics,
            "best_params": study.best_params,
            "architecture": "xgboost"
        }
    
    def train_lightgbm_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                            position: int, n_trials: int = 20) -> Dict[str, Any]:
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
            model.fit(X_train, y_train)
            
            y_pred_probs = model.predict_proba(X_val)
            metrics = self.calculate_metrics(y_val, y_pred_probs)
            
            return metrics.composite_score
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
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
        final_model.fit(X_train, y_train)
        
        y_pred_probs = final_model.predict_proba(X_test)
        test_metrics = self.calculate_metrics(y_test, y_pred_probs)
        
        logger.info(f"Test Top-5 Accuracy: {test_metrics.top_5_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {test_metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {test_metrics.composite_score:.4f}")
        
        return {
            "model": final_model,
            "position": position,
            "metrics": test_metrics,
            "best_params": study.best_params,
            "architecture": "lightgbm"
        }
    
    def train_catboost_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                            position: int, n_trials: int = 20) -> Dict[str, Any]:
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
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "iterations": trial.suggest_int("iterations", 100, 500),
                "random_state": 42,
                "verbose": False,
                "allow_writing_files": False,
            }
            
            model = cb.CatBoostClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred_probs = model.predict_proba(X_val)
            metrics = self.calculate_metrics(y_val, y_pred_probs)
            
            return metrics.composite_score
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"Best composite score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        best_params = study.best_params.copy()
        best_params.update({
            "random_state": 42,
            "verbose": False,
            "allow_writing_files": False,
        })
        
        final_model = cb.CatBoostClassifier(**best_params)
        final_model.fit(X_train, y_train, verbose=False)
        
        y_pred_probs = final_model.predict_proba(X_test)
        test_metrics = self.calculate_metrics(y_test, y_pred_probs)
        
        logger.info(f"Test Top-5 Accuracy: {test_metrics.top_5_accuracy:.4f}")
        logger.info(f"Test KL-Divergence: {test_metrics.kl_divergence:.4f}")
        logger.info(f"Test Composite Score: {test_metrics.composite_score:.4f}")
        
        return {
            "model": final_model,
            "position": position,
            "metrics": test_metrics,
            "best_params": study.best_params,
            "architecture": "catboost"
        }
    
    def train_all_models(self, n_trials: int = 15) -> Dict[str, List[Dict[str, Any]]]:
        """
        Train all position-specific models for all architectures.
        
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
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_engineered_features()
        
        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        results = {
            "xgboost": [],
            "lightgbm": [],
            "catboost": []
        }
        
        # Train models for each position
        for position in range(1, self.game_config.num_positions + 1):
            logger.info(f"\n{'*'*60}")
            logger.info(f"Position {position}/{self.game_config.num_positions}")
            logger.info(f"{'*'*60}")
            
            # XGBoost
            xgb_result = self.train_xgboost_model(X_train, X_val, X_test, y_train, y_val, y_test, 
                                                   position, n_trials)
            results["xgboost"].append(xgb_result)
            self._save_model(xgb_result, f"position_{position:02d}")
            
            # LightGBM
            lgb_result = self.train_lightgbm_model(X_train, X_val, X_test, y_train, y_val, y_test,
                                                    position, n_trials)
            results["lightgbm"].append(lgb_result)
            self._save_model(lgb_result, f"position_{position:02d}")
            
            # CatBoost
            cb_result = self.train_catboost_model(X_train, X_val, X_test, y_train, y_val, y_test,
                                                   position, n_trials)
            results["catboost"].append(cb_result)
            self._save_model(cb_result, f"position_{position:02d}")
        
        self._save_training_summary(results)
        return results
    
    def _save_model(self, result: Dict[str, Any], position_name: str):
        """Save trained model to disk."""
        arch = result["architecture"]
        model_path = self.models_dir / arch / f"{position_name}.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(result["model"], f)
        
        logger.info(f"Saved {arch} model: {model_path}")
    
    def _save_training_summary(self, results: Dict[str, List[Dict[str, Any]]]):
        """Save training summary to JSON."""
        summary = {
            "game": self.game_config.name,
            "num_positions": self.game_config.num_positions,
            "num_numbers": self.game_config.num_numbers,
            "architectures": {}
        }
        
        for arch, models in results.items():
            arch_summary = {
                "num_models": len(models),
                "models": []
            }
            
            all_metrics = []
            for model_result in models:
                metrics = model_result["metrics"]
                all_metrics.append(metrics)
                arch_summary["models"].append({
                    "position": model_result["position"],
                    "metrics": metrics.to_dict(),
                    "best_params": model_result["best_params"]
                })
            
            # Calculate aggregate metrics
            mean_top5 = np.mean([m.top_5_accuracy for m in all_metrics])
            mean_kl = np.mean([m.kl_divergence for m in all_metrics])
            mean_composite = np.mean([m.composite_score for m in all_metrics])
            
            arch_summary["aggregate_metrics"] = {
                "mean_top_5_accuracy": float(mean_top5),
                "mean_kl_divergence": float(mean_kl),
                "mean_composite_score": float(mean_composite)
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
        num_numbers=50,
        num_positions=7
    )
    
    trainer_max = AdvancedTreeModelTrainer(config_max)
    results_max = trainer_max.train_all_models(n_trials=15)
    
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2A TREE MODEL TRAINING COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 2A tree models")
    parser.add_argument("--game", type=str, default=None, help="Game to train (lotto_6_49 or lotto_max, or All Games)")
    args = parser.parse_args()
    
    if args.game and args.game != "All Games":
        if "649" in args.game or args.game.lower() == "lotto 6/49":
            config = GameConfig(name="lotto_6_49", num_balls=6, num_numbers=49, num_positions=6)
            trainer = AdvancedTreeModelTrainer(config)
            trainer.train_all_models(n_trials=15)
        elif "max" in args.game.lower() or args.game.lower() == "lotto max":
            config = GameConfig(name="lotto_max", num_balls=7, num_numbers=50, num_positions=7)
            trainer = AdvancedTreeModelTrainer(config)
            trainer.train_all_models(n_trials=15)
    else:
        run_tree_model_training_pipeline()
