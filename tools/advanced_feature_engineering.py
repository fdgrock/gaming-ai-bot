"""
Advanced Feature Engineering Pipeline for Lottery Prediction
Implements temporal features, rolling statistics, and auxiliary tasks
Supports both Lotto 649 (6-ball, 49 numbers) and Lotto Max (7-ball, 50 numbers)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    """Configuration for different lottery games"""
    game_name: str
    num_balls: int
    num_numbers: int  # 49 for 649, 50 for Max
    lookback_window: int = 100
    min_draws_for_feature: int = 10


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering pipeline for lottery data
    
    Features generated:
    1. Temporal Features: time_since_last_seen, rolling_frequency (50/100), rolling_mean_interval
    2. Global Draw Features: even_count, sum, std_dev, and rolling statistics
    3. Auxiliary Targets: Skip-Gram co-occurrence, Distribution Forecasting
    """
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.lookback_window = config.lookback_window
        self.min_draws = config.min_draws_for_feature
        
    def generate_temporal_features(
        self, 
        draws: pd.DataFrame, 
        num_draws_back: int = 100
    ) -> pd.DataFrame:
        """
        Generate temporal features for each number in each draw
        
        Args:
            draws: DataFrame with columns ['draw_id', 'date', 'numbers'] (numbers as list or array)
            num_draws_back: How many draws back to look for statistics
            
        Returns:
            DataFrame with engineered temporal features
        """
        n_draws = len(draws)
        n_numbers = self.config.num_numbers
        
        # Initialize feature arrays
        features_data = []
        
        for draw_idx in range(num_draws_back, n_draws):
            current_draw = draws.iloc[draw_idx]
            current_numbers = set(current_draw['numbers'])
            
            # Get historical draws for lookback window
            history_start = max(0, draw_idx - num_draws_back)
            history_draws = draws.iloc[history_start:draw_idx].reset_index(drop=True)
            history_numbers = [set(history_draws.iloc[i]['numbers']) for i in range(len(history_draws))]
            
            # For each number, calculate temporal features
            draw_rows = []
            for num in range(1, n_numbers + 1):
                num_in_current = 1 if num in current_numbers else 0

                # 1. Time since last seen (draws ago)
                time_since_last_seen = None
                for i in range(len(history_numbers) - 1, -1, -1):
                    if num in history_numbers[i]:
                        time_since_last_seen = len(history_numbers) - 1 - i
                        break
                time_since_last_seen = time_since_last_seen if time_since_last_seen is not None else len(history_numbers) + 1

                # 2. Rolling frequencies
                freq_50 = sum(1 for nums in history_numbers[-50:] if num in nums)
                freq_100 = sum(1 for nums in history_numbers[-100:] if num in nums)

                # 3. Rolling mean interval (average gaps between appearances)
                appearances = [i for i, nums in enumerate(history_numbers) if num in nums]
                if len(appearances) > 1:
                    intervals = [appearances[j + 1] - appearances[j] for j in range(len(appearances) - 1)]
                    rolling_mean_interval = float(np.mean(intervals))
                else:
                    rolling_mean_interval = float(len(history_numbers)) if appearances else 0.0

                draw_rows.append({
                    'draw_idx': draw_idx,
                    'number': num,
                    'time_since_last_seen': time_since_last_seen,
                    'rolling_freq_50': freq_50,
                    'rolling_freq_100': freq_100,
                    'rolling_mean_interval': rolling_mean_interval,
                    'target': num_in_current,
                })

            # Draw-level context: z-score normalize key features relative to this draw
            tsl_vals = np.array([r['time_since_last_seen'] for r in draw_rows], dtype=float)
            f50_vals = np.array([r['rolling_freq_50'] for r in draw_rows], dtype=float)
            tsl_mean, tsl_std = tsl_vals.mean(), max(tsl_vals.std(), 1e-8)
            f50_mean, f50_std = f50_vals.mean(), max(f50_vals.std(), 1e-8)
            for row in draw_rows:
                row['rel_time_since_last_seen'] = (row['time_since_last_seen'] - tsl_mean) / tsl_std
                row['rel_rolling_freq_50'] = (row['rolling_freq_50'] - f50_mean) / f50_std
            features_data.extend(draw_rows)
        
        return pd.DataFrame(features_data)
    
    def generate_global_draw_features(self, draws: pd.DataFrame) -> pd.DataFrame:
        """
        Generate global features for each draw (not per-number)
        
        Args:
            draws: DataFrame with columns ['draw_id', 'date', 'numbers']
            
        Returns:
            DataFrame with global draw features
        """
        n_draws = len(draws)
        global_features = []
        
        for draw_idx in range(self.min_draws, n_draws):
            current_numbers = np.array(draws.iloc[draw_idx]['numbers'])
            
            # Current draw statistics
            even_count = int(np.sum(current_numbers % 2 == 0))
            sum_numbers = int(np.sum(current_numbers))
            std_numbers = float(np.std(current_numbers))
            sorted_nums = sorted(current_numbers.tolist())
            n_balls = len(sorted_nums)
            n_numbers = self.config.num_numbers

            # Combination features
            min_sum = sum(range(1, n_balls + 1))
            max_sum = sum(range(n_numbers - n_balls + 1, n_numbers + 1))
            sum_range = max(max_sum - min_sum, 1)
            sum_decile = int(np.clip(int((sum_numbers - min_sum) / sum_range * 10) + 1, 1, 10))

            decade_size = 10
            decade_coverage = len({(n - 1) // decade_size for n in sorted_nums})

            consecutive_count = sum(
                1 for i in range(len(sorted_nums) - 1) if sorted_nums[i + 1] == sorted_nums[i] + 1
            )
            min_max_range = sorted_nums[-1] - sorted_nums[0] if n_balls > 1 else 0

            # Rolling statistics (previous 20 draws)
            window_size = 20
            start_idx = max(0, draw_idx - window_size)

            rolling_sums = []
            rolling_even_counts = []
            for i in range(start_idx, draw_idx):
                prev_numbers = np.array(draws.iloc[i]['numbers'])
                rolling_sums.append(np.sum(prev_numbers))
                rolling_even_counts.append(np.sum(prev_numbers % 2 == 0))

            rolling_mean_sum = float(np.mean(rolling_sums)) if rolling_sums else 0.0
            rolling_mean_even = float(np.mean(rolling_even_counts)) if rolling_even_counts else 0.0
            rolling_std_sum = float(np.std(rolling_sums)) if len(rolling_sums) > 1 else 0.0

            global_features.append({
                'draw_idx': draw_idx,
                'even_count': even_count,
                'sum_numbers': sum_numbers,
                'std_numbers': std_numbers,
                'rolling_mean_sum': rolling_mean_sum,
                'rolling_mean_even': rolling_mean_even,
                'rolling_std_sum': rolling_std_sum,
                'sum_decile': sum_decile,
                'decade_coverage': decade_coverage,
                'consecutive_count': consecutive_count,
                'min_max_range': min_max_range,
            })
        
        return pd.DataFrame(global_features)
    
    def generate_skipgram_targets(
        self, 
        draws: pd.DataFrame,
        mask_ratio: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate Skip-Gram style co-occurrence targets
        Treat draws as sequences and learn number co-occurrence patterns
        
        Args:
            draws: DataFrame with draws
            mask_ratio: Ratio of numbers to mask in a draw
            
        Returns:
            DataFrame with co-occurrence patterns
        """
        skipgram_data = []
        
        for draw_idx, row in draws.iterrows():
            numbers = np.array(row['numbers'])
            n_to_mask = max(1, int(len(numbers) * mask_ratio))
            
            # Randomly select numbers to mask (as context)
            context_idx = np.random.choice(len(numbers), n_to_mask, replace=False)
            context = numbers[context_idx]
            target_idx = [i for i in range(len(numbers)) if i not in context_idx]
            targets = numbers[target_idx]
            
            skipgram_data.append({
                'draw_idx': draw_idx,
                'context_numbers': tuple(sorted(context)),
                'target_numbers': tuple(sorted(targets)),
                'all_numbers': tuple(sorted(numbers))
            })
        
        return pd.DataFrame(skipgram_data)
    
    def generate_distribution_targets(self, draws: pd.DataFrame) -> pd.DataFrame:
        """
        Generate distribution forecasting targets
        Instead of predicting exact numbers, predict probability distribution
        
        Args:
            draws: DataFrame with draws
            
        Returns:
            DataFrame with distribution targets (one-hot encoded)
        """
        n_draws = len(draws)
        n_numbers = self.config.num_numbers
        
        dist_data = []
        
        for draw_idx in range(1, n_draws):
            current_numbers = draws.iloc[draw_idx]['numbers']
            
            # One-hot encode: 1 for numbers in draw, 0 otherwise
            one_hot = np.zeros(n_numbers)
            for num in current_numbers:
                if 1 <= num <= n_numbers:
                    one_hot[num - 1] = 1
            
            dist_data.append({
                'draw_idx': draw_idx,
                'distribution': one_hot,
                'distribution_sum': np.sum(one_hot)
            })
        
        return pd.DataFrame(dist_data)
    
    def create_train_val_test_split(
        self,
        n_samples: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Create temporal integrity preserving train/val/test split
        
        Args:
            n_samples: Total number of samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Tuples: (train_start, train_end), (val_start, val_end), (test_start, test_end)
        """
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        return (
            (0, train_end),
            (train_end, val_end),
            (val_end, n_samples)
        )
    
    def create_sequences(
        self,
        features: pd.DataFrame,
        lookback: int = 100,
        target_col: str = 'target'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D sequences for neural networks
        Shape: [n_sequences, lookback, n_features]
        
        Args:
            features: Engineered features DataFrame
            lookback: Number of draws to look back
            target_col: Column name for target variable
            
        Returns:
            (X sequences, y targets)
        """
        draw_indices = features['draw_idx'].unique()
        draw_indices = sorted(draw_indices)
        
        X_sequences = []
        y_targets = []
        
        for i in range(lookback, len(draw_indices)):
            # Get features for lookback window
            lookback_indices = draw_indices[i-lookback:i]
            lookback_features = features[features['draw_idx'].isin(lookback_indices)]
            
            # Create feature matrix [lookback * n_numbers, n_feature_cols]
            feature_cols = [col for col in features.columns 
                          if col not in ['draw_idx', 'number', target_col]]
            
            seq = lookback_features[feature_cols].values.reshape(lookback, -1)
            X_sequences.append(seq)
            
            # Get targets for next draw
            next_draw_features = features[features['draw_idx'] == draw_indices[i]]
            y_targets.append(next_draw_features[target_col].values)
        
        return np.array(X_sequences), np.array(y_targets)
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate KL-divergence between two probability distributions
        
        Args:
            p: True probability distribution
            q: Predicted probability distribution
            
        Returns:
            KL-divergence value
        """
        # Avoid log(0)
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * (np.log(p) - np.log(q)))
    
    def evaluate_top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Calculate Top-K accuracy
        
        Args:
            y_true: True binary targets [n_samples, n_numbers]
            y_pred_probs: Predicted probabilities [n_samples, n_numbers]
            k: Number of top predictions to consider
            
        Returns:
            Top-K accuracy score
        """
        correct = 0
        for i in range(len(y_true)):
            true_numbers = np.where(y_true[i] == 1)[0]
            top_k_pred = np.argsort(y_pred_probs[i])[-k:]
            
            if len(np.intersect1d(true_numbers, top_k_pred)) > 0:
                correct += 1
        
        return correct / len(y_true)
    
    def save_engineered_dataset(
        self,
        temporal_features: pd.DataFrame,
        global_features: pd.DataFrame,
        skipgram_targets: pd.DataFrame,
        distribution_targets: pd.DataFrame,
        output_dir: Path,
        draws_df: pd.DataFrame = None,
    ) -> None:
        """
        Save all engineered features to disk, including a full feature_schema.json
        that records every column, phase version, and trainer-specific parameters
        needed for inference-time validation.
        """
        import sys as _sys
        import platform as _platform

        try:
            from model_card_utils import (
                derive_feature_cols, build_feature_schema,
                save_feature_schema, SCHEMA_VERSION, collect_package_versions,
            )
            _mcu_available = True
        except ImportError:
            _mcu_available = False

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save features
        temporal_features.to_parquet(output_dir / 'temporal_features.parquet', index=False)
        global_features.to_parquet(output_dir / 'global_features.parquet', index=False)
        skipgram_targets.to_parquet(output_dir / 'skipgram_targets.parquet', index=False)
        distribution_targets.to_parquet(output_dir / 'distribution_targets.parquet', index=False)

        # Derive the actual feature columns each trainer will use
        _non_feat = {"draw_index", "draw_idx", "number", "target"}
        _temp_feat_cols = [c for c in temporal_features.columns if c not in _non_feat]
        _glob_feat_cols = [c for c in global_features.columns
                           if c not in _non_feat and c not in {'draw_idx', 'draw_index'}]

        # Draw index range for traceability
        _draw_col = 'draw_idx' if 'draw_idx' in temporal_features.columns else \
                    'draw_index' if 'draw_index' in temporal_features.columns else None
        _draw_range: dict = {}
        if _draw_col:
            _draw_range = {
                "min": int(temporal_features[_draw_col].min()),
                "max": int(temporal_features[_draw_col].max()),
                "unique_draws": int(temporal_features[_draw_col].nunique()),
            }

        # Build enriched metadata.json
        metadata = {
            'schema_version': SCHEMA_VERSION if _mcu_available else '2.0',
            'phase': 'Phase_A_E_2026',
            'game': self.config.game_name,
            'num_balls': self.config.num_balls,
            'num_numbers': self.config.num_numbers,
            'num_temporal_rows': len(temporal_features),
            'num_global_rows': len(global_features),
            'created_at': datetime.now().isoformat(),
            'python_version': _sys.version,
            'platform': _platform.platform(),
            'draw_index_range': _draw_range,
            'feature_columns': {
                'temporal': temporal_features.columns.tolist(),
                'temporal_feature_cols': _temp_feat_cols,
                'temporal_feature_count': len(_temp_feat_cols),
                'global': global_features.columns.tolist(),
                'global_feature_cols': _glob_feat_cols,
                'global_feature_count': len(_glob_feat_cols),
            },
            'new_features_phase_c': [
                'sum_decile', 'decade_coverage', 'consecutive_count',
                'min_max_range', 'rolling_std_sum',
                'rel_time_since_last_seen', 'rel_rolling_freq_50',
            ],
            'excluded_from_X': sorted(_non_feat),
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Write per-trainer feature_schema.json files that each trainer will
        # copy into its model directory at training time.
        _n = len(temporal_features)
        _n_train = int(0.70 * _n)
        _n_val = int(0.15 * _n)
        _n_test = _n - _n_train - _n_val

        _trainer_configs = [
            # (model_type, window_size, lookback, stride)
            ("tree",        None, None,  None),
            ("lstm",        None, 100,   10),
            ("cnn",         50,   None,  5),
            ("transformer", 20,   None,  5),
            ("lstm_ensemble",      None, 100, 10),
            ("transformer_ensemble", None, None, None),
        ]

        if _mcu_available:
            for _mt, _ws, _lb, _st in _trainer_configs:
                _schema = build_feature_schema(
                    model_type=_mt,
                    game=self.config.game_name,
                    feature_cols=_temp_feat_cols,
                    n_samples=_n,
                    n_train=_n_train,
                    n_val=_n_val,
                    n_test=_n_test,
                    window_size=_ws,
                    lookback=_lb,
                    stride=_st,
                    draw_index_range=_draw_range,
                    extra={
                        'phase': 'Phase_A_E_2026',
                        'global_feature_cols': _glob_feat_cols,
                    },
                )
                save_feature_schema(
                    _schema, output_dir / f'feature_schema_{_mt}.json'
                )
            logger.info(f"Saved feature_schema_*.json for {len(_trainer_configs)} trainer types")

        logger.info(f"Saved engineered dataset to {output_dir}")
    
    @staticmethod
    def load_engineered_dataset(input_dir: Path) -> Dict:
        """
        Load engineered features from disk
        
        Args:
            input_dir: Input directory path
            
        Returns:
            Dictionary with all feature DataFrames and metadata
        """
        input_dir = Path(input_dir)
        
        return {
            'temporal_features': pd.read_parquet(input_dir / 'temporal_features.parquet'),
            'global_features': pd.read_parquet(input_dir / 'global_features.parquet'),
            'skipgram_targets': pd.read_parquet(input_dir / 'skipgram_targets.parquet'),
            'distribution_targets': pd.read_parquet(input_dir / 'distribution_targets.parquet'),
            'metadata': json.load(open(input_dir / 'metadata.json'))
        }


def create_combined_features(
    temporal_features: pd.DataFrame,
    global_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine temporal and global features into unified feature set
    """
    # Merge on draw_idx
    combined = temporal_features.merge(
        global_features,
        on='draw_idx',
        how='left'
    )
    
    # Fill NaN values in rolling statistics
    for col in global_features.columns:
        if col != 'draw_idx':
            combined[col] = combined[col].fillna(combined[col].mean())
    
    return combined
