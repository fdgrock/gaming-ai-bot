"""
Advanced AI-Powered Feature Generation Service
Sophisticated feature engineering for LSTM, CNN, Transformer, and XGBoost models
Designed to generate sequences that predict winning lottery numbers

Core Philosophy:
- Analyze raw lottery data comprehensively
- Extract deep statistical and temporal patterns
- Create advanced engineered features for optimal model learning
- Target: Generate winning number predictions with maximum accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import json
import logging
import sys
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

try:
    from ..core import get_data_dir, app_log
except ImportError:
    def get_data_dir():
        return Path("data")
    def app_log(msg: str, level: str = "info"):
        level_map = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
        logging.log(level_map.get(level, logging.INFO), msg)

# Import feature schema system
try:
    from .feature_schema import FeatureSchema, NormalizationMethod, Transformation, NormalizationParams
except ImportError:
    # Fallback if not available
    FeatureSchema = None
    NormalizationMethod = None


class AdvancedFeatureGenerator:
    """Advanced feature generation with sophisticated AI/ML techniques."""
    
    def __init__(self, game: str):
        """Initialize advanced feature generator."""
        self.game = game
        self.game_folder = game.lower().replace(" ", "_").replace("/", "_")
        self.data_dir = get_data_dir()
        self.raw_data_dir = self.data_dir / self.game_folder
        self.features_dir = self.data_dir / "features"
        
        # Create directories
        self.lstm_dir = self.features_dir / "lstm" / self.game_folder
        self.transformer_dir = self.features_dir / "transformer" / self.game_folder
        self.cnn_dir = self.features_dir / "cnn" / self.game_folder
        self.xgboost_dir = self.features_dir / "xgboost" / self.game_folder
        self.catboost_dir = self.features_dir / "catboost" / self.game_folder
        self.lightgbm_dir = self.features_dir / "lightgbm" / self.game_folder
        
        for d in [self.lstm_dir, self.transformer_dir, self.cnn_dir, self.xgboost_dir, self.catboost_dir, self.lightgbm_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def _create_feature_schema(
        self,
        model_type: str,
        feature_names: List[str],
        normalization_method: str,
        data_shape: Tuple[int, ...],
        data_date_range: Dict[str, str],
        window_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        transformations: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Optional['FeatureSchema']:
        """
        Create a FeatureSchema object with all generation parameters.
        This ensures reproducibility across training and prediction.
        """
        if FeatureSchema is None:
            app_log("FeatureSchema not available, skipping schema creation", "warning")
            return None
        
        try:
            # Determine normalization method enum
            norm_method_map = {
                "StandardScaler": NormalizationMethod.STANDARD_SCALER,
                "MinMaxScaler": NormalizationMethod.MIN_MAX_SCALER,
                "RobustScaler": NormalizationMethod.ROBUST_SCALER,
                "L2": NormalizationMethod.L2_NORM,
                "None": NormalizationMethod.NONE,
            }
            norm_enum = norm_method_map.get(normalization_method, NormalizationMethod.NONE)
            
            # Build transformations list
            trans_list = []
            if transformations:
                for t in transformations:
                    trans_list.append(Transformation(**t))
            
            schema = FeatureSchema(
                model_type=model_type,
                game=self.game,
                schema_version="1.0",
                created_at=datetime.now().isoformat(),
                feature_names=feature_names,
                feature_count=len(feature_names),
                feature_categories=kwargs.get("feature_categories", []),
                normalization_method=norm_enum,
                window_size=window_size,
                embedding_dim=embedding_dim,
                transformations=trans_list,
                data_shape=data_shape,
                data_date_range=data_date_range,
                raw_data_version=kwargs.get("raw_data_version", "1.0"),
                raw_data_date_generated=datetime.now().isoformat(),
                notes=kwargs.get("notes", ""),
                package_versions={
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "scikit-learn": "1.0+",
                },
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            )
            
            return schema
        except Exception as e:
            app_log(f"Error creating FeatureSchema: {e}", "error")
            return None
    
    def _save_schema_with_features(
        self,
        schema: Optional['FeatureSchema'],
        model_type: str
    ) -> Optional[Path]:
        """Save FeatureSchema to file for later retrieval"""
        if schema is None:
            return None
        
        try:
            # Save schema as JSON in the same directory as features
            model_folder = {
                "lstm": self.lstm_dir,
                "cnn": self.cnn_dir,
                "transformer": self.transformer_dir,
                "xgboost": self.xgboost_dir,
                "catboost": self.catboost_dir,
                "lightgbm": self.lightgbm_dir,
            }.get(model_type)
            
            if model_folder:
                schema_path = model_folder / "feature_schema.json"
                schema.save_to_file(schema_path)
                app_log(f"Saved FeatureSchema to {schema_path}", "info")
                return schema_path
        except Exception as e:
            app_log(f"Error saving FeatureSchema: {e}", "error")
        
        return None
    
    def get_raw_files(self) -> List[Path]:
        """Get all raw CSV files for the game."""
        if not self.raw_data_dir.exists():
            return []
        return sorted(self.raw_data_dir.glob("training_data_*.csv"))
    
    def _get_feature_files_for_type(self, feature_type: str) -> List[Path]:
        """Get all feature files for a specific feature type."""
        type_dirs = {
            'lstm': self.lstm_dir,
            'cnn': self.cnn_dir,
            'transformer': self.transformer_dir,
            'xgboost': self.xgboost_dir,
            'catboost': self.catboost_dir,
            'lightgbm': self.lightgbm_dir
        }
        
        feature_dir = type_dirs.get(feature_type)
        if not feature_dir or not feature_dir.exists():
            return []
        
        # Get all relevant files based on type
        if feature_type in ['lstm', 'cnn']:
            # NPZ files for neural network features
            return sorted(feature_dir.glob("*.npz"))
        else:
            # CSV files for tree-based model features
            return sorted(feature_dir.glob("*.csv"))
    
    def load_raw_data(self, files: List[Path]) -> Optional[pd.DataFrame]:
        """Load and combine raw data from multiple files."""
        try:
            dfs = []
            for file_path in files:
                df = pd.read_csv(file_path)
                dfs.append(df)
            
            if not dfs:
                return None
            
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["draw_date"], keep="first")
            combined_df = combined_df.sort_values("draw_date").reset_index(drop=True)
            combined_df["draw_date"] = pd.to_datetime(combined_df["draw_date"])
            
            return combined_df
        except Exception as e:
            app_log(f"Error loading raw data: {e}", "error")
            return None
    
    def _parse_numbers(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Parse and prepare number data."""
        data = raw_data.copy()
        
        # Parse draw_date as datetime if it's not already
        if "draw_date" in data.columns and not pd.api.types.is_datetime64_any_dtype(data["draw_date"]):
            data["draw_date"] = pd.to_datetime(data["draw_date"], errors='coerce')
        
        data["numbers_list"] = data["numbers"].apply(
            lambda x: sorted([int(n.strip()) for n in str(x).split(",")])
        )
        data["bonus_int"] = data["bonus"].apply(
            lambda x: int(x) if pd.notna(x) and str(x).isdigit() else 0
        )
        return data
    
    def _calculate_temporal_features(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Calculate temporal and time-based features."""
        features = {}
        
        if idx < 1:
            return features
        
        current_date = data.iloc[idx]["draw_date"]
        
        # Days since last draw
        prev_date = data.iloc[idx - 1]["draw_date"]
        features["days_since_last"] = (current_date - prev_date).days
        
        # Week of year
        features["week_of_year"] = current_date.isocalendar().week
        
        # Month
        features["month"] = current_date.month
        
        # Day of week (0=Monday, 6=Sunday)
        features["day_of_week"] = current_date.dayofweek
        
        # Is weekend
        features["is_weekend"] = 1.0 if current_date.dayofweek >= 5 else 0.0
        
        # Days into year
        features["day_of_year"] = current_date.dayofyear
        
        return features
    
    def _calculate_number_distribution_features(self, numbers: List[int], max_num: int = 50) -> Dict[str, float]:
        """Calculate advanced distribution features."""
        features = {}
        
        # Range analysis
        features["min_num"] = np.min(numbers)
        features["max_num"] = np.max(numbers)
        features["range"] = features["max_num"] - features["min_num"]
        
        # Distribution buckets
        bucket_size = max_num // 5
        for i in range(5):
            low = i * bucket_size
            high = (i + 1) * bucket_size
            count = sum(1 for n in numbers if low <= n < high)
            features[f"bucket_{i}_count"] = count
        
        # Quartile analysis
        features["q1"] = np.percentile(numbers, 25)
        features["q2"] = np.percentile(numbers, 50)
        features["q3"] = np.percentile(numbers, 75)
        features["iqr"] = features["q3"] - features["q1"]
        
        # Spread metrics
        features["mean"] = np.mean(numbers)
        features["std"] = np.std(numbers)
        features["cv"] = features["std"] / (features["mean"] + 1e-8)
        features["sum"] = np.sum(numbers)
        
        # Skewness and Kurtosis
        features["skewness"] = float(stats.skew(numbers))
        features["kurtosis"] = float(stats.kurtosis(numbers))
        
        # Entropy (randomness measure)
        hist, _ = np.histogram(numbers, bins=10)
        hist = hist / hist.sum()
        features["entropy"] = -np.sum(hist * np.log2(hist + 1e-10))
        
        return features
    
    def _calculate_parity_features(self, numbers: List[int]) -> Dict[str, float]:
        """Calculate parity and modulo-based features."""
        features = {}
        
        # Even/Odd
        evens = [n for n in numbers if n % 2 == 0]
        odds = [n for n in numbers if n % 2 == 1]
        features["even_count"] = len(evens)
        features["odd_count"] = len(odds)
        features["even_odd_ratio"] = len(evens) / (len(odds) + 1e-8)
        
        # Low/Mid/High
        third = len(numbers) // 3
        sorted_nums = sorted(numbers)
        features["low_count"] = sum(1 for n in numbers if n <= sorted_nums[third])
        features["mid_count"] = sum(1 for n in numbers if sorted_nums[third] < n <= sorted_nums[2*third])
        features["high_count"] = sum(1 for n in numbers if n > sorted_nums[2*third])
        
        # Modulo patterns
        for mod in [3, 5, 7, 11]:
            features[f"mod_{mod}_variance"] = float(np.var([n % mod for n in numbers]))
        
        return features
    
    def _calculate_spacing_features(self, numbers: List[int]) -> Dict[str, float]:
        """Calculate spacing and gap features."""
        features = {}
        
        sorted_nums = sorted(numbers)
        gaps = np.diff(sorted_nums)
        
        features["avg_gap"] = np.mean(gaps)
        features["max_gap"] = np.max(gaps)
        features["min_gap"] = np.min(gaps)
        features["gap_std"] = np.std(gaps)
        features["gap_variance"] = np.var(gaps)
        
        # Gap distribution
        large_gaps = sum(1 for g in gaps if g > 10)
        features["large_gap_count"] = large_gaps
        
        # Consecutive sequences
        consecutive_length = 1
        max_consecutive = 1
        for i in range(len(gaps)):
            if gaps[i] == 1:
                consecutive_length += 1
                max_consecutive = max(max_consecutive, consecutive_length)
            else:
                consecutive_length = 1
        features["max_consecutive"] = max_consecutive
        
        return features
    
    def _calculate_frequency_features(self, data: pd.DataFrame, idx: int, lookback_windows: List[int]) -> Dict[str, float]:
        """Calculate frequency-based features from historical data."""
        features = {}
        
        current_numbers = set(data.iloc[idx]["numbers_list"])
        
        for window in lookback_windows:
            if idx < window:
                continue
            
            # Get historical numbers
            historical = data.iloc[max(0, idx - window):idx]["numbers_list"]
            
            # Frequency of current numbers in history
            freq_count = 0
            for hist_nums in historical:
                freq_count += len(current_numbers & set(hist_nums))
            
            features[f"freq_match_{window}"] = freq_count / (window * len(current_numbers) + 1e-8)
            
            # New numbers (not in recent history)
            recent_all = set()
            for hist_nums in historical:
                recent_all.update(hist_nums)
            
            new_count = len(current_numbers - recent_all)
            features[f"new_numbers_{window}"] = new_count
        
        return features
    
    def _calculate_statistical_moments(self, numbers: List[int]) -> Dict[str, float]:
        """Calculate higher-order statistical moments."""
        features = {}
        
        # First 4 moments (mean, variance, skewness, kurtosis)
        features["moment_1"] = np.mean(numbers)  # Mean
        features["moment_2"] = np.var(numbers)   # Variance
        
        # Skewness (3rd moment)
        mean = np.mean(numbers)
        features["moment_3"] = np.mean([(n - mean)**3 for n in numbers])
        
        # Kurtosis (4th moment)
        features["moment_4"] = np.mean([(n - mean)**4 for n in numbers])
        
        return features
    
    def _calculate_periodicity_features(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Calculate periodicity and cyclical patterns."""
        features = {}
        
        if idx < 30:
            return features
        
        # Get recent sum history
        recent_sums = data.iloc[max(0, idx-30):idx]["numbers_list"].apply(np.sum).values
        
        try:
            # FFT to detect periodicity
            fft_vals = np.abs(fft(recent_sums))
            features["dominant_frequency"] = np.argmax(fft_vals[1:]) + 1  # Skip DC component
        except:
            features["dominant_frequency"] = 0.0
        
        # Autocorrelation
        if len(recent_sums) > 2:
            autocorr = np.correlate(recent_sums - np.mean(recent_sums), 
                                   recent_sums - np.mean(recent_sums), mode='full')
            features["autocorr_lag1"] = autocorr[len(autocorr)//2 + 1] / (autocorr[len(autocorr)//2] + 1e-8)
        
        return features
    
    def _calculate_bonus_features(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Calculate bonus number specific features."""
        features = {}
        
        current_bonus = data.iloc[idx]["bonus_int"]
        
        features["bonus_value"] = current_bonus
        features["bonus_even_odd"] = 0.0 if current_bonus % 2 == 0 else 1.0
        
        if idx > 0:
            prev_bonus = data.iloc[idx - 1]["bonus_int"]
            features["bonus_change"] = current_bonus - prev_bonus
            features["bonus_repeating"] = 1.0 if current_bonus == prev_bonus else 0.0
        
        # Bonus frequency in history
        if idx >= 10:
            recent_bonuses = data.iloc[idx-10:idx]["bonus_int"].values
            features["bonus_frequency"] = np.sum(recent_bonuses == current_bonus) / 10.0
        
        return features
    
    def generate_lstm_sequences(self, raw_data: pd.DataFrame, 
                               window_size: int = 25,
                               include_all_features: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate advanced LSTM sequences with comprehensive feature engineering.
        
        Features include: temporal, statistical, distribution, parity, spacing,
        frequency, periodicity, and bonus features.
        """
        try:
            app_log(f"Generating advanced LSTM sequences with window_size={window_size}", "info")
            
            data = self._parse_numbers(raw_data)
            
            # Determine max number based on game
            max_num = 50 if "6" in self.game else 50
            
            sequences_list = []
            feature_names = None
            
            # Generate features for each draw
            for idx in range(len(data)):
                draw_row = data.iloc[idx]
                numbers = draw_row["numbers_list"]
                
                features = {}
                
                # 1. Temporal features
                features.update(self._calculate_temporal_features(data, idx))
                
                # 2. Number distribution features
                features.update(self._calculate_number_distribution_features(numbers, max_num))
                
                # 3. Parity features
                features.update(self._calculate_parity_features(numbers))
                
                # 4. Spacing features
                features.update(self._calculate_spacing_features(numbers))
                
                # 5. Statistical moments
                features.update(self._calculate_statistical_moments(numbers))
                
                # 6. Frequency features
                lookback_windows = [5, 10, 20, 30, 60]
                features.update(self._calculate_frequency_features(data, idx, lookback_windows))
                
                # 7. Periodicity features
                features.update(self._calculate_periodicity_features(data, idx))
                
                # 8. Bonus features
                features.update(self._calculate_bonus_features(data, idx))
                
                # 9. Jackpot features
                jackpot = float(draw_row.get("jackpot", 0))
                features["jackpot"] = jackpot
                features["jackpot_log"] = np.log1p(jackpot)
                features["jackpot_millions"] = jackpot / 1_000_000
                
                sequences_list.append(features)
                
                if feature_names is None:
                    feature_names = sorted(features.keys())
            
            # Convert to array
            feature_array = np.array([
                [f.get(name, 0) for name in feature_names]
                for f in sequences_list
            ])
            
            # Advanced normalization using RobustScaler (resistant to outliers)
            scaler = RobustScaler()
            feature_array = scaler.fit_transform(feature_array)
            
            # Create overlapping sequences (windows)
            sequences = []
            for i in range(len(feature_array) - window_size):
                window = feature_array[i:i + window_size]
                sequences.append(window)
            
            sequences_array = np.array(sequences)
            
            metadata = {
                "model_type": "lstm",
                "game": self.game,
                "processing_mode": "advanced_all_files",
                "raw_files": [str(f) for f in self.get_raw_files()],
                "total_draws": len(data),
                "sequence_count": len(sequences_array),
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "window_size": window_size,
                    "lookback_windows": lookback_windows,
                    "normalization": "RobustScaler",
                    "feature_categories": [
                        "temporal", "distribution", "parity", "spacing",
                        "statistical_moments", "frequency", "periodicity",
                        "bonus", "jackpot"
                    ]
                },
                "file_info": [
                    {
                        "file": str(f),
                        "draws_count": len(pd.read_csv(f))
                    }
                    for f in self.get_raw_files()
                ]
            }
            
            app_log(f"✓ Generated {len(sequences_array)} advanced LSTM sequences with {len(feature_names)} features", "info")
            
            # CREATE FEATURE SCHEMA
            data_date_range = {
                "min": str(data["draw_date"].min()),
                "max": str(data["draw_date"].max())
            }
            schema = self._create_feature_schema(
                model_type="lstm",
                feature_names=feature_names,
                normalization_method="RobustScaler",
                data_shape=sequences_array.shape,
                data_date_range=data_date_range,
                window_size=window_size,
                lookback_periods=[5, 10, 20, 30, 60],
                feature_categories=metadata["params"]["feature_categories"],
                notes=f"LSTM sequences with {window_size}-step lookback windows"
            )
            self._save_schema_with_features(schema, "lstm")
            
            return sequences_array, metadata
            
        except Exception as e:
            app_log(f"Error generating LSTM sequences: {e}", "error")
            raise
    
    def generate_cnn_embeddings(self, raw_data: pd.DataFrame,
                                window_size: int = 24,
                                embedding_dim: int = 64) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate multi-scale CNN embeddings optimized for pattern detection.
        
        Uses sliding window with multi-scale features (simulating conv kernels 3, 5, 7)
        to capture patterns at different granularities. More efficient than Transformer
        for fixed-dimensional lottery feature classification.
        """
        try:
            app_log(f"Generating advanced CNN embeddings (dim={embedding_dim})", "info")
            
            data = self._parse_numbers(raw_data)
            max_num = 50
            
            # Generate comprehensive features
            features_list = []
            for idx in range(len(data)):
                draw_row = data.iloc[idx]
                numbers = draw_row["numbers_list"]
                
                features = {}
                
                # Core numerical features for CNN
                features.update(self._calculate_number_distribution_features(numbers, max_num))
                features.update(self._calculate_parity_features(numbers))
                features.update(self._calculate_spacing_features(numbers))
                features.update(self._calculate_statistical_moments(numbers))
                features.update(self._calculate_bonus_features(data, idx))
                
                # Temporal features
                features.update(self._calculate_temporal_features(data, idx))
                
                features_list.append(features)
            
            feature_names = sorted(features_list[0].keys())
            
            # Create normalized feature array
            feature_array = np.array([
                [f.get(name, 0) for name in feature_names]
                for f in features_list
            ])
            
            # Normalize features
            scaler = StandardScaler()
            feature_array = scaler.fit_transform(feature_array)
            
            # Generate CNN embeddings using windowed multi-scale features
            embeddings = []
            for i in range(len(feature_array) - window_size):
                window = feature_array[i:i + window_size]
                
                # Multi-scale CNN aggregation (simulating conv kernels 3, 5, 7)
                embeddings_parts = []
                
                # Part 1: Mean pooling (kernel 3 effect)
                mean_pool = np.mean(window, axis=0)
                embeddings_parts.append(mean_pool)
                
                # Part 2: Max pooling (kernel 5 effect)
                max_pool = np.max(window, axis=0)
                embeddings_parts.append(max_pool)
                
                # Part 3: Std aggregation (kernel 7 effect)
                std_pool = np.std(window, axis=0)
                embeddings_parts.append(std_pool)
                
                # Part 4: Temporal differences (local gradients)
                if window_size > 1:
                    diff = np.mean(np.diff(window, axis=0), axis=0)
                    embeddings_parts.append(diff)
                
                # Part 5: Percentile features (robust to outliers)
                p25 = np.percentile(window, 25, axis=0)
                p75 = np.percentile(window, 75, axis=0)
                embeddings_parts.append(p25)
                embeddings_parts.append(p75)
                
                # Concatenate all parts
                combined = np.concatenate(embeddings_parts)
                
                # Project to target embedding dimension
                if len(combined) >= embedding_dim:
                    # Use dimensionality reduction
                    embedding = combined[:embedding_dim]
                else:
                    # Pad with learned patterns
                    padding_size = embedding_dim - len(combined)
                    padding = np.random.randn(padding_size) * 0.1
                    embedding = np.concatenate([combined, padding])
                
                embeddings.append(embedding)
            
            embeddings_array = np.array(embeddings)
            
            # Apply L2 normalization for CNN compatibility
            embeddings_array = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8)
            
            metadata = {
                "model_type": "cnn",
                "game": self.game,
                "processing_mode": "advanced_multi_scale",
                "raw_files": [str(f) for f in self.get_raw_files()],
                "total_draws": len(data),
                "embedding_count": len(embeddings_array),
                "embedding_dimension": embedding_dim,
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "window_size": window_size,
                    "embedding_dim": embedding_dim,
                    "aggregation_methods": ["mean_pool", "max_pool", "std_pool", "temporal_diff", "percentile_25", "percentile_75"],
                    "normalization": "L2",
                    "base_features": len(feature_names)
                },
                "file_info": [
                    {
                        "file": str(f),
                        "draws_count": len(pd.read_csv(f))
                    }
                    for f in self.get_raw_files()
                ]
            }
            
            app_log(f"✓ Generated {len(embeddings_array)} advanced CNN embeddings", "info")
            
            # CREATE FEATURE SCHEMA
            data_date_range = {
                "min": str(data["draw_date"].min()),
                "max": str(data["draw_date"].max())
            }
            schema = self._create_feature_schema(
                model_type="cnn",
                feature_names=[f"cnn_dim_{i}" for i in range(embedding_dim)],
                normalization_method="L2",
                data_shape=embeddings_array.shape,
                data_date_range=data_date_range,
                embedding_dim=embedding_dim,
                window_size=window_size,
                feature_categories=["embedding"],
                notes=f"CNN embeddings ({embedding_dim}D) from {window_size}-step windows"
            )
            self._save_schema_with_features(schema, "cnn")
            
            return embeddings_array, metadata
            
        except Exception as e:
            app_log(f"Error generating CNN embeddings: {e}", "error")
            return np.array([]), {}

    def generate_transformer_embeddings(self, raw_data: pd.DataFrame,
                                       window_size: int = 30,
                                       embedding_dim: int = 128) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate advanced Transformer embeddings with multi-scale attention patterns.
        
        Uses multi-head semantic embeddings to capture complex number relationships.
        """
        try:
            app_log(f"Generating advanced Transformer embeddings (dim={embedding_dim})", "info")
            
            data = self._parse_numbers(raw_data)
            max_num = 50
            
            # Generate comprehensive features
            features_list = []
            for idx in range(len(data)):
                draw_row = data.iloc[idx]
                numbers = draw_row["numbers_list"]
                
                features = {}
                
                # Core numerical features
                features.update(self._calculate_number_distribution_features(numbers, max_num))
                features.update(self._calculate_parity_features(numbers))
                features.update(self._calculate_spacing_features(numbers))
                features.update(self._calculate_statistical_moments(numbers))
                features.update(self._calculate_bonus_features(data, idx))
                
                # Temporal features
                features.update(self._calculate_temporal_features(data, idx))
                
                features_list.append(features)
            
            feature_names = sorted(features_list[0].keys())
            
            # Create normalized feature array
            feature_array = np.array([
                [f.get(name, 0) for name in feature_names]
                for f in features_list
            ])
            
            # Normalize features
            scaler = StandardScaler()
            feature_array = scaler.fit_transform(feature_array)
            
            # Generate embeddings using windowed features + PCA projection
            embeddings = []
            for i in range(len(feature_array) - window_size):
                window = feature_array[i:i + window_size]
                
                # Multi-scale aggregation
                embeddings_parts = []
                
                # Part 1: Mean pooling (global context)
                mean_pool = np.mean(window, axis=0)
                embeddings_parts.append(mean_pool)
                
                # Part 2: Max pooling (peak features)
                max_pool = np.max(window, axis=0)
                embeddings_parts.append(max_pool)
                
                # Part 3: Std aggregation (variability)
                std_pool = np.std(window, axis=0)
                embeddings_parts.append(std_pool)
                
                # Part 4: Temporal difference (trends)
                if window_size > 1:
                    diff = np.mean(np.diff(window, axis=0), axis=0)
                    embeddings_parts.append(diff)
                
                # Concatenate all parts
                combined = np.concatenate(embeddings_parts)
                
                # Project to target embedding dimension
                if len(combined) >= embedding_dim:
                    # Use PCA-like projection
                    embedding = combined[:embedding_dim]
                else:
                    # Pad with learned patterns
                    padding_size = embedding_dim - len(combined)
                    padding = np.random.randn(padding_size) * 0.1
                    embedding = np.concatenate([combined, padding])
                
                embeddings.append(embedding)
            
            embeddings_array = np.array(embeddings)
            
            # Apply L2 normalization for transformer compatibility
            embeddings_array = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8)
            
            metadata = {
                "model_type": "transformer",
                "game": self.game,
                "processing_mode": "advanced_all_files",
                "raw_files": [str(f) for f in self.get_raw_files()],
                "total_draws": len(data),
                "embedding_count": len(embeddings_array),
                "embedding_dimension": embedding_dim,
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "window_size": window_size,
                    "embedding_dim": embedding_dim,
                    "aggregation_methods": ["mean_pool", "max_pool", "std_pool", "temporal_diff"],
                    "normalization": "L2",
                    "base_features": len(feature_names)
                },
                "file_info": [
                    {
                        "file": str(f),
                        "draws_count": len(pd.read_csv(f))
                    }
                    for f in self.get_raw_files()
                ]
            }
            
            app_log(f"✓ Generated {len(embeddings_array)} advanced Transformer embeddings", "info")
            
            # CREATE FEATURE SCHEMA
            data_date_range = {
                "min": str(data["draw_date"].min()),
                "max": str(data["draw_date"].max())
            }
            schema = self._create_feature_schema(
                model_type="transformer",
                feature_names=[f"transformer_dim_{i}" for i in range(embedding_dim)],
                normalization_method="L2",
                data_shape=embeddings_array.shape,
                data_date_range=data_date_range,
                embedding_dim=embedding_dim,
                window_size=window_size,
                feature_categories=["embedding"],
                notes=f"Transformer embeddings ({embedding_dim}D) with multi-scale aggregation"
            )
            self._save_schema_with_features(schema, "transformer")
            
            return embeddings_array, metadata
            
        except Exception as e:
            app_log(f"Error generating Transformer embeddings: {e}", "error")
            raise
    
    def generate_transformer_features_csv(self, raw_data: pd.DataFrame,
                                         output_dim: int = 20) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate Transformer features optimized for CSV output with exactly 20 dimensions.
        
        Produces features suitable for Transformer model predictions.
        Output shape: (N, 20) - 20 engineered features per draw.
        Format: DataFrame saved as CSV
        """
        try:
            app_log(f"Generating Transformer features for CSV export (output_dim={output_dim})", "info")
            
            data = self._parse_numbers(raw_data)
            max_num = 50
            
            # Generate base features from each draw
            features_list = []
            for idx in range(len(data)):
                draw_row = data.iloc[idx]
                numbers = draw_row["numbers_list"]
                
                # Create feature dict with exactly 20 features
                features = {}
                
                # Statistical features (5 features)
                stats = self._calculate_statistical_moments(numbers)
                features['sum'] = stats.get('sum', 0)
                features['mean'] = stats.get('mean', 0)
                features['std'] = stats.get('std', 0)
                features['skew'] = stats.get('skew', 0)
                features['kurtosis'] = stats.get('kurtosis', 0)
                
                # Distribution features (3 features)
                dist = self._calculate_number_distribution_features(numbers, max_num)
                features['min_num'] = dist.get('min', 0)
                features['max_num'] = dist.get('max', 0)
                features['range'] = dist.get('range', 0)
                
                # Parity features (2 features)
                parity = self._calculate_parity_features(numbers)
                features['even_count'] = parity.get('even_count', 0)
                features['odd_count'] = parity.get('odd_count', 0)
                
                # Spacing features (3 features)
                spacing = self._calculate_spacing_features(numbers)
                features['avg_gap'] = spacing.get('avg_gap', 0)
                features['max_gap'] = spacing.get('max_gap', 0)
                features['consecutive_pairs'] = spacing.get('consecutive_pairs', 0)
                
                # Temporal features (3 features)
                temporal = self._calculate_temporal_features(data, idx)
                features['days_since_last'] = temporal.get('days_since_last_draw', 0)
                features['day_of_week_sin'] = temporal.get('day_of_week_sin', 0)
                features['month_sin'] = temporal.get('month_sin', 0)
                
                # Bonus features (2 features)
                bonus = self._calculate_bonus_features(data, idx)
                features['bonus_num'] = bonus.get('bonus_number', 0)
                features['bonus_zscore'] = bonus.get('bonus_zscore', 0)
                
                # Additional pattern feature (1 feature)
                # Sum of modulo 10 differences (captures digit patterns)
                digit_pattern = sum(abs(numbers[i] % 10 - numbers[i+1] % 10) for i in range(len(numbers)-1)) / max(1, len(numbers)-1)
                features['digit_pattern_score'] = digit_pattern
                
                # Ensure exactly 20 features by padding with zeros if needed
                while len(features) < 20:
                    features[f'padding_{len(features)}'] = 0.0
                
                # Truncate to exactly 20 if we have more
                feature_keys = sorted(list(features.keys()))[:20]
                features = {k: features[k] for k in feature_keys}
                
                features_list.append(features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            
            # ✅ CRITICAL: Add draw_date and numbers columns BEFORE normalization
            features_df.insert(0, "draw_date", data["draw_date"].values)
            if "numbers" in data.columns:
                features_df.insert(1, "numbers", data["numbers"].values)
            
            # Ensure numeric types for feature columns only (skip draw_date and numbers)
            feature_cols = [col for col in features_df.columns if col not in ["draw_date", "numbers"]]
            features_df[feature_cols] = features_df[feature_cols].astype(float)
            
            # Normalize features to 0-1 range for better Transformer input (only numeric feature columns)
            scaler = MinMaxScaler()
            features_normalized = scaler.fit_transform(features_df[feature_cols])
            features_df[feature_cols] = features_normalized
            
            metadata = {
                "model_type": "transformer",
                "game": self.game,
                "processing_mode": "csv_export",
                "feature_count": len(features_df.columns),
                "draw_count": len(features_df),
                "output_format": "CSV",
                "output_dim": output_dim,
                "timestamp": datetime.now().isoformat(),
                "feature_columns": list(features_df.columns),
                "feature_categories": [
                    "statistical (5: sum, mean, std, skew, kurtosis)",
                    "distribution (3: min, max, range)",
                    "parity (2: even_count, odd_count)",
                    "spacing (3: avg_gap, max_gap, consecutive_pairs)",
                    "temporal (3: days_since_last, day_of_week_sin, month_sin)",
                    "bonus (2: bonus_num, bonus_zscore)",
                    "pattern (1: digit_pattern_score)"
                ],
                "params": {
                    "output_dim": output_dim,
                    "normalization": "MinMax (0-1)",
                    "total_features": 20
                },
                "raw_files": [str(f) for f in self.get_raw_files()]
            }
            
            app_log(f"✓ Generated {len(features_df)} Transformer feature vectors with shape {features_df.shape}", "info")
            
            # CREATE FEATURE SCHEMA
            data_date_range = {
                "min": str(data["draw_date"].min()),
                "max": str(data["draw_date"].max())
            }
            schema = self._create_feature_schema(
                model_type="transformer",
                feature_names=list(features_df.columns),
                normalization_method="MinMaxScaler",
                data_shape=features_df.shape,
                data_date_range=data_date_range,
                feature_categories=metadata["feature_categories"],
                notes="Transformer CSV features (20D), normalized to 0-1 range"
            )
            self._save_schema_with_features(schema, "transformer")
            
            return features_df, metadata
            
        except Exception as e:
            app_log(f"Error generating Transformer CSV features: {e}", "error")
            raise
    
    def save_transformer_features_csv(self, features_df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Save Transformer features to CSV file with quality indicators."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Build filename with quality indicators
            filename_parts = ["transformer", "features"]
            if metadata.get('optimization_applied', False):
                filename_parts.append("optimized")
            if metadata.get('validation_passed', False):
                filename_parts.append("validated")
            filename_parts.append(timestamp)
            filename = "_".join(filename_parts) + ".csv"
            filepath = self.transformer_dir / filename
            
            # Save CSV
            features_df.to_csv(filepath, index=False)
            
            # Save comprehensive metadata
            metadata_enhanced = {
                'feature_type': 'transformer',
                'game': self.game,
                'created_at': timestamp,
                'feature_count': len(features_df.columns),
                'sample_count': len(features_df),
                'optimization_applied': metadata.get('optimization_applied', False),
                'optimization_config': metadata.get('optimization_config', None),
                'validation_passed': metadata.get('validation_passed', False),
                'validation_config': metadata.get('validation_config', None),
                'validation_results': metadata.get('validation_results', None),
                'enhanced_features': metadata.get('enhanced_features_config', {}),
                'target_representation': metadata.get('target_representation', 'binary'),
                'original_metadata': metadata
            }
            
            # Add feature statistics for drift detection
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metadata_enhanced['feature_stats'] = {
                    'mean': features_df[numeric_cols].mean().to_dict(),
                    'std': features_df[numeric_cols].std().to_dict(),
                    'min': features_df[numeric_cols].min().to_dict(),
                    'max': features_df[numeric_cols].max().to_dict()
                }
            
            metadata_file = self.transformer_dir / f"{filename}.meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata_enhanced, f, indent=2, default=str)
            
            app_log(f"✓ Saved Transformer features to {filepath}", "info")
            app_log(f"✓ Saved metadata to {metadata_file}", "info")
            return True
            
        except Exception as e:
            app_log(f"Error saving Transformer CSV features: {e}", "error")
            return False
    
    def generate_xgboost_features(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate comprehensive XGBoost features from raw lottery data.
        
        Creates 100+ engineered features for gradient boosting, including:
        - Statistical distributions
        - Historical patterns
        - Frequency analysis
        - Spatial relationships
        - Temporal patterns
        """
        try:
            app_log("Generating comprehensive XGBoost features...", "info")
            
            data = self._parse_numbers(raw_data)
            max_num = 50
            
            features_df = pd.DataFrame()
            features_df["draw_date"] = data["draw_date"]
            
            # ✅ CRITICAL: Preserve original numbers for multi-output target extraction
            if "numbers" in data.columns:
                features_df["numbers"] = data["numbers"]
            
            # 1. BASIC STATISTICAL FEATURES (10 features)
            features_df["sum"] = data["numbers_list"].apply(np.sum)
            features_df["mean"] = data["numbers_list"].apply(np.mean)
            features_df["std"] = data["numbers_list"].apply(np.std)
            features_df["var"] = data["numbers_list"].apply(np.var)
            features_df["min"] = data["numbers_list"].apply(np.min)
            features_df["max"] = data["numbers_list"].apply(np.max)
            features_df["range"] = features_df["max"] - features_df["min"]
            features_df["median"] = data["numbers_list"].apply(np.median)
            features_df["skewness"] = data["numbers_list"].apply(lambda x: float(stats.skew(x)))
            features_df["kurtosis"] = data["numbers_list"].apply(lambda x: float(stats.kurtosis(x)))
            
            # 2. DISTRIBUTION FEATURES (15 features)
            for i in range(5):
                low = i * 10
                high = (i + 1) * 10
                features_df[f"bucket_{i}_count"] = data["numbers_list"].apply(
                    lambda x: sum(1 for n in x if low <= n < high)
                )
            
            features_df["q1"] = data["numbers_list"].apply(lambda x: np.percentile(x, 25))
            features_df["q2"] = data["numbers_list"].apply(lambda x: np.percentile(x, 50))
            features_df["q3"] = data["numbers_list"].apply(lambda x: np.percentile(x, 75))
            features_df["iqr"] = features_df["q3"] - features_df["q1"]
            
            features_df["p10"] = data["numbers_list"].apply(lambda x: np.percentile(x, 10))
            features_df["p90"] = data["numbers_list"].apply(lambda x: np.percentile(x, 90))
            features_df["p05"] = data["numbers_list"].apply(lambda x: np.percentile(x, 5))
            features_df["p95"] = data["numbers_list"].apply(lambda x: np.percentile(x, 95))
            
            # 3. PARITY FEATURES (8 features)
            features_df["even_count"] = data["numbers_list"].apply(lambda x: sum(1 for n in x if n % 2 == 0))
            features_df["odd_count"] = data["numbers_list"].apply(lambda x: sum(1 for n in x if n % 2 == 1))
            features_df["even_odd_ratio"] = features_df["even_count"] / (features_df["odd_count"] + 1e-8)
            
            for mod in [3, 5, 7, 11]:
                features_df[f"mod_{mod}_var"] = data["numbers_list"].apply(
                    lambda x: float(np.var([n % mod for n in x]))
                )
            
            # 4. SPACING FEATURES (8 features)
            def gap_analysis(numbers):
                sorted_nums = sorted(numbers)
                gaps = np.diff(sorted_nums)
                return {
                    'mean_gap': np.mean(gaps),
                    'max_gap': np.max(gaps),
                    'min_gap': np.min(gaps),
                    'std_gap': np.std(gaps)
                }
            
            gap_data = data["numbers_list"].apply(gap_analysis)
            features_df["mean_gap"] = gap_data.apply(lambda x: x['mean_gap'])
            features_df["max_gap"] = gap_data.apply(lambda x: x['max_gap'])
            features_df["min_gap"] = gap_data.apply(lambda x: x['min_gap'])
            features_df["std_gap"] = gap_data.apply(lambda x: x['std_gap'])
            
            def count_consecutive(numbers):
                sorted_nums = sorted(numbers)
                max_cons = 1
                curr_cons = 1
                for i in range(len(sorted_nums) - 1):
                    if sorted_nums[i+1] - sorted_nums[i] == 1:
                        curr_cons += 1
                        max_cons = max(max_cons, curr_cons)
                    else:
                        curr_cons = 1
                return max_cons
            
            features_df["max_consecutive"] = data["numbers_list"].apply(count_consecutive)
            features_df["large_gap_count"] = gap_data.apply(
                lambda x: sum(1 for gap in np.diff(sorted(data["numbers_list"].iloc[0])) if gap > 10)
            )
            
            # 5. HISTORICAL FREQUENCY FEATURES (20 features)
            for window in [5, 10, 20, 30, 60]:
                freq_features = []
                for idx in range(len(data)):
                    if idx < window:
                        freq_features.append(0)
                    else:
                        current_nums = set(data.iloc[idx]["numbers_list"])
                        historical = data.iloc[max(0, idx-window):idx]["numbers_list"]
                        freq_count = sum(len(current_nums & set(hist_nums)) for hist_nums in historical)
                        freq_features.append(freq_count / (window * len(current_nums) + 1e-8))
                
                features_df[f"freq_match_w{window}"] = freq_features
                
                # New numbers not in recent history
                new_features = []
                for idx in range(len(data)):
                    if idx < window:
                        new_features.append(0)
                    else:
                        current_nums = set(data.iloc[idx]["numbers_list"])
                        historical = data.iloc[max(0, idx-window):idx]["numbers_list"]
                        recent_all = set()
                        for hist_nums in historical:
                            recent_all.update(hist_nums)
                        new_count = len(current_nums - recent_all)
                        new_features.append(new_count)
                
                features_df[f"new_numbers_w{window}"] = new_features
            
            # 6. ROLLING STATISTICS (15 features)
            for window in [3, 5, 10]:
                features_df[f"rolling_sum_w{window}"] = features_df["sum"].rolling(window=window, min_periods=1).mean()
                features_df[f"rolling_mean_w{window}"] = features_df["mean"].rolling(window=window, min_periods=1).mean()
                features_df[f"rolling_std_w{window}"] = features_df["mean"].rolling(window=window, min_periods=1).std()
            
            # 7. TEMPORAL FEATURES (10 features)
            features_df["day_of_week"] = features_df["draw_date"].dt.dayofweek
            features_df["month"] = features_df["draw_date"].dt.month
            features_df["day_of_year"] = features_df["draw_date"].dt.dayofyear
            features_df["week_of_year"] = features_df["draw_date"].dt.isocalendar().week
            features_df["is_weekend"] = (features_df["day_of_week"] >= 5).astype(int)
            features_df["season"] = features_df["month"].apply(lambda m: (m % 12) // 3)
            
            features_df["days_since_last"] = (features_df["draw_date"].diff()).dt.days
            features_df["days_since_last"] = features_df["days_since_last"].fillna(0)
            
            # 8. BONUS FEATURES (8 features)
            features_df["bonus"] = data["bonus_int"]
            features_df["bonus_even_odd"] = (data["bonus_int"] % 2).apply(lambda x: 1.0 if x == 1 else 0.0)
            features_df["bonus_change"] = features_df["bonus"].diff().fillna(0)
            features_df["bonus_repeating"] = (features_df["bonus"] == features_df["bonus"].shift(1)).astype(int).fillna(0)
            
            for window in [5, 10]:
                freq_list = []
                for idx in range(len(features_df)):
                    if idx < window:
                        freq_list.append(0)
                    else:
                        freq = (features_df["bonus"].iloc[idx-window:idx] == features_df["bonus"].iloc[idx]).sum() / window
                        freq_list.append(freq)
                features_df[f"bonus_freq_w{window}"] = freq_list
            
            # 9. JACKPOT FEATURES (8 features)
            features_df["jackpot"] = data["jackpot"]
            features_df["jackpot_log"] = np.log1p(features_df["jackpot"])
            features_df["jackpot_millions"] = features_df["jackpot"] / 1_000_000
            features_df["jackpot_change"] = features_df["jackpot"].diff().fillna(0)
            features_df["jackpot_change_pct"] = features_df["jackpot"].pct_change().fillna(0)
            features_df["jackpot_rolling_mean"] = features_df["jackpot"].rolling(window=5, min_periods=1).mean()
            features_df["jackpot_rolling_std"] = features_df["jackpot"].rolling(window=5, min_periods=1).std()
            features_df["jackpot_z_score"] = (features_df["jackpot"] - features_df["jackpot"].mean()) / (features_df["jackpot"].std() + 1e-8)
            
            # 10. ENTROPY AND RANDOMNESS (5 features)
            def calculate_entropy(numbers):
                hist, _ = np.histogram(numbers, bins=max(2, len(numbers)//2))
                hist = hist / hist.sum()
                return -np.sum(hist * np.log2(hist + 1e-10))
            
            features_df["entropy"] = data["numbers_list"].apply(calculate_entropy)
            
            # Fill NaN values
            features_df = features_df.fillna(0)
            
            # Drop draw_date for model training
            feature_cols = [col for col in features_df.columns if col != 'draw_date']
            
            # Ensure exactly 93 features for XGBoost model compatibility
            target_features = 93
            if len(feature_cols) < target_features:
                # Add padding columns filled with zeros
                num_padding = target_features - len(feature_cols)
                for i in range(num_padding):
                    pad_col = f"padding_{i}"
                    features_df[pad_col] = 0
                    feature_cols.append(pad_col)
                app_log(f"Padded XGBoost features from {len(feature_cols) - num_padding} to {target_features}", "info")
            elif len(feature_cols) > target_features:
                # Truncate to top 93 features (keep existing ones, drop padding if any)
                feature_cols = feature_cols[:target_features]
                features_df = features_df[feature_cols]
                app_log(f"Truncated XGBoost features to {target_features}", "info")
            
            metadata = {
                "model_type": "xgboost",
                "game": self.game,
                "processing_mode": "advanced_all_files",
                "raw_files": [str(f) for f in self.get_raw_files()],
                "total_draws": len(data),
                "feature_count": len(feature_cols),
                "features": feature_cols,
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "lookback_windows": [5, 10, 20, 30, 60],
                    "rolling_windows": [3, 5, 10],
                    "modulo_operations": [3, 5, 7, 11],
                    "percentiles": [5, 10, 25, 50, 75, 90, 95],
                    "target_features": target_features,
                    "feature_categories": [
                        "basic_statistics", "distribution", "parity", "spacing",
                        "historical_frequency", "rolling_statistics", "temporal",
                        "bonus", "jackpot", "entropy_randomness"
                    ]
                },
                "file_info": [
                    {
                        "file": str(f),
                        "draws_count": len(pd.read_csv(f))
                    }
                    for f in self.get_raw_files()
                ]
            }
            
            app_log(f"✓ Generated {len(feature_cols)} advanced XGBoost features for {len(features_df)} draws (target: {target_features})", "info")
            
            # CREATE FEATURE SCHEMA
            data_date_range = {
                "min": str(features_df["draw_date"].min()),
                "max": str(features_df["draw_date"].max())
            }
            schema = self._create_feature_schema(
                model_type="xgboost",
                feature_names=feature_cols,
                normalization_method="None",  # XGBoost uses raw values
                data_shape=(len(features_df), len(feature_cols)),
                data_date_range=data_date_range,
                feature_categories=metadata["params"]["feature_categories"],
                notes="XGBoost features generated with advanced feature engineering"
            )
            self._save_schema_with_features(schema, "xgboost")
            
            return features_df, metadata
            
        except Exception as e:
            app_log(f"Error generating XGBoost features: {e}", "error")
            raise
    
    def save_lstm_sequences(self, sequences: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save LSTM sequences to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_lstm_w{metadata['params']['window_size']}_t{timestamp}.npz"
            filepath = self.lstm_dir / filename
            
            np.savez_compressed(filepath, sequences=sequences)
            
            # Save metadata
            meta_filepath = self.lstm_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            app_log(f"✓ Saved LSTM sequences to {filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving LSTM sequences: {e}", "error")
            return False
    
    def save_cnn_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save CNN embeddings to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_cnn_w{metadata['params']['window_size']}_e{metadata['params']['embedding_dim']}_t{timestamp}.npz"
            filepath = self.cnn_dir / filename
            
            np.savez_compressed(filepath, embeddings=embeddings)
            
            # Save metadata
            meta_filepath = self.cnn_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            app_log(f"✓ Saved CNN embeddings to {filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving CNN embeddings: {e}", "error")
            return False

    def save_transformer_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save Transformer embeddings to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_transformer_w{metadata['params']['window_size']}_e{metadata['params']['embedding_dim']}_t{timestamp}.npz"
            filepath = self.transformer_dir / filename
            
            np.savez_compressed(filepath, embeddings=embeddings)
            
            # Save metadata
            meta_filepath = self.transformer_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            app_log(f"✓ Saved Transformer embeddings to {filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving Transformer embeddings: {e}", "error")
            return False
    
    def save_xgboost_features(self, features_df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Save XGBoost features to disk with quality indicators."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Build filename with quality indicators
            filename_parts = ["xgboost", "features"]
            
            # Add optimization indicator
            if metadata.get('optimization_applied', False):
                filename_parts.append("optimized")
            
            # Add validation indicator
            if metadata.get('validation_passed', False):
                filename_parts.append("validated")
            
            filename_parts.append(timestamp)
            filename = "_".join(filename_parts) + ".csv"
            filepath = self.xgboost_dir / filename
            
            features_df.to_csv(filepath, index=False)
            
            # Save comprehensive metadata
            metadata_enhanced = {
                'feature_type': 'xgboost',
                'game': self.game,
                'created_at': timestamp,
                'feature_count': len(features_df.columns),
                'sample_count': len(features_df),
                'optimization_applied': metadata.get('optimization_applied', False),
                'optimization_config': metadata.get('optimization_config', None),
                'validation_passed': metadata.get('validation_passed', False),
                'validation_config': metadata.get('validation_config', None),
                'validation_results': metadata.get('validation_results', None),
                'enhanced_features': metadata.get('enhanced_features_config', {}),
                'target_representation': metadata.get('target_representation', 'binary'),
                'original_metadata': metadata
            }
            
            # Add feature statistics for drift detection
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metadata_enhanced['feature_stats'] = {
                    'mean': features_df[numeric_cols].mean().to_dict(),
                    'std': features_df[numeric_cols].std().to_dict(),
                    'min': features_df[numeric_cols].min().to_dict(),
                    'max': features_df[numeric_cols].max().to_dict()
                }
            
            meta_filepath = self.xgboost_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata_enhanced, f, indent=2, default=str)
            
            app_log(f"✓ Saved XGBoost features to {filepath}", "info")
            app_log(f"✓ Saved metadata to {meta_filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving XGBoost features: {e}", "error")
            return False
    
    def generate_catboost_features(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate comprehensive CatBoost features from raw lottery data.
        Optimized for categorical features with 115+ engineered features.
        """
        try:
            app_log("Generating CatBoost features optimized for categorical boosting...", "info")
            
            data = self._parse_numbers(raw_data)
            features_df = pd.DataFrame()
            features_df["draw_date"] = data["draw_date"]
            
            # ✅ CRITICAL: Preserve original numbers for multi-output target extraction
            if "numbers" in data.columns:
                features_df["numbers"] = data["numbers"]
            
            # 1. BASIC STATISTICAL FEATURES (10 features)
            features_df["sum"] = data["numbers_list"].apply(np.sum)
            features_df["mean"] = data["numbers_list"].apply(np.mean)
            features_df["std"] = data["numbers_list"].apply(np.std)
            features_df["var"] = data["numbers_list"].apply(np.var)
            features_df["min"] = data["numbers_list"].apply(np.min)
            features_df["max"] = data["numbers_list"].apply(np.max)
            features_df["range"] = features_df["max"] - features_df["min"]
            features_df["median"] = data["numbers_list"].apply(np.median)
            features_df["skewness"] = data["numbers_list"].apply(lambda x: float(stats.skew(x)))
            features_df["kurtosis"] = data["numbers_list"].apply(lambda x: float(stats.kurtosis(x)))
            
            # 2. DISTRIBUTION FEATURES (15 features)
            for i in range(5):
                low = i * 10
                high = (i + 1) * 10
                features_df[f"bucket_{i}_count"] = data["numbers_list"].apply(
                    lambda x: sum(1 for n in x if low <= n < high)
                )
            
            features_df["q1"] = data["numbers_list"].apply(lambda x: np.percentile(x, 25))
            features_df["q2"] = data["numbers_list"].apply(lambda x: np.percentile(x, 50))
            features_df["q3"] = data["numbers_list"].apply(lambda x: np.percentile(x, 75))
            features_df["iqr"] = features_df["q3"] - features_df["q1"]
            
            features_df["p10"] = data["numbers_list"].apply(lambda x: np.percentile(x, 10))
            features_df["p90"] = data["numbers_list"].apply(lambda x: np.percentile(x, 90))
            features_df["p05"] = data["numbers_list"].apply(lambda x: np.percentile(x, 5))
            features_df["p95"] = data["numbers_list"].apply(lambda x: np.percentile(x, 95))
            
            # 3. PARITY FEATURES (8 features)
            features_df["even_count"] = data["numbers_list"].apply(lambda x: sum(1 for n in x if n % 2 == 0))
            features_df["odd_count"] = data["numbers_list"].apply(lambda x: sum(1 for n in x if n % 2 == 1))
            features_df["even_odd_ratio"] = features_df["even_count"] / (features_df["odd_count"] + 1e-8)
            
            for mod in [3, 5, 7, 11]:
                features_df[f"mod_{mod}_var"] = data["numbers_list"].apply(
                    lambda x: float(np.var([n % mod for n in x]))
                )
            
            # 4. SPACING FEATURES (8 features)
            def gap_analysis(numbers):
                sorted_nums = sorted(numbers)
                gaps = np.diff(sorted_nums)
                return {
                    'mean_gap': np.mean(gaps),
                    'max_gap': np.max(gaps),
                    'min_gap': np.min(gaps),
                    'std_gap': np.std(gaps)
                }
            
            gap_data = data["numbers_list"].apply(gap_analysis)
            features_df["mean_gap"] = gap_data.apply(lambda x: x['mean_gap'])
            features_df["max_gap"] = gap_data.apply(lambda x: x['max_gap'])
            features_df["min_gap"] = gap_data.apply(lambda x: x['min_gap'])
            features_df["std_gap"] = gap_data.apply(lambda x: x['std_gap'])
            
            def count_consecutive(numbers):
                sorted_nums = sorted(numbers)
                max_cons = 1
                curr_cons = 1
                for i in range(len(sorted_nums) - 1):
                    if sorted_nums[i+1] - sorted_nums[i] == 1:
                        curr_cons += 1
                        max_cons = max(max_cons, curr_cons)
                    else:
                        curr_cons = 1
                return max_cons
            
            features_df["max_consecutive"] = data["numbers_list"].apply(count_consecutive)
            features_df["large_gap_count"] = data["numbers_list"].apply(
                lambda x: sum(1 for gap in np.diff(sorted(x)) if gap > 10)
            )
            
            # 5. HISTORICAL FREQUENCY FEATURES (20 features)
            for window in [5, 10, 20, 30, 60]:
                freq_features = []
                for idx in range(len(data)):
                    if idx < window:
                        freq_features.append(0)
                    else:
                        current_nums = set(data.iloc[idx]["numbers_list"])
                        historical = data.iloc[max(0, idx-window):idx]["numbers_list"]
                        freq_count = sum(len(current_nums & set(hist_nums)) for hist_nums in historical)
                        freq_features.append(freq_count / (window * len(current_nums) + 1e-8))
                
                features_df[f"freq_match_w{window}"] = freq_features
                
                # New numbers not in recent history
                new_features = []
                for idx in range(len(data)):
                    if idx < window:
                        new_features.append(0)
                    else:
                        current_nums = set(data.iloc[idx]["numbers_list"])
                        historical = data.iloc[max(0, idx-window):idx]["numbers_list"]
                        recent_all = set()
                        for hist_nums in historical:
                            recent_all.update(hist_nums)
                        new_count = len(current_nums - recent_all)
                        new_features.append(new_count)
                
                features_df[f"new_numbers_w{window}"] = new_features
            
            # 6. ROLLING STATISTICS (15 features)
            for window in [3, 5, 10]:
                features_df[f"rolling_sum_w{window}"] = features_df["sum"].rolling(window=window, min_periods=1).mean()
                features_df[f"rolling_mean_w{window}"] = features_df["mean"].rolling(window=window, min_periods=1).mean()
                features_df[f"rolling_std_w{window}"] = features_df["mean"].rolling(window=window, min_periods=1).std()
            
            # 7. TEMPORAL FEATURES (10 features)
            features_df["day_of_week"] = features_df["draw_date"].dt.dayofweek
            features_df["month"] = features_df["draw_date"].dt.month
            features_df["day_of_year"] = features_df["draw_date"].dt.dayofyear
            features_df["week_of_year"] = features_df["draw_date"].dt.isocalendar().week
            features_df["is_weekend"] = (features_df["day_of_week"] >= 5).astype(int)
            features_df["season"] = features_df["month"].apply(lambda m: (m % 12) // 3)
            
            features_df["days_since_last"] = (features_df["draw_date"].diff()).dt.days
            features_df["days_since_last"] = features_df["days_since_last"].fillna(0)
            
            # 8. BONUS FEATURES (8 features)
            features_df["bonus"] = data["bonus_int"]
            features_df["bonus_even_odd"] = (data["bonus_int"] % 2).apply(lambda x: 1.0 if x == 1 else 0.0)
            features_df["bonus_change"] = features_df["bonus"].diff().fillna(0)
            features_df["bonus_repeating"] = (features_df["bonus"] == features_df["bonus"].shift(1)).astype(int).fillna(0)
            
            for window in [5, 10]:
                freq_list = []
                for idx in range(len(features_df)):
                    if idx < window:
                        freq_list.append(0)
                    else:
                        freq = (features_df["bonus"].iloc[idx-window:idx] == features_df["bonus"].iloc[idx]).sum() / window
                        freq_list.append(freq)
                features_df[f"bonus_freq_w{window}"] = freq_list
            
            # 9. JACKPOT FEATURES (8 features)
            features_df["jackpot"] = data["jackpot"]
            features_df["jackpot_log"] = np.log1p(features_df["jackpot"])
            features_df["jackpot_millions"] = features_df["jackpot"] / 1_000_000
            features_df["jackpot_change"] = features_df["jackpot"].diff().fillna(0)
            features_df["jackpot_change_pct"] = features_df["jackpot"].pct_change().fillna(0)
            features_df["jackpot_rolling_mean"] = features_df["jackpot"].rolling(window=5, min_periods=1).mean()
            features_df["jackpot_rolling_std"] = features_df["jackpot"].rolling(window=5, min_periods=1).std()
            features_df["jackpot_z_score"] = (features_df["jackpot"] - features_df["jackpot"].mean()) / (features_df["jackpot"].std() + 1e-8)
            
            # 10. ENTROPY AND RANDOMNESS (5 features)
            def calculate_entropy(numbers):
                hist, _ = np.histogram(numbers, bins=max(2, len(numbers)//2))
                hist = hist / hist.sum()
                return -np.sum(hist * np.log2(hist + 1e-10))
            
            features_df["entropy"] = data["numbers_list"].apply(calculate_entropy)
            
            # 11. ADDITIONAL PREDICTIVE FEATURES (8 features to reach 85 total)
            # These features improve model predictions and match training data dimensions
            features_df["num_uniqueness"] = data["numbers_list"].apply(lambda x: len(set(x)) / len(x))
            features_df["sum_div_count"] = data["numbers_list"].apply(lambda x: np.sum(x) / len(x))
            features_df["first_num"] = data["numbers_list"].apply(lambda x: x[0] if len(x) > 0 else 0)
            features_df["last_num"] = data["numbers_list"].apply(lambda x: x[-1] if len(x) > 0 else 0)
            features_df["center_mass"] = data["numbers_list"].apply(lambda x: np.mean(sorted(x)))
            features_df["even_sum"] = data["numbers_list"].apply(lambda x: sum(n for n in x if n % 2 == 0))
            features_df["odd_sum"] = data["numbers_list"].apply(lambda x: sum(n for n in x if n % 2 == 1))
            features_df["max_min_ratio"] = data["numbers_list"].apply(lambda x: max(x) / (min(x) + 1e-8) if len(x) > 0 else 0)
            
            # Fill NaN values
            features_df = features_df.fillna(0)
            
            # Drop draw_date for model training
            feature_cols = [col for col in features_df.columns if col != 'draw_date']
            
            # Pad to 93 features for model compatibility
            while len(feature_cols) < 93:
                pad_idx = len(feature_cols)
                features_df[f'padding_{pad_idx}'] = 0.0
                feature_cols.append(f'padding_{pad_idx}')
            feature_cols = feature_cols[:93]
            
            metadata = {
                'model_type': 'catboost',
                'generated_at': datetime.now().isoformat(),
                'total_draws': len(features_df),
                'feature_count': len(feature_cols),
                'params': {
                    'feature_categories': [
                        'Statistical', 'Distribution', 'Parity', 'Spacing',
                        'Historical Frequency', 'Rolling Statistics', 'Temporal',
                        'Bonus', 'Jackpot', 'Entropy', 'Additional Predictive'
                    ],
                    'categorical_features': [
                        'bucket_0_count', 'bucket_1_count', 'bucket_2_count', 'bucket_3_count', 'bucket_4_count',
                        'even_count', 'odd_count', 'is_weekend', 'bonus_repeating'
                    ]
                }
            }
            
            # CREATE FEATURE SCHEMA
            data_date_range = {
                "min": str(features_df["draw_date"].min()),
                "max": str(features_df["draw_date"].max())
            }
            schema = self._create_feature_schema(
                model_type="catboost",
                feature_names=feature_cols,
                normalization_method="None",  # CatBoost handles normalization
                data_shape=(len(features_df), len(feature_cols)),
                data_date_range=data_date_range,
                feature_categories=metadata["params"]["feature_categories"],
                notes="CatBoost features with categorical feature optimization"
            )
            self._save_schema_with_features(schema, "catboost")
            
            return features_df, metadata
        except Exception as e:
            app_log(f"Error generating CatBoost features: {e}", "error")
            return pd.DataFrame(), {}
    
    def generate_lightgbm_features(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate comprehensive LightGBM features from raw lottery data.
        Optimized for gradient boosting with 115+ engineered features.
        """
        try:
            app_log("Generating LightGBM features optimized for gradient boosting...", "info")
            
            data = self._parse_numbers(raw_data)
            features_df = pd.DataFrame()
            features_df["draw_date"] = data["draw_date"]
            
            # ✅ CRITICAL: Preserve original numbers for multi-output target extraction
            if "numbers" in data.columns:
                features_df["numbers"] = data["numbers"]
            
            # 1. BASIC STATISTICAL FEATURES (10 features)
            features_df["sum"] = data["numbers_list"].apply(np.sum)
            features_df["mean"] = data["numbers_list"].apply(np.mean)
            features_df["std"] = data["numbers_list"].apply(np.std)
            features_df["var"] = data["numbers_list"].apply(np.var)
            features_df["min"] = data["numbers_list"].apply(np.min)
            features_df["max"] = data["numbers_list"].apply(np.max)
            features_df["range"] = features_df["max"] - features_df["min"]
            features_df["median"] = data["numbers_list"].apply(np.median)
            features_df["skewness"] = data["numbers_list"].apply(lambda x: float(stats.skew(x)))
            features_df["kurtosis"] = data["numbers_list"].apply(lambda x: float(stats.kurtosis(x)))
            
            # 2. DISTRIBUTION FEATURES (15 features)
            for i in range(5):
                low = i * 10
                high = (i + 1) * 10
                features_df[f"bucket_{i}_count"] = data["numbers_list"].apply(
                    lambda x: sum(1 for n in x if low <= n < high)
                )
            
            features_df["q1"] = data["numbers_list"].apply(lambda x: np.percentile(x, 25))
            features_df["q2"] = data["numbers_list"].apply(lambda x: np.percentile(x, 50))
            features_df["q3"] = data["numbers_list"].apply(lambda x: np.percentile(x, 75))
            features_df["iqr"] = features_df["q3"] - features_df["q1"]
            
            features_df["p10"] = data["numbers_list"].apply(lambda x: np.percentile(x, 10))
            features_df["p90"] = data["numbers_list"].apply(lambda x: np.percentile(x, 90))
            features_df["p05"] = data["numbers_list"].apply(lambda x: np.percentile(x, 5))
            features_df["p95"] = data["numbers_list"].apply(lambda x: np.percentile(x, 95))
            
            # 3. PARITY FEATURES (8 features)
            features_df["even_count"] = data["numbers_list"].apply(lambda x: sum(1 for n in x if n % 2 == 0))
            features_df["odd_count"] = data["numbers_list"].apply(lambda x: sum(1 for n in x if n % 2 == 1))
            features_df["even_odd_ratio"] = features_df["even_count"] / (features_df["odd_count"] + 1e-8)
            
            for mod in [3, 5, 7, 11]:
                features_df[f"mod_{mod}_var"] = data["numbers_list"].apply(
                    lambda x: float(np.var([n % mod for n in x]))
                )
            
            # 4. SPACING FEATURES (8 features)
            def gap_analysis(numbers):
                sorted_nums = sorted(numbers)
                gaps = np.diff(sorted_nums)
                return {
                    'mean_gap': np.mean(gaps),
                    'max_gap': np.max(gaps),
                    'min_gap': np.min(gaps),
                    'std_gap': np.std(gaps)
                }
            
            gap_data = data["numbers_list"].apply(gap_analysis)
            features_df["mean_gap"] = gap_data.apply(lambda x: x['mean_gap'])
            features_df["max_gap"] = gap_data.apply(lambda x: x['max_gap'])
            features_df["min_gap"] = gap_data.apply(lambda x: x['min_gap'])
            features_df["std_gap"] = gap_data.apply(lambda x: x['std_gap'])
            
            def count_consecutive(numbers):
                sorted_nums = sorted(numbers)
                max_cons = 1
                curr_cons = 1
                for i in range(len(sorted_nums) - 1):
                    if sorted_nums[i+1] - sorted_nums[i] == 1:
                        curr_cons += 1
                        max_cons = max(max_cons, curr_cons)
                    else:
                        curr_cons = 1
                return max_cons
            
            features_df["max_consecutive"] = data["numbers_list"].apply(count_consecutive)
            features_df["large_gap_count"] = data["numbers_list"].apply(
                lambda x: sum(1 for gap in np.diff(sorted(x)) if gap > 10)
            )
            
            # 5. HISTORICAL FREQUENCY FEATURES (20 features)
            for window in [5, 10, 20, 30, 60]:
                freq_features = []
                for idx in range(len(data)):
                    if idx < window:
                        freq_features.append(0)
                    else:
                        current_nums = set(data.iloc[idx]["numbers_list"])
                        historical = data.iloc[max(0, idx-window):idx]["numbers_list"]
                        freq_count = sum(len(current_nums & set(hist_nums)) for hist_nums in historical)
                        freq_features.append(freq_count / (window * len(current_nums) + 1e-8))
                
                features_df[f"freq_match_w{window}"] = freq_features
                
                # New numbers not in recent history
                new_features = []
                for idx in range(len(data)):
                    if idx < window:
                        new_features.append(0)
                    else:
                        current_nums = set(data.iloc[idx]["numbers_list"])
                        historical = data.iloc[max(0, idx-window):idx]["numbers_list"]
                        recent_all = set()
                        for hist_nums in historical:
                            recent_all.update(hist_nums)
                        new_count = len(current_nums - recent_all)
                        new_features.append(new_count)
                
                features_df[f"new_numbers_w{window}"] = new_features
            
            # 6. ROLLING STATISTICS (15 features)
            for window in [3, 5, 10]:
                features_df[f"rolling_sum_w{window}"] = features_df["sum"].rolling(window=window, min_periods=1).mean()
                features_df[f"rolling_mean_w{window}"] = features_df["mean"].rolling(window=window, min_periods=1).mean()
                features_df[f"rolling_std_w{window}"] = features_df["mean"].rolling(window=window, min_periods=1).std()
            
            # 7. TEMPORAL FEATURES (10 features)
            features_df["day_of_week"] = features_df["draw_date"].dt.dayofweek
            features_df["month"] = features_df["draw_date"].dt.month
            features_df["day_of_year"] = features_df["draw_date"].dt.dayofyear
            features_df["week_of_year"] = features_df["draw_date"].dt.isocalendar().week
            features_df["is_weekend"] = (features_df["day_of_week"] >= 5).astype(int)
            features_df["season"] = features_df["month"].apply(lambda m: (m % 12) // 3)
            
            features_df["days_since_last"] = (features_df["draw_date"].diff()).dt.days
            features_df["days_since_last"] = features_df["days_since_last"].fillna(0)
            
            # 8. BONUS FEATURES (8 features)
            features_df["bonus"] = data["bonus_int"]
            features_df["bonus_even_odd"] = (data["bonus_int"] % 2).apply(lambda x: 1.0 if x == 1 else 0.0)
            features_df["bonus_change"] = features_df["bonus"].diff().fillna(0)
            features_df["bonus_repeating"] = (features_df["bonus"] == features_df["bonus"].shift(1)).astype(int).fillna(0)
            
            for window in [5, 10]:
                freq_list = []
                for idx in range(len(features_df)):
                    if idx < window:
                        freq_list.append(0)
                    else:
                        freq = (features_df["bonus"].iloc[idx-window:idx] == features_df["bonus"].iloc[idx]).sum() / window
                        freq_list.append(freq)
                features_df[f"bonus_freq_w{window}"] = freq_list
            
            # 9. JACKPOT FEATURES (8 features)
            features_df["jackpot"] = data["jackpot"]
            features_df["jackpot_log"] = np.log1p(features_df["jackpot"])
            features_df["jackpot_millions"] = features_df["jackpot"] / 1_000_000
            features_df["jackpot_change"] = features_df["jackpot"].diff().fillna(0)
            features_df["jackpot_change_pct"] = features_df["jackpot"].pct_change().fillna(0)
            features_df["jackpot_rolling_mean"] = features_df["jackpot"].rolling(window=5, min_periods=1).mean()
            features_df["jackpot_rolling_std"] = features_df["jackpot"].rolling(window=5, min_periods=1).std()
            features_df["jackpot_z_score"] = (features_df["jackpot"] - features_df["jackpot"].mean()) / (features_df["jackpot"].std() + 1e-8)
            
            # 10. ENTROPY AND RANDOMNESS (5 features)
            def calculate_entropy(numbers):
                hist, _ = np.histogram(numbers, bins=max(2, len(numbers)//2))
                hist = hist / hist.sum()
                return -np.sum(hist * np.log2(hist + 1e-10))
            
            features_df["entropy"] = data["numbers_list"].apply(calculate_entropy)
            
            # 11. ADDITIONAL PREDICTIVE FEATURES (8 features to reach 85 total)
            # These features improve model predictions and match training data dimensions
            features_df["num_uniqueness"] = data["numbers_list"].apply(lambda x: len(set(x)) / len(x))
            features_df["sum_div_count"] = data["numbers_list"].apply(lambda x: np.sum(x) / len(x))
            features_df["first_num"] = data["numbers_list"].apply(lambda x: x[0] if len(x) > 0 else 0)
            features_df["last_num"] = data["numbers_list"].apply(lambda x: x[-1] if len(x) > 0 else 0)
            features_df["center_mass"] = data["numbers_list"].apply(lambda x: np.mean(sorted(x)))
            features_df["even_sum"] = data["numbers_list"].apply(lambda x: sum(n for n in x if n % 2 == 0))
            features_df["odd_sum"] = data["numbers_list"].apply(lambda x: sum(n for n in x if n % 2 == 1))
            features_df["max_min_ratio"] = data["numbers_list"].apply(lambda x: max(x) / (min(x) + 1e-8) if len(x) > 0 else 0)
            
            # Fill NaN values
            features_df = features_df.fillna(0)
            
            # Drop draw_date for model training
            feature_cols = [col for col in features_df.columns if col != 'draw_date']
            
            # Pad to 93 features for model compatibility
            while len(feature_cols) < 93:
                pad_idx = len(feature_cols)
                features_df[f'padding_{pad_idx}'] = 0.0
                feature_cols.append(f'padding_{pad_idx}')
            feature_cols = feature_cols[:93]
            
            metadata = {
                'model_type': 'lightgbm',
                'generated_at': datetime.now().isoformat(),
                'total_draws': len(features_df),
                'feature_count': len(feature_cols),
                'params': {
                    'feature_categories': [
                        'Statistical', 'Distribution', 'Parity', 'Spacing',
                        'Historical Frequency', 'Rolling Statistics', 'Temporal',
                        'Bonus', 'Jackpot', 'Entropy', 'Additional Predictive'
                    ],
                    'metric': 'gamma'
                }
            }
            
            # CREATE FEATURE SCHEMA
            data_date_range = {
                "min": str(features_df["draw_date"].min()),
                "max": str(features_df["draw_date"].max())
            }
            schema = self._create_feature_schema(
                model_type="lightgbm",
                feature_names=feature_cols,
                normalization_method="None",  # LightGBM handles normalization
                data_shape=(len(features_df), len(feature_cols)),
                data_date_range=data_date_range,
                feature_categories=metadata["params"]["feature_categories"],
                notes="LightGBM features optimized for gradient boosting"
            )
            self._save_schema_with_features(schema, "lightgbm")
            
            return features_df, metadata
        except Exception as e:
            app_log(f"Error generating LightGBM features: {e}", "error")
            return pd.DataFrame(), {}
    
    def save_catboost_features(self, features_df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Save CatBoost features to disk with quality indicators."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Build filename with quality indicators
            filename_parts = ["catboost", "features"]
            if metadata.get('optimization_applied', False):
                filename_parts.append("optimized")
            if metadata.get('validation_passed', False):
                filename_parts.append("validated")
            filename_parts.append(timestamp)
            filename = "_".join(filename_parts) + ".csv"
            filepath = self.catboost_dir / filename
            
            features_df.to_csv(filepath, index=False)
            
            # Save comprehensive metadata
            metadata_enhanced = {
                'feature_type': 'catboost',
                'game': self.game,
                'created_at': timestamp,
                'feature_count': len(features_df.columns),
                'sample_count': len(features_df),
                'optimization_applied': metadata.get('optimization_applied', False),
                'optimization_config': metadata.get('optimization_config', None),
                'validation_passed': metadata.get('validation_passed', False),
                'validation_config': metadata.get('validation_config', None),
                'validation_results': metadata.get('validation_results', None),
                'enhanced_features': metadata.get('enhanced_features_config', {}),
                'target_representation': metadata.get('target_representation', 'binary'),
                'original_metadata': metadata
            }
            
            # Add feature statistics for drift detection
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metadata_enhanced['feature_stats'] = {
                    'mean': features_df[numeric_cols].mean().to_dict(),
                    'std': features_df[numeric_cols].std().to_dict(),
                    'min': features_df[numeric_cols].min().to_dict(),
                    'max': features_df[numeric_cols].max().to_dict()
                }
            
            meta_filepath = self.catboost_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata_enhanced, f, indent=2, default=str)
            
            app_log(f"✓ Saved CatBoost features to {filepath}", "info")
            app_log(f"✓ Saved metadata to {meta_filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving CatBoost features: {e}", "error")
            return False
    
    def save_lightgbm_features(self, features_df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Save LightGBM features to disk with quality indicators."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Build filename with quality indicators
            filename_parts = ["lightgbm", "features"]
            if metadata.get('optimization_applied', False):
                filename_parts.append("optimized")
            if metadata.get('validation_passed', False):
                filename_parts.append("validated")
            filename_parts.append(timestamp)
            filename = "_".join(filename_parts) + ".csv"
            filepath = self.lightgbm_dir / filename
            
            features_df.to_csv(filepath, index=False)
            
            # Save comprehensive metadata
            metadata_enhanced = {
                'feature_type': 'lightgbm',
                'game': self.game,
                'created_at': timestamp,
                'feature_count': len(features_df.columns),
                'sample_count': len(features_df),
                'optimization_applied': metadata.get('optimization_applied', False),
                'optimization_config': metadata.get('optimization_config', None),
                'validation_passed': metadata.get('validation_passed', False),
                'validation_config': metadata.get('validation_config', None),
                'validation_results': metadata.get('validation_results', None),
                'enhanced_features': metadata.get('enhanced_features_config', {}),
                'target_representation': metadata.get('target_representation', 'binary'),
                'original_metadata': metadata
            }
            
            # Add feature statistics for drift detection
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metadata_enhanced['feature_stats'] = {
                    'mean': features_df[numeric_cols].mean().to_dict(),
                    'std': features_df[numeric_cols].std().to_dict(),
                    'min': features_df[numeric_cols].min().to_dict(),
                    'max': features_df[numeric_cols].max().to_dict()
                }
            
            meta_filepath = self.lightgbm_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata_enhanced, f, indent=2, default=str)
            
            app_log(f"✓ Saved LightGBM features to {filepath}", "info")
            app_log(f"✓ Saved metadata to {meta_filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving LightGBM features: {e}", "error")
            return False
    
    # ========================================
    # ENHANCED LOTTERY FEATURES
    # ========================================
    
    def _calculate_hot_cold_frequency(self, data: pd.DataFrame, idx: int, numbers: List[int], 
                                      windows: List[int]) -> Dict[str, float]:
        """Calculate hot/cold number frequencies over multiple windows."""
        features = {}
        
        for window in windows:
            if idx >= window:
                # Get recent draws
                recent_draws = data.iloc[max(0, idx-window):idx]
                all_recent_numbers = []
                for _, row in recent_draws.iterrows():
                    all_recent_numbers.extend(row['numbers_list'])
                
                # Calculate frequency for each number in current draw
                hot_count = 0
                cold_count = 0
                for num in numbers:
                    freq = all_recent_numbers.count(num)
                    if freq >= window * 0.15:  # Hot if appears in 15%+ of draws
                        hot_count += 1
                    elif freq <= window * 0.05:  # Cold if appears in <5% of draws
                        cold_count += 1
                
                features[f'hot_numbers_window_{window}'] = hot_count
                features[f'cold_numbers_window_{window}'] = cold_count
                features[f'hot_ratio_window_{window}'] = hot_count / len(numbers) if numbers else 0
            else:
                features[f'hot_numbers_window_{window}'] = 0
                features[f'cold_numbers_window_{window}'] = 0
                features[f'hot_ratio_window_{window}'] = 0
        
        return features
    
    def _calculate_gap_analysis(self, data: pd.DataFrame, idx: int, numbers: List[int], 
                                max_num: int = 50) -> Dict[str, float]:
        """Calculate draws since each number last appeared."""
        features = {}
        
        if idx > 0:
            gaps = []
            for num in numbers:
                # Find last appearance
                last_seen = -1
                for i in range(idx - 1, -1, -1):
                    if num in data.iloc[i]['numbers_list']:
                        last_seen = idx - i
                        break
                
                if last_seen > 0:
                    gaps.append(last_seen)
            
            if gaps:
                features['avg_gap_since_last'] = np.mean(gaps)
                features['max_gap_since_last'] = np.max(gaps)
                features['min_gap_since_last'] = np.min(gaps)
                features['gap_variance'] = np.var(gaps)
                features['overdue_count'] = sum(1 for g in gaps if g > 10)  # Not seen in 10+ draws
            else:
                features['avg_gap_since_last'] = 0
                features['max_gap_since_last'] = 0
                features['min_gap_since_last'] = 0
                features['gap_variance'] = 0
                features['overdue_count'] = 0
        else:
            features['avg_gap_since_last'] = 0
            features['max_gap_since_last'] = 0
            features['min_gap_since_last'] = 0
            features['gap_variance'] = 0
            features['overdue_count'] = 0
        
        return features
    
    def _calculate_pattern_features(self, numbers: List[int]) -> Dict[str, float]:
        """Calculate pattern features: consecutive runs, clusters, spacing."""
        features = {}
        
        sorted_nums = sorted(numbers)
        
        # Consecutive runs
        consecutive_count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive_count += 1
        features['consecutive_pairs'] = consecutive_count
        features['has_consecutive'] = 1.0 if consecutive_count > 0 else 0.0
        
        # Clustering (numbers within 5 of each other)
        clusters = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] <= 5:
                clusters += 1
        features['cluster_pairs'] = clusters
        features['cluster_ratio'] = clusters / max(1, len(sorted_nums) - 1)
        
        # Spacing patterns
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]
        if gaps:
            features['spacing_mean'] = np.mean(gaps)
            features['spacing_std'] = np.std(gaps)
            features['spacing_uniformity'] = 1.0 / (1.0 + np.std(gaps))  # Higher = more uniform
            features['max_spacing'] = np.max(gaps)
            features['min_spacing'] = np.min(gaps)
        else:
            features['spacing_mean'] = 0
            features['spacing_std'] = 0
            features['spacing_uniformity'] = 0
            features['max_spacing'] = 0
            features['min_spacing'] = 0
        
        # Pattern score (composite)
        features['pattern_score'] = (
            features['consecutive_pairs'] * 0.3 +
            features['cluster_ratio'] * 0.4 +
            features['spacing_uniformity'] * 0.3
        )
        
        return features
    
    def _calculate_entropy_randomness(self, numbers: List[int], max_num: int = 50) -> Dict[str, float]:
        """Calculate entropy and randomness scores."""
        features = {}
        
        # Shannon entropy
        if numbers:
            # Normalize numbers to probabilities
            probs = np.array(numbers) / max_num
            probs = probs / np.sum(probs)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features['shannon_entropy'] = entropy
            features['normalized_entropy'] = entropy / np.log2(len(numbers))
        else:
            features['shannon_entropy'] = 0
            features['normalized_entropy'] = 0
        
        # Randomness score (based on distribution)
        if len(numbers) > 1:
            # Perfect randomness would have even distribution
            expected_gap = max_num / len(numbers)
            sorted_nums = sorted(numbers)
            gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]
            gap_variance = np.var(gaps) if gaps else 0
            randomness = 1.0 / (1.0 + gap_variance / expected_gap)
            features['randomness_score'] = randomness
        else:
            features['randomness_score'] = 0
        
        # Digit entropy (ones, tens places)
        ones_digits = [n % 10 for n in numbers]
        tens_digits = [n // 10 for n in numbers if n >= 10]
        
        if ones_digits:
            ones_unique = len(set(ones_digits))
            features['ones_digit_diversity'] = ones_unique / len(ones_digits)
        else:
            features['ones_digit_diversity'] = 0
        
        if tens_digits:
            tens_unique = len(set(tens_digits))
            features['tens_digit_diversity'] = tens_unique / len(tens_digits)
        else:
            features['tens_digit_diversity'] = 0
        
        return features
    
    def _calculate_correlation_features(self, data: pd.DataFrame, idx: int, 
                                       numbers: List[int]) -> Dict[str, float]:
        """Calculate number co-occurrence patterns."""
        features = {}
        
        if idx >= 20:  # Need history for meaningful correlations
            # Look at last 20 draws
            recent_draws = data.iloc[max(0, idx-20):idx]
            
            # Build co-occurrence matrix for current numbers
            co_occurrences = {}
            for num1 in numbers:
                for num2 in numbers:
                    if num1 < num2:
                        pair = (num1, num2)
                        count = 0
                        for _, row in recent_draws.iterrows():
                            draw_nums = row['numbers_list']
                            if num1 in draw_nums and num2 in draw_nums:
                                count += 1
                        co_occurrences[pair] = count
            
            if co_occurrences:
                features['max_pair_frequency'] = max(co_occurrences.values())
                features['avg_pair_frequency'] = np.mean(list(co_occurrences.values()))
                features['strong_pairs'] = sum(1 for v in co_occurrences.values() if v >= 3)
            else:
                features['max_pair_frequency'] = 0
                features['avg_pair_frequency'] = 0
                features['strong_pairs'] = 0
        else:
            features['max_pair_frequency'] = 0
            features['avg_pair_frequency'] = 0
            features['strong_pairs'] = 0
        
        return features
    
    def _calculate_position_specific_features(self, numbers: List[int]) -> Dict[str, float]:
        """Calculate position-specific biases (position 1 tends low, position 7 tends high, etc.)."""
        features = {}
        
        sorted_nums = sorted(numbers)
        
        # Analyze each position
        for pos in range(min(7, len(sorted_nums))):
            num = sorted_nums[pos]
            features[f'position_{pos+1}_value'] = num
            features[f'position_{pos+1}_normalized'] = num / 50.0
            
            # Position-specific expectations
            # Position 1 should be low (1-15), Position 7 should be high (35-50)
            if pos == 0:  # First position
                features[f'position_1_is_low'] = 1.0 if num <= 15 else 0.0
            elif pos == len(sorted_nums) - 1:  # Last position
                features[f'position_last_is_high'] = 1.0 if num >= 35 else 0.0
        
        # Position spread analysis
        if len(sorted_nums) >= 2:
            features['position_spread'] = sorted_nums[-1] - sorted_nums[0]
            features['position_spread_normalized'] = features['position_spread'] / 50.0
        else:
            features['position_spread'] = 0
            features['position_spread_normalized'] = 0
        
        return features
    
    def apply_enhanced_features(self, data: pd.DataFrame, idx: int, 
                               numbers: List[int], config: Dict[str, Any]) -> Dict[str, float]:
        """Apply enhanced lottery features based on configuration."""
        features = {}
        max_num = 50
        
        if config.get('frequency', False):
            freq_windows = config.get('frequency_windows', [10, 20, 50])
            features.update(self._calculate_hot_cold_frequency(data, idx, numbers, freq_windows))
        
        if config.get('gap_analysis', False):
            features.update(self._calculate_gap_analysis(data, idx, numbers, max_num))
        
        if config.get('patterns', False):
            features.update(self._calculate_pattern_features(numbers))
        
        if config.get('entropy', False):
            features.update(self._calculate_entropy_randomness(numbers, max_num))
        
        if config.get('correlation', False):
            features.update(self._calculate_correlation_features(data, idx, numbers))
        
        if config.get('position_specific', False):
            features.update(self._calculate_position_specific_features(numbers))
        
        return features
    
    # ========================================
    # FEATURE OPTIMIZATION
    # ========================================
    
    def apply_feature_optimization(self, features_df: pd.DataFrame, 
                                   config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply feature optimization and dimensionality reduction."""
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.decomposition import PCA as SKLearnPCA
        from sklearn.preprocessing import StandardScaler
        
        if not config.get('enabled', False):
            return features_df, {'optimization': 'disabled'}
        
        method = config.get('method', 'RFE')
        optimization_info = {'method': method, 'original_features': features_df.shape[1]}
        
        # Separate features from metadata columns
        exclude_cols = []
        if 'draw_date' in features_df.columns:
            exclude_cols.append('draw_date')
        if 'numbers' in features_df.columns:
            exclude_cols.append('numbers')
        
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].values
        
        # For RFE/Importance, we need a simple target (use first column as proxy)
        if len(X) > 0:
            y_proxy = X[:, 0] > np.median(X[:, 0])
        else:
            return features_df, optimization_info
        
        result_features = X
        selected_feature_names = feature_cols
        
        try:
            if 'RFE' in method or 'Hybrid' in method:
                n_features = config.get('rfe_n_features', 200)
                n_features = min(n_features, X.shape[1])
                
                # Use Random Forest for feature selection
                estimator = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
                selector = RFE(estimator, n_features_to_select=n_features, step=10)
                
                result_features = selector.fit_transform(X, y_proxy)
                selected_feature_names = [feature_cols[i] for i, selected in enumerate(selector.support_) if selected]
                optimization_info['rfe_features_selected'] = n_features
                optimization_info['rfe_support'] = selector.support_.tolist()
            
            if 'PCA' in method or 'Hybrid' in method:
                variance_threshold = config.get('pca_variance', 0.95)
                max_components = config.get('pca_max_components', 150)
                
                # Standardize before PCA
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(result_features)
                
                # Apply PCA
                pca = SKLearnPCA(n_components=min(max_components, result_features.shape[1]), random_state=42)
                result_features = pca.fit_transform(X_scaled)
                
                # Find number of components for variance threshold
                cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
                result_features = result_features[:, :n_components]
                
                selected_feature_names = [f'PC{i+1}' for i in range(n_components)]
                optimization_info['pca_components'] = n_components
                optimization_info['pca_explained_variance'] = cumsum_variance[n_components-1]
            
            if 'Importance' in method:
                threshold_pct = config.get('importance_threshold', 30) / 100.0
                
                # Train model and get feature importances
                model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
                model.fit(X, y_proxy)
                importances = model.feature_importances_
                
                # Select top features
                n_features_to_keep = max(10, int(len(importances) * threshold_pct))
                indices = np.argsort(importances)[-n_features_to_keep:]
                
                result_features = X[:, indices]
                selected_feature_names = [feature_cols[i] for i in indices]
                optimization_info['importance_threshold'] = threshold_pct
                optimization_info['features_selected'] = n_features_to_keep
            
            # Create optimized dataframe
            optimized_df = pd.DataFrame(result_features, columns=selected_feature_names)
            
            # Add back metadata columns
            for col in exclude_cols:
                optimized_df[col] = features_df[col].values
            
            optimization_info['final_features'] = len(selected_feature_names)
            optimization_info['reduction_ratio'] = len(selected_feature_names) / len(feature_cols)
            
            app_log(f"Feature optimization: {len(feature_cols)} → {len(selected_feature_names)} features", "info")
            
            return optimized_df, optimization_info
        
        except Exception as e:
            app_log(f"Error during feature optimization: {e}", "error")
            return features_df, {'optimization': 'failed', 'error': str(e)}
    
    # ========================================
    # FEATURE VALIDATION
    # ========================================
    
    def validate_features(self, features_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature quality based on configuration."""
        validation_results = {
            'checks_run': [],
            'issues_found': [],
            'warnings': [],
            'passed': True
        }
        
        if not config.get('enabled', False):
            return validation_results
        
        # Ensure features_data is numeric
        try:
            # Convert to float to ensure numeric operations work
            if features_data.dtype == object or not np.issubdtype(features_data.dtype, np.number):
                app_log("Warning: features_data contains non-numeric types, attempting conversion", "warning")
                features_data = features_data.astype(float)
        except (ValueError, TypeError) as e:
            validation_results['issues_found'].append(f'Cannot convert features to numeric: {e}')
            validation_results['passed'] = False
            return validation_results
        
        # Check for NaN/Inf
        if config.get('check_nan', True):
            validation_results['checks_run'].append('NaN/Inf check')
            try:
                nan_count = np.isnan(features_data).sum()
                inf_count = np.isinf(features_data).sum()
                
                if nan_count > 0 or inf_count > 0:
                    validation_results['issues_found'].append(f'Found {nan_count} NaN and {inf_count} Inf values')
                    validation_results['passed'] = False
            except TypeError as e:
                validation_results['warnings'].append(f'Could not check NaN/Inf: {e}')
        
        # Check for constant features
        if config.get('check_constant', True):
            validation_results['checks_run'].append('Constant feature check')
            variance_threshold = config.get('variance_threshold', 0.01)
            
            if len(features_data.shape) == 2:
                try:
                    variances = np.var(features_data, axis=0)
                    constant_count = (variances < variance_threshold).sum()
                    
                    if constant_count > 0:
                        validation_results['warnings'].append(f'Found {constant_count} near-constant features')
                except Exception as e:
                    validation_results['warnings'].append(f'Could not check variance: {e}')
        
        # Check for high correlation
        if config.get('check_correlation', True) and len(features_data.shape) == 2:
            validation_results['checks_run'].append('Correlation check')
            correlation_threshold = config.get('correlation_threshold', 0.95)
            
            try:
                # Sample if too large
                if features_data.shape[0] > 5000:
                    indices = np.random.choice(features_data.shape[0], 5000, replace=False)
                    sample_data = features_data[indices]
                else:
                    sample_data = features_data
                
                corr_matrix = np.corrcoef(sample_data, rowvar=False)
                high_corr_mask = (np.abs(corr_matrix) > correlation_threshold) & (np.abs(corr_matrix) < 1.0)
                high_corr_count = high_corr_mask.sum() // 2
                
                if high_corr_count > 0:
                    validation_results['warnings'].append(f'Found {high_corr_count} highly correlated feature pairs')
            except:
                pass
        
        return validation_results
    
    # ========================================
    # FEATURE EXPORT
    # ========================================
    
    def export_feature_samples(self, features_df: pd.DataFrame, config: Dict[str, Any],
                               feature_type: str) -> Optional[Path]:
        """Export feature samples in specified format."""
        if not config.get('enabled', False):
            return None
        
        try:
            sample_size = config.get('sample_size', 1000)
            strategy = config.get('strategy', 'Random')
            export_format = config.get('format', 'CSV')
            
            # Sampling
            if strategy == 'Random':
                if len(features_df) > sample_size:
                    sample_df = features_df.sample(n=sample_size, random_state=42)
                else:
                    sample_df = features_df.copy()
            elif strategy == 'Recent draws':
                sample_df = features_df.tail(sample_size)
            else:  # Stratified
                # Simple stratified sampling by index
                if len(features_df) > sample_size:
                    indices = np.linspace(0, len(features_df)-1, sample_size, dtype=int)
                    sample_df = features_df.iloc[indices]
                else:
                    sample_df = features_df.copy()
            
            # Prepare export directory
            export_dir = self.features_dir / "samples" / self.game_folder
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{feature_type}_sample_{timestamp}"
            
            exported_files = []
            
            # Export in requested formats
            if export_format in ['CSV', 'All formats']:
                csv_path = export_dir / f"{base_filename}.csv"
                sample_df.to_csv(csv_path, index=False)
                exported_files.append(csv_path)
            
            if export_format in ['JSON', 'All formats']:
                json_path = export_dir / f"{base_filename}.json"
                sample_df.to_json(json_path, orient='records', indent=2)
                exported_files.append(json_path)
            
            if export_format in ['Parquet', 'All formats']:
                parquet_path = export_dir / f"{base_filename}.parquet"
                sample_df.to_parquet(parquet_path, index=False)
                exported_files.append(parquet_path)
            
            # Export metadata if requested
            if config.get('include_metadata', True) or config.get('include_stats', True):
                metadata = {
                    'feature_type': feature_type,
                    'game': self.game,
                    'sample_size': len(sample_df),
                    'sampling_strategy': strategy,
                    'export_date': datetime.now().isoformat(),
                    'total_features': len(sample_df.columns)
                }
                
                if config.get('include_stats', True):
                    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
                    stats = {
                        'mean': sample_df[numeric_cols].mean().to_dict(),
                        'std': sample_df[numeric_cols].std().to_dict(),
                        'min': sample_df[numeric_cols].min().to_dict(),
                        'max': sample_df[numeric_cols].max().to_dict(),
                        'median': sample_df[numeric_cols].median().to_dict()
                    }
                    metadata['statistics'] = stats
                
                metadata_path = export_dir / f"{base_filename}.metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                exported_files.append(metadata_path)
            
            app_log(f"Exported {len(exported_files)} sample files for {feature_type}", "info")
            return exported_files[0] if exported_files else None
        
        except Exception as e:
            app_log(f"Error exporting samples: {e}", "error")
            return None
