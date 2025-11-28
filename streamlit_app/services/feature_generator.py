"""
Advanced Feature Generation Service
Generates LSTM sequences, Transformer embeddings, and XGBoost advanced features
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import json
import logging

try:
    from ..core import get_data_dir, app_log
except ImportError:
    def get_data_dir():
        return Path("data")
    def app_log(msg: str, level: str = "info"):
        logging.log(logging.INFO if level == "info" else logging.WARNING if level == "warning" else logging.ERROR, msg)


class FeatureGenerator:
    """Generates features for machine learning models."""
    
    def __init__(self, game: str):
        """Initialize feature generator for a specific game."""
        self.game = game
        self.game_folder = game.lower().replace(" ", "_").replace("/", "_")
        self.data_dir = get_data_dir()
        self.raw_data_dir = self.data_dir / self.game_folder
        self.features_dir = self.data_dir / "features"
        
        # Create feature directories if they don't exist
        self.lstm_dir = self.features_dir / "lstm" / self.game_folder
        self.transformer_dir = self.features_dir / "transformer" / self.game_folder
        self.xgboost_dir = self.features_dir / "xgboost" / self.game_folder
        
        self.lstm_dir.mkdir(parents=True, exist_ok=True)
        self.transformer_dir.mkdir(parents=True, exist_ok=True)
        self.xgboost_dir.mkdir(parents=True, exist_ok=True)
    
    def get_raw_files(self) -> List[Path]:
        """Get all raw CSV files for the game."""
        if not self.raw_data_dir.exists():
            return []
        return sorted(self.raw_data_dir.glob("training_data_*.csv"))
    
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
            
            return combined_df
        except Exception as e:
            app_log(f"Error loading raw data: {e}", "error")
            return None
    
    def generate_lstm_sequences(self, raw_data: pd.DataFrame, 
                               window_size: int = 25,
                               include_statistics: bool = True,
                               include_trends: bool = True,
                               normalize_features: bool = True,
                               rolling_windows: List[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate LSTM sequences from raw lottery data.
        
        Args:
            raw_data: DataFrame with columns: draw_date, numbers, bonus, jackpot
            window_size: Number of past draws to use for each sequence
            include_statistics: Include statistical features
            include_trends: Include trend features
            normalize_features: Normalize features to [0, 1]
            rolling_windows: List of rolling window sizes for trend calculation
            
        Returns:
            Tuple of (sequences array, metadata dict)
        """
        if rolling_windows is None:
            rolling_windows = [5, 10, 20, 30]
        
        try:
            # Parse numbers
            raw_data = raw_data.copy()
            raw_data["numbers_list"] = raw_data["numbers"].apply(
                lambda x: list(map(int, str(x).split(",")))
            )
            raw_data["bonus_int"] = raw_data["bonus"].apply(
                lambda x: int(x) if pd.notna(x) and str(x).isdigit() else 0
            )
            
            features_list = []
            
            # Extract features for each draw
            for idx in range(len(raw_data)):
                draw_row = raw_data.iloc[idx]
                numbers = draw_row["numbers_list"]
                
                # Basic features from current draw
                features = {
                    "sum_numbers": np.sum(numbers),
                    "mean_numbers": np.mean(numbers),
                    "std_numbers": np.std(numbers),
                    "min_number": np.min(numbers),
                    "max_number": np.max(numbers),
                    "range": np.max(numbers) - np.min(numbers),
                    "bonus": draw_row["bonus_int"],
                    "jackpot": float(draw_row.get("jackpot", 0))
                }
                
                # Add statistical features if enabled
                if include_statistics:
                    features["median_numbers"] = np.median(numbers)
                    features["skew"] = float(pd.Series(numbers).skew())
                    features["kurtosis"] = float(pd.Series(numbers).kurtosis())
                
                # Add trend features if enabled and we have history
                if include_trends and idx > max(rolling_windows):
                    history_slice = raw_data.iloc[max(0, idx-30):idx]
                    
                    for win in rolling_windows:
                        if idx >= win:
                            window_data = raw_data.iloc[idx-win:idx]
                            window_sums = window_data["numbers_list"].apply(lambda x: np.sum(x))
                            
                            features[f"trend_sum_{win}"] = float(window_sums.mean())
                            features[f"trend_std_{win}"] = float(window_sums.std())
                
                features_list.append(features)
            
            # Convert to array
            feature_names = sorted(features_list[0].keys())
            feature_array = np.array([
                [f.get(name, 0) for name in feature_names] 
                for f in features_list
            ])
            
            # Normalize if requested
            if normalize_features:
                feature_array = (feature_array - feature_array.min(axis=0)) / (feature_array.max(axis=0) - feature_array.min(axis=0) + 1e-8)
            
            # Create sequences (rolling window)
            sequences = []
            for i in range(len(feature_array) - window_size):
                sequences.append(feature_array[i:i+window_size])
            
            sequences_array = np.array(sequences)
            
            # Metadata
            metadata = {
                "model_type": "lstm",
                "game": self.game,
                "processing_mode": "all_files",
                "raw_files": [str(f) for f in self.get_raw_files()],
                "file_info": [
                    {
                        "file": str(f),
                        "draws_count": len(pd.read_csv(f))
                    }
                    for f in self.get_raw_files()
                ],
                "total_draws": len(raw_data),
                "consistent_draws": len(feature_array),
                "original_draws": len(raw_data),
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "window": window_size,
                    "include_statistics": include_statistics,
                    "include_trends": include_trends,
                    "normalize_features": normalize_features,
                    "rolling_windows": rolling_windows,
                    "target_type": "Next Draw"
                },
                "feature_count": len(feature_names),
                "sequence_count": len(sequences_array),
                "feature_names": feature_names
            }
            
            app_log(f"Generated {len(sequences_array)} LSTM sequences with {len(feature_names)} features", "info")
            return sequences_array, metadata
            
        except Exception as e:
            app_log(f"Error generating LSTM sequences: {e}", "error")
            raise
    
    def generate_transformer_embeddings(self, raw_data: pd.DataFrame,
                                       window_size: int = 30,
                                       embedding_dim: int = 128,
                                       include_statistics: bool = True,
                                       normalize_features: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate Transformer embeddings from raw lottery data.
        
        Args:
            raw_data: DataFrame with lottery draw data
            window_size: Context window size
            embedding_dim: Embedding dimension
            include_statistics: Include statistical features
            normalize_features: Normalize features
            
        Returns:
            Tuple of (embeddings array, metadata dict)
        """
        try:
            raw_data = raw_data.copy()
            raw_data["numbers_list"] = raw_data["numbers"].apply(
                lambda x: list(map(int, str(x).split(",")))
            )
            
            # Create base features
            features_list = []
            for idx in range(len(raw_data)):
                draw_row = raw_data.iloc[idx]
                numbers = draw_row["numbers_list"]
                
                features = {
                    "sum": np.sum(numbers),
                    "mean": np.mean(numbers),
                    "std": np.std(numbers),
                    "min": np.min(numbers),
                    "max": np.max(numbers),
                    "range": np.max(numbers) - np.min(numbers),
                    "variance": np.var(numbers),
                }
                
                if include_statistics:
                    features["median"] = np.median(numbers)
                    features["q1"] = np.percentile(numbers, 25)
                    features["q3"] = np.percentile(numbers, 75)
                    features["iqr"] = features["q3"] - features["q1"]
                
                features_list.append(features)
            
            feature_names = sorted(features_list[0].keys())
            feature_array = np.array([
                [f.get(name, 0) for name in feature_names]
                for f in features_list
            ])
            
            if normalize_features:
                feature_array = (feature_array - feature_array.min(axis=0)) / (feature_array.max(axis=0) - feature_array.min(axis=0) + 1e-8)
            
            # Create embedding sequences
            embeddings = []
            for i in range(len(feature_array) - window_size):
                window = feature_array[i:i+window_size]
                # Project to embedding dimension using PCA-like approach
                # For now, use average pooling and linear projection simulation
                pooled = np.mean(window, axis=0)
                # Repeat to match embedding dimension
                if len(pooled) < embedding_dim:
                    embedding = np.concatenate([
                        pooled,
                        np.random.randn(embedding_dim - len(pooled)) * 0.1
                    ])
                else:
                    embedding = pooled[:embedding_dim]
                embeddings.append(embedding)
            
            embeddings_array = np.array(embeddings)
            
            metadata = {
                "model_type": "transformer",
                "game": self.game,
                "processing_mode": "all_files",
                "raw_files": [str(f) for f in self.get_raw_files()],
                "file_info": [
                    {
                        "file": str(f),
                        "draws_count": len(pd.read_csv(f))
                    }
                    for f in self.get_raw_files()
                ],
                "total_draws": len(raw_data),
                "consistent_draws": len(embeddings_array),
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "window": window_size,
                    "embedding_dim": embedding_dim,
                    "include_statistics": include_statistics,
                    "normalize_features": normalize_features
                },
                "embedding_count": len(embeddings_array),
                "embedding_dimension": embedding_dim
            }
            
            app_log(f"Generated {len(embeddings_array)} Transformer embeddings with dimension {embedding_dim}", "info")
            return embeddings_array, metadata
            
        except Exception as e:
            app_log(f"Error generating Transformer embeddings: {e}", "error")
            raise
    
    def generate_xgboost_features(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate advanced features for XGBoost models.
        
        Args:
            raw_data: DataFrame with lottery draw data
            
        Returns:
            Tuple of (features DataFrame, metadata dict)
        """
        try:
            raw_data = raw_data.copy()
            raw_data["numbers_list"] = raw_data["numbers"].apply(
                lambda x: list(map(int, str(x).split(",")))
            )
            
            features_df = pd.DataFrame()
            features_df["draw_date"] = raw_data["draw_date"]
            
            # Statistical features
            features_df["sum_numbers"] = raw_data["numbers_list"].apply(np.sum)
            features_df["mean_numbers"] = raw_data["numbers_list"].apply(np.mean)
            features_df["std_numbers"] = raw_data["numbers_list"].apply(np.std)
            features_df["min_number"] = raw_data["numbers_list"].apply(np.min)
            features_df["max_number"] = raw_data["numbers_list"].apply(np.max)
            features_df["range"] = features_df["max_number"] - features_df["min_number"]
            features_df["median_numbers"] = raw_data["numbers_list"].apply(np.median)
            
            # Distribution features
            features_df["skew"] = raw_data["numbers_list"].apply(lambda x: float(pd.Series(x).skew()))
            features_df["kurtosis"] = raw_data["numbers_list"].apply(lambda x: float(pd.Series(x).kurtosis()))
            
            # Spacing features
            def calculate_spacing(numbers):
                if len(numbers) < 2:
                    return 0
                spacing = np.diff(sorted(numbers))
                return np.mean(spacing)
            
            features_df["avg_spacing"] = raw_data["numbers_list"].apply(calculate_spacing)
            
            # Even/Odd count
            features_df["even_count"] = raw_data["numbers_list"].apply(lambda x: sum(1 for n in x if n % 2 == 0))
            features_df["odd_count"] = raw_data["numbers_list"].apply(lambda x: sum(1 for n in x if n % 2 == 1))
            
            # Low/High count (assuming 49 is max)
            features_df["low_count"] = raw_data["numbers_list"].apply(lambda x: sum(1 for n in x if n <= 24))
            features_df["high_count"] = raw_data["numbers_list"].apply(lambda x: sum(1 for n in x if n > 24))
            
            # Jackpot features
            features_df["jackpot"] = raw_data["jackpot"]
            features_df["jackpot_log"] = np.log1p(features_df["jackpot"])
            
            # Bonus features
            features_df["bonus"] = raw_data["bonus"]
            
            # Consecutive numbers
            def count_consecutive(numbers):
                sorted_nums = sorted(numbers)
                consecutive_count = 0
                for i in range(len(sorted_nums) - 1):
                    if sorted_nums[i+1] - sorted_nums[i] == 1:
                        consecutive_count += 1
                return consecutive_count
            
            features_df["consecutive_count"] = raw_data["numbers_list"].apply(count_consecutive)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features_df[f"rolling_sum_{window}"] = features_df["sum_numbers"].rolling(window=window, min_periods=1).mean()
                features_df[f"rolling_std_{window}"] = features_df["mean_numbers"].rolling(window=window, min_periods=1).std()
            
            # Fill NaN values
            features_df = features_df.fillna(0)
            
            metadata = {
                "model_type": "xgboost",
                "game": self.game,
                "processing_mode": "all_files",
                "raw_files": [str(f) for f in self.get_raw_files()],
                "file_info": [
                    {
                        "file": str(f),
                        "draws_count": len(pd.read_csv(f))
                    }
                    for f in self.get_raw_files()
                ],
                "total_draws": len(raw_data),
                "timestamp": datetime.now().isoformat(),
                "feature_count": len(features_df.columns) - 1,  # Exclude draw_date
                "feature_names": list(features_df.columns)
            }
            
            app_log(f"Generated XGBoost features with {metadata['feature_count']} features for {len(features_df)} draws", "info")
            return features_df, metadata
            
        except Exception as e:
            app_log(f"Error generating XGBoost features: {e}", "error")
            raise
    
    def save_lstm_sequences(self, sequences: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save LSTM sequences to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"all_files_advanced_seq_w{metadata['params']['window']}.npz"
            filepath = self.lstm_dir / filename
            
            np.savez_compressed(filepath, sequences=sequences)
            
            # Save metadata
            meta_filepath = self.lstm_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata, f, indent=2)
            
            app_log(f"Saved LSTM sequences to {filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving LSTM sequences: {e}", "error")
            return False
    
    def save_transformer_embeddings(self, embeddings: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Save Transformer embeddings to disk."""
        try:
            filename = f"all_files_advanced_embed_w{metadata['params']['window']}_e{metadata['params']['embedding_dim']}.npz"
            filepath = self.transformer_dir / filename
            
            np.savez_compressed(filepath, embeddings=embeddings)
            
            # Save metadata
            meta_filepath = self.transformer_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata, f, indent=2)
            
            app_log(f"Saved Transformer embeddings to {filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving Transformer embeddings: {e}", "error")
            return False
    
    def save_xgboost_features(self, features_df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
        """Save XGBoost features to disk."""
        try:
            filename = "all_files_advanced_features.csv"
            filepath = self.xgboost_dir / filename
            
            features_df.to_csv(filepath, index=False)
            
            # Save metadata
            meta_filepath = self.xgboost_dir / f"{filename}.meta.json"
            with open(meta_filepath, "w") as f:
                json.dump(metadata, f, indent=2)
            
            app_log(f"Saved XGBoost features to {filepath}", "info")
            return True
        except Exception as e:
            app_log(f"Error saving XGBoost features: {e}", "error")
            return False
