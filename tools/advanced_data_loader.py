"""
Advanced Data Loader and Preprocessor for Lottery Games
Loads historical lottery data and prepares it for advanced feature engineering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class LotteryDataLoader:
    """Load and validate historical lottery data"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir) if data_dir else Path('data')
    
    def load_game_data(self, game_name: str) -> pd.DataFrame:
        """
        Load historical draws for a specific game
        Combines all yearly training data files
        
        Args:
            game_name: Either 'lotto_6_49' or 'lotto_max'
            
        Returns:
            DataFrame with columns: ['draw_id', 'date', 'numbers']
        """
        game_dir = self.data_dir / game_name
        
        # Debug logging
        logger.info(f"Looking for game data in: {game_dir.absolute()}")
        logger.info(f"Data dir exists: {self.data_dir.exists()}")
        logger.info(f"Game dir exists: {game_dir.exists()}")
        
        if not game_dir.exists():
            # Try to list what's in data_dir
            try:
                contents = list(self.data_dir.iterdir())
                logger.error(f"Data directory contents: {[d.name for d in contents]}")
            except:
                pass
            raise ValueError(f"Game directory not found: {game_dir}")
        
        # Find all training data files
        training_files = sorted(game_dir.glob('training_data_*.csv'))
        
        if not training_files:
            raise FileNotFoundError(f"No training data files found in {game_dir}")
        
        # Load and combine all files
        dfs = []
        for file_path in training_files:
            df_year = pd.read_csv(file_path)
            dfs.append(df_year)
            logger.info(f"Loaded {len(df_year)} draws from {file_path.name}")
        
        df = pd.concat(dfs, ignore_index=True)
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Parse numbers column - handle various formats
        if 'numbers' in df.columns:
            df['numbers'] = df['numbers'].apply(self._parse_numbers)
        elif 'ball_1' in df.columns:
            # If data has individual ball columns, combine them
            ball_cols = [col for col in df.columns if col.startswith('ball_')]
            df['numbers'] = df[sorted(ball_cols)].apply(
                lambda row: tuple(int(x) for x in row if pd.notna(x)),
                axis=1
            )
        else:
            raise ValueError("Cannot identify numbers column in data")
        
        # Parse date column
        if 'date' in df.columns or 'draw_date' in df.columns:
            date_col = 'date' if 'date' in df.columns else 'draw_date'
            df['date'] = pd.to_datetime(df[date_col])
        else:
            # If no date, create sequential dates
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        
        # Create draw_id if it doesn't exist
        if 'draw_id' not in df.columns:
            df['draw_id'] = range(len(df))
        
        # Select relevant columns
        df = df[['draw_id', 'date', 'numbers']].copy()
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} draws for {game_name}")
        return df
    
    @staticmethod
    def _parse_numbers(val) -> Tuple[int, ...]:
        """Parse numbers from various formats"""
        if isinstance(val, (list, tuple)):
            return tuple(sorted([int(x) for x in val]))
        elif isinstance(val, str):
            # Try comma-separated
            if ',' in val:
                return tuple(sorted([int(x.strip()) for x in val.split(',')]))
            # Try space-separated
            elif ' ' in val:
                return tuple(sorted([int(x.strip()) for x in val.split()]))
            # Try single string of digits
            else:
                return tuple(sorted([int(x) for x in val.strip()]))
        else:
            raise ValueError(f"Cannot parse numbers: {val}")
    
    def validate_draws(self, df: pd.DataFrame, game_name: str) -> bool:
        """
        Validate that draws conform to game rules
        
        Args:
            df: DataFrame with draws
            game_name: Game identifier
            
        Returns:
            True if valid, raises exception otherwise
        """
        if game_name == 'lotto_6_49':
            expected_balls = 6
            max_number = 49
        elif game_name == 'lotto_max':
            expected_balls = 7
            max_number = 50
        else:
            raise ValueError(f"Unknown game: {game_name}")
        
        invalid_indices = []
        for idx, row in df.iterrows():
            numbers = row['numbers']
            
            # Check number of balls - LENIENT: allow both 6 and 7 for Lotto Max during transition
            if game_name == 'lotto_6_49' and len(numbers) != 6:
                invalid_indices.append(idx)
            elif game_name == 'lotto_max' and len(numbers) not in [6, 7]:
                invalid_indices.append(idx)
            
            # Check number range
            elif any(n < 1 or n > max_number for n in numbers):
                invalid_indices.append(idx)
            
            # Check for duplicates
            elif len(set(numbers)) != len(numbers):
                invalid_indices.append(idx)
        
        if invalid_indices:
            logger.warning(f"Found {len(invalid_indices)} invalid draws, removing them")
            df = df.drop(invalid_indices).reset_index(drop=True)
        
        logger.info(f"Validation complete for {len(df)} draws")
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get basic statistics about the draws
        
        Args:
            df: DataFrame with draws
            
        Returns:
            Dictionary with statistics
        """
        all_numbers = []
        for numbers in df['numbers']:
            all_numbers.extend(numbers)
        
        all_numbers = np.array(all_numbers)
        
        return {
            'total_draws': len(df),
            'date_range': (df['date'].min(), df['date'].max()),
            'avg_number_frequency': len(all_numbers) / (len(df) * len(df.iloc[0]['numbers'])),
            'most_common_numbers': self._get_most_common(all_numbers, k=10),
            'least_common_numbers': self._get_least_common(all_numbers, k=10)
        }
    
    @staticmethod
    def _get_most_common(numbers: np.ndarray, k: int = 10) -> List[Tuple[int, int]]:
        """Get k most common numbers with frequencies"""
        unique, counts = np.unique(numbers, return_counts=True)
        return sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:k]
    
    @staticmethod
    def _get_least_common(numbers: np.ndarray, k: int = 10) -> List[Tuple[int, int]]:
        """Get k least common numbers with frequencies"""
        unique, counts = np.unique(numbers, return_counts=True)
        return sorted(zip(unique, counts), key=lambda x: x[1])[:k]


class DataPreprocessor:
    """Preprocess and clean lottery data"""
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate draws"""
        df = df.drop_duplicates(subset=['numbers'])
        logger.info(f"After removing duplicates: {len(df)} draws")
        return df
    
    @staticmethod
    def filter_date_range(
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter draws by date range"""
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        logger.info(f"After date filtering: {len(df)} draws")
        return df
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle any missing values in the dataset"""
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values:\n{null_counts}")
            df = df.dropna()
        
        return df
    
    @staticmethod
    def normalize_number_ranges(
        df: pd.DataFrame,
        game_name: str
    ) -> pd.DataFrame:
        """Ensure numbers are in valid range for the game"""
        if game_name == 'lotto_6_49':
            max_number = 49
        elif game_name == 'lotto_max':
            max_number = 50
        else:
            raise ValueError(f"Unknown game: {game_name}")
        
        def validate_and_fix(numbers):
            valid = tuple(n for n in numbers if 1 <= n <= max_number)
            if len(valid) != len(numbers):
                logger.warning(f"Invalid numbers removed: {set(numbers) - set(valid)}")
            return valid
        
        df['numbers'] = df['numbers'].apply(validate_and_fix)
        df = df[df['numbers'].apply(len) > 0]  # Remove empty draws
        
        return df


def prepare_game_dataset(
    game_name: str,
    data_dir: Path = None,
    remove_dups: bool = True,
    min_draws: int = 100
) -> pd.DataFrame:
    """
    End-to-end dataset preparation
    
    Args:
        game_name: Game identifier
        data_dir: Data directory path
        remove_dups: Whether to remove duplicate draws
        min_draws: Minimum number of draws required
        
    Returns:
        Prepared DataFrame
    """
    loader = LotteryDataLoader(data_dir)
    preprocessor = DataPreprocessor()
    
    # Load data
    df = loader.load_game_data(game_name)
    
    # Validate and clean (returns cleaned DataFrame)
    df = loader.validate_draws(df, game_name)
    
    # Preprocess
    if remove_dups:
        df = preprocessor.remove_duplicates(df)
    
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.normalize_number_ranges(df, game_name)
    
    # Check minimum draws
    if len(df) < min_draws:
        raise ValueError(f"Not enough draws: {len(df)} < {min_draws}")
    
    # Log statistics
    stats = loader.get_statistics(df)
    logger.info(f"Dataset statistics for {game_name}:")
    logger.info(f"  Total draws: {stats['total_draws']}")
    logger.info(f"  Date range: {stats['date_range']}")
    logger.info(f"  Most common: {stats['most_common_numbers'][:5]}")
    
    return df
