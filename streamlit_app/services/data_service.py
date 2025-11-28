"""
Enhanced Data Service - Business Logic Extracted from Monolithic App

This module provides comprehensive data management services combining:
1. Original DataManager and StatisticsManager functionality
2. Extracted business logic from monolithic app.py (data loading, statistics)
3. Enhanced service foundation with BaseService integration

Extracted Functions Integrated:
- load_historical_data() -> load_historical_data()
- get_latest_draw() -> get_latest_draw()
- calculate_game_stats() -> calculate_statistics()

Enhanced Features:
- BaseService integration with dependency injection
- Comprehensive data validation and integrity checking
- Enhanced statistical analysis and reporting
- Flexible data loading from multiple sources
- Clean separation from UI dependencies
"""

import pandas as pd
import numpy as np
import json
import pickle
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import os
import hashlib
from pathlib import Path

# Phase 2 Service Integration
from .base_service import BaseService, ServiceValidationMixin
from ..core.exceptions import DataError, ValidationError, safe_execute
from ..core.utils import sanitize_game_name

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data loading, saving, and basic operations.
    
    This class handles all data I/O operations including loading from various
    sources, saving data in different formats, and maintaining data integrity.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.db_path = self.data_dir / 'lottery_data.db'
        self.ensure_data_directory()
        self.init_database()
    
    def ensure_data_directory(self) -> None:
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Data directory: {self.data_dir}")
    
    def init_database(self) -> None:
        """Initialize SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create drawings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS drawings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_type TEXT NOT NULL,
                        draw_date DATE NOT NULL,
                        numbers TEXT NOT NULL,
                        bonus_numbers TEXT,
                        jackpot_amount REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(game_type, draw_date)
                    )
                ''')
                
                # Create predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        game_type TEXT NOT NULL,
                        numbers TEXT NOT NULL,
                        confidence REAL,
                        strategy TEXT,
                        model_name TEXT,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        result TEXT DEFAULT 'pending'
                    )
                ''')
                
                # Create statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stat_type TEXT NOT NULL,
                        stat_data TEXT NOT NULL,
                        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def load_drawing_data(self, game_type: str = None, 
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load drawing data from database.
        
        Args:
            game_type: Specific game type to load (None for all)
            limit: Maximum number of records to load
            
        Returns:
            DataFrame with drawing data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM drawings"
                params = []
                
                if game_type:
                    query += " WHERE game_type = ?"
                    params.append(game_type)
                
                query += " ORDER BY draw_date DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    # Parse numbers from JSON strings
                    df['numbers'] = df['numbers'].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else x
                    )
                    
                    if 'bonus_numbers' in df.columns:
                        df['bonus_numbers'] = df['bonus_numbers'].apply(
                            lambda x: json.loads(x) if isinstance(x, str) and x else []
                        )
                    
                    # Convert date column
                    df['draw_date'] = pd.to_datetime(df['draw_date'])
                
                logger.info(f"📊 Loaded {len(df)} drawing records")
                return df
                
        except Exception as e:
            logger.error(f"❌ Failed to load drawing data: {e}")
            return pd.DataFrame()
    
    def save_drawing_data(self, data: Union[pd.DataFrame, Dict[str, Any], 
                                          List[Dict[str, Any]]]) -> bool:
        """
        Save drawing data to database.
        
        Args:
            data: Drawing data to save
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if isinstance(data, pd.DataFrame):
                    # Convert DataFrame to records
                    records = data.to_dict('records')
                elif isinstance(data, dict):
                    records = [data]
                else:
                    records = data
                
                for record in records:
                    # Convert numbers to JSON strings
                    if 'numbers' in record and isinstance(record['numbers'], list):
                        record['numbers'] = json.dumps(record['numbers'])
                    
                    if 'bonus_numbers' in record and isinstance(record['bonus_numbers'], list):
                        record['bonus_numbers'] = json.dumps(record['bonus_numbers'])
                    
                    # Insert with conflict resolution
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO drawings 
                        (game_type, draw_date, numbers, bonus_numbers, jackpot_amount)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        record.get('game_type'),
                        record.get('draw_date'),
                        record.get('numbers'),
                        record.get('bonus_numbers'),
                        record.get('jackpot_amount')
                    ))
                
                conn.commit()
                logger.info(f"✅ Saved {len(records)} drawing records")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save drawing data: {e}")
            return False
    
    def load_data_from_file(self, file_path: str, 
                           file_format: str = 'auto') -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
            file_format: File format ('auto', 'csv', 'json', 'excel')
            
        Returns:
            DataFrame with loaded data
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return pd.DataFrame()
            
            # Auto-detect format
            if file_format == 'auto':
                file_format = file_path.suffix.lower()[1:]  # Remove dot
            
            # Load based on format
            if file_format == 'csv':
                df = pd.read_csv(file_path)
            elif file_format == 'json':
                df = pd.read_json(file_path)
            elif file_format in ['xlsx', 'xls', 'excel']:
                df = pd.read_excel(file_path)
            elif file_format == 'pickle':
                df = pd.read_pickle(file_path)
            else:
                logger.error(f"❌ Unsupported file format: {file_format}")
                return pd.DataFrame()
            
            logger.info(f"📁 Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from file: {e}")
            return pd.DataFrame()
    
    def save_data_to_file(self, data: pd.DataFrame, file_path: str, 
                         file_format: str = 'auto') -> bool:
        """
        Save data to file.
        
        Args:
            data: DataFrame to save
            file_path: Path to save file
            file_format: File format ('auto', 'csv', 'json', 'excel')
            
        Returns:
            Success status
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Auto-detect format
            if file_format == 'auto':
                file_format = file_path.suffix.lower()[1:]
            
            # Save based on format
            if file_format == 'csv':
                data.to_csv(file_path, index=False)
            elif file_format == 'json':
                data.to_json(file_path, orient='records', indent=2)
            elif file_format in ['xlsx', 'excel']:
                data.to_excel(file_path, index=False)
            elif file_format == 'pickle':
                data.to_pickle(file_path)
            else:
                logger.error(f"❌ Unsupported file format: {file_format}")
                return False
            
            logger.info(f"💾 Saved {len(data)} records to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save data to file: {e}")
            return False
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            data: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            cleaned_data = data.copy()
            
            # Remove duplicates
            initial_count = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            
            if len(cleaned_data) < initial_count:
                logger.info(f"🧹 Removed {initial_count - len(cleaned_data)} duplicate records")
            
            # Handle missing values
            if cleaned_data.isnull().any().any():
                # Fill missing dates with interpolation
                if 'draw_date' in cleaned_data.columns:
                    cleaned_data['draw_date'] = pd.to_datetime(cleaned_data['draw_date'])
                    cleaned_data = cleaned_data.dropna(subset=['draw_date'])
                
                # Remove records with missing numbers
                if 'numbers' in cleaned_data.columns:
                    cleaned_data = cleaned_data.dropna(subset=['numbers'])
                
                logger.info("🧹 Handled missing values")
            
            # Validate number ranges (example for typical lottery)
            if 'numbers' in cleaned_data.columns:
                def validate_numbers(numbers):
                    if isinstance(numbers, list):
                        return all(1 <= num <= 70 for num in numbers)  # Example range
                    return False
                
                valid_mask = cleaned_data['numbers'].apply(validate_numbers)
                invalid_count = (~valid_mask).sum()
                
                if invalid_count > 0:
                    logger.warning(f"⚠️ Found {invalid_count} records with invalid numbers")
                    cleaned_data = cleaned_data[valid_mask]
            
            logger.info(f"✅ Data cleaning complete. Final records: {len(cleaned_data)}")
            return cleaned_data
            
        except Exception as e:
            logger.error(f"❌ Failed to clean data: {e}")
            return data
    
    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate data quality report.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Quality report dictionary
        """
        try:
            report = {
                'total_records': len(data),
                'columns': list(data.columns),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.astype(str).to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'duplicates': data.duplicated().sum()
            }
            
            # Date range analysis
            if 'draw_date' in data.columns:
                date_col = pd.to_datetime(data['draw_date'], errors='coerce')
                report['date_range'] = {
                    'earliest': str(date_col.min()),
                    'latest': str(date_col.max()),
                    'span_days': (date_col.max() - date_col.min()).days
                }
            
            # Number analysis
            if 'numbers' in data.columns:
                all_numbers = []
                for numbers in data['numbers']:
                    if isinstance(numbers, list):
                        all_numbers.extend(numbers)
                
                if all_numbers:
                    report['numbers_analysis'] = {
                        'total_numbers': len(all_numbers),
                        'unique_numbers': len(set(all_numbers)),
                        'min_number': min(all_numbers),
                        'max_number': max(all_numbers),
                        'avg_number': np.mean(all_numbers)
                    }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate quality report: {e}")
            return {}
    
    def backup_data(self, backup_name: Optional[str] = None) -> bool:
        """
        Create data backup.
        
        Args:
            backup_name: Custom backup name
            
        Returns:
            Success status
        """
        try:
            if backup_name is None:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_dir = self.data_dir / 'backups'
            backup_dir.mkdir(exist_ok=True)
            
            backup_path = backup_dir / f"{backup_name}.db"
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"💾 Created backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return False
    
    @staticmethod
    def health_check() -> bool:
        """Check data manager health."""
        return True


class HistoryManager:
    """
    Manages historical data and prediction history.
    
    This class handles storage and retrieval of prediction history,
    performance tracking, and historical analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize history manager."""
        self.config = config or {}
        self.data_manager = DataManager(config)
        self.max_history_days = self.config.get('max_history_days', 365)
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """
        Save prediction to history.
        
        Args:
            prediction_data: Prediction data to save
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert numbers to JSON string
                numbers_json = json.dumps(prediction_data.get('numbers', []))
                
                cursor.execute('''
                    INSERT INTO predictions 
                    (game_type, numbers, confidence, strategy, model_name)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    prediction_data.get('game_type'),
                    numbers_json,
                    prediction_data.get('confidence'),
                    prediction_data.get('strategy'),
                    prediction_data.get('model_name')
                ))
                
                conn.commit()
                logger.info("✅ Saved prediction to history")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save prediction: {e}")
            return False
    
    def load_prediction_history(self, game_type: str = None, 
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load prediction history.
        
        Args:
            game_type: Specific game type (None for all)
            limit: Maximum number of records
            
        Returns:
            List of prediction records
        """
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                query = "SELECT * FROM predictions"
                params = []
                
                if game_type:
                    query += " WHERE game_type = ?"
                    params.append(game_type)
                
                query += " ORDER BY generated_at DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    # Parse numbers from JSON
                    df['numbers'] = df['numbers'].apply(json.loads)
                    df['generated_at'] = pd.to_datetime(df['generated_at'])
                    
                    return df.to_dict('records')
                
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to load prediction history: {e}")
            return []
    
    def update_prediction_result(self, prediction_id: int, 
                                result: str, matches: Optional[List[int]] = None) -> bool:
        """
        Update prediction result after drawing.
        
        Args:
            prediction_id: Prediction ID
            result: Result status ('win', 'partial', 'loss')
            matches: List of matching numbers
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                cursor = conn.cursor()
                
                result_data = {
                    'result': result,
                    'matches': matches or [],
                    'checked_at': datetime.now().isoformat()
                }
                
                cursor.execute('''
                    UPDATE predictions 
                    SET result = ? 
                    WHERE id = ?
                ''', (json.dumps(result_data), prediction_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Updated prediction {prediction_id} result: {result}")
                    return True
                else:
                    logger.warning(f"⚠️ Prediction {prediction_id} not found")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Failed to update prediction result: {e}")
            return False
    
    def get_performance_metrics(self, game_type: str = None, 
                               days: int = 30) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            game_type: Specific game type
            days: Number of days to analyze
            
        Returns:
            Performance metrics dictionary
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.data_manager.db_path) as conn:
                query = """
                    SELECT * FROM predictions 
                    WHERE generated_at >= ?
                """
                params = [cutoff_date.isoformat()]
                
                if game_type:
                    query += " AND game_type = ?"
                    params.append(game_type)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return {'total_predictions': 0}
                
                # Parse results
                results = []
                for result_str in df['result']:
                    try:
                        if result_str and result_str != 'pending':
                            result_data = json.loads(result_str)
                            results.append(result_data.get('result', 'pending'))
                        else:
                            results.append('pending')
                    except:
                        results.append('pending')
                
                df['parsed_result'] = results
                
                # Calculate metrics
                total = len(df)
                wins = (df['parsed_result'] == 'win').sum()
                partials = (df['parsed_result'] == 'partial').sum()
                losses = (df['parsed_result'] == 'loss').sum()
                pending = (df['parsed_result'] == 'pending').sum()
                
                metrics = {
                    'total_predictions': total,
                    'wins': wins,
                    'partial_wins': partials,
                    'losses': losses,
                    'pending': pending,
                    'win_rate': wins / max(total - pending, 1),
                    'success_rate': (wins + partials) / max(total - pending, 1),
                    'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0
                }
                
                # Strategy breakdown
                if 'strategy' in df.columns:
                    strategy_counts = df['strategy'].value_counts().to_dict()
                    metrics['strategy_usage'] = strategy_counts
                
                # Model breakdown
                if 'model_name' in df.columns:
                    model_counts = df['model_name'].value_counts().to_dict()
                    metrics['model_usage'] = model_counts
                
                return metrics
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return {}
    
    def cleanup_old_history(self) -> bool:
        """
        Remove old history records beyond retention period.
        
        Returns:
            Success status
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            
            with sqlite3.connect(self.data_manager.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM predictions 
                    WHERE generated_at < ?
                ''', (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {deleted_count} old prediction records")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old history: {e}")
            return False
    
    @staticmethod
    def health_check() -> bool:
        """Check history manager health."""
        return True


class StatisticsManager:
    """
    Manages statistical calculations and analysis.
    
    This class handles number frequency analysis, trend detection,
    pattern recognition, and other statistical operations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize statistics manager."""
        self.config = config or {}
        self.data_manager = DataManager(config)
    
    def calculate_number_frequencies(self, game_type: str, 
                                   days: Optional[int] = None) -> Dict[int, int]:
        """
        Calculate number frequency statistics.
        
        Args:
            game_type: Game type to analyze
            days: Number of recent days to analyze (None for all)
            
        Returns:
            Dictionary mapping numbers to frequencies
        """
        try:
            # Load drawing data
            df = self.data_manager.load_drawing_data(game_type)
            
            if df.empty:
                return {}
            
            # Filter by date if specified
            if days:
                cutoff_date = datetime.now() - timedelta(days=days)
                df = df[df['draw_date'] >= cutoff_date]
            
            # Count number frequencies
            frequencies = {}
            for numbers in df['numbers']:
                if isinstance(numbers, list):
                    for num in numbers:
                        frequencies[num] = frequencies.get(num, 0) + 1
            
            logger.info(f"📊 Calculated frequencies for {len(frequencies)} numbers")
            return frequencies
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate number frequencies: {e}")
            return {}
    
    def analyze_hot_cold_numbers(self, game_type: str, 
                                days: int = 90) -> Dict[str, List[int]]:
        """
        Analyze hot and cold numbers.
        
        Args:
            game_type: Game type to analyze
            days: Number of recent days to analyze
            
        Returns:
            Dictionary with hot and cold numbers
        """
        try:
            frequencies = self.calculate_number_frequencies(game_type, days)
            
            if not frequencies:
                return {'hot': [], 'cold': []}
            
            # Sort by frequency
            sorted_numbers = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
            
            total_numbers = len(sorted_numbers)
            hot_cutoff = max(1, total_numbers // 4)  # Top 25%
            cold_cutoff = max(1, total_numbers // 4)  # Bottom 25%
            
            hot_numbers = [num for num, freq in sorted_numbers[:hot_cutoff]]
            cold_numbers = [num for num, freq in sorted_numbers[-cold_cutoff:]]
            
            logger.info(f"🔥 Found {len(hot_numbers)} hot and {len(cold_numbers)} cold numbers")
            
            return {
                'hot': hot_numbers,
                'cold': cold_numbers,
                'frequencies': frequencies
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze hot/cold numbers: {e}")
            return {'hot': [], 'cold': []}
    
    def detect_patterns(self, game_type: str, 
                       pattern_type: str = 'consecutive') -> Dict[str, Any]:
        """
        Detect number patterns in historical data.
        
        Args:
            game_type: Game type to analyze
            pattern_type: Type of pattern to detect
            
        Returns:
            Pattern analysis results
        """
        try:
            df = self.data_manager.load_drawing_data(game_type)
            
            if df.empty:
                return {}
            
            patterns = {'type': pattern_type, 'findings': []}
            
            if pattern_type == 'consecutive':
                # Find consecutive number patterns
                consecutive_counts = {}
                
                for numbers in df['numbers']:
                    if isinstance(numbers, list):
                        sorted_nums = sorted(numbers)
                        consecutive = 0
                        
                        for i in range(len(sorted_nums) - 1):
                            if sorted_nums[i + 1] == sorted_nums[i] + 1:
                                consecutive += 1
                        
                        consecutive_counts[consecutive] = consecutive_counts.get(consecutive, 0) + 1
                
                patterns['findings'] = consecutive_counts
                
            elif pattern_type == 'sum_range':
                # Analyze sum ranges
                sums = []
                for numbers in df['numbers']:
                    if isinstance(numbers, list):
                        sums.append(sum(numbers))
                
                if sums:
                    patterns['findings'] = {
                        'min_sum': min(sums),
                        'max_sum': max(sums),
                        'avg_sum': np.mean(sums),
                        'std_sum': np.std(sums)
                    }
            
            elif pattern_type == 'odd_even':
                # Analyze odd/even distribution
                odd_even_counts = {}
                
                for numbers in df['numbers']:
                    if isinstance(numbers, list):
                        odd_count = sum(1 for num in numbers if num % 2 == 1)
                        even_count = len(numbers) - odd_count
                        
                        pattern_key = f"{odd_count}odd_{even_count}even"
                        odd_even_counts[pattern_key] = odd_even_counts.get(pattern_key, 0) + 1
                
                patterns['findings'] = odd_even_counts
            
            logger.info(f"🔍 Detected {pattern_type} patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Failed to detect patterns: {e}")
            return {}
    
    def calculate_overdue_numbers(self, game_type: str, 
                                 max_number: int = 70) -> Dict[str, Any]:
        """
        Calculate overdue numbers (haven't appeared recently).
        
        Args:
            game_type: Game type to analyze
            max_number: Maximum number in the game
            
        Returns:
            Overdue analysis results
        """
        try:
            df = self.data_manager.load_drawing_data(game_type)
            
            if df.empty:
                return {}
            
            # Sort by date (most recent first)
            df = df.sort_values('draw_date', ascending=False)
            
            # Track last appearance of each number
            last_appearance = {}
            all_numbers = set(range(1, max_number + 1))
            
            for idx, row in df.iterrows():
                if isinstance(row['numbers'], list):
                    for num in row['numbers']:
                        if num not in last_appearance:
                            last_appearance[num] = idx
            
            # Find numbers that haven't appeared
            missing_numbers = all_numbers - set(last_appearance.keys())
            
            # Calculate overdue periods
            overdue_data = {}
            for num, last_idx in last_appearance.items():
                overdue_data[num] = last_idx  # Number of drawings since last appearance
            
            # Sort by overdue period
            sorted_overdue = sorted(overdue_data.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'overdue_numbers': dict(sorted_overdue[:10]),  # Top 10 overdue
                'missing_numbers': list(missing_numbers),
                'total_drawings': len(df)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate overdue numbers: {e}")
            return {}
    
    def save_statistics(self, stat_type: str, stat_data: Dict[str, Any]) -> bool:
        """
        Save calculated statistics.
        
        Args:
            stat_type: Type of statistics
            stat_data: Statistics data
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO statistics (stat_type, stat_data)
                    VALUES (?, ?)
                ''', (stat_type, json.dumps(stat_data)))
                
                conn.commit()
                logger.info(f"✅ Saved {stat_type} statistics")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to save statistics: {e}")
            return False
    
    def load_statistics(self, stat_type: str) -> Optional[Dict[str, Any]]:
        """
        Load saved statistics.
        
        Args:
            stat_type: Type of statistics to load
            
        Returns:
            Statistics data or None
        """
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT stat_data FROM statistics 
                    WHERE stat_type = ? 
                    ORDER BY calculated_at DESC 
                    LIMIT 1
                ''', (stat_type,))
                
                result = cursor.fetchone()
                
                if result:
                    return json.loads(result[0])
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load statistics: {e}")
            return None
    
    @staticmethod
    def health_check() -> bool:
        """Check statistics manager health."""
        return True
    
    # =============================================================================
    # EXTRACTED BUSINESS LOGIC FROM MONOLITHIC APP.PY
    # =============================================================================
    
    def load_historical_data_from_files(self, game_name: str, limit: int = 1000) -> pd.DataFrame:
        """
        Load historical draw data for a game from CSV files.
        
        Extracted from: load_historical_data() in original app.py (Line 268)
        Enhanced with: Better error handling, flexible date parsing, data validation
        
        Args:
            game_name: Name of the lottery game
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with historical data
        """
        try:
            from ..core.utils import sanitize_game_name
            
            game_key = sanitize_game_name(game_name)
            game_dir = self.data_dir / game_key
            all_data = []
            
            if not game_dir.exists():
                logger.info(f"📁 No data directory found for {game_name}")
                return pd.DataFrame()
            
            # Process all CSV files in game directory
            csv_files = sorted(game_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
                    logger.debug(f"📊 Loaded {len(df)} records from {csv_file}")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {csv_file}: {e}")
                    continue
            
            if not all_data:
                logger.info(f"📊 No valid CSV data found for {game_name}")
                return pd.DataFrame()
            
            # Combine all DataFrames
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by draw_date if available
            if 'draw_date' in combined_df.columns:
                try:
                    # Use flexible date parsing to handle different formats
                    combined_df['draw_date'] = pd.to_datetime(
                        combined_df['draw_date'], 
                        format='mixed', 
                        errors='coerce'
                    )
                    
                    # Remove any rows where date parsing failed
                    initial_count = len(combined_df)
                    combined_df = combined_df.dropna(subset=['draw_date'])
                    
                    if len(combined_df) < initial_count:
                        logger.warning(f"⚠️ Removed {initial_count - len(combined_df)} rows with invalid dates")
                    
                    # Sort by date (most recent first)
                    combined_df = combined_df.sort_values('draw_date', ascending=False)
                    
                except Exception as e:
                    logger.error(f"❌ Error parsing dates in historical data: {e}")
                    # Continue without date sorting
            
            # Apply limit
            result_df = combined_df.head(limit)
            
            logger.info(f"✅ Loaded {len(result_df)} historical records for {game_name}")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical data for {game_name}: {e}")
            return pd.DataFrame()
    
    def get_latest_draw(self, game_name: str) -> Dict[str, Any]:
        """
        Get the most recent draw for a game.
        
        Extracted from: get_latest_draw() in original app.py (Line 298)
        Enhanced with: Type validation, comprehensive error handling
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            Dictionary with latest draw information
        """
        try:
            df = self.load_historical_data_from_files(game_name, limit=1)
            
            if df.empty:
                logger.info(f"📊 No historical data found for {game_name}")
                return {}
            
            row = df.iloc[0]
            
            # Extract draw information with safe access
            draw_info = {
                'draw_date': self._safe_get_value(row, 'draw_date'),
                'numbers': self._safe_get_value(row, 'numbers', ''),
                'bonus': self._safe_get_value(row, 'bonus'),
                'jackpot': self._safe_get_value(row, 'jackpot')
            }
            
            # Convert date to string if it's a pandas timestamp
            if hasattr(draw_info['draw_date'], 'isoformat'):
                draw_info['draw_date'] = draw_info['draw_date'].isoformat()
            
            logger.info(f"✅ Retrieved latest draw for {game_name}: {draw_info['draw_date']}")
            return draw_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest draw for {game_name}: {e}")
            return {}
    
    def calculate_game_statistics(self, game_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for a game.
        
        Extracted from: calculate_game_stats() in original app.py (Line 785)
        Enhanced with: More comprehensive statistics, better error handling
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            Dictionary with game statistics
        """
        try:
            df = self.load_historical_data_from_files(game_name)
            
            if df.empty:
                logger.info(f"📊 No data available for statistics calculation: {game_name}")
                return {
                    'total_draws': 0,
                    'avg_jackpot': 0,
                    'last_jackpot': 0,
                    'most_frequent_numbers': [],
                    'data_range': None,
                    'completeness': 0.0
                }
            
            # Basic statistics
            stats = {
                'total_draws': len(df),
                'data_range': self._calculate_date_range(df),
                'completeness': self._calculate_data_completeness(df)
            }
            
            # Jackpot statistics
            if 'jackpot' in df.columns:
                jackpot_series = pd.to_numeric(df['jackpot'], errors='coerce')
                valid_jackpots = jackpot_series.dropna()
                
                stats.update({
                    'avg_jackpot': float(valid_jackpots.mean()) if len(valid_jackpots) > 0 else 0,
                    'last_jackpot': float(valid_jackpots.iloc[0]) if len(valid_jackpots) > 0 else 0,
                    'max_jackpot': float(valid_jackpots.max()) if len(valid_jackpots) > 0 else 0,
                    'min_jackpot': float(valid_jackpots.min()) if len(valid_jackpots) > 0 else 0
                })
            else:
                stats.update({
                    'avg_jackpot': 0,
                    'last_jackpot': 0,
                    'max_jackpot': 0,
                    'min_jackpot': 0
                })
            
            # Number frequency analysis
            stats['most_frequent_numbers'] = self._calculate_number_frequency(df)
            
            # Additional statistics
            stats.update({
                'recent_draws_30_days': self._count_recent_draws(df, days=30),
                'recent_draws_90_days': self._count_recent_draws(df, days=90),
                'draw_frequency': self._calculate_draw_frequency(df)
            })
            
            logger.info(f"✅ Calculated statistics for {game_name}: {stats['total_draws']} draws")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate statistics for {game_name}: {e}")
            return {}
    
    def _safe_get_value(self, row: pd.Series, column: str, default: Any = None) -> Any:
        """Safely get value from pandas Series."""
        try:
            if column in row.index:
                value = row[column]
                # Handle pandas NaN/None values
                if pd.isna(value):
                    return default
                return value
            return default
        except Exception:
            return default
    
    def _calculate_date_range(self, df: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Calculate date range from DataFrame."""
        try:
            if 'draw_date' not in df.columns:
                return None
            
            dates = pd.to_datetime(df['draw_date'], errors='coerce').dropna()
            if dates.empty:
                return None
            
            return {
                'earliest': dates.min().isoformat(),
                'latest': dates.max().isoformat(),
                'span_days': (dates.max() - dates.min()).days
            }
        except Exception:
            return None
    
    def _calculate_data_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness percentage."""
        try:
            if df.empty:
                return 0.0
            
            # Calculate percentage of non-null values across all columns
            total_cells = df.size
            non_null_cells = df.notna().sum().sum()
            
            return (non_null_cells / total_cells) * 100 if total_cells > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_number_frequency(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """Calculate most frequent numbers from historical data."""
        try:
            if 'numbers' not in df.columns:
                return []
            
            all_numbers = []
            
            for numbers_str in df['numbers'].dropna():
                try:
                    # Parse numbers from string (comma-separated)
                    numbers = [int(x.strip()) for x in str(numbers_str).split(',')]
                    all_numbers.extend(numbers)
                except (ValueError, AttributeError):
                    continue
            
            if not all_numbers:
                return []
            
            # Count frequency and return top 10
            from collections import Counter
            frequency = Counter(all_numbers)
            return frequency.most_common(10)
            
        except Exception as e:
            logger.error(f"❌ Error calculating number frequency: {e}")
            return []
    
    def _count_recent_draws(self, df: pd.DataFrame, days: int) -> int:
        """Count draws within specified number of days."""
        try:
            if 'draw_date' not in df.columns:
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=days)
            dates = pd.to_datetime(df['draw_date'], errors='coerce').dropna()
            
            recent_draws = dates[dates > cutoff_date]
            return len(recent_draws)
            
        except Exception:
            return 0
    
    def _calculate_draw_frequency(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate average frequency of draws."""
        try:
            if 'draw_date' not in df.columns or len(df) < 2:
                return None
            
            dates = pd.to_datetime(df['draw_date'], errors='coerce').dropna().sort_values()
            
            if len(dates) < 2:
                return None
            
            # Calculate average days between draws
            date_diffs = dates.diff().dt.days.dropna()
            
            return {
                'avg_days_between_draws': float(date_diffs.mean()),
                'min_days_between_draws': int(date_diffs.min()),
                'max_days_between_draws': int(date_diffs.max()),
                'draws_per_week': 7 / date_diffs.mean() if date_diffs.mean() > 0 else 0
            }
            
        except Exception:
            return None
    
    def validate_data_integrity(self, game_name: str) -> Dict[str, Any]:
        """
        Validate data integrity for a game.
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            Dictionary with integrity check results
        """
        try:
            df = self.load_historical_data_from_files(game_name)
            
            if df.empty:
                return {
                    'is_valid': False,
                    'issues': ['No data found'],
                    'recommendations': ['Load historical data for the game']
                }
            
            issues = []
            recommendations = []
            
            # Check for required columns
            required_columns = ['draw_date', 'numbers']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {', '.join(missing_columns)}")
                recommendations.append("Ensure data has draw_date and numbers columns")
            
            # Check for null values in critical columns
            for col in ['draw_date', 'numbers']:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        issues.append(f"{null_count} null values in {col}")
                        recommendations.append(f"Clean or impute null values in {col}")
            
            # Check date format consistency
            if 'draw_date' in df.columns:
                try:
                    pd.to_datetime(df['draw_date'], errors='raise')
                except Exception:
                    issues.append("Inconsistent date formats")
                    recommendations.append("Standardize date formats")
            
            # Check for duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                issues.append(f"{duplicates} duplicate records found")
                recommendations.append("Remove duplicate records")
            
            # Data freshness check
            if 'draw_date' in df.columns:
                latest_date = pd.to_datetime(df['draw_date'], errors='coerce').max()
                if pd.notna(latest_date):
                    days_since_last = (datetime.now() - latest_date.to_pydatetime()).days
                    if days_since_last > 30:
                        issues.append(f"Data is {days_since_last} days old")
                        recommendations.append("Update with recent draw data")
            
            integrity_result = {
                'is_valid': len(issues) == 0,
                'issues': issues,
                'recommendations': recommendations,
                'total_records': len(df),
                'completeness_score': self._calculate_data_completeness(df)
            }
            
            logger.info(f"✅ Data integrity check for {game_name}: {'PASS' if integrity_result['is_valid'] else 'FAIL'}")
            return integrity_result
            
        except Exception as e:
            logger.error(f"❌ Failed to validate data integrity for {game_name}: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'recommendations': ['Check data accessibility and format']
            }


# =============================================================================
# ENHANCED DATA SERVICE WITH BASE SERVICE INTEGRATION  
# =============================================================================

class DataService(BaseService, ServiceValidationMixin):
    """
    Enhanced Data Service integrating Phase 2 service foundation.
    
    Combines:
    - BaseService patterns (dependency injection, logging, error handling)
    - Original DataManager and StatisticsManager functionality
    - Extracted business logic from monolithic app.py
    """
    
    def _setup_service(self) -> None:
        """Initialize data service with integrated functionality."""
        self.log_operation("setup", status="info", action="initializing data service")
        
        # Initialize managers with config
        manager_config = {
            'data_dir': self.config.data_dir,
        }
        
        self.data_manager = DataManager(manager_config)
        self.statistics_manager = StatisticsManager(self.data_manager)
        
        self.log_operation("setup", status="success", 
                          data_dir=manager_config['data_dir'])
    
    # Delegate to managers with enhanced error handling
    def load_historical_data(self, game_name: str, limit: int = 1000) -> pd.DataFrame:
        """Load historical data with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.data_manager.load_historical_data_from_files,
            "load_historical_data",
            game_name=game_key,
            default_return=pd.DataFrame(),
            limit=limit
        )
    
    def get_latest_draw(self, game_name: str) -> Dict[str, Any]:
        """Get latest draw with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.data_manager.get_latest_draw,
            "get_latest_draw",
            game_name=game_key,
            default_return={}
        )
    
    def calculate_statistics(self, game_name: str) -> Dict[str, Any]:
        """Calculate statistics with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.data_manager.calculate_game_statistics,
            "calculate_statistics",
            game_name=game_key,
            default_return={}
        )
    
    def validate_data_integrity(self, game_name: str) -> Dict[str, Any]:
        """Validate data integrity with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.data_manager.validate_data_integrity,
            "validate_data_integrity",
            game_name=game_key,
            default_return={'is_valid': False, 'issues': ['Validation failed']}
        )
    
    def load_drawing_data(self, game_type: str = None, 
                         start_date: datetime = None, 
                         end_date: datetime = None) -> pd.DataFrame:
        """Load drawing data with service-level error handling."""
        self.validate_initialized()
        
        return self.safe_execute_operation(
            self.data_manager.load_drawing_data,
            "load_drawing_data",
            default_return=pd.DataFrame(),
            game_type=game_type,
            start_date=start_date,
            end_date=end_date
        )
    
    def save_drawing_data(self, data: pd.DataFrame, game_type: str) -> bool:
        """Save drawing data with service-level error handling."""
        self.validate_initialized()
        
        return self.safe_execute_operation(
            self.data_manager.save_drawing_data,
            "save_drawing_data",
            default_return=False,
            data=data,
            game_type=game_type
        )
    
    def calculate_number_frequency(self, game_type: str = None,
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> pd.DataFrame:
        """Calculate number frequency with service-level error handling."""
        self.validate_initialized()
        
        return self.safe_execute_operation(
            self.statistics_manager.calculate_number_frequency,
            "calculate_number_frequency",
            default_return=pd.DataFrame(),
            game_type=game_type,
            start_date=start_date,
            end_date=end_date
        )
    
    def _service_health_check(self) -> Optional[Dict[str, Any]]:
        """Data service specific health check."""
        health = {
            'healthy': True,
            'issues': []
        }
        
        # Check DataManager health
        if not self.data_manager.health_check():
            health['healthy'] = False
            health['issues'].append("DataManager health check failed")
        
        # Check StatisticsManager health
        if not self.statistics_manager.health_check():
            health['healthy'] = False
            health['issues'].append("StatisticsManager health check failed")
        
        # Check data directory access
        try:
            if not Path(self.config.data_dir).exists():
                Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            health['healthy'] = False
            health['issues'].append(f"Cannot access data directory: {e}")
        
        # Check database connectivity
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                conn.execute("SELECT 1").fetchone()
        except Exception as e:
            health['healthy'] = False
            health['issues'].append(f"Database connectivity issues: {e}")
        
        return health

    # Legacy Data Path Methods for Integration
    def get_features_path(self, model_type: str, game_type: str) -> Path:
        """Get path for features data: data/features/{model_type}/{game_type}/"""
        self.validate_initialized()
        game_key = sanitize_game_name(game_type)
        features_path = Path(self.config.data_dir) / "features" / model_type / game_key
        features_path.mkdir(parents=True, exist_ok=True)
        return features_path
    
    def get_raw_data_path(self, game_type: str) -> Path:
        """Get path for raw CSV data: data/{game_type}/"""
        self.validate_initialized()
        game_key = sanitize_game_name(game_type)
        raw_data_path = Path(self.config.data_dir) / game_key
        raw_data_path.mkdir(parents=True, exist_ok=True)
        return raw_data_path
    
    def load_features_data(self, model_type: str, game_type: str, feature_file: str = None) -> Union[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """Load features data from legacy path structure."""
        features_path = self.get_features_path(model_type, game_type)
        
        if feature_file:
            file_path = features_path / feature_file
            if not file_path.exists():
                raise FileNotFoundError(f"Feature file not found: {file_path}")
        else:
            # Find latest comprehensive feature file
            csv_files = list(features_path.glob("*comprehensive*.csv"))
            npz_files = list(features_path.glob("*comprehensive*.npz"))
            ultra_files = list(features_path.glob("*ultra*.csv"))
            
            if ultra_files:
                file_path = max(ultra_files, key=os.path.getmtime)
            elif csv_files:
                file_path = max(csv_files, key=os.path.getmtime)
            elif npz_files:
                file_path = max(npz_files, key=os.path.getmtime)
            else:
                raise FileNotFoundError(f"No feature files found in {features_path}")
        
        return self.safe_execute_operation(
            self._load_feature_file,
            "load_features_data",
            default_return=pd.DataFrame(),
            file_path=file_path
        )
    
    def _load_feature_file(self, file_path: Path) -> Union[pd.DataFrame, np.ndarray, Dict[str, Any]]:
        """Load feature file based on extension."""
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.npz':
            return np.load(file_path, allow_pickle=True)
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported feature file format: {file_path.suffix}")
    
    def load_raw_training_data(self, game_type: str, year: str = None) -> pd.DataFrame:
        """Load raw training data from legacy path structure."""
        raw_data_path = self.get_raw_data_path(game_type)
        
        if year:
            file_path = raw_data_path / f"training_data_{year}.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Training data not found for year {year}: {file_path}")
        else:
            # Get latest training data file
            csv_files = list(raw_data_path.glob("training_data_*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No training data files found in {raw_data_path}")
            file_path = max(csv_files, key=os.path.getmtime)
        
        return self.safe_execute_operation(
            pd.read_csv,
            "load_raw_training_data",
            default_return=pd.DataFrame(),
            filepath_or_buffer=file_path
        )
    
    def get_available_features(self, model_type: str, game_type: str) -> List[str]:
        """Get list of available feature files for a model type and game."""
        features_path = self.get_features_path(model_type, game_type)
        if not features_path.exists():
            return []
        
        feature_files = []
        for ext in ['.csv', '.npz', '.json']:
            feature_files.extend([f.name for f in features_path.glob(f"*{ext}")])
        
        return sorted(feature_files)
    
    def get_available_raw_data_years(self, game_type: str) -> List[str]:
        """Get list of available years for raw training data."""
        raw_data_path = self.get_raw_data_path(game_type)
        if not raw_data_path.exists():
            return []
        
        csv_files = list(raw_data_path.glob("training_data_*.csv"))
        years = []
        for file in csv_files:
            # Extract year from filename like "training_data_2024.csv"
            year_match = file.stem.split('_')[-1]
            if year_match.isdigit():
                years.append(year_match)
        
        return sorted(years)