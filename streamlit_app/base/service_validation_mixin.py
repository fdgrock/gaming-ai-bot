"""
Service Validation Mixin

Provides validation utilities for service classes including
parameter validation, data type checking, and business rule validation.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import logging

from ..core.exceptions import ValidationError


class ServiceValidationMixin:
    """
    Mixin class providing validation utilities for services.
    
    This mixin can be used with BaseService to add comprehensive
    validation capabilities to service methods.
    """
    
    def validate_not_none(self, value: Any, name: str) -> Any:
        """Validate that a value is not None."""
        if value is None:
            raise ValidationError(name, value, "Value cannot be None")
        return value
    
    def validate_not_empty(self, value: Union[str, list, dict], name: str) -> Any:
        """Validate that a value is not empty."""
        if not value:
            raise ValidationError(name, value, "Value cannot be empty")
        return value
    
    def validate_type(self, value: Any, expected_type: type, name: str) -> Any:
        """Validate that a value is of the expected type."""
        if not isinstance(value, expected_type):
            raise ValidationError(
                name, value, 
                f"Expected {expected_type.__name__}, got {type(value).__name__}"
            )
        return value
    
    def validate_dataframe(self, df: pd.DataFrame, name: str, 
                          min_rows: int = 1, required_columns: List[str] = None) -> pd.DataFrame:
        """
        Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            name: Parameter name for error messages
            min_rows: Minimum number of rows required
            required_columns: List of required column names
            
        Returns:
            Validated DataFrame
        """
        self.validate_type(df, pd.DataFrame, name)
        
        if len(df) < min_rows:
            raise ValidationError(
                name, f"rows={len(df)}", 
                f"DataFrame must have at least {min_rows} rows"
            )
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValidationError(
                    name, f"columns={list(df.columns)}", 
                    f"Missing required columns: {missing_cols}"
                )
        
        return df
    
    def validate_numeric(self, value: Union[int, float], name: str,
                        min_value: Optional[float] = None,
                        max_value: Optional[float] = None) -> Union[int, float]:
        """
        Validate a numeric value.
        
        Args:
            value: Value to validate
            name: Parameter name
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Validated value
        """
        if not isinstance(value, (int, float)) or np.isnan(value):
            raise ValidationError(name, value, "Must be a valid number")
        
        if min_value is not None and value < min_value:
            raise ValidationError(name, value, f"Must be >= {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(name, value, f"Must be <= {max_value}")
        
        return value
    
    def validate_string(self, value: str, name: str,
                       min_length: Optional[int] = None,
                       max_length: Optional[int] = None,
                       allowed_values: Optional[List[str]] = None) -> str:
        """
        Validate a string value.
        
        Args:
            value: String to validate
            name: Parameter name
            min_length: Minimum string length
            max_length: Maximum string length  
            allowed_values: List of allowed values
            
        Returns:
            Validated string
        """
        self.validate_type(value, str, name)
        
        if min_length is not None and len(value) < min_length:
            raise ValidationError(name, value, f"Must be at least {min_length} characters")
        
        if max_length is not None and len(value) > max_length:
            raise ValidationError(name, value, f"Must be at most {max_length} characters")
        
        if allowed_values is not None and value not in allowed_values:
            raise ValidationError(name, value, f"Must be one of: {allowed_values}")
        
        return value
    
    def validate_dict(self, value: dict, name: str,
                     required_keys: Optional[List[str]] = None,
                     allowed_keys: Optional[List[str]] = None) -> dict:
        """
        Validate a dictionary.
        
        Args:
            value: Dictionary to validate
            name: Parameter name
            required_keys: List of required keys
            allowed_keys: List of allowed keys
            
        Returns:
            Validated dictionary
        """
        self.validate_type(value, dict, name)
        
        if required_keys:
            missing_keys = [key for key in required_keys if key not in value]
            if missing_keys:
                raise ValidationError(name, value, f"Missing required keys: {missing_keys}")
        
        if allowed_keys:
            invalid_keys = [key for key in value.keys() if key not in allowed_keys]
            if invalid_keys:
                raise ValidationError(name, value, f"Invalid keys: {invalid_keys}")
        
        return value
    
    def validate_list(self, value: list, name: str,
                     min_length: Optional[int] = None,
                     max_length: Optional[int] = None,
                     item_type: Optional[type] = None) -> list:
        """
        Validate a list.
        
        Args:
            value: List to validate
            name: Parameter name
            min_length: Minimum list length
            max_length: Maximum list length
            item_type: Required type for list items
            
        Returns:
            Validated list
        """
        self.validate_type(value, list, name)
        
        if min_length is not None and len(value) < min_length:
            raise ValidationError(name, value, f"Must have at least {min_length} items")
        
        if max_length is not None and len(value) > max_length:
            raise ValidationError(name, value, f"Must have at most {max_length} items")
        
        if item_type is not None:
            for i, item in enumerate(value):
                if not isinstance(item, item_type):
                    raise ValidationError(
                        f"{name}[{i}]", item, 
                        f"All items must be of type {item_type.__name__}"
                    )
        
        return value
    
    def validate_date_range(self, start_date: datetime, end_date: datetime,
                           max_days: Optional[int] = None) -> tuple:
        """
        Validate a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            max_days: Maximum allowed days in range
            
        Returns:
            Tuple of (start_date, end_date)
        """
        self.validate_type(start_date, datetime, "start_date")
        self.validate_type(end_date, datetime, "end_date")
        
        if start_date >= end_date:
            raise ValidationError(
                "date_range", f"{start_date} to {end_date}",
                "Start date must be before end date"
            )
        
        if max_days is not None:
            days_diff = (end_date - start_date).days
            if days_diff > max_days:
                raise ValidationError(
                    "date_range", f"{days_diff} days",
                    f"Date range cannot exceed {max_days} days"
                )
        
        return start_date, end_date
    
    def validate_probability(self, value: float, name: str) -> float:
        """Validate a probability value (0.0 to 1.0)."""
        return self.validate_numeric(value, name, min_value=0.0, max_value=1.0)
    
    def validate_percentage(self, value: float, name: str) -> float:
        """Validate a percentage value (0.0 to 100.0)."""
        return self.validate_numeric(value, name, min_value=0.0, max_value=100.0)
    
    def validate_positive_integer(self, value: int, name: str) -> int:
        """Validate a positive integer."""
        self.validate_type(value, int, name)
        if value <= 0:
            raise ValidationError(name, value, "Must be a positive integer")
        return value
    
    def validate_game_name(self, game_name: str) -> str:
        """
        Validate and normalize game name.
        
        Args:
            game_name: Raw game name
            
        Returns:
            Normalized game name
        """
        # Common game name mappings
        game_mappings = {
            'powerball': 'powerball',
            'mega millions': 'megamillions',
            'mega_millions': 'megamillions',
            'megamillions': 'megamillions',
            'lotto': 'lotto',
            'pick3': 'pick3',
            'pick4': 'pick4',
            'daily4': 'pick4',
            'fantasy5': 'fantasy5',
            'fantasy_5': 'fantasy5'
        }
        
        if not game_name:
            raise ValidationError("game_name", game_name, "Game name cannot be empty")
        
        # Normalize to lowercase and remove extra spaces
        normalized = game_name.lower().strip()
        
        # Apply mappings
        mapped_name = game_mappings.get(normalized, normalized)
        
        # Validate against known games
        valid_games = list(game_mappings.values())
        if mapped_name not in valid_games:
            raise ValidationError(
                "game_name", game_name,
                f"Unknown game. Valid games: {valid_games}"
            )
        
        return mapped_name
    
    def validate_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Validated configuration
        """
        required_keys = ['model_type', 'parameters']
        self.validate_dict(config, "model_config", required_keys=required_keys)
        
        # Validate model type
        valid_model_types = [
            'xgboost', 'lstm', 'transformer', 'random_forest',
            'linear_regression', 'neural_network'
        ]
        self.validate_string(
            config['model_type'], "model_type",
            allowed_values=valid_model_types
        )
        
        # Validate parameters
        self.validate_dict(config['parameters'], "parameters")
        
        return config
    
    def validate_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate training data format and quality.
        
        Args:
            data: Training data DataFrame
            
        Returns:
            Validated DataFrame
        """
        # Basic DataFrame validation
        self.validate_dataframe(data, "training_data", min_rows=10)
        
        # Check for required columns (basic lottery data structure)
        required_cols = ['draw_date', 'numbers']
        self.validate_dataframe(data, "training_data", required_columns=required_cols)
        
        # Validate draw_date column
        if not pd.api.types.is_datetime64_any_dtype(data['draw_date']):
            try:
                data['draw_date'] = pd.to_datetime(data['draw_date'])
            except:
                raise ValidationError(
                    "training_data.draw_date", 
                    data['draw_date'].dtype,
                    "draw_date column must be convertible to datetime"
                )
        
        # Check for missing values in critical columns
        critical_columns = ['draw_date', 'numbers']
        for col in critical_columns:
            if data[col].isnull().any():
                null_count = data[col].isnull().sum()
                raise ValidationError(
                    f"training_data.{col}", f"{null_count} nulls",
                    f"Column {col} cannot contain null values"
                )
        
        return data
    
    def validate_prediction_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate prediction input data.
        
        Args:
            data: Prediction input dictionary
            
        Returns:
            Validated input data
        """
        required_keys = ['game_name', 'prediction_date']
        self.validate_dict(data, "prediction_input", required_keys=required_keys)
        
        # Validate game name
        data['game_name'] = self.validate_game_name(data['game_name'])
        
        # Validate prediction date
        if isinstance(data['prediction_date'], str):
            try:
                data['prediction_date'] = datetime.fromisoformat(data['prediction_date'])
            except ValueError:
                raise ValidationError(
                    "prediction_date", data['prediction_date'],
                    "Invalid date format. Use ISO format (YYYY-MM-DD)"
                )
        
        self.validate_type(data['prediction_date'], datetime, "prediction_date")
        
        return data