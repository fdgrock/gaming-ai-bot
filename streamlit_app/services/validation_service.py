"""
Validation service module for the lottery prediction system.

This module provides comprehensive validation services including input validation,
configuration validation, and business rule enforcement.
"""

import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import ipaddress
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation level enumeration."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Validation result structure."""
    field: str
    level: ValidationLevel
    message: str
    value: Any
    expected: Optional[Any] = None
    suggestion: Optional[str] = None


class ValidationReport:
    """Comprehensive validation report."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.summary = {
            'total_checks': 0,
            'errors': 0,
            'warnings': 0,
            'info': 0,
            'passed': 0
        }
    
    def add_result(self, result: ValidationResult) -> None:
        """Add validation result."""
        self.results.append(result)
        self.summary['total_checks'] += 1
        
        if result.level == ValidationLevel.ERROR:
            self.summary['errors'] += 1
        elif result.level == ValidationLevel.WARNING:
            self.summary['warnings'] += 1
        elif result.level == ValidationLevel.INFO:
            self.summary['info'] += 1
    
    def add_success(self, field: str, message: str = "Validation passed") -> None:
        """Add successful validation."""
        self.summary['total_checks'] += 1
        self.summary['passed'] += 1
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return self.summary['errors'] == 0
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return self.summary['warnings'] > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'valid': self.is_valid(),
            'summary': self.summary,
            'results': [
                {
                    'field': r.field,
                    'level': r.level.value,
                    'message': r.message,
                    'value': str(r.value) if r.value is not None else None,
                    'expected': str(r.expected) if r.expected is not None else None,
                    'suggestion': r.suggestion
                }
                for r in self.results
            ]
        }


class DataValidator:
    """
    Validates data inputs and formats.
    
    This class provides comprehensive data validation including
    type checking, range validation, format validation, and business rules.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize data validator."""
        self.config = config or {}
        self.strict_mode = self.config.get('strict_mode', False)
        
        # Validation rules
        self.number_rules = {
            'min_value': self.config.get('min_number', 1),
            'max_value': self.config.get('max_number', 70),
            'max_numbers': self.config.get('max_numbers', 10)
        }
        
        self.date_rules = {
            'min_date': self.config.get('min_date', datetime(2000, 1, 1)),
            'max_date': self.config.get('max_date', datetime.now() + timedelta(days=365))
        }
    
    def validate_lottery_numbers(self, numbers: List[int], 
                                game_config: Dict[str, Any] = None) -> ValidationReport:
        """
        Validate lottery number selection.
        
        Args:
            numbers: List of lottery numbers
            game_config: Game-specific configuration
            
        Returns:
            Validation report
        """
        report = ValidationReport()
        
        try:
            # Use game config or defaults
            config = game_config or {}
            min_number = config.get('min_number', self.number_rules['min_value'])
            max_number = config.get('max_number', self.number_rules['max_value'])
            num_numbers = config.get('num_numbers', 6)
            
            # Check if numbers is a list
            if not isinstance(numbers, list):
                report.add_result(ValidationResult(
                    field='numbers',
                    level=ValidationLevel.ERROR,
                    message='Numbers must be provided as a list',
                    value=type(numbers).__name__,
                    expected='list'
                ))
                return report
            
            # Check number count
            if len(numbers) != num_numbers:
                report.add_result(ValidationResult(
                    field='numbers.count',
                    level=ValidationLevel.ERROR,
                    message=f'Expected {num_numbers} numbers, got {len(numbers)}',
                    value=len(numbers),
                    expected=num_numbers
                ))
            else:
                report.add_success('numbers.count', f'Correct number count: {num_numbers}')
            
            # Check for duplicates
            if len(set(numbers)) != len(numbers):
                duplicates = [x for x in numbers if numbers.count(x) > 1]
                report.add_result(ValidationResult(
                    field='numbers.duplicates',
                    level=ValidationLevel.ERROR,
                    message=f'Duplicate numbers found: {duplicates}',
                    value=duplicates,
                    suggestion='Remove duplicate numbers'
                ))
            else:
                report.add_success('numbers.duplicates', 'No duplicate numbers found')
            
            # Check number ranges
            invalid_numbers = [n for n in numbers if not isinstance(n, int) or n < min_number or n > max_number]
            if invalid_numbers:
                report.add_result(ValidationResult(
                    field='numbers.range',
                    level=ValidationLevel.ERROR,
                    message=f'Numbers out of valid range [{min_number}, {max_number}]: {invalid_numbers}',
                    value=invalid_numbers,
                    expected=f'Range [{min_number}, {max_number}]'
                ))
            else:
                report.add_success('numbers.range', f'All numbers within valid range [{min_number}, {max_number}]')
            
            # Check for non-integer values
            non_integers = [n for n in numbers if not isinstance(n, int)]
            if non_integers:
                report.add_result(ValidationResult(
                    field='numbers.type',
                    level=ValidationLevel.ERROR,
                    message=f'Non-integer values found: {non_integers}',
                    value=non_integers,
                    expected='integers'
                ))
            else:
                report.add_success('numbers.type', 'All numbers are integers')
            
            # Business rule validations
            self._validate_number_patterns(numbers, report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Number validation failed: {e}")
            report.add_result(ValidationResult(
                field='validation.error',
                level=ValidationLevel.ERROR,
                message=f'Validation process failed: {e}',
                value=str(e)
            ))
            return report
    
    def _validate_number_patterns(self, numbers: List[int], report: ValidationReport) -> None:
        """Validate number patterns and business rules."""
        try:
            # Check for all consecutive numbers
            sorted_numbers = sorted(numbers)
            consecutive_count = 0
            for i in range(len(sorted_numbers) - 1):
                if sorted_numbers[i + 1] == sorted_numbers[i] + 1:
                    consecutive_count += 1
            
            if consecutive_count == len(numbers) - 1:
                report.add_result(ValidationResult(
                    field='numbers.pattern.consecutive',
                    level=ValidationLevel.WARNING,
                    message='All numbers are consecutive',
                    value=numbers,
                    suggestion='Consider mixing consecutive and non-consecutive numbers'
                ))
            
            # Check for all even or all odd
            all_even = all(n % 2 == 0 for n in numbers)
            all_odd = all(n % 2 == 1 for n in numbers)
            
            if all_even:
                report.add_result(ValidationResult(
                    field='numbers.pattern.parity',
                    level=ValidationLevel.WARNING,
                    message='All numbers are even',
                    value=numbers,
                    suggestion='Consider mixing even and odd numbers'
                ))
            elif all_odd:
                report.add_result(ValidationResult(
                    field='numbers.pattern.parity',
                    level=ValidationLevel.WARNING,
                    message='All numbers are odd',
                    value=numbers,
                    suggestion='Consider mixing even and odd numbers'
                ))
            
            # Check number distribution
            if len(numbers) >= 5:
                low_count = sum(1 for n in numbers if n <= 35)  # Assuming max 70
                high_count = len(numbers) - low_count
                
                if low_count == 0:
                    report.add_result(ValidationResult(
                        field='numbers.distribution',
                        level=ValidationLevel.WARNING,
                        message='No numbers in low range (1-35)',
                        value=f'Low: {low_count}, High: {high_count}',
                        suggestion='Consider including some low numbers'
                    ))
                elif high_count == 0:
                    report.add_result(ValidationResult(
                        field='numbers.distribution',
                        level=ValidationLevel.WARNING,
                        message='No numbers in high range (36-70)',
                        value=f'Low: {low_count}, High: {high_count}',
                        suggestion='Consider including some high numbers'
                    ))
            
        except Exception as e:
            logger.error(f"❌ Pattern validation failed: {e}")
    
    def validate_game_configuration(self, config: Dict[str, Any]) -> ValidationReport:
        """
        Validate game configuration.
        
        Args:
            config: Game configuration dictionary
            
        Returns:
            Validation report
        """
        report = ValidationReport()
        
        try:
            # Required fields
            required_fields = ['game_name', 'num_numbers', 'max_number']
            for field in required_fields:
                if field not in config:
                    report.add_result(ValidationResult(
                        field=f'config.{field}',
                        level=ValidationLevel.ERROR,
                        message=f'Required field missing: {field}',
                        value=None,
                        expected='non-null value'
                    ))
                else:
                    report.add_success(f'config.{field}', f'Required field present: {field}')
            
            # Validate specific fields
            if 'num_numbers' in config:
                num_numbers = config['num_numbers']
                if not isinstance(num_numbers, int) or num_numbers <= 0 or num_numbers > 20:
                    report.add_result(ValidationResult(
                        field='config.num_numbers',
                        level=ValidationLevel.ERROR,
                        message='num_numbers must be a positive integer between 1 and 20',
                        value=num_numbers,
                        expected='1-20'
                    ))
                else:
                    report.add_success('config.num_numbers', f'Valid num_numbers: {num_numbers}')
            
            if 'max_number' in config:
                max_number = config['max_number']
                if not isinstance(max_number, int) or max_number < config.get('num_numbers', 6):
                    report.add_result(ValidationResult(
                        field='config.max_number',
                        level=ValidationLevel.ERROR,
                        message='max_number must be an integer greater than num_numbers',
                        value=max_number,
                        expected=f'> {config.get("num_numbers", 6)}'
                    ))
                else:
                    report.add_success('config.max_number', f'Valid max_number: {max_number}')
            
            if 'min_number' in config:
                min_number = config['min_number']
                if not isinstance(min_number, int) or min_number < 1:
                    report.add_result(ValidationResult(
                        field='config.min_number',
                        level=ValidationLevel.ERROR,
                        message='min_number must be a positive integer',
                        value=min_number,
                        expected='>= 1'
                    ))
                else:
                    report.add_success('config.min_number', f'Valid min_number: {min_number}')
            
            # Validate draw schedule if present
            if 'draw_schedule' in config:
                self._validate_draw_schedule(config['draw_schedule'], report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Config validation failed: {e}")
            report.add_result(ValidationResult(
                field='validation.error',
                level=ValidationLevel.ERROR,
                message=f'Configuration validation failed: {e}',
                value=str(e)
            ))
            return report
    
    def _validate_draw_schedule(self, schedule: Dict[str, Any], report: ValidationReport) -> None:
        """Validate draw schedule configuration."""
        try:
            if 'days' in schedule:
                days = schedule['days']
                valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                
                if isinstance(days, list):
                    invalid_days = [day for day in days if day.lower() not in valid_days]
                    if invalid_days:
                        report.add_result(ValidationResult(
                            field='config.draw_schedule.days',
                            level=ValidationLevel.ERROR,
                            message=f'Invalid day names: {invalid_days}',
                            value=invalid_days,
                            expected=valid_days
                        ))
                    else:
                        report.add_success('config.draw_schedule.days', f'Valid draw days: {days}')
                else:
                    report.add_result(ValidationResult(
                        field='config.draw_schedule.days',
                        level=ValidationLevel.ERROR,
                        message='Draw days must be a list',
                        value=type(days).__name__,
                        expected='list'
                    ))
            
            if 'time' in schedule:
                time_str = schedule['time']
                if not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', time_str):
                    report.add_result(ValidationResult(
                        field='config.draw_schedule.time',
                        level=ValidationLevel.ERROR,
                        message='Invalid time format',
                        value=time_str,
                        expected='HH:MM format'
                    ))
                else:
                    report.add_success('config.draw_schedule.time', f'Valid draw time: {time_str}')
            
        except Exception as e:
            logger.error(f"❌ Draw schedule validation failed: {e}")
    
    def validate_prediction_data(self, prediction_data: Dict[str, Any]) -> ValidationReport:
        """
        Validate prediction data structure.
        
        Args:
            prediction_data: Prediction data dictionary
            
        Returns:
            Validation report
        """
        report = ValidationReport()
        
        try:
            # Check required fields
            required_fields = ['predictions', 'confidence', 'strategy']
            for field in required_fields:
                if field not in prediction_data:
                    report.add_result(ValidationResult(
                        field=f'prediction.{field}',
                        level=ValidationLevel.ERROR,
                        message=f'Required field missing: {field}',
                        value=None,
                        expected='non-null value'
                    ))
                else:
                    report.add_success(f'prediction.{field}', f'Required field present: {field}')
            
            # Validate predictions array
            if 'predictions' in prediction_data:
                predictions = prediction_data['predictions']
                if not isinstance(predictions, list):
                    report.add_result(ValidationResult(
                        field='prediction.predictions',
                        level=ValidationLevel.ERROR,
                        message='Predictions must be a list',
                        value=type(predictions).__name__,
                        expected='list'
                    ))
                elif not predictions:
                    report.add_result(ValidationResult(
                        field='prediction.predictions',
                        level=ValidationLevel.ERROR,
                        message='Predictions list cannot be empty',
                        value=len(predictions),
                        expected='> 0'
                    ))
                else:
                    # Validate each prediction
                    for i, pred in enumerate(predictions):
                        if isinstance(pred, list):
                            pred_report = self.validate_lottery_numbers(pred)
                            if not pred_report.is_valid():
                                report.add_result(ValidationResult(
                                    field=f'prediction.predictions[{i}]',
                                    level=ValidationLevel.ERROR,
                                    message=f'Invalid prediction at index {i}',
                                    value=pred,
                                    suggestion='Check number format and ranges'
                                ))
                        else:
                            report.add_result(ValidationResult(
                                field=f'prediction.predictions[{i}]',
                                level=ValidationLevel.ERROR,
                                message=f'Prediction at index {i} must be a list of numbers',
                                value=type(pred).__name__,
                                expected='list'
                            ))
            
            # Validate confidence
            if 'confidence' in prediction_data:
                confidence = prediction_data['confidence']
                if not isinstance(confidence, (int, float)):
                    report.add_result(ValidationResult(
                        field='prediction.confidence',
                        level=ValidationLevel.ERROR,
                        message='Confidence must be a number',
                        value=type(confidence).__name__,
                        expected='number'
                    ))
                elif not (0 <= confidence <= 1):
                    report.add_result(ValidationResult(
                        field='prediction.confidence',
                        level=ValidationLevel.ERROR,
                        message='Confidence must be between 0 and 1',
                        value=confidence,
                        expected='0.0 - 1.0'
                    ))
                else:
                    report.add_success('prediction.confidence', f'Valid confidence: {confidence}')
            
            # Validate strategy
            if 'strategy' in prediction_data:
                strategy = prediction_data['strategy']
                valid_strategies = ['aggressive', 'balanced', 'conservative', 'custom']
                if strategy not in valid_strategies:
                    report.add_result(ValidationResult(
                        field='prediction.strategy',
                        level=ValidationLevel.WARNING,
                        message=f'Unknown strategy: {strategy}',
                        value=strategy,
                        expected=valid_strategies,
                        suggestion='Use a recognized strategy name'
                    ))
                else:
                    report.add_success('prediction.strategy', f'Valid strategy: {strategy}')
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Prediction data validation failed: {e}")
            report.add_result(ValidationResult(
                field='validation.error',
                level=ValidationLevel.ERROR,
                message=f'Prediction validation failed: {e}',
                value=str(e)
            ))
            return report
    
    def validate_historical_data(self, data: pd.DataFrame) -> ValidationReport:
        """
        Validate historical lottery data.
        
        Args:
            data: Historical data DataFrame
            
        Returns:
            Validation report
        """
        report = ValidationReport()
        
        try:
            # Check if DataFrame is empty
            if data.empty:
                report.add_result(ValidationResult(
                    field='data.size',
                    level=ValidationLevel.ERROR,
                    message='Historical data is empty',
                    value=0,
                    expected='> 0 rows'
                ))
                return report
            else:
                report.add_success('data.size', f'Data contains {len(data)} records')
            
            # Check required columns
            required_columns = ['draw_date', 'numbers']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                report.add_result(ValidationResult(
                    field='data.columns',
                    level=ValidationLevel.ERROR,
                    message=f'Missing required columns: {missing_columns}',
                    value=list(data.columns),
                    expected=required_columns
                ))
            else:
                report.add_success('data.columns', 'All required columns present')
            
            # Validate date column
            if 'draw_date' in data.columns:
                self._validate_date_column(data['draw_date'], report)
            
            # Validate numbers column
            if 'numbers' in data.columns:
                self._validate_numbers_column(data['numbers'], report)
            
            # Check for duplicates
            if 'draw_date' in data.columns:
                duplicate_dates = data['draw_date'].duplicated().sum()
                if duplicate_dates > 0:
                    report.add_result(ValidationResult(
                        field='data.duplicates',
                        level=ValidationLevel.WARNING,
                        message=f'Found {duplicate_dates} duplicate draw dates',
                        value=duplicate_dates,
                        suggestion='Remove or verify duplicate entries'
                    ))
                else:
                    report.add_success('data.duplicates', 'No duplicate draw dates found')
            
            # Check data quality
            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                report.add_result(ValidationResult(
                    field='data.missing_values',
                    level=ValidationLevel.WARNING,
                    message=f'Found {missing_values} missing values',
                    value=missing_values,
                    suggestion='Clean or impute missing values'
                ))
            else:
                report.add_success('data.missing_values', 'No missing values found')
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Historical data validation failed: {e}")
            report.add_result(ValidationResult(
                field='validation.error',
                level=ValidationLevel.ERROR,
                message=f'Historical data validation failed: {e}',
                value=str(e)
            ))
            return report
    
    def _validate_date_column(self, date_series: pd.Series, report: ValidationReport) -> None:
        """Validate date column."""
        try:
            # Try to convert to datetime
            try:
                date_series = pd.to_datetime(date_series, errors='coerce')
                invalid_dates = date_series.isnull().sum()
                
                if invalid_dates > 0:
                    report.add_result(ValidationResult(
                        field='data.draw_date.format',
                        level=ValidationLevel.ERROR,
                        message=f'Found {invalid_dates} invalid date formats',
                        value=invalid_dates,
                        suggestion='Ensure dates are in valid format'
                    ))
                else:
                    report.add_success('data.draw_date.format', 'All dates have valid format')
                
                # Check date range
                if not date_series.empty:
                    min_date = date_series.min()
                    max_date = date_series.max()
                    
                    if min_date < self.date_rules['min_date']:
                        report.add_result(ValidationResult(
                            field='data.draw_date.range',
                            level=ValidationLevel.WARNING,
                            message=f'Dates before minimum allowed: {min_date}',
                            value=str(min_date),
                            expected=f'>= {self.date_rules["min_date"]}'
                        ))
                    
                    if max_date > self.date_rules['max_date']:
                        report.add_result(ValidationResult(
                            field='data.draw_date.range',
                            level=ValidationLevel.WARNING,
                            message=f'Future dates found: {max_date}',
                            value=str(max_date),
                            expected=f'<= {self.date_rules["max_date"]}'
                        ))
                
            except Exception as e:
                report.add_result(ValidationResult(
                    field='data.draw_date.conversion',
                    level=ValidationLevel.ERROR,
                    message=f'Date conversion failed: {e}',
                    value=str(e),
                    suggestion='Check date format and values'
                ))
            
        except Exception as e:
            logger.error(f"❌ Date column validation failed: {e}")
    
    def _validate_numbers_column(self, numbers_series: pd.Series, report: ValidationReport) -> None:
        """Validate numbers column."""
        try:
            invalid_count = 0
            
            for idx, numbers in numbers_series.items():
                try:
                    # Try to parse numbers
                    if isinstance(numbers, str):
                        # Try to parse JSON string
                        try:
                            parsed_numbers = json.loads(numbers)
                        except:
                            # Try to parse comma-separated
                            parsed_numbers = [int(x.strip()) for x in numbers.split(',')]
                    elif isinstance(numbers, list):
                        parsed_numbers = numbers
                    else:
                        invalid_count += 1
                        continue
                    
                    # Validate parsed numbers
                    if not isinstance(parsed_numbers, list):
                        invalid_count += 1
                        continue
                    
                    # Check for valid integers
                    if not all(isinstance(n, int) for n in parsed_numbers):
                        invalid_count += 1
                        continue
                        
                except:
                    invalid_count += 1
            
            if invalid_count > 0:
                report.add_result(ValidationResult(
                    field='data.numbers.format',
                    level=ValidationLevel.ERROR,
                    message=f'Found {invalid_count} records with invalid number format',
                    value=invalid_count,
                    suggestion='Ensure numbers are in list or comma-separated format'
                ))
            else:
                report.add_success('data.numbers.format', 'All number records have valid format')
            
        except Exception as e:
            logger.error(f"❌ Numbers column validation failed: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check data validator health."""
        return True


class ConfigValidator:
    """
    Validates application and model configurations.
    
    This class ensures that configuration files and settings
    are properly formatted and contain valid values.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize config validator."""
        self.config = config or {}
    
    def validate_app_config(self, app_config: Dict[str, Any]) -> ValidationReport:
        """
        Validate application configuration.
        
        Args:
            app_config: Application configuration dictionary
            
        Returns:
            Validation report
        """
        report = ValidationReport()
        
        try:
            # Validate database configuration
            if 'database' in app_config:
                self._validate_database_config(app_config['database'], report)
            
            # Validate cache configuration
            if 'cache' in app_config:
                self._validate_cache_config(app_config['cache'], report)
            
            # Validate logging configuration
            if 'logging' in app_config:
                self._validate_logging_config(app_config['logging'], report)
            
            # Validate API configuration
            if 'api' in app_config:
                self._validate_api_config(app_config['api'], report)
            
            # Validate security settings
            if 'security' in app_config:
                self._validate_security_config(app_config['security'], report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ App config validation failed: {e}")
            report.add_result(ValidationResult(
                field='validation.error',
                level=ValidationLevel.ERROR,
                message=f'Application config validation failed: {e}',
                value=str(e)
            ))
            return report
    
    def _validate_database_config(self, db_config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate database configuration."""
        try:
            # Check connection string or parameters
            if 'connection_string' in db_config:
                conn_str = db_config['connection_string']
                if not isinstance(conn_str, str) or not conn_str.strip():
                    report.add_result(ValidationResult(
                        field='config.database.connection_string',
                        level=ValidationLevel.ERROR,
                        message='Connection string must be a non-empty string',
                        value=type(conn_str).__name__,
                        expected='non-empty string'
                    ))
                else:
                    report.add_success('config.database.connection_string', 'Valid connection string')
            
            # Check pool settings
            if 'pool_size' in db_config:
                pool_size = db_config['pool_size']
                if not isinstance(pool_size, int) or pool_size <= 0:
                    report.add_result(ValidationResult(
                        field='config.database.pool_size',
                        level=ValidationLevel.ERROR,
                        message='Pool size must be a positive integer',
                        value=pool_size,
                        expected='> 0'
                    ))
                else:
                    report.add_success('config.database.pool_size', f'Valid pool size: {pool_size}')
            
            # Check timeout settings
            if 'timeout' in db_config:
                timeout = db_config['timeout']
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    report.add_result(ValidationResult(
                        field='config.database.timeout',
                        level=ValidationLevel.ERROR,
                        message='Timeout must be a positive number',
                        value=timeout,
                        expected='> 0'
                    ))
                else:
                    report.add_success('config.database.timeout', f'Valid timeout: {timeout}')
            
        except Exception as e:
            logger.error(f"❌ Database config validation failed: {e}")
    
    def _validate_cache_config(self, cache_config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate cache configuration."""
        try:
            # Check memory limits
            if 'max_memory_mb' in cache_config:
                max_memory = cache_config['max_memory_mb']
                if not isinstance(max_memory, (int, float)) or max_memory <= 0:
                    report.add_result(ValidationResult(
                        field='config.cache.max_memory_mb',
                        level=ValidationLevel.ERROR,
                        message='Max memory must be a positive number',
                        value=max_memory,
                        expected='> 0'
                    ))
                else:
                    report.add_success('config.cache.max_memory_mb', f'Valid memory limit: {max_memory}MB')
            
            # Check TTL settings
            if 'default_ttl' in cache_config:
                ttl = cache_config['default_ttl']
                if not isinstance(ttl, int) or ttl <= 0:
                    report.add_result(ValidationResult(
                        field='config.cache.default_ttl',
                        level=ValidationLevel.ERROR,
                        message='Default TTL must be a positive integer',
                        value=ttl,
                        expected='> 0'
                    ))
                else:
                    report.add_success('config.cache.default_ttl', f'Valid TTL: {ttl}s')
            
            # Check cache directory
            if 'cache_dir' in cache_config:
                cache_dir = cache_config['cache_dir']
                if not isinstance(cache_dir, str):
                    report.add_result(ValidationResult(
                        field='config.cache.cache_dir',
                        level=ValidationLevel.ERROR,
                        message='Cache directory must be a string',
                        value=type(cache_dir).__name__,
                        expected='string'
                    ))
                else:
                    # Check if directory is writable
                    try:
                        Path(cache_dir).mkdir(parents=True, exist_ok=True)
                        report.add_success('config.cache.cache_dir', f'Valid cache directory: {cache_dir}')
                    except Exception as e:
                        report.add_result(ValidationResult(
                            field='config.cache.cache_dir',
                            level=ValidationLevel.ERROR,
                            message=f'Cache directory not accessible: {e}',
                            value=cache_dir,
                            suggestion='Check directory permissions'
                        ))
            
        except Exception as e:
            logger.error(f"❌ Cache config validation failed: {e}")
    
    def _validate_logging_config(self, logging_config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate logging configuration."""
        try:
            # Check log level
            if 'level' in logging_config:
                level = logging_config['level']
                valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                if level not in valid_levels:
                    report.add_result(ValidationResult(
                        field='config.logging.level',
                        level=ValidationLevel.ERROR,
                        message=f'Invalid log level: {level}',
                        value=level,
                        expected=valid_levels
                    ))
                else:
                    report.add_success('config.logging.level', f'Valid log level: {level}')
            
            # Check log file path
            if 'file' in logging_config:
                log_file = logging_config['file']
                if log_file:
                    try:
                        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                        report.add_success('config.logging.file', f'Valid log file path: {log_file}')
                    except Exception as e:
                        report.add_result(ValidationResult(
                            field='config.logging.file',
                            level=ValidationLevel.ERROR,
                            message=f'Log file path not accessible: {e}',
                            value=log_file,
                            suggestion='Check directory permissions'
                        ))
            
        except Exception as e:
            logger.error(f"❌ Logging config validation failed: {e}")
    
    def _validate_api_config(self, api_config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate API configuration."""
        try:
            # Check host
            if 'host' in api_config:
                host = api_config['host']
                if not isinstance(host, str):
                    report.add_result(ValidationResult(
                        field='config.api.host',
                        level=ValidationLevel.ERROR,
                        message='Host must be a string',
                        value=type(host).__name__,
                        expected='string'
                    ))
                else:
                    # Try to validate IP address
                    try:
                        ipaddress.ip_address(host)
                        report.add_success('config.api.host', f'Valid IP address: {host}')
                    except:
                        # Could be hostname
                        if re.match(r'^[a-zA-Z0-9.-]+$', host):
                            report.add_success('config.api.host', f'Valid hostname: {host}')
                        else:
                            report.add_result(ValidationResult(
                                field='config.api.host',
                                level=ValidationLevel.WARNING,
                                message=f'Host format questionable: {host}',
                                value=host,
                                suggestion='Verify host format'
                            ))
            
            # Check port
            if 'port' in api_config:
                port = api_config['port']
                if not isinstance(port, int) or not (1 <= port <= 65535):
                    report.add_result(ValidationResult(
                        field='config.api.port',
                        level=ValidationLevel.ERROR,
                        message='Port must be an integer between 1 and 65535',
                        value=port,
                        expected='1-65535'
                    ))
                else:
                    report.add_success('config.api.port', f'Valid port: {port}')
            
        except Exception as e:
            logger.error(f"❌ API config validation failed: {e}")
    
    def _validate_security_config(self, security_config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate security configuration."""
        try:
            # Check secret key
            if 'secret_key' in security_config:
                secret_key = security_config['secret_key']
                if not isinstance(secret_key, str):
                    report.add_result(ValidationResult(
                        field='config.security.secret_key',
                        level=ValidationLevel.ERROR,
                        message='Secret key must be a string',
                        value=type(secret_key).__name__,
                        expected='string'
                    ))
                elif len(secret_key) < 32:
                    report.add_result(ValidationResult(
                        field='config.security.secret_key',
                        level=ValidationLevel.WARNING,
                        message='Secret key should be at least 32 characters',
                        value=len(secret_key),
                        expected='>= 32 characters',
                        suggestion='Use a longer, more secure secret key'
                    ))
                else:
                    report.add_success('config.security.secret_key', 'Valid secret key length')
            
            # Check session timeout
            if 'session_timeout' in security_config:
                timeout = security_config['session_timeout']
                if not isinstance(timeout, int) or timeout <= 0:
                    report.add_result(ValidationResult(
                        field='config.security.session_timeout',
                        level=ValidationLevel.ERROR,
                        message='Session timeout must be a positive integer',
                        value=timeout,
                        expected='> 0'
                    ))
                else:
                    report.add_success('config.security.session_timeout', f'Valid timeout: {timeout}s')
            
        except Exception as e:
            logger.error(f"❌ Security config validation failed: {e}")
    
    def validate_model_config(self, model_config: Dict[str, Any], 
                             model_type: str) -> ValidationReport:
        """
        Validate model configuration.
        
        Args:
            model_config: Model configuration dictionary
            model_type: Type of model being configured
            
        Returns:
            Validation report
        """
        report = ValidationReport()
        
        try:
            # Common model parameters
            if 'learning_rate' in model_config:
                lr = model_config['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                    report.add_result(ValidationResult(
                        field='model.learning_rate',
                        level=ValidationLevel.ERROR,
                        message='Learning rate must be between 0 and 1',
                        value=lr,
                        expected='0 < lr <= 1'
                    ))
                else:
                    report.add_success('model.learning_rate', f'Valid learning rate: {lr}')
            
            if 'epochs' in model_config:
                epochs = model_config['epochs']
                if not isinstance(epochs, int) or epochs <= 0:
                    report.add_result(ValidationResult(
                        field='model.epochs',
                        level=ValidationLevel.ERROR,
                        message='Epochs must be a positive integer',
                        value=epochs,
                        expected='> 0'
                    ))
                else:
                    report.add_success('model.epochs', f'Valid epochs: {epochs}')
            
            # Model-specific validation
            if model_type == 'ensemble':
                self._validate_ensemble_config(model_config, report)
            elif model_type == 'neural_network':
                self._validate_neural_network_config(model_config, report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Model config validation failed: {e}")
            report.add_result(ValidationResult(
                field='validation.error',
                level=ValidationLevel.ERROR,
                message=f'Model config validation failed: {e}',
                value=str(e)
            ))
            return report
    
    def _validate_ensemble_config(self, config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate ensemble model configuration."""
        try:
            if 'n_estimators' in config:
                n_estimators = config['n_estimators']
                if not isinstance(n_estimators, int) or n_estimators <= 0:
                    report.add_result(ValidationResult(
                        field='model.n_estimators',
                        level=ValidationLevel.ERROR,
                        message='Number of estimators must be a positive integer',
                        value=n_estimators,
                        expected='> 0'
                    ))
                else:
                    report.add_success('model.n_estimators', f'Valid estimators: {n_estimators}')
            
            if 'max_depth' in config:
                max_depth = config['max_depth']
                if max_depth is not None and (not isinstance(max_depth, int) or max_depth <= 0):
                    report.add_result(ValidationResult(
                        field='model.max_depth',
                        level=ValidationLevel.ERROR,
                        message='Max depth must be None or a positive integer',
                        value=max_depth,
                        expected='None or > 0'
                    ))
                else:
                    report.add_success('model.max_depth', f'Valid max depth: {max_depth}')
            
        except Exception as e:
            logger.error(f"❌ Ensemble config validation failed: {e}")
    
    def _validate_neural_network_config(self, config: Dict[str, Any], report: ValidationReport) -> None:
        """Validate neural network configuration."""
        try:
            if 'hidden_layers' in config:
                hidden_layers = config['hidden_layers']
                if not isinstance(hidden_layers, list):
                    report.add_result(ValidationResult(
                        field='model.hidden_layers',
                        level=ValidationLevel.ERROR,
                        message='Hidden layers must be a list',
                        value=type(hidden_layers).__name__,
                        expected='list'
                    ))
                elif not all(isinstance(layer, int) and layer > 0 for layer in hidden_layers):
                    report.add_result(ValidationResult(
                        field='model.hidden_layers',
                        level=ValidationLevel.ERROR,
                        message='All hidden layer sizes must be positive integers',
                        value=hidden_layers,
                        expected='list of positive integers'
                    ))
                else:
                    report.add_success('model.hidden_layers', f'Valid hidden layers: {hidden_layers}')
            
            if 'activation' in config:
                activation = config['activation']
                valid_activations = ['relu', 'sigmoid', 'tanh', 'linear']
                if activation not in valid_activations:
                    report.add_result(ValidationResult(
                        field='model.activation',
                        level=ValidationLevel.ERROR,
                        message=f'Invalid activation function: {activation}',
                        value=activation,
                        expected=valid_activations
                    ))
                else:
                    report.add_success('model.activation', f'Valid activation: {activation}')
            
        except Exception as e:
            logger.error(f"❌ Neural network config validation failed: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check config validator health."""
        return True