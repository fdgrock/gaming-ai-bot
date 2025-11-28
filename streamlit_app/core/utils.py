"""
Utility functions for the lottery prediction system.

This module contains shared utility functions that are used across 
the application for data processing, file handling, game management,
and other common operations.
"""

import os
import json
import glob as _glob
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import pytz
import pandas as pd
import numpy as np
from pathlib import Path

from .exceptions import DataProcessingError, ValidationError
from .logger import app_log


def get_est_now() -> datetime:
    """Get current time in EST timezone."""
    est = pytz.timezone('America/New_York')
    return datetime.now(est)


def get_est_timestamp() -> str:
    """Get current timestamp string in EST timezone."""
    return get_est_now().strftime('%Y%m%d%H%M%S')


def get_est_isoformat() -> str:
    """Get current time in EST as ISO format string."""
    return get_est_now().isoformat()


def safe_load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Safely load JSON file with error handling for empty or malformed files.
    
    Args:
        file_path: Path to the JSON file to load
        
    Returns:
        Dictionary containing JSON data, or None if loading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Check if file is empty
        if not content:
            app_log(f"Empty JSON file found: {file_path}", "warning")
            return None
            
        # Try to parse JSON
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            app_log(f"Invalid JSON in file {file_path}: {e}", "warning")
            return None
            
    except FileNotFoundError:
        app_log(f"JSON file not found: {file_path}", "warning")
        return None
    except Exception as e:
        app_log(f"Error reading JSON file {file_path}: {e}", "error")
        return None


def safe_save_json(file_path: Union[str, Path], data: Dict[str, Any], 
                   ensure_dirs: bool = True) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        file_path: Path where to save the JSON file
        data: Dictionary to save as JSON
        ensure_dirs: Whether to create parent directories if they don't exist
        
    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        
        if ensure_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return True
        
    except Exception as e:
        app_log(f"Error saving JSON file {file_path}: {e}", "error")
        return False


def sanitize_game_name(name: str) -> str:
    """
    Create a filesystem-safe folder name from a human game name.
    
    Args:
        name: Human-readable game name
        
    Returns:
        Sanitized name safe for use as directory/file name
    """
    if not name:
        return "unknown_game"
        
    return name.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def get_available_games() -> List[str]:
    """
    Get list of available games - return the two supported lottery games.
    
    Returns:
        List of supported game names
    """
    supported_games = ["Lotto Max", "Lotto 6/49"]
    available_games = []
    
    # Check if data exists for each supported game
    for game in supported_games:
        game_dir = os.path.join("data", sanitize_game_name(game))
        if os.path.exists(game_dir):
            # Check if there are actual data files
            data_files = _glob.glob(os.path.join(game_dir, "*.csv"))
            if data_files:
                available_games.append(game)
    
    # Always return both games, even if no data exists (user can upload data)
    return supported_games


def compute_next_draw_date(game: str) -> date:
    """
    Return a reasonable next-draw date for the given game.
    
    Based on current lottery schedule analysis (as of November 17, 2025):
    - Lotto 6/49: Wednesday & Saturday | Last: 2025-11-15 (Sat) | Next: 2025-11-19 (Wed)
    - Lotto Max: Tuesday, Wednesday, Friday, Saturday | Last: 2025-11-14 (Fri) | Next: 2025-11-18 (Tue)
    
    Args:
        game: Game name
        
    Returns:
        Date of the next expected draw
    """
    today = date.today()
    
    # Game draw schedules (weekday numbers: Monday=0, Sunday=6)
    schedules = {
        "lotto_max": [1, 4],  # Tuesday(1), Friday(4)
        "lotto_6/49": [2, 5],  # Wednesday(2), Saturday(5)
        "lotto_6_49": [2, 5],  # Alternative naming
    }
    
    game_key = sanitize_game_name(game)
    draw_days = schedules.get(game_key, [2, 5])  # Default to Wed/Sat
    
    # Find next occurrence
    for day_offset in range(0, 14):
        candidate_date = today + timedelta(days=day_offset)
        if candidate_date.weekday() in draw_days:
            return candidate_date
            
    # Fallback: one week from today
    return today + timedelta(days=7)


def validate_lottery_numbers(numbers: List[int], game_name: str) -> bool:
    """
    Validate lottery numbers for a specific game.
    
    Args:
        numbers: List of lottery numbers to validate
        game_name: Name of the game for validation rules
        
    Returns:
        True if numbers are valid, False otherwise
        
    Raises:
        ValidationError: If validation fails with detailed reason
    """
    if not numbers:
        raise ValidationError("numbers", numbers, "Numbers list cannot be empty")
    
    game_key = sanitize_game_name(game_name)
    
    # Game-specific validation rules
    game_rules = {
        "lotto_max": {
            "count": 7,
            "min_number": 1,
            "max_number": 50,
            "unique": True
        },
        "lotto_6_49": {
            "count": 6,
            "min_number": 1,
            "max_number": 49,
            "unique": True
        }
    }
    
    rules = game_rules.get(game_key)
    if not rules:
        app_log(f"No validation rules for game: {game_name}", "warning")
        return True  # Allow unknown games
    
    # Check count
    if len(numbers) != rules["count"]:
        raise ValidationError(
            "numbers", 
            numbers, 
            f"Expected {rules['count']} numbers, got {len(numbers)}"
        )
    
    # Check range
    for num in numbers:
        if not isinstance(num, int):
            raise ValidationError("numbers", num, "All numbers must be integers")
        if not (rules["min_number"] <= num <= rules["max_number"]):
            raise ValidationError(
                "numbers", 
                num, 
                f"Number {num} is outside valid range {rules['min_number']}-{rules['max_number']}"
            )
    
    # Check uniqueness
    if rules["unique"] and len(set(numbers)) != len(numbers):
        raise ValidationError("numbers", numbers, "All numbers must be unique")
    
    return True


def format_lottery_numbers(numbers: List[int], sorted_output: bool = True) -> str:
    """
    Format lottery numbers for display.
    
    Args:
        numbers: List of lottery numbers
        sorted_output: Whether to sort numbers before formatting
        
    Returns:
        Formatted string representation of numbers
    """
    if not numbers:
        return ""
    
    if sorted_output:
        numbers = sorted(numbers)
    
    return ", ".join(str(num) for num in numbers)


def parse_lottery_numbers(numbers_str: str) -> List[int]:
    """
    Parse lottery numbers from string format.
    
    Args:
        numbers_str: String containing lottery numbers (comma or space separated)
        
    Returns:
        List of parsed integers
        
    Raises:
        ValidationError: If parsing fails
    """
    if not numbers_str or not numbers_str.strip():
        return []
    
    try:
        # Handle both comma and space separation
        numbers_str = numbers_str.replace(',', ' ')
        parts = numbers_str.strip().split()
        
        numbers = []
        for part in parts:
            part = part.strip()
            if part:
                numbers.append(int(part))
        
        return numbers
        
    except ValueError as e:
        raise ValidationError("numbers_str", numbers_str, f"Invalid number format: {e}")


def ensure_directory_exists(directory: Union[str, Path]) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        app_log(f"Failed to create directory {directory}: {e}", "error")
        return False


def clean_filename(filename: str) -> str:
    """
    Clean filename to remove invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename safe for filesystem
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove extra spaces and dots
    filename = filename.strip(' .')
    
    # Ensure it's not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB, or 0 if file doesn't exist
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except (OSError, FileNotFoundError):
        return 0.0


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change (positive for increase, negative for decrease)
    """
    if old_value == 0:
        return 100.0 if new_value > 0 else 0.0
    
    return ((new_value - old_value) / old_value) * 100


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def save_npz_and_meta(path_npz: Union[str, Path], X: np.ndarray, y: np.ndarray, 
                      meta: Dict[str, Any]) -> bool:
    """
    Save numpy arrays and metadata with error handling.
    
    Args:
        path_npz: Path for NPZ file
        X: Feature array
        y: Target array  
        meta: Metadata dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Save NPZ file
        try:
            np.savez_compressed(path_npz, X=X, y=y)
        except Exception as e:
            app_log(f"Failed to save NPZ file, trying joblib: {e}", "warning")
            import joblib
            joblib.dump({'X': X, 'y': y}, str(path_npz) + '.joblib')
        
        # Save metadata
        meta_path = str(path_npz) + '.meta.json'
        return safe_save_json(meta_path, meta)
        
    except Exception as e:
        app_log(f"Failed to save NPZ and metadata: {e}", "error")
        return False


def load_npz_and_meta(path_npz: Union[str, Path]) -> Optional[tuple]:
    """
    Load numpy arrays and metadata.
    
    Args:
        path_npz: Path to NPZ file
        
    Returns:
        Tuple of (X, y, metadata) or None if loading fails
    """
    try:
        # Try loading NPZ file
        try:
            data = np.load(path_npz)
            X, y = data['X'], data['y']
        except Exception:
            # Try joblib fallback
            import joblib
            data = joblib.load(str(path_npz) + '.joblib')
            X, y = data['X'], data['y']
        
        # Load metadata
        meta_path = str(path_npz) + '.meta.json'
        meta = safe_load_json(meta_path) or {}
        
        return X, y, meta
        
    except Exception as e:
        app_log(f"Failed to load NPZ and metadata: {e}", "error")
        return None


def retry_operation(operation, max_retries: int = 3, delay: float = 1.0, 
                   backoff_factor: float = 2.0) -> Any:
    """
    Retry an operation with exponential backoff.
    
    Args:
        operation: Callable to retry
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        
    Returns:
        Result of the operation
        
    Raises:
        Exception: The last exception if all retries fail
    """
    import time
    
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                app_log(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}), "
                       f"retrying in {current_delay:.1f}s: {e}", "warning")
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                app_log(f"Operation failed after {max_retries + 1} attempts: {e}", "error")
    
    raise last_exception


def save_npz_and_meta(path_npz: Union[str, Path], X: np.ndarray, y: np.ndarray, 
                      meta: Dict[str, Any]) -> bool:
    """
    Save training data to NPZ format with metadata (extracted from monolithic app.py).
    
    Args:
        path_npz: Path for the NPZ file
        X: Training features
        y: Training targets
        meta: Metadata dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        np.savez_compressed(path_npz, X=X, y=y)
        
        # Save metadata
        meta_path = str(path_npz) + '.meta.json'
        with open(meta_path, 'w', encoding='utf-8') as mf:
            json.dump(meta, mf, indent=2)
        
        return True
        
    except Exception as e:
        app_log(f"Failed to save npz file {path_npz}: {e}", "error")
        try:
            # Fallback to joblib
            import joblib
            joblib.dump({'X': X, 'y': y}, str(path_npz) + '.joblib')
            return True
        except Exception as e2:
            app_log(f"Failed to save joblib backup {path_npz}: {e2}", "error")
            return False


def compute_next_draw_date(game: str) -> date:
    """
    Compute next draw date for a given game.
    
    Based on current lottery schedule analysis (as of November 17, 2025):
    - Lotto 6/49: Wednesday & Saturday | Last: 2025-11-15 (Sat) | Next: 2025-11-19 (Wed)
    - Lotto Max: Tuesday, Wednesday, Friday, Saturday | Last: 2025-11-14 (Fri) | Next: 2025-11-18 (Tue)
    
    Args:
        game: Game name
        
    Returns:
        Date of next draw
    """
    today = date.today()
    
    # Game draw schedules
    schedules = {
        "lotto_max": [1, 4],     # Tuesday(1), Friday(4)
        "lotto_6/49": [2, 5],    # Wednesday(2), Saturday(5)
        "lotto_6_49": [2, 5],    # Alt naming
    }
    
    game_key = sanitize_game_name(game)
    draw_days = schedules.get(game_key, [2, 5])  # Default to Wed/Sat
    
    # Find next draw date
    for d in range(0, 14):  # Look ahead 2 weeks max
        candidate = today + timedelta(days=d)
        if candidate.weekday() in draw_days:
            return candidate
    
    # Fallback - next week
    return today + timedelta(days=7)


def validate_game_name(game_name: str) -> bool:
    """
    Validate if a game name is supported.
    
    Args:
        game_name: Game name to validate
        
    Returns:
        True if game is supported, False otherwise
    """
    supported_games = ["Lotto Max", "Lotto 6/49"]
    return game_name in supported_games


def format_numbers(numbers: Union[List[int], str], sort: bool = True) -> str:
    """
    Format number combination for display.
    
    Args:
        numbers: List of numbers or comma-separated string
        sort: Whether to sort the numbers
        
    Returns:
        Formatted number string
    """
    if isinstance(numbers, str):
        try:
            number_list = [int(x.strip()) for x in numbers.split(',')]
        except ValueError:
            return str(numbers)
    else:
        number_list = list(numbers)
    
    if sort:
        number_list.sort()
    
    return ', '.join(map(str, number_list))


def parse_numbers(numbers_str: str) -> List[int]:
    """
    Parse number string into list of integers.
    
    Args:
        numbers_str: Comma-separated number string
        
    Returns:
        List of integers
        
    Raises:
        ValidationError: If parsing fails
    """
    try:
        numbers = [int(x.strip()) for x in numbers_str.split(',') if x.strip()]
        return numbers
    except ValueError as e:
        from .exceptions import ValidationError
        raise ValidationError(f"Invalid number format: {numbers_str}", 
                            field_name="numbers", field_value=numbers_str)


def calculate_number_stats(numbers_list: List[List[int]], 
                          game_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistical information about number combinations.
    
    Args:
        numbers_list: List of number combinations
        game_config: Game configuration
        
    Returns:
        Dictionary with statistics
    """
    if not numbers_list:
        return {"total_combinations": 0, "most_frequent": [], "least_frequent": []}
    
    from collections import Counter
    
    # Flatten all numbers
    all_numbers = []
    for combination in numbers_list:
        all_numbers.extend(combination)
    
    # Calculate frequency
    frequency = Counter(all_numbers)
    most_frequent = frequency.most_common(10)
    least_frequent = frequency.most_common()[-10:] if len(frequency) >= 10 else []
    
    return {
        "total_combinations": len(numbers_list),
        "most_frequent": most_frequent,
        "least_frequent": least_frequent,
        "total_numbers_drawn": len(all_numbers),
        "unique_numbers": len(frequency),
        "average_frequency": len(all_numbers) / len(frequency) if frequency else 0
    }


def validate_number_combination(numbers: List[int], 
                               game_config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a number combination against game rules.
    
    Args:
        numbers: List of numbers to validate
        game_config: Game configuration with validation rules
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not numbers:
        return False, "No numbers provided"
    
    # Check count
    expected_count = game_config.get("number_count", 6)
    if len(numbers) != expected_count:
        return False, f"Expected {expected_count} numbers, got {len(numbers)}"
    
    # Check range
    number_range = game_config.get("number_range", [1, 49])
    min_num, max_num = number_range
    
    for num in numbers:
        if not isinstance(num, int):
            return False, f"All numbers must be integers, got {type(num)}"
        if num < min_num or num > max_num:
            return False, f"Number {num} outside valid range {min_num}-{max_num}"
    
    # Check uniqueness
    if game_config.get("validation_rules", {}).get("unique_numbers", True):
        if len(set(numbers)) != len(numbers):
            return False, "All numbers must be unique"
    
    return True, "Valid combination"


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with default fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer with default fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    try:
        if value is None or value == '':
            return default
        return int(float(value))  # Handle string floats like "1.0"
    except (ValueError, TypeError):
        return default


def format_accuracy(accuracy: Union[float, str]) -> str:
    """
    Format accuracy value for display.
    
    Args:
        accuracy: Accuracy value (0-1 range or percentage string)
        
    Returns:
        Formatted percentage string
    """
    try:
        if isinstance(accuracy, str):
            # Try to parse percentage string
            if accuracy.endswith('%'):
                return accuracy
            acc_val = float(accuracy)
        else:
            acc_val = float(accuracy)
        
        # Convert to percentage
        if acc_val <= 1.0:
            return f"{acc_val * 100:.1f}%"
        else:
            return f"{acc_val:.1f}%"
            
    except (ValueError, TypeError):
        return "N/A"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def extract_version_from_path(file_path: Union[str, Path]) -> Optional[str]:
    """
    Extract version number from file path or name.
    
    Args:
        file_path: Path to examine
        
    Returns:
        Version string if found, None otherwise
    """
    import re
    
    path_str = str(file_path)
    
    # Look for version patterns like v1.2.3, version_1.2, etc.
    version_patterns = [
        r'v(\d+\.?\d*\.?\d*)',
        r'version[_-]?(\d+\.?\d*\.?\d*)',
        r'(\d+\.?\d*\.?\d*)',
    ]
    
    for pattern in version_patterns:
        match = re.search(pattern, path_str, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def clean_prediction_data(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean prediction data by removing numpy types and ensuring JSON serializable.
    
    Args:
        prediction_data: Raw prediction data
        
    Returns:
        Cleaned prediction data
    """
    def clean_numpy_types(obj):
        """Recursively clean numpy types from nested data."""
        if isinstance(obj, dict):
            return {k: clean_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    return clean_numpy_types(prediction_data)


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_age_days(file_path: Union[str, Path]) -> float:
    """
    Get age of file in days.
    
    Args:
        file_path: Path to file
        
    Returns:
        Age in days, or -1 if file doesn't exist
    """
    try:
        from datetime import datetime
        import os
        
        file_path = Path(file_path)
        if not file_path.exists():
            return -1
        
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        now = datetime.now()
        age = (now - file_time).total_seconds() / 86400  # seconds to days
        
        return age
        
    except Exception:
        return -1


def is_file_recent(file_path: Union[str, Path], max_age_hours: float = 24) -> bool:
    """
    Check if file was modified recently.
    
    Args:
        file_path: Path to file
        max_age_hours: Maximum age in hours to consider recent
        
    Returns:
        True if file is recent, False otherwise
    """
    age_days = get_file_age_days(file_path)
    if age_days < 0:
        return False
    
    max_age_days = max_age_hours / 24
    return age_days <= max_age_days