"""
Training Logger Utility
Provides real-time logging and progress tracking for model training with Streamlit integration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import json


class TrainingLogHandler(logging.Handler):
    """Custom logging handler that stores logs in memory for UI display."""
    
    def __init__(self):
        super().__init__()
        self.logs: List[Dict[str, Any]] = []
        self.max_logs = 1000  # Limit to prevent memory bloat
    
    def emit(self, record):
        """Store log record."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": self.format(record),
                "module": record.module
            }
            self.logs.append(log_entry)
            
            # Keep only recent logs
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs:]
        except Exception:
            self.handleError(record)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all stored logs."""
        return self.logs.copy()
    
    def clear_logs(self):
        """Clear all stored logs."""
        self.logs = []
    
    def get_latest_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get the latest N logs."""
        return self.logs[-count:] if self.logs else []


class TrainingProgressTracker:
    """Track training progress with per-model and overall completion percentages."""
    
    def __init__(self, total_models: int, trainer_name: str):
        self.total_models = total_models
        self.trainer_name = trainer_name
        self.completed_models = 0
        self.current_model = None
        self.current_model_progress = 0.0
        self.model_details: Dict[str, Dict] = {}
        self.start_time = datetime.now()
    
    def start_model(self, model_name: str):
        """Mark a model training as started."""
        self.current_model = model_name
        self.current_model_progress = 0.0
        self.model_details[model_name] = {
            "status": "in_progress",
            "start_time": datetime.now().isoformat(),
            "progress": 0.0,
            "completion_time": None
        }
    
    def update_model_progress(self, progress_pct: float):
        """Update current model progress (0-100)."""
        self.current_model_progress = min(100.0, max(0.0, progress_pct))
        if self.current_model and self.current_model in self.model_details:
            self.model_details[self.current_model]["progress"] = self.current_model_progress
    
    def complete_model(self, model_name: str = None):
        """Mark a model as completed."""
        model = model_name or self.current_model
        if model:
            self.completed_models += 1
            if model in self.model_details:
                self.model_details[model]["status"] = "completed"
                self.model_details[model]["completion_time"] = datetime.now().isoformat()
            self.current_model = None
            self.current_model_progress = 0.0
    
    def fail_model(self, model_name: str, error: str):
        """Mark a model as failed."""
        if model_name in self.model_details:
            self.model_details[model_name]["status"] = "failed"
            self.model_details[model_name]["error"] = error
    
    def get_overall_progress(self) -> float:
        """Get overall progress percentage (0-100)."""
        if self.total_models == 0:
            return 0.0
        
        # Completed models count as 100%, current model counts as its progress
        completed_pct = (self.completed_models / self.total_models) * 100
        current_pct = (self.current_model_progress / self.total_models)
        return min(100.0, completed_pct + current_pct)
    
    def get_per_model_progress(self) -> Dict[str, Dict]:
        """Get progress for each model."""
        return self.model_details.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete progress summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            "trainer": self.trainer_name,
            "total_models": self.total_models,
            "completed_models": self.completed_models,
            "overall_progress_pct": self.get_overall_progress(),
            "current_model": self.current_model,
            "current_model_progress_pct": self.current_model_progress,
            "elapsed_seconds": elapsed,
            "model_details": self.get_per_model_progress(),
            "timestamp": datetime.now().isoformat()
        }


def setup_training_logger(name: str, log_file: Optional[Path] = None) -> tuple:
    """
    Setup a logger with both file and in-memory handlers.
    
    Returns:
        Tuple of (logger, TrainingLogHandler)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add in-memory handler
    memory_handler = TrainingLogHandler()
    memory_handler.setFormatter(detailed_formatter)
    logger.addHandler(memory_handler)
    
    # Add file handler if log file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    return logger, memory_handler
