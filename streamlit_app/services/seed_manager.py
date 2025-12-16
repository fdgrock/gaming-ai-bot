"""
Automatic Seed Management System for Prediction Generation

Manages seed allocation across different model types to ensure:
1. No duplicate seeds across different models (prevents identical predictions)
2. Automatic tracking and incrementation
3. Persistent seed state across sessions
4. Easy reset functionality
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class SeedManager:
    """
    Manages automatic seed allocation for prediction generation.
    
    Seed Ranges (0-999):
    - XGBoost: 0-99
    - CatBoost: 100-199
    - LightGBM: 200-299
    - LSTM: 300-399
    - CNN: 400-499
    - Transformer: 500-599
    - Ensemble: 600-699
    - Reserved: 700-999 (future models)
    """
    
    # Define seed ranges for each model type
    SEED_RANGES = {
        "xgboost": {"start": 0, "end": 99},
        "catboost": {"start": 100, "end": 199},
        "lightgbm": {"start": 200, "end": 299},
        "lstm": {"start": 300, "end": 399},
        "cnn": {"start": 400, "end": 499},
        "transformer": {"start": 500, "end": 599},
        "ensemble": {"start": 600, "end": 699}
    }
    
    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize seed manager.
        
        Args:
            state_file: Path to seed state file. If None, uses default location.
        """
        if state_file is None:
            state_file = Path(__file__).parent.parent.parent / "data" / "seed_state.json"
        
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load seed state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Ensure all model types have entries
                    for model_type in self.SEED_RANGES.keys():
                        if model_type not in state:
                            state[model_type] = self.SEED_RANGES[model_type]["start"]
                    return state
            except (json.JSONDecodeError, IOError):
                pass
        
        # Initialize with start values
        return {
            model_type: range_info["start"]
            for model_type, range_info in self.SEED_RANGES.items()
        }
    
    def _save_state(self) -> None:
        """Save seed state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save seed state: {e}")
    
    def get_next_seed(self, model_type: str) -> int:
        """
        Get the next available seed for a model type.
        
        Args:
            model_type: Model type (xgboost, catboost, lstm, etc.)
        
        Returns:
            Next seed value for this model type
        """
        model_type = model_type.lower()
        
        if model_type not in self.SEED_RANGES:
            raise ValueError(f"Unknown model type: {model_type}. Valid types: {list(self.SEED_RANGES.keys())}")
        
        # Get current seed
        current_seed = self.state.get(model_type, self.SEED_RANGES[model_type]["start"])
        
        # Increment for next time
        range_info = self.SEED_RANGES[model_type]
        next_seed = current_seed + 1
        
        # Wrap around if exceeded range
        if next_seed > range_info["end"]:
            next_seed = range_info["start"]
        
        # Update state
        self.state[model_type] = next_seed
        self._save_state()
        
        return current_seed
    
    def peek_next_seed(self, model_type: str) -> int:
        """
        Preview the next seed without consuming it.
        
        Args:
            model_type: Model type (xgboost, catboost, lstm, etc.)
        
        Returns:
            Next seed value (without incrementing)
        """
        model_type = model_type.lower()
        
        if model_type not in self.SEED_RANGES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.state.get(model_type, self.SEED_RANGES[model_type]["start"])
    
    def reset_model_seeds(self, model_type: str) -> None:
        """
        Reset seeds for a specific model type back to start.
        
        Args:
            model_type: Model type to reset
        """
        model_type = model_type.lower()
        
        if model_type not in self.SEED_RANGES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.state[model_type] = self.SEED_RANGES[model_type]["start"]
        self._save_state()
    
    def reset_all_seeds(self) -> None:
        """Reset all model seeds back to their starting values."""
        self.state = {
            model_type: range_info["start"]
            for model_type, range_info in self.SEED_RANGES.items()
        }
        self._save_state()
    
    def get_seed_info(self, model_type: str) -> Dict:
        """
        Get detailed seed information for a model type.
        
        Returns:
            Dict with current, next, range_start, range_end, usage_percentage
        """
        model_type = model_type.lower()
        
        if model_type not in self.SEED_RANGES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        range_info = self.SEED_RANGES[model_type]
        current = self.state.get(model_type, range_info["start"])
        
        # Calculate usage
        total_range = range_info["end"] - range_info["start"] + 1
        used = (current - range_info["start"]) % total_range
        usage_pct = (used / total_range) * 100
        
        return {
            "model_type": model_type,
            "current_seed": current,
            "range_start": range_info["start"],
            "range_end": range_info["end"],
            "total_capacity": total_range,
            "seeds_used": used,
            "usage_percentage": round(usage_pct, 1)
        }
    
    def get_all_seed_info(self) -> Dict[str, Dict]:
        """Get seed information for all model types."""
        return {
            model_type: self.get_seed_info(model_type)
            for model_type in self.SEED_RANGES.keys()
        }
    
    def export_state(self) -> str:
        """Export current state as formatted string."""
        info = self.get_all_seed_info()
        lines = ["Seed Manager State", "=" * 50]
        
        for model_type, details in info.items():
            lines.append(
                f"{model_type.upper():12s} | "
                f"Current: {details['current_seed']:3d} | "
                f"Range: {details['range_start']:3d}-{details['range_end']:3d} | "
                f"Used: {details['seeds_used']}/{details['total_capacity']} ({details['usage_percentage']:.1f}%)"
            )
        
        return "\n".join(lines)
