"""
Validation and Testing Utilities

Comprehensive validation tools for data integrity, system health,
and testing support for the lottery prediction system.
"""

import json
import re
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import logging


class DataValidator:
    """Advanced data validation for lottery prediction system."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = logging.getLogger(f"{__name__}.DataValidator")
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different data types."""
        return {
            "lottery_numbers": {
                "powerball": {
                    "main_numbers": {"count": 5, "min": 1, "max": 69},
                    "bonus_numbers": {"count": 1, "min": 1, "max": 26}
                },
                "mega_millions": {
                    "main_numbers": {"count": 5, "min": 1, "max": 70},
                    "bonus_numbers": {"count": 1, "min": 1, "max": 25}
                }
            },
            "confidence_score": {"min": 0.0, "max": 1.0},
            "date_format": r"^\d{4}-\d{2}-\d{2}$",
            "strategy_types": ["conservative", "balanced", "aggressive"],
            "engine_types": ["ensemble", "neural_network", "pattern", "frequency"]
        }
    
    def validate_lottery_numbers(self, numbers: List[int], game: str, is_bonus: bool = False) -> Dict[str, Any]:
        """Validate lottery numbers for specific game."""
        result = {"valid": True, "errors": [], "warnings": []}
        
        if game not in self.validation_rules["lottery_numbers"]:
            result["errors"].append(f"Unknown game type: {game}")
            result["valid"] = False
            return result
        
        rules = self.validation_rules["lottery_numbers"][game]
        number_type = "bonus_numbers" if is_bonus else "main_numbers"
        rule = rules[number_type]
        
        # Check count
        if len(numbers) != rule["count"]:
            result["errors"].append(
                f"Invalid {number_type} count: expected {rule['count']}, got {len(numbers)}"
            )
            result["valid"] = False
        
        # Check range and duplicates
        seen_numbers = set()
        for num in numbers:
            if not isinstance(num, int):
                result["errors"].append(f"Non-integer number found: {num}")
                result["valid"] = False
                continue
            
            if num < rule["min"] or num > rule["max"]:
                result["errors"].append(
                    f"Number {num} out of range: must be between {rule['min']} and {rule['max']}"
                )
                result["valid"] = False
            
            if num in seen_numbers:
                result["errors"].append(f"Duplicate number found: {num}")
                result["valid"] = False
            
            seen_numbers.add(num)
        
        return result
    
    def validate_prediction_structure(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the structure and content of a prediction."""
        result = {"valid": True, "errors": [], "warnings": [], "score": 100}
        
        # Required fields
        required_fields = ["numbers", "confidence", "strategy", "engine", "game"]
        for field in required_fields:
            if field not in prediction:
                result["errors"].append(f"Missing required field: {field}")
                result["valid"] = False
                result["score"] -= 20
        
        # Validate numbers
        if "numbers" in prediction and "game" in prediction:
            numbers_validation = self.validate_lottery_numbers(
                prediction["numbers"], 
                prediction["game"]
            )
            if not numbers_validation["valid"]:
                result["errors"].extend(numbers_validation["errors"])
                result["valid"] = False
                result["score"] -= 30
        
        # Validate bonus numbers
        if "bonus" in prediction and prediction["bonus"] is not None and "game" in prediction:
            bonus_validation = self.validate_lottery_numbers(
                [prediction["bonus"]], 
                prediction["game"], 
                is_bonus=True
            )
            if not bonus_validation["valid"]:
                result["errors"].extend(bonus_validation["errors"])
                result["valid"] = False
                result["score"] -= 15
        
        # Validate confidence
        if "confidence" in prediction:
            conf = prediction["confidence"]
            rules = self.validation_rules["confidence_score"]
            if not isinstance(conf, (int, float)) or conf < rules["min"] or conf > rules["max"]:
                result["errors"].append(f"Invalid confidence: must be between {rules['min']} and {rules['max']}")
                result["valid"] = False
                result["score"] -= 25
        
        # Validate strategy
        if "strategy" in prediction:
            strategy = prediction["strategy"]
            if strategy not in self.validation_rules["strategy_types"]:
                result["warnings"].append(f"Unknown strategy: {strategy}")
                result["score"] -= 5
        
        # Validate engine
        if "engine" in prediction:
            engine = prediction["engine"]
            if engine not in self.validation_rules["engine_types"]:
                result["warnings"].append(f"Unknown engine: {engine}")
                result["score"] -= 5
        
        # Validate timestamp if present
        if "generated_at" in prediction:
            timestamp = prediction["generated_at"]
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                result["warnings"].append("Invalid timestamp format")
                result["score"] -= 5
        
        return result
    
    def validate_historical_data(self, data: List[Dict[str, Any]], game: str) -> Dict[str, Any]:
        """Validate historical lottery data for consistency and completeness."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "total_records": len(data),
                "valid_records": 0,
                "invalid_records": 0,
                "date_range": None,
                "missing_fields": {},
                "data_quality_score": 0
            }
        }
        
        if not data:
            result["errors"].append("No data provided")
            result["valid"] = False
            return result
        
        valid_count = 0
        dates = []
        field_counts = {}
        
        for i, record in enumerate(data):
            record_valid = True
            
            # Check required fields
            required_fields = ["draw_date", "numbers", "game"]
            for field in required_fields:
                if field not in record:
                    result["errors"].append(f"Record {i}: Missing field '{field}'")
                    result["statistics"]["missing_fields"][field] = result["statistics"]["missing_fields"].get(field, 0) + 1
                    record_valid = False
                else:
                    field_counts[field] = field_counts.get(field, 0) + 1
            
            # Validate date
            if "draw_date" in record:
                if not re.match(self.validation_rules["date_format"], record["draw_date"]):
                    result["errors"].append(f"Record {i}: Invalid date format")
                    record_valid = False
                else:
                    try:
                        date_obj = datetime.strptime(record["draw_date"], "%Y-%m-%d")
                        dates.append(date_obj)
                    except ValueError:
                        result["errors"].append(f"Record {i}: Invalid date value")
                        record_valid = False
            
            # Validate numbers
            if "numbers" in record:
                try:
                    numbers = record["numbers"]
                    if isinstance(numbers, str):
                        numbers = json.loads(numbers)
                    
                    numbers_validation = self.validate_lottery_numbers(numbers, game)
                    if not numbers_validation["valid"]:
                        result["errors"].extend([f"Record {i}: {error}" for error in numbers_validation["errors"]])
                        record_valid = False
                        
                except (json.JSONDecodeError, TypeError):
                    result["errors"].append(f"Record {i}: Invalid numbers format")
                    record_valid = False
            
            if record_valid:
                valid_count += 1
        
        # Update statistics
        result["statistics"]["valid_records"] = valid_count
        result["statistics"]["invalid_records"] = len(data) - valid_count
        
        if dates:
            result["statistics"]["date_range"] = {
                "start": min(dates).strftime("%Y-%m-%d"),
                "end": max(dates).strftime("%Y-%m-%d"),
                "span_days": (max(dates) - min(dates)).days
            }
        
        # Calculate data quality score
        quality_score = (valid_count / len(data)) * 100
        result["statistics"]["data_quality_score"] = round(quality_score, 2)
        
        if quality_score < 90:
            result["warnings"].append(f"Data quality score is low: {quality_score:.1f}%")
        
        if result["statistics"]["invalid_records"] > 0:
            result["valid"] = False
        
        return result
    
    def validate_model_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model performance metrics."""
        result = {"valid": True, "errors": [], "warnings": [], "analysis": {}}
        
        required_metrics = ["accuracy", "precision", "recall", "f1_score"]
        
        for metric in required_metrics:
            if metric not in performance_data:
                result["errors"].append(f"Missing performance metric: {metric}")
                result["valid"] = False
                continue
            
            value = performance_data[metric]
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                result["errors"].append(f"Invalid {metric} value: must be between 0 and 1")
                result["valid"] = False
            elif value < 0.5:
                result["warnings"].append(f"Low {metric} value: {value:.3f}")
        
        # Performance analysis
        if result["valid"]:
            accuracy = performance_data.get("accuracy", 0)
            precision = performance_data.get("precision", 0)
            recall = performance_data.get("recall", 0)
            f1 = performance_data.get("f1_score", 0)
            
            result["analysis"] = {
                "overall_performance": "excellent" if accuracy > 0.8 else "good" if accuracy > 0.6 else "poor",
                "balanced_metrics": abs(precision - recall) < 0.1,
                "f1_accuracy_correlation": abs(f1 - accuracy) < 0.1,
                "recommendations": []
            }
            
            if precision > recall + 0.1:
                result["analysis"]["recommendations"].append("Model is conservative - consider increasing recall")
            elif recall > precision + 0.1:
                result["analysis"]["recommendations"].append("Model is aggressive - consider increasing precision")
            
            if accuracy < 0.7:
                result["analysis"]["recommendations"].append("Consider retraining model with more data")
        
        return result


class TestDataGenerator:
    """Generate test data for development and testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize test data generator with random seed."""
        import random
        random.seed(seed)
        self.random = random
        self.logger = logging.getLogger(f"{__name__}.TestDataGenerator")
    
    def generate_historical_data(self, game: str, days: int = 365, start_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate realistic historical lottery data."""
        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = datetime.now() - timedelta(days=days)
        
        # Game configurations
        game_configs = {
            "powerball": {"main_count": 5, "main_max": 69, "bonus_max": 26},
            "mega_millions": {"main_count": 5, "main_max": 70, "bonus_max": 25}
        }
        
        if game not in game_configs:
            raise ValueError(f"Unknown game: {game}")
        
        config = game_configs[game]
        data = []
        
        # Generate draws (typically 3 times per week)
        draw_dates = []
        current_date = start
        while current_date <= datetime.now() and len(draw_dates) < days:
            # Powerball: Monday, Wednesday, Saturday
            # Mega Millions: Tuesday, Friday
            if game == "powerball" and current_date.weekday() in [0, 2, 5]:
                draw_dates.append(current_date)
            elif game == "mega_millions" and current_date.weekday() in [1, 4]:
                draw_dates.append(current_date)
            
            current_date += timedelta(days=1)
        
        for i, draw_date in enumerate(draw_dates):
            # Generate main numbers
            main_numbers = sorted(self.random.sample(range(1, config["main_max"] + 1), config["main_count"]))
            
            # Generate bonus number
            bonus = self.random.randint(1, config["bonus_max"])
            
            # Generate realistic jackpot (grows over time)
            base_jackpot = 20000000  # $20M base
            growth_factor = 1 + (i % 10) * 0.1  # Grows every 10 draws
            jackpot = int(base_jackpot * growth_factor * self.random.uniform(0.8, 2.5))
            
            # Generate multiplier (some games have this)
            multiplier = self.random.choice([1, 2, 3, 4, 5, 10]) if self.random.random() < 0.3 else 1
            
            record = {
                "id": i + 1,
                "game": game,
                "draw_date": draw_date.strftime("%Y-%m-%d"),
                "numbers": json.dumps(main_numbers),
                "bonus": bonus,
                "jackpot": jackpot,
                "multiplier": multiplier,
                "winners": {
                    "jackpot": 1 if self.random.random() < 0.01 else 0,
                    "match_5": self.random.randint(0, 5),
                    "match_4": self.random.randint(5, 50),
                    "match_3": self.random.randint(50, 500)
                }
            }
            
            data.append(record)
        
        self.logger.info(f"Generated {len(data)} historical records for {game}")
        return data
    
    def generate_predictions(self, game: str, count: int = 10, strategies: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Generate realistic prediction data."""
        if strategies is None:
            strategies = ["conservative", "balanced", "aggressive"]
        
        engines = ["ensemble", "neural_network", "pattern", "frequency"]
        predictions = []
        
        # Game configurations
        game_configs = {
            "powerball": {"main_count": 5, "main_max": 69, "bonus_max": 26},
            "mega_millions": {"main_count": 5, "main_max": 70, "bonus_max": 25}
        }
        
        config = game_configs.get(game, {"main_count": 5, "main_max": 69, "bonus_max": 26})
        
        for i in range(count):
            strategy = self.random.choice(strategies)
            engine = self.random.choice(engines)
            
            # Generate numbers
            main_numbers = sorted(self.random.sample(range(1, config["main_max"] + 1), config["main_count"]))
            bonus = self.random.randint(1, config["bonus_max"])
            
            # Generate confidence based on strategy
            base_confidence = {"conservative": 0.65, "balanced": 0.72, "aggressive": 0.78}
            confidence = base_confidence[strategy] + self.random.uniform(-0.05, 0.05)
            confidence = max(0.0, min(1.0, confidence))
            
            prediction = {
                "id": i + 1,
                "game": game,
                "numbers": main_numbers,
                "bonus": bonus,
                "confidence": round(confidence, 3),
                "strategy": strategy,
                "engine": engine,
                "generated_at": (datetime.now() - timedelta(minutes=self.random.randint(0, 1440))).isoformat(),
                "metadata": {
                    "model_version": f"{self.random.randint(1, 3)}.{self.random.randint(0, 9)}",
                    "features_used": self.random.sample(
                        ["frequency", "patterns", "trends", "statistics", "seasonal"], 
                        self.random.randint(2, 4)
                    )
                }
            }
            
            predictions.append(prediction)
        
        self.logger.info(f"Generated {count} predictions for {game}")
        return predictions
    
    def generate_performance_data(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """Generate realistic model performance data."""
        # Base performance varies by model type
        base_performance = {
            "ensemble": {"accuracy": 0.75, "precision": 0.72, "recall": 0.68},
            "neural_network": {"accuracy": 0.68, "precision": 0.65, "recall": 0.70},
            "pattern": {"accuracy": 0.62, "precision": 0.60, "recall": 0.64},
            "frequency": {"accuracy": 0.58, "precision": 0.55, "recall": 0.60}
        }
        
        base = base_performance.get(model_name, base_performance["ensemble"])
        
        # Add some realistic variation
        performance = {}
        for metric, value in base.items():
            performance[metric] = max(0.0, min(1.0, value + self.random.uniform(-0.05, 0.05)))
        
        # Calculate F1 score
        performance["f1_score"] = 2 * (performance["precision"] * performance["recall"]) / (performance["precision"] + performance["recall"])
        
        # Generate time series data
        daily_performance = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d")
            daily_perf = {
                "date": date,
                "accuracy": max(0.0, min(1.0, performance["accuracy"] + self.random.uniform(-0.1, 0.1))),
                "predictions_made": self.random.randint(10, 100),
                "response_time": self.random.uniform(0.5, 3.0)
            }
            daily_performance.append(daily_perf)
        
        return {
            "model_name": model_name,
            "overall_performance": performance,
            "daily_performance": daily_performance,
            "training_data": {
                "samples": self.random.randint(1000, 10000),
                "features": self.random.randint(10, 50),
                "last_trained": (datetime.now() - timedelta(days=self.random.randint(1, 30))).isoformat()
            }
        }
    
    def generate_user_preferences(self, user_count: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic user preference data."""
        preferences_list = []
        
        games = ["powerball", "mega_millions"]
        strategies = ["conservative", "balanced", "aggressive"]
        themes = ["light", "dark", "auto"]
        
        for i in range(user_count):
            user_id = f"user_{i+1:04d}"
            
            # Generate some favorite numbers
            favorite_numbers = sorted(self.random.sample(range(1, 50), self.random.randint(3, 7)))
            
            preferences = {
                "user_id": user_id,
                "preferences": {
                    "default_game": self.random.choice(games),
                    "default_strategy": self.random.choice(strategies),
                    "number_of_predictions": self.random.choice([3, 5, 7, 10]),
                    "exclude_recent_numbers": self.random.choice([True, False]),
                    "favorite_numbers": favorite_numbers,
                    "notification_settings": {
                        "email_enabled": self.random.choice([True, False]),
                        "sms_enabled": self.random.choice([True, False]),
                        "push_enabled": self.random.choice([True, False])
                    },
                    "display_settings": {
                        "theme": self.random.choice(themes),
                        "show_confidence": self.random.choice([True, False]),
                        "show_statistics": self.random.choice([True, False]),
                        "chart_type": self.random.choice(["bar", "line", "pie"])
                    }
                },
                "created_at": (datetime.now() - timedelta(days=self.random.randint(1, 365))).isoformat(),
                "last_active": (datetime.now() - timedelta(hours=self.random.randint(1, 168))).isoformat()
            }
            
            preferences_list.append(preferences)
        
        self.logger.info(f"Generated preferences for {user_count} users")
        return preferences_list


# Export validation and test utilities
__all__ = ["DataValidator", "TestDataGenerator"]