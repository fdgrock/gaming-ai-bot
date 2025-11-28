"""
Test fixtures and sample data for lottery prediction system tests.

This module contains sample data files and fixtures used across
different test modules.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Set random seed for reproducible test data
np.random.seed(42)

# Sample lottery game configurations
SAMPLE_GAME_CONFIGS = {
    "powerball": {
        "name": "Powerball",
        "code": "powerball",
        "num_numbers": 5,
        "max_number": 69,
        "min_number": 1,
        "bonus_numbers": 1,
        "bonus_max": 26,
        "draw_days": ["monday", "wednesday", "saturday"],
        "draw_time": "22:59",
        "timezone": "US/Eastern"
    },
    "mega_millions": {
        "name": "Mega Millions",
        "code": "mega_millions",
        "num_numbers": 5,
        "max_number": 70,
        "min_number": 1,
        "bonus_numbers": 1,
        "bonus_max": 25,
        "draw_days": ["tuesday", "friday"],
        "draw_time": "23:00",
        "timezone": "US/Eastern"
    },
    "test_game": {
        "name": "Test Game",
        "code": "test_game",
        "num_numbers": 6,
        "max_number": 49,
        "min_number": 1,
        "bonus_numbers": 0,
        "bonus_max": 0,
        "draw_days": ["sunday"],
        "draw_time": "20:00",
        "timezone": "UTC"
    }
}

# Sample historical lottery data
def generate_sample_historical_data(game="powerball", days=365):
    """Generate sample historical lottery data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    config = SAMPLE_GAME_CONFIGS[game]
    dates = pd.date_range(start=start_date, end=end_date, freq='3D')
    
    data = []
    for i, date in enumerate(dates):
        # Generate realistic lottery numbers
        main_numbers = sorted(
            np.random.choice(
                range(config["min_number"], config["max_number"] + 1),
                config["num_numbers"],
                replace=False
            ).tolist()
        )
        
        # Add bonus number if applicable
        bonus = None
        if config["bonus_numbers"] > 0:
            bonus = np.random.randint(1, config["bonus_max"] + 1)
        
        # Generate realistic jackpot
        base_jackpot = 20000000  # $20M base
        growth = np.random.exponential(0.1) * 100000000  # Exponential growth
        jackpot = int(base_jackpot + growth)
        
        data.append({
            "id": i + 1,
            "draw_date": date.strftime("%Y-%m-%d"),
            "game": game,
            "numbers": json.dumps(main_numbers),
            "bonus": bonus,
            "jackpot": jackpot,
            "multiplier": np.random.choice([1, 2, 3, 4, 5, 10], p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02]),
            "winners": {
                "jackpot": 1 if np.random.random() < 0.01 else 0,
                "match_5": np.random.randint(0, 5),
                "match_4": np.random.randint(5, 50),
                "match_3": np.random.randint(50, 500)
            }
        })
    
    return data

# Sample prediction data
SAMPLE_PREDICTIONS = [
    {
        "id": 1,
        "numbers": [5, 12, 20, 24, 29],
        "bonus": 4,
        "confidence": 0.75,
        "strategy": "balanced",
        "engine": "ensemble",
        "game": "powerball",
        "generated_at": "2023-12-01T10:30:00",
        "metadata": {
            "model_version": "1.0",
            "training_data_size": 1000,
            "features_used": ["frequency", "patterns", "trends"]
        }
    },
    {
        "id": 2,
        "numbers": [8, 15, 27, 33, 48],
        "bonus": 12,
        "confidence": 0.68,
        "strategy": "aggressive",
        "engine": "neural_network",
        "game": "powerball",
        "generated_at": "2023-12-01T10:31:00",
        "metadata": {
            "model_version": "2.1",
            "training_epochs": 100,
            "loss": 0.02
        }
    },
    {
        "id": 3,
        "numbers": [11, 22, 33, 44, 55],
        "bonus": 26,
        "confidence": 0.62,
        "strategy": "conservative",
        "engine": "pattern",
        "game": "powerball",
        "generated_at": "2023-12-01T10:32:00",
        "metadata": {
            "pattern_score": 0.8,
            "frequency_weight": 0.6
        }
    }
]

# Sample statistics data
SAMPLE_STATISTICS = {
    "frequency_analysis": {
        str(i): np.random.randint(10, 100) for i in range(1, 70)
    },
    "number_statistics": {
        "most_frequent": [
            {"number": 23, "frequency": 45, "percentage": 12.5},
            {"number": 12, "frequency": 42, "percentage": 11.7},
            {"number": 45, "frequency": 40, "percentage": 11.1}
        ],
        "least_frequent": [
            {"number": 3, "frequency": 8, "percentage": 2.2},
            {"number": 14, "frequency": 9, "percentage": 2.5},
            {"number": 67, "frequency": 10, "percentage": 2.8}
        ]
    },
    "pattern_analysis": {
        "consecutive_numbers": {
            "frequency": 25,
            "percentage": 6.9,
            "examples": [[1, 2, 3, 15, 25], [22, 23, 24, 35, 45]]
        },
        "even_odd_distribution": {
            "all_even": {"count": 5, "percentage": 1.4},
            "all_odd": {"count": 8, "percentage": 2.2},
            "mixed": {"count": 348, "percentage": 96.4}
        },
        "sum_ranges": {
            "low_sum": {"range": "50-100", "count": 45, "percentage": 12.5},
            "medium_sum": {"range": "100-200", "count": 180, "percentage": 50.0},
            "high_sum": {"range": "200-300", "count": 135, "percentage": 37.5}
        }
    },
    "draw_patterns": {
        "day_frequency": {
            "monday": 52,
            "wednesday": 52,
            "saturday": 52
        },
        "month_distribution": {
            str(i): np.random.randint(8, 15) for i in range(1, 13)
        }
    },
    "jackpot_analysis": {
        "average": 125000000,
        "median": 89000000,
        "max": 500000000,
        "min": 20000000,
        "std_dev": 75000000,
        "growth_rate": 0.15
    }
}

# Sample user preferences
SAMPLE_USER_PREFERENCES = {
    "user_id": "test_user_123",
    "preferences": {
        "default_game": "powerball",
        "default_strategy": "balanced",
        "number_of_predictions": 5,
        "exclude_recent_numbers": True,
        "favorite_numbers": [7, 13, 21, 35],
        "notification_settings": {
            "email_enabled": True,
            "sms_enabled": False,
            "push_enabled": True
        },
        "display_settings": {
            "theme": "light",
            "show_confidence": True,
            "show_statistics": True,
            "chart_type": "bar"
        }
    }
}

# Sample model configurations
SAMPLE_MODEL_CONFIGS = {
    "ensemble": {
        "name": "Ensemble Model",
        "version": "1.0",
        "algorithms": ["random_forest", "gradient_boosting", "svm"],
        "weights": [0.4, 0.35, 0.25],
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1
        },
        "performance": {
            "accuracy": 0.75,
            "precision": 0.72,
            "recall": 0.68,
            "f1_score": 0.70
        }
    },
    "neural_network": {
        "name": "Neural Network Model",
        "version": "2.1",
        "architecture": {
            "input_layer": 100,
            "hidden_layers": [64, 32, 16],
            "output_layer": 69,
            "activation": "relu",
            "dropout": 0.2
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss_function": "categorical_crossentropy"
        },
        "performance": {
            "accuracy": 0.68,
            "loss": 0.02,
            "val_accuracy": 0.65,
            "val_loss": 0.025
        }
    }
}

# Sample cache data
SAMPLE_CACHE_DATA = {
    "prediction_cache": {
        "key": "predictions_powerball_balanced_2023-12-01",
        "value": SAMPLE_PREDICTIONS[:2],
        "ttl": 3600,
        "created_at": "2023-12-01T10:30:00",
        "access_count": 5
    },
    "statistics_cache": {
        "key": "statistics_powerball_2023-12",
        "value": SAMPLE_STATISTICS,
        "ttl": 7200,
        "created_at": "2023-12-01T09:00:00",
        "access_count": 12
    }
}

# Sample export data
SAMPLE_EXPORT_DATA = {
    "csv_export": {
        "filename": "predictions_20231201.csv",
        "format": "csv",
        "data": pd.DataFrame(SAMPLE_PREDICTIONS),
        "metadata": {
            "rows": len(SAMPLE_PREDICTIONS),
            "columns": 8,
            "file_size": "2.5KB",
            "generated_at": "2023-12-01T11:00:00"
        }
    },
    "json_export": {
        "filename": "statistics_20231201.json",
        "format": "json",
        "data": SAMPLE_STATISTICS,
        "metadata": {
            "size": "15.2KB",
            "generated_at": "2023-12-01T11:05:00"
        }
    }
}

# Sample error scenarios
SAMPLE_ERRORS = {
    "validation_errors": [
        {
            "field": "numbers",
            "error": "Invalid number range",
            "value": [0, 5, 10, 15, 20],
            "expected": "Numbers between 1 and 69"
        },
        {
            "field": "confidence",
            "error": "Invalid confidence value",
            "value": 1.5,
            "expected": "Value between 0 and 1"
        }
    ],
    "service_errors": [
        {
            "service": "data_service",
            "error": "Database connection failed",
            "code": "DB_CONNECTION_ERROR",
            "timestamp": "2023-12-01T10:00:00"
        },
        {
            "service": "prediction_service",
            "error": "Model not found",
            "code": "MODEL_NOT_FOUND",
            "timestamp": "2023-12-01T10:05:00"
        }
    ]
}

# Performance benchmarks
SAMPLE_PERFORMANCE_DATA = {
    "prediction_generation": {
        "average_time": 1.2,
        "max_time": 3.5,
        "min_time": 0.8,
        "samples": 100,
        "success_rate": 0.98
    },
    "database_queries": {
        "average_time": 0.05,
        "max_time": 0.2,
        "min_time": 0.01,
        "queries_per_second": 200,
        "cache_hit_rate": 0.85
    },
    "model_training": {
        "ensemble_time": 45.2,
        "neural_network_time": 120.5,
        "pattern_analysis_time": 15.8,
        "total_time": 181.5
    }
}

def save_fixtures_to_files(output_dir: Path):
    """Save all sample data to JSON files for use in tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fixtures = {
        "game_configs.json": SAMPLE_GAME_CONFIGS,
        "predictions.json": SAMPLE_PREDICTIONS,
        "statistics.json": SAMPLE_STATISTICS,
        "user_preferences.json": SAMPLE_USER_PREFERENCES,
        "model_configs.json": SAMPLE_MODEL_CONFIGS,
        "cache_data.json": SAMPLE_CACHE_DATA,
        "export_data.json": SAMPLE_EXPORT_DATA,
        "errors.json": SAMPLE_ERRORS,
        "performance.json": SAMPLE_PERFORMANCE_DATA
    }
    
    for filename, data in fixtures.items():
        with open(output_dir / filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # Save historical data as CSV
    historical_data = generate_sample_historical_data("powerball", 365)
    df = pd.DataFrame(historical_data)
    df.to_csv(output_dir / "historical_data.csv", index=False)
    
    print(f"✅ Test fixtures saved to {output_dir}")

if __name__ == "__main__":
    # Generate and save fixtures when run directly
    fixtures_dir = Path(__file__).parent / "data"
    save_fixtures_to_files(fixtures_dir)