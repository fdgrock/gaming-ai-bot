"""
Pytest configuration and shared fixtures for the lottery prediction system tests.

This module provides common test configuration, fixtures, and utilities
for all test modules in the project.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import sqlite3
import json
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))

from streamlit_app.configs import AppConfig, DatabaseConfig, CacheConfig, AIConfig
from streamlit_app.services import ServiceManager


# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ai: mark test as AI/ML related"
    )
    config.addinivalue_line(
        "markers", "database: mark test as database related"
    )
    config.addinivalue_line(
        "markers", "cache: mark test as cache related"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark tests in integration/ directory as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark AI engine tests
        if "ai_engines" in str(item.fspath):
            item.add_marker(pytest.mark.ai)
        
        # Mark database-related tests
        if any(keyword in str(item.fspath).lower() for keyword in ["database", "data_service", "sql"]):
            item.add_marker(pytest.mark.database)
        
        # Mark cache-related tests
        if "cache" in str(item.fspath).lower():
            item.add_marker(pytest.mark.cache)


# ==================== BASIC FIXTURES ====================

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def temp_dir():
    """Provide a temporary directory for tests."""
    temp_path = tempfile.mkdtemp(prefix="lottery_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing UI components."""
    with pytest.MonkeyPatch.context() as mp:
        mock_st = Mock()
        mock_st.write = Mock()
        mock_st.title = Mock()
        mock_st.header = Mock()
        mock_st.subheader = Mock()
        mock_st.text = Mock()
        mock_st.markdown = Mock()
        mock_st.error = Mock()
        mock_st.warning = Mock()
        mock_st.info = Mock()
        mock_st.success = Mock()
        mock_st.container = Mock()
        mock_st.columns = Mock(return_value=[Mock(), Mock()])
        mock_st.sidebar = Mock()
        mock_st.session_state = {}
        mock_st.rerun = Mock()
        
        mp.setattr("streamlit", mock_st)
        yield mock_st


# ==================== CONFIGURATION FIXTURES ====================

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return AppConfig(
        app_name="Test Lottery System",
        version="2.0.0-test",
        debug=True,
        database=DatabaseConfig(
            connection_string="sqlite:///:memory:",
            pool_size=1,
            echo=False,
            backup_enabled=False
        ),
        cache=CacheConfig(
            enabled=False,
            max_memory_mb=64,
            default_ttl=300,
            cache_dir="test_cache",
            persistent_cache=False
        ),
        ai=AIConfig(
            ensemble_enabled=True,
            neural_network_enabled=False,  # Disable for faster tests
            quantum_enabled=False,
            pattern_enabled=True,
            model_timeout=5,
            confidence_threshold=0.0
        )
    )


@pytest.fixture
def test_database_config():
    """Provide test database configuration."""
    return DatabaseConfig(
        connection_string="sqlite:///:memory:",
        pool_size=1,
        pool_timeout=5,
        echo=False,
        backup_enabled=False
    )


@pytest.fixture
def test_cache_config():
    """Provide test cache configuration."""
    return CacheConfig(
        enabled=True,
        max_memory_mb=32,
        default_ttl=60,
        cache_dir="test_cache",
        cleanup_interval=30,
        persistent_cache=False
    )


# ==================== DATA FIXTURES ====================

@pytest.fixture
def sample_lottery_numbers():
    """Provide sample lottery numbers for testing."""
    return [1, 12, 23, 34, 45, 56]


@pytest.fixture
def sample_powerball_numbers():
    """Provide sample Powerball numbers."""
    return {
        'main_numbers': [5, 12, 20, 24, 29],
        'powerball': 4,
        'power_play': 2
    }


@pytest.fixture
def sample_historical_data():
    """Provide sample historical lottery data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='3D')
    data = []
    
    np.random.seed(42)  # For reproducible tests
    
    for date in dates:
        # Generate realistic lottery numbers
        main_numbers = sorted(np.random.choice(range(1, 70), 5, replace=False).tolist())
        powerball = np.random.randint(1, 27)
        
        data.append({
            'draw_date': date.strftime('%Y-%m-%d'),
            'game': 'powerball',
            'numbers': json.dumps(main_numbers),
            'powerball': powerball,
            'jackpot': np.random.randint(20, 500) * 1000000
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_prediction_data():
    """Provide sample prediction data."""
    return {
        'predictions': [
            {
                'numbers': [2, 15, 28, 35, 42, 58],
                'confidence': 0.75,
                'strategy': 'balanced',
                'engine': 'ensemble',
                'generated_at': datetime.now().isoformat()
            },
            {
                'numbers': [7, 18, 31, 44, 49, 63],
                'confidence': 0.68,
                'strategy': 'aggressive',
                'engine': 'pattern',
                'generated_at': datetime.now().isoformat()
            }
        ],
        'metadata': {
            'game': 'powerball',
            'total_predictions': 2,
            'generation_time': 1.23
        }
    }


@pytest.fixture
def sample_statistics_data():
    """Provide sample statistics data."""
    return {
        'frequency_analysis': {str(i): np.random.randint(5, 50) for i in range(1, 70)},
        'hot_numbers': [12, 23, 45, 56, 67],
        'cold_numbers': [3, 14, 25, 36, 47],
        'most_common_pairs': [(12, 23), (45, 56), (23, 45)],
        'draw_frequency': {
            'monday': 52,
            'wednesday': 52,
            'saturday': 52
        },
        'jackpot_statistics': {
            'average': 125000000,
            'median': 89000000,
            'max': 500000000,
            'min': 20000000
        }
    }


# ==================== SERVICE FIXTURES ====================

@pytest.fixture
def mock_service_manager(test_config):
    """Provide a mock service manager for testing."""
    manager = Mock(spec=ServiceManager)
    manager.config = test_config
    manager.services = {}
    manager.health_check = Mock(return_value={'healthy': True, 'services': {}})
    manager.get_service = Mock(side_effect=lambda name: manager.services.get(name))
    manager.register_service = Mock(side_effect=lambda name, service: manager.services.update({name: service}))
    return manager


@pytest.fixture
def mock_data_service(sample_historical_data, sample_statistics_data):
    """Provide a mock data service for testing."""
    service = Mock()
    service.get_historical_data = Mock(return_value=sample_historical_data)
    service.get_statistics = Mock(return_value=sample_statistics_data)
    service.add_draw_data = Mock(return_value=True)
    service.health_check = Mock(return_value=True)
    return service


@pytest.fixture
def mock_prediction_service(sample_prediction_data):
    """Provide a mock prediction service for testing."""
    service = Mock()
    service.generate_predictions = Mock(return_value=sample_prediction_data)
    service.get_prediction_confidence = Mock(return_value=0.75)
    service.health_check = Mock(return_value=True)
    return service


@pytest.fixture
def mock_cache_service():
    """Provide a mock cache service for testing."""
    service = Mock()
    service.cache = {}
    service.get = Mock(side_effect=lambda key, default=None: service.cache.get(key, default))
    service.set = Mock(side_effect=lambda key, value, ttl=None: service.cache.update({key: value}))
    service.delete = Mock(side_effect=lambda key: service.cache.pop(key, None))
    service.clear = Mock(side_effect=lambda: service.cache.clear())
    service.get_cache_stats = Mock(return_value={'hits': 10, 'misses': 5, 'size': 3})
    service.health_check = Mock(return_value=True)
    return service


# ==================== DATABASE FIXTURES ====================

@pytest.fixture
def in_memory_db():
    """Provide an in-memory SQLite database for testing."""
    conn = sqlite3.connect(':memory:')
    
    # Create test tables
    conn.execute('''
        CREATE TABLE lottery_draws (
            id INTEGER PRIMARY KEY,
            draw_date TEXT NOT NULL,
            game TEXT NOT NULL,
            numbers TEXT NOT NULL,
            powerball INTEGER,
            jackpot INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            game TEXT NOT NULL,
            numbers TEXT NOT NULL,
            confidence REAL,
            strategy TEXT,
            engine TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def populated_test_db(in_memory_db, sample_historical_data):
    """Provide a populated test database."""
    conn = in_memory_db
    
    # Insert sample data
    for _, row in sample_historical_data.iterrows():
        conn.execute(
            'INSERT INTO lottery_draws (draw_date, game, numbers, powerball, jackpot) VALUES (?, ?, ?, ?, ?)',
            (row['draw_date'], row['game'], row['numbers'], row['powerball'], row['jackpot'])
        )
    
    conn.commit()
    return conn


# ==================== AI ENGINE FIXTURES ====================

@pytest.fixture
def mock_ensemble_engine():
    """Provide a mock ensemble engine for testing."""
    engine = Mock()
    engine.predict = Mock(return_value={
        'numbers': [5, 12, 20, 24, 29],
        'confidence': 0.75,
        'method': 'ensemble'
    })
    engine.train = Mock(return_value=True)
    engine.health_check = Mock(return_value=True)
    return engine


@pytest.fixture
def mock_neural_engine():
    """Provide a mock neural network engine for testing."""
    engine = Mock()
    engine.predict = Mock(return_value={
        'numbers': [8, 15, 27, 33, 48],
        'confidence': 0.68,
        'method': 'neural_network'
    })
    engine.train = Mock(return_value=True)
    engine.health_check = Mock(return_value=True)
    return engine


@pytest.fixture
def mock_pattern_engine():
    """Provide a mock pattern engine for testing."""
    engine = Mock()
    engine.predict = Mock(return_value={
        'numbers': [11, 22, 33, 44, 55],
        'confidence': 0.62,
        'method': 'pattern_analysis'
    })
    engine.analyze_patterns = Mock(return_value={'hot_numbers': [11, 22, 33]})
    engine.health_check = Mock(return_value=True)
    return engine


# ==================== COMPONENT FIXTURES ====================

@pytest.fixture
def sample_chart_data():
    """Provide sample data for chart testing."""
    return {
        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        'values': [10, 15, 13, 17, 20],
        'categories': ['A', 'B', 'C', 'D', 'E']
    }


@pytest.fixture
def sample_table_data():
    """Provide sample table data for testing."""
    return pd.DataFrame({
        'Number': [1, 2, 3, 4, 5],
        'Frequency': [15, 12, 18, 9, 21],
        'Last_Seen': ['2023-12-01', '2023-11-28', '2023-12-03', '2023-11-25', '2023-12-02'],
        'Category': ['Hot', 'Cold', 'Hot', 'Cold', 'Hot']
    })


# ==================== VALIDATION FIXTURES ====================

@pytest.fixture
def valid_lottery_config():
    """Provide valid lottery game configuration."""
    return {
        'game_name': 'Test Powerball',
        'num_numbers': 5,
        'max_number': 69,
        'min_number': 1,
        'bonus_numbers': 1,
        'bonus_max': 26,
        'draw_schedule': {
            'days': ['monday', 'wednesday', 'saturday'],
            'time': '22:59'
        }
    }


@pytest.fixture
def invalid_lottery_config():
    """Provide invalid lottery game configuration for testing validation."""
    return {
        'game_name': '',  # Invalid: empty name
        'num_numbers': 0,  # Invalid: zero numbers
        'max_number': 5,   # Invalid: max less than num_numbers
        'min_number': -1,  # Invalid: negative min
        'draw_schedule': {
            'days': ['invalid_day'],  # Invalid day
            'time': '25:00'  # Invalid time
        }
    }


# ==================== EXPORT FIXTURES ====================

@pytest.fixture
def sample_export_data():
    """Provide sample export data."""
    return {
        'predictions': [
            {'numbers': [1, 2, 3, 4, 5], 'confidence': 0.8},
            {'numbers': [6, 7, 8, 9, 10], 'confidence': 0.7}
        ],
        'statistics': {'total': 100, 'average': 50},
        'metadata': {'exported_at': datetime.now().isoformat()}
    }


# ==================== PERFORMANCE FIXTURES ====================

@pytest.fixture
def performance_timer():
    """Provide a performance timer for testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
            return (self.end_time - self.start_time).total_seconds()
    
    return Timer()


# ==================== CLEANUP FIXTURES ====================

@pytest.fixture(autouse=True)
def cleanup_test_files(temp_dir):
    """Automatically clean up test files after each test."""
    yield
    # Cleanup happens automatically with temp_dir fixture


# ==================== ERROR HANDLING FIXTURES ====================

@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing logging behavior."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.critical = Mock()
    return logger


# ==================== ASYNC FIXTURES ====================

@pytest.fixture
def event_loop():
    """Provide event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ==================== PARAMETRIZED FIXTURES ====================

@pytest.fixture(params=['powerball', 'mega_millions', 'euromillions'])
def lottery_game_type(request):
    """Parametrized fixture for different lottery game types."""
    return request.param


@pytest.fixture(params=[5, 10, 20])
def prediction_count(request):
    """Parametrized fixture for different prediction counts."""
    return request.param


@pytest.fixture(params=['aggressive', 'balanced', 'conservative'])
def prediction_strategy(request):
    """Parametrized fixture for different prediction strategies."""
    return request.param


# ==================== HELPER FUNCTIONS ====================

def create_test_user():
    """Helper function to create test user data."""
    return {
        'id': 'test_user_123',
        'name': 'Test User',
        'preferences': {
            'default_game': 'powerball',
            'default_strategy': 'balanced'
        }
    }


def assert_valid_lottery_numbers(numbers, min_num=1, max_num=69, count=5):
    """Helper function to validate lottery numbers in tests."""
    assert isinstance(numbers, list), "Numbers must be a list"
    assert len(numbers) == count, f"Must have exactly {count} numbers"
    assert all(isinstance(n, int) for n in numbers), "All numbers must be integers"
    assert all(min_num <= n <= max_num for n in numbers), f"Numbers must be between {min_num} and {max_num}"
    assert len(set(numbers)) == len(numbers), "Numbers must be unique"


def assert_valid_prediction(prediction):
    """Helper function to validate prediction structure in tests."""
    assert isinstance(prediction, dict), "Prediction must be a dictionary"
    assert 'numbers' in prediction, "Prediction must have numbers"
    assert 'confidence' in prediction, "Prediction must have confidence"
    assert isinstance(prediction['confidence'], (int, float)), "Confidence must be numeric"
    assert 0 <= prediction['confidence'] <= 1, "Confidence must be between 0 and 1"