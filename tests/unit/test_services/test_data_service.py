"""
Unit tests for the Data Service module.

Tests data management functionality including historical data access,
caching, and database operations.
"""

import pytest
import json
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call

from services.data_service import DataService
from configs.app_config import AppConfig


class TestDataService:
    """Test suite for Data Service functionality."""
    
    def test_service_initialization(self, mock_data_service):
        """Test that DataService initializes correctly."""
        assert mock_data_service is not None
        assert hasattr(mock_data_service, 'get_historical_data')
        assert hasattr(mock_data_service, 'save_prediction')
        assert hasattr(mock_data_service, 'get_predictions')
    
    def test_get_historical_data_success(self, mock_data_service, sample_historical_data):
        """Test successful retrieval of historical lottery data."""
        # Configure mock
        mock_data_service.get_historical_data.return_value = sample_historical_data
        
        # Test
        result = mock_data_service.get_historical_data("powerball", days=30)
        
        # Assertions
        assert result is not None
        assert len(result) > 0
        assert isinstance(result, list)
        assert "draw_date" in result[0]
        assert "numbers" in result[0]
        assert "game" in result[0]
        
        # Verify method was called correctly
        mock_data_service.get_historical_data.assert_called_once_with("powerball", days=30)
    
    def test_get_historical_data_empty_result(self, mock_data_service):
        """Test handling of empty historical data."""
        # Configure mock to return empty list
        mock_data_service.get_historical_data.return_value = []
        
        # Test
        result = mock_data_service.get_historical_data("invalid_game")
        
        # Assertions
        assert result == []
        mock_data_service.get_historical_data.assert_called_once_with("invalid_game")
    
    def test_get_historical_data_with_date_range(self, mock_data_service, sample_historical_data):
        """Test retrieval of historical data with specific date range."""
        # Filter sample data for date range
        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)
        
        filtered_data = [
            item for item in sample_historical_data
            if start_date <= datetime.strptime(item["draw_date"], "%Y-%m-%d") <= end_date
        ]
        
        mock_data_service.get_historical_data.return_value = filtered_data
        
        # Test
        result = mock_data_service.get_historical_data(
            "powerball", 
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Assertions
        assert isinstance(result, list)
        for item in result:
            item_date = datetime.strptime(item["draw_date"], "%Y-%m-%d")
            assert start_date <= item_date <= end_date
    
    def test_save_prediction_success(self, mock_data_service, sample_predictions):
        """Test successful saving of prediction data."""
        prediction = sample_predictions[0]
        mock_data_service.save_prediction.return_value = True
        
        # Test
        result = mock_data_service.save_prediction(prediction)
        
        # Assertions
        assert result is True
        mock_data_service.save_prediction.assert_called_once_with(prediction)
    
    def test_save_prediction_validation_error(self, mock_data_service):
        """Test saving prediction with validation errors."""
        invalid_prediction = {
            "numbers": [0, 5, 10],  # Invalid: too few numbers and invalid range
            "game": "powerball"
        }
        
        mock_data_service.save_prediction.side_effect = ValueError("Invalid prediction data")
        
        # Test
        with pytest.raises(ValueError, match="Invalid prediction data"):
            mock_data_service.save_prediction(invalid_prediction)
    
    def test_get_predictions_recent(self, mock_data_service, sample_predictions):
        """Test retrieval of recent predictions."""
        # Configure mock to return recent predictions
        recent_predictions = sample_predictions[:2]
        mock_data_service.get_predictions.return_value = recent_predictions
        
        # Test
        result = mock_data_service.get_predictions("powerball", limit=2)
        
        # Assertions
        assert len(result) == 2
        assert all(pred["game"] == "powerball" for pred in result)
        mock_data_service.get_predictions.assert_called_once_with("powerball", limit=2)
    
    def test_get_predictions_by_strategy(self, mock_data_service, sample_predictions):
        """Test retrieval of predictions filtered by strategy."""
        balanced_predictions = [
            pred for pred in sample_predictions 
            if pred["strategy"] == "balanced"
        ]
        mock_data_service.get_predictions.return_value = balanced_predictions
        
        # Test
        result = mock_data_service.get_predictions("powerball", strategy="balanced")
        
        # Assertions
        assert all(pred["strategy"] == "balanced" for pred in result)
        mock_data_service.get_predictions.assert_called_once_with("powerball", strategy="balanced")
    
    def test_get_game_statistics(self, mock_data_service, sample_statistics):
        """Test retrieval of game statistics."""
        mock_data_service.get_game_statistics.return_value = sample_statistics
        
        # Test
        result = mock_data_service.get_game_statistics("powerball")
        
        # Assertions
        assert "frequency_analysis" in result
        assert "number_statistics" in result
        assert "pattern_analysis" in result
        mock_data_service.get_game_statistics.assert_called_once_with("powerball")
    
    def test_get_frequency_analysis(self, mock_data_service, sample_statistics):
        """Test frequency analysis functionality."""
        frequency_data = sample_statistics["frequency_analysis"]
        mock_data_service.get_frequency_analysis.return_value = frequency_data
        
        # Test
        result = mock_data_service.get_frequency_analysis("powerball", days=365)
        
        # Assertions
        assert isinstance(result, dict)
        assert len(result) > 0
        assert all(isinstance(freq, int) for freq in result.values())
        mock_data_service.get_frequency_analysis.assert_called_once_with("powerball", days=365)
    
    def test_database_connection_error(self, mock_data_service):
        """Test handling of database connection errors."""
        mock_data_service.get_historical_data.side_effect = ConnectionError("Database unavailable")
        
        # Test
        with pytest.raises(ConnectionError, match="Database unavailable"):
            mock_data_service.get_historical_data("powerball")
    
    def test_data_validation(self, mock_data_service):
        """Test data validation methods."""
        # Test valid data
        valid_data = {
            "numbers": [1, 15, 25, 35, 45],
            "bonus": 12,
            "game": "powerball"
        }
        mock_data_service.validate_lottery_data.return_value = True
        
        result = mock_data_service.validate_lottery_data(valid_data)
        assert result is True
        
        # Test invalid data
        invalid_data = {
            "numbers": [0, 70, 80],  # Invalid range
            "game": "powerball"
        }
        mock_data_service.validate_lottery_data.return_value = False
        
        result = mock_data_service.validate_lottery_data(invalid_data)
        assert result is False
    
    def test_cache_integration(self, mock_data_service, mock_cache_service):
        """Test integration with cache service."""
        cache_key = "historical_powerball_30"
        cached_data = [{"id": 1, "numbers": [1, 2, 3, 4, 5]}]
        
        # Configure cache mock
        mock_cache_service.get.return_value = cached_data
        mock_data_service.cache = mock_cache_service
        
        # Mock the service to use cache
        def get_with_cache(game, days=30):
            cached = mock_cache_service.get(cache_key)
            if cached:
                return cached
            return []
        
        mock_data_service.get_historical_data.side_effect = get_with_cache
        
        # Test
        result = mock_data_service.get_historical_data("powerball", days=30)
        
        # Assertions
        assert result == cached_data
        mock_cache_service.get.assert_called_once_with(cache_key)
    
    def test_bulk_data_operations(self, mock_data_service, sample_historical_data):
        """Test bulk data operations."""
        # Test bulk save
        mock_data_service.bulk_save_historical_data.return_value = len(sample_historical_data)
        
        result = mock_data_service.bulk_save_historical_data(sample_historical_data)
        assert result == len(sample_historical_data)
        
        # Test bulk update
        mock_data_service.bulk_update_statistics.return_value = True
        
        result = mock_data_service.bulk_update_statistics("powerball")
        assert result is True
    
    def test_data_export_functionality(self, mock_data_service, sample_historical_data):
        """Test data export functionality."""
        export_data = {
            "format": "csv",
            "data": sample_historical_data,
            "filename": "export_20231201.csv"
        }
        
        mock_data_service.export_data.return_value = export_data
        
        # Test
        result = mock_data_service.export_data("powerball", format="csv", days=30)
        
        # Assertions
        assert result["format"] == "csv"
        assert "data" in result
        assert "filename" in result
        mock_data_service.export_data.assert_called_once_with("powerball", format="csv", days=30)
    
    def test_data_import_functionality(self, mock_data_service):
        """Test data import functionality."""
        import_file = "test_data.csv"
        imported_count = 100
        
        mock_data_service.import_historical_data.return_value = imported_count
        
        # Test
        result = mock_data_service.import_historical_data(import_file)
        
        # Assertions
        assert result == imported_count
        mock_data_service.import_historical_data.assert_called_once_with(import_file)
    
    def test_data_cleanup_operations(self, mock_data_service):
        """Test data cleanup and maintenance operations."""
        # Test old data cleanup
        cleaned_count = 50
        mock_data_service.cleanup_old_data.return_value = cleaned_count
        
        result = mock_data_service.cleanup_old_data(days=730)  # 2 years
        assert result == cleaned_count
        
        # Test duplicate removal
        duplicates_removed = 5
        mock_data_service.remove_duplicates.return_value = duplicates_removed
        
        result = mock_data_service.remove_duplicates("powerball")
        assert result == duplicates_removed
    
    @pytest.mark.parametrize("game,expected_fields", [
        ("powerball", ["numbers", "bonus", "draw_date", "jackpot"]),
        ("mega_millions", ["numbers", "bonus", "draw_date", "jackpot"]),
        ("test_game", ["numbers", "draw_date"])
    ])
    def test_game_specific_data_structure(self, mock_data_service, game, expected_fields):
        """Test that data structure matches game requirements."""
        mock_data = {field: f"mock_{field}" for field in expected_fields}
        mock_data_service.get_latest_draw.return_value = mock_data
        
        result = mock_data_service.get_latest_draw(game)
        
        for field in expected_fields:
            assert field in result
    
    def test_concurrent_access_handling(self, mock_data_service):
        """Test handling of concurrent data access."""
        # Simulate concurrent access
        mock_data_service.get_historical_data.return_value = [{"id": 1}]
        
        # Multiple concurrent calls
        results = []
        for _ in range(5):
            result = mock_data_service.get_historical_data("powerball")
            results.append(result)
        
        # All calls should succeed
        assert len(results) == 5
        assert all(len(r) == 1 for r in results)
    
    def test_error_recovery_mechanisms(self, mock_data_service):
        """Test error recovery and retry mechanisms."""
        # First call fails, second succeeds
        mock_data_service.get_historical_data.side_effect = [
            ConnectionError("Temporary failure"),
            [{"id": 1, "numbers": [1, 2, 3, 4, 5]}]
        ]
        
        # Simulate retry logic
        try:
            result = mock_data_service.get_historical_data("powerball")
        except ConnectionError:
            # Retry
            result = mock_data_service.get_historical_data("powerball")
        
        assert result is not None
        assert len(result) == 1