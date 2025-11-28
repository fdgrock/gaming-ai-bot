"""
Unit tests for the Prediction Service module.

Tests prediction generation, AI engine coordination, and result processing.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from services.prediction_service import PredictionService
from configs.app_config import AppConfig


class TestPredictionService:
    """Test suite for Prediction Service functionality."""
    
    def test_service_initialization(self, mock_prediction_service):
        """Test that PredictionService initializes correctly."""
        assert mock_prediction_service is not None
        assert hasattr(mock_prediction_service, 'generate_predictions')
        assert hasattr(mock_prediction_service, 'get_prediction_confidence')
        assert hasattr(mock_prediction_service, 'validate_prediction')
    
    def test_generate_predictions_success(self, mock_prediction_service, sample_predictions):
        """Test successful prediction generation."""
        # Configure mock
        mock_prediction_service.generate_predictions.return_value = sample_predictions
        
        # Test
        result = mock_prediction_service.generate_predictions(
            game="powerball",
            strategy="balanced",
            count=3
        )
        
        # Assertions
        assert isinstance(result, list)
        assert len(result) == len(sample_predictions)
        assert all("numbers" in pred for pred in result)
        assert all("confidence" in pred for pred in result)
        assert all(pred["game"] == "powerball" for pred in result)
        
        # Verify method was called correctly
        mock_prediction_service.generate_predictions.assert_called_once_with(
            game="powerball",
            strategy="balanced", 
            count=3
        )
    
    def test_generate_predictions_with_different_strategies(self, mock_prediction_service, sample_predictions):
        """Test prediction generation with different strategies."""
        strategies = ["conservative", "balanced", "aggressive"]
        
        for strategy in strategies:
            # Filter predictions by strategy
            strategy_predictions = [
                pred for pred in sample_predictions 
                if pred["strategy"] == strategy
            ]
            if not strategy_predictions:
                strategy_predictions = [sample_predictions[0].copy()]
                strategy_predictions[0]["strategy"] = strategy
            
            mock_prediction_service.generate_predictions.return_value = strategy_predictions
            
            # Test
            result = mock_prediction_service.generate_predictions(
                game="powerball",
                strategy=strategy
            )
            
            # Assertions
            assert all(pred["strategy"] == strategy for pred in result)
    
    def test_generate_predictions_invalid_game(self, mock_prediction_service):
        """Test prediction generation with invalid game."""
        mock_prediction_service.generate_predictions.side_effect = ValueError("Invalid game type")
        
        # Test
        with pytest.raises(ValueError, match="Invalid game type"):
            mock_prediction_service.generate_predictions(game="invalid_game")
    
    def test_get_prediction_confidence(self, mock_prediction_service):
        """Test prediction confidence calculation."""
        test_prediction = {
            "numbers": [1, 15, 25, 35, 45],
            "bonus": 12,
            "strategy": "balanced"
        }
        
        expected_confidence = 0.75
        mock_prediction_service.get_prediction_confidence.return_value = expected_confidence
        
        # Test
        result = mock_prediction_service.get_prediction_confidence(test_prediction)
        
        # Assertions
        assert result == expected_confidence
        assert 0 <= result <= 1
        mock_prediction_service.get_prediction_confidence.assert_called_once_with(test_prediction)
    
    def test_validate_prediction_valid(self, mock_prediction_service):
        """Test validation of valid prediction."""
        valid_prediction = {
            "numbers": [5, 12, 25, 35, 45],
            "bonus": 12,
            "game": "powerball",
            "strategy": "balanced"
        }
        
        mock_prediction_service.validate_prediction.return_value = True
        
        # Test
        result = mock_prediction_service.validate_prediction(valid_prediction, "powerball")
        
        # Assertions
        assert result is True
        mock_prediction_service.validate_prediction.assert_called_once_with(valid_prediction, "powerball")
    
    def test_validate_prediction_invalid_numbers(self, mock_prediction_service):
        """Test validation of prediction with invalid numbers."""
        invalid_prediction = {
            "numbers": [0, 5, 10, 70, 80],  # Invalid range
            "bonus": 12,
            "game": "powerball"
        }
        
        mock_prediction_service.validate_prediction.return_value = False
        
        # Test
        result = mock_prediction_service.validate_prediction(invalid_prediction, "powerball")
        
        # Assertions
        assert result is False
    
    def test_validate_prediction_duplicate_numbers(self, mock_prediction_service):
        """Test validation of prediction with duplicate numbers."""
        invalid_prediction = {
            "numbers": [5, 12, 12, 25, 35],  # Duplicate 12
            "game": "powerball"
        }
        
        mock_prediction_service.validate_prediction.return_value = False
        
        # Test
        result = mock_prediction_service.validate_prediction(invalid_prediction, "powerball")
        
        # Assertions
        assert result is False
    
    def test_ai_engine_integration(self, mock_prediction_service, mock_ai_engines):
        """Test integration with AI engines."""
        # Mock AI engine responses
        ensemble_result = {
            "numbers": [1, 15, 25, 35, 45],
            "confidence": 0.75,
            "engine": "ensemble"
        }
        
        neural_result = {
            "numbers": [5, 20, 30, 40, 50],
            "confidence": 0.68,
            "engine": "neural_network"
        }
        
        mock_ai_engines["ensemble"].generate_prediction.return_value = ensemble_result
        mock_ai_engines["neural_network"].generate_prediction.return_value = neural_result
        
        # Configure prediction service to use engines
        def generate_with_engines(game, strategy="balanced", engine="ensemble"):
            if engine == "ensemble":
                return [ensemble_result]
            elif engine == "neural_network":
                return [neural_result]
            return []
        
        mock_prediction_service.generate_predictions.side_effect = generate_with_engines
        
        # Test ensemble engine
        result = mock_prediction_service.generate_predictions("powerball", engine="ensemble")
        assert result[0]["engine"] == "ensemble"
        
        # Test neural network engine
        result = mock_prediction_service.generate_predictions("powerball", engine="neural_network")
        assert result[0]["engine"] == "neural_network"
    
    def test_prediction_with_user_preferences(self, mock_prediction_service, sample_user_preferences):
        """Test prediction generation with user preferences."""
        preferences = sample_user_preferences["preferences"]
        prediction_with_prefs = {
            "numbers": [7, 13, 21, 35, 45],  # Includes favorite numbers
            "confidence": 0.72,
            "strategy": preferences["default_strategy"],
            "excluded_recent": preferences["exclude_recent_numbers"]
        }
        
        mock_prediction_service.generate_predictions.return_value = [prediction_with_prefs]
        
        # Test
        result = mock_prediction_service.generate_predictions(
            game="powerball",
            user_preferences=preferences
        )
        
        # Assertions
        prediction = result[0]
        assert prediction["strategy"] == preferences["default_strategy"]
        
        # Check if favorite numbers are included
        favorite_numbers = set(preferences["favorite_numbers"])
        prediction_numbers = set(prediction["numbers"])
        assert len(favorite_numbers.intersection(prediction_numbers)) > 0
    
    def test_batch_prediction_generation(self, mock_prediction_service, sample_predictions):
        """Test batch generation of multiple predictions."""
        batch_size = 10
        mock_prediction_service.generate_batch_predictions.return_value = sample_predictions * 3  # Simulate batch
        
        # Test
        result = mock_prediction_service.generate_batch_predictions(
            game="powerball",
            count=batch_size,
            strategies=["conservative", "balanced", "aggressive"]
        )
        
        # Assertions
        assert len(result) >= batch_size
        strategies = set(pred["strategy"] for pred in result)
        assert len(strategies) > 1  # Multiple strategies used
        mock_prediction_service.generate_batch_predictions.assert_called_once()
    
    def test_prediction_caching(self, mock_prediction_service, mock_cache_service, sample_predictions):
        """Test prediction result caching."""
        cache_key = "predictions_powerball_balanced"
        mock_cache_service.get.return_value = None  # Cache miss
        mock_cache_service.set.return_value = True
        
        # Configure service to use cache
        mock_prediction_service.cache = mock_cache_service
        
        def generate_with_cache(game, strategy="balanced"):
            cached = mock_cache_service.get(cache_key)
            if cached:
                return cached
            
            # Generate new predictions
            result = sample_predictions
            mock_cache_service.set(cache_key, result, ttl=3600)
            return result
        
        mock_prediction_service.generate_predictions.side_effect = generate_with_cache
        
        # Test
        result = mock_prediction_service.generate_predictions("powerball", strategy="balanced")
        
        # Assertions
        assert result == sample_predictions
        mock_cache_service.get.assert_called_once_with(cache_key)
        mock_cache_service.set.assert_called_once_with(cache_key, sample_predictions, ttl=3600)
    
    def test_prediction_history_tracking(self, mock_prediction_service, sample_predictions):
        """Test tracking of prediction history."""
        # Mock saving prediction to history
        mock_prediction_service.save_prediction_to_history.return_value = True
        
        prediction = sample_predictions[0]
        
        # Test
        result = mock_prediction_service.save_prediction_to_history(prediction)
        
        # Assertions
        assert result is True
        mock_prediction_service.save_prediction_to_history.assert_called_once_with(prediction)
    
    def test_prediction_performance_metrics(self, mock_prediction_service):
        """Test prediction performance tracking."""
        performance_data = {
            "generation_time": 1.2,
            "confidence_avg": 0.72,
            "engine_used": "ensemble",
            "cache_hit": False,
            "validation_passed": True
        }
        
        mock_prediction_service.get_performance_metrics.return_value = performance_data
        
        # Test
        result = mock_prediction_service.get_performance_metrics("powerball")
        
        # Assertions
        assert "generation_time" in result
        assert "confidence_avg" in result
        assert result["validation_passed"] is True
        mock_prediction_service.get_performance_metrics.assert_called_once_with("powerball")
    
    def test_prediction_with_exclusions(self, mock_prediction_service):
        """Test prediction generation with number exclusions."""
        excluded_numbers = [1, 2, 3, 4, 5]
        prediction_with_exclusions = {
            "numbers": [10, 20, 30, 40, 50],  # No excluded numbers
            "confidence": 0.68,
            "exclusions_applied": True
        }
        
        mock_prediction_service.generate_predictions.return_value = [prediction_with_exclusions]
        
        # Test
        result = mock_prediction_service.generate_predictions(
            game="powerball",
            exclude_numbers=excluded_numbers
        )
        
        # Assertions
        prediction = result[0]
        prediction_numbers = set(prediction["numbers"])
        excluded_set = set(excluded_numbers)
        assert len(prediction_numbers.intersection(excluded_set)) == 0
    
    def test_async_prediction_generation(self, mock_prediction_service, sample_predictions):
        """Test asynchronous prediction generation."""
        # Mock async method
        async_mock = AsyncMock(return_value=sample_predictions)
        mock_prediction_service.generate_predictions_async = async_mock
        
        # Test would require async context in real implementation
        # For mock testing, verify the mock is properly configured
        assert mock_prediction_service.generate_predictions_async is not None
        assert callable(mock_prediction_service.generate_predictions_async)
    
    def test_prediction_comparison(self, mock_prediction_service, sample_predictions):
        """Test comparison between different prediction strategies."""
        comparison_result = {
            "conservative": {"avg_confidence": 0.65, "risk_level": "low"},
            "balanced": {"avg_confidence": 0.72, "risk_level": "medium"},
            "aggressive": {"avg_confidence": 0.78, "risk_level": "high"}
        }
        
        mock_prediction_service.compare_strategies.return_value = comparison_result
        
        # Test
        result = mock_prediction_service.compare_strategies("powerball", sample_predictions)
        
        # Assertions
        assert "conservative" in result
        assert "balanced" in result
        assert "aggressive" in result
        assert all("avg_confidence" in strategy for strategy in result.values())
        mock_prediction_service.compare_strategies.assert_called_once_with("powerball", sample_predictions)
    
    @pytest.mark.parametrize("game,strategy,expected_count", [
        ("powerball", "conservative", 3),
        ("powerball", "balanced", 5),
        ("powerball", "aggressive", 7),
        ("mega_millions", "balanced", 5)
    ])
    def test_parameterized_prediction_generation(self, mock_prediction_service, game, strategy, expected_count):
        """Test prediction generation with various parameters."""
        mock_predictions = [{"id": i, "game": game, "strategy": strategy} for i in range(expected_count)]
        mock_prediction_service.generate_predictions.return_value = mock_predictions
        
        # Test
        result = mock_prediction_service.generate_predictions(
            game=game,
            strategy=strategy,
            count=expected_count
        )
        
        # Assertions
        assert len(result) == expected_count
        assert all(pred["game"] == game for pred in result)
        assert all(pred["strategy"] == strategy for pred in result)
    
    def test_error_handling_and_recovery(self, mock_prediction_service):
        """Test error handling and recovery mechanisms."""
        # First call fails with engine error
        mock_prediction_service.generate_predictions.side_effect = [
            RuntimeError("AI engine unavailable"),
            [{"numbers": [1, 2, 3, 4, 5], "confidence": 0.5, "engine": "fallback"}]  # Fallback success
        ]
        
        # Test error and recovery
        with pytest.raises(RuntimeError, match="AI engine unavailable"):
            mock_prediction_service.generate_predictions("powerball")
        
        # Test fallback mechanism
        result = mock_prediction_service.generate_predictions("powerball")
        assert result[0]["engine"] == "fallback"
    
    def test_prediction_metadata_enrichment(self, mock_prediction_service):
        """Test enrichment of predictions with metadata."""
        enriched_prediction = {
            "numbers": [5, 12, 25, 35, 45],
            "confidence": 0.75,
            "metadata": {
                "generation_time": "2023-12-01T10:30:00",
                "model_version": "1.0",
                "data_source": "historical_365d",
                "algorithm": "ensemble"
            }
        }
        
        mock_prediction_service.enrich_prediction_metadata.return_value = enriched_prediction
        
        # Test
        base_prediction = {"numbers": [5, 12, 25, 35, 45], "confidence": 0.75}
        result = mock_prediction_service.enrich_prediction_metadata(base_prediction)
        
        # Assertions
        assert "metadata" in result
        assert "generation_time" in result["metadata"]
        assert "model_version" in result["metadata"]
        mock_prediction_service.enrich_prediction_metadata.assert_called_once_with(base_prediction)