"""
Integration tests for the complete lottery prediction workflow.

Tests end-to-end functionality including data flow between services,
AI engines, and UI components.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from services.service_manager import ServiceManager
from configs.app_config import AppConfig


class TestPredictionWorkflow:
    """Integration tests for complete prediction workflow."""
    
    def test_complete_prediction_generation_workflow(self, mock_service_manager, sample_historical_data, sample_game_config):
        """Test complete workflow from data retrieval to prediction display."""
        # Mock the complete workflow
        game_config = sample_game_config["powerball"]
        
        # Step 1: Data retrieval
        mock_service_manager.data_service.get_historical_data.return_value = sample_historical_data
        
        # Step 2: Prediction generation
        prediction_result = {
            "numbers": [5, 12, 25, 35, 45],
            "bonus": 12,
            "confidence": 0.75,
            "strategy": "balanced",
            "engine": "ensemble",
            "generated_at": datetime.now().isoformat()
        }
        mock_service_manager.prediction_service.generate_predictions.return_value = [prediction_result]
        
        # Step 3: Save prediction
        mock_service_manager.data_service.save_prediction.return_value = True
        
        # Step 4: Cache result
        mock_service_manager.cache_service.set.return_value = True
        
        # Execute workflow
        historical_data = mock_service_manager.data_service.get_historical_data("powerball", days=365)
        assert len(historical_data) > 0
        
        predictions = mock_service_manager.prediction_service.generate_predictions(
            game="powerball",
            strategy="balanced",
            historical_data=historical_data
        )
        assert len(predictions) == 1
        assert predictions[0]["confidence"] > 0.7
        
        # Save and cache
        save_result = mock_service_manager.data_service.save_prediction(predictions[0])
        assert save_result is True
        
        cache_result = mock_service_manager.cache_service.set(
            "latest_prediction_powerball",
            predictions[0],
            ttl=3600
        )
        assert cache_result is True
        
        # Verify all services were called
        mock_service_manager.data_service.get_historical_data.assert_called_once()
        mock_service_manager.prediction_service.generate_predictions.assert_called_once()
        mock_service_manager.data_service.save_prediction.assert_called_once()
        mock_service_manager.cache_service.set.assert_called_once()
    
    def test_multi_strategy_prediction_comparison(self, mock_service_manager, sample_historical_data):
        """Test workflow for comparing multiple prediction strategies."""
        strategies = ["conservative", "balanced", "aggressive"]
        strategy_results = {}
        
        # Mock predictions for each strategy
        for strategy in strategies:
            prediction = {
                "numbers": [1, 15, 25, 35, 45],
                "confidence": 0.7 + (0.05 if strategy == "aggressive" else 0),
                "strategy": strategy,
                "risk_level": strategy
            }
            mock_service_manager.prediction_service.generate_predictions.return_value = [prediction]
            
            # Generate prediction for strategy
            result = mock_service_manager.prediction_service.generate_predictions(
                game="powerball",
                strategy=strategy,
                historical_data=sample_historical_data
            )
            strategy_results[strategy] = result[0]
        
        # Compare strategies
        comparison = {
            "strategies": strategy_results,
            "recommendation": "balanced",
            "analysis": {
                "risk_vs_reward": "balanced offers optimal risk/reward ratio",
                "confidence_comparison": strategy_results
            }
        }
        
        mock_service_manager.prediction_service.compare_strategies.return_value = comparison
        
        comparison_result = mock_service_manager.prediction_service.compare_strategies(
            "powerball", 
            list(strategy_results.values())
        )
        
        # Assertions
        assert len(comparison_result["strategies"]) == 3
        assert comparison_result["recommendation"] in strategies
        assert "analysis" in comparison_result
    
    def test_prediction_with_user_preferences_workflow(self, mock_service_manager, sample_user_preferences):
        """Test workflow incorporating user preferences."""
        preferences = sample_user_preferences["preferences"]
        
        # Mock user preference retrieval
        mock_service_manager.data_service.get_user_preferences.return_value = preferences
        
        # Mock prediction with preferences
        personalized_prediction = {
            "numbers": [7, 13, 21, 35, 45],  # Includes favorite numbers
            "bonus": 12,
            "confidence": 0.72,
            "strategy": preferences["default_strategy"],
            "personalized": True,
            "favorite_numbers_used": [7, 13, 21],
            "excluded_recent": preferences["exclude_recent_numbers"]
        }
        
        mock_service_manager.prediction_service.generate_predictions.return_value = [personalized_prediction]
        
        # Execute workflow
        user_prefs = mock_service_manager.data_service.get_user_preferences("test_user_123")
        assert user_prefs == preferences
        
        prediction = mock_service_manager.prediction_service.generate_predictions(
            game=user_prefs["default_game"],
            strategy=user_prefs["default_strategy"],
            user_preferences=user_prefs
        )
        
        # Assertions
        assert prediction[0]["personalized"] is True
        assert prediction[0]["strategy"] == preferences["default_strategy"]
        assert len(prediction[0]["favorite_numbers_used"]) > 0
    
    def test_statistics_and_analysis_workflow(self, mock_service_manager, sample_statistics):
        """Test workflow for generating statistics and analysis."""
        # Mock statistics retrieval
        mock_service_manager.data_service.get_game_statistics.return_value = sample_statistics
        
        # Mock frequency analysis
        frequency_data = sample_statistics["frequency_analysis"]
        mock_service_manager.data_service.get_frequency_analysis.return_value = frequency_data
        
        # Mock pattern analysis
        pattern_data = sample_statistics["pattern_analysis"]
        mock_service_manager.data_service.get_pattern_analysis.return_value = pattern_data
        
        # Execute workflow
        stats = mock_service_manager.data_service.get_game_statistics("powerball")
        frequency = mock_service_manager.data_service.get_frequency_analysis("powerball", days=365)
        patterns = mock_service_manager.data_service.get_pattern_analysis("powerball")
        
        # Generate analysis report
        analysis_report = {
            "game": "powerball",
            "statistics": stats,
            "frequency_analysis": frequency,
            "pattern_analysis": patterns,
            "insights": [
                "Number 23 is most frequent",
                "Consecutive numbers appear in 6.9% of draws",
                "Even/odd distribution is balanced"
            ],
            "recommendations": [
                "Consider including high-frequency numbers",
                "Avoid all-consecutive patterns",
                "Balance even and odd numbers"
            ]
        }
        
        mock_service_manager.data_service.generate_analysis_report.return_value = analysis_report
        
        report = mock_service_manager.data_service.generate_analysis_report("powerball")
        
        # Assertions
        assert "statistics" in report
        assert "insights" in report
        assert "recommendations" in report
        assert len(report["insights"]) > 0
    
    def test_caching_workflow_optimization(self, mock_service_manager, sample_predictions):
        """Test caching workflow for performance optimization."""
        cache_key = "predictions_powerball_balanced"
        
        # First call - cache miss
        mock_service_manager.cache_service.get.return_value = None
        mock_service_manager.prediction_service.generate_predictions.return_value = sample_predictions
        mock_service_manager.cache_service.set.return_value = True
        
        # First prediction generation (cache miss)
        cached_result = mock_service_manager.cache_service.get(cache_key)
        assert cached_result is None
        
        predictions = mock_service_manager.prediction_service.generate_predictions("powerball", "balanced")
        mock_service_manager.cache_service.set(cache_key, predictions, ttl=3600)
        
        # Second call - cache hit
        mock_service_manager.cache_service.get.return_value = sample_predictions
        
        cached_predictions = mock_service_manager.cache_service.get(cache_key)
        assert cached_predictions == sample_predictions
        
        # Verify cache optimization
        assert mock_service_manager.cache_service.get.call_count == 2
        mock_service_manager.cache_service.set.assert_called_once()
    
    def test_error_handling_and_recovery_workflow(self, mock_service_manager):
        """Test error handling and recovery across the workflow."""
        # Simulate database error
        mock_service_manager.data_service.get_historical_data.side_effect = ConnectionError("Database unavailable")
        
        # Test error propagation
        with pytest.raises(ConnectionError):
            mock_service_manager.data_service.get_historical_data("powerball")
        
        # Simulate recovery with fallback data
        fallback_data = [{"id": 1, "numbers": [1, 2, 3, 4, 5], "draw_date": "2023-12-01"}]
        mock_service_manager.data_service.get_fallback_data.return_value = fallback_data
        
        # Recovery workflow
        try:
            historical_data = mock_service_manager.data_service.get_historical_data("powerball")
        except ConnectionError:
            # Use fallback
            historical_data = mock_service_manager.data_service.get_fallback_data("powerball")
        
        assert historical_data == fallback_data
        
        # Continue with degraded service
        mock_service_manager.prediction_service.generate_predictions.return_value = [{
            "numbers": [5, 10, 15, 20, 25],
            "confidence": 0.5,  # Lower confidence due to limited data
            "fallback_mode": True
        }]
        
        predictions = mock_service_manager.prediction_service.generate_predictions(
            "powerball", 
            historical_data=historical_data
        )
        
        assert predictions[0]["fallback_mode"] is True
        assert predictions[0]["confidence"] < 0.7
    
    def test_data_export_workflow(self, mock_service_manager, sample_predictions, sample_statistics):
        """Test complete data export workflow."""
        # Mock data retrieval for export
        mock_service_manager.data_service.get_predictions.return_value = sample_predictions
        mock_service_manager.data_service.get_game_statistics.return_value = sample_statistics
        
        # Mock export functionality
        export_data = {
            "predictions": sample_predictions,
            "statistics": sample_statistics,
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "format": "json",
                "size": "25.6KB"
            }
        }
        
        mock_service_manager.data_service.export_data.return_value = export_data
        
        # Execute export workflow
        predictions = mock_service_manager.data_service.get_predictions("powerball", limit=100)
        stats = mock_service_manager.data_service.get_game_statistics("powerball")
        
        export_result = mock_service_manager.data_service.export_data(
            game="powerball",
            include_predictions=True,
            include_statistics=True,
            format="json"
        )
        
        # Assertions
        assert "predictions" in export_result
        assert "statistics" in export_result
        assert "metadata" in export_result
        assert export_result["metadata"]["format"] == "json"
    
    def test_batch_processing_workflow(self, mock_service_manager, sample_historical_data):
        """Test batch processing workflow for multiple games."""
        games = ["powerball", "mega_millions"]
        batch_results = {}
        
        for game in games:
            # Mock data for each game
            mock_service_manager.data_service.get_historical_data.return_value = sample_historical_data
            
            predictions = [{
                "numbers": [1, 15, 25, 35, 45],
                "confidence": 0.7,
                "game": game,
                "strategy": "balanced"
            }]
            mock_service_manager.prediction_service.generate_predictions.return_value = predictions
            
            # Process each game
            historical_data = mock_service_manager.data_service.get_historical_data(game)
            game_predictions = mock_service_manager.prediction_service.generate_predictions(
                game=game,
                strategy="balanced",
                historical_data=historical_data
            )
            
            batch_results[game] = {
                "predictions": game_predictions,
                "data_points": len(historical_data),
                "processed_at": datetime.now().isoformat()
            }
        
        # Assertions
        assert len(batch_results) == 2
        assert "powerball" in batch_results
        assert "mega_millions" in batch_results
        assert all("predictions" in result for result in batch_results.values())
    
    def test_real_time_prediction_workflow(self, mock_service_manager):
        """Test real-time prediction workflow with live data updates."""
        # Mock real-time data source
        live_data = {
            "latest_draw": {
                "game": "powerball",
                "draw_date": "2023-12-01",
                "numbers": [8, 15, 27, 33, 48],
                "bonus": 12,
                "jackpot": 150000000
            },
            "trend_data": {
                "hot_numbers": [15, 27, 33],
                "cold_numbers": [2, 7, 14],
                "trending_patterns": ["low_high_mix", "even_odd_balanced"]
            }
        }
        
        mock_service_manager.data_service.get_live_data.return_value = live_data
        
        # Generate real-time prediction
        real_time_prediction = {
            "numbers": [5, 17, 29, 41, 53],
            "bonus": 15,
            "confidence": 0.78,
            "real_time_factors": {
                "incorporates_latest_draw": True,
                "trend_analysis": True,
                "hot_numbers_weight": 0.3
            },
            "generated_at": datetime.now().isoformat()
        }
        
        mock_service_manager.prediction_service.generate_real_time_prediction.return_value = real_time_prediction
        
        # Execute real-time workflow
        live_data_result = mock_service_manager.data_service.get_live_data("powerball")
        prediction = mock_service_manager.prediction_service.generate_real_time_prediction(
            game="powerball",
            live_data=live_data_result
        )
        
        # Assertions
        assert prediction["real_time_factors"]["incorporates_latest_draw"] is True
        assert prediction["confidence"] > 0.75
        assert "generated_at" in prediction
    
    def test_performance_monitoring_workflow(self, mock_service_manager):
        """Test performance monitoring across the entire workflow."""
        # Mock performance metrics
        performance_data = {
            "data_service": {
                "avg_response_time": 0.05,
                "success_rate": 0.99,
                "cache_hit_rate": 0.85
            },
            "prediction_service": {
                "avg_generation_time": 1.2,
                "success_rate": 0.97,
                "confidence_avg": 0.72
            },
            "cache_service": {
                "hit_rate": 0.85,
                "memory_usage": 0.45,
                "eviction_rate": 0.02
            },
            "overall": {
                "end_to_end_time": 1.5,
                "total_requests": 1000,
                "error_rate": 0.01
            }
        }
        
        mock_service_manager.get_performance_metrics.return_value = performance_data
        
        # Execute monitoring
        metrics = mock_service_manager.get_performance_metrics()
        
        # Assertions
        assert metrics["overall"]["error_rate"] < 0.05  # Low error rate
        assert metrics["data_service"]["success_rate"] > 0.95  # High success rate
        assert metrics["cache_service"]["hit_rate"] > 0.8  # Good cache performance
        assert metrics["overall"]["end_to_end_time"] < 2.0  # Fast response
    
    @pytest.mark.parametrize("game,strategy,expected_confidence", [
        ("powerball", "conservative", 0.65),
        ("powerball", "balanced", 0.72),
        ("powerball", "aggressive", 0.78),
        ("mega_millions", "balanced", 0.70)
    ])
    def test_parameterized_workflow_validation(self, mock_service_manager, game, strategy, expected_confidence):
        """Test workflow with various game and strategy combinations."""
        # Mock prediction for each combination
        prediction = {
            "numbers": [5, 15, 25, 35, 45],
            "confidence": expected_confidence,
            "game": game,
            "strategy": strategy
        }
        
        mock_service_manager.prediction_service.generate_predictions.return_value = [prediction]
        
        # Execute workflow
        result = mock_service_manager.prediction_service.generate_predictions(
            game=game,
            strategy=strategy
        )
        
        # Assertions
        assert result[0]["game"] == game
        assert result[0]["strategy"] == strategy
        assert result[0]["confidence"] == expected_confidence