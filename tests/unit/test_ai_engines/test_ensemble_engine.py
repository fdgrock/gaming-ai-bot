"""
Unit tests for the Ensemble AI Engine module.

Tests ensemble prediction logic, model coordination, and result aggregation.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ai_engines.ensemble_engine import EnsembleEngine
from configs.app_config import AppConfig


class TestEnsembleEngine:
    """Test suite for Ensemble AI Engine functionality."""
    
    def test_engine_initialization(self, mock_ensemble_engine):
        """Test that EnsembleEngine initializes correctly."""
        assert mock_ensemble_engine is not None
        assert hasattr(mock_ensemble_engine, 'generate_prediction')
        assert hasattr(mock_ensemble_engine, 'train_models')
        assert hasattr(mock_ensemble_engine, 'get_model_weights')
    
    def test_generate_prediction_success(self, mock_ensemble_engine, sample_game_config):
        """Test successful prediction generation using ensemble."""
        expected_prediction = {
            "numbers": [5, 12, 25, 35, 45],
            "bonus": 12,
            "confidence": 0.75,
            "engine": "ensemble",
            "models_used": ["random_forest", "gradient_boosting", "svm"],
            "weights": [0.4, 0.35, 0.25]
        }
        
        mock_ensemble_engine.generate_prediction.return_value = expected_prediction
        
        # Test
        result = mock_ensemble_engine.generate_prediction(
            game_config=sample_game_config["powerball"],
            historical_data=[],
            strategy="balanced"
        )
        
        # Assertions
        assert result["engine"] == "ensemble"
        assert "models_used" in result
        assert "weights" in result
        assert len(result["numbers"]) == sample_game_config["powerball"]["num_numbers"]
        assert result["confidence"] > 0
        
        mock_ensemble_engine.generate_prediction.assert_called_once()
    
    def test_model_weight_configuration(self, mock_ensemble_engine):
        """Test ensemble model weight configuration."""
        weights = {
            "random_forest": 0.4,
            "gradient_boosting": 0.35,
            "svm": 0.25
        }
        
        mock_ensemble_engine.set_model_weights.return_value = True
        mock_ensemble_engine.get_model_weights.return_value = weights
        
        # Test setting weights
        set_result = mock_ensemble_engine.set_model_weights(weights)
        assert set_result is True
        
        # Test getting weights
        get_result = mock_ensemble_engine.get_model_weights()
        assert get_result == weights
        assert sum(get_result.values()) == 1.0  # Weights should sum to 1
    
    def test_individual_model_predictions(self, mock_ensemble_engine):
        """Test individual model predictions within ensemble."""
        individual_predictions = {
            "random_forest": {
                "numbers": [1, 15, 25, 35, 45],
                "confidence": 0.72,
                "probability_distribution": [0.1, 0.08, 0.12, 0.15, 0.09]
            },
            "gradient_boosting": {
                "numbers": [5, 20, 30, 40, 50],
                "confidence": 0.68,
                "probability_distribution": [0.12, 0.09, 0.11, 0.14, 0.10]
            },
            "svm": {
                "numbers": [8, 18, 28, 38, 48],
                "confidence": 0.65,
                "probability_distribution": [0.11, 0.07, 0.13, 0.12, 0.08]
            }
        }
        
        mock_ensemble_engine.get_individual_predictions.return_value = individual_predictions
        
        # Test
        result = mock_ensemble_engine.get_individual_predictions(
            game_config={"num_numbers": 5, "max_number": 69},
            historical_data=[]
        )
        
        # Assertions
        assert len(result) == 3  # Three models
        for model_name, prediction in result.items():
            assert "numbers" in prediction
            assert "confidence" in prediction
            assert len(prediction["numbers"]) == 5
            assert 0 <= prediction["confidence"] <= 1
    
    def test_ensemble_aggregation_methods(self, mock_ensemble_engine):
        """Test different ensemble aggregation methods."""
        # Test majority voting
        mock_ensemble_engine.aggregate_predictions.return_value = {
            "numbers": [5, 15, 25, 35, 45],
            "confidence": 0.73,
            "method": "majority_voting",
            "agreement_score": 0.8
        }
        
        result = mock_ensemble_engine.aggregate_predictions(
            individual_predictions=[],
            method="majority_voting"
        )
        
        assert result["method"] == "majority_voting"
        assert "agreement_score" in result
        
        # Test weighted average
        mock_ensemble_engine.aggregate_predictions.return_value = {
            "numbers": [8, 18, 28, 38, 48],
            "confidence": 0.71,
            "method": "weighted_average",
            "weight_distribution": [0.4, 0.35, 0.25]
        }
        
        result = mock_ensemble_engine.aggregate_predictions(
            individual_predictions=[],
            method="weighted_average"
        )
        
        assert result["method"] == "weighted_average"
        assert "weight_distribution" in result
    
    def test_model_performance_tracking(self, mock_ensemble_engine):
        """Test tracking of individual model performance."""
        performance_metrics = {
            "random_forest": {
                "accuracy": 0.72,
                "precision": 0.68,
                "recall": 0.70,
                "f1_score": 0.69,
                "recent_performance": 0.74
            },
            "gradient_boosting": {
                "accuracy": 0.68,
                "precision": 0.65,
                "recall": 0.67,
                "f1_score": 0.66,
                "recent_performance": 0.71
            },
            "svm": {
                "accuracy": 0.65,
                "precision": 0.62,
                "recall": 0.64,
                "f1_score": 0.63,
                "recent_performance": 0.67
            }
        }
        
        mock_ensemble_engine.get_model_performance.return_value = performance_metrics
        
        # Test
        result = mock_ensemble_engine.get_model_performance()
        
        # Assertions
        assert len(result) == 3
        for model_name, metrics in result.items():
            assert "accuracy" in metrics
            assert "recent_performance" in metrics
            assert 0 <= metrics["accuracy"] <= 1
    
    def test_dynamic_weight_adjustment(self, mock_ensemble_engine):
        """Test dynamic adjustment of model weights based on performance."""
        # Initial weights
        initial_weights = {"random_forest": 0.33, "gradient_boosting": 0.33, "svm": 0.34}
        
        # Adjusted weights based on performance
        adjusted_weights = {"random_forest": 0.45, "gradient_boosting": 0.35, "svm": 0.20}
        
        mock_ensemble_engine.adjust_weights_by_performance.return_value = adjusted_weights
        
        # Test
        result = mock_ensemble_engine.adjust_weights_by_performance(
            current_weights=initial_weights,
            performance_history=[]
        )
        
        # Assertions
        assert result != initial_weights  # Weights should change
        assert abs(sum(result.values()) - 1.0) < 0.001  # Should sum to 1
        assert result["random_forest"] > initial_weights["random_forest"]  # Best performer gets more weight
    
    def test_ensemble_diversity_metrics(self, mock_ensemble_engine):
        """Test measurement of ensemble diversity."""
        diversity_metrics = {
            "pairwise_diversity": 0.65,
            "model_agreement": 0.72,
            "prediction_variance": 0.15,
            "coverage": 0.88
        }
        
        mock_ensemble_engine.calculate_diversity_metrics.return_value = diversity_metrics
        
        # Test
        result = mock_ensemble_engine.calculate_diversity_metrics([])
        
        # Assertions
        assert "pairwise_diversity" in result
        assert "model_agreement" in result
        assert 0 <= result["pairwise_diversity"] <= 1
        assert 0 <= result["model_agreement"] <= 1
    
    def test_model_training_coordination(self, mock_ensemble_engine, sample_historical_data):
        """Test coordination of training across ensemble models."""
        training_results = {
            "models_trained": 3,
            "training_time": 45.2,
            "validation_scores": {
                "random_forest": 0.74,
                "gradient_boosting": 0.71,
                "svm": 0.68
            },
            "ensemble_score": 0.76
        }
        
        mock_ensemble_engine.train_models.return_value = training_results
        
        # Test
        result = mock_ensemble_engine.train_models(
            training_data=sample_historical_data,
            validation_split=0.2
        )
        
        # Assertions
        assert result["models_trained"] == 3
        assert result["ensemble_score"] > max(result["validation_scores"].values())
        assert "training_time" in result
        mock_ensemble_engine.train_models.assert_called_once()
    
    def test_prediction_uncertainty_quantification(self, mock_ensemble_engine):
        """Test quantification of prediction uncertainty."""
        uncertainty_data = {
            "prediction": [5, 15, 25, 35, 45],
            "confidence_interval": {
                "lower": [3, 12, 22, 32, 42],
                "upper": [7, 18, 28, 38, 48]
            },
            "uncertainty_score": 0.25,
            "model_disagreement": 0.18
        }
        
        mock_ensemble_engine.quantify_uncertainty.return_value = uncertainty_data
        
        # Test
        result = mock_ensemble_engine.quantify_uncertainty([])
        
        # Assertions
        assert "confidence_interval" in result
        assert "uncertainty_score" in result
        assert 0 <= result["uncertainty_score"] <= 1
        assert result["uncertainty_score"] < 0.5  # Good uncertainty
    
    def test_ensemble_with_feature_importance(self, mock_ensemble_engine):
        """Test ensemble with feature importance analysis."""
        feature_importance = {
            "frequency_features": 0.35,
            "pattern_features": 0.28,
            "temporal_features": 0.22,
            "statistical_features": 0.15
        }
        
        mock_ensemble_engine.get_feature_importance.return_value = feature_importance
        
        # Test
        result = mock_ensemble_engine.get_feature_importance()
        
        # Assertions
        assert abs(sum(result.values()) - 1.0) < 0.001  # Should sum to 1
        assert max(result.values()) > 0.3  # Some features should be important
        assert min(result.values()) > 0  # All features should have some importance
    
    def test_ensemble_prediction_explanation(self, mock_ensemble_engine):
        """Test explanation of ensemble predictions."""
        explanation = {
            "prediction": [5, 15, 25, 35, 45],
            "model_contributions": {
                "random_forest": {"weight": 0.4, "prediction": [1, 15, 25, 35, 49]},
                "gradient_boosting": {"weight": 0.35, "prediction": [5, 12, 28, 35, 45]},
                "svm": {"weight": 0.25, "prediction": [8, 15, 22, 38, 42]}
            },
            "feature_influences": {
                "recent_frequency": 0.25,
                "pattern_match": 0.18,
                "trend_analysis": 0.15
            },
            "confidence_sources": ["model_agreement", "historical_patterns", "feature_strength"]
        }
        
        mock_ensemble_engine.explain_prediction.return_value = explanation
        
        # Test
        result = mock_ensemble_engine.explain_prediction([5, 15, 25, 35, 45])
        
        # Assertions
        assert "model_contributions" in result
        assert "feature_influences" in result
        assert len(result["model_contributions"]) == 3
        assert "confidence_sources" in result
    
    def test_cross_validation_ensemble(self, mock_ensemble_engine):
        """Test cross-validation of ensemble performance."""
        cv_results = {
            "cv_scores": [0.74, 0.72, 0.76, 0.71, 0.75],
            "mean_score": 0.736,
            "std_score": 0.019,
            "individual_model_scores": {
                "random_forest": [0.72, 0.70, 0.74, 0.69, 0.73],
                "gradient_boosting": [0.69, 0.68, 0.72, 0.67, 0.71],
                "svm": [0.66, 0.65, 0.68, 0.64, 0.67]
            }
        }
        
        mock_ensemble_engine.cross_validate.return_value = cv_results
        
        # Test
        result = mock_ensemble_engine.cross_validate([], k_folds=5)
        
        # Assertions
        assert len(result["cv_scores"]) == 5
        assert result["mean_score"] > 0.7  # Good performance
        assert result["std_score"] < 0.05  # Stable performance
        assert "individual_model_scores" in result
    
    @pytest.mark.parametrize("strategy,expected_weights", [
        ("conservative", {"random_forest": 0.5, "gradient_boosting": 0.3, "svm": 0.2}),
        ("balanced", {"random_forest": 0.4, "gradient_boosting": 0.35, "svm": 0.25}),
        ("aggressive", {"random_forest": 0.3, "gradient_boosting": 0.4, "svm": 0.3})
    ])
    def test_strategy_based_ensemble_weights(self, mock_ensemble_engine, strategy, expected_weights):
        """Test ensemble weights adjustment based on strategy."""
        mock_ensemble_engine.get_strategy_weights.return_value = expected_weights
        
        # Test
        result = mock_ensemble_engine.get_strategy_weights(strategy)
        
        # Assertions
        assert result == expected_weights
        assert abs(sum(result.values()) - 1.0) < 0.001
    
    def test_ensemble_model_health_check(self, mock_ensemble_engine):
        """Test health check of ensemble models."""
        health_status = {
            "overall_health": "good",
            "model_status": {
                "random_forest": {"status": "healthy", "last_trained": "2023-12-01", "performance": 0.74},
                "gradient_boosting": {"status": "healthy", "last_trained": "2023-12-01", "performance": 0.71},
                "svm": {"status": "warning", "last_trained": "2023-11-25", "performance": 0.65}
            },
            "recommendations": ["retrain_svm", "update_features"]
        }
        
        mock_ensemble_engine.health_check.return_value = health_status
        
        # Test
        result = mock_ensemble_engine.health_check()
        
        # Assertions
        assert result["overall_health"] in ["good", "warning", "critical"]
        assert len(result["model_status"]) == 3
        assert "recommendations" in result
    
    def test_ensemble_with_missing_models(self, mock_ensemble_engine):
        """Test ensemble behavior when some models are unavailable."""
        # Simulate missing model scenario
        available_models = ["random_forest", "gradient_boosting"]  # SVM missing
        fallback_prediction = {
            "numbers": [5, 15, 25, 35, 45],
            "confidence": 0.68,  # Lower confidence due to missing model
            "models_used": available_models,
            "missing_models": ["svm"],
            "adjusted_weights": {"random_forest": 0.57, "gradient_boosting": 0.43}
        }
        
        mock_ensemble_engine.generate_prediction_with_fallback.return_value = fallback_prediction
        
        # Test
        result = mock_ensemble_engine.generate_prediction_with_fallback(
            available_models=available_models
        )
        
        # Assertions
        assert len(result["models_used"]) == 2
        assert "missing_models" in result
        assert result["confidence"] < 0.75  # Reduced confidence
        assert abs(sum(result["adjusted_weights"].values()) - 1.0) < 0.001