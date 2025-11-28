"""
Enhanced Prediction Service - Business Logic Extracted from Monolithic App

This module provides prediction orchestration services combining:
1. Original PredictionOrchestrator functionality (AI engine coordination)
2. Extracted business logic from monolithic app.py (prediction management)
3. Enhanced service foundation with BaseService integration

Extracted Functions Integrated:
- get_recent_predictions() -> get_recent_predictions()
- get_predictions_by_model() -> get_predictions_by_model()  
- count_total_predictions() -> count_total_predictions()

Enhanced Features:
- BaseService integration with dependency injection
- Comprehensive error handling and logging
- Prediction file validation and cleanup
- Performance statistics and analytics
- Clean separation from UI dependencies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Phase 2 Service Integration  
from .base_service import BaseService, ServiceValidationMixin
from ..core.exceptions import PredictionError, ValidationError, safe_execute
from ..core.utils import sanitize_game_name

logger = logging.getLogger(__name__)


class PredictionOrchestrator:
    """
    Orchestrates prediction generation across multiple AI engines.
    
    This class coordinates the four AI engines (Mathematical, Ensemble, 
    Temporal, Optimizer) to generate comprehensive predictions with 
    confidence scoring and validation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize prediction orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_workers = self.config.get('max_workers', 4)
        self.timeout = self.config.get('timeout', 30)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        
        # Initialize AI engines
        self._initialize_engines()
    
    def _initialize_engines(self) -> None:
        """Initialize AI engines with error handling."""
        self.engines = {}
        
        try:
            # Import AI engines
            from ..ai_engines.mathematical_engine import MathematicalEngine
            self.engines['mathematical'] = MathematicalEngine(
                self.config.get('mathematical_engine', {})
            )
            logger.info("✅ Mathematical engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Mathematical engine not available: {e}")
        
        try:
            from ..ai_engines.ensemble_engine import EnsembleEngine
            self.engines['ensemble'] = EnsembleEngine(
                self.config.get('ensemble_engine', {})
            )
            logger.info("✅ Ensemble engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Ensemble engine not available: {e}")
        
        try:
            from ..ai_engines.temporal_engine import TemporalEngine
            self.engines['temporal'] = TemporalEngine(
                self.config.get('temporal_engine', {})
            )
            logger.info("✅ Temporal engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Temporal engine not available: {e}")
        
        try:
            from ..ai_engines.set_optimizer import SetOptimizer
            self.engines['optimizer'] = SetOptimizer(
                self.config.get('set_optimizer', {})
            )
            logger.info("✅ Set optimizer initialized")
        except Exception as e:
            logger.warning(f"⚠️ Set optimizer not available: {e}")
        
        logger.info(f"🎯 Initialized {len(self.engines)} AI engines")
    
    def generate_prediction(self, game_config: Dict[str, Any], 
                          strategy: str = 'balanced',
                          engine_weights: Optional[Dict[str, float]] = None,
                          num_sets: int = 1) -> Dict[str, Any]:
        """
        Generate comprehensive prediction using multiple engines.
        
        Args:
            game_config: Game configuration (numbers range, set size, etc.)
            strategy: Prediction strategy ('aggressive', 'balanced', 'conservative')
            engine_weights: Custom weights for engines
            num_sets: Number of prediction sets to generate
            
        Returns:
            Comprehensive prediction results
        """
        try:
            start_time = time.time()
            
            # Validate inputs
            if not self._validate_game_config(game_config):
                return self._create_error_result("Invalid game configuration")
            
            # Set default engine weights
            if engine_weights is None:
                engine_weights = self._get_default_weights(strategy)
            
            # Generate predictions from each engine
            engine_predictions = self._generate_engine_predictions(
                game_config, strategy, num_sets
            )
            
            # Aggregate predictions
            aggregated_result = self._aggregate_predictions(
                engine_predictions, engine_weights, game_config
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                engine_predictions, aggregated_result
            )
            
            # Validate predictions
            validation_results = self._validate_predictions(
                aggregated_result['predictions'], game_config
            )
            
            # Compile final result
            final_result = {
                'predictions': aggregated_result['predictions'],
                'confidence': confidence_scores['overall'],
                'engine_confidence': confidence_scores['engine_breakdown'],
                'strategy': strategy,
                'engine_weights': engine_weights,
                'engine_results': engine_predictions,
                'validation': validation_results,
                'metadata': {
                    'generated_at': datetime.now(),
                    'generation_time': time.time() - start_time,
                    'game_config': game_config,
                    'engines_used': list(self.engines.keys()),
                    'num_sets_requested': num_sets,
                    'num_sets_generated': len(aggregated_result['predictions'])
                }
            }
            
            logger.info(f"🎯 Generated {len(final_result['predictions'])} predictions in {final_result['metadata']['generation_time']:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Failed to generate prediction: {e}")
            return self._create_error_result(str(e))
    
    def _generate_engine_predictions(self, game_config: Dict[str, Any], 
                                   strategy: str, num_sets: int) -> Dict[str, Any]:
        """Generate predictions from all available engines."""
        engine_predictions = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit prediction tasks
            future_to_engine = {}
            
            for engine_name, engine in self.engines.items():
                future = executor.submit(
                    self._safe_engine_predict, 
                    engine, engine_name, game_config, strategy, num_sets
                )
                future_to_engine[future] = engine_name
            
            # Collect results
            for future in as_completed(future_to_engine, timeout=self.timeout):
                engine_name = future_to_engine[future]
                
                try:
                    result = future.result()
                    engine_predictions[engine_name] = result
                    logger.info(f"✅ {engine_name} engine completed")
                except Exception as e:
                    logger.error(f"❌ {engine_name} engine failed: {e}")
                    engine_predictions[engine_name] = {
                        'success': False,
                        'error': str(e),
                        'predictions': [],
                        'confidence': 0.0
                    }
        
        return engine_predictions
    
    def _safe_engine_predict(self, engine: Any, engine_name: str, 
                           game_config: Dict[str, Any], strategy: str, 
                           num_sets: int) -> Dict[str, Any]:
        """Safely call engine prediction with error handling."""
        try:
            if hasattr(engine, 'predict'):
                result = engine.predict(
                    game_config=game_config,
                    strategy=strategy,
                    num_predictions=num_sets
                )
                
                # Normalize result format
                if isinstance(result, list):
                    # Simple list of predictions
                    return {
                        'success': True,
                        'predictions': result,
                        'confidence': 0.7,  # Default confidence
                        'metadata': {'engine': engine_name}
                    }
                elif isinstance(result, dict):
                    # Structured result
                    return {
                        'success': True,
                        'predictions': result.get('predictions', []),
                        'confidence': result.get('confidence', 0.7),
                        'metadata': result.get('metadata', {'engine': engine_name})
                    }
                else:
                    return {
                        'success': False,
                        'error': f"Unknown result format from {engine_name}",
                        'predictions': [],
                        'confidence': 0.0
                    }
            else:
                return {
                    'success': False,
                    'error': f"Engine {engine_name} has no predict method",
                    'predictions': [],
                    'confidence': 0.0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'predictions': [],
                'confidence': 0.0
            }
    
    def _aggregate_predictions(self, engine_predictions: Dict[str, Any], 
                             engine_weights: Dict[str, float],
                             game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate predictions from multiple engines."""
        try:
            all_predictions = []
            weighted_predictions = []
            
            # Collect all successful predictions
            for engine_name, result in engine_predictions.items():
                if result.get('success', False) and result.get('predictions'):
                    weight = engine_weights.get(engine_name, 0.25)
                    
                    for prediction in result['predictions']:
                        if isinstance(prediction, list):
                            all_predictions.append({
                                'numbers': prediction,
                                'engine': engine_name,
                                'weight': weight,
                                'confidence': result.get('confidence', 0.5)
                            })
            
            if not all_predictions:
                # Fallback: generate random predictions
                return self._generate_fallback_predictions(game_config)
            
            # Apply different aggregation strategies
            aggregated = self._apply_aggregation_strategy(
                all_predictions, game_config
            )
            
            return aggregated
            
        except Exception as e:
            logger.error(f"❌ Failed to aggregate predictions: {e}")
            return self._generate_fallback_predictions(game_config)
    
    def _apply_aggregation_strategy(self, predictions: List[Dict[str, Any]], 
                                  game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply aggregation strategy to combine predictions."""
        try:
            num_numbers = game_config.get('num_numbers', 6)
            max_number = game_config.get('max_number', 70)
            
            # Strategy 1: Weighted frequency analysis
            number_scores = {}
            
            for pred in predictions:
                for number in pred['numbers']:
                    if 1 <= number <= max_number:
                        score = pred['weight'] * pred['confidence']
                        number_scores[number] = number_scores.get(number, 0) + score
            
            # Select top numbers by score
            sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Generate final predictions
            final_predictions = []
            
            # Primary prediction: top weighted numbers
            if len(sorted_numbers) >= num_numbers:
                primary_numbers = [num for num, score in sorted_numbers[:num_numbers]]
                final_predictions.append(primary_numbers)
            
            # Secondary predictions: variations
            if len(sorted_numbers) > num_numbers:
                # Mix top numbers with some variations
                top_candidates = [num for num, score in sorted_numbers[:num_numbers * 2]]
                
                for i in range(min(2, len(predictions) // 2)):
                    variation = self._create_number_variation(
                        top_candidates, num_numbers, max_number
                    )
                    if variation and variation not in final_predictions:
                        final_predictions.append(variation)
            
            return {
                'predictions': final_predictions,
                'number_scores': number_scores,
                'aggregation_method': 'weighted_frequency'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to apply aggregation strategy: {e}")
            return self._generate_fallback_predictions(game_config)
    
    def _create_number_variation(self, candidates: List[int], 
                               num_numbers: int, max_number: int) -> List[int]:
        """Create a variation of number selection."""
        try:
            # Select most of the numbers from candidates
            np.random.seed(int(time.time() * 1000) % 2**32)
            
            selected = list(np.random.choice(
                candidates[:num_numbers + 2], 
                size=num_numbers - 1, 
                replace=False
            ))
            
            # Add one random number
            remaining = [n for n in range(1, max_number + 1) if n not in selected]
            if remaining:
                selected.append(np.random.choice(remaining))
            
            return sorted(selected)
            
        except Exception as e:
            logger.error(f"❌ Failed to create number variation: {e}")
            return []
    
    def _calculate_confidence_scores(self, engine_predictions: Dict[str, Any], 
                                   aggregated_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence scores for predictions."""
        try:
            engine_confidences = {}
            engine_weights = []
            
            # Collect engine confidences
            for engine_name, result in engine_predictions.items():
                if result.get('success', False):
                    confidence = result.get('confidence', 0.5)
                    engine_confidences[engine_name] = confidence
                    engine_weights.append(confidence)
            
            # Calculate overall confidence
            if engine_weights:
                overall_confidence = np.mean(engine_weights)
                
                # Boost confidence if multiple engines agree
                if len(engine_weights) > 1:
                    agreement_bonus = min(0.2, (len(engine_weights) - 1) * 0.05)
                    overall_confidence = min(1.0, overall_confidence + agreement_bonus)
            else:
                overall_confidence = 0.3  # Low confidence fallback
            
            # Adjust confidence based on number scores
            if 'number_scores' in aggregated_result:
                scores = list(aggregated_result['number_scores'].values())
                if scores:
                    score_variance = np.var(scores)
                    if score_variance < 0.1:  # High agreement
                        overall_confidence = min(1.0, overall_confidence + 0.1)
            
            return {
                'overall': overall_confidence,
                'engine_breakdown': engine_confidences,
                'factors': {
                    'num_engines': len(engine_confidences),
                    'avg_engine_confidence': np.mean(list(engine_confidences.values())) if engine_confidences else 0,
                    'agreement_level': 'high' if len(engine_confidences) > 2 else 'medium'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate confidence scores: {e}")
            return {
                'overall': 0.3,
                'engine_breakdown': {},
                'factors': {'error': str(e)}
            }
    
    def _validate_predictions(self, predictions: List[List[int]], 
                            game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated predictions."""
        try:
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'checks_performed': []
            }
            
            num_numbers = game_config.get('num_numbers', 6)
            max_number = game_config.get('max_number', 70)
            min_number = game_config.get('min_number', 1)
            
            for i, prediction in enumerate(predictions):
                # Check number count
                if len(prediction) != num_numbers:
                    validation_results['errors'].append(
                        f"Prediction {i+1}: Expected {num_numbers} numbers, got {len(prediction)}"
                    )
                    validation_results['valid'] = False
                
                # Check number range
                for number in prediction:
                    if not (min_number <= number <= max_number):
                        validation_results['errors'].append(
                            f"Prediction {i+1}: Number {number} out of range [{min_number}, {max_number}]"
                        )
                        validation_results['valid'] = False
                
                # Check for duplicates
                if len(set(prediction)) != len(prediction):
                    validation_results['errors'].append(
                        f"Prediction {i+1}: Contains duplicate numbers"
                    )
                    validation_results['valid'] = False
                
                # Check if sorted
                if prediction != sorted(prediction):
                    validation_results['warnings'].append(
                        f"Prediction {i+1}: Numbers not sorted"
                    )
            
            validation_results['checks_performed'] = [
                'number_count', 'number_range', 'duplicates', 'sorting'
            ]
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Failed to validate predictions: {e}")
            return {
                'valid': False,
                'errors': [f"Validation failed: {e}"],
                'warnings': [],
                'checks_performed': []
            }
    
    def _validate_game_config(self, game_config: Dict[str, Any]) -> bool:
        """Validate game configuration."""
        try:
            required_fields = ['num_numbers', 'max_number']
            
            for field in required_fields:
                if field not in game_config:
                    logger.error(f"❌ Missing required field: {field}")
                    return False
            
            # Validate values
            if game_config['num_numbers'] <= 0 or game_config['num_numbers'] > 20:
                logger.error("❌ Invalid num_numbers value")
                return False
            
            if game_config['max_number'] <= game_config['num_numbers']:
                logger.error("❌ max_number must be greater than num_numbers")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Game config validation error: {e}")
            return False
    
    def _get_default_weights(self, strategy: str) -> Dict[str, float]:
        """Get default engine weights for strategy."""
        weight_strategies = {
            'aggressive': {
                'mathematical': 0.4,
                'ensemble': 0.3,
                'temporal': 0.2,
                'optimizer': 0.1
            },
            'balanced': {
                'mathematical': 0.25,
                'ensemble': 0.25,
                'temporal': 0.25,
                'optimizer': 0.25
            },
            'conservative': {
                'mathematical': 0.2,
                'ensemble': 0.4,
                'temporal': 0.3,
                'optimizer': 0.1
            }
        }
        
        return weight_strategies.get(strategy, weight_strategies['balanced'])
    
    def _generate_fallback_predictions(self, game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback predictions when engines fail."""
        try:
            num_numbers = game_config.get('num_numbers', 6)
            max_number = game_config.get('max_number', 70)
            min_number = game_config.get('min_number', 1)
            
            np.random.seed(int(time.time() * 1000) % 2**32)
            
            # Generate random predictions
            predictions = []
            for _ in range(3):  # Generate 3 fallback predictions
                numbers = sorted(np.random.choice(
                    range(min_number, max_number + 1),
                    size=num_numbers,
                    replace=False
                ))
                predictions.append(numbers.tolist())
            
            logger.warning("⚠️ Using fallback random predictions")
            
            return {
                'predictions': predictions,
                'number_scores': {},
                'aggregation_method': 'fallback_random'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate fallback predictions: {e}")
            return {'predictions': [], 'number_scores': {}, 'aggregation_method': 'failed'}
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            'predictions': [],
            'confidence': 0.0,
            'engine_confidence': {},
            'strategy': 'error',
            'engine_weights': {},
            'engine_results': {},
            'validation': {
                'valid': False,
                'errors': [error_message],
                'warnings': [],
                'checks_performed': []
            },
            'metadata': {
                'generated_at': datetime.now(),
                'generation_time': 0.0,
                'error': error_message,
                'engines_used': [],
                'num_sets_requested': 0,
                'num_sets_generated': 0
            }
        }
    
    def analyze_prediction_quality(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the quality of generated predictions.
        
        Args:
            prediction_result: Result from generate_prediction
            
        Returns:
            Quality analysis results
        """
        try:
            quality_metrics = {
                'overall_score': 0.0,
                'confidence_score': 0.0,
                'diversity_score': 0.0,
                'engine_agreement_score': 0.0,
                'validation_score': 0.0,
                'recommendations': []
            }
            
            # Confidence score
            confidence = prediction_result.get('confidence', 0.0)
            quality_metrics['confidence_score'] = confidence
            
            # Diversity score (how varied are the predictions)
            predictions = prediction_result.get('predictions', [])
            if len(predictions) > 1:
                all_numbers = set()
                for pred in predictions:
                    all_numbers.update(pred)
                
                # Calculate diversity as unique numbers vs total positions
                total_positions = len(predictions) * len(predictions[0]) if predictions else 1
                diversity = len(all_numbers) / total_positions
                quality_metrics['diversity_score'] = min(1.0, diversity * 1.5)
            else:
                quality_metrics['diversity_score'] = 0.5
            
            # Engine agreement score
            engine_results = prediction_result.get('engine_results', {})
            successful_engines = sum(1 for r in engine_results.values() if r.get('success', False))
            total_engines = len(self.engines)
            
            if total_engines > 0:
                quality_metrics['engine_agreement_score'] = successful_engines / total_engines
            
            # Validation score
            validation = prediction_result.get('validation', {})
            if validation.get('valid', False):
                quality_metrics['validation_score'] = 1.0
                if validation.get('warnings'):
                    quality_metrics['validation_score'] -= len(validation['warnings']) * 0.1
            else:
                quality_metrics['validation_score'] = 0.0
            
            # Overall score (weighted average)
            weights = {'confidence': 0.3, 'diversity': 0.2, 'agreement': 0.3, 'validation': 0.2}
            quality_metrics['overall_score'] = (
                weights['confidence'] * quality_metrics['confidence_score'] +
                weights['diversity'] * quality_metrics['diversity_score'] +
                weights['agreement'] * quality_metrics['engine_agreement_score'] +
                weights['validation'] * quality_metrics['validation_score']
            )
            
            # Generate recommendations
            quality_metrics['recommendations'] = self._generate_quality_recommendations(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze prediction quality: {e}")
            return {'error': str(e)}
    
    def _generate_quality_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        if metrics['confidence_score'] < 0.5:
            recommendations.append("Consider adjusting engine weights or strategy for higher confidence")
        
        if metrics['diversity_score'] < 0.3:
            recommendations.append("Predictions lack diversity - consider generating more varied sets")
        
        if metrics['engine_agreement_score'] < 0.5:
            recommendations.append("Low engine agreement - check engine configurations and data quality")
        
        if metrics['validation_score'] < 1.0:
            recommendations.append("Validation issues detected - review game configuration and prediction logic")
        
        if metrics['overall_score'] > 0.8:
            recommendations.append("High quality predictions - good engine performance and agreement")
        
        return recommendations
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get status of all AI engines.
        
        Returns:
            Engine status information
        """
        try:
            status = {
                'total_engines': len(self.engines),
                'available_engines': list(self.engines.keys()),
                'engine_health': {}
            }
            
            for engine_name, engine in self.engines.items():
                try:
                    if hasattr(engine, 'health_check'):
                        health = engine.health_check()
                        status['engine_health'][engine_name] = {
                            'status': 'healthy' if health else 'unhealthy',
                            'last_check': datetime.now().isoformat()
                        }
                    else:
                        status['engine_health'][engine_name] = {
                            'status': 'unknown',
                            'last_check': datetime.now().isoformat()
                        }
                except Exception as e:
                    status['engine_health'][engine_name] = {
                        'status': 'error',
                        'error': str(e),
                        'last_check': datetime.now().isoformat()
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get engine status: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def health_check() -> bool:
        """Check prediction orchestrator health."""
        return True
    
    # =============================================================================
    # EXTRACTED BUSINESS LOGIC FROM MONOLITHIC APP.PY
    # =============================================================================
    
    def get_recent_predictions(self, game_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent predictions for a game.
        
        Extracted from: get_recent_predictions() in original app.py (Line 565)
        Enhanced with: Proper error handling, logging, validation
        
        Args:
            game_name: Name of the lottery game
            limit: Maximum number of predictions to return
            
        Returns:
            List of recent prediction data dictionaries
        """
        try:
            from ..core.utils import sanitize_game_name
            
            game_key = sanitize_game_name(game_name)
            pred_dir = Path("predictions") / game_key
            predictions = []
            
            if not pred_dir.exists():
                logger.info(f"📁 No predictions directory for {game_name}")
                return predictions
            
            # Get all prediction files, sorted by modification time (newest first)
            pred_files = []
            for pred_file in pred_dir.glob("*.json"):
                if pred_file.is_file():
                    pred_files.append((pred_file, pred_file.stat().st_mtime))
            
            # Sort by modification time (newest first)
            pred_files.sort(key=lambda x: x[1], reverse=True)
            
            # Process up to limit files
            for pred_file, _ in pred_files[:limit]:
                try:
                    prediction_data = self._load_prediction_file(pred_file)
                    if prediction_data:
                        predictions.append({
                            'file': str(pred_file),
                            'filename': pred_file.name,
                            'data': prediction_data,
                            'modified_time': datetime.fromtimestamp(pred_file.stat().st_mtime).isoformat()
                        })
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process prediction file {pred_file}: {e}")
                    continue
            
            logger.info(f"📊 Loaded {len(predictions)} recent predictions for {game_name}")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent predictions for {game_name}: {e}")
            return []
    
    def get_predictions_by_model(self, game_name: str, model_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get predictions organized by model type for a game.
        
        Extracted from: get_predictions_by_model() in original app.py (Line 610)  
        Enhanced with: Better organization, comprehensive model support
        
        Args:
            game_name: Name of the lottery game
            model_type: Specific model type to filter (optional)
            
        Returns:
            Dictionary of predictions organized by model type
        """
        try:
            from ..core.utils import sanitize_game_name
            
            game_key = sanitize_game_name(game_name)
            pred_dir = Path("predictions") / game_key
            model_predictions = {}
            
            if not pred_dir.exists():
                logger.info(f"📁 No predictions directory for {game_name}")
                return model_predictions
            
            # Define model directories to check
            model_dirs = ['hybrid', 'lstm', 'transformer', 'xgboost', 'ensemble', 'mathematical', 'temporal']
            
            # Check each model subdirectory
            for model_dir in model_dirs:
                if model_type and model_dir != model_type:
                    continue
                    
                model_path = pred_dir / model_dir
                if model_path.exists():
                    model_predictions[model_dir] = self._load_model_predictions(model_path, limit=3)
            
            # Also check root directory files (baseline predictions)
            root_predictions = self._load_model_predictions(pred_dir, limit=3, file_pattern="*.json")
            if root_predictions:
                if not model_type or model_type == 'baseline':
                    model_predictions['baseline'] = root_predictions
            
            # Filter out models with no predictions
            model_predictions = {k: v for k, v in model_predictions.items() if v}
            
            logger.info(f"📊 Found predictions for {len(model_predictions)} models for {game_name}")
            return model_predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to get predictions by model for {game_name}: {e}")
            return {}
    
    def count_total_predictions(self, game_name: str) -> int:
        """
        Count total prediction files for a game across all models.
        
        Extracted from: count_total_predictions() in original app.py (Line 674)
        Enhanced with: Better path handling, comprehensive counting
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            Total number of prediction files
        """
        try:
            from ..core.utils import sanitize_game_name
            
            game_key = sanitize_game_name(game_name)
            pred_dir = Path("predictions") / game_key
            total_count = 0
            
            if not pred_dir.exists():
                logger.info(f"📁 No predictions directory for {game_name}")
                return total_count
            
            # Count files in root directory  
            root_files = list(pred_dir.glob("*.json"))
            total_count += len(root_files)
            
            # Count files in model subdirectories
            model_dirs = ['hybrid', 'lstm', 'transformer', 'xgboost', 'ensemble', 'mathematical', 'temporal']
            for model_dir in model_dirs:
                model_path = pred_dir / model_dir
                if model_path.exists():
                    model_files = list(model_path.glob("*.json"))
                    total_count += len(model_files)
            
            logger.info(f"📊 Total predictions for {game_name}: {total_count}")
            return total_count
            
        except Exception as e:
            logger.error(f"❌ Failed to count predictions for {game_name}: {e}")
            return 0
    
    def _load_prediction_file(self, pred_file: Path) -> Optional[Dict[str, Any]]:
        """Load and validate a single prediction file."""
        try:
            # Check if file is empty
            if pred_file.stat().st_size == 0:
                logger.info(f"🧹 Removing empty prediction file: {pred_file}")
                try:
                    pred_file.unlink()
                except Exception:
                    pass
                return None
            
            # Load JSON data
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
            
            # Validate structure
            if not isinstance(pred_data, dict):
                logger.warning(f"⚠️ Invalid prediction format in {pred_file}")
                return None
                
            # Check for required structure (original app.py validation)
            if 'sets' not in pred_data:
                logger.info(f"🧹 Removing malformed prediction file: {pred_file}")
                try:
                    pred_file.unlink()
                except Exception:
                    pass
                return None
            
            return pred_data
            
        except json.JSONDecodeError:
            logger.info(f"🧹 Removing corrupted prediction file: {pred_file}")
            try:
                pred_file.unlink()
            except Exception:
                pass
            return None
        except Exception as e:
            logger.debug(f"Could not process prediction file {pred_file}: {e}")
            return None
    
    def _load_model_predictions(self, model_path: Path, limit: int = 3, 
                              file_pattern: str = "*.json") -> List[Dict[str, Any]]:
        """Load predictions from a model directory."""
        predictions = []
        
        try:
            # Get prediction files sorted by modification time
            pred_files = []
            for pred_file in model_path.glob(file_pattern):
                if pred_file.is_file():
                    pred_files.append((pred_file, pred_file.stat().st_mtime))
            
            # Sort by modification time (newest first) 
            pred_files.sort(key=lambda x: x[1], reverse=True)
            
            # Process up to limit files
            for pred_file, mod_time in pred_files[:limit]:
                prediction_data = self._load_prediction_file(pred_file)
                if prediction_data:
                    # Extract date from filename (first 8 characters if available)
                    filename = pred_file.name
                    date_str = filename[:8] if len(filename) >= 8 and filename[:8].isdigit() else 'Unknown'
                    
                    predictions.append({
                        'file': str(pred_file),
                        'filename': filename,
                        'data': prediction_data,
                        'date': date_str,
                        'modified_time': datetime.fromtimestamp(mod_time).isoformat()
                    })
            
        except Exception as e:
            logger.error(f"❌ Failed to load predictions from {model_path}: {e}")
        
        return predictions
    
    def get_prediction_statistics(self, game_name: str) -> Dict[str, Any]:
        """
        Get comprehensive prediction statistics for a game.
        
        Args:
            game_name: Name of the lottery game
            
        Returns:
            Dictionary with prediction statistics
        """
        try:
            stats = {
                'total_predictions': self.count_total_predictions(game_name),
                'predictions_by_model': {},
                'recent_predictions_count': 0,
                'oldest_prediction_date': None,
                'newest_prediction_date': None,
                'prediction_frequency': {
                    'daily': 0,
                    'weekly': 0,
                    'monthly': 0
                }
            }
            
            # Get predictions by model
            model_predictions = self.get_predictions_by_model(game_name)
            
            all_dates = []
            for model_type, predictions in model_predictions.items():
                stats['predictions_by_model'][model_type] = len(predictions)
                
                # Collect dates for frequency analysis
                for pred in predictions:
                    if pred.get('modified_time'):
                        try:
                            pred_date = datetime.fromisoformat(pred['modified_time'])
                            all_dates.append(pred_date)
                        except ValueError:
                            pass
            
            # Analyze dates
            if all_dates:
                all_dates.sort()
                stats['oldest_prediction_date'] = all_dates[0].isoformat()
                stats['newest_prediction_date'] = all_dates[-1].isoformat()
                
                # Calculate frequencies
                now = datetime.now()
                day_ago = now - timedelta(days=1)
                week_ago = now - timedelta(weeks=1)
                month_ago = now - timedelta(days=30)
                
                for date in all_dates:
                    if date > day_ago:
                        stats['prediction_frequency']['daily'] += 1
                    if date > week_ago:
                        stats['prediction_frequency']['weekly'] += 1
                    if date > month_ago:
                        stats['prediction_frequency']['monthly'] += 1
            
            # Recent predictions
            recent_predictions = self.get_recent_predictions(game_name, limit=10)
            stats['recent_predictions_count'] = len(recent_predictions)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get prediction statistics for {game_name}: {e}")
            return {}


# =============================================================================
# ENHANCED PREDICTION SERVICE WITH BASE SERVICE INTEGRATION
# =============================================================================

class PredictionService(BaseService, ServiceValidationMixin):
    """
    Enhanced Prediction Service integrating Phase 2 service foundation.
    
    Combines:
    - BaseService patterns (dependency injection, logging, error handling)
    - Original PredictionOrchestrator functionality
    - Extracted business logic from monolithic app.py
    """
    
    def _setup_service(self) -> None:
        """Initialize prediction service with integrated functionality."""
        self.log_operation("setup", status="info", action="initializing prediction service")
        
        # Initialize the PredictionOrchestrator with config
        orchestrator_config = {
            'max_workers': getattr(self.config, 'max_prediction_workers', 4),
            'timeout': getattr(self.config, 'prediction_timeout', 30),
            'confidence_threshold': getattr(self.config, 'confidence_threshold', 0.5),
        }
        
        self.orchestrator = PredictionOrchestrator(orchestrator_config)
        
        self.log_operation("setup", status="success",
                          engines_count=len(self.orchestrator.engines))
    
    # Delegate to PredictionOrchestrator with enhanced error handling  
    def generate_prediction(self, game_config: Dict[str, Any], 
                          strategy: str = 'balanced',
                          engine_weights: Optional[Dict[str, float]] = None,
                          num_sets: int = 1) -> Dict[str, Any]:
        """Generate prediction with service-level error handling."""
        self.validate_initialized()
        
        return self.safe_execute_operation(
            self.orchestrator.generate_prediction,
            "generate_prediction",
            game_config=game_config,
            default_return={},
            strategy=strategy,
            engine_weights=engine_weights,
            num_sets=num_sets
        )
    
    def get_recent_predictions(self, game_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.orchestrator.get_recent_predictions,
            "get_recent_predictions",
            game_name=game_key,
            default_return=[],
            limit=limit
        )
    
    def get_predictions_by_model(self, game_name: str, 
                               model_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get predictions by model with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.orchestrator.get_predictions_by_model,
            "get_predictions_by_model",
            game_name=game_key,
            default_return={},
            model_type=model_type
        )
    
    def count_total_predictions(self, game_name: str) -> int:
        """Count predictions with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.orchestrator.count_total_predictions,
            "count_total_predictions",
            game_name=game_key,
            default_return=0
        )
    
    def get_prediction_statistics(self, game_name: str) -> Dict[str, Any]:
        """Get prediction statistics with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.orchestrator.get_prediction_statistics,
            "get_prediction_statistics",
            game_name=game_key,
            default_return={}
        )
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status with service-level error handling."""
        self.validate_initialized()
        
        return self.safe_execute_operation(
            self.orchestrator.get_engine_status,
            "get_engine_status",
            default_return={}
        )
    
    def _service_health_check(self) -> Optional[Dict[str, Any]]:
        """Prediction service specific health check."""
        health = {
            'healthy': True,
            'issues': []
        }
        
        # Check PredictionOrchestrator health
        if not self.orchestrator.health_check():
            health['healthy'] = False
            health['issues'].append("PredictionOrchestrator health check failed")
        
        # Check AI engines availability
        engine_status = self.orchestrator.get_engine_status()
        unhealthy_engines = []
        
        for engine_name, engine_health in engine_status.get('engine_health', {}).items():
            if engine_health.get('status') != 'healthy':
                unhealthy_engines.append(engine_name)
        
        if unhealthy_engines:
            health['issues'].append(f"Unhealthy engines: {', '.join(unhealthy_engines)}")
        
        # Check predictions directory access
        try:
            predictions_dir = Path("predictions")
            if not predictions_dir.exists():
                predictions_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            health['healthy'] = False
            health['issues'].append(f"Cannot access predictions directory: {e}")
        
        return health

    # Legacy Prediction Path Methods for Integration
    def get_legacy_predictions_path(self, game_type: str, model_type: str = None) -> Path:
        """Get path for legacy predictions: predictions/{game_type}/{model_type}/ or predictions/{game_type}/hybrid/"""
        self.validate_initialized()
        game_key = sanitize_game_name(game_type)
        
        if model_type == 'hybrid' or model_type is None:
            predictions_path = Path("predictions") / game_key / "hybrid"
        else:
            predictions_path = Path("predictions") / game_key / model_type
            
        predictions_path.mkdir(parents=True, exist_ok=True)
        return predictions_path
    
    def load_legacy_predictions(self, game_type: str, model_type: str = None, date: str = None) -> List[Dict[str, Any]]:
        """Load predictions from legacy path structure."""
        predictions_path = self.get_legacy_predictions_path(game_type, model_type)
        
        if not predictions_path.exists():
            return []
        
        predictions = []
        
        # Look for JSON prediction files
        if date:
            # Load specific date predictions
            pattern = f"*{date}*.json"
        else:
            # Load all prediction files
            pattern = "*.json"
        
        for pred_file in predictions_path.glob(pattern):
            try:
                with open(pred_file, 'r') as f:
                    prediction_data = json.load(f)
                    
                # Add metadata from filename
                prediction_data['file_name'] = pred_file.name
                prediction_data['file_path'] = str(pred_file)
                prediction_data['model_type'] = model_type or 'hybrid'
                prediction_data['game_type'] = game_type
                
                predictions.append(prediction_data)
                
            except Exception as e:
                logger.warning(f"Failed to load prediction file {pred_file}: {e}")
                continue
        
        # Sort by creation time (newest first)
        predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return predictions
    
    def save_legacy_prediction(self, game_type: str, model_type: str, prediction_data: Dict[str, Any]) -> bool:
        """Save prediction to legacy path structure."""
        predictions_path = self.get_legacy_predictions_path(game_type, model_type)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        
        if model_type == 'hybrid':
            # Include all model types in hybrid filename
            models_used = prediction_data.get('models_used', ['lstm', 'transformer', 'xgboost'])
            models_str = '_'.join(models_used)
            filename = f"{timestamp}_hybrid_{models_str}.json"
        else:
            filename = f"{timestamp}_{model_type}_prediction.json"
        
        file_path = predictions_path / filename
        
        # Add metadata to prediction
        prediction_data.update({
            'timestamp': datetime.now().isoformat(),
            'game_type': game_type,
            'model_type': model_type,
            'file_path': str(file_path)
        })
        
        try:
            with open(file_path, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            
            logger.info(f"Saved prediction to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save prediction to {file_path}: {e}")
            return False
    
    def get_available_prediction_dates(self, game_type: str, model_type: str = None) -> List[str]:
        """Get available prediction dates from legacy path structure."""
        predictions_path = self.get_legacy_predictions_path(game_type, model_type)
        
        if not predictions_path.exists():
            return []
        
        dates = set()
        for pred_file in predictions_path.glob("*.json"):
            # Extract date from filename like "20250924_hybrid_lstm_transformer_xgboost.json"
            parts = pred_file.stem.split('_')
            if parts and len(parts[0]) == 8 and parts[0].isdigit():
                dates.add(parts[0])
        
        return sorted(list(dates), reverse=True)
    
    def get_prediction_summary(self, game_type: str) -> Dict[str, Any]:
        """Get summary of all predictions for a game type."""
        game_key = sanitize_game_name(game_type)
        game_predictions_dir = Path("predictions") / game_key
        
        if not game_predictions_dir.exists():
            return {
                'game_type': game_type,
                'model_types': {},
                'total_predictions': 0,
                'latest_date': None
            }
        
        summary = {
            'game_type': game_type,
            'model_types': {},
            'total_predictions': 0,
            'latest_date': None
        }
        
        latest_date = None
        
        # Check each model type directory
        for model_dir in game_predictions_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_type = model_dir.name
            predictions = self.load_legacy_predictions(game_type, model_type)
            dates = self.get_available_prediction_dates(game_type, model_type)
            
            summary['model_types'][model_type] = {
                'count': len(predictions),
                'dates': dates,
                'latest_date': dates[0] if dates else None
            }
            
            summary['total_predictions'] += len(predictions)
            
            # Track overall latest date
            if dates and (not latest_date or dates[0] > latest_date):
                latest_date = dates[0]
        
        summary['latest_date'] = latest_date
        return summary
    
    def get_latest_hybrid_prediction(self, game_type: str) -> Optional[Dict[str, Any]]:
        """Get the most recent hybrid prediction for a game type."""
        hybrid_predictions = self.load_legacy_predictions(game_type, 'hybrid')
        if hybrid_predictions:
            return hybrid_predictions[0]  # Already sorted by timestamp
        return None