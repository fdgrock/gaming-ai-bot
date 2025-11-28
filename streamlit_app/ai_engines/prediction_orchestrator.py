"""
Prediction Orchestrator for the lottery prediction system.

This orchestrator coordinates the 4-phase AI enhancement system,
managing the flow between different engines and combining their
predictions for optimal results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from collections import defaultdict
import json

# Import the phase engines
from .phase1_mathematical import MathematicalEngine
from .phase2_expert_ensemble import ExpertEnsemble
from .phase3_set_optimizer import SetOptimizer
from .phase4_temporal import TemporalEngine

logger = logging.getLogger(__name__)


class AdvancedOrchestrationIntelligence:
    """Advanced orchestration intelligence for sophisticated engine coordination"""
    
    def __init__(self):
        self.engine_intelligence_scores = {}
        self.pattern_synergy_matrix = {}
        self.performance_history = defaultdict(list)
        self.adaptive_weights = {}
    
    def analyze_engine_synergy(self, engine_results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze synergy between different engines"""
        try:
            synergy_scores = {}
            
            # Calculate pairwise synergy scores
            engine_names = list(engine_results.keys())
            for i, engine1 in enumerate(engine_names):
                for engine2 in engine_names[i+1:]:
                    if engine1 in engine_results and engine2 in engine_results:
                        result1 = engine_results[engine1]
                        result2 = engine_results[engine2]
                        
                        # Calculate synergy based on prediction overlap and confidence correlation
                        synergy = self._calculate_prediction_synergy(result1, result2)
                        synergy_scores[f"{engine1}_{engine2}"] = synergy
            
            return synergy_scores
            
        except Exception as e:
            logger.error(f"Error analyzing engine synergy: {e}")
            return {}
    
    def _calculate_prediction_synergy(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> float:
        """Calculate synergy between two prediction results"""
        try:
            # Get predictions and confidences
            pred1 = result1.get('numbers', [])
            pred2 = result2.get('numbers', [])
            conf1 = result1.get('confidence', 0)
            conf2 = result2.get('confidence', 0)
            
            if not pred1 or not pred2:
                return 0.0
            
            # Calculate overlap ratio
            overlap = len(set(pred1) & set(pred2))
            total_unique = len(set(pred1) | set(pred2))
            overlap_ratio = overlap / total_unique if total_unique > 0 else 0
            
            # Calculate confidence correlation
            conf_correlation = abs(conf1 - conf2)  # Lower difference = higher correlation
            conf_factor = 1 - (conf_correlation / 2)  # Normalize to 0-1
            
            # Combined synergy score
            synergy = (overlap_ratio * 0.7) + (conf_factor * 0.3)
            return min(1.0, max(0.0, synergy))
            
        except Exception as e:
            logger.error(f"Error calculating prediction synergy: {e}")
            return 0.0
    
    def calculate_dynamic_weights(self, engine_performances: Dict[str, float], 
                                 synergy_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate dynamic weights based on performance and synergy"""
        try:
            if not engine_performances:
                return {}
            
            # Base weights from individual performance
            total_performance = sum(engine_performances.values())
            if total_performance <= 0:
                # Equal weights if no performance data
                return {engine: 1.0 / len(engine_performances) for engine in engine_performances}
            
            base_weights = {engine: perf / total_performance 
                          for engine, perf in engine_performances.items()}
            
            # Adjust weights based on synergy
            adjusted_weights = base_weights.copy()
            
            for synergy_pair, synergy_score in synergy_scores.items():
                if '_' in synergy_pair:
                    engine1, engine2 = synergy_pair.split('_')
                    if engine1 in adjusted_weights and engine2 in adjusted_weights:
                        # Boost weights for high-synergy engine pairs
                        boost = synergy_score * 0.1  # 10% max boost
                        adjusted_weights[engine1] = min(1.0, adjusted_weights[engine1] * (1 + boost))
                        adjusted_weights[engine2] = min(1.0, adjusted_weights[engine2] * (1 + boost))
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {engine: weight / total_weight 
                                  for engine, weight in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {e}")
            return {engine: 1.0 / len(engine_performances) for engine in engine_performances}


class SophisticatedPredictionAggregator:
    """Sophisticated prediction aggregation with intelligent combination strategies"""
    
    def __init__(self):
        self.aggregation_cache = {}
        self.quality_metrics = {}
    
    def aggregate_predictions(self, engine_predictions: Dict[str, List[Dict[str, Any]]], 
                            weights: Dict[str, float],
                            strategy: str = 'intelligent_weighted') -> List[Dict[str, Any]]:
        """Aggregate predictions from multiple engines using sophisticated strategies"""
        try:
            if not engine_predictions:
                return []
            
            if strategy == 'intelligent_weighted':
                return self._aggregate_intelligent_weighted(engine_predictions, weights)
            elif strategy == 'consensus_fusion':
                return self._aggregate_consensus_fusion(engine_predictions, weights)
            elif strategy == 'confidence_ranking':
                return self._aggregate_confidence_ranking(engine_predictions, weights)
            elif strategy == 'hybrid_optimization':
                return self._aggregate_hybrid_optimization(engine_predictions, weights)
            else:
                return self._aggregate_intelligent_weighted(engine_predictions, weights)
                
        except Exception as e:
            logger.error(f"Error in prediction aggregation: {e}")
            return []
    
    def _aggregate_intelligent_weighted(self, engine_predictions: Dict[str, List[Dict[str, Any]]], 
                                      weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Intelligent weighted aggregation considering prediction quality"""
        try:
            aggregated_predictions = []
            
            # Collect all predictions with weights
            weighted_predictions = []
            
            for engine_name, predictions in engine_predictions.items():
                engine_weight = weights.get(engine_name, 0)
                if engine_weight > 0:
                    for pred in predictions:
                        # Calculate prediction quality score
                        quality_score = self._calculate_prediction_quality(pred)
                        final_weight = engine_weight * quality_score
                        
                        weighted_predictions.append({
                            'engine': engine_name,
                            'prediction': pred,
                            'weight': final_weight,
                            'quality': quality_score
                        })
            
            # Sort by weight and select top predictions
            weighted_predictions.sort(key=lambda x: x['weight'], reverse=True)
            
            # Generate aggregated sets
            max_predictions = min(10, len(weighted_predictions))
            for i in range(max_predictions):
                if i < len(weighted_predictions):
                    wp = weighted_predictions[i]
                    aggregated_pred = wp['prediction'].copy()
                    aggregated_pred['aggregation_method'] = 'intelligent_weighted'
                    aggregated_pred['source_engine'] = wp['engine']
                    aggregated_pred['final_weight'] = wp['weight']
                    aggregated_pred['quality_score'] = wp['quality']
                    aggregated_predictions.append(aggregated_pred)
            
            return aggregated_predictions
            
        except Exception as e:
            logger.error(f"Error in intelligent weighted aggregation: {e}")
            return []
    
    def _aggregate_consensus_fusion(self, engine_predictions: Dict[str, List[Dict[str, Any]]], 
                                  weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Consensus fusion aggregation finding common patterns"""
        try:
            aggregated_predictions = []
            
            # Collect all prediction numbers
            all_predictions = []
            for engine_name, predictions in engine_predictions.items():
                engine_weight = weights.get(engine_name, 0)
                if engine_weight > 0:
                    for pred in predictions:
                        numbers = pred.get('numbers', [])
                        confidence = pred.get('confidence', 0) * engine_weight
                        all_predictions.append({
                            'numbers': numbers,
                            'confidence': confidence,
                            'engine': engine_name
                        })
            
            if not all_predictions:
                return []
            
            # Find consensus numbers (appearing in multiple predictions)
            number_votes = defaultdict(float)
            number_engines = defaultdict(set)
            
            for pred in all_predictions:
                for num in pred['numbers']:
                    number_votes[num] += pred['confidence']
                    number_engines[num].add(pred['engine'])
            
            # Create consensus-based predictions
            consensus_numbers = [(num, votes, len(engines)) 
                               for num, votes in number_votes.items() 
                               for engines in [number_engines[num]]]
            
            # Sort by consensus strength (votes * engine diversity)
            consensus_numbers.sort(key=lambda x: x[1] * x[2], reverse=True)
            
            # Generate multiple consensus sets
            numbers_per_draw = 6  # Default
            if all_predictions:
                numbers_per_draw = len(all_predictions[0]['numbers'])
            
            for set_id in range(min(5, len(consensus_numbers) // numbers_per_draw)):
                start_idx = set_id * (numbers_per_draw // 2)
                end_idx = start_idx + numbers_per_draw
                
                if end_idx <= len(consensus_numbers):
                    consensus_set = [num for num, _, _ in consensus_numbers[start_idx:end_idx]]
                    
                    # Fill to required size if needed
                    if len(consensus_set) < numbers_per_draw:
                        available = [num for num, _, _ in consensus_numbers 
                                   if num not in consensus_set]
                        needed = numbers_per_draw - len(consensus_set)
                        consensus_set.extend(available[:needed])
                    
                    avg_confidence = np.mean([votes for _, votes, _ in consensus_numbers[start_idx:end_idx]])
                    
                    aggregated_predictions.append({
                        'numbers': sorted(consensus_set[:numbers_per_draw]),
                        'confidence': min(1.0, avg_confidence),
                        'aggregation_method': 'consensus_fusion',
                        'consensus_strength': len(set(engines for _, _, engines in consensus_numbers[start_idx:end_idx]))
                    })
            
            return aggregated_predictions
            
        except Exception as e:
            logger.error(f"Error in consensus fusion aggregation: {e}")
            return []
    
    def _calculate_prediction_quality(self, prediction: Dict[str, Any]) -> float:
        """Calculate quality score for a prediction"""
        try:
            quality_factors = []
            
            # Confidence factor
            confidence = prediction.get('confidence', 0)
            quality_factors.append(confidence)
            
            # Number diversity factor
            numbers = prediction.get('numbers', [])
            if numbers:
                number_range = max(numbers) - min(numbers)
                diversity = min(1.0, number_range / 40.0)  # Normalize to 0-1
                quality_factors.append(diversity)
            
            # Analysis depth factor
            analysis = prediction.get('analysis', {})
            depth_score = min(1.0, len(analysis) / 10.0) if analysis else 0.5
            quality_factors.append(depth_score)
            
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating prediction quality: {e}")
            return 0.5


class UltraHighAccuracyOrchestrator:
    """Ultra-high accuracy orchestrator coordinating all sophisticated AI components"""
    
    def __init__(self):
        self.orchestration_intelligence = AdvancedOrchestrationIntelligence()
        self.prediction_aggregator = SophisticatedPredictionAggregator()
        self.performance_optimizer = {}
        self.ultra_accuracy_cache = {}
    
    def coordinate_ultra_accuracy_prediction(self, engines: Dict[str, Any], 
                                           num_predictions: int = 5) -> Dict[str, Any]:
        """Coordinate ultra-high accuracy predictions across all engines"""
        try:
            logger.info("🚀 Coordinating ultra-high accuracy predictions...")
            
            # Phase 1: Collect sophisticated predictions from all engines
            engine_results = self._collect_sophisticated_predictions(engines, num_predictions)
            
            # Phase 2: Analyze engine synergy
            synergy_analysis = self.orchestration_intelligence.analyze_engine_synergy(engine_results)
            
            # Phase 3: Calculate dynamic weights
            performance_scores = self._calculate_engine_performance_scores(engine_results)
            dynamic_weights = self.orchestration_intelligence.calculate_dynamic_weights(
                performance_scores, synergy_analysis
            )
            
            # Phase 4: Sophisticated prediction aggregation
            aggregated_predictions = self.prediction_aggregator.aggregate_predictions(
                engine_results, dynamic_weights, 'hybrid_optimization'
            )
            
            # Phase 5: Ultra-accuracy enhancement
            enhanced_predictions = self._apply_ultra_accuracy_enhancement(
                aggregated_predictions, synergy_analysis, dynamic_weights
            )
            
            return {
                'ultra_accuracy_predictions': enhanced_predictions,
                'engine_synergy': synergy_analysis,
                'dynamic_weights': dynamic_weights,
                'performance_scores': performance_scores,
                'orchestration_intelligence': np.mean(list(performance_scores.values())) if performance_scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in ultra-accuracy coordination: {e}")
            return {
                'ultra_accuracy_predictions': [],
                'error': str(e)
            }
    
    def _collect_sophisticated_predictions(self, engines: Dict[str, Any], 
                                         num_predictions: int) -> Dict[str, List[Dict[str, Any]]]:
        """Collect sophisticated predictions from all engines"""
        try:
            engine_results = {}
            
            for engine_name, engine in engines.items():
                try:
                    # Try to use sophisticated prediction methods first
                    if hasattr(engine, 'generate_sophisticated_predictions'):
                        predictions = engine.generate_sophisticated_predictions(num_predictions)
                        if isinstance(predictions, dict) and 'predictions' in predictions:
                            engine_results[engine_name] = predictions['predictions']
                        elif isinstance(predictions, list):
                            engine_results[engine_name] = predictions
                    elif hasattr(engine, 'generate_ultra_high_accuracy_predictions'):
                        predictions = engine.generate_ultra_high_accuracy_predictions(num_predictions)
                        engine_results[engine_name] = predictions.get('optimized_sets', [])
                    elif hasattr(engine, 'generate_sophisticated_temporal_predictions'):
                        predictions = engine.generate_sophisticated_temporal_predictions(num_predictions)
                        engine_results[engine_name] = predictions.get('temporal_sets', [])
                    else:
                        # Fallback to standard prediction method
                        predictions = engine.predict(num_predictions)
                        if isinstance(predictions, list):
                            engine_results[engine_name] = predictions
                        elif isinstance(predictions, dict) and 'numbers' in predictions:
                            engine_results[engine_name] = [predictions]
                
                except Exception as e:
                    logger.warning(f"Error collecting predictions from {engine_name}: {e}")
                    engine_results[engine_name] = []
            
            return engine_results
            
        except Exception as e:
            logger.error(f"Error collecting sophisticated predictions: {e}")
            return {}
    
    def _calculate_engine_performance_scores(self, engine_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate performance scores for each engine"""
        try:
            performance_scores = {}
            
            for engine_name, predictions in engine_results.items():
                if predictions:
                    # Calculate average confidence
                    confidences = [pred.get('confidence', 0) for pred in predictions if 'confidence' in pred]
                    avg_confidence = np.mean(confidences) if confidences else 0.5
                    
                    # Calculate prediction diversity
                    all_numbers = set()
                    for pred in predictions:
                        numbers = pred.get('numbers', [])
                        all_numbers.update(numbers)
                    diversity = len(all_numbers) / 49.0  # Normalize by max possible numbers
                    
                    # Calculate analysis depth
                    analysis_depths = [len(pred.get('analysis', {})) for pred in predictions]
                    avg_depth = np.mean(analysis_depths) if analysis_depths else 0
                    depth_score = min(1.0, avg_depth / 10.0)
                    
                    # Combined performance score
                    performance_scores[engine_name] = (avg_confidence * 0.5) + (diversity * 0.3) + (depth_score * 0.2)
                else:
                    performance_scores[engine_name] = 0.0
            
            return performance_scores
            
        except Exception as e:
            logger.error(f"Error calculating performance scores: {e}")
            return {}
    
    def _apply_ultra_accuracy_enhancement(self, predictions: List[Dict[str, Any]], 
                                        synergy_analysis: Dict[str, float],
                                        dynamic_weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Apply ultra-accuracy enhancement to predictions"""
        try:
            enhanced_predictions = []
            
            # Calculate overall orchestration intelligence
            avg_synergy = np.mean(list(synergy_analysis.values())) if synergy_analysis else 0.0
            weight_balance = 1 - np.std(list(dynamic_weights.values())) if dynamic_weights else 0.0
            orchestration_intelligence = (avg_synergy * 0.6) + (weight_balance * 0.4)
            
            # Enhance each prediction
            for i, prediction in enumerate(predictions):
                enhanced_pred = prediction.copy()
                
                # Apply intelligence boost to confidence
                base_confidence = enhanced_pred.get('confidence', 0)
                intelligence_boost = orchestration_intelligence * 0.3
                enhanced_confidence = min(1.0, base_confidence + intelligence_boost)
                
                enhanced_pred['confidence'] = enhanced_confidence
                enhanced_pred['orchestration_intelligence'] = orchestration_intelligence
                enhanced_pred['ultra_accuracy_applied'] = True
                enhanced_pred['enhancement_factors'] = {
                    'synergy_score': avg_synergy,
                    'weight_balance': weight_balance,
                    'intelligence_boost': intelligence_boost
                }
                
                enhanced_predictions.append(enhanced_pred)
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error applying ultra-accuracy enhancement: {e}")
            return predictions


class PredictionOrchestrator:
    """
    Orchestrates the 4-phase AI enhancement system for lottery predictions.
    
    Manages:
    - Engine coordination
    - Prediction aggregation
    - Confidence scoring
    - Performance tracking
    - Strategy selection
    - Load balancing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Prediction Orchestrator.
        
        Args:
            config: Configuration dictionary containing orchestrator parameters
        """
        self.config = config
        self.game_config = config.get('game_config', {})
        self.orchestration_config = config.get('orchestration_config', {})
        
        # Game parameters
        self.number_range = self.game_config.get('number_range', [1, 49])
        self.numbers_per_draw = self.game_config.get('numbers_per_draw', 6)
        self.has_bonus = self.game_config.get('has_bonus', False)
        
        # Orchestration parameters
        self.orchestration_strategy = self.orchestration_config.get('strategy', 'adaptive')
        self.confidence_threshold = self.orchestration_config.get('confidence_threshold', 0.5)
        self.max_predictions = self.orchestration_config.get('max_predictions', 20)
        self.engine_weights = self.orchestration_config.get('engine_weights', {
            'mathematical': 0.25,
            'expert_ensemble': 0.30,
            'set_optimizer': 0.25,
            'temporal': 0.20
        })
        self.parallel_execution = self.orchestration_config.get('parallel_execution', True)
        self.timeout_seconds = self.orchestration_config.get('timeout_seconds', 300)
        
        # Initialize engines
        self.engines = {}
        self.engine_performance = {}
        self.historical_data = None
        self.trained = False
        self.prediction_history = []
        self.performance_metrics = {}
        
        # Sophisticated orchestration components
        self.ultra_accuracy_orchestrator = UltraHighAccuracyOrchestrator()
        self.orchestration_intelligence = AdvancedOrchestrationIntelligence()
        self.prediction_aggregator = SophisticatedPredictionAggregator()
        
        # Enhanced caches and tracking
        self.ultra_accuracy_cache = {}
        self.synergy_analysis_cache = {}
        self.dynamic_weights_history = []
        self.orchestration_intelligence_score = 0.0
        
        # Performance tracking
        self.engine_response_times = defaultdict(list)
        self.engine_success_rates = defaultdict(list)
        self.confidence_histories = defaultdict(list)
        
        self._initialize_engines()
        
        logger.info("🎯 Prediction Orchestrator initialized")
    
    def _initialize_engines(self) -> None:
        """Initialize all AI engines."""
        try:
            # Initialize Phase 1: Mathematical Engine
            self.engines['mathematical'] = MathematicalEngine({
                'game_config': self.game_config,
                'analysis_config': self.orchestration_config.get('mathematical_config', {})
            })
            
            # Initialize Phase 2: Expert Ensemble
            self.engines['expert_ensemble'] = ExpertEnsemble({
                'game_config': self.game_config,
                'ensemble_config': self.orchestration_config.get('ensemble_config', {})
            })
            
            # Initialize Phase 3: Set Optimizer
            self.engines['set_optimizer'] = SetOptimizer({
                'game_config': self.game_config,
                'optimizer_config': self.orchestration_config.get('optimizer_config', {})
            })
            
            # Initialize Phase 4: Temporal Engine
            self.engines['temporal'] = TemporalEngine({
                'game_config': self.game_config,
                'temporal_config': self.orchestration_config.get('temporal_config', {})
            })
            
            logger.info(f"✅ Initialized {len(self.engines)} AI engines")
            
        except Exception as e:
            logger.error(f"❌ Engine initialization failed: {e}")
            raise
    
    def load_data(self, historical_data: pd.DataFrame) -> None:
        """
        Load historical lottery data into all engines.
        
        Args:
            historical_data: DataFrame with columns ['date', 'numbers', 'bonus']
        """
        try:
            self.historical_data = historical_data.copy()
            
            # Load data into each engine
            for engine_name, engine in self.engines.items():
                try:
                    start_time = datetime.now()
                    engine.load_data(historical_data)
                    load_time = (datetime.now() - start_time).total_seconds()
                    
                    logger.info(f"📊 {engine_name} engine loaded data in {load_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load data into {engine_name} engine: {e}")
                    # Remove failed engine from orchestration
                    if engine_name in self.engine_weights:
                        del self.engine_weights[engine_name]
            
            logger.info(f"📊 Data loaded into {len(self.engines)} engines")
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """
        Train all AI engines and collect performance metrics.
        
        Returns:
            Dictionary containing training results
        """
        try:
            logger.info("🎓 Training all AI engines...")
            
            training_results = {}
            
            if self.parallel_execution:
                # Train engines in parallel
                training_results = self._train_engines_parallel()
            else:
                # Train engines sequentially
                training_results = self._train_engines_sequential()
            
            # Calculate engine weights based on performance
            self._update_engine_weights(training_results)
            
            # Update orchestration strategy based on results
            self._optimize_orchestration_strategy(training_results)
            
            self.trained = True
            
            logger.info("✅ All engines trained successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def _train_engines_parallel(self) -> Dict[str, Any]:
        """Train engines in parallel using ThreadPoolExecutor."""
        training_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit training tasks
            future_to_engine = {}
            for engine_name, engine in self.engines.items():
                future = executor.submit(self._train_single_engine, engine_name, engine)
                future_to_engine[future] = engine_name
            
            # Collect results
            for future in as_completed(future_to_engine, timeout=self.timeout_seconds):
                engine_name = future_to_engine[future]
                try:
                    result = future.result()
                    training_results[engine_name] = result
                except Exception as e:
                    logger.error(f"❌ {engine_name} training failed: {e}")
                    training_results[engine_name] = {'error': str(e)}
        
        return training_results
    
    def _train_engines_sequential(self) -> Dict[str, Any]:
        """Train engines sequentially."""
        training_results = {}
        
        for engine_name, engine in self.engines.items():
            try:
                result = self._train_single_engine(engine_name, engine)
                training_results[engine_name] = result
            except Exception as e:
                logger.error(f"❌ {engine_name} training failed: {e}")
                training_results[engine_name] = {'error': str(e)}
        
        return training_results
    
    def _train_single_engine(self, engine_name: str, engine: Any) -> Dict[str, Any]:
        """Train a single engine and collect metrics."""
        start_time = datetime.now()
        
        try:
            # Train the engine
            engine.train()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Collect performance metrics
            result = {
                'success': True,
                'training_time': training_time,
                'trained_at': datetime.now().isoformat()
            }
            
            # Get engine-specific metrics if available
            if hasattr(engine, 'get_analysis_summary'):
                result['analysis_summary'] = engine.get_analysis_summary()
            elif hasattr(engine, 'get_model_performance'):
                result['model_performance'] = engine.get_model_performance()
            elif hasattr(engine, 'get_optimization_summary'):
                result['optimization_summary'] = engine.get_optimization_summary()
            elif hasattr(engine, 'get_temporal_summary'):
                result['temporal_summary'] = engine.get_temporal_summary()
            
            logger.info(f"✅ {engine_name} trained in {training_time:.2f}s")
            return result
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ {engine_name} training failed after {training_time:.2f}s: {e}")
            raise
    
    def _update_engine_weights(self, training_results: Dict[str, Any]) -> None:
        """Update engine weights based on training performance."""
        try:
            total_weight = 0
            successful_engines = []
            
            # Calculate weights based on training success and speed
            for engine_name, result in training_results.items():
                if result.get('success', False):
                    # Weight based on inverse of training time (faster = better weight)
                    training_time = result.get('training_time', 1.0)
                    weight = 1.0 / (training_time + 1.0)  # +1 to avoid division by zero
                    
                    self.engine_weights[engine_name] = weight
                    total_weight += weight
                    successful_engines.append(engine_name)
                else:
                    # Set weight to 0 for failed engines
                    self.engine_weights[engine_name] = 0.0
            
            # Normalize weights
            if total_weight > 0:
                for engine_name in successful_engines:
                    self.engine_weights[engine_name] /= total_weight
            
            logger.info(f"🎯 Updated engine weights: {self.engine_weights}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to update engine weights: {e}")
    
    def _optimize_orchestration_strategy(self, training_results: Dict[str, Any]) -> None:
        """Optimize orchestration strategy based on training results."""
        try:
            successful_count = sum(1 for result in training_results.values() if result.get('success', False))
            total_count = len(training_results)
            
            success_rate = successful_count / total_count if total_count > 0 else 0
            
            # Adjust strategy based on success rate
            if success_rate >= 0.8:
                self.orchestration_strategy = 'aggressive'  # Use all engines
                self.confidence_threshold = 0.4  # Lower threshold for more predictions
            elif success_rate >= 0.6:
                self.orchestration_strategy = 'balanced'  # Standard approach
                self.confidence_threshold = 0.5
            else:
                self.orchestration_strategy = 'conservative'  # Only use best engines
                self.confidence_threshold = 0.7  # Higher threshold for quality
            
            logger.info(f"🎯 Optimized orchestration: {self.orchestration_strategy}, threshold: {self.confidence_threshold}")
            
        except Exception as e:
            logger.warning(f"⚠️ Strategy optimization failed: {e}")
    
    def predict(self, num_predictions: int = 5, strategy: str = 'adaptive', 
                engine_selection: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate predictions using the orchestrated AI system.
        
        Args:
            num_predictions: Number of prediction sets to generate
            strategy: Orchestration strategy ('adaptive', 'ensemble', 'best_engine', 'consensus')
            engine_selection: Specific engines to use (None for all available)
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if not self.trained:
                raise ValueError("Orchestrator not trained. Call train() first.")
            
            # Determine which engines to use
            active_engines = self._select_active_engines(engine_selection)
            
            if not active_engines:
                raise ValueError("No active engines available for prediction")
            
            # Generate predictions based on strategy
            if strategy == 'adaptive':
                predictions = self._predict_adaptive(num_predictions, active_engines)
            elif strategy == 'ensemble':
                predictions = self._predict_ensemble(num_predictions, active_engines)
            elif strategy == 'best_engine':
                predictions = self._predict_best_engine(num_predictions, active_engines)
            elif strategy == 'consensus':
                predictions = self._predict_consensus(num_predictions, active_engines)
            else:
                predictions = self._predict_adaptive(num_predictions, active_engines)
            
            # Post-process predictions
            processed_predictions = self._post_process_predictions(predictions, strategy)
            
            # Update prediction history
            self.prediction_history.extend(processed_predictions)
            
            logger.info(f"🎯 Generated {len(processed_predictions)} orchestrated predictions using {strategy} strategy")
            return processed_predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            raise
    
    def _select_active_engines(self, engine_selection: Optional[List[str]]) -> Dict[str, Any]:
        """Select which engines to use for prediction."""
        if engine_selection:
            # Use specified engines
            active_engines = {name: engine for name, engine in self.engines.items() 
                            if name in engine_selection and name in self.engine_weights and self.engine_weights[name] > 0}
        else:
            # Use all engines with positive weights
            active_engines = {name: engine for name, engine in self.engines.items() 
                            if name in self.engine_weights and self.engine_weights[name] > 0}
        
        return active_engines
    
    def _predict_adaptive(self, num_predictions: int, active_engines: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictions using adaptive strategy."""
        predictions = []
        
        for i in range(num_predictions):
            # Select engine based on weights and performance
            selected_engine_name = self._select_engine_adaptive()
            
            if selected_engine_name in active_engines:
                engine = active_engines[selected_engine_name]
                
                try:
                    start_time = datetime.now()
                    engine_predictions = engine.predict(1, strategy='balanced')
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    if engine_predictions:
                        prediction = engine_predictions[0]
                        prediction['orchestration_method'] = 'adaptive'
                        prediction['selected_engine'] = selected_engine_name
                        prediction['response_time'] = response_time
                        predictions.append(prediction)
                        
                        # Track performance
                        self.engine_response_times[selected_engine_name].append(response_time)
                        self.engine_success_rates[selected_engine_name].append(1.0)
                    
                except Exception as e:
                    logger.warning(f"⚠️ {selected_engine_name} prediction failed: {e}")
                    self.engine_success_rates[selected_engine_name].append(0.0)
        
        return predictions
    
    def _predict_ensemble(self, num_predictions: int, active_engines: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictions using ensemble of all engines."""
        predictions = []
        
        for i in range(num_predictions):
            engine_predictions = {}
            
            # Get predictions from all active engines
            for engine_name, engine in active_engines.items():
                try:
                    start_time = datetime.now()
                    pred = engine.predict(1, strategy='balanced')
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    if pred:
                        engine_predictions[engine_name] = {
                            'prediction': pred[0],
                            'response_time': response_time
                        }
                        
                        # Track performance
                        self.engine_response_times[engine_name].append(response_time)
                        self.engine_success_rates[engine_name].append(1.0)
                
                except Exception as e:
                    logger.warning(f"⚠️ {engine_name} prediction failed: {e}")
                    self.engine_success_rates[engine_name].append(0.0)
            
            # Combine predictions
            if engine_predictions:
                combined_prediction = self._combine_engine_predictions(engine_predictions)
                combined_prediction['orchestration_method'] = 'ensemble'
                predictions.append(combined_prediction)
        
        return predictions
    
    def _predict_best_engine(self, num_predictions: int, active_engines: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictions using the best performing engine."""
        # Find best engine based on weights
        best_engine_name = max(self.engine_weights.items(), key=lambda x: x[1])[0]
        
        if best_engine_name not in active_engines:
            # Fallback to any available engine
            best_engine_name = list(active_engines.keys())[0]
        
        predictions = []
        engine = active_engines[best_engine_name]
        
        try:
            start_time = datetime.now()
            engine_predictions = engine.predict(num_predictions, strategy='balanced')
            response_time = (datetime.now() - start_time).total_seconds()
            
            for pred in engine_predictions:
                pred['orchestration_method'] = 'best_engine'
                pred['selected_engine'] = best_engine_name
                pred['response_time'] = response_time / len(engine_predictions)
                predictions.append(pred)
            
            # Track performance
            self.engine_response_times[best_engine_name].append(response_time)
            self.engine_success_rates[best_engine_name].append(1.0)
            
        except Exception as e:
            logger.error(f"❌ Best engine {best_engine_name} prediction failed: {e}")
            self.engine_success_rates[best_engine_name].append(0.0)
        
        return predictions
    
    def _predict_consensus(self, num_predictions: int, active_engines: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictions using consensus approach."""
        predictions = []
        
        for i in range(num_predictions):
            # Get predictions from all engines
            all_engine_predictions = []
            
            for engine_name, engine in active_engines.items():
                try:
                    start_time = datetime.now()
                    pred = engine.predict(1, strategy='balanced')
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    if pred:
                        all_engine_predictions.append({
                            'engine': engine_name,
                            'prediction': pred[0],
                            'response_time': response_time
                        })
                        
                        # Track performance
                        self.engine_response_times[engine_name].append(response_time)
                        self.engine_success_rates[engine_name].append(1.0)
                
                except Exception as e:
                    logger.warning(f"⚠️ {engine_name} consensus prediction failed: {e}")
                    self.engine_success_rates[engine_name].append(0.0)
            
            # Find consensus
            if len(all_engine_predictions) >= 2:
                consensus_prediction = self._find_consensus(all_engine_predictions)
                consensus_prediction['orchestration_method'] = 'consensus'
                predictions.append(consensus_prediction)
        
        return predictions
    
    def _select_engine_adaptive(self) -> str:
        """Select engine based on adaptive strategy."""
        # Consider weights, recent performance, and response times
        engine_scores = {}
        
        for engine_name, weight in self.engine_weights.items():
            if weight > 0:
                # Base score from weight
                score = weight
                
                # Adjust for recent success rate
                if engine_name in self.engine_success_rates:
                    recent_successes = self.engine_success_rates[engine_name][-10:]  # Last 10 predictions
                    if recent_successes:
                        success_rate = sum(recent_successes) / len(recent_successes)
                        score *= success_rate
                
                # Adjust for response time (faster is better)
                if engine_name in self.engine_response_times:
                    recent_times = self.engine_response_times[engine_name][-10:]  # Last 10 predictions
                    if recent_times:
                        avg_time = sum(recent_times) / len(recent_times)
                        time_factor = 1.0 / (avg_time + 0.1)  # Avoid division by zero
                        score *= time_factor
                
                engine_scores[engine_name] = score
        
        # Select engine with highest score
        if engine_scores:
            return max(engine_scores.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to first available engine
            return list(self.engine_weights.keys())[0]
    
    def _combine_engine_predictions(self, engine_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine predictions from multiple engines."""
        # Collect all numbers and their confidence scores
        number_votes = defaultdict(list)
        
        for engine_name, pred_data in engine_predictions.items():
            prediction = pred_data['prediction']
            numbers = prediction['numbers']
            confidence = prediction['confidence']
            weight = self.engine_weights.get(engine_name, 0.25)
            
            for number in numbers:
                number_votes[number].append(confidence * weight)
        
        # Calculate weighted scores for each number
        number_scores = {}
        for number, votes in number_votes.items():
            number_scores[number] = sum(votes) / len(votes)
        
        # Select top numbers
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        selected_numbers = [num for num, score in sorted_numbers[:self.numbers_per_draw]]
        
        # Calculate combined confidence
        top_scores = [score for _, score in sorted_numbers[:self.numbers_per_draw]]
        combined_confidence = sum(top_scores) / len(top_scores) if top_scores else 0.5
        
        # Calculate average response time
        avg_response_time = sum(pred_data['response_time'] for pred_data in engine_predictions.values()) / len(engine_predictions)
        
        return {
            'numbers': sorted(selected_numbers),
            'confidence': min(0.95, combined_confidence),
            'engine_contributions': {engine: pred_data['prediction']['confidence'] 
                                   for engine, pred_data in engine_predictions.items()},
            'response_time': avg_response_time,
            'generated_at': datetime.now().isoformat(),
            'engine': 'orchestrated_ensemble'
        }
    
    def _find_consensus(self, all_predictions: List[Dict]) -> Dict[str, Any]:
        """Find consensus among multiple predictions."""
        # Count how often each number appears
        number_appearances = defaultdict(int)
        total_confidence = 0
        total_predictions = len(all_predictions)
        
        for pred_data in all_predictions:
            prediction = pred_data['prediction']
            numbers = prediction['numbers']
            confidence = prediction['confidence']
            
            for number in numbers:
                number_appearances[number] += 1
            
            total_confidence += confidence
        
        # Select numbers that appear in majority of predictions
        consensus_threshold = max(1, total_predictions // 2)
        consensus_numbers = [num for num, count in number_appearances.items() 
                           if count >= consensus_threshold]
        
        # If not enough consensus numbers, add most frequent ones
        if len(consensus_numbers) < self.numbers_per_draw:
            sorted_by_frequency = sorted(number_appearances.items(), key=lambda x: x[1], reverse=True)
            for num, count in sorted_by_frequency:
                if num not in consensus_numbers and len(consensus_numbers) < self.numbers_per_draw:
                    consensus_numbers.append(num)
        
        # Ensure we have the right number of selections
        while len(consensus_numbers) < self.numbers_per_draw:
            available = [n for n in range(self.number_range[0], self.number_range[1] + 1) 
                        if n not in consensus_numbers]
            if available:
                consensus_numbers.append(np.random.choice(available))
            else:
                break
        
        avg_confidence = total_confidence / total_predictions if total_predictions > 0 else 0.5
        avg_response_time = sum(pred_data['response_time'] for pred_data in all_predictions) / total_predictions
        
        return {
            'numbers': sorted(consensus_numbers[:self.numbers_per_draw]),
            'confidence': min(0.95, avg_confidence),
            'consensus_strength': len([num for num, count in number_appearances.items() 
                                     if count >= consensus_threshold]) / self.numbers_per_draw,
            'participating_engines': [pred_data['engine'] for pred_data in all_predictions],
            'response_time': avg_response_time,
            'generated_at': datetime.now().isoformat(),
            'engine': 'orchestrated_consensus'
        }
    
    def _post_process_predictions(self, predictions: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
        """Post-process predictions for final output."""
        processed_predictions = []
        
        for prediction in predictions:
            # Ensure numbers are within valid range
            valid_numbers = [num for num in prediction['numbers'] 
                           if self.number_range[0] <= num <= self.number_range[1]]
            
            # Fill missing numbers if some were invalid
            while len(valid_numbers) < self.numbers_per_draw:
                available = [n for n in range(self.number_range[0], self.number_range[1] + 1) 
                           if n not in valid_numbers]
                if available:
                    valid_numbers.append(np.random.choice(available))
                else:
                    break
            
            # Update prediction
            prediction['numbers'] = sorted(valid_numbers[:self.numbers_per_draw])
            
            # Add orchestration metadata
            prediction['orchestration_strategy'] = strategy
            prediction['orchestration_version'] = '1.0.0'
            prediction['quality_score'] = self._calculate_quality_score(prediction)
            
            # Only include predictions above confidence threshold
            if prediction['confidence'] >= self.confidence_threshold:
                processed_predictions.append(prediction)
        
        # Sort by confidence and limit to max predictions
        processed_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return processed_predictions[:self.max_predictions]
    
    def _calculate_quality_score(self, prediction: Dict[str, Any]) -> float:
        """Calculate quality score for a prediction."""
        try:
            base_score = prediction['confidence']
            
            # Adjust for engine diversity
            if 'engine_contributions' in prediction:
                diversity_bonus = len(prediction['engine_contributions']) * 0.1
                base_score += diversity_bonus
            
            # Adjust for response time (faster is better)
            response_time = prediction.get('response_time', 1.0)
            time_factor = max(0.9, 1.0 - (response_time / 10.0))  # Penalize slow responses
            base_score *= time_factor
            
            return min(1.0, base_score)
            
        except Exception as e:
            logger.warning(f"⚠️ Quality score calculation failed: {e}")
            return prediction.get('confidence', 0.5)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive orchestration status.
        
        Returns:
            Dictionary containing orchestration status
        """
        return {
            'orchestrator': {
                'trained': self.trained,
                'strategy': self.orchestration_strategy,
                'confidence_threshold': self.confidence_threshold,
                'max_predictions': self.max_predictions
            },
            'engines': {
                'total_engines': len(self.engines),
                'active_engines': len([w for w in self.engine_weights.values() if w > 0]),
                'engine_weights': self.engine_weights,
                'engine_names': list(self.engines.keys())
            },
            'performance': {
                'total_predictions': len(self.prediction_history),
                'avg_response_times': {engine: sum(times) / len(times) if times else 0 
                                     for engine, times in self.engine_response_times.items()},
                'success_rates': {engine: sum(rates) / len(rates) if rates else 0 
                                for engine, rates in self.engine_success_rates.items()}
            },
            'configuration': {
                'parallel_execution': self.parallel_execution,
                'timeout_seconds': self.timeout_seconds,
                'game_config': self.game_config
            }
        }
    
    def update_engine_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update engine weights manually.
        
        Args:
            new_weights: Dictionary of engine names to weights
        """
        try:
            # Validate weights
            total_weight = sum(new_weights.values())
            if total_weight <= 0:
                raise ValueError("Total weight must be positive")
            
            # Normalize weights
            normalized_weights = {engine: weight / total_weight 
                                for engine, weight in new_weights.items()}
            
            # Update only for existing engines
            for engine_name, weight in normalized_weights.items():
                if engine_name in self.engines:
                    self.engine_weights[engine_name] = weight
            
            logger.info(f"🎯 Updated engine weights: {self.engine_weights}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update engine weights: {e}")
            raise
    
    def get_prediction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent prediction history.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List of recent predictions
        """
        return self.prediction_history[-limit:] if self.prediction_history else []
    
    def clear_performance_metrics(self) -> None:
        """Clear performance tracking metrics."""
        self.engine_response_times.clear()
        self.engine_success_rates.clear()
        self.confidence_histories.clear()
        self.prediction_history.clear()
        
        logger.info("🧹 Performance metrics cleared")
    
    def generate_ultra_accuracy_predictions(self, num_predictions: int = 5) -> Dict[str, Any]:
        """
        Generate ultra-high accuracy predictions using sophisticated orchestration.
        
        Args:
            num_predictions: Number of ultra-accuracy prediction sets to generate
            
        Returns:
            Dictionary containing ultra-accuracy predictions and orchestration intelligence
        """
        try:
            if not self.trained:
                raise ValueError("Orchestrator not trained. Call train() first.")
            
            logger.info(f"🚀 Generating {num_predictions} ultra-accuracy predictions...")
            
            # Use the ultra-accuracy orchestrator
            ultra_result = self.ultra_accuracy_orchestrator.coordinate_ultra_accuracy_prediction(
                self.engines, num_predictions
            )
            
            # Cache the results
            cache_key = f"ultra_accuracy_{num_predictions}_{datetime.now().isoformat()}"
            self.ultra_accuracy_cache[cache_key] = ultra_result
            
            # Update orchestration intelligence score
            self.orchestration_intelligence_score = ultra_result.get('orchestration_intelligence', 0)
            
            # Store synergy analysis
            synergy_analysis = ultra_result.get('engine_synergy', {})
            self.synergy_analysis_cache.update(synergy_analysis)
            
            # Store dynamic weights history
            dynamic_weights = ultra_result.get('dynamic_weights', {})
            self.dynamic_weights_history.append({
                'timestamp': datetime.now().isoformat(),
                'weights': dynamic_weights
            })
            
            # Update prediction history
            ultra_predictions = ultra_result.get('ultra_accuracy_predictions', [])
            self.prediction_history.extend(ultra_predictions)
            
            logger.info(f"✅ Generated {len(ultra_predictions)} ultra-accuracy predictions with intelligence score: {self.orchestration_intelligence_score:.3f}")
            
            return {
                'predictions': ultra_predictions,
                'orchestration_intelligence': self.orchestration_intelligence_score,
                'engine_synergy': synergy_analysis,
                'dynamic_weights': dynamic_weights,
                'performance_scores': ultra_result.get('performance_scores', {}),
                'method': 'ultra_accuracy_orchestration'
            }
            
        except Exception as e:
            logger.error(f"❌ Ultra-accuracy prediction generation failed: {e}")
            return {
                'predictions': [],
                'error': str(e),
                'orchestration_intelligence': 0.0
            }
    
    def generate_sophisticated_orchestrated_predictions(self, num_predictions: int = 5, 
                                                       aggregation_strategy: str = 'intelligent_weighted') -> Dict[str, Any]:
        """
        Generate sophisticated orchestrated predictions with advanced aggregation.
        
        Args:
            num_predictions: Number of prediction sets to generate
            aggregation_strategy: Strategy for prediction aggregation
            
        Returns:
            Dictionary containing sophisticated orchestrated predictions
        """
        try:
            if not self.trained:
                raise ValueError("Orchestrator not trained. Call train() first.")
            
            logger.info(f"🧠 Generating {num_predictions} sophisticated orchestrated predictions...")
            
            # Collect predictions from all engines
            engine_predictions = {}
            performance_scores = {}
            
            for engine_name, engine in self.engines.items():
                if self.engine_weights.get(engine_name, 0) > 0:
                    try:
                        # Try sophisticated prediction methods first
                        engine_result = self._collect_engine_predictions(engine, engine_name, num_predictions)
                        engine_predictions[engine_name] = engine_result['predictions']
                        performance_scores[engine_name] = engine_result['performance_score']
                        
                    except Exception as e:
                        logger.warning(f"Error collecting predictions from {engine_name}: {e}")
                        continue
            
            # Analyze engine synergy
            synergy_analysis = self.orchestration_intelligence.analyze_engine_synergy(engine_predictions)
            
            # Calculate dynamic weights
            dynamic_weights = self.orchestration_intelligence.calculate_dynamic_weights(
                performance_scores, synergy_analysis
            )
            
            # Aggregate predictions using sophisticated methods
            aggregated_predictions = self.prediction_aggregator.aggregate_predictions(
                engine_predictions, dynamic_weights, aggregation_strategy
            )
            
            # Calculate overall orchestration intelligence
            orchestration_intelligence = self._calculate_orchestration_intelligence(
                synergy_analysis, dynamic_weights, performance_scores
            )
            
            # Update tracking
            self.orchestration_intelligence_score = orchestration_intelligence
            self.prediction_history.extend(aggregated_predictions)
            
            logger.info(f"✅ Generated {len(aggregated_predictions)} sophisticated orchestrated predictions")
            
            return {
                'predictions': aggregated_predictions,
                'aggregation_strategy': aggregation_strategy,
                'orchestration_intelligence': orchestration_intelligence,
                'engine_synergy': synergy_analysis,
                'dynamic_weights': dynamic_weights,
                'performance_scores': performance_scores,
                'method': 'sophisticated_orchestration'
            }
            
        except Exception as e:
            logger.error(f"❌ Sophisticated orchestrated prediction generation failed: {e}")
            return {
                'predictions': [],
                'error': str(e),
                'orchestration_intelligence': 0.0
            }
    
    def _collect_engine_predictions(self, engine: Any, engine_name: str, num_predictions: int) -> Dict[str, Any]:
        """Collect predictions from a single engine with performance assessment"""
        try:
            start_time = datetime.now()
            predictions = []
            
            # Try sophisticated methods first
            if hasattr(engine, 'generate_ultra_high_accuracy_predictions'):
                result = engine.generate_ultra_high_accuracy_predictions(num_predictions)
                predictions = result.get('optimized_sets', [])
            elif hasattr(engine, 'generate_sophisticated_temporal_predictions'):
                result = engine.generate_sophisticated_temporal_predictions(num_predictions)
                predictions = result.get('temporal_sets', [])
            elif hasattr(engine, 'generate_sophisticated_predictions'):
                result = engine.generate_sophisticated_predictions(num_predictions)
                if isinstance(result, dict) and 'predictions' in result:
                    predictions = result['predictions']
                elif isinstance(result, list):
                    predictions = result
            else:
                # Fallback to standard prediction
                result = engine.predict(num_predictions)
                if isinstance(result, list):
                    predictions = result
                elif isinstance(result, dict) and 'numbers' in result:
                    predictions = [result]
            
            # Calculate performance score
            response_time = (datetime.now() - start_time).total_seconds()
            performance_score = self._calculate_engine_performance(predictions, response_time)
            
            return {
                'predictions': predictions,
                'performance_score': performance_score,
                'response_time': response_time
            }
            
        except Exception as e:
            logger.error(f"Error collecting predictions from {engine_name}: {e}")
            return {
                'predictions': [],
                'performance_score': 0.0,
                'response_time': 0.0
            }
    
    def _calculate_engine_performance(self, predictions: List[Dict[str, Any]], response_time: float) -> float:
        """Calculate performance score for an engine"""
        try:
            if not predictions:
                return 0.0
            
            # Confidence factor
            confidences = [pred.get('confidence', 0) for pred in predictions]
            avg_confidence = np.mean(confidences) if confidences else 0.5
            
            # Response time factor (faster is better, cap at 10 seconds)
            time_factor = max(0.1, 1.0 - (response_time / 10.0))
            
            # Prediction count factor
            count_factor = min(1.0, len(predictions) / 5.0)
            
            # Combined performance score
            performance = (avg_confidence * 0.5) + (time_factor * 0.3) + (count_factor * 0.2)
            return min(1.0, max(0.0, performance))
            
        except Exception as e:
            logger.error(f"Error calculating engine performance: {e}")
            return 0.5
    
    def _calculate_orchestration_intelligence(self, synergy_analysis: Dict[str, float], 
                                            dynamic_weights: Dict[str, float], 
                                            performance_scores: Dict[str, float]) -> float:
        """Calculate overall orchestration intelligence score"""
        try:
            intelligence_factors = []
            
            # Synergy factor
            if synergy_analysis:
                avg_synergy = np.mean(list(synergy_analysis.values()))
                intelligence_factors.append(avg_synergy)
            
            # Weight balance factor
            if dynamic_weights:
                weight_std = np.std(list(dynamic_weights.values()))
                weight_balance = 1.0 - min(1.0, weight_std * 2)  # Lower std = better balance
                intelligence_factors.append(weight_balance)
            
            # Performance factor
            if performance_scores:
                avg_performance = np.mean(list(performance_scores.values()))
                intelligence_factors.append(avg_performance)
            
            # Engine diversity factor
            active_engines = len([w for w in dynamic_weights.values() if w > 0.1])
            diversity_factor = min(1.0, active_engines / 4.0)  # Normalize to 4 engines
            intelligence_factors.append(diversity_factor)
            
            return np.mean(intelligence_factors) if intelligence_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating orchestration intelligence: {e}")
            return 0.5
    
    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Get comprehensive orchestration summary"""
        try:
            return {
                'orchestration_intelligence': self.orchestration_intelligence_score,
                'active_engines': len([e for e in self.engine_weights.values() if e > 0]),
                'total_predictions': len(self.prediction_history),
                'engine_weights': self.engine_weights,
                'last_synergy_analysis': list(self.synergy_analysis_cache.keys())[-1] if self.synergy_analysis_cache else None,
                'ultra_accuracy_sessions': len(self.ultra_accuracy_cache),
                'dynamic_weights_history_length': len(self.dynamic_weights_history),
                'sophisticated_capabilities': [
                    'ultra_accuracy_orchestration',
                    'sophisticated_aggregation',
                    'dynamic_weight_calculation',
                    'engine_synergy_analysis',
                    'orchestration_intelligence'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting orchestration summary: {e}")
            return {'error': str(e)}
    
    def export_orchestration_config(self) -> Dict[str, Any]:
        """
        Export current orchestration configuration.
        
        Returns:
            Dictionary containing complete configuration
        """
        return {
            'orchestration_config': self.orchestration_config,
            'engine_weights': self.engine_weights,
            'orchestration_strategy': self.orchestration_strategy,
            'confidence_threshold': self.confidence_threshold,
            'game_config': self.game_config,
            'performance_summary': {
                'total_predictions': len(self.prediction_history),
                'engines_trained': len([engine for engine in self.engines.keys() 
                                      if self.engine_weights.get(engine, 0) > 0])
            }
        }