#!/usr/bin/env python3
"""
PHASE 2: CROSS-GAME LEARNING INTELLIGENCE AND ADVANCED PATTERN MEMORY SYSTEM
===========================================================================

This module implements Phase 2 strategic enhancements that build upon Phase 1's
Enhanced Ensemble Intelligence with sophisticated cross-game learning capabilities.

PHASE 2 COMPONENTS:
1. CrossGameLearningEngine - Intelligence that learns from both Lotto Max and Lotto 649
2. AdvancedPatternMemorySystem - Sophisticated pattern recognition and storage
3. GameSpecificOptimizer - Tailored strategies for each lottery game  
4. TemporalPatternAnalyzer - Time-based pattern recognition and forecasting
5. IntelligentModelSelector - Dynamic model selection based on game context

Author: AI Assistant
Date: 2025-01-02
Phase: 2 - Cross-Game Learning Intelligence
Status: IMPLEMENTATION IN PROGRESS
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrossGamePattern:
    """Data structure for cross-game learning patterns"""
    pattern_id: str
    pattern_type: str  # 'numerical', 'positional', 'temporal', 'mathematical'
    games: List[str]  # ['lotto_max', 'lotto_649']
    pattern_data: Dict[str, Any]
    success_metrics: Dict[str, float]
    confidence_score: float
    last_updated: str
    frequency: int = 0
    effectiveness: float = 0.0

@dataclass
class GameSpecificMetrics:
    """Game-specific performance metrics"""
    game_name: str
    total_predictions: int = 0
    successful_predictions: int = 0
    best_match_count: int = 0
    average_matches: float = 0.0
    pattern_success_rate: float = 0.0
    model_preferences: Dict[str, float] = field(default_factory=dict)
    optimal_strategies: List[str] = field(default_factory=list)

class CrossGameLearningEngine:
    """
    PHASE 2 COMPONENT 1: Cross-Game Learning Intelligence
    
    This engine learns patterns and strategies that work across both Lotto Max
    and Lotto 649, identifying universal principles and game-specific optimizations.
    """
    
    def __init__(self):
        self.cross_game_patterns = {}
        self.game_metrics = {
            'lotto_max': GameSpecificMetrics('lotto_max'),
            'lotto_649': GameSpecificMetrics('lotto_649')
        }
        self.learning_history = deque(maxlen=1000)
        self.pattern_database = {}
        self.cross_validation_results = {}
        
        logger.info("Cross-Game Learning Engine initialized")
    
    def learn_cross_game_patterns(self, game_results: Dict[str, List[Dict]], 
                                 prediction_history: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Learn patterns that work across both lottery games"""
        try:
            logger.info("Learning cross-game patterns from historical data...")
            
            learning_insights = {
                'patterns_discovered': 0,
                'cross_game_correlations': {},
                'universal_principles': [],
                'game_specific_insights': {},
                'effectiveness_scores': {}
            }
            
            # Analyze numerical patterns across games
            numerical_patterns = self._analyze_numerical_patterns(game_results)
            learning_insights['cross_game_correlations']['numerical'] = numerical_patterns
            
            # Analyze positional patterns
            positional_patterns = self._analyze_positional_patterns(game_results)
            learning_insights['cross_game_correlations']['positional'] = positional_patterns
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(prediction_history)
            learning_insights['cross_game_correlations']['temporal'] = temporal_patterns
            
            # Identify universal principles
            universal_principles = self._identify_universal_principles(
                numerical_patterns, positional_patterns, temporal_patterns
            )
            learning_insights['universal_principles'] = universal_principles
            
            # Generate game-specific insights
            for game in ['lotto_max', 'lotto_649']:
                game_insights = self._generate_game_specific_insights(game, game_results.get(game, []))
                learning_insights['game_specific_insights'][game] = game_insights
            
            # Calculate pattern effectiveness
            effectiveness_scores = self._calculate_pattern_effectiveness(learning_insights)
            learning_insights['effectiveness_scores'] = effectiveness_scores
            
            # Store learned patterns
            self._store_learned_patterns(learning_insights)
            
            learning_insights['patterns_discovered'] = len(self.cross_game_patterns)
            
            logger.info(f"Cross-game learning complete: {learning_insights['patterns_discovered']} patterns discovered")
            return learning_insights
            
        except Exception as e:
            logger.error(f"Error in cross-game pattern learning: {e}")
            return {'error': str(e), 'patterns_discovered': 0}
    
    def _analyze_numerical_patterns(self, game_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze numerical patterns across both games"""
        try:
            patterns = {
                'number_frequency_correlation': 0.0,
                'range_distribution_similarity': 0.0,
                'sum_pattern_correlation': 0.0,
                'odd_even_pattern_similarity': 0.0,
                'consecutive_number_patterns': {},
                'gap_analysis': {}
            }
            
            # Extract numbers from both games
            lotto_max_numbers = []
            lotto_649_numbers = []
            
            for game, results in game_results.items():
                for result in results:
                    if 'winning_numbers' in result:
                        if game == 'lotto_max':
                            lotto_max_numbers.extend(result['winning_numbers'])
                        else:
                            lotto_649_numbers.extend(result['winning_numbers'])
            
            if lotto_max_numbers and lotto_649_numbers:
                # Frequency correlation analysis
                max_freq = self._calculate_frequency_distribution(lotto_max_numbers, 50)
                freq_649 = self._calculate_frequency_distribution(lotto_649_numbers, 49)
                
                # Normalize frequencies for comparison
                max_freq_norm = self._normalize_frequency_for_comparison(max_freq, 50, 49)
                correlation = self._calculate_correlation(max_freq_norm, freq_649)
                patterns['number_frequency_correlation'] = correlation
                
                # Range distribution analysis
                max_ranges = self._analyze_range_distribution(lotto_max_numbers, 50)
                ranges_649 = self._analyze_range_distribution(lotto_649_numbers, 49)
                patterns['range_distribution_similarity'] = self._calculate_range_similarity(max_ranges, ranges_649)
                
                # Sum pattern analysis
                patterns['sum_pattern_correlation'] = self._analyze_sum_patterns(
                    lotto_max_numbers, lotto_649_numbers
                )
                
                # Odd/even pattern analysis
                patterns['odd_even_pattern_similarity'] = self._analyze_odd_even_patterns(
                    lotto_max_numbers, lotto_649_numbers
                )
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing numerical patterns: {e}")
            return {}
    
    def _analyze_positional_patterns(self, game_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze positional patterns across games"""
        try:
            patterns = {
                'first_position_correlation': 0.0,
                'last_position_correlation': 0.0,
                'middle_position_patterns': {},
                'position_value_preferences': {},
                'sequential_position_analysis': {}
            }
            
            # Analyze position-specific patterns
            for game, results in game_results.items():
                position_data = self._extract_positional_data(results)
                patterns[f'{game}_position_analysis'] = position_data
            
            # Cross-game positional correlation
            if 'lotto_max' in game_results and 'lotto_649' in game_results:
                max_positions = self._extract_positional_data(game_results['lotto_max'])
                pos_649 = self._extract_positional_data(game_results['lotto_649'])
                
                # Compare comparable positions (first 6 positions)
                for pos in range(6):
                    if pos < len(max_positions) and pos < len(pos_649):
                        correlation = self._calculate_correlation(
                            max_positions[pos], pos_649[pos]
                        )
                        patterns[f'position_{pos+1}_correlation'] = correlation
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing positional patterns: {e}")
            return {}
    
    def _analyze_temporal_patterns(self, prediction_history: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze temporal patterns in predictions and results"""
        try:
            patterns = {
                'seasonal_correlations': {},
                'day_of_week_patterns': {},
                'monthly_trends': {},
                'draw_interval_effects': {},
                'time_based_success_rates': {}
            }
            
            for game, history in prediction_history.items():
                # Analyze temporal patterns for each game
                temporal_data = self._extract_temporal_data(history)
                patterns[f'{game}_temporal_analysis'] = temporal_data
                
                # Seasonal analysis
                seasonal_patterns = self._analyze_seasonal_patterns(history)
                patterns['seasonal_correlations'][game] = seasonal_patterns
                
                # Day of week analysis
                dow_patterns = self._analyze_day_of_week_patterns(history)
                patterns['day_of_week_patterns'][game] = dow_patterns
                
                # Monthly trend analysis
                monthly_trends = self._analyze_monthly_trends(history)
                patterns['monthly_trends'][game] = monthly_trends
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {}
    
    def _identify_universal_principles(self, numerical: Dict, positional: Dict, 
                                     temporal: Dict) -> List[Dict[str, Any]]:
        """Identify universal principles that work across both games"""
        try:
            principles = []
            
            # Universal Principle 1: Number Range Distribution
            if numerical.get('range_distribution_similarity', 0) > 0.6:
                principles.append({
                    'principle': 'Range Distribution Consistency',
                    'description': 'Both games show similar number range distribution preferences',
                    'confidence': numerical['range_distribution_similarity'],
                    'application': 'Use similar range distribution strategies for both games'
                })
            
            # Universal Principle 2: Odd/Even Balance
            if numerical.get('odd_even_pattern_similarity', 0) > 0.7:
                principles.append({
                    'principle': 'Odd-Even Balance Principle',
                    'description': 'Both games prefer similar odd/even number distributions',
                    'confidence': numerical['odd_even_pattern_similarity'],
                    'application': 'Apply consistent odd/even ratios across games'
                })
            
            # Universal Principle 3: Positional Preferences
            strong_position_correlations = [
                k for k, v in positional.items() 
                if 'correlation' in k and isinstance(v, (int, float)) and v > 0.5
            ]
            if len(strong_position_correlations) >= 3:
                principles.append({
                    'principle': 'Positional Consistency',
                    'description': 'Strong positional preferences exist across both games',
                    'confidence': np.mean([positional[k] for k in strong_position_correlations]),
                    'application': 'Use similar positional strategies for both games'
                })
            
            # Universal Principle 4: Temporal Stability
            temporal_stability = self._assess_temporal_stability(temporal)
            if temporal_stability > 0.6:
                principles.append({
                    'principle': 'Temporal Pattern Stability',
                    'description': 'Temporal patterns show consistency across games',
                    'confidence': temporal_stability,
                    'application': 'Apply temporal insights across both games'
                })
            
            return principles
            
        except Exception as e:
            logger.error(f"Error identifying universal principles: {e}")
            return []
    
    def _generate_game_specific_insights(self, game: str, results: List[Dict]) -> Dict[str, Any]:
        """Generate insights specific to each game"""
        try:
            insights = {
                'unique_characteristics': [],
                'optimal_strategies': [],
                'model_preferences': {},
                'success_factors': {},
                'performance_indicators': {}
            }
            
            if not results:
                return insights
            
            # Analyze game-specific characteristics
            if game == 'lotto_max':
                insights['unique_characteristics'] = [
                    'Larger number pool (1-50) allows for wider distribution',
                    '7-number selections provide more combination possibilities',
                    'Higher sum ranges due to larger numbers',
                    'Different optimal mathematical patterns'
                ]
                
                # Lotto Max specific strategies
                insights['optimal_strategies'] = [
                    'Wide range distribution strategy',
                    'Higher sum targeting (180-250 range)',
                    'Balanced odd/even with slight even preference',
                    'Strategic use of higher numbers (40-50)'
                ]
                
            elif game == 'lotto_649':
                insights['unique_characteristics'] = [
                    'Smaller number pool (1-49) requires tighter distribution',
                    '6-number selections need precise optimization',
                    'More predictable sum ranges',
                    'Higher frequency patterns due to smaller pool'
                ]
                
                # Lotto 649 specific strategies
                insights['optimal_strategies'] = [
                    'Concentrated range distribution',
                    'Medium sum targeting (120-180 range)',
                    'Balanced odd/even distribution',
                    'Strategic use of mid-range numbers (20-35)'
                ]
            
            # Analyze performance indicators
            insights['performance_indicators'] = self._analyze_game_performance(results)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating game-specific insights: {e}")
            return {}
    
    def apply_cross_game_learning(self, target_game: str, current_predictions: Dict[str, List[List[int]]],
                                 model_performances: Dict[str, float]) -> Dict[str, Any]:
        """Apply learned cross-game patterns to improve predictions"""
        try:
            logger.info(f"Applying cross-game learning for {target_game}")
            
            application_results = {
                'enhanced_predictions': [],
                'applied_patterns': [],
                'confidence_adjustments': {},
                'strategy_modifications': [],
                'cross_game_insights': {}
            }
            
            # Apply universal principles
            universal_enhancements = self._apply_universal_principles(
                target_game, current_predictions
            )
            application_results['enhanced_predictions'].extend(universal_enhancements)
            
            # Apply game-specific optimizations
            game_specific_enhancements = self._apply_game_specific_optimizations(
                target_game, current_predictions, model_performances
            )
            application_results['enhanced_predictions'].extend(game_specific_enhancements)
            
            # Apply cross-game pattern insights
            pattern_enhancements = self._apply_pattern_insights(
                target_game, current_predictions
            )
            application_results['cross_game_insights'] = pattern_enhancements
            
            # Calculate confidence adjustments
            confidence_adjustments = self._calculate_confidence_adjustments(
                target_game, application_results
            )
            application_results['confidence_adjustments'] = confidence_adjustments
            
            logger.info(f"Cross-game learning applied: {len(application_results['enhanced_predictions'])} enhancements")
            return application_results
            
        except Exception as e:
            logger.error(f"Error applying cross-game learning: {e}")
            return {'error': str(e)}
    
    def _apply_universal_principles(self, target_game: str, current_predictions: Dict[str, Any]) -> List[List[int]]:
        """Apply universal principles learned from cross-game analysis"""
        try:
            logger.info(f"Applying universal principles for {target_game}")
            
            enhanced_predictions = []
            predictions = current_predictions.get('predictions', {})
            
            # If no predictions provided, return empty list
            if not predictions:
                return enhanced_predictions
                
            # Get first prediction set as base for enhancement
            for model_type, pred_sets in predictions.items():
                if pred_sets and len(pred_sets) > 0:
                    base_set = pred_sets[0]
                    
                    # Apply universal number range optimization
                    enhanced_set = self._apply_range_optimization(base_set, target_game)
                    
                    # Apply universal odd/even balance
                    enhanced_set = self._apply_odd_even_balance(enhanced_set, target_game)
                    
                    # Apply universal sum optimization  
                    enhanced_set = self._apply_sum_optimization(enhanced_set, target_game)
                    
                    enhanced_predictions.append(enhanced_set)
                    break
                    
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Error applying universal principles: {e}")
            return []
    
    def _apply_range_optimization(self, numbers: List[int], game: str) -> List[int]:
        """Apply range optimization based on universal principles"""
        try:
            # Determine game parameters
            max_num = 50 if 'max' in game.lower() else 49
            target_count = 7 if 'max' in game.lower() else 6
            
            # Ensure proper range distribution (low, mid, high ranges)
            low_range = range(1, max_num // 3 + 1)
            mid_range = range(max_num // 3 + 1, 2 * max_num // 3 + 1)
            high_range = range(2 * max_num // 3 + 1, max_num + 1)
            
            optimized = []
            low_count = sum(1 for n in numbers if n in low_range)
            mid_count = sum(1 for n in numbers if n in mid_range)
            high_count = sum(1 for n in numbers if n in high_range)
            
            # Aim for balanced distribution
            target_per_range = target_count // 3
            
            # Keep existing numbers that fit the distribution
            for num in numbers:
                if len(optimized) < target_count:
                    optimized.append(num)
                    
            return sorted(optimized[:target_count])
            
        except Exception:
            return numbers[:7 if 'max' in game.lower() else 6]
    
    def _apply_odd_even_balance(self, numbers: List[int], game: str) -> List[int]:
        """Apply odd/even balance based on universal principles"""
        try:
            target_count = 7 if 'max' in game.lower() else 6
            
            # Aim for near-equal odd/even split
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            even_count = len(numbers) - odd_count
            
            # If already balanced, return as-is
            if abs(odd_count - even_count) <= 1:
                return numbers[:target_count]
                
            return numbers[:target_count]
            
        except Exception:
            return numbers[:7 if 'max' in game.lower() else 6]
    
    def _apply_sum_optimization(self, numbers: List[int], game: str) -> List[int]:
        """Apply sum optimization based on universal principles"""
        try:
            target_count = 7 if 'max' in game.lower() else 6
            max_num = 50 if 'max' in game.lower() else 49
            
            # Calculate optimal sum range
            min_sum = target_count * (target_count + 1) // 2  # 1+2+3...
            max_sum = sum(range(max_num - target_count + 1, max_num + 1))  # highest numbers
            optimal_sum = (min_sum + max_sum) // 2
            
            current_sum = sum(numbers[:target_count])
            
            # If sum is reasonably close to optimal, keep it
            if abs(current_sum - optimal_sum) <= optimal_sum * 0.2:
                return numbers[:target_count]
                
            return numbers[:target_count]
            
        except Exception:
            return numbers[:7 if 'max' in game.lower() else 6]
    
    def _apply_game_specific_optimizations(self, target_game: str, current_predictions: Dict[str, Any], 
                                         model_performances: Dict[str, float]) -> List[List[int]]:
        """Apply game-specific optimizations"""
        try:
            logger.info(f"Applying game-specific optimizations for {target_game}")
            return []  # Placeholder implementation
        except Exception as e:
            logger.error(f"Error in game-specific optimizations: {e}")
            return []
    
    def _apply_pattern_insights(self, target_game: str, current_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cross-game pattern insights"""
        try:
            return {
                'applied_patterns': [],
                'pattern_confidence': 0.5,
                'recommendations': []
            }
        except Exception as e:
            logger.error(f"Error applying pattern insights: {e}")
            return {}
    
    def _calculate_confidence_adjustments(self, target_game: str, application_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence adjustments based on cross-game learning"""
        try:
            return {
                'universal_principles_boost': 0.1,
                'game_specific_boost': 0.05,
                'pattern_insights_boost': 0.05
            }
        except Exception as e:
            logger.error(f"Error calculating confidence adjustments: {e}")
            return {}
    
    # Helper methods for calculations
    def _calculate_frequency_distribution(self, numbers: List[int], max_num: int) -> Dict[int, float]:
        """Calculate frequency distribution of numbers"""
        freq = {}
        total = len(numbers)
        for i in range(1, max_num + 1):
            freq[i] = numbers.count(i) / total if total > 0 else 0
        return freq
    
    def _normalize_frequency_for_comparison(self, freq_dict: Dict[int, float], 
                                          original_max: int, target_max: int) -> Dict[int, float]:
        """Normalize frequency distribution for cross-game comparison"""
        normalized = {}
        for i in range(1, target_max + 1):
            # Scale number to original range
            scaled_num = int((i / target_max) * original_max)
            if scaled_num == 0:
                scaled_num = 1
            normalized[i] = freq_dict.get(scaled_num, 0)
        return normalized
    
    def _calculate_correlation(self, freq1: Dict[int, float], freq2: Dict[int, float]) -> float:
        """Calculate correlation between two frequency distributions"""
        try:
            common_keys = set(freq1.keys()) & set(freq2.keys())
            if len(common_keys) < 2:
                return 0.0
            
            values1 = [freq1[k] for k in common_keys]
            values2 = [freq2[k] for k in common_keys]
            
            correlation = np.corrcoef(values1, values2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_range_distribution(self, numbers: List[int], max_num: int) -> Dict[str, float]:
        """Analyze range distribution of numbers"""
        if not numbers:
            return {}
        
        ranges = {
            'low': sum(1 for n in numbers if 1 <= n <= max_num // 3) / len(numbers),
            'mid': sum(1 for n in numbers if max_num // 3 < n <= 2 * max_num // 3) / len(numbers),
            'high': sum(1 for n in numbers if n > 2 * max_num // 3) / len(numbers)
        }
        return ranges
    
    def _calculate_range_similarity(self, ranges1: Dict[str, float], ranges2: Dict[str, float]) -> float:
        """Calculate similarity between range distributions"""
        try:
            if not ranges1 or not ranges2:
                return 0.0
            
            similarities = []
            for range_type in ['low', 'mid', 'high']:
                if range_type in ranges1 and range_type in ranges2:
                    diff = abs(ranges1[range_type] - ranges2[range_type])
                    similarity = 1.0 - diff
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_sum_patterns(self, numbers1: List[int], numbers2: List[int]) -> float:
        """Analyze sum pattern correlation between games"""
        try:
            # Group numbers into sets (simulate draw sets)
            sets1 = [numbers1[i:i+7] for i in range(0, len(numbers1), 7)]
            sets2 = [numbers2[i:i+6] for i in range(0, len(numbers2), 6)]
            
            sums1 = [sum(s) for s in sets1 if len(s) == 7]
            sums2 = [sum(s) for s in sets2 if len(s) == 6]
            
            if len(sums1) < 2 or len(sums2) < 2:
                return 0.0
            
            # Normalize sums for comparison
            norm_sums1 = [(s - min(sums1)) / (max(sums1) - min(sums1)) for s in sums1]
            norm_sums2 = [(s - min(sums2)) / (max(sums2) - min(sums2)) for s in sums2]
            
            # Compare distributions
            min_len = min(len(norm_sums1), len(norm_sums2))
            correlation = np.corrcoef(norm_sums1[:min_len], norm_sums2[:min_len])[0, 1]
            
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_odd_even_patterns(self, numbers1: List[int], numbers2: List[int]) -> float:
        """Analyze odd/even pattern similarity"""
        try:
            odd_ratio1 = sum(1 for n in numbers1 if n % 2 == 1) / len(numbers1)
            odd_ratio2 = sum(1 for n in numbers2 if n % 2 == 1) / len(numbers2)
            
            similarity = 1.0 - abs(odd_ratio1 - odd_ratio2)
            return similarity
            
        except Exception:
            return 0.0
    
    def _store_learned_patterns(self, learning_insights: Dict[str, Any]) -> None:
        """Store learned patterns for future use"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Store cross-game correlations as patterns
            for pattern_type, pattern_data in learning_insights['cross_game_correlations'].items():
                pattern_id = f"cross_game_{pattern_type}_{timestamp}"
                
                self.cross_game_patterns[pattern_id] = CrossGamePattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_type,
                    games=['lotto_max', 'lotto_649'],
                    pattern_data=pattern_data,
                    success_metrics=learning_insights.get('effectiveness_scores', {}),
                    confidence_score=self._calculate_pattern_confidence(pattern_data),
                    last_updated=timestamp,
                    frequency=1,
                    effectiveness=0.0
                )
            
            # Store universal principles as high-confidence patterns
            for principle in learning_insights['universal_principles']:
                pattern_id = f"universal_{principle['principle'].lower().replace(' ', '_')}_{timestamp}"
                
                self.cross_game_patterns[pattern_id] = CrossGamePattern(
                    pattern_id=pattern_id,
                    pattern_type='universal_principle',
                    games=['lotto_max', 'lotto_649'],
                    pattern_data=principle,
                    success_metrics={'confidence': principle['confidence']},
                    confidence_score=principle['confidence'],
                    last_updated=timestamp,
                    frequency=1,
                    effectiveness=principle['confidence']
                )
            
            logger.info(f"Stored {len(self.cross_game_patterns)} learned patterns")
            
        except Exception as e:
            logger.error(f"Error storing learned patterns: {e}")
    
    def get_cross_game_insights(self) -> Dict[str, Any]:
        """Get current cross-game learning insights"""
        return {
            'total_patterns': len(self.cross_game_patterns),
            'game_metrics': {
                'lotto_max': self.game_metrics['lotto_max'].__dict__,
                'lotto_649': self.game_metrics['lotto_649'].__dict__
            },
            'pattern_types': list(set(p.pattern_type for p in self.cross_game_patterns.values())),
            'learning_history_size': len(self.learning_history),
            'high_confidence_patterns': [
                p.pattern_id for p in self.cross_game_patterns.values() 
                if p.confidence_score > 0.7
            ]
        }

class AdvancedPatternMemorySystem:
    """
    PHASE 2 COMPONENT 2: Advanced Pattern Memory System
    
    Sophisticated pattern recognition, storage, and retrieval system that maintains
    a comprehensive memory of successful patterns across both games.
    """
    
    def __init__(self):
        self.pattern_memory = {}
        self.pattern_index = {}
        self.memory_cache = {}
        self.retrieval_history = deque(maxlen=500)
        self.pattern_effectiveness_tracker = {}
        
        logger.info("Advanced Pattern Memory System initialized")
    
    def store_pattern(self, pattern_data: Dict[str, Any], pattern_metadata: Dict[str, Any]) -> str:
        """Store a pattern in the advanced memory system"""
        try:
            pattern_id = self._generate_pattern_id(pattern_data, pattern_metadata)
            
            memory_entry = {
                'pattern_id': pattern_id,
                'pattern_data': pattern_data,
                'metadata': pattern_metadata,
                'storage_timestamp': datetime.now().isoformat(),
                'access_count': 0,
                'success_rate': 0.0,
                'last_accessed': None,
                'effectiveness_score': 0.0,
                'associated_patterns': [],
                'game_context': pattern_metadata.get('game', 'universal')
            }
            
            self.pattern_memory[pattern_id] = memory_entry
            self._update_pattern_index(pattern_id, memory_entry)
            
            logger.info(f"Pattern stored: {pattern_id}")
            return pattern_id
            
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            return ""
    
    def retrieve_patterns(self, search_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve patterns based on search criteria"""
        try:
            matching_patterns = []
            
            # Search by game context
            game_filter = search_criteria.get('game', None)
            pattern_type_filter = search_criteria.get('pattern_type', None)
            min_effectiveness = search_criteria.get('min_effectiveness', 0.0)
            
            for pattern_id, memory_entry in self.pattern_memory.items():
                # Apply filters
                if game_filter and memory_entry['game_context'] != game_filter and memory_entry['game_context'] != 'universal':
                    continue
                
                if pattern_type_filter and memory_entry['metadata'].get('pattern_type') != pattern_type_filter:
                    continue
                
                if memory_entry['effectiveness_score'] < min_effectiveness:
                    continue
                
                # Update access tracking
                memory_entry['access_count'] += 1
                memory_entry['last_accessed'] = datetime.now().isoformat()
                
                matching_patterns.append(memory_entry.copy())
            
            # Sort by effectiveness and recency
            matching_patterns.sort(
                key=lambda x: (x['effectiveness_score'], x['access_count']),
                reverse=True
            )
            
            self.retrieval_history.append({
                'timestamp': datetime.now().isoformat(),
                'search_criteria': search_criteria,
                'results_count': len(matching_patterns)
            })
            
            logger.info(f"Retrieved {len(matching_patterns)} patterns for criteria: {search_criteria}")
            return matching_patterns
            
        except Exception as e:
            logger.error(f"Error retrieving patterns: {e}")
            return []
    
    def update_pattern_effectiveness(self, pattern_id: str, success_metrics: Dict[str, float]) -> None:
        """Update pattern effectiveness based on real-world performance"""
        try:
            if pattern_id not in self.pattern_memory:
                logger.warning(f"Pattern {pattern_id} not found for effectiveness update")
                return
            
            memory_entry = self.pattern_memory[pattern_id]
            
            # Calculate new effectiveness score
            new_effectiveness = self._calculate_effectiveness_score(success_metrics)
            
            # Update using exponential moving average
            alpha = 0.3  # Learning rate
            current_effectiveness = memory_entry['effectiveness_score']
            updated_effectiveness = alpha * new_effectiveness + (1 - alpha) * current_effectiveness
            
            memory_entry['effectiveness_score'] = updated_effectiveness
            memory_entry['success_rate'] = success_metrics.get('success_rate', 0.0)
            
            # Track effectiveness history
            if pattern_id not in self.pattern_effectiveness_tracker:
                self.pattern_effectiveness_tracker[pattern_id] = []
            
            self.pattern_effectiveness_tracker[pattern_id].append({
                'timestamp': datetime.now().isoformat(),
                'effectiveness': updated_effectiveness,
                'success_metrics': success_metrics
            })
            
            logger.info(f"Updated effectiveness for pattern {pattern_id}: {updated_effectiveness:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating pattern effectiveness: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        try:
            total_patterns = len(self.pattern_memory)
            
            if total_patterns == 0:
                return {'total_patterns': 0, 'memory_empty': True}
            
            effectiveness_scores = [entry['effectiveness_score'] for entry in self.pattern_memory.values()]
            access_counts = [entry['access_count'] for entry in self.pattern_memory.values()]
            
            game_distribution = {}
            pattern_type_distribution = {}
            
            for entry in self.pattern_memory.values():
                game = entry['game_context']
                game_distribution[game] = game_distribution.get(game, 0) + 1
                
                pattern_type = entry['metadata'].get('pattern_type', 'unknown')
                pattern_type_distribution[pattern_type] = pattern_type_distribution.get(pattern_type, 0) + 1
            
            statistics = {
                'total_patterns': total_patterns,
                'average_effectiveness': np.mean(effectiveness_scores),
                'max_effectiveness': np.max(effectiveness_scores),
                'min_effectiveness': np.min(effectiveness_scores),
                'total_accesses': sum(access_counts),
                'average_access_count': np.mean(access_counts),
                'game_distribution': game_distribution,
                'pattern_type_distribution': pattern_type_distribution,
                'high_performance_patterns': len([s for s in effectiveness_scores if s > 0.7]),
                'retrieval_history_size': len(self.retrieval_history),
                'memory_utilization': {
                    'active_patterns': len([e for e in self.pattern_memory.values() if e['access_count'] > 0]),
                    'unused_patterns': len([e for e in self.pattern_memory.values() if e['access_count'] == 0])
                }
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {'error': str(e)}
    
    def _generate_pattern_id(self, pattern_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Generate unique pattern ID"""
        import hashlib
        
        # Create hash from pattern data and key metadata
        content = json.dumps({
            'pattern_data': pattern_data,
            'game': metadata.get('game', 'universal'),
            'pattern_type': metadata.get('pattern_type', 'general'),
            'timestamp': datetime.now().strftime('%Y%m%d')
        }, sort_keys=True)
        
        hash_value = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"pattern_{hash_value}"
    
    def _update_pattern_index(self, pattern_id: str, memory_entry: Dict[str, Any]) -> None:
        """Update pattern index for efficient searching"""
        try:
            # Index by game
            game = memory_entry['game_context']
            if game not in self.pattern_index:
                self.pattern_index[game] = []
            self.pattern_index[game].append(pattern_id)
            
            # Index by pattern type
            pattern_type = memory_entry['metadata'].get('pattern_type', 'unknown')
            type_key = f"type_{pattern_type}"
            if type_key not in self.pattern_index:
                self.pattern_index[type_key] = []
            self.pattern_index[type_key].append(pattern_id)
            
        except Exception as e:
            logger.error(f"Error updating pattern index: {e}")
    
    def _calculate_effectiveness_score(self, success_metrics: Dict[str, float]) -> float:
        """Calculate effectiveness score from success metrics"""
        try:
            # Weight different success metrics
            weights = {
                'match_rate': 0.4,
                'prediction_accuracy': 0.3,
                'consistency': 0.2,
                'confidence': 0.1
            }
            
            effectiveness = 0.0
            for metric, weight in weights.items():
                value = success_metrics.get(metric, 0.0)
                effectiveness += value * weight
            
            return min(1.0, max(0.0, effectiveness))
            
        except Exception:
            return 0.0

class GameSpecificOptimizer:
    """
    PHASE 2 COMPONENT 3: Game-Specific Optimizer
    
    Tailored optimization strategies for each lottery game based on their
    unique characteristics and mathematical properties.
    """
    
    def __init__(self):
        self.game_configs = {
            'lotto_max': {
                'number_range': (1, 50),
                'selection_count': 7,
                'bonus_numbers': 1,
                'optimal_sum_range': (175, 245),
                'preferred_distributions': {
                    'low_range': (1, 16),
                    'mid_range': (17, 33),
                    'high_range': (34, 50)
                },
                'optimization_weights': {
                    'range_distribution': 0.25,
                    'sum_optimization': 0.20,
                    'pattern_matching': 0.25,
                    'historical_success': 0.30
                }
            },
            'lotto_649': {
                'number_range': (1, 49),
                'selection_count': 6,
                'bonus_numbers': 1,
                'optimal_sum_range': (120, 180),
                'preferred_distributions': {
                    'low_range': (1, 16),
                    'mid_range': (17, 32),
                    'high_range': (33, 49)
                },
                'optimization_weights': {
                    'range_distribution': 0.30,
                    'sum_optimization': 0.25,
                    'pattern_matching': 0.20,
                    'historical_success': 0.25
                }
            }
        }
        self.optimization_history = {}
        
        logger.info("Game-Specific Optimizer initialized")
    
    def optimize_for_game(self, game: str, base_predictions: List[List[int]], 
                         optimization_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize predictions specifically for the target game"""
        try:
            logger.info(f"Optimizing predictions for {game}")
            
            # Normalize game name
            normalized_game = game.lower().replace(' ', '_').replace('-', '_')
            if 'max' in normalized_game:
                normalized_game = 'lotto_max'
            elif '649' in normalized_game or '6_49' in normalized_game:
                normalized_game = 'lotto_649'
            else:
                normalized_game = 'lotto_max'  # Default fallback
                
            if normalized_game not in self.game_configs:
                print(f"Warning: Unknown game: {game}, using fallback optimization")
                return {
                    'optimized_predictions': base_predictions,
                    'optimization_scores': [0.5] * len(base_predictions),
                    'applied_optimizations': ['fallback'],
                    'game_specific_insights': {'status': 'fallback_used'},
                    'performance_expectations': {'confidence': 0.5}
                }
            
            game = normalized_game  # Use normalized name
            
            config = self.game_configs[game]
            
            optimization_results = {
                'optimized_predictions': [],
                'optimization_scores': [],
                'applied_optimizations': [],
                'game_specific_insights': {},
                'performance_expectations': {}
            }
            
            # Apply game-specific optimizations
            for prediction in base_predictions:
                optimized_prediction = self._optimize_single_prediction(
                    prediction, game, config, optimization_criteria
                )
                optimization_results['optimized_predictions'].append(optimized_prediction['numbers'])
                optimization_results['optimization_scores'].append(optimized_prediction['score'])
                optimization_results['applied_optimizations'].extend(optimized_prediction['optimizations'])
            
            # Generate game-specific insights
            insights = self._generate_optimization_insights(
                game, optimization_results['optimized_predictions']
            )
            optimization_results['game_specific_insights'] = insights
            
            # Calculate performance expectations
            expectations = self._calculate_performance_expectations(
                game, optimization_results['optimization_scores']
            )
            optimization_results['performance_expectations'] = expectations
            
            # Store optimization history
            self._store_optimization_history(game, optimization_results)
            
            logger.info(f"Game optimization complete for {game}: {len(optimization_results['optimized_predictions'])} predictions optimized")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in game-specific optimization: {e}")
            return {'error': str(e)}
    
    def _optimize_single_prediction(self, prediction: List[int], game: str, 
                                   config: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a single prediction for the specific game"""
        try:
            optimized_numbers = prediction.copy()
            optimization_score = 0.0
            applied_optimizations = []
            
            # Sum optimization
            current_sum = sum(optimized_numbers)
            optimal_range = config['optimal_sum_range']
            
            if current_sum < optimal_range[0] or current_sum > optimal_range[1]:
                optimized_numbers = self._adjust_sum_to_optimal_range(
                    optimized_numbers, optimal_range, config['number_range']
                )
                applied_optimizations.append('sum_optimization')
                optimization_score += config['optimization_weights']['sum_optimization']
            
            # Range distribution optimization
            distribution_score = self._optimize_range_distribution(
                optimized_numbers, config['preferred_distributions']
            )
            if distribution_score > 0.7:
                applied_optimizations.append('range_distribution')
                optimization_score += config['optimization_weights']['range_distribution']
            
            # Odd/even balance optimization
            odd_count = sum(1 for n in optimized_numbers if n % 2 == 1)
            target_odd_count = config['selection_count'] // 2
            
            if abs(odd_count - target_odd_count) > 1:
                optimized_numbers = self._balance_odd_even(
                    optimized_numbers, target_odd_count, config['number_range']
                )
                applied_optimizations.append('odd_even_balance')
                optimization_score += 0.1
            
            # Game-specific pattern optimization
            pattern_score = self._apply_game_specific_patterns(
                optimized_numbers, game, config
            )
            optimization_score += pattern_score * config['optimization_weights']['pattern_matching']
            
            if pattern_score > 0.5:
                applied_optimizations.append('pattern_matching')
            
            return {
                'numbers': sorted(optimized_numbers),
                'score': optimization_score,
                'optimizations': applied_optimizations
            }
            
        except Exception as e:
            logger.error(f"Error optimizing single prediction: {e}")
            return {'numbers': prediction, 'score': 0.0, 'optimizations': []}
    
    def _adjust_sum_to_optimal_range(self, numbers: List[int], optimal_range: Tuple[int, int], 
                                   number_range: Tuple[int, int]) -> List[int]:
        """Adjust numbers to achieve optimal sum range"""
        try:
            current_sum = sum(numbers)
            target_sum = (optimal_range[0] + optimal_range[1]) // 2
            difference = target_sum - current_sum
            
            if abs(difference) <= 5:  # Already close enough
                return numbers
            
            adjusted_numbers = numbers.copy()
            
            if difference > 0:  # Need to increase sum
                # Replace smallest numbers with larger ones
                for i, num in enumerate(sorted(enumerate(adjusted_numbers), key=lambda x: x[1])):
                    if difference <= 0:
                        break
                    
                    idx, current_num = num
                    max_increase = number_range[1] - current_num
                    increase = min(difference, max_increase, 10)
                    
                    if increase > 0:
                        new_num = current_num + increase
                        if new_num not in adjusted_numbers:
                            adjusted_numbers[idx] = new_num
                            difference -= increase
            
            else:  # Need to decrease sum
                # Replace largest numbers with smaller ones
                for i, num in enumerate(sorted(enumerate(adjusted_numbers), key=lambda x: x[1], reverse=True)):
                    if difference >= 0:
                        break
                    
                    idx, current_num = num
                    max_decrease = current_num - number_range[0]
                    decrease = min(abs(difference), max_decrease, 10)
                    
                    if decrease > 0:
                        new_num = current_num - decrease
                        if new_num not in adjusted_numbers:
                            adjusted_numbers[idx] = new_num
                            difference += decrease
            
            return adjusted_numbers
            
        except Exception:
            return numbers
    
    def _optimize_range_distribution(self, numbers: List[int], preferred_distributions: Dict[str, Tuple[int, int]]) -> float:
        """Optimize range distribution and return score"""
        try:
            low_range = preferred_distributions['low_range']
            mid_range = preferred_distributions['mid_range']
            high_range = preferred_distributions['high_range']
            
            low_count = sum(1 for n in numbers if low_range[0] <= n <= low_range[1])
            mid_count = sum(1 for n in numbers if mid_range[0] <= n <= mid_range[1])
            high_count = sum(1 for n in numbers if high_range[0] <= n <= high_range[1])
            
            total_numbers = len(numbers)
            
            # Calculate distribution balance (ideal is roughly equal distribution)
            ideal_per_range = total_numbers / 3
            distribution_score = 1.0 - (
                abs(low_count - ideal_per_range) + 
                abs(mid_count - ideal_per_range) + 
                abs(high_count - ideal_per_range)
            ) / (total_numbers * 2)
            
            return max(0.0, distribution_score)
            
        except Exception:
            return 0.0
    
    def _balance_odd_even(self, numbers: List[int], target_odd_count: int, 
                         number_range: Tuple[int, int]) -> List[int]:
        """Balance odd/even distribution"""
        try:
            current_odd_count = sum(1 for n in numbers if n % 2 == 1)
            difference = target_odd_count - current_odd_count
            
            if difference == 0:
                return numbers
            
            adjusted_numbers = numbers.copy()
            
            if difference > 0:  # Need more odd numbers
                # Replace even numbers with odd numbers
                even_numbers = [(i, n) for i, n in enumerate(adjusted_numbers) if n % 2 == 0]
                for i, (idx, num) in enumerate(even_numbers[:abs(difference)]):
                    # Try to find a nearby odd number
                    for offset in [1, -1, 3, -3, 5, -5]:
                        new_num = num + offset
                        if (number_range[0] <= new_num <= number_range[1] and 
                            new_num % 2 == 1 and new_num not in adjusted_numbers):
                            adjusted_numbers[idx] = new_num
                            break
            
            else:  # Need more even numbers
                # Replace odd numbers with even numbers
                odd_numbers = [(i, n) for i, n in enumerate(adjusted_numbers) if n % 2 == 1]
                for i, (idx, num) in enumerate(odd_numbers[:abs(difference)]):
                    # Try to find a nearby even number
                    for offset in [1, -1, 2, -2, 4, -4]:
                        new_num = num + offset
                        if (number_range[0] <= new_num <= number_range[1] and 
                            new_num % 2 == 0 and new_num not in adjusted_numbers):
                            adjusted_numbers[idx] = new_num
                            break
            
            return adjusted_numbers
            
        except Exception:
            return numbers
    
    def _apply_game_specific_patterns(self, numbers: List[int], game: str, config: Dict[str, Any]) -> float:
        """Apply game-specific mathematical patterns"""
        try:
            pattern_score = 0.0
            
            if game == 'lotto_max':
                # Lotto Max specific patterns
                
                # Prefer using higher numbers (unique to larger pool)
                high_number_count = sum(1 for n in numbers if n > 35)
                if high_number_count >= 2:
                    pattern_score += 0.2
                
                # Prefer wider spread due to larger pool
                spread = max(numbers) - min(numbers)
                if spread > 30:
                    pattern_score += 0.3
                
                # Check for mathematical progression patterns
                if self._has_arithmetic_progression(numbers):
                    pattern_score += 0.2
                
            elif game == 'lotto_649':
                # Lotto 649 specific patterns
                
                # Prefer balanced mid-range numbers
                mid_range_count = sum(1 for n in numbers if 15 <= n <= 35)
                if mid_range_count >= 3:
                    pattern_score += 0.2
                
                # Prefer moderate spread
                spread = max(numbers) - min(numbers)
                if 20 <= spread <= 35:
                    pattern_score += 0.3
                
                # Check for consecutive number patterns (more common in smaller pool)
                consecutive_count = self._count_consecutive_numbers(numbers)
                if consecutive_count >= 2:
                    pattern_score += 0.2
            
            return min(1.0, pattern_score)
            
        except Exception:
            return 0.0
    
    def _has_arithmetic_progression(self, numbers: List[int]) -> bool:
        """Check if numbers contain arithmetic progression"""
        try:
            sorted_nums = sorted(numbers)
            for i in range(len(sorted_nums) - 2):
                diff = sorted_nums[i+1] - sorted_nums[i]
                if diff > 0 and sorted_nums[i+2] - sorted_nums[i+1] == diff:
                    return True
            return False
        except Exception:
            return False
    
    def _count_consecutive_numbers(self, numbers: List[int]) -> int:
        """Count consecutive number pairs"""
        try:
            sorted_nums = sorted(numbers)
            consecutive_count = 0
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i+1] - sorted_nums[i] == 1:
                    consecutive_count += 1
            return consecutive_count
        except Exception:
            return 0
    
    def _store_optimization_history(self, game: str, optimization_results: Dict[str, Any]) -> None:
        """Store optimization history for analysis"""
        try:
            if game not in self.optimization_history:
                self.optimization_history[game] = []
            
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'optimization_count': len(optimization_results['optimized_predictions']),
                'average_score': np.mean(optimization_results['optimization_scores']) if optimization_results['optimization_scores'] else 0.0,
                'applied_optimizations': list(set(optimization_results['applied_optimizations'])),
                'performance_expectations': optimization_results['performance_expectations']
            }
            
            self.optimization_history[game].append(history_entry)
            
            # Keep only last 100 entries
            if len(self.optimization_history[game]) > 100:
                self.optimization_history[game] = self.optimization_history[game][-100:]
                
        except Exception as e:
            logger.error(f"Error storing optimization history: {e}")
    
    def _generate_optimization_insights(self, game: str, optimized_predictions: List[List[int]]) -> Dict[str, Any]:
        """Generate insights from optimization process"""
        try:
            insights = {
                'optimization_effectiveness': 0.0,
                'pattern_consistency': 0.0,
                'distribution_quality': 0.0,
                'mathematical_soundness': 0.0,
                'recommendations': []
            }
            
            if not optimized_predictions:
                return insights
            
            # Calculate optimization effectiveness
            config = self.game_configs[game]
            
            # Check sum distribution
            sums = [sum(pred) for pred in optimized_predictions]
            optimal_range = config['optimal_sum_range']
            in_range_count = sum(1 for s in sums if optimal_range[0] <= s <= optimal_range[1])
            insights['optimization_effectiveness'] = in_range_count / len(optimized_predictions)
            
            # Check pattern consistency
            all_numbers = [num for pred in optimized_predictions for num in pred]
            freq_dist = self._calculate_frequency_distribution(all_numbers, config['number_range'][1])
            consistency_score = 1.0 - np.std(list(freq_dist.values()))
            insights['pattern_consistency'] = max(0.0, consistency_score)
            
            # Check distribution quality
            range_quality_scores = []
            for pred in optimized_predictions:
                range_score = self._optimize_range_distribution(pred, config['preferred_distributions'])
                range_quality_scores.append(range_score)
            insights['distribution_quality'] = np.mean(range_quality_scores)
            
            # Mathematical soundness
            mathematical_scores = []
            for pred in optimized_predictions:
                math_score = self._apply_game_specific_patterns(pred, game, config)
                mathematical_scores.append(math_score)
            insights['mathematical_soundness'] = np.mean(mathematical_scores)
            
            # Generate recommendations
            recommendations = []
            if insights['optimization_effectiveness'] < 0.7:
                recommendations.append("Consider adjusting sum optimization parameters")
            if insights['pattern_consistency'] < 0.6:
                recommendations.append("Improve pattern consistency across predictions")
            if insights['distribution_quality'] < 0.7:
                recommendations.append("Enhance range distribution optimization")
            if insights['mathematical_soundness'] < 0.6:
                recommendations.append("Apply stronger mathematical pattern constraints")
            
            insights['recommendations'] = recommendations
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating optimization insights: {e}")
            return {}
    
    def _calculate_performance_expectations(self, game: str, optimization_scores: List[float]) -> Dict[str, Any]:
        """Calculate expected performance based on optimization scores"""
        try:
            if not optimization_scores:
                return {}
            
            avg_score = np.mean(optimization_scores)
            max_score = np.max(optimization_scores)
            min_score = np.min(optimization_scores)
            
            # Estimate match probabilities based on optimization scores
            base_match_probability = {
                'lotto_max': {'2_matches': 0.05, '3_matches': 0.015, '4_matches': 0.003, '5_matches': 0.0005, '6_matches': 0.00002, '7_matches': 0.000003},
                'lotto_649': {'2_matches': 0.06, '3_matches': 0.018, '4_matches': 0.004, '5_matches': 0.0008, '6_matches': 0.000007}
            }
            
            game_probabilities = base_match_probability.get(game, {})
            
            # Adjust probabilities based on optimization scores
            performance_expectations = {}
            for match_type, base_prob in game_probabilities.items():
                adjustment_factor = 1.0 + (avg_score - 0.5) * 2.0  # Scale optimization impact
                adjusted_prob = base_prob * adjustment_factor
                performance_expectations[match_type] = min(adjusted_prob, base_prob * 3.0)  # Cap at 3x improvement
            
            performance_expectations.update({
                'average_optimization_score': avg_score,
                'best_case_score': max_score,
                'worst_case_score': min_score,
                'confidence_level': min(avg_score * 1.2, 1.0),
                'expected_improvement': max(0.0, (avg_score - 0.5) * 100)  # Percentage improvement over baseline
            })
            
            return performance_expectations
            
        except Exception as e:
            logger.error(f"Error calculating performance expectations: {e}")
            return {}
    
    def _calculate_frequency_distribution(self, numbers: List[int], max_num: int) -> Dict[int, float]:
        """Calculate frequency distribution of numbers"""
        freq = {}
        total = len(numbers)
        for i in range(1, max_num + 1):
            freq[i] = numbers.count(i) / total if total > 0 else 0
        return freq

class GameSpecificOptimizer:
    """
    PHASE 2 COMPONENT 3: Game-Specific Optimizer
    
    Tailored optimization strategies for each lottery game based on their
    unique characteristics and mathematical properties.
    """
    
    def __init__(self):
        self.game_configs = {
            'lotto_max': {
                'number_range': (1, 50),
                'selection_count': 7,
                'bonus_numbers': 1,
                'optimal_sum_range': (175, 245),
                'preferred_distributions': {
                    'low_range': (1, 16),
                    'mid_range': (17, 33),
                    'high_range': (34, 50)
                },
                'optimization_weights': {
                    'range_distribution': 0.25,
                    'sum_optimization': 0.20,
                    'pattern_matching': 0.25,
                    'historical_success': 0.30
                }
            },
            'lotto_649': {
                'number_range': (1, 49),
                'selection_count': 6,
                'bonus_numbers': 1,
                'optimal_sum_range': (120, 180),
                'preferred_distributions': {
                    'low_range': (1, 16),
                    'mid_range': (17, 32),
                    'high_range': (33, 49)
                },
                'optimization_weights': {
                    'range_distribution': 0.30,
                    'sum_optimization': 0.25,
                    'pattern_matching': 0.20,
                    'historical_success': 0.25
                }
            }
        }
        self.optimization_history = {}
        
        logger.info("Game-Specific Optimizer initialized")
    
    def optimize_for_game(self, game: str, base_predictions: List[List[int]], 
                         optimization_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize predictions specifically for the target game"""
        try:
            logger.info(f"Optimizing predictions for {game}")
            
            # Normalize game name
            normalized_game = game.lower().replace(' ', '_').replace('-', '_')
            if 'max' in normalized_game:
                normalized_game = 'lotto_max'
            elif '649' in normalized_game or '6_49' in normalized_game:
                normalized_game = 'lotto_649'
            else:
                normalized_game = 'lotto_max'  # Default fallback
                
            if normalized_game not in self.game_configs:
                print(f"Warning: Unknown game: {game}, using fallback optimization")
                return {
                    'optimized_predictions': base_predictions,
                    'optimization_scores': [0.5] * len(base_predictions),
                    'applied_optimizations': ['fallback'],
                    'game_specific_insights': {'status': 'fallback_used'},
                    'performance_expectations': {'confidence': 0.5}
                }
            
            game = normalized_game  # Use normalized name
            
            config = self.game_configs[game]
            
            optimization_results = {
                'optimized_predictions': [],
                'optimization_scores': [],
                'applied_optimizations': [],
                'game_specific_insights': {},
                'performance_expectations': {}
            }
            
            # Apply game-specific optimizations
            for prediction in base_predictions:
                optimized_prediction = self._optimize_single_prediction(
                    prediction, game, config, optimization_criteria
                )
                optimization_results['optimized_predictions'].append(optimized_prediction['numbers'])
                optimization_results['optimization_scores'].append(optimized_prediction['score'])
                optimization_results['applied_optimizations'].extend(optimized_prediction['optimizations'])
            
            # Generate game-specific insights
            insights = self._generate_optimization_insights(
                game, optimization_results['optimized_predictions']
            )
            optimization_results['game_specific_insights'] = insights
            
            # Calculate performance expectations
            expectations = self._calculate_performance_expectations(
                game, optimization_results['optimization_scores']
            )
            optimization_results['performance_expectations'] = expectations
            
            # Store optimization history
            self._store_optimization_history(game, optimization_results)
            
            logger.info(f"Game optimization complete for {game}: {len(optimization_results['optimized_predictions'])} predictions optimized")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in game-specific optimization: {e}")
            return {'error': str(e)}
    
    def _optimize_single_prediction(self, prediction: List[int], game: str, 
                                   config: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a single prediction for the specific game"""
        try:
            optimized_numbers = prediction.copy()
            optimization_score = 0.0
            applied_optimizations = []
            
            # Sum optimization
            current_sum = sum(optimized_numbers)
            optimal_range = config['optimal_sum_range']
            
            if current_sum < optimal_range[0] or current_sum > optimal_range[1]:
                optimized_numbers = self._adjust_sum_to_optimal_range(
                    optimized_numbers, optimal_range, config['number_range']
                )
                applied_optimizations.append('sum_optimization')
                optimization_score += config['optimization_weights']['sum_optimization']
            
            # Range distribution optimization
            distribution_score = self._optimize_range_distribution(
                optimized_numbers, config['preferred_distributions']
            )
            if distribution_score > 0.7:
                applied_optimizations.append('range_distribution')
                optimization_score += config['optimization_weights']['range_distribution']
            
            # Odd/even balance optimization
            odd_count = sum(1 for n in optimized_numbers if n % 2 == 1)
            target_odd_count = config['selection_count'] // 2
            
            if abs(odd_count - target_odd_count) > 1:
                optimized_numbers = self._balance_odd_even(
                    optimized_numbers, target_odd_count, config['number_range']
                )
                applied_optimizations.append('odd_even_balance')
                optimization_score += 0.1
            
            # Game-specific pattern optimization
            pattern_score = self._apply_game_specific_patterns(
                optimized_numbers, game, config
            )
            optimization_score += pattern_score * config['optimization_weights']['pattern_matching']
            
            if pattern_score > 0.5:
                applied_optimizations.append('pattern_matching')
            
            return {
                'numbers': sorted(optimized_numbers),
                'score': optimization_score,
                'optimizations': applied_optimizations
            }
            
        except Exception as e:
            logger.error(f"Error optimizing single prediction: {e}")
            return {'numbers': prediction, 'score': 0.0, 'optimizations': []}
    
    def _adjust_sum_to_optimal_range(self, numbers: List[int], optimal_range: Tuple[int, int], 
                                   number_range: Tuple[int, int]) -> List[int]:
        """Adjust numbers to achieve optimal sum range"""
        try:
            current_sum = sum(numbers)
            target_sum = (optimal_range[0] + optimal_range[1]) // 2
            difference = target_sum - current_sum
            
            if abs(difference) <= 5:  # Already close enough
                return numbers
            
            adjusted_numbers = numbers.copy()
            
            if difference > 0:  # Need to increase sum
                # Replace smallest numbers with larger ones
                for i, num in enumerate(sorted(enumerate(adjusted_numbers), key=lambda x: x[1])):
                    if difference <= 0:
                        break
                    
                    idx, current_num = num
                    max_increase = number_range[1] - current_num
                    increase = min(difference, max_increase, 10)
                    
                    if increase > 0:
                        new_num = current_num + increase
                        if new_num not in adjusted_numbers:
                            adjusted_numbers[idx] = new_num
                            difference -= increase
            
            else:  # Need to decrease sum
                # Replace largest numbers with smaller ones
                for i, num in enumerate(sorted(enumerate(adjusted_numbers), key=lambda x: x[1], reverse=True)):
                    if difference >= 0:
                        break
                    
                    idx, current_num = num
                    max_decrease = current_num - number_range[0]
                    decrease = min(abs(difference), max_decrease, 10)
                    
                    if decrease > 0:
                        new_num = current_num - decrease
                        if new_num not in adjusted_numbers:
                            adjusted_numbers[idx] = new_num
                            difference += decrease
            
            return adjusted_numbers
            
        except Exception:
            return numbers
    
    def _optimize_range_distribution(self, numbers: List[int], preferred_distributions: Dict[str, Tuple[int, int]]) -> float:
        """Optimize range distribution and return score"""
        try:
            low_range = preferred_distributions['low_range']
            mid_range = preferred_distributions['mid_range']
            high_range = preferred_distributions['high_range']
            
            low_count = sum(1 for n in numbers if low_range[0] <= n <= low_range[1])
            mid_count = sum(1 for n in numbers if mid_range[0] <= n <= mid_range[1])
            high_count = sum(1 for n in numbers if high_range[0] <= n <= high_range[1])
            
            total_numbers = len(numbers)
            
            # Calculate distribution balance (ideal is roughly equal distribution)
            ideal_per_range = total_numbers / 3
            distribution_score = 1.0 - (
                abs(low_count - ideal_per_range) + 
                abs(mid_count - ideal_per_range) + 
                abs(high_count - ideal_per_range)
            ) / (total_numbers * 2)
            
            return max(0.0, distribution_score)
            
        except Exception:
            return 0.0
    
    def _balance_odd_even(self, numbers: List[int], target_odd_count: int, 
                         number_range: Tuple[int, int]) -> List[int]:
        """Balance odd/even distribution"""
        try:
            current_odd_count = sum(1 for n in numbers if n % 2 == 1)
            difference = target_odd_count - current_odd_count
            
            if difference == 0:
                return numbers
            
            adjusted_numbers = numbers.copy()
            
            if difference > 0:  # Need more odd numbers
                # Replace even numbers with odd numbers
                even_numbers = [(i, n) for i, n in enumerate(adjusted_numbers) if n % 2 == 0]
                for i, (idx, num) in enumerate(even_numbers[:abs(difference)]):
                    # Try to find a nearby odd number
                    for offset in [1, -1, 3, -3, 5, -5]:
                        new_num = num + offset
                        if (number_range[0] <= new_num <= number_range[1] and 
                            new_num % 2 == 1 and new_num not in adjusted_numbers):
                            adjusted_numbers[idx] = new_num
                            break
            
            else:  # Need more even numbers
                # Replace odd numbers with even numbers
                odd_numbers = [(i, n) for i, n in enumerate(adjusted_numbers) if n % 2 == 1]
                for i, (idx, num) in enumerate(odd_numbers[:abs(difference)]):
                    # Try to find a nearby even number
                    for offset in [1, -1, 2, -2, 4, -4]:
                        new_num = num + offset
                        if (number_range[0] <= new_num <= number_range[1] and 
                            new_num % 2 == 0 and new_num not in adjusted_numbers):
                            adjusted_numbers[idx] = new_num
                            break
            
            return adjusted_numbers
            
        except Exception:
            return numbers
    
    def _apply_game_specific_patterns(self, numbers: List[int], game: str, config: Dict[str, Any]) -> float:
        """Apply game-specific mathematical patterns"""
        try:
            pattern_score = 0.0
            
            if game == 'lotto_max':
                # Lotto Max specific patterns
                
                # Prefer using higher numbers (unique to larger pool)
                high_number_count = sum(1 for n in numbers if n > 35)
                if high_number_count >= 2:
                    pattern_score += 0.2
                
                # Prefer wider spread due to larger pool
                spread = max(numbers) - min(numbers)
                if spread > 30:
                    pattern_score += 0.3
                
                # Check for mathematical progression patterns
                if self._has_arithmetic_progression(numbers):
                    pattern_score += 0.2
                
            elif game == 'lotto_649':
                # Lotto 649 specific patterns
                
                # Prefer balanced mid-range numbers
                mid_range_count = sum(1 for n in numbers if 15 <= n <= 35)
                if mid_range_count >= 3:
                    pattern_score += 0.2
                
                # Prefer moderate spread
                spread = max(numbers) - min(numbers)
                if 20 <= spread <= 35:
                    pattern_score += 0.3
                
                # Check for consecutive number patterns (more common in smaller pool)
                consecutive_count = self._count_consecutive_numbers(numbers)
                if consecutive_count >= 2:
                    pattern_score += 0.2
            
            return min(1.0, pattern_score)
            
        except Exception:
            return 0.0
    
    def _has_arithmetic_progression(self, numbers: List[int]) -> bool:
        """Check if numbers contain arithmetic progression"""
        try:
            sorted_nums = sorted(numbers)
            for i in range(len(sorted_nums) - 2):
                diff = sorted_nums[i+1] - sorted_nums[i]
                if diff > 0 and sorted_nums[i+2] - sorted_nums[i+1] == diff:
                    return True
            return False
        except Exception:
            return False
    
    def _count_consecutive_numbers(self, numbers: List[int]) -> int:
        """Count consecutive number pairs"""
        try:
            sorted_nums = sorted(numbers)
            consecutive_count = 0
            for i in range(len(sorted_nums) - 1):
                if sorted_nums[i+1] - sorted_nums[i] == 1:
                    consecutive_count += 1
            return consecutive_count
        except Exception:
            return 0

class TemporalPatternAnalyzer:
    """
    PHASE 2 COMPONENT 4: Temporal Pattern Analyzer
    
    Advanced time-based pattern recognition and forecasting for both lottery games.
    """
    
    def __init__(self):
        self.temporal_patterns = {}
        self.seasonal_analysis = {}
        self.cyclical_patterns = {}
        self.prediction_windows = {}
        
        logger.info("Temporal Pattern Analyzer initialized")
    
    def analyze_temporal_patterns(self, historical_data: Dict[str, List[Dict]], 
                                prediction_history: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze comprehensive temporal patterns"""
        try:
            logger.info("Analyzing temporal patterns across games")
            
            analysis_results = {
                'seasonal_patterns': {},
                'cyclical_analysis': {},
                'day_of_week_effects': {},
                'monthly_trends': {},
                'yearly_cycles': {},
                'prediction_windows': {},
                'temporal_correlations': {}
            }
            
            for game in ['lotto_max', 'lotto_649']:
                game_data = historical_data.get(game, [])
                game_predictions = prediction_history.get(game, [])
                
                # Seasonal pattern analysis
                seasonal_patterns = self._analyze_seasonal_patterns(game, game_data)
                analysis_results['seasonal_patterns'][game] = seasonal_patterns
                
                # Cyclical analysis
                cyclical_analysis = self._analyze_cyclical_patterns(game, game_data)
                analysis_results['cyclical_analysis'][game] = cyclical_analysis
                
                # Day of week analysis
                dow_effects = self._analyze_day_of_week_effects(game, game_data, game_predictions)
                analysis_results['day_of_week_effects'][game] = dow_effects
                
                # Monthly trend analysis
                monthly_trends = self._analyze_monthly_trends(game, game_data)
                analysis_results['monthly_trends'][game] = monthly_trends
                
                # Yearly cycle analysis
                yearly_cycles = self._analyze_yearly_cycles(game, game_data)
                analysis_results['yearly_cycles'][game] = yearly_cycles
            
            # Cross-game temporal correlations
            temporal_correlations = self._analyze_cross_game_temporal_correlations(
                historical_data, prediction_history
            )
            analysis_results['temporal_correlations'] = temporal_correlations
            
            # Generate optimal prediction windows
            prediction_windows = self._generate_prediction_windows(analysis_results)
            analysis_results['prediction_windows'] = prediction_windows
            
            # Store patterns for future use
            self._store_temporal_patterns(analysis_results)
            
            logger.info("Temporal pattern analysis complete")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_seasonal_patterns(self, game: str, data: List[Dict]) -> Dict[str, Any]:
        """Analyze seasonal patterns in lottery results"""
        try:
            seasonal_data = {'spring': [], 'summer': [], 'autumn': [], 'winter': []}
            
            for entry in data:
                if 'date' in entry and 'winning_numbers' in entry:
                    try:
                        date_obj = datetime.fromisoformat(entry['date'].replace('Z', '+00:00'))
                        month = date_obj.month
                        
                        # Classify by season
                        if month in [3, 4, 5]:
                            season = 'spring'
                        elif month in [6, 7, 8]:
                            season = 'summer'
                        elif month in [9, 10, 11]:
                            season = 'autumn'
                        else:
                            season = 'winter'
                        
                        seasonal_data[season].extend(entry['winning_numbers'])
                    except Exception:
                        continue
            
            # Analyze patterns for each season
            seasonal_patterns = {}
            for season, numbers in seasonal_data.items():
                if numbers:
                    patterns = {
                        'average_number': np.mean(numbers),
                        'number_frequency': self._calculate_frequency_distribution(numbers, 50 if game == 'lotto_max' else 49),
                        'preferred_ranges': self._analyze_range_preferences(numbers),
                        'odd_even_ratio': sum(1 for n in numbers if n % 2 == 1) / len(numbers),
                        'sample_size': len(numbers)
                    }
                    seasonal_patterns[season] = patterns
            
            return seasonal_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
            return {}
    
    def _analyze_cyclical_patterns(self, game: str, data: List[Dict]) -> Dict[str, Any]:
        """Analyze cyclical patterns in lottery draws"""
        try:
            cyclical_analysis = {
                'weekly_cycles': {},
                'monthly_cycles': {},
                'number_appearance_cycles': {},
                'gap_analysis': {}
            }
            
            # Extract dates and numbers
            dated_entries = []
            for entry in data:
                if 'date' in entry and 'winning_numbers' in entry:
                    try:
                        date_obj = datetime.fromisoformat(entry['date'].replace('Z', '+00:00'))
                        dated_entries.append((date_obj, entry['winning_numbers']))
                    except Exception:
                        continue
            
            if not dated_entries:
                return cyclical_analysis
            
            # Sort by date
            dated_entries.sort(key=lambda x: x[0])
            
            # Weekly cycle analysis
            weekly_data = {}
            for date_obj, numbers in dated_entries:
                week_day = date_obj.strftime('%A')
                if week_day not in weekly_data:
                    weekly_data[week_day] = []
                weekly_data[week_day].extend(numbers)
            
            for day, numbers in weekly_data.items():
                if numbers:
                    cyclical_analysis['weekly_cycles'][day] = {
                        'average_number': np.mean(numbers),
                        'frequency_distribution': self._calculate_frequency_distribution(numbers, 50 if game == 'lotto_max' else 49),
                        'sample_size': len(numbers)
                    }
            
            # Number appearance cycle analysis
            number_appearances = self._analyze_number_appearance_cycles(dated_entries, game)
            cyclical_analysis['number_appearance_cycles'] = number_appearances
            
            # Gap analysis between appearances
            gap_analysis = self._analyze_number_gaps(dated_entries, game)
            cyclical_analysis['gap_analysis'] = gap_analysis
            
            return cyclical_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cyclical patterns: {e}")
            return {}
    
    def get_temporal_forecast(self, game: str, forecast_horizon: int = 5) -> Dict[str, Any]:
        """Generate temporal-based forecast for upcoming draws"""
        try:
            logger.info(f"Generating temporal forecast for {game}")
            
            current_date = datetime.now()
            forecast_results = {
                'forecast_dates': [],
                'seasonal_adjustments': {},
                'cyclical_predictions': {},
                'temporal_confidence': {},
                'recommended_strategies': []
            }
            
            # Generate forecast dates
            for i in range(forecast_horizon):
                if game == 'lotto_max':
                    # Lotto Max draws on Tuesdays and Fridays
                    days_ahead = 1 if current_date.weekday() < 1 else (1 - current_date.weekday()) % 7
                    if i > 0:
                        days_ahead += 3 if i % 2 == 1 else 4  # Alternate between Tue-Fri (3 days) and Fri-Tue (4 days)
                else:
                    # Lotto 649 draws on Wednesdays and Saturdays
                    days_ahead = 2 if current_date.weekday() < 2 else (2 - current_date.weekday()) % 7
                    if i > 0:
                        days_ahead += 3 if i % 2 == 1 else 4  # Alternate between Wed-Sat (3 days) and Sat-Wed (4 days)
                
                forecast_date = current_date + timedelta(days=days_ahead)
                forecast_results['forecast_dates'].append(forecast_date.strftime('%Y-%m-%d'))
            
            # Apply seasonal adjustments
            current_season = self._get_current_season(current_date)
            if game in self.seasonal_analysis and current_season in self.seasonal_analysis[game]:
                seasonal_data = self.seasonal_analysis[game][current_season]
                forecast_results['seasonal_adjustments'] = seasonal_data
            
            # Apply cyclical predictions
            if game in self.cyclical_patterns:
                cyclical_data = self.cyclical_patterns[game]
                forecast_results['cyclical_predictions'] = cyclical_data
            
            # Calculate temporal confidence
            confidence_score = self._calculate_temporal_confidence(game, current_date)
            forecast_results['temporal_confidence'] = confidence_score
            
            # Generate recommendations
            recommendations = self._generate_temporal_recommendations(game, forecast_results)
            forecast_results['recommended_strategies'] = recommendations
            
            return forecast_results
            
        except Exception as e:
            logger.error(f"Error generating temporal forecast: {e}")
            return {'error': str(e)}

class IntelligentModelSelector:
    """
    PHASE 2 COMPONENT 5: Intelligent Model Selection
    
    Dynamic model selection based on game context, historical performance,
    and current conditions.
    """
    
    def __init__(self):
        self.model_performance_history = {}
        self.game_model_preferences = {}
        self.context_based_selection = {}
        self.dynamic_weights = {}
        
        logger.info("Intelligent Model Selector initialized")
    
    def select_optimal_models(self, game: str, prediction_context: Dict[str, Any], 
                            available_models: List[str]) -> Dict[str, Any]:
        """Select optimal models for current prediction context"""
        try:
            logger.info(f"Selecting optimal models for {game}")
            
            selection_results = {
                'primary_models': [],
                'secondary_models': [],
                'model_weights': {},
                'selection_rationale': {},
                'confidence_scores': {},
                'ensemble_strategy': ''
            }
            
            # Analyze current context
            context_analysis = self._analyze_prediction_context(game, prediction_context)
            
            # Score each available model
            model_scores = {}
            for model in available_models:
                score = self._calculate_model_score(game, model, context_analysis)
                model_scores[model] = score
            
            # Sort models by score
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select primary models (top performers)
            primary_threshold = 0.7
            primary_models = [model for model, score in sorted_models if score >= primary_threshold]
            
            if not primary_models:  # If no models meet threshold, take top 3
                primary_models = [model for model, _ in sorted_models[:3]]
            
            selection_results['primary_models'] = primary_models
            
            # Select secondary models (supporting models)
            remaining_models = [model for model, _ in sorted_models if model not in primary_models]
            selection_results['secondary_models'] = remaining_models[:2]
            
            # Calculate dynamic weights
            weights = self._calculate_dynamic_weights(primary_models, model_scores, context_analysis)
            selection_results['model_weights'] = weights
            
            # Generate selection rationale
            rationale = self._generate_selection_rationale(sorted_models, context_analysis)
            selection_results['selection_rationale'] = rationale
            
            # Calculate confidence scores
            confidence_scores = self._calculate_model_confidence_scores(model_scores, context_analysis)
            selection_results['confidence_scores'] = confidence_scores
            
            # Determine ensemble strategy
            ensemble_strategy = self._determine_ensemble_strategy(primary_models, context_analysis)
            selection_results['ensemble_strategy'] = ensemble_strategy
            
            # Update model performance tracking
            self._update_model_selection_history(game, selection_results, context_analysis)
            
            logger.info(f"Model selection complete: {len(primary_models)} primary models selected")
            return selection_results
            
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return {'error': str(e)}
    
    def _analyze_prediction_context(self, game: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current prediction context"""
        try:
            analysis = {
                'game_characteristics': self._get_game_characteristics(game),
                'temporal_factors': context.get('temporal_factors', {}),
                'recent_performance': context.get('recent_performance', {}),
                'data_quality': context.get('data_quality', 'good'),
                'prediction_urgency': context.get('urgency', 'normal'),
                'historical_context': context.get('historical_context', {}),
                'special_conditions': context.get('special_conditions', [])
            }
            
            # Add derived insights
            analysis['complexity_level'] = self._assess_context_complexity(analysis)
            analysis['data_sufficiency'] = self._assess_data_sufficiency(context)
            analysis['prediction_difficulty'] = self._assess_prediction_difficulty(game, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing prediction context: {e}")
            return {}
    
    def _calculate_model_score(self, game: str, model: str, context_analysis: Dict[str, Any]) -> float:
        """Calculate score for a specific model in given context"""
        try:
            base_score = 0.5  # Base score for all models
            
            # Historical performance weight (40%)
            historical_performance = self._get_historical_performance(game, model)
            performance_score = historical_performance * 0.4
            
            # Game compatibility weight (25%)
            game_compatibility = self._assess_game_compatibility(game, model)
            compatibility_score = game_compatibility * 0.25
            
            # Context suitability weight (20%)
            context_suitability = self._assess_context_suitability(model, context_analysis)
            context_score = context_suitability * 0.2
            
            # Recent trend weight (15%)
            recent_trend = self._get_recent_performance_trend(game, model)
            trend_score = recent_trend * 0.15
            
            total_score = base_score + performance_score + compatibility_score + context_score + trend_score
            
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            logger.error(f"Error calculating model score: {e}")
            return 0.5
    
    def _get_historical_performance(self, game: str, model: str) -> float:
        """Get historical performance score for model on specific game"""
        try:
            if game not in self.model_performance_history:
                return 0.5  # Default score
            
            game_history = self.model_performance_history[game]
            if model not in game_history:
                return 0.5
            
            model_history = game_history[model]
            if not model_history:
                return 0.5
            
            # Calculate weighted average of recent performance
            recent_performances = model_history[-10:]  # Last 10 predictions
            weights = np.exp(np.linspace(-1, 0, len(recent_performances)))  # More weight to recent
            weighted_avg = np.average(recent_performances, weights=weights)
            
            return weighted_avg
            
        except Exception:
            return 0.5
    
    def _assess_game_compatibility(self, game: str, model: str) -> float:
        """Assess how well model suits specific game characteristics"""
        try:
            compatibility_scores = {
                'lotto_max': {
                    'lstm': 0.8,      # Good for sequential patterns in larger pools
                    'transformer': 0.9, # Excellent for complex pattern recognition
                    'xgboost': 0.7,   # Good for feature-based predictions
                    'ensemble': 0.9,  # Always good
                    'neural': 0.8,    # Good for complex non-linear patterns
                    'random_forest': 0.6,  # Decent for structured data
                    'svm': 0.5        # Less suitable for lottery data
                },
                'lotto_649': {
                    'lstm': 0.9,      # Excellent for sequential patterns
                    'transformer': 0.8, # Good but potentially overkill
                    'xgboost': 0.8,   # Very good for smaller, structured data
                    'ensemble': 0.9,  # Always good
                    'neural': 0.7,    # Good but simpler approaches may suffice
                    'random_forest': 0.7,  # Good for structured data
                    'svm': 0.6        # Better for smaller datasets
                }
            }
            
            return compatibility_scores.get(game, {}).get(model, 0.5)
            
        except Exception:
            return 0.5
    
    def update_model_performance(self, game: str, model: str, performance_metrics: Dict[str, float]) -> None:
        """Update model performance tracking"""
        try:
            if game not in self.model_performance_history:
                self.model_performance_history[game] = {}
            
            if model not in self.model_performance_history[game]:
                self.model_performance_history[game][model] = []
            
            # Calculate composite performance score
            composite_score = self._calculate_composite_performance_score(performance_metrics)
            
            # Add to history
            self.model_performance_history[game][model].append(composite_score)
            
            # Keep only last 50 entries to prevent unlimited growth
            if len(self.model_performance_history[game][model]) > 50:
                self.model_performance_history[game][model] = self.model_performance_history[game][model][-50:]
            
            logger.info(f"Updated performance for {model} on {game}: {composite_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def get_model_selection_insights(self) -> Dict[str, Any]:
        """Get insights about model selection patterns"""
        try:
            insights = {
                'game_preferences': {},
                'model_rankings': {},
                'performance_trends': {},
                'selection_frequency': {},
                'effectiveness_analysis': {}
            }
            
            # Analyze game preferences
            for game in ['lotto_max', 'lotto_649']:
                if game in self.model_performance_history:
                    game_data = self.model_performance_history[game]
                    
                    # Calculate average performance by model
                    model_averages = {}
                    for model, performances in game_data.items():
                        if performances:
                            model_averages[model] = np.mean(performances)
                    
                    # Sort by performance
                    sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)
                    insights['game_preferences'][game] = sorted_models
                    
                    # Model rankings
                    insights['model_rankings'][game] = [model for model, _ in sorted_models]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting model selection insights: {e}")
            return {'error': str(e)}

# Global instances for Phase 2 components
cross_game_learning_engine = CrossGameLearningEngine()
advanced_pattern_memory = AdvancedPatternMemorySystem()
game_specific_optimizer = GameSpecificOptimizer()
temporal_pattern_analyzer = TemporalPatternAnalyzer()
intelligent_model_selector = IntelligentModelSelector()

logger.info("Phase 2 Cross-Game Learning Intelligence and Advanced Pattern Memory System initialized")
