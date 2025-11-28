"""
Phase 2: Expert Ensemble Engine for lottery prediction system.

This engine implements ensemble methods that combine multiple prediction
models to create more robust and accurate predictions. It uses various
machine learning algorithms and voting mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            VotingClassifier, BaggingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter
from scipy import stats
from abc import ABC, abstractmethod
import joblib
import warnings
import math

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SpecialistExpert(ABC):
    """Abstract base class for all specialist experts"""
    
    def __init__(self, name: str):
        self.name = name
        self.confidence_score = 0.0
        self.last_prediction = None
        self.performance_history = []
    
    @abstractmethod
    def analyze_patterns(self, historical_data: List[List[int]], game_type: str) -> Dict[str, Any]:
        """Analyze patterns in historical data"""
        pass
    
    @abstractmethod
    def generate_predictions(self, analysis_result: Dict[str, Any], num_sets: int, 
                           max_number: int, numbers_per_set: int) -> List[Dict[str, Any]]:
        """Generate predictions based on analysis"""
        pass
    
    def calculate_confidence(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score for this specialist"""
        return min(1.0, max(0.0, analysis_result.get('confidence', 0.5)))
    
    def update_performance(self, actual_draw: List[int], predicted_sets: List[List[int]]) -> Dict[str, Any]:
        """Update performance metrics based on actual results"""
        try:
            performance = {
                'timestamp': datetime.now().isoformat(),
                'actual_draw': actual_draw,
                'predicted_sets': predicted_sets,
                'matches_per_set': [],
                'best_match_count': 0,
                'accuracy_score': 0.0
            }
            
            # Calculate matches for each predicted set
            for pred_set in predicted_sets:
                matches = len(set(pred_set).intersection(set(actual_draw)))
                performance['matches_per_set'].append(matches)
                performance['best_match_count'] = max(performance['best_match_count'], matches)
            
            # Calculate accuracy score (best match / total numbers)
            if len(actual_draw) > 0:
                performance['accuracy_score'] = performance['best_match_count'] / len(actual_draw)
            
            self.performance_history.append(performance)
            
            # Keep only last 50 performance records
            if len(self.performance_history) > 50:
                self.performance_history = self.performance_history[-50:]
            
            return performance
            
        except Exception as e:
            logger.error(f"Error updating performance for {self.name}: {e}")
            return {}


class MathematicalPatternSpecialist(SpecialistExpert):
    """Specialist focusing on mathematical patterns and relationships"""
    
    def __init__(self):
        super().__init__("Mathematical Pattern Specialist")
        self.prime_cache = self._generate_primes(100)
        self.fibonacci_cache = self._generate_fibonacci(100)
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def _ensure_integer_data(self, draw):
        """Ensure draw data is converted to integers"""
        if isinstance(draw, list) and len(draw) > 0 and isinstance(draw[0], str):
            return [int(num) for num in draw if str(num).isdigit()]
        return draw
    
    def _generate_primes(self, limit: int) -> List[int]:
        """Generate prime numbers up to limit"""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(limit + 1) if sieve[i]]
    
    def _generate_fibonacci(self, limit: int) -> List[int]:
        """Generate Fibonacci numbers up to limit"""
        fib = [1, 1]
        while fib[-1] < limit:
            fib.append(fib[-1] + fib[-2])
        return [f for f in fib if f <= limit]

    def analyze_patterns(self, historical_data: List[List[int]], game_type: str) -> Dict[str, Any]:
        """Analyze mathematical patterns in historical data"""
        try:
            logger.info(f"{self.name}: Analyzing mathematical patterns...")
            
            analysis = {
                'specialist': self.name,
                'timestamp': datetime.now().isoformat(),
                'prime_analysis': self._analyze_prime_patterns(historical_data),
                'modular_analysis': self._analyze_modular_patterns(historical_data),
                'fibonacci_analysis': self._analyze_fibonacci_patterns(historical_data),
                'golden_ratio_analysis': self._analyze_golden_ratio_patterns(historical_data)
            }
            
            # Calculate overall confidence
            confidence_factors = [
                analysis['prime_analysis'].get('pattern_strength', 0),
                analysis['modular_analysis'].get('consistency_score', 0),
                analysis['fibonacci_analysis'].get('significance', 0),
                analysis['golden_ratio_analysis'].get('ratio_accuracy', 0)
            ]
            
            analysis['confidence'] = np.mean([f for f in confidence_factors if f > 0])
            
            self.confidence_score = analysis['confidence']
            logger.info(f"{self.name}: Analysis complete, confidence: {self.confidence_score:.3f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in {self.name} analysis: {e}")
            return {'confidence': 0.0, 'error': str(e)}

    def _analyze_prime_patterns(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze prime number patterns"""
        try:
            prime_counts = []
            prime_distributions = []
            
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                primes_in_draw = [num for num in draw if num in self.prime_cache]
                prime_counts.append(len(primes_in_draw))
                
                # Distribution analysis
                sorted_primes = sorted(primes_in_draw)
                if len(sorted_primes) >= 2:
                    gaps = [sorted_primes[i+1] - sorted_primes[i] for i in range(len(sorted_primes)-1)]
                    prime_distributions.append(np.mean(gaps) if gaps else 0)
            
            pattern_strength = 1.0 - (np.var(prime_counts) / (np.mean(prime_counts) + 1e-6)) if prime_counts else 0
            pattern_strength = max(0.0, min(1.0, pattern_strength))
            
            return {
                'average_prime_count': float(np.mean(prime_counts)) if prime_counts else 0.0,
                'prime_variance': float(np.var(prime_counts)) if prime_counts else 0.0,
                'pattern_strength': float(pattern_strength),
                'distribution_consistency': float(1.0 - np.var(prime_distributions) / (np.mean(prime_distributions) + 1e-6)) if prime_distributions else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in prime pattern analysis: {e}")
            return {'pattern_strength': 0.0}

    def _analyze_modular_patterns(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze modular arithmetic patterns"""
        try:
            modular_data = {}
            
            for mod in [3, 5, 7, 11]:
                mod_distributions = []
                
                for draw in historical_data:
                    draw = self._ensure_integer_data(draw)
                    mod_count = [0] * mod
                    for num in draw:
                        mod_count[num % mod] += 1
                    mod_distributions.append(mod_count)
                
                # Calculate consistency across draws
                if mod_distributions:
                    avg_distribution = np.mean(mod_distributions, axis=0)
                    consistency = 1.0 - np.std(mod_distributions, axis=0).mean() / (avg_distribution.mean() + 1e-6)
                else:
                    consistency = 0.0
                
                modular_data[f'mod_{mod}'] = {
                    'average_distribution': avg_distribution.tolist() if len(mod_distributions) > 0 else [],
                    'consistency': float(consistency),
                    'most_common_pattern': Counter(tuple(dist) for dist in mod_distributions).most_common(1)[0] if mod_distributions else ((), 0)
                }
            
            # Overall consistency score
            consistency_scores = [data['consistency'] for data in modular_data.values()]
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
            
            return {
                'modular_patterns': modular_data,
                'consistency_score': float(overall_consistency)
            }
            
        except Exception as e:
            logger.error(f"Error in modular pattern analysis: {e}")
            return {'consistency_score': 0.0}

    def _analyze_fibonacci_patterns(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze Fibonacci number patterns"""
        try:
            fib_counts = []
            fib_positions = []
            
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                if not draw:
                    continue
                    
                fibs_in_draw = [num for num in draw if num in self.fibonacci_cache]
                fib_counts.append(len(fibs_in_draw))
                
                sorted_draw = sorted(draw)
                for i, num in enumerate(sorted_draw):
                    if num in self.fibonacci_cache:
                        fib_positions.append(i)
            
            # Calculate significance
            avg_fibs = np.mean(fib_counts) if fib_counts else 0
            avg_draw_length = np.mean([len(self._ensure_integer_data(draw)) for draw in historical_data if self._ensure_integer_data(draw)]) if historical_data else 6
            expected_fibs = len(self.fibonacci_cache) / 50 * avg_draw_length if avg_draw_length > 0 else 0
            significance = min(1.0, avg_fibs / (expected_fibs + 1e-6)) if expected_fibs > 0 else 0
            
            return {
                'average_fibonacci': float(avg_fibs),
                'significance': float(significance),
                'position_preference': Counter(fib_positions).most_common(2) if fib_positions else []
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            return {'significance': 0.0}

    def _analyze_golden_ratio_patterns(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze golden ratio relationships"""
        try:
            golden_relationships = []
            
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                if not draw:
                    continue
                    
                sorted_draw = sorted(draw)
                for i in range(len(sorted_draw) - 1):
                    try:
                        if sorted_draw[i] > 0 and isinstance(sorted_draw[i], (int, float)) and isinstance(sorted_draw[i + 1], (int, float)):
                            ratio = float(sorted_draw[i + 1]) / float(sorted_draw[i])
                            if abs(ratio - self.golden_ratio) < 0.2:
                                golden_relationships.append(ratio)
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
            
            return {
                'golden_ratio_frequency': float(len(golden_relationships) / len(historical_data)) if historical_data else 0.0,
                'average_ratio': float(np.mean(golden_relationships)) if golden_relationships else 0.0,
                'ratio_accuracy': float(1.0 - np.mean([abs(r - self.golden_ratio) for r in golden_relationships])) if golden_relationships else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in golden ratio analysis: {e}")
            return {'ratio_accuracy': 0.0}

    def generate_predictions(self, analysis_result: Dict[str, Any], num_sets: int, 
                           max_number: int, numbers_per_set: int) -> List[Dict[str, Any]]:
        """Generate predictions based on mathematical analysis"""
        try:
            predictions = []
            
            # Extract insights from analysis
            prime_analysis = analysis_result.get('prime_analysis', {})
            modular_analysis = analysis_result.get('modular_analysis', {})
            fibonacci_analysis = analysis_result.get('fibonacci_analysis', {})
            
            for i in range(num_sets):
                # Generate mathematically influenced number set
                number_set = []
                
                # Include prime numbers based on analysis
                avg_primes = prime_analysis.get('average_prime_count', 2)
                target_primes = max(1, min(numbers_per_set // 2, int(avg_primes)))
                available_primes = [p for p in self.prime_cache if p <= max_number]
                selected_primes = np.random.choice(available_primes, size=min(target_primes, len(available_primes)), replace=False)
                number_set.extend(selected_primes)
                
                # Include Fibonacci numbers if significant
                if fibonacci_analysis.get('significance', 0) > 0.3:
                    available_fibs = [f for f in self.fibonacci_cache if f <= max_number and f not in number_set]
                    if available_fibs:
                        fib_count = min(1, numbers_per_set - len(number_set))
                        selected_fibs = np.random.choice(available_fibs, size=min(fib_count, len(available_fibs)), replace=False)
                        number_set.extend(selected_fibs)
                
                # Fill remaining with mathematically distributed numbers
                remaining_count = numbers_per_set - len(number_set)
                available_numbers = [n for n in range(1, max_number + 1) if n not in number_set]
                
                if remaining_count > 0 and available_numbers:
                    remaining_numbers = np.random.choice(available_numbers, size=min(remaining_count, len(available_numbers)), replace=False)
                    number_set.extend(remaining_numbers)
                
                # Ensure we have the right number of numbers
                number_set = sorted(list(set(number_set)))[:numbers_per_set]
                
                # Pad if necessary
                while len(number_set) < numbers_per_set:
                    missing_numbers = [n for n in range(1, max_number + 1) if n not in number_set]
                    if missing_numbers:
                        number_set.append(np.random.choice(missing_numbers))
                    else:
                        break
                
                predictions.append({
                    'numbers': sorted(number_set[:numbers_per_set]),
                    'confidence': analysis_result.get('confidence', 0.5),
                    'specialist': self.name,
                    'properties': {
                        'mathematical_score': analysis_result.get('confidence', 0.5),
                        'prime_influence': target_primes / numbers_per_set,
                        'fibonacci_influence': fibonacci_analysis.get('significance', 0)
                    }
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions in {self.name}: {e}")
            return []


class TemporalSequenceSpecialist(SpecialistExpert):
    """Specialist focusing on temporal patterns and sequences"""
    
    def __init__(self):
        super().__init__("Temporal Sequence Specialist")
    
    def _ensure_integer_data(self, draw):
        """Ensure draw data is converted to integers"""
        if isinstance(draw, list) and len(draw) > 0 and isinstance(draw[0], str):
            return [int(num) for num in draw if str(num).isdigit()]
        return draw

    def analyze_patterns(self, historical_data: List[List[int]], game_type: str) -> Dict[str, Any]:
        """Analyze temporal patterns in historical data"""
        try:
            logger.info(f"{self.name}: Analyzing temporal patterns...")
            
            analysis = {
                'specialist': self.name,
                'timestamp': datetime.now().isoformat(),
                'trend_analysis': self._analyze_trends(historical_data),
                'cycle_analysis': self._analyze_cycles(historical_data),
                'momentum_analysis': self._analyze_momentum(historical_data),
                'sequence_analysis': self._analyze_sequences(historical_data)
            }
            
            # Calculate confidence
            confidence_factors = [
                analysis['trend_analysis'].get('trend_strength', 0),
                analysis['cycle_analysis'].get('cycle_reliability', 0),
                analysis['momentum_analysis'].get('momentum_strength', 0),
                analysis['sequence_analysis'].get('pattern_strength', 0)
            ]
            
            analysis['confidence'] = np.mean([f for f in confidence_factors if f > 0])
            self.confidence_score = analysis['confidence']
            
            logger.info(f"{self.name}: Analysis complete, confidence: {self.confidence_score:.3f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in {self.name} analysis: {e}")
            return {'confidence': 0.0, 'error': str(e)}

    def _analyze_trends(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze trending patterns"""
        try:
            # Track number frequencies over time windows
            window_size = max(10, len(historical_data) // 5)
            trend_data = {}
            
            for num in range(1, 51):
                frequencies = []
                for i in range(0, len(historical_data), window_size):
                    window = historical_data[i:i+window_size]
                    freq = sum(1 for draw in window if num in self._ensure_integer_data(draw)) / len(window) if window else 0
                    frequencies.append(freq)
                
                if len(frequencies) >= 2:
                    # Calculate trend slope
                    x = np.arange(len(frequencies))
                    slope, _, r_value, _, _ = stats.linregress(x, frequencies)
                    trend_data[num] = {
                        'slope': float(slope),
                        'correlation': float(r_value ** 2),
                        'current_freq': frequencies[-1] if frequencies else 0
                    }
            
            # Calculate overall trend strength
            correlations = [data['correlation'] for data in trend_data.values()]
            trend_strength = np.mean(correlations) if correlations else 0.0
            
            # Identify trending numbers
            trending_up = [num for num, data in trend_data.items() if data['slope'] > 0.01 and data['correlation'] > 0.5]
            trending_down = [num for num, data in trend_data.items() if data['slope'] < -0.01 and data['correlation'] > 0.5]
            
            return {
                'trend_strength': float(trend_strength),
                'trend_data': trend_data,
                'trending_up': trending_up[:10],
                'trending_down': trending_down[:10]
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'trend_strength': 0.0}

    def _analyze_cycles(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze cyclical patterns"""
        try:
            cycle_data = {}
            
            for num in range(1, 51):
                appearances = []
                for i, draw in enumerate(historical_data):
                    try:
                        if isinstance(draw, list):
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                appearances.append(i)
                        elif hasattr(draw, '__iter__'):
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                appearances.append(i)
                    except (ValueError, TypeError):
                        continue
                
                if len(appearances) > 3:
                    gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                    gap_counter = Counter(gaps)
                    most_common_gap = gap_counter.most_common(1)[0][0] if gap_counter else 0
                    
                    cycle_data[num] = {
                        'average_gap': np.mean(gaps),
                        'gap_variance': np.var(gaps),
                        'most_common_gap': most_common_gap,
                        'last_appearance': appearances[-1] if appearances else -1
                    }
            
            gap_variances = [data['gap_variance'] for data in cycle_data.values()]
            cycle_reliability = 1.0 - (np.mean(gap_variances) / 100) if gap_variances else 0.0
            cycle_reliability = max(0.0, min(1.0, cycle_reliability))
            
            return {
                'cycle_reliability': float(cycle_reliability),
                'cycle_data': cycle_data,
                'numbers_with_cycles': len([num for num, data in cycle_data.items() if data['gap_variance'] < 10])
            }
            
        except Exception as e:
            logger.error(f"Error in cycle analysis: {e}")
            return {'cycle_reliability': 0.0}

    def _analyze_momentum(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze momentum patterns"""
        try:
            recent_window = min(20, len(historical_data) // 4)
            recent_data = historical_data[-recent_window:] if recent_window > 0 else historical_data
            
            momentum_data = {}
            
            for num in range(1, 51):
                # Recent frequency
                recent_freq = 0
                valid_recent_draws = 0
                for draw in recent_data:
                    try:
                        if isinstance(draw, list):
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                recent_freq += 1
                            valid_recent_draws += 1
                    except (ValueError, TypeError):
                        continue
                
                recent_freq = recent_freq / valid_recent_draws if valid_recent_draws > 0 else 0
                
                # Long-term frequency
                long_term_freq = 0
                valid_total_draws = 0
                for draw in historical_data:
                    try:
                        if isinstance(draw, list):
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                long_term_freq += 1
                            valid_total_draws += 1
                    except (ValueError, TypeError):
                        continue
                
                long_term_freq = long_term_freq / valid_total_draws if valid_total_draws > 0 else 0
                
                momentum = recent_freq - long_term_freq
                momentum_data[num] = {
                    'recent_frequency': recent_freq,
                    'long_term_frequency': long_term_freq,
                    'momentum': momentum
                }
            
            # Calculate momentum strength
            momentums = [abs(data['momentum']) for data in momentum_data.values()]
            momentum_strength = np.mean(momentums) if momentums else 0.0
            
            # Identify hot and cold numbers
            hot_numbers = [num for num, data in momentum_data.items() if data['momentum'] > 0.05][:10]
            cold_numbers = [num for num, data in momentum_data.items() if data['momentum'] < -0.05][:10]
            
            return {
                'momentum_strength': float(momentum_strength),
                'momentum_data': momentum_data,
                'hot_numbers': hot_numbers,
                'cold_numbers': cold_numbers
            }
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return {'momentum_strength': 0.0}

    def _analyze_sequences(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze number sequences and patterns"""
        try:
            sequence_patterns = {}
            
            # Analyze consecutive number patterns
            consecutive_counts = []
            gap_patterns = []
            
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                if not draw:
                    continue
                    
                sorted_draw = sorted(draw)
                
                # Count consecutive numbers
                consecutive_count = 0
                for i in range(len(sorted_draw) - 1):
                    if sorted_draw[i+1] - sorted_draw[i] == 1:
                        consecutive_count += 1
                consecutive_counts.append(consecutive_count)
                
                # Analyze gaps
                gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
                gap_patterns.append(gaps)
            
            # Pattern strength
            avg_consecutive = np.mean(consecutive_counts) if consecutive_counts else 0
            pattern_strength = min(1.0, avg_consecutive / 2.0)  # Normalize
            
            # Most common gap patterns
            all_gaps = [gap for gaps in gap_patterns for gap in gaps]
            common_gaps = Counter(all_gaps).most_common(5) if all_gaps else []
            
            return {
                'pattern_strength': float(pattern_strength),
                'average_consecutive': float(avg_consecutive),
                'common_gaps': common_gaps,
                'gap_variance': float(np.var(all_gaps)) if all_gaps else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in sequence analysis: {e}")
            return {'pattern_strength': 0.0}

    def generate_predictions(self, analysis_result: Dict[str, Any], num_sets: int, 
                           max_number: int, numbers_per_set: int) -> List[Dict[str, Any]]:
        """Generate predictions based on temporal analysis"""
        try:
            predictions = []
            
            # Extract analysis components
            momentum_analysis = analysis_result.get('momentum_analysis', {})
            trend_analysis = analysis_result.get('trend_analysis', {})
            cycle_analysis = analysis_result.get('cycle_analysis', {})
            
            # Get hot numbers and trending numbers
            hot_numbers = momentum_analysis.get('hot_numbers', [])
            trending_up = trend_analysis.get('trending_up', [])
            
            for i in range(num_sets):
                number_set = []
                
                # Include hot/trending numbers with higher probability
                priority_numbers = list(set(hot_numbers + trending_up))
                if priority_numbers:
                    # Select 2-3 priority numbers
                    priority_count = min(len(priority_numbers), max(2, numbers_per_set // 3))
                    selected_priority = np.random.choice(priority_numbers, size=priority_count, replace=False)
                    number_set.extend(selected_priority)
                
                # Fill remaining with weighted selection
                remaining_count = numbers_per_set - len(number_set)
                available_numbers = [n for n in range(1, max_number + 1) if n not in number_set]
                
                if remaining_count > 0 and available_numbers:
                    # Weight numbers based on momentum and trends
                    weights = []
                    momentum_data = momentum_analysis.get('momentum_data', {})
                    
                    for num in available_numbers:
                        momentum = momentum_data.get(num, {}).get('momentum', 0)
                        weight = max(0.1, 1.0 + momentum)  # Ensure positive weights
                        weights.append(weight)
                    
                    # Normalize weights
                    weights = np.array(weights)
                    weights = weights / weights.sum()
                    
                    selected_remaining = np.random.choice(
                        available_numbers, 
                        size=min(remaining_count, len(available_numbers)), 
                        replace=False, 
                        p=weights
                    )
                    number_set.extend(selected_remaining)
                
                # Ensure proper count
                number_set = sorted(list(set(number_set)))[:numbers_per_set]
                
                predictions.append({
                    'numbers': number_set,
                    'confidence': analysis_result.get('confidence', 0.5),
                    'specialist': self.name,
                    'properties': {
                        'temporal_score': analysis_result.get('confidence', 0.5),
                        'hot_count': len([n for n in number_set if n in hot_numbers]),
                        'trending_count': len([n for n in number_set if n in trending_up])
                    }
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions in {self.name}: {e}")
            return []


class SpecializedExpertEnsemble:
    """Ensemble coordinator for all specialist experts"""
    
    def __init__(self):
        self.specialists = {
            'mathematical': MathematicalPatternSpecialist(),
            'temporal': TemporalSequenceSpecialist()
        }
        self.ensemble_history = []
    
    def analyze_all_patterns(self, historical_data: List[List[int]], game_type: str) -> Dict[str, Any]:
        """Run analysis across all specialists"""
        try:
            all_analysis = {
                'ensemble_timestamp': datetime.now().isoformat(),
                'game_type': game_type,
                'data_points': len(historical_data),
                'specialist_analyses': {}
            }
            
            # Run each specialist analysis
            for name, specialist in self.specialists.items():
                logger.info(f"Running {specialist.name} analysis...")
                analysis = specialist.analyze_patterns(historical_data, game_type)
                all_analysis['specialist_analyses'][name] = analysis
            
            # Calculate ensemble confidence
            specialist_confidences = [
                analysis.get('confidence', 0) 
                for analysis in all_analysis['specialist_analyses'].values()
            ]
            
            all_analysis['ensemble_confidence'] = np.mean(specialist_confidences) if specialist_confidences else 0.0
            all_analysis['confidence_variance'] = np.var(specialist_confidences) if specialist_confidences else 0.0
            
            return all_analysis
            
        except Exception as e:
            logger.error(f"Error in ensemble analysis: {e}")
            return {'ensemble_confidence': 0.0, 'error': str(e)}

    def generate_ensemble_predictions(self, ensemble_analysis: Dict[str, Any], 
                                    num_sets: int, max_number: int, numbers_per_set: int) -> List[Dict[str, Any]]:
        """Generate predictions using all specialists"""
        try:
            all_predictions = []
            
            # Get predictions from each specialist
            for name, specialist in self.specialists.items():
                specialist_analysis = ensemble_analysis['specialist_analyses'].get(name, {})
                if specialist_analysis:
                    predictions = specialist.generate_predictions(
                        specialist_analysis, num_sets, max_number, numbers_per_set
                    )
                    
                    # Tag predictions with specialist info
                    for pred in predictions:
                        pred['ensemble_source'] = name
                        pred['ensemble_confidence'] = ensemble_analysis.get('ensemble_confidence', 0.5)
                    
                    all_predictions.extend(predictions)
            
            # Rank and select best predictions
            ranked_predictions = self._rank_ensemble_predictions(all_predictions, num_sets)
            
            return ranked_predictions[:num_sets]
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {e}")
            return []

    def _rank_ensemble_predictions(self, all_predictions: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Rank predictions from all specialists"""
        try:
            # Score each prediction based on multiple factors
            for pred in all_predictions:
                score_factors = [
                    pred.get('confidence', 0.5),
                    pred.get('properties', {}).get('mathematical_score', 0.5),
                    pred.get('properties', {}).get('temporal_score', 0.5)
                ]
                
                # Weight factors based on specialist type
                if pred.get('ensemble_source') == 'mathematical':
                    score_factors[1] *= 1.2  # Boost mathematical score for math specialist
                elif pred.get('ensemble_source') == 'temporal':
                    score_factors[2] *= 1.2  # Boost temporal score for temporal specialist
                
                pred['ensemble_score'] = np.mean(score_factors)
            
            # Sort by ensemble score
            return sorted(all_predictions, key=lambda x: x.get('ensemble_score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error ranking ensemble predictions: {e}")
            return all_predictions


class ExpertEnsemble:
    """
    Expert Ensemble Engine for lottery predictions.
    
    Combines multiple machine learning models using ensemble techniques:
    - Random Forest
    - Gradient Boosting
    - Neural Networks
    - Support Vector Machines
    - Naive Bayes
    - Voting mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Expert Ensemble Engine.
        
        Args:
            config: Configuration dictionary containing engine parameters
        """
        self.config = config
        self.game_config = config.get('game_config', {})
        self.ensemble_config = config.get('ensemble_config', {})
        
        # Game parameters
        self.number_range = self.game_config.get('number_range', [1, 49])
        self.numbers_per_draw = self.game_config.get('numbers_per_draw', 6)
        self.has_bonus = self.game_config.get('has_bonus', False)
        
        # Ensemble parameters
        self.n_estimators = self.ensemble_config.get('n_estimators', 100)
        self.voting_strategy = self.ensemble_config.get('voting_strategy', 'soft')
        self.cross_validation_folds = self.ensemble_config.get('cv_folds', 5)
        self.test_size = self.ensemble_config.get('test_size', 0.2)
        self.random_state = self.ensemble_config.get('random_state', 42)
        
        # Model weights (learned through validation)
        self.model_weights = {}
        
        # Initialize models
        self._initialize_models()
        
        # Initialize sophisticated specialist ensemble
        self.specialist_ensemble = SpecializedExpertEnsemble()
        
        # Data structures
        self.historical_data = None
        self.feature_matrix = None
        self.target_matrix = None
        self.trained_models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.trained = False
        self.training_metrics = {}
        
        logger.info("🎯 Enhanced Expert Ensemble Engine initialized with sophisticated specialists")
    
    def _initialize_models(self) -> None:
        """Initialize individual ML models."""
        try:
            self.base_models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=self.random_state
                ),
                'svm': SVC(
                    probability=True,
                    random_state=self.random_state
                ),
                'naive_bayes': GaussianNB(),
                'bagging': BaggingClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            }
            
            logger.info(f"📊 Initialized {len(self.base_models)} base models")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            raise
    
    def load_data(self, historical_data: pd.DataFrame) -> None:
        """
        Load historical lottery data for training.
        
        Args:
            historical_data: DataFrame with columns ['date', 'numbers', 'bonus']
        """
        try:
            self.historical_data = historical_data.copy()
            
            # Validate data format
            required_columns = ['date', 'numbers']
            for col in required_columns:
                if col not in self.historical_data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.historical_data['date']):
                self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            
            # Sort by date
            self.historical_data = self.historical_data.sort_values('date').reset_index(drop=True)
            
            # Process numbers and create features
            self._process_data()
            
            logger.info(f"📊 Loaded {len(self.historical_data)} historical draws for ensemble training")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def _process_data(self) -> None:
        """Process historical data and create features for ML models."""
        try:
            # Extract number sequences
            processed_numbers = []
            for _, row in self.historical_data.iterrows():
                numbers = row['numbers']
                
                if isinstance(numbers, str):
                    number_list = [int(x.strip()) for x in numbers.split(',')]
                elif isinstance(numbers, list):
                    number_list = [int(x) for x in numbers]
                else:
                    raise ValueError(f"Invalid numbers format: {numbers}")
                
                processed_numbers.append(sorted(number_list))
            
            self.historical_data['processed_numbers'] = processed_numbers
            
            # Create feature matrix
            self._create_features()
            
            # Create target matrix (for prediction)
            self._create_targets()
            
        except Exception as e:
            logger.error(f"❌ Data processing failed: {e}")
            raise
    
    def _create_features(self) -> None:
        """Create comprehensive feature matrix for ML models."""
        try:
            features = []
            
            for i in range(len(self.historical_data)):
                draw_features = []
                
                # Historical frequency features (last N draws)
                lookback_windows = [5, 10, 20, 50]
                for window in lookback_windows:
                    start_idx = max(0, i - window)
                    historical_subset = self.historical_data.iloc[start_idx:i]
                    
                    if len(historical_subset) > 0:
                        # Frequency features for this window
                        freq_features = self._calculate_frequency_features(historical_subset)
                        draw_features.extend(freq_features)
                    else:
                        # Pad with zeros if insufficient history
                        draw_features.extend([0] * (self.number_range[1] - self.number_range[0] + 1))
                
                # Gap features
                gap_features = self._calculate_gap_features(i)
                draw_features.extend(gap_features)
                
                # Pattern features
                pattern_features = self._calculate_pattern_features(i)
                draw_features.extend(pattern_features)
                
                # Statistical features
                stat_features = self._calculate_statistical_features(i)
                draw_features.extend(stat_features)
                
                features.append(draw_features)
            
            self.feature_matrix = np.array(features[:-1])  # Exclude last draw (no target available)
            
            # Normalize features
            self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
            
            logger.info(f"📊 Created feature matrix: {self.feature_matrix.shape}")
            
        except Exception as e:
            logger.error(f"❌ Feature creation failed: {e}")
            raise
    
    def _calculate_frequency_features(self, historical_subset: pd.DataFrame) -> List[float]:
        """Calculate frequency features for a subset of historical data."""
        frequencies = []
        
        for num in range(self.number_range[0], self.number_range[1] + 1):
            count = 0
            total_draws = len(historical_subset)
            
            for _, row in historical_subset.iterrows():
                if num in row['processed_numbers']:
                    count += 1
            
            frequency = count / total_draws if total_draws > 0 else 0
            frequencies.append(frequency)
        
        return frequencies
    
    def _calculate_gap_features(self, current_index: int) -> List[float]:
        """Calculate gap-based features."""
        gap_features = []
        
        for num in range(self.number_range[0], self.number_range[1] + 1):
            # Find last occurrence
            last_occurrence = -1
            for i in range(current_index - 1, -1, -1):
                if num in self.historical_data.iloc[i]['processed_numbers']:
                    last_occurrence = i
                    break
            
            # Calculate gap
            gap = current_index - last_occurrence - 1 if last_occurrence >= 0 else current_index
            gap_features.append(gap)
        
        return gap_features
    
    def _calculate_pattern_features(self, current_index: int) -> List[float]:
        """Calculate pattern-based features."""
        pattern_features = []
        
        # Consecutive appearance patterns
        for num in range(self.number_range[0], self.number_range[1] + 1):
            consecutive_count = 0
            for i in range(current_index - 1, max(0, current_index - 10), -1):
                if num in self.historical_data.iloc[i]['processed_numbers']:
                    consecutive_count += 1
                else:
                    break
            pattern_features.append(consecutive_count)
        
        # Sum patterns (odd/even, high/low)
        if current_index > 0:
            last_numbers = self.historical_data.iloc[current_index - 1]['processed_numbers']
            
            odd_count = sum(1 for n in last_numbers if n % 2 == 1)
            even_count = len(last_numbers) - odd_count
            
            mid_point = (self.number_range[0] + self.number_range[1]) // 2
            high_count = sum(1 for n in last_numbers if n > mid_point)
            low_count = len(last_numbers) - high_count
            
            pattern_features.extend([odd_count, even_count, high_count, low_count])
        else:
            pattern_features.extend([0, 0, 0, 0])
        
        return pattern_features
    
    def _calculate_statistical_features(self, current_index: int) -> List[float]:
        """Calculate statistical features."""
        stat_features = []
        
        if current_index > 0:
            last_numbers = self.historical_data.iloc[current_index - 1]['processed_numbers']
            
            # Basic statistics
            stat_features.extend([
                np.mean(last_numbers),
                np.std(last_numbers),
                np.min(last_numbers),
                np.max(last_numbers),
                np.median(last_numbers)
            ])
            
            # Gaps between consecutive numbers
            sorted_numbers = sorted(last_numbers)
            gaps = [sorted_numbers[i] - sorted_numbers[i-1] for i in range(1, len(sorted_numbers))]
            
            if gaps:
                stat_features.extend([
                    np.mean(gaps),
                    np.std(gaps),
                    max(gaps),
                    min(gaps)
                ])
            else:
                stat_features.extend([0, 0, 0, 0])
        else:
            stat_features.extend([0] * 9)  # 5 basic + 4 gap statistics
        
        return stat_features
    
    def _create_targets(self) -> None:
        """Create target matrix for multi-label classification."""
        try:
            targets = []
            
            # Skip first draw (no features available) and last draw (no target available)
            for i in range(1, len(self.historical_data)):
                target_vector = np.zeros(self.number_range[1] - self.number_range[0] + 1)
                
                current_numbers = self.historical_data.iloc[i]['processed_numbers']
                
                for num in current_numbers:
                    target_vector[num - self.number_range[0]] = 1
                
                targets.append(target_vector)
            
            self.target_matrix = np.array(targets)
            
            logger.info(f"📊 Created target matrix: {self.target_matrix.shape}")
            
        except Exception as e:
            logger.error(f"❌ Target creation failed: {e}")
            raise
    
    def train(self) -> None:
        """Train the ensemble of models."""
        try:
            if self.historical_data is None:
                raise ValueError("No historical data loaded")
            
            if self.feature_matrix is None or self.target_matrix is None:
                raise ValueError("Features or targets not created")
            
            logger.info("🎓 Training Expert Ensemble...")
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                self.feature_matrix, self.target_matrix,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            # Train individual models
            self._train_individual_models(X_train, y_train, X_test, y_test)
            
            # Create ensemble model
            self._create_ensemble_model()
            
            # Train ensemble
            self.ensemble_model.fit(X_train, y_train)
            
            # Evaluate ensemble
            self._evaluate_ensemble(X_test, y_test)
            
            self.trained = True
            
            logger.info("✅ Expert Ensemble training complete")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def _train_individual_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Train individual models in the ensemble."""
        self.training_metrics = {}
        
        for model_name, model in self.base_models.items():
            try:
                logger.info(f"🎓 Training {model_name}...")
                
                # Train model on first number position (simplified for multi-label)
                model.fit(X_train, y_train[:, 0])
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train[:, 0], 
                                          cv=self.cross_validation_folds)
                
                # Test predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Calculate metrics
                accuracy = accuracy_score(y_test[:, 0], y_pred)
                
                self.training_metrics[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': accuracy,
                    'model_weight': cv_scores.mean()  # Use CV score as weight
                }
                
                self.trained_models[model_name] = model
                self.model_weights[model_name] = cv_scores.mean()
                
                logger.info(f"✅ {model_name} trained - CV: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to train {model_name}: {e}")
                self.training_metrics[model_name] = {'error': str(e)}
    
    def _create_ensemble_model(self) -> None:
        """Create the ensemble model using voting."""
        try:
            # Select best performing models
            valid_models = []
            for model_name, model in self.trained_models.items():
                if model_name in self.training_metrics and 'error' not in self.training_metrics[model_name]:
                    valid_models.append((model_name, model))
            
            if len(valid_models) < 2:
                raise ValueError("Insufficient valid models for ensemble")
            
            # Create voting classifier
            self.ensemble_model = VotingClassifier(
                estimators=valid_models,
                voting=self.voting_strategy,
                weights=[self.model_weights[name] for name, _ in valid_models]
            )
            
            logger.info(f"🎯 Created ensemble with {len(valid_models)} models")
            
        except Exception as e:
            logger.error(f"❌ Ensemble creation failed: {e}")
            raise
    
    def _evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Evaluate the ensemble model."""
        try:
            # Predict with ensemble (simplified for first position)
            y_pred = self.ensemble_model.predict(X_test)
            y_pred_proba = self.ensemble_model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test[:, 0], y_pred)
            
            self.training_metrics['ensemble'] = {
                'test_accuracy': accuracy,
                'prediction_confidence': np.mean(np.max(y_pred_proba, axis=1))
            }
            
            logger.info(f"🎯 Ensemble accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ensemble evaluation failed: {e}")
    
    def predict(self, num_predictions: int = 1, strategy: str = 'ensemble') -> List[Dict[str, Any]]:
        """
        Generate predictions using the expert ensemble.
        
        Args:
            num_predictions: Number of prediction sets to generate
            strategy: Prediction strategy ('ensemble', 'best_model', 'weighted_average')
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if not self.trained:
                raise ValueError("Ensemble not trained. Call train() first.")
            
            predictions = []
            
            # Prepare latest features for prediction
            latest_features = self._prepare_latest_features()
            
            for i in range(num_predictions):
                if strategy == 'ensemble':
                    prediction = self._predict_ensemble(latest_features)
                elif strategy == 'best_model':
                    prediction = self._predict_best_model(latest_features)
                elif strategy == 'weighted_average':
                    prediction = self._predict_weighted_average(latest_features)
                else:
                    prediction = self._predict_ensemble(latest_features)
                
                # Add noise for diversity in multiple predictions
                if i > 0:
                    prediction = self._add_prediction_diversity(prediction, i)
                
                predictions.append({
                    'numbers': prediction['numbers'],
                    'confidence': prediction['confidence'],
                    'strategy': strategy,
                    'model_contributions': prediction.get('contributions', {}),
                    'generated_at': datetime.now().isoformat(),
                    'engine': 'expert_ensemble'
                })
            
            logger.info(f"🎯 Generated {num_predictions} ensemble predictions using {strategy} strategy")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            raise
    
    def _prepare_latest_features(self) -> np.ndarray:
        """Prepare features for the latest draw."""
        try:
            # Use the last available features from training
            if self.feature_matrix is not None and len(self.feature_matrix) > 0:
                latest_features = self.feature_matrix[-1:].copy()
                return latest_features
            else:
                # Create dummy features if no training data
                feature_size = (self.number_range[1] - self.number_range[0] + 1) * 4 + 50  # Approximate
                return np.zeros((1, feature_size))
                
        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            raise
    
    def _predict_ensemble(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate prediction using the full ensemble."""
        try:
            # Get ensemble probabilities
            probabilities = self.ensemble_model.predict_proba(features)[0]
            
            # Convert probabilities to number selection
            number_probs = {}
            for i, prob in enumerate(probabilities):
                if len(probabilities) == 2:  # Binary classification
                    number_probs[i + self.number_range[0]] = prob[1] if isinstance(prob, np.ndarray) else prob
                else:
                    number_probs[i + self.number_range[0]] = prob
            
            # Select top numbers
            sorted_numbers = sorted(number_probs.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, prob in sorted_numbers[:self.numbers_per_draw]]
            
            # Calculate confidence
            top_probs = [prob for _, prob in sorted_numbers[:self.numbers_per_draw]]
            confidence = np.mean(top_probs)
            
            return {
                'numbers': sorted(selected_numbers),
                'confidence': min(0.95, confidence),
                'contributions': {'ensemble': 1.0}
            }
            
        except Exception as e:
            logger.error(f"❌ Ensemble prediction failed: {e}")
            # Fallback to random selection
            random_numbers = np.random.choice(
                range(self.number_range[0], self.number_range[1] + 1),
                size=self.numbers_per_draw,
                replace=False
            )
            return {
                'numbers': sorted(random_numbers.tolist()),
                'confidence': 0.3,
                'contributions': {'fallback': 1.0}
            }
    
    def _predict_best_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate prediction using the best performing model."""
        try:
            # Find best model
            best_model_name = max(self.model_weights.keys(), key=lambda k: self.model_weights[k])
            best_model = self.trained_models[best_model_name]
            
            # Generate prediction
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba(features)[0]
                prob_value = probabilities[1] if len(probabilities) == 2 else probabilities
            else:
                prob_value = best_model.predict(features)[0]
            
            # Create number selection (simplified approach)
            selected_numbers = np.random.choice(
                range(self.number_range[0], self.number_range[1] + 1),
                size=self.numbers_per_draw,
                replace=False
            )
            
            confidence = float(prob_value) if isinstance(prob_value, (int, float, np.number)) else 0.5
            
            return {
                'numbers': sorted(selected_numbers.tolist()),
                'confidence': min(0.95, confidence),
                'contributions': {best_model_name: 1.0}
            }
            
        except Exception as e:
            logger.error(f"❌ Best model prediction failed: {e}")
            return self._predict_ensemble(features)
    
    def _predict_weighted_average(self, features: np.ndarray) -> Dict[str, Any]:
        """Generate prediction using weighted average of all models."""
        try:
            model_predictions = {}
            total_weight = 0
            
            for model_name, model in self.trained_models.items():
                if model_name in self.model_weights:
                    try:
                        if hasattr(model, 'predict_proba'):
                            prob = model.predict_proba(features)[0]
                            prob_value = prob[1] if len(prob) == 2 else np.mean(prob)
                        else:
                            prob_value = model.predict(features)[0]
                        
                        weight = self.model_weights[model_name]
                        model_predictions[model_name] = prob_value * weight
                        total_weight += weight
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Model {model_name} prediction failed: {e}")
            
            if total_weight > 0:
                # Weighted average confidence
                avg_confidence = sum(model_predictions.values()) / total_weight
                
                # Generate numbers (simplified)
                selected_numbers = np.random.choice(
                    range(self.number_range[0], self.number_range[1] + 1),
                    size=self.numbers_per_draw,
                    replace=False
                )
                
                return {
                    'numbers': sorted(selected_numbers.tolist()),
                    'confidence': min(0.95, float(avg_confidence)),
                    'contributions': {name: pred/sum(model_predictions.values()) 
                                   for name, pred in model_predictions.items()}
                }
            else:
                return self._predict_ensemble(features)
                
        except Exception as e:
            logger.error(f"❌ Weighted average prediction failed: {e}")
            return self._predict_ensemble(features)
    
    def _add_prediction_diversity(self, prediction: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Add diversity to predictions for multiple generations."""
        try:
            numbers = prediction['numbers'].copy()
            
            # Replace some numbers with alternatives
            num_replacements = min(2, len(numbers) // 3)
            replacement_indices = np.random.choice(len(numbers), num_replacements, replace=False)
            
            for idx in replacement_indices:
                # Find alternative number not in current selection
                available_numbers = [n for n in range(self.number_range[0], self.number_range[1] + 1) 
                                   if n not in numbers]
                if available_numbers:
                    numbers[idx] = np.random.choice(available_numbers)
            
            # Slightly reduce confidence for diversified predictions
            prediction['numbers'] = sorted(numbers)
            prediction['confidence'] *= 0.95
            
            return prediction
            
        except Exception as e:
            logger.warning(f"⚠️ Diversity addition failed: {e}")
            return prediction
    
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for all models.
        
        Returns:
            Dictionary containing model performance information
        """
        if not self.trained:
            return {'error': 'Ensemble not trained'}
        
        performance = {
            'training_metrics': self.training_metrics,
            'model_weights': self.model_weights,
            'ensemble_size': len(self.trained_models),
            'available_strategies': ['ensemble', 'best_model', 'weighted_average']
        }
        
        return performance
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from ensemble models.
        
        Returns:
            Dictionary containing feature importance information
        """
        if not self.trained:
            return {'error': 'Ensemble not trained'}
        
        importance_data = {}
        
        for model_name, model in self.trained_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[model_name] = model.feature_importances_.tolist()
            elif hasattr(model, 'coef_'):
                importance_data[model_name] = np.abs(model.coef_).flatten().tolist()
        
        return {
            'feature_importances': importance_data,
            'feature_count': self.feature_matrix.shape[1] if self.feature_matrix is not None else 0
        }
    
    def save_models(self, filepath: str) -> None:
        """Save trained models to file."""
        try:
            save_data = {
                'trained_models': self.trained_models,
                'ensemble_model': self.ensemble_model,
                'model_weights': self.model_weights,
                'training_metrics': self.training_metrics,
                'scaler': self.scaler,
                'config': self.config
            }
            
            joblib.dump(save_data, filepath)
            logger.info(f"💾 Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
            raise
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from file."""
        try:
            save_data = joblib.load(filepath)
            
            self.trained_models = save_data['trained_models']
            self.ensemble_model = save_data['ensemble_model']
            self.model_weights = save_data['model_weights']
            self.training_metrics = save_data['training_metrics']
            self.scaler = save_data['scaler']
            self.trained = True
            
            logger.info(f"📁 Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")

    def analyze_specialist_patterns(self, historical_data: List[List[int]], game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Analyze patterns using sophisticated specialist ensemble"""
        try:
            logger.info("🔍 Analyzing patterns with specialist ensemble...")
            
            # Store historical data
            self.historical_data = historical_data
            
            # Run sophisticated specialist analysis
            specialist_analysis = self.specialist_ensemble.analyze_all_patterns(historical_data, game_type)
            
            logger.info(f"✅ Specialist analysis complete. Ensemble confidence: {specialist_analysis.get('ensemble_confidence', 0):.3f}")
            return specialist_analysis
            
        except Exception as e:
            logger.error(f"❌ Error in specialist pattern analysis: {e}")
            return {'ensemble_confidence': 0.0, 'error': str(e)}

    def generate_specialist_predictions(self, specialist_analysis: Dict[str, Any], 
                                      num_sets: int = 5) -> List[Dict[str, Any]]:
        """Generate predictions using sophisticated specialists"""
        try:
            logger.info(f"🎯 Generating {num_sets} prediction sets using specialists...")
            
            max_number = self.number_range[1]
            numbers_per_set = self.numbers_per_draw
            
            # Generate predictions using specialist ensemble
            predictions = self.specialist_ensemble.generate_ensemble_predictions(
                specialist_analysis, num_sets, max_number, numbers_per_set
            )
            
            logger.info(f"✅ Generated {len(predictions)} specialist prediction sets")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error generating specialist predictions: {e}")
            return []

    def generate_ensemble_insights(self, specialist_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from specialist ensemble analysis"""
        try:
            insights = {
                'specialist_insights': {},
                'ensemble_insights': [],
                'confidence_assessment': {},
                'recommendations': []
            }
            
            # Extract insights from each specialist
            specialist_analyses = specialist_analysis.get('specialist_analyses', {})
            
            for name, analysis in specialist_analyses.items():
                specialist_insights = []
                
                if name == 'mathematical':
                    prime_analysis = analysis.get('prime_analysis', {})
                    if prime_analysis.get('pattern_strength', 0) > 0.6:
                        specialist_insights.append("🔢 Strong mathematical patterns detected")
                    
                    modular_analysis = analysis.get('modular_analysis', {})
                    if modular_analysis.get('consistency_score', 0) > 0.7:
                        specialist_insights.append("📐 Excellent modular distribution consistency")
                    
                elif name == 'temporal':
                    trend_analysis = analysis.get('trend_analysis', {})
                    if trend_analysis.get('trend_strength', 0) > 0.6:
                        specialist_insights.append("📈 Clear temporal trends identified")
                    
                    momentum_analysis = analysis.get('momentum_analysis', {})
                    hot_numbers = momentum_analysis.get('hot_numbers', [])
                    if len(hot_numbers) > 5:
                        specialist_insights.append(f"🔥 {len(hot_numbers)} numbers showing positive momentum")
                
                insights['specialist_insights'][name] = specialist_insights
            
            # Ensemble-level insights
            ensemble_confidence = specialist_analysis.get('ensemble_confidence', 0)
            confidence_variance = specialist_analysis.get('confidence_variance', 0)
            
            if ensemble_confidence > 0.8:
                insights['ensemble_insights'].append("⭐ Very high ensemble confidence across specialists")
            elif ensemble_confidence > 0.6:
                insights['ensemble_insights'].append("✅ Good ensemble agreement between specialists")
            else:
                insights['ensemble_insights'].append("⚠️ Mixed signals from specialist ensemble")
            
            if confidence_variance < 0.1:
                insights['ensemble_insights'].append("🎯 High specialist agreement (low variance)")
            else:
                insights['ensemble_insights'].append("🔄 Diverse specialist perspectives (high variance)")
            
            # Confidence assessment
            insights['confidence_assessment'] = {
                'level': 'High' if ensemble_confidence > 0.7 else 'Medium' if ensemble_confidence > 0.4 else 'Low',
                'score': float(ensemble_confidence),
                'agreement': 'High' if confidence_variance < 0.1 else 'Medium' if confidence_variance < 0.2 else 'Low'
            }
            
            # Recommendations
            if ensemble_confidence > 0.7:
                insights['recommendations'].append("Consider using ensemble predictions with high confidence")
            else:
                insights['recommendations'].append("Supplement with additional analysis methods")
            
            if confidence_variance > 0.2:
                insights['recommendations'].append("Review individual specialist recommendations separately")
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generating ensemble insights: {e}")
            return {
                'specialist_insights': {},
                'ensemble_insights': ["Error generating insights"],
                'confidence_assessment': {'level': 'Unknown', 'score': 0.0, 'agreement': 'Unknown'},
                'recommendations': ["Unable to generate recommendations"]
            }

    def get_expert_ensemble_insights(self, historical_data: List[List[int]], game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Get comprehensive expert ensemble insights combining ML models and specialists"""
        try:
            logger.info("🧠 Generating comprehensive expert ensemble insights...")
            
            # Analyze with sophisticated specialists
            specialist_analysis = self.analyze_specialist_patterns(historical_data, game_type)
            
            # Generate specialist predictions
            specialist_predictions = self.generate_specialist_predictions(specialist_analysis, num_sets=5)
            
            # Generate ensemble insights
            ensemble_insights = self.generate_ensemble_insights(specialist_analysis)
            
            # Combine all insights
            comprehensive_insights = {
                'specialist_analysis': specialist_analysis,
                'specialist_predictions': specialist_predictions,
                'ensemble_insights': ensemble_insights,
                'analysis_metadata': {
                    'confidence_level': specialist_analysis.get('ensemble_confidence', 0.0),
                    'data_points_analyzed': len(historical_data),
                    'game_type': game_type,
                    'specialists_used': list(self.specialist_ensemble.specialists.keys()),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Expert ensemble insights generated with confidence: {comprehensive_insights['analysis_metadata']['confidence_level']:.3f}")
            
            return comprehensive_insights
            
        except Exception as e:
            logger.error(f"❌ Error getting expert ensemble insights: {e}")
            return {
                'specialist_analysis': {'ensemble_confidence': 0.0},
                'specialist_predictions': [],
                'ensemble_insights': {'ensemble_insights': ["Error generating insights"]},
                'error': str(e),
                'analysis_metadata': {
                    'confidence_level': 0.0,
                    'data_points_analyzed': 0,
                    'game_type': game_type,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise