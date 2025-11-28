#!/usr/bin/env python3
"""
Specialized Expert Ensemble for Ultra-High Accuracy Lottery Prediction
This module implements specialized expert models:
- Mathematical Pattern Specialist
- Temporal Sequence Specialist  
- Frequency Pattern Specialist
- Hot/Cold Number Specialist
- Position-based Specialist
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import math
from scipy import stats
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

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
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def _generate_fibonacci(self, limit: int) -> List[int]:
        """Generate Fibonacci numbers up to limit"""
        fib = [1, 1]
        while fib[-1] < limit:
            fib.append(fib[-1] + fib[-2])
        return [f for f in fib if f <= limit]
    
    def analyze_patterns(self, historical_data: List[List[int]], game_type: str) -> Dict[str, Any]:
        """Analyze mathematical patterns in historical data"""
        try:
            # Ensure data is in correct format - validate the entire dataset first
            validated_data = []
            for draw in historical_data:
                if isinstance(draw, (list, tuple)):
                    # Convert any string numbers to integers
                    validated_draw = []
                    for num in draw:
                        try:
                            if isinstance(num, str) and num.strip().isdigit():
                                validated_draw.append(int(num))
                            elif isinstance(num, (int, float)):
                                validated_draw.append(int(num))
                        except (ValueError, TypeError):
                            continue
                    if validated_draw:  # Only add non-empty draws
                        validated_data.append(validated_draw)
            
            if not validated_data:
                return {'confidence': 0.0, 'error': 'No valid data after conversion'}
            
            analysis = {
                'pattern_type': 'mathematical',
                'prime_analysis': self._analyze_prime_patterns(validated_data),
                'modular_analysis': self._analyze_modular_patterns(validated_data),
                'fibonacci_analysis': self._analyze_fibonacci_patterns(validated_data),
                'golden_ratio_analysis': self._analyze_golden_ratio_patterns(validated_data),
                'arithmetic_progression': self._analyze_arithmetic_progressions(validated_data),
                'sum_analysis': self._analyze_sum_patterns(validated_data)
            }
            
            # Calculate overall confidence
            confidence_factors = [
                analysis['prime_analysis'].get('pattern_strength', 0),
                analysis['modular_analysis'].get('consistency_score', 0),
                analysis['fibonacci_analysis'].get('significance', 0),
                analysis['sum_analysis'].get('stability', 0)
            ]
            
            analysis['confidence'] = np.mean([f for f in confidence_factors if f > 0])
            self.confidence_score = analysis['confidence']
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in mathematical pattern analysis: {e}")
            return {'confidence': 0.0, 'error': str(e)}
    
    def _analyze_prime_patterns(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze prime number patterns"""
        try:
            prime_counts = []
            prime_positions = []
            
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                primes_in_draw = [num for num in draw if num in self.prime_cache]
                prime_counts.append(len(primes_in_draw))
                
                # Analyze prime positions within the sorted draw
                sorted_draw = sorted(draw)
                for i, num in enumerate(sorted_draw):
                    if num in self.prime_cache:
                        prime_positions.append(i)
            
            return {
                'average_primes': float(np.mean(prime_counts)) if prime_counts else 0.0,
                'prime_stability': float(1.0 - np.std(prime_counts) / (np.mean(prime_counts) + 1e-6)) if prime_counts else 0.0,
                'pattern_strength': float(len(prime_counts) / len(historical_data)) if historical_data else 0.0,
                'preferred_positions': Counter(prime_positions).most_common(3) if prime_positions else []
            }
            
        except Exception as e:
            logger.error(f"Error in prime pattern analysis: {e}")
            return {}
    
    def _analyze_modular_patterns(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze modular arithmetic patterns"""
        try:
            modular_data = {}
            
            for mod in [3, 5, 7, 11]:
                mod_distributions = []
                
                for draw in historical_data:
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
                # Ensure integer data
                draw = self._ensure_integer_data(draw)
                if not draw:  # Skip empty draws
                    continue
                    
                fibs_in_draw = [num for num in draw if num in self.fibonacci_cache]
                fib_counts.append(len(fibs_in_draw))
                
                sorted_draw = sorted(draw)
                for i, num in enumerate(sorted_draw):
                    if num in self.fibonacci_cache:
                        fib_positions.append(i)
            
            # Calculate significance
            avg_fibs = np.mean(fib_counts) if fib_counts else 0
            # Use average draw length instead of assuming historical_data[0] exists
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
                # Ensure integer data
                draw = self._ensure_integer_data(draw)
                if not draw:  # Skip empty draws
                    continue
                    
                sorted_draw = sorted(draw)
                for i in range(len(sorted_draw) - 1):
                    try:
                        # Ensure both values are numeric and positive
                        if sorted_draw[i] > 0 and isinstance(sorted_draw[i], (int, float)) and isinstance(sorted_draw[i + 1], (int, float)):
                            ratio = float(sorted_draw[i + 1]) / float(sorted_draw[i])
                            if abs(ratio - self.golden_ratio) < 0.2:  # Tolerance
                                golden_relationships.append(ratio)
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue  # Skip invalid data
            
            return {
                'golden_ratio_frequency': float(len(golden_relationships) / len(historical_data)) if historical_data else 0.0,
                'average_ratio': float(np.mean(golden_relationships)) if golden_relationships else 0.0,
                'ratio_accuracy': float(1.0 - np.mean([abs(r - self.golden_ratio) for r in golden_relationships])) if golden_relationships else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in golden ratio analysis: {e}")
            return {}
    
    def _analyze_arithmetic_progressions(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze arithmetic progressions in draws"""
        try:
            progression_counts = []
            common_differences = []
            
            for draw in historical_data:
                # Ensure integer data
                draw = self._ensure_integer_data(draw)
                if not draw:  # Skip empty draws
                    continue
                    
                sorted_draw = sorted(draw)
                progressions = 0
                
                # Check for arithmetic progressions of length 3+
                for i in range(len(sorted_draw) - 2):
                    for j in range(i + 1, len(sorted_draw) - 1):
                        try:
                            # Ensure both values are integers for arithmetic operations
                            val_i = int(sorted_draw[i])
                            val_j = int(sorted_draw[j])
                            diff = val_j - val_i
                            expected_third = val_j + diff
                            
                            if expected_third in [int(x) for x in sorted_draw[j+1:]]:
                                progressions += 1
                                common_differences.append(diff)
                        except (ValueError, TypeError):
                            continue  # Skip invalid data
                
                progression_counts.append(progressions)
            
            return {
                'average_progressions': float(np.mean(progression_counts)) if progression_counts else 0.0,
                'progression_stability': float(1.0 - np.std(progression_counts) / (np.mean(progression_counts) + 1e-6)) if progression_counts else 0.0,
                'common_differences': Counter(common_differences).most_common(5) if common_differences else []
            }
            
        except Exception as e:
            logger.error(f"Error in arithmetic progression analysis: {e}")
            return {}
    
    def _analyze_sum_patterns(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze sum patterns in draws"""
        try:
            sums = []
            for draw in historical_data:
                # Ensure integer data and calculate sum safely
                draw = self._ensure_integer_data(draw)
                if draw:  # Skip empty draws
                    try:
                        draw_sum = sum(int(x) for x in draw)
                        sums.append(draw_sum)
                    except (ValueError, TypeError):
                        continue  # Skip invalid data
            
            if not sums:
                return {}
            
            # Calculate sum statistics
            avg_sum = np.mean(sums)
            std_sum = np.std(sums)
            stability = 1.0 - (std_sum / avg_sum) if avg_sum > 0 else 0.0
            
            # Check for patterns in sum modulo
            sum_mod_7 = [s % 7 for s in sums]
            mod_distribution = Counter(sum_mod_7)
            mod_evenness = 1.0 - (max(mod_distribution.values()) - min(mod_distribution.values())) / len(sums) if sums else 0.0
            
            return {
                'average_sum': float(avg_sum),
                'sum_stability': float(stability),
                'mod_evenness': float(mod_evenness),
                'stability': float((stability + mod_evenness) / 2)
            }
            
        except Exception as e:
            logger.error(f"Error in sum pattern analysis: {e}")
            return {'stability': 0.0}
    
    def generate_predictions(self, analysis_result: Dict[str, Any], num_sets: int, 
                           max_number: int, numbers_per_set: int) -> List[Dict[str, Any]]:
        """Generate predictions based on mathematical analysis"""
        try:
            predictions = []
            
            for set_id in range(num_sets):
                numbers = []
                
                # Strategy 1: Prime-focused set
                if set_id == 0:
                    # Include primes based on analysis
                    prime_analysis = analysis_result.get('prime_analysis', {})
                    target_primes = max(1, int(prime_analysis.get('average_primes', 2)))
                    
                    # Select primes
                    available_primes = [p for p in self.prime_cache if p <= max_number]
                    selected_primes = np.random.choice(available_primes, min(target_primes, len(available_primes)), replace=False)
                    numbers.extend(selected_primes)
                    
                    # Fill with non-primes
                    non_primes = [n for n in range(1, max_number + 1) if n not in self.prime_cache]
                    remaining = numbers_per_set - len(numbers)
                    if remaining > 0:
                        additional = np.random.choice(non_primes, min(remaining, len(non_primes)), replace=False)
                        numbers.extend(additional)
                
                # Strategy 2: Fibonacci-focused set
                elif set_id == 1:
                    # Include Fibonacci numbers
                    available_fibs = [f for f in self.fibonacci_cache if f <= max_number]
                    target_fibs = min(2, len(available_fibs))
                    
                    if target_fibs > 0:
                        selected_fibs = np.random.choice(available_fibs, target_fibs, replace=False)
                        numbers.extend(selected_fibs)
                    
                    # Fill with other numbers
                    non_fibs = [n for n in range(1, max_number + 1) if n not in self.fibonacci_cache]
                    remaining = numbers_per_set - len(numbers)
                    if remaining > 0:
                        additional = np.random.choice(non_fibs, min(remaining, len(non_fibs)), replace=False)
                        numbers.extend(additional)
                
                # Strategy 3: Modular balance set
                else:
                    # Create balanced modular distribution
                    for mod_class in range(7):  # Use mod 7
                        if len(numbers) < numbers_per_set:
                            candidates = [n for n in range(1, max_number + 1) if n % 7 == mod_class and n not in numbers]
                            if candidates:
                                numbers.append(np.random.choice(candidates))
                    
                    # Fill remaining slots
                    remaining = numbers_per_set - len(numbers)
                    if remaining > 0:
                        available = [n for n in range(1, max_number + 1) if n not in numbers]
                        if len(available) >= remaining:
                            additional = np.random.choice(available, remaining, replace=False)
                            numbers.extend(additional)
                
                # Ensure we have the right number of unique numbers
                numbers = list(set(numbers))
                while len(numbers) < numbers_per_set:
                    available = [n for n in range(1, max_number + 1) if n not in numbers]
                    if available:
                        numbers.append(np.random.choice(available))
                
                numbers = sorted(numbers[:numbers_per_set])
                
                # Calculate mathematical properties of this set
                set_properties = self._calculate_set_properties(numbers)
                
                predictions.append({
                    'set_id': set_id + 1,
                    'numbers': numbers,
                    'confidence': self.confidence_score * set_properties.get('mathematical_score', 0.8),
                    'strategy': f'Mathematical Strategy {set_id + 1}',
                    'properties': set_properties,
                    'specialist': self.name
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating mathematical predictions: {e}")
            return []
    
    def _calculate_set_properties(self, numbers: List[int]) -> Dict[str, Any]:
        """Calculate mathematical properties of a number set"""
        try:
            properties = {}
            
            # Prime content
            primes_in_set = [n for n in numbers if n in self.prime_cache]
            properties['prime_count'] = len(primes_in_set)
            properties['prime_ratio'] = len(primes_in_set) / len(numbers) if numbers else 0
            
            # Fibonacci content
            fibs_in_set = [n for n in numbers if n in self.fibonacci_cache]
            properties['fibonacci_count'] = len(fibs_in_set)
            
            # Sum properties
            total_sum = sum(numbers)
            properties['sum'] = total_sum
            properties['sum_mod_7'] = total_sum % 7
            
            # Distribution properties
            if len(numbers) > 1:
                gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
                properties['average_gap'] = np.mean(gaps)
                properties['gap_variance'] = np.var(gaps)
            
            # Overall mathematical score
            score_factors = [
                min(1.0, properties['prime_ratio'] * 3),  # Prime content
                min(1.0, properties['fibonacci_count'] / 2),  # Fibonacci content
                1.0 - abs(properties['sum_mod_7'] - 3.5) / 3.5 if 'sum_mod_7' in properties else 0.5  # Modular balance
            ]
            
            properties['mathematical_score'] = np.mean(score_factors)
            
            return properties
            
        except Exception as e:
            logger.error(f"Error calculating set properties: {e}")
            return {'mathematical_score': 0.5}


class TemporalSequenceSpecialist(SpecialistExpert):
    """Specialist focusing on temporal patterns and sequences"""
    
    def __init__(self):
        super().__init__("Temporal Sequence Specialist")
        self.sequence_patterns = {}
        self.trend_analysis = {}
    
    def analyze_patterns(self, historical_data: List[List[int]], game_type: str) -> Dict[str, Any]:
        """Analyze temporal patterns in historical data"""
        try:
            # Ensure data is in correct format - validate the entire dataset first
            validated_data = []
            for draw in historical_data:
                if isinstance(draw, (list, tuple)):
                    # Convert any string numbers to integers
                    validated_draw = []
                    for num in draw:
                        try:
                            if isinstance(num, str) and num.strip().isdigit():
                                validated_draw.append(int(num))
                            elif isinstance(num, (int, float)):
                                validated_draw.append(int(num))
                        except (ValueError, TypeError):
                            continue
                    if validated_draw:  # Only add non-empty draws
                        validated_data.append(validated_draw)
            
            if not validated_data:
                return {'confidence': 0.0, 'error': 'No valid data after conversion'}
            
            analysis = {
                'pattern_type': 'temporal',
                'trend_analysis': self._analyze_trends(validated_data),
                'sequence_patterns': self._analyze_sequences(validated_data),
                'cyclical_patterns': self._analyze_cycles(validated_data),
                'momentum_analysis': self._analyze_momentum(validated_data),
                'time_decay_analysis': self._analyze_time_decay(validated_data)
            }
            
            # Calculate confidence based on pattern strength
            confidence_factors = [
                analysis['trend_analysis'].get('trend_strength', 0),
                analysis['sequence_patterns'].get('pattern_consistency', 0),
                analysis['cyclical_patterns'].get('cycle_reliability', 0),
                analysis['momentum_analysis'].get('momentum_stability', 0)
            ]
            
            analysis['confidence'] = np.mean([f for f in confidence_factors if f > 0])
            self.confidence_score = analysis['confidence']
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
            return {'confidence': 0.0, 'error': str(e)}
    
    def _analyze_trends(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze number trends over time"""
        try:
            # Track each number's frequency over time windows
            window_size = 10
            trend_data = {}
            
            for num in range(1, 51):  # Assume max 50 numbers
                frequencies = []
                
                for i in range(len(historical_data) - window_size + 1):
                    window_data = historical_data[i:i + window_size]
                    # Count frequency with proper data validation
                    freq = 0
                    for draw in window_data:
                        # Ensure draw is list of integers
                        try:
                            if isinstance(draw, list):
                                # Convert to integers if needed
                                int_draw = [int(x) for x in draw if str(x).isdigit()]
                                if num in int_draw:
                                    freq += 1
                            elif hasattr(draw, '__iter__'):  # Handle other iterables
                                int_draw = [int(x) for x in draw if str(x).isdigit()]
                                if num in int_draw:
                                    freq += 1
                        except (ValueError, TypeError):
                            continue  # Skip invalid data
                    frequencies.append(freq)
                
                if frequencies:
                    # Calculate trend (slope)
                    x = np.arange(len(frequencies))
                    if len(frequencies) > 1:
                        slope, _, r_value, _, _ = stats.linregress(x, frequencies)
                        trend_data[num] = {
                            'slope': slope,
                            'r_squared': r_value ** 2,
                            'recent_freq': frequencies[-5:] if len(frequencies) >= 5 else frequencies,
                            'trend_direction': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'
                        }
            
            # Overall trend strength
            slopes = [data['slope'] for data in trend_data.values()]
            r_squares = [data['r_squared'] for data in trend_data.values()]
            
            trend_strength = np.mean(r_squares) if r_squares else 0.0
            
            # Identify trending numbers
            increasing_numbers = [num for num, data in trend_data.items() if data['trend_direction'] == 'increasing']
            decreasing_numbers = [num for num, data in trend_data.items() if data['trend_direction'] == 'decreasing']
            
            return {
                'trend_strength': float(trend_strength),
                'trending_up': increasing_numbers[:10],  # Top 10
                'trending_down': decreasing_numbers[:10],  # Top 10
                'trend_data': trend_data
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'trend_strength': 0.0}
    
    def _analyze_sequences(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze sequential patterns"""
        try:
            # Look for repeating subsequences
            sequence_patterns = defaultdict(int)
            
            for i in range(len(historical_data) - 2):
                # Get 3-draw sequences
                seq = (tuple(historical_data[i]), tuple(historical_data[i+1]), tuple(historical_data[i+2]))
                sequence_patterns[seq] += 1
            
            # Find most common patterns
            common_patterns = Counter(sequence_patterns).most_common(5)
            
            # Calculate pattern consistency
            if sequence_patterns:
                total_sequences = sum(sequence_patterns.values())
                max_frequency = max(sequence_patterns.values())
                pattern_consistency = max_frequency / total_sequences
            else:
                pattern_consistency = 0.0
            
            # Analyze position-based sequences
            position_sequences = self._analyze_position_sequences(historical_data)
            
            return {
                'pattern_consistency': float(pattern_consistency),
                'common_patterns': common_patterns,
                'position_sequences': position_sequences,
                'total_patterns_found': len(sequence_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error in sequence analysis: {e}")
            return {'pattern_consistency': 0.0}
    
    def _analyze_position_sequences(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze sequences within draw positions"""
        try:
            position_data = {}
            
            for pos in range(6):  # Assume 6-7 number draws
                position_sequences = []
                
                for draw in historical_data:
                    if pos < len(draw):
                        sorted_draw = sorted(draw)
                        position_sequences.append(sorted_draw[pos])
                
                if len(position_sequences) > 3:
                    # Look for patterns in consecutive position values
                    differences = [position_sequences[i+1] - position_sequences[i] for i in range(len(position_sequences)-1)]
                    
                    position_data[pos] = {
                        'average_value': np.mean(position_sequences),
                        'value_variance': np.var(position_sequences),
                        'average_difference': np.mean(differences) if differences else 0,
                        'difference_variance': np.var(differences) if differences else 0
                    }
            
            return position_data
            
        except Exception as e:
            logger.error(f"Error in position sequence analysis: {e}")
            return {}
    
    def _analyze_cycles(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze cyclical patterns"""
        try:
            # Look for cyclical patterns in number appearances
            cycle_data = {}
            
            for num in range(1, 51):
                appearances = []
                for i, draw in enumerate(historical_data):
                    # Ensure proper data validation
                    try:
                        if isinstance(draw, list):
                            # Convert to integers if needed
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                appearances.append(i)
                        elif hasattr(draw, '__iter__'):  # Handle other iterables
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                appearances.append(i)
                    except (ValueError, TypeError):
                        continue  # Skip invalid data
                
                if len(appearances) > 3:
                    # Calculate gaps between appearances
                    gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                    
                    # Look for cyclical patterns
                    gap_counter = Counter(gaps)
                    most_common_gap = gap_counter.most_common(1)[0][0] if gap_counter else 0
                    
                    cycle_data[num] = {
                        'average_gap': np.mean(gaps),
                        'gap_variance': np.var(gaps),
                        'most_common_gap': most_common_gap,
                        'last_appearance': appearances[-1] if appearances else -1
                    }
            
            # Calculate cycle reliability
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
            # Recent vs long-term frequency comparison
            recent_window = min(20, len(historical_data) // 4)
            recent_data = historical_data[-recent_window:] if recent_window > 0 else historical_data
            
            momentum_data = {}
            
            for num in range(1, 51):
                # Recent frequency with data validation
                recent_freq = 0
                valid_recent_draws = 0
                for draw in recent_data:
                    try:
                        if isinstance(draw, list):
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                recent_freq += 1
                            valid_recent_draws += 1
                        elif hasattr(draw, '__iter__'):
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                recent_freq += 1
                            valid_recent_draws += 1
                    except (ValueError, TypeError):
                        continue
                
                recent_freq = recent_freq / valid_recent_draws if valid_recent_draws > 0 else 0
                
                # Long-term frequency with data validation
                long_term_freq = 0
                valid_total_draws = 0
                for draw in historical_data:
                    try:
                        if isinstance(draw, list):
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                long_term_freq += 1
                            valid_total_draws += 1
                        elif hasattr(draw, '__iter__'):
                            int_draw = [int(x) for x in draw if str(x).isdigit()]
                            if num in int_draw:
                                long_term_freq += 1
                            valid_total_draws += 1
                    except (ValueError, TypeError):
                        continue
                
                long_term_freq = long_term_freq / valid_total_draws if valid_total_draws > 0 else 0
                
                # Momentum score
                momentum = recent_freq - long_term_freq
                
                momentum_data[num] = {
                    'recent_frequency': recent_freq,
                    'long_term_frequency': long_term_freq,
                    'momentum': momentum,
                    'momentum_direction': 'positive' if momentum > 0.05 else 'negative' if momentum < -0.05 else 'neutral'
                }
            
            # Momentum stability
            momentum_values = [data['momentum'] for data in momentum_data.values()]
            momentum_stability = 1.0 - np.std(momentum_values) if momentum_values else 0.0
            momentum_stability = max(0.0, min(1.0, momentum_stability))
            
            # Hot and cold numbers based on momentum
            hot_numbers = [num for num, data in momentum_data.items() if data['momentum_direction'] == 'positive']
            cold_numbers = [num for num, data in momentum_data.items() if data['momentum_direction'] == 'negative']
            
            return {
                'momentum_stability': float(momentum_stability),
                'hot_numbers': hot_numbers[:15],
                'cold_numbers': cold_numbers[:15],
                'momentum_data': momentum_data
            }
            
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return {'momentum_stability': 0.0}
    
    def _analyze_time_decay(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze time decay patterns"""
        try:
            # Calculate weighted frequencies with exponential decay
            decay_rate = 0.95  # Numbers from recent draws are weighted more heavily
            
            time_decay_data = {}
            
            for num in range(1, 51):
                weighted_frequency = 0.0
                total_weight = 0.0
                
                for i, draw in enumerate(reversed(historical_data)):
                    weight = decay_rate ** i
                    if num in draw:
                        weighted_frequency += weight
                    total_weight += weight
                
                if total_weight > 0:
                    time_decay_data[num] = weighted_frequency / total_weight
                else:
                    time_decay_data[num] = 0.0
            
            # Calculate decay stability
            decay_values = list(time_decay_data.values())
            decay_stability = 1.0 - np.std(decay_values) if decay_values else 0.0
            
            return {
                'decay_stability': float(decay_stability),
                'time_weighted_frequencies': time_decay_data,
                'top_recent_numbers': sorted(time_decay_data.items(), key=lambda x: x[1], reverse=True)[:15]
            }
            
        except Exception as e:
            logger.error(f"Error in time decay analysis: {e}")
            return {'decay_stability': 0.0}
    
    def generate_predictions(self, analysis_result: Dict[str, Any], num_sets: int, 
                           max_number: int, numbers_per_set: int) -> List[Dict[str, Any]]:
        """Generate predictions based on temporal analysis"""
        try:
            predictions = []
            
            # Extract analysis components
            trend_analysis = analysis_result.get('trend_analysis', {})
            momentum_analysis = analysis_result.get('momentum_analysis', {})
            cycle_analysis = analysis_result.get('cycle_analysis', {})
            time_decay_analysis = analysis_result.get('time_decay_analysis', {})
            
            for set_id in range(num_sets):
                numbers = []
                
                # Strategy 1: Trend-based set
                if set_id == 0:
                    trending_up = trend_analysis.get('trending_up', [])[:max_number]
                    hot_numbers = momentum_analysis.get('hot_numbers', [])[:max_number]
                    
                    # Combine trending and hot numbers
                    candidates = list(set(trending_up + hot_numbers))
                    np.random.shuffle(candidates)
                    numbers.extend(candidates[:numbers_per_set//2])
                    
                    # Fill with time-weighted selections
                    time_weighted = time_decay_analysis.get('top_recent_numbers', [])
                    for num, weight in time_weighted:
                        if len(numbers) < numbers_per_set and num not in numbers and num <= max_number:
                            numbers.append(num)
                
                # Strategy 2: Momentum-based set
                elif set_id == 1:
                    momentum_data = momentum_analysis.get('momentum_data', {})
                    
                    # Select numbers with strong positive momentum
                    positive_momentum = [(num, data['momentum']) for num, data in momentum_data.items() 
                                       if data['momentum'] > 0 and num <= max_number]
                    positive_momentum.sort(key=lambda x: x[1], reverse=True)
                    
                    for num, momentum in positive_momentum[:numbers_per_set]:
                        numbers.append(num)
                    
                    # Fill remaining with balanced selections
                    if len(numbers) < numbers_per_set:
                        neutral_numbers = [num for num, data in momentum_data.items() 
                                         if data['momentum_direction'] == 'neutral' and num not in numbers and num <= max_number]
                        np.random.shuffle(neutral_numbers)
                        numbers.extend(neutral_numbers[:numbers_per_set - len(numbers)])
                
                # Strategy 3: Cycle-based set
                else:
                    cycle_data = cycle_analysis.get('cycle_data', {})
                    
                    # Select numbers that are "due" based on their cycles
                    current_draw_index = len(analysis_result.get('historical_data', []))
                    due_numbers = []
                    
                    for num, data in cycle_data.items():
                        if num <= max_number:
                            last_appearance = data.get('last_appearance', -1)
                            average_gap = data.get('average_gap', 10)
                            
                            # Calculate how "due" this number is
                            draws_since_last = current_draw_index - last_appearance - 1
                            due_score = draws_since_last / (average_gap + 1e-6)
                            
                            if due_score > 0.8:  # Number is overdue
                                due_numbers.append((num, due_score))
                    
                    # Sort by due score and select top numbers
                    due_numbers.sort(key=lambda x: x[1], reverse=True)
                    for num, score in due_numbers[:numbers_per_set]:
                        numbers.append(num)
                    
                    # Fill remaining with random selections from available numbers
                    if len(numbers) < numbers_per_set:
                        available = [n for n in range(1, max_number + 1) if n not in numbers]
                        np.random.shuffle(available)
                        numbers.extend(available[:numbers_per_set - len(numbers)])
                
                # Ensure we have the right number of unique numbers
                numbers = list(set(numbers))
                while len(numbers) < numbers_per_set:
                    available = [n for n in range(1, max_number + 1) if n not in numbers]
                    if available:
                        numbers.append(np.random.choice(available))
                
                numbers = sorted(numbers[:numbers_per_set])
                
                # Calculate temporal properties
                set_properties = self._calculate_temporal_properties(numbers, analysis_result)
                
                predictions.append({
                    'set_id': set_id + 1,
                    'numbers': numbers,
                    'confidence': self.confidence_score * set_properties.get('temporal_score', 0.8),
                    'strategy': f'Temporal Strategy {set_id + 1}',
                    'properties': set_properties,
                    'specialist': self.name
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating temporal predictions: {e}")
            return []
    
    def _calculate_temporal_properties(self, numbers: List[int], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate temporal properties of a number set"""
        try:
            properties = {}
            
            # Momentum analysis
            momentum_data = analysis_result.get('momentum_analysis', {}).get('momentum_data', {})
            positive_momentum_count = sum(1 for num in numbers if momentum_data.get(num, {}).get('momentum', 0) > 0)
            properties['positive_momentum_ratio'] = positive_momentum_count / len(numbers) if numbers else 0
            
            # Trend analysis
            trend_analysis = analysis_result.get('trend_analysis', {})
            trending_up = trend_analysis.get('trending_up', [])
            trending_count = sum(1 for num in numbers if num in trending_up)
            properties['trending_ratio'] = trending_count / len(numbers) if numbers else 0
            
            # Time decay weighted score
            time_decay_data = analysis_result.get('time_decay_analysis', {}).get('time_weighted_frequencies', {})
            time_weights = [time_decay_data.get(num, 0) for num in numbers]
            properties['average_time_weight'] = np.mean(time_weights) if time_weights else 0
            
            # Overall temporal score
            score_factors = [
                properties['positive_momentum_ratio'],
                properties['trending_ratio'],
                min(1.0, properties['average_time_weight'] * 3)  # Scale time weight
            ]
            
            properties['temporal_score'] = np.mean([f for f in score_factors if f >= 0])
            
            return properties
            
        except Exception as e:
            logger.error(f"Error calculating temporal properties: {e}")
            return {'temporal_score': 0.5}


class SpecializedExpertEnsemble:
    """Ensemble coordinator for all specialist experts"""
    
    def __init__(self):
        self.specialists = {
            'mathematical': MathematicalPatternSpecialist(),
            'temporal': TemporalSequenceSpecialist()
            # Will add more specialists in subsequent phases
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
                
                pred['ensemble_score'] = np.mean([f for f in score_factors if f > 0])
            
            # Sort by ensemble score
            ranked = sorted(all_predictions, key=lambda x: x['ensemble_score'], reverse=True)
            
            # Ensure diversity - don't select too many from same specialist
            selected = []
            specialist_counts = defaultdict(int)
            max_per_specialist = max(1, target_count // len(self.specialists))
            
            for pred in ranked:
                source = pred.get('ensemble_source', 'unknown')
                if len(selected) < target_count and specialist_counts[source] < max_per_specialist:
                    selected.append(pred)
                    specialist_counts[source] += 1
            
            # Fill remaining slots with best remaining predictions
            while len(selected) < target_count and len(selected) < len(ranked):
                for pred in ranked:
                    if pred not in selected and len(selected) < target_count:
                        selected.append(pred)
                        break
            
            return selected
            
        except Exception as e:
            logger.error(f"Error ranking ensemble predictions: {e}")
            return all_predictions[:target_count]
    
    def get_ensemble_insights(self, ensemble_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from ensemble analysis"""
        try:
            insights = {
                'specialist_insights': {},
                'ensemble_insights': [],
                'confidence_assessment': {},
                'recommendations': []
            }
            
            # Get insights from each specialist
            for name, analysis in ensemble_analysis.get('specialist_analyses', {}).items():
                specialist = self.specialists.get(name)
                if specialist and analysis:
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
            ensemble_confidence = ensemble_analysis.get('ensemble_confidence', 0)
            confidence_variance = ensemble_analysis.get('confidence_variance', 0)
            
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
            logger.error(f"Error generating ensemble insights: {e}")
            return {
                'specialist_insights': {},
                'ensemble_insights': ["Error generating insights"],
                'confidence_assessment': {'level': 'Unknown', 'score': 0.0, 'agreement': 'Unknown'},
                'recommendations': ["Unable to generate recommendations"]
            }
