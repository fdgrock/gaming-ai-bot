"""
Phase 3: Set Optimizer Engine for lottery prediction system.

This engine focuses on optimizing lottery number combinations for maximum
coverage and hit rate potential. It uses combinatorial optimization,
coverage analysis, and strategic set generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime
import logging
from itertools import combinations, product
from collections import defaultdict, Counter
import math
from scipy.optimize import minimize
from scipy import stats
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SetOptimizationStrategy(ABC):
    """Abstract base class for set optimization strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.performance_history = []
    
    @abstractmethod
    def optimize_sets(self, base_predictions: List[List[int]], game_config: Dict[str, Any], 
                     target_sets: int) -> List[Dict[str, Any]]:
        """Optimize a collection of number sets"""
        pass
    
    def calculate_set_coverage(self, sets: List[List[int]], max_number: int) -> float:
        """Calculate what percentage of possible numbers are covered by the sets"""
        all_numbers = set()
        for number_set in sets:
            all_numbers.update(number_set)
        return len(all_numbers) / max_number
    
    def calculate_set_diversity(self, sets: List[List[int]]) -> float:
        """Calculate diversity between sets (lower overlap = higher diversity)"""
        if len(sets) <= 1:
            return 1.0
        
        total_diversity = 0
        comparisons = 0
        
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                set1 = set(sets[i])
                set2 = set(sets[j])
                
                overlap = len(set1.intersection(set2))
                union_size = len(set1.union(set2))
                
                diversity = 1 - (overlap / union_size) if union_size > 0 else 0
                total_diversity += diversity
                comparisons += 1
        
        return total_diversity / comparisons if comparisons > 0 else 0


class CoverageOptimizer(SetOptimizationStrategy):
    """Optimizes sets for maximum number coverage"""
    
    def __init__(self):
        super().__init__("Coverage Optimizer")
    
    def optimize_sets(self, base_predictions: List[List[int]], game_config: Dict[str, Any], 
                     target_sets: int) -> List[Dict[str, Any]]:
        """Generate sets that maximize number coverage"""
        try:
            max_number = game_config.get('max_number', 49)
            numbers_per_set = game_config.get('numbers_per_set', 6)
            
            optimized_sets = []
            used_numbers = set()
            
            # Strategy: Greedy coverage maximization
            for set_id in range(target_sets):
                current_set = []
                
                # Calculate unused number frequencies from base predictions
                number_frequencies = self._calculate_unused_frequencies(
                    base_predictions, used_numbers, max_number
                )
                
                # Select numbers to maximize coverage while maintaining quality
                while len(current_set) < numbers_per_set:
                    best_number = self._select_best_coverage_number(
                        number_frequencies, current_set, used_numbers, max_number
                    )
                    
                    if best_number is not None:
                        current_set.append(best_number)
                        used_numbers.add(best_number)
                    else:
                        # Fill with random unused numbers if needed
                        unused_numbers = [n for n in range(1, max_number + 1) 
                                        if n not in current_set and n not in used_numbers]
                        if unused_numbers:
                            current_set.append(np.random.choice(unused_numbers))
                
                # Calculate coverage metrics
                coverage_score = self._calculate_coverage_contribution(current_set, used_numbers, max_number)
                set_quality = self._calculate_set_quality(current_set, base_predictions)
                
                optimized_sets.append({
                    'set_id': set_id + 1,
                    'numbers': sorted(current_set),
                    'strategy': 'Coverage Optimization',
                    'coverage_score': coverage_score,
                    'set_quality': set_quality,
                    'combined_score': (coverage_score * 0.6 + set_quality * 0.4),
                    'optimizer': self.name
                })
            
            return optimized_sets
            
        except Exception as e:
            logger.error(f"Error in coverage optimization: {e}")
            return []

    def _calculate_unused_frequencies(self, base_predictions: List[List[int]], 
                                    used_numbers: Set[int], max_number: int) -> Dict[int, float]:
        """Calculate frequencies of unused numbers from base predictions"""
        frequencies = defaultdict(float)
        
        for prediction in base_predictions:
            for number in prediction:
                if number not in used_numbers and 1 <= number <= max_number:
                    frequencies[number] += 1
        
        # Normalize frequencies
        total = sum(frequencies.values())
        if total > 0:
            for num in frequencies:
                frequencies[num] /= total
        
        return frequencies

    def _select_best_coverage_number(self, number_frequencies: Dict[int, float], 
                                   current_set: List[int], used_numbers: Set[int], 
                                   max_number: int) -> Optional[int]:
        """Select the best number for coverage optimization"""
        candidates = []
        
        for number, frequency in number_frequencies.items():
            if number not in current_set and number not in used_numbers:
                # Score based on frequency and gap-filling potential
                gap_bonus = self._calculate_gap_bonus(number, current_set, max_number)
                total_score = frequency + gap_bonus
                candidates.append((number, total_score))
        
        if candidates:
            # Select number with highest score
            best_candidate = max(candidates, key=lambda x: x[1])
            return best_candidate[0]
        
        return None

    def _calculate_gap_bonus(self, number: int, current_set: List[int], max_number: int) -> float:
        """Calculate bonus for filling gaps in number distribution"""
        if not current_set:
            return 0.1
        
        sorted_set = sorted(current_set)
        min_existing = min(sorted_set)
        max_existing = max(sorted_set)
        
        # Bonus for extending range or filling gaps
        if number < min_existing or number > max_existing:
            range_bonus = 0.2  # Bonus for extending range
        else:
            range_bonus = 0.1  # Smaller bonus for filling gaps
        
        return range_bonus

    def _calculate_coverage_contribution(self, number_set: List[int], 
                                       all_used_numbers: Set[int], max_number: int) -> float:
        """Calculate how much this set contributes to overall coverage"""
        unique_contributions = len([n for n in number_set if n not in all_used_numbers])
        return unique_contributions / len(number_set) if number_set else 0

    def _calculate_set_quality(self, number_set: List[int], 
                             base_predictions: List[List[int]]) -> float:
        """Calculate quality score based on how well set aligns with base predictions"""
        if not base_predictions:
            return 0.5
        
        # Count how many numbers from this set appear in base predictions
        appearance_count = 0
        total_appearances = 0
        
        for base_set in base_predictions:
            base_set_numbers = set(base_set)
            matches = len(set(number_set).intersection(base_set_numbers))
            appearance_count += matches
            total_appearances += len(base_set)
        
        # Quality is based on alignment with base predictions
        if total_appearances > 0:
            return appearance_count / total_appearances
        else:
            return 0.5


class ComplementarySetGenerator(SetOptimizationStrategy):
    """Generates complementary sets that fill gaps in base predictions"""
    
    def __init__(self):
        super().__init__("Complementary Set Generator")
    
    def optimize_sets(self, base_predictions: List[List[int]], game_config: Dict[str, Any], 
                     target_sets: int) -> List[Dict[str, Any]]:
        """Generate sets that complement base predictions"""
        try:
            max_number = game_config.get('max_number', 49)
            numbers_per_set = game_config.get('numbers_per_set', 6)
            
            # Analyze gaps in base predictions
            gap_analysis = self._analyze_prediction_gaps(base_predictions, max_number)
            
            optimized_sets = []
            
            for set_id in range(target_sets):
                if set_id == 0:
                    # First set: Fill the biggest gaps
                    current_set = self._generate_gap_filling_set(
                        gap_analysis, numbers_per_set, max_number
                    )
                elif set_id == 1:
                    # Second set: Contrarian approach (opposite patterns)
                    current_set = self._generate_contrarian_set(
                        base_predictions, numbers_per_set, max_number
                    )
                else:
                    # Additional sets: Balanced complementary approach
                    current_set = self._generate_balanced_complement_set(
                        base_predictions, optimized_sets, numbers_per_set, max_number
                    )
                
                # Calculate complementary properties
                gap_fill_score = self._calculate_gap_fill_score(current_set, gap_analysis)
                complementary_score = self._calculate_complementary_score(current_set, base_predictions)
                
                optimized_sets.append({
                    'set_id': set_id + 1,
                    'numbers': sorted(current_set),
                    'strategy': 'Complementary Generation',
                    'gap_fill_score': gap_fill_score,
                    'complementary_score': complementary_score,
                    'combined_score': (gap_fill_score * 0.5 + complementary_score * 0.5),
                    'optimizer': self.name
                })
            
            return optimized_sets
            
        except Exception as e:
            logger.error(f"Error in complementary set generation: {e}")
            return []

    def _analyze_prediction_gaps(self, base_predictions: List[List[int]], 
                               max_number: int) -> Dict[str, Any]:
        """Analyze gaps and under-represented numbers in base predictions"""
        # Count frequency of each number
        number_frequencies = Counter()
        for pred_set in base_predictions:
            for number in pred_set:
                if 1 <= number <= max_number:
                    number_frequencies[number] += 1
        
        # Identify gaps
        total_predictions = len(base_predictions)
        expected_frequency = total_predictions * 6 / max_number  # Rough expectation
        
        under_represented = []
        over_represented = []
        
        for number in range(1, max_number + 1):
            frequency = number_frequencies.get(number, 0)
            deviation = frequency - expected_frequency
            
            if deviation < -expected_frequency * 0.3:  # 30% below expected
                under_represented.append((number, abs(deviation)))
            elif deviation > expected_frequency * 0.3:  # 30% above expected
                over_represented.append((number, deviation))
        
        return {
            'under_represented': sorted(under_represented, key=lambda x: x[1], reverse=True),
            'over_represented': sorted(over_represented, key=lambda x: x[1], reverse=True),
            'frequencies': dict(number_frequencies),
            'expected_frequency': expected_frequency
        }

    def _generate_gap_filling_set(self, gap_analysis: Dict[str, Any], 
                                numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a set that fills the biggest gaps"""
        under_represented = gap_analysis['under_represented']
        current_set = []
        
        # Start with most under-represented numbers
        for number, _ in under_represented:
            if len(current_set) < numbers_per_set:
                current_set.append(number)
        
        # Fill remaining slots with balanced selection
        while len(current_set) < numbers_per_set:
            available = [n for n in range(1, max_number + 1) if n not in current_set]
            if available:
                # Prefer numbers with lower representation
                frequencies = gap_analysis['frequencies']
                weights = [1.0 / (frequencies.get(n, 0) + 1) for n in available]
                weights = np.array(weights) / sum(weights)
                selected = np.random.choice(available, p=weights)
                current_set.append(selected)
        
        return current_set

    def _generate_contrarian_set(self, base_predictions: List[List[int]], 
                               numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a contrarian set (opposite of base prediction patterns)"""
        # Identify most common patterns in base predictions
        number_frequencies = Counter()
        for pred_set in base_predictions:
            for number in pred_set:
                number_frequencies[number] += 1
        
        # Select least common numbers
        least_common = [num for num, _ in number_frequencies.most_common()[-max_number:]]
        current_set = []
        
        # Start with least common numbers
        for number in least_common:
            if len(current_set) < numbers_per_set and 1 <= number <= max_number:
                current_set.append(number)
        
        # Fill remaining with rarely used numbers
        while len(current_set) < numbers_per_set:
            available = [n for n in range(1, max_number + 1) 
                        if n not in current_set and number_frequencies.get(n, 0) < 5]
            if available:
                current_set.append(np.random.choice(available))
            else:
                # Fallback to any available number
                all_available = [n for n in range(1, max_number + 1) if n not in current_set]
                if all_available:
                    current_set.append(np.random.choice(all_available))
        
        return current_set

    def _generate_balanced_complement_set(self, base_predictions: List[List[int]], 
                                        existing_sets: List[Dict[str, Any]], 
                                        numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a balanced complementary set"""
        # Track numbers already used in optimization
        used_numbers = set()
        for opt_set in existing_sets:
            used_numbers.update(opt_set['numbers'])
        
        # Track numbers from base predictions
        prediction_numbers = set()
        for pred_set in base_predictions:
            prediction_numbers.update(pred_set)
        
        current_set = []
        
        # Balance between new numbers and strategic selections
        new_numbers = [n for n in range(1, max_number + 1) 
                      if n not in used_numbers and n not in prediction_numbers]
        
        # Include some completely new numbers
        new_count = min(numbers_per_set // 2, len(new_numbers))
        if new_numbers:
            selected_new = np.random.choice(new_numbers, size=new_count, replace=False)
            current_set.extend(selected_new)
        
        # Fill remaining with strategic numbers
        while len(current_set) < numbers_per_set:
            available = [n for n in range(1, max_number + 1) if n not in current_set]
            if available:
                current_set.append(np.random.choice(available))
        
        return current_set

    def _calculate_gap_fill_score(self, number_set: List[int], 
                                gap_analysis: Dict[str, Any]) -> float:
        """Calculate how well this set fills identified gaps"""
        under_represented = {num for num, _ in gap_analysis['under_represented']}
        gap_fills = len(set(number_set).intersection(under_represented))
        return gap_fills / len(number_set) if number_set else 0

    def _calculate_complementary_score(self, number_set: List[int], 
                                     base_predictions: List[List[int]]) -> float:
        """Calculate how complementary this set is to base predictions"""
        prediction_numbers = set()
        for pred_set in base_predictions:
            prediction_numbers.update(pred_set)
        
        # Score based on uniqueness from base predictions
        unique_numbers = len([n for n in number_set if n not in prediction_numbers])
        return unique_numbers / len(number_set) if number_set else 0


class BalancedDistributionOptimizer(SetOptimizationStrategy):
    """Optimizes sets for balanced mathematical distribution"""
    
    def __init__(self):
        super().__init__("Balanced Distribution Optimizer")

    def optimize_sets(self, base_predictions: List[List[int]], game_config: Dict[str, Any], 
                     target_sets: int) -> List[Dict[str, Any]]:
        """Generate sets with optimal balanced distribution"""
        try:
            max_number = game_config.get('max_number', 49)
            numbers_per_set = game_config.get('numbers_per_set', 6)
            
            optimized_sets = []
            
            for set_id in range(target_sets):
                if set_id == 0:
                    # Perfect mathematical balance
                    current_set = self._generate_perfect_balance_set(numbers_per_set, max_number)
                elif set_id == 1:
                    # Fibonacci-based balance
                    current_set = self._generate_fibonacci_balance_set(numbers_per_set, max_number)
                else:
                    # Statistical balance based on common patterns
                    current_set = self._generate_statistical_balance_set(
                        base_predictions, numbers_per_set, max_number
                    )
                
                # Calculate balance metrics
                range_balance = self._calculate_range_balance(current_set, max_number)
                sum_balance = self._calculate_sum_balance(current_set, max_number, numbers_per_set)
                parity_balance = self._calculate_parity_balance(current_set)
                distribution_score = self._calculate_distribution_score(current_set, max_number)
                
                optimized_sets.append({
                    'set_id': set_id + 1,
                    'numbers': sorted(current_set),
                    'strategy': 'Balanced Distribution',
                    'range_balance': range_balance,
                    'sum_balance': sum_balance,
                    'parity_balance': parity_balance,
                    'distribution_score': distribution_score,
                    'combined_score': np.mean([range_balance, sum_balance, parity_balance, distribution_score]),
                    'optimizer': self.name
                })
            
            return optimized_sets
            
        except Exception as e:
            logger.error(f"Error in balanced distribution optimization: {e}")
            return []

    def _generate_perfect_balance_set(self, numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a perfectly balanced set mathematically"""
        # Divide the number range into equal segments
        segment_size = max_number / numbers_per_set
        current_set = []
        
        for i in range(numbers_per_set):
            # Select from each segment
            segment_start = int(i * segment_size) + 1
            segment_end = int((i + 1) * segment_size)
            
            # Ensure we don't exceed max_number
            segment_end = min(segment_end, max_number)
            
            if segment_start <= segment_end:
                # Pick middle of segment for perfect balance
                middle = (segment_start + segment_end) // 2
                current_set.append(middle)
        
        # Ensure we have exactly the right number of unique numbers
        current_set = list(set(current_set))
        while len(current_set) < numbers_per_set:
            available = [n for n in range(1, max_number + 1) if n not in current_set]
            if available:
                current_set.append(np.random.choice(available))
        
        return current_set[:numbers_per_set]

    def _generate_fibonacci_balance_set(self, numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a set using Fibonacci ratios for balance"""
        # Generate Fibonacci numbers up to max_number
        fib = [1, 1]
        while fib[-1] < max_number:
            fib.append(fib[-1] + fib[-2])
        
        fib_in_range = [f for f in fib if f <= max_number]
        
        current_set = []
        
        # Include some Fibonacci numbers
        if len(fib_in_range) >= 2:
            fib_count = min(numbers_per_set // 2, len(fib_in_range))
            selected_fibs = np.random.choice(fib_in_range, fib_count, replace=False)
            current_set.extend(selected_fibs)
        
        # Fill remaining with numbers that maintain golden ratio relationships
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        while len(current_set) < numbers_per_set:
            if current_set:
                # Try to find numbers that create golden ratio relationships
                last_number = max(current_set)
                target = int(last_number * golden_ratio)
                
                if target <= max_number and target not in current_set:
                    current_set.append(target)
                else:
                    # Fallback to available numbers
                    available = [n for n in range(1, max_number + 1) if n not in current_set]
                    if available:
                        current_set.append(np.random.choice(available))
            else:
                current_set.append(np.random.randint(1, max_number + 1))
        
        return current_set[:numbers_per_set]

    def _generate_statistical_balance_set(self, base_predictions: List[List[int]], 
                                        numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a statistically balanced set"""
        current_set = []
        
        # Analyze range distribution in base predictions
        if base_predictions:
            range_segments = self._analyze_range_distribution(base_predictions, max_number, 3)
            
            # Select numbers from each range segment proportionally
            for segment, target_count in range_segments.items():
                segment_start, segment_end = segment
                available_in_segment = list(range(segment_start, segment_end + 1))
                
                if available_in_segment and target_count > 0:
                    selected_count = min(target_count, len(available_in_segment))
                    selected = np.random.choice(available_in_segment, size=selected_count, replace=False)
                    current_set.extend(selected)
        
        # Fill remaining slots
        while len(current_set) < numbers_per_set:
            available = [n for n in range(1, max_number + 1) if n not in current_set]
            if available:
                current_set.append(np.random.choice(available))
        
        return current_set[:numbers_per_set]

    def _analyze_range_distribution(self, base_predictions: List[List[int]], 
                                  max_number: int, num_segments: int) -> Dict[Tuple[int, int], int]:
        """Analyze how numbers are distributed across range segments"""
        segment_size = max_number // num_segments
        segments = {}
        
        for i in range(num_segments):
            start = i * segment_size + 1
            end = min((i + 1) * segment_size, max_number)
            segments[(start, end)] = 0
        
        # Count numbers in each segment from base predictions
        total_numbers = 0
        for pred_set in base_predictions:
            for number in pred_set:
                total_numbers += 1
                for (start, end), count in segments.items():
                    if start <= number <= end:
                        segments[(start, end)] += 1
                        break
        
        # Convert to target distribution
        numbers_per_set = 6  # Assuming 6 numbers per set
        for segment in segments:
            proportion = segments[segment] / total_numbers if total_numbers > 0 else 1/num_segments
            segments[segment] = max(1, int(proportion * numbers_per_set))
        
        return segments

    def _calculate_range_balance(self, number_set: List[int], max_number: int) -> float:
        """Calculate how well balanced the set is across the number range"""
        if not number_set:
            return 0.0
        
        sorted_set = sorted(number_set)
        range_span = sorted_set[-1] - sorted_set[0]
        max_possible_span = max_number - 1
        
        return range_span / max_possible_span if max_possible_span > 0 else 0

    def _calculate_sum_balance(self, number_set: List[int], max_number: int, 
                             expected_count: int) -> float:
        """Calculate how close the sum is to expected average"""
        if not number_set:
            return 0.0
        
        actual_sum = sum(number_set)
        expected_sum = (max_number + 1) * expected_count / 2
        
        # Normalized distance from expected
        max_deviation = expected_sum
        deviation = abs(actual_sum - expected_sum)
        
        return max(0, 1 - (deviation / max_deviation)) if max_deviation > 0 else 0

    def _calculate_parity_balance(self, number_set: List[int]) -> float:
        """Calculate balance between odd and even numbers"""
        if not number_set:
            return 0.0
        
        even_count = sum(1 for n in number_set if n % 2 == 0)
        odd_count = len(number_set) - even_count
        
        # Perfect balance would be 50/50
        total = len(number_set)
        ideal_split = total / 2
        
        balance = 1 - abs(even_count - ideal_split) / ideal_split if ideal_split > 0 else 0
        return max(0, balance)

    def _calculate_distribution_score(self, number_set: List[int], max_number: int) -> float:
        """Calculate overall distribution quality score"""
        if not number_set or len(number_set) < 2:
            return 0.0
        
        sorted_set = sorted(number_set)
        gaps = [sorted_set[i+1] - sorted_set[i] for i in range(len(sorted_set) - 1)]
        
        # Ideal gap would be evenly distributed
        ideal_gap = max_number / len(number_set)
        
        # Score based on how close gaps are to ideal
        gap_scores = [1 - abs(gap - ideal_gap) / ideal_gap for gap in gaps]
        return np.mean(gap_scores) if gap_scores else 0


class UltraHighAccuracySetOptimizer:
    """Main set optimizer coordinating all optimization strategies"""
    
    def __init__(self):
        self.optimizers = {
            'coverage': CoverageOptimizer(),
            'complementary': ComplementarySetGenerator(),
            'balanced': BalancedDistributionOptimizer()
        }
        self.optimization_history = []
    
    def optimize_prediction_sets(self, base_predictions: List[List[int]], 
                               game_config: Dict[str, Any], target_sets: int = 10) -> Dict[str, Any]:
        """Perform comprehensive set optimization using all strategies"""
        try:
            logger.info("🔧 Performing ultra-high accuracy set optimization...")
            
            optimization_result = {
                'timestamp': datetime.now().isoformat(),
                'base_predictions_count': len(base_predictions),
                'target_sets': target_sets,
                'game_config': game_config,
                'optimized_sets': [],
                'strategy_results': {}
            }
            
            all_optimized_sets = []
            strategy_scores = {}
            
            # Run each optimization strategy
            for strategy_name, optimizer in self.optimizers.items():
                logger.info(f"Running {strategy_name} optimization...")
                
                strategy_sets = optimizer.optimize_sets(base_predictions, game_config, target_sets)
                
                # Tag sets with strategy info
                for opt_set in strategy_sets:
                    opt_set['optimization_strategy'] = strategy_name
                
                all_optimized_sets.extend(strategy_sets)
                
                # Calculate strategy performance
                if strategy_sets:
                    avg_score = np.mean([s.get('combined_score', 0) for s in strategy_sets])
                    strategy_scores[strategy_name] = {
                        'average_score': float(avg_score),
                        'sets_generated': len(strategy_sets)
                    }
                
                optimization_result['strategy_results'][strategy_name] = {
                    'sets_generated': len(strategy_sets),
                    'average_score': strategy_scores.get(strategy_name, {}).get('average_score', 0)
                }
            
            # Rank and select best sets across all strategies
            ranked_sets = self._rank_optimized_sets(all_optimized_sets, target_sets)
            optimization_result['optimized_sets'] = ranked_sets
            
            optimization_result['optimization_scores'] = strategy_scores
            
            # Calculate overall optimization quality
            optimization_result['overall_quality'] = self._calculate_overall_quality(
                optimization_result['optimized_sets'], base_predictions, game_config
            )
            
            self.optimization_history.append(optimization_result)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in set-based optimization: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'optimized_sets': [],
                'overall_quality': 0.0
            }

    def _rank_optimized_sets(self, all_sets: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """Rank optimized sets by multiple criteria"""
        try:
            # Calculate comprehensive scores for each set
            for opt_set in all_sets:
                score_factors = []
                
                # Include existing combined score
                if 'combined_score' in opt_set:
                    score_factors.append(opt_set['combined_score'])
                
                # Add strategy-specific bonuses
                strategy = opt_set.get('optimization_strategy', '')
                if strategy == 'coverage':
                    coverage_bonus = opt_set.get('coverage_score', 0) * 0.3
                    score_factors.append(coverage_bonus)
                elif strategy == 'complementary':
                    comp_bonus = opt_set.get('complementary_score', 0) * 0.3
                    score_factors.append(comp_bonus)
                elif strategy == 'balanced':
                    balance_bonus = opt_set.get('distribution_score', 0) * 0.3
                    score_factors.append(balance_bonus)
                
                # Calculate final ranking score
                opt_set['ranking_score'] = np.mean(score_factors) if score_factors else 0.5
            
            # Sort by ranking score
            ranked = sorted(all_sets, key=lambda x: x['ranking_score'], reverse=True)
            
            # Ensure diversity in selected sets
            selected = []
            strategy_counts = defaultdict(int)
            max_per_strategy = max(1, target_count // len(self.optimizers))
            
            # First pass: select best from each strategy
            for opt_set in ranked:
                strategy = opt_set.get('optimization_strategy', 'unknown')
                if (len(selected) < target_count and 
                    strategy_counts[strategy] < max_per_strategy):
                    selected.append(opt_set)
                    strategy_counts[strategy] += 1
            
            # Second pass: fill remaining slots with best available
            for opt_set in ranked:
                if opt_set not in selected and len(selected) < target_count:
                    selected.append(opt_set)
            
            return selected
            
        except Exception as e:
            logger.error(f"Error ranking optimized sets: {e}")
            return all_sets[:target_count]

    def _calculate_overall_quality(self, optimized_sets: List[Dict[str, Any]], 
                                 base_predictions: List[List[int]], 
                                 game_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality metrics for the optimization"""
        try:
            if not optimized_sets:
                return {'overall_score': 0.0}
            
            max_number = game_config.get('max_number', 49)
            
            # Extract just the number sets
            number_sets = [opt_set['numbers'] for opt_set in optimized_sets]
            
            # Calculate quality metrics
            coverage = self.optimizers['coverage'].calculate_set_coverage(number_sets, max_number)
            diversity = self.optimizers['coverage'].calculate_set_diversity(number_sets)
            
            # Calculate improvement over base predictions
            base_coverage = self.optimizers['coverage'].calculate_set_coverage(base_predictions, max_number)
            base_diversity = self.optimizers['coverage'].calculate_set_diversity(base_predictions)
            
            coverage_improvement = max(0, coverage - base_coverage)
            diversity_improvement = max(0, diversity - base_diversity)
            
            # Overall quality score
            quality_factors = [
                coverage,
                diversity,
                coverage_improvement * 2,  # Weight improvements higher
                diversity_improvement * 2
            ]
            
            overall_score = np.mean(quality_factors)
            
            return {
                'overall_score': float(overall_score),
                'coverage': float(coverage),
                'diversity': float(diversity),
                'coverage_improvement': float(coverage_improvement),
                'diversity_improvement': float(diversity_improvement)
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return {'overall_score': 0.0}


class SetOptimizer:
    """
    Set Optimizer Engine for lottery predictions.
    
    Implements coverage optimization strategies:
    - Maximum coverage analysis
    - Hit rate optimization
    - Combinatorial coverage
    - Strategic set generation
    - Cluster-based optimization
    - Wheeling systems
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Set Optimizer Engine.
        
        Args:
            config: Configuration dictionary containing engine parameters
        """
        self.config = config
        self.game_config = config.get('game_config', {})
        self.optimizer_config = config.get('optimizer_config', {})
        
        # Game parameters
        self.number_range = self.game_config.get('number_range', [1, 49])
        self.numbers_per_draw = self.game_config.get('numbers_per_draw', 6)
        self.has_bonus = self.game_config.get('has_bonus', False)
        
        # Optimization parameters
        self.coverage_strategy = self.optimizer_config.get('coverage_strategy', 'balanced')
        self.max_combinations = self.optimizer_config.get('max_combinations', 10000)
        self.hit_rate_target = self.optimizer_config.get('hit_rate_target', 0.3)
        self.diversity_factor = self.optimizer_config.get('diversity_factor', 0.7)
        self.clustering_enabled = self.optimizer_config.get('clustering_enabled', True)
        self.wheeling_system = self.optimizer_config.get('wheeling_system', 'abbreviated')
        
        # Data structures
        self.historical_data = None
        self.number_frequencies = {}
        self.combination_patterns = {}
        self.coverage_matrix = None
        self.optimal_sets = []
        self.cluster_assignments = {}
        self.wheeling_systems = {}
        self.trained = False
        
        # Initialize sophisticated set optimizer
        self.ultra_optimizer = UltraHighAccuracySetOptimizer()
        
        logger.info("🎯 Enhanced Set Optimizer Engine initialized with sophisticated strategies")
    
    def load_data(self, historical_data: pd.DataFrame) -> None:
        """
        Load historical lottery data for optimization analysis.
        
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
            
            # Process numbers format
            self._process_numbers_format()
            
            logger.info(f"📊 Loaded {len(self.historical_data)} historical draws for optimization")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def _process_numbers_format(self) -> None:
        """Process numbers column to ensure consistent format."""
        processed_numbers = []
        
        for _, row in self.historical_data.iterrows():
            numbers = row['numbers']
            
            if isinstance(numbers, str):
                number_list = [int(x.strip()) for x in numbers.split(',')]
            elif isinstance(numbers, list):
                number_list = [int(x) for x in numbers]
            else:
                raise ValueError(f"Invalid numbers format: {numbers}")
            
            # Validate number range
            for num in number_list:
                if not (self.number_range[0] <= num <= self.number_range[1]):
                    raise ValueError(f"Number {num} outside valid range {self.number_range}")
            
            processed_numbers.append(sorted(number_list))
        
        self.historical_data['processed_numbers'] = processed_numbers
    
    def analyze_coverage_patterns(self) -> Dict[str, Any]:
        """
        Analyze coverage patterns in historical data.
        
        Returns:
            Dictionary containing coverage analysis results
        """
        try:
            # Analyze individual number frequencies
            self._analyze_number_frequencies()
            
            # Analyze combination patterns
            self._analyze_combination_patterns()
            
            # Build coverage matrix
            self._build_coverage_matrix()
            
            # Analyze hit patterns
            hit_analysis = self._analyze_hit_patterns()
            
            coverage_results = {
                'number_frequencies': self.number_frequencies,
                'combination_patterns': self.combination_patterns,
                'hit_analysis': hit_analysis,
                'coverage_matrix_shape': self.coverage_matrix.shape if self.coverage_matrix is not None else None
            }
            
            logger.info("📊 Coverage pattern analysis complete")
            return coverage_results
            
        except Exception as e:
            logger.error(f"❌ Coverage analysis failed: {e}")
            raise
    
    def _analyze_number_frequencies(self) -> None:
        """Analyze individual number frequencies and patterns."""
        all_numbers = []
        for numbers in self.historical_data['processed_numbers']:
            all_numbers.extend(numbers)
        
        frequency_counter = Counter(all_numbers)
        total_draws = len(self.historical_data)
        
        for num in range(self.number_range[0], self.number_range[1] + 1):
            count = frequency_counter.get(num, 0)
            frequency = count / total_draws if total_draws > 0 else 0
            
            self.number_frequencies[num] = {
                'count': count,
                'frequency': frequency,
                'hit_rate': frequency,
                'expected_frequency': self.numbers_per_draw / (self.number_range[1] - self.number_range[0] + 1)
            }
    
    def _analyze_combination_patterns(self) -> None:
        """Analyze patterns in number combinations."""
        # Analyze pair frequencies
        pair_frequencies = defaultdict(int)
        triple_frequencies = defaultdict(int)
        
        for numbers in self.historical_data['processed_numbers']:
            # Analyze pairs
            for pair in combinations(numbers, 2):
                pair_frequencies[tuple(sorted(pair))] += 1
            
            # Analyze triples
            for triple in combinations(numbers, 3):
                triple_frequencies[tuple(sorted(triple))] += 1
        
        # Convert to relative frequencies
        total_draws = len(self.historical_data)
        
        self.combination_patterns = {
            'pairs': {pair: count / total_draws for pair, count in pair_frequencies.items()},
            'triples': {triple: count / total_draws for triple, count in triple_frequencies.items()},
            'total_pairs': len(pair_frequencies),
            'total_triples': len(triple_frequencies)
        }
    
    def _build_coverage_matrix(self) -> None:
        """Build coverage matrix for optimization analysis."""
        try:
            num_count = self.number_range[1] - self.number_range[0] + 1
            
            # Create matrix where rows are draws and columns are numbers
            coverage_matrix = np.zeros((len(self.historical_data), num_count))
            
            for i, numbers in enumerate(self.historical_data['processed_numbers']):
                for num in numbers:
                    coverage_matrix[i, num - self.number_range[0]] = 1
            
            self.coverage_matrix = coverage_matrix
            
        except Exception as e:
            logger.error(f"❌ Coverage matrix building failed: {e}")
            raise
    
    def _analyze_hit_patterns(self) -> Dict[str, Any]:
        """Analyze hit patterns for different combination strategies."""
        hit_analysis = {
            'single_number_hits': {},
            'pair_hits': {},
            'coverage_scores': {},
            'optimal_combinations': []
        }
        
        # Analyze single number hit rates
        for num in range(self.number_range[0], self.number_range[1] + 1):
            hits = sum(1 for numbers in self.historical_data['processed_numbers'] if num in numbers)
            hit_rate = hits / len(self.historical_data)
            hit_analysis['single_number_hits'][num] = hit_rate
        
        # Analyze pair hit rates (sample due to computational complexity)
        sample_pairs = list(combinations(range(self.number_range[0], self.number_range[1] + 1), 2))[:1000]
        
        for pair in sample_pairs:
            hits = sum(1 for numbers in self.historical_data['processed_numbers'] 
                      if pair[0] in numbers and pair[1] in numbers)
            hit_rate = hits / len(self.historical_data)
            if hit_rate > 0:
                hit_analysis['pair_hits'][pair] = hit_rate
        
        return hit_analysis
    
    def optimize_sets(self, strategy: str = 'balanced') -> List[Dict[str, Any]]:
        """
        Optimize number sets based on specified strategy.
        
        Args:
            strategy: Optimization strategy ('balanced', 'coverage', 'hit_rate', 'diversity')
            
        Returns:
            List of optimized number sets
        """
        try:
            if strategy == 'balanced':
                optimal_sets = self._optimize_balanced_sets()
            elif strategy == 'coverage':
                optimal_sets = self._optimize_coverage_sets()
            elif strategy == 'hit_rate':
                optimal_sets = self._optimize_hit_rate_sets()
            elif strategy == 'diversity':
                optimal_sets = self._optimize_diversity_sets()
            else:
                optimal_sets = self._optimize_balanced_sets()
            
            self.optimal_sets = optimal_sets
            
            logger.info(f"🎯 Generated {len(optimal_sets)} optimized sets using {strategy} strategy")
            return optimal_sets
            
        except Exception as e:
            logger.error(f"❌ Set optimization failed: {e}")
            raise
    
    def _optimize_balanced_sets(self) -> List[Dict[str, Any]]:
        """Optimize sets using balanced approach."""
        sets = []
        
        # Create sets balancing frequency, coverage, and diversity
        for i in range(min(10, self.max_combinations // 100)):
            # Start with high-frequency numbers
            high_freq_numbers = sorted(self.number_frequencies.items(), 
                                     key=lambda x: x[1]['frequency'], reverse=True)[:15]
            
            # Add some medium frequency numbers for balance
            medium_freq_numbers = sorted(self.number_frequencies.items(), 
                                       key=lambda x: x[1]['frequency'])[15:35]
            
            # Combine with strategic selection
            candidate_numbers = [num for num, _ in high_freq_numbers[:10]] + \
                              [num for num, _ in medium_freq_numbers[:10]]
            
            # Select final combination
            selected_numbers = self._select_optimal_combination(candidate_numbers, 'balanced')
            
            # Calculate scores
            coverage_score = self._calculate_coverage_score(selected_numbers)
            hit_rate_score = self._calculate_hit_rate_score(selected_numbers)
            diversity_score = self._calculate_diversity_score(selected_numbers)
            
            combined_score = (coverage_score + hit_rate_score + diversity_score) / 3
            
            sets.append({
                'numbers': selected_numbers,
                'coverage_score': coverage_score,
                'hit_rate_score': hit_rate_score,
                'diversity_score': diversity_score,
                'combined_score': combined_score,
                'strategy': 'balanced'
            })
        
        return sorted(sets, key=lambda x: x['combined_score'], reverse=True)
    
    def _optimize_coverage_sets(self) -> List[Dict[str, Any]]:
        """Optimize sets for maximum coverage."""
        sets = []
        
        # Use greedy approach for coverage maximization
        for i in range(min(5, self.max_combinations // 200)):
            selected_numbers = []
            available_numbers = list(range(self.number_range[0], self.number_range[1] + 1))
            
            # Greedy selection for maximum coverage
            for _ in range(self.numbers_per_draw):
                best_number = None
                best_coverage = -1
                
                for num in available_numbers:
                    test_set = selected_numbers + [num]
                    coverage = self._calculate_coverage_score(test_set)
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_number = num
                
                if best_number is not None:
                    selected_numbers.append(best_number)
                    available_numbers.remove(best_number)
            
            coverage_score = self._calculate_coverage_score(selected_numbers)
            hit_rate_score = self._calculate_hit_rate_score(selected_numbers)
            diversity_score = self._calculate_diversity_score(selected_numbers)
            
            sets.append({
                'numbers': sorted(selected_numbers),
                'coverage_score': coverage_score,
                'hit_rate_score': hit_rate_score,
                'diversity_score': diversity_score,
                'combined_score': coverage_score * 0.6 + hit_rate_score * 0.3 + diversity_score * 0.1,
                'strategy': 'coverage'
            })
        
        return sorted(sets, key=lambda x: x['coverage_score'], reverse=True)
    
    def _optimize_hit_rate_sets(self) -> List[Dict[str, Any]]:
        """Optimize sets for maximum hit rate potential."""
        sets = []
        
        # Focus on numbers with highest individual hit rates
        hit_rates = [(num, data['hit_rate']) for num, data in self.number_frequencies.items()]
        sorted_by_hit_rate = sorted(hit_rates, key=lambda x: x[1], reverse=True)
        
        # Create combinations of high hit rate numbers
        for i in range(min(8, self.max_combinations // 150)):
            if i == 0:
                # Best hit rate numbers
                selected_numbers = [num for num, _ in sorted_by_hit_rate[:self.numbers_per_draw]]
            else:
                # Mix of high and medium hit rate numbers
                high_hit = [num for num, _ in sorted_by_hit_rate[:10]]
                medium_hit = [num for num, _ in sorted_by_hit_rate[10:25]]
                
                selected_numbers = np.random.choice(
                    high_hit + medium_hit, 
                    size=self.numbers_per_draw, 
                    replace=False
                ).tolist()
            
            coverage_score = self._calculate_coverage_score(selected_numbers)
            hit_rate_score = self._calculate_hit_rate_score(selected_numbers)
            diversity_score = self._calculate_diversity_score(selected_numbers)
            
            sets.append({
                'numbers': sorted(selected_numbers),
                'coverage_score': coverage_score,
                'hit_rate_score': hit_rate_score,
                'diversity_score': diversity_score,
                'combined_score': hit_rate_score * 0.6 + coverage_score * 0.3 + diversity_score * 0.1,
                'strategy': 'hit_rate'
            })
        
        return sorted(sets, key=lambda x: x['hit_rate_score'], reverse=True)
    
    def _optimize_diversity_sets(self) -> List[Dict[str, Any]]:
        """Optimize sets for maximum diversity."""
        sets = []
        
        if self.clustering_enabled:
            # Use clustering to create diverse sets
            clusters = self._create_number_clusters()
            
            for i in range(min(6, self.max_combinations // 180)):
                selected_numbers = []
                
                # Select numbers from different clusters
                for cluster_id in range(min(self.numbers_per_draw, len(clusters))):
                    if cluster_id < len(clusters) and len(clusters[cluster_id]) > 0:
                        selected_numbers.append(np.random.choice(clusters[cluster_id]))
                
                # Fill remaining slots if needed
                while len(selected_numbers) < self.numbers_per_draw:
                    available = [n for n in range(self.number_range[0], self.number_range[1] + 1) 
                               if n not in selected_numbers]
                    if available:
                        selected_numbers.append(np.random.choice(available))
                    else:
                        break
                
                coverage_score = self._calculate_coverage_score(selected_numbers)
                hit_rate_score = self._calculate_hit_rate_score(selected_numbers)
                diversity_score = self._calculate_diversity_score(selected_numbers)
                
                sets.append({
                    'numbers': sorted(selected_numbers),
                    'coverage_score': coverage_score,
                    'hit_rate_score': hit_rate_score,
                    'diversity_score': diversity_score,
                    'combined_score': diversity_score * 0.5 + coverage_score * 0.3 + hit_rate_score * 0.2,
                    'strategy': 'diversity'
                })
        else:
            # Simple diversity approach
            for i in range(min(6, self.max_combinations // 180)):
                selected_numbers = []
                available_numbers = list(range(self.number_range[0], self.number_range[1] + 1))
                
                # Spread numbers across range
                step = len(available_numbers) // self.numbers_per_draw
                for j in range(self.numbers_per_draw):
                    start_idx = j * step
                    end_idx = min(start_idx + step, len(available_numbers))
                    if start_idx < len(available_numbers):
                        selected_numbers.append(available_numbers[start_idx + (i % step)])
                
                coverage_score = self._calculate_coverage_score(selected_numbers)
                hit_rate_score = self._calculate_hit_rate_score(selected_numbers)
                diversity_score = self._calculate_diversity_score(selected_numbers)
                
                sets.append({
                    'numbers': sorted(selected_numbers),
                    'coverage_score': coverage_score,
                    'hit_rate_score': hit_rate_score,
                    'diversity_score': diversity_score,
                    'combined_score': diversity_score * 0.5 + coverage_score * 0.3 + hit_rate_score * 0.2,
                    'strategy': 'diversity'
                })
        
        return sorted(sets, key=lambda x: x['diversity_score'], reverse=True)
    
    def _create_number_clusters(self) -> List[List[int]]:
        """Create clusters of numbers based on appearance patterns."""
        try:
            if self.coverage_matrix is None:
                return []
            
            # Transpose to get number patterns across draws
            number_patterns = self.coverage_matrix.T
            
            # Apply K-means clustering
            n_clusters = min(8, self.numbers_per_draw + 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(number_patterns)
            
            # Group numbers by cluster
            clusters = [[] for _ in range(n_clusters)]
            for i, label in enumerate(cluster_labels):
                number = i + self.number_range[0]
                clusters[label].append(number)
            
            return [cluster for cluster in clusters if len(cluster) > 0]
            
        except Exception as e:
            logger.warning(f"⚠️ Clustering failed: {e}")
            return []
    
    def _select_optimal_combination(self, candidate_numbers: List[int], strategy: str) -> List[int]:
        """Select optimal combination from candidate numbers."""
        if len(candidate_numbers) <= self.numbers_per_draw:
            return candidate_numbers[:self.numbers_per_draw]
        
        best_combination = None
        best_score = -1
        
        # Sample combinations due to computational complexity
        max_combinations_to_test = min(1000, math.comb(len(candidate_numbers), self.numbers_per_draw))
        
        for _ in range(max_combinations_to_test):
            combination = np.random.choice(candidate_numbers, size=self.numbers_per_draw, replace=False).tolist()
            
            if strategy == 'balanced':
                score = (self._calculate_coverage_score(combination) + 
                        self._calculate_hit_rate_score(combination) + 
                        self._calculate_diversity_score(combination)) / 3
            elif strategy == 'coverage':
                score = self._calculate_coverage_score(combination)
            elif strategy == 'hit_rate':
                score = self._calculate_hit_rate_score(combination)
            else:
                score = self._calculate_diversity_score(combination)
            
            if score > best_score:
                best_score = score
                best_combination = combination
        
        return sorted(best_combination) if best_combination else candidate_numbers[:self.numbers_per_draw]
    
    def _calculate_coverage_score(self, numbers: List[int]) -> float:
        """Calculate coverage score for a set of numbers."""
        try:
            if not numbers:
                return 0.0
            
            # Calculate how well this combination covers historical patterns
            coverage_count = 0
            total_draws = len(self.historical_data)
            
            for historical_numbers in self.historical_data['processed_numbers']:
                matches = len(set(numbers) & set(historical_numbers))
                if matches >= 2:  # At least 2 numbers match
                    coverage_count += matches / self.numbers_per_draw
            
            return coverage_count / total_draws if total_draws > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ Coverage score calculation failed: {e}")
            return 0.0
    
    def _calculate_hit_rate_score(self, numbers: List[int]) -> float:
        """Calculate hit rate score for a set of numbers."""
        try:
            if not numbers:
                return 0.0
            
            # Average hit rate of individual numbers
            total_hit_rate = sum(self.number_frequencies[num]['hit_rate'] for num in numbers)
            return total_hit_rate / len(numbers)
            
        except Exception as e:
            logger.warning(f"⚠️ Hit rate score calculation failed: {e}")
            return 0.0
    
    def _calculate_diversity_score(self, numbers: List[int]) -> float:
        """Calculate diversity score for a set of numbers."""
        try:
            if len(numbers) < 2:
                return 0.0
            
            # Calculate spread across number range
            sorted_numbers = sorted(numbers)
            gaps = [sorted_numbers[i] - sorted_numbers[i-1] for i in range(1, len(sorted_numbers))]
            
            # Ideal gap would be evenly distributed
            total_range = self.number_range[1] - self.number_range[0]
            ideal_gap = total_range / self.numbers_per_draw
            
            # Calculate how close gaps are to ideal
            gap_score = 1.0 - (sum(abs(gap - ideal_gap) for gap in gaps) / (len(gaps) * ideal_gap))
            
            # Add odd/even balance
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            even_count = len(numbers) - odd_count
            balance_score = 1.0 - abs(odd_count - even_count) / len(numbers)
            
            return (gap_score + balance_score) / 2
            
        except Exception as e:
            logger.warning(f"⚠️ Diversity score calculation failed: {e}")
            return 0.0
    
    def train(self) -> None:
        """Train the set optimizer on historical data."""
        try:
            if self.historical_data is None:
                raise ValueError("No historical data loaded")
            
            logger.info("🎓 Training Set Optimizer...")
            
            # Analyze coverage patterns
            self.analyze_coverage_patterns()
            
            # Generate optimal sets for different strategies
            strategies = ['balanced', 'coverage', 'hit_rate', 'diversity']
            all_optimal_sets = []
            
            for strategy in strategies:
                strategy_sets = self.optimize_sets(strategy)
                all_optimal_sets.extend(strategy_sets[:3])  # Top 3 from each strategy
            
            self.optimal_sets = all_optimal_sets
            self.trained = True
            
            logger.info("✅ Set Optimizer training complete")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def predict(self, num_predictions: int = 1, strategy: str = 'balanced') -> List[Dict[str, Any]]:
        """
        Generate predictions using set optimization.
        
        Args:
            num_predictions: Number of prediction sets to generate
            strategy: Optimization strategy ('balanced', 'coverage', 'hit_rate', 'diversity')
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if not self.trained:
                raise ValueError("Optimizer not trained. Call train() first.")
            
            predictions = []
            
            # Get optimized sets for the specified strategy
            strategy_sets = [s for s in self.optimal_sets if s['strategy'] == strategy]
            if not strategy_sets:
                # Fallback to any available sets
                strategy_sets = self.optimal_sets[:num_predictions]
            
            for i in range(num_predictions):
                if i < len(strategy_sets):
                    optimal_set = strategy_sets[i]
                    numbers = optimal_set['numbers']
                    confidence = optimal_set['combined_score']
                else:
                    # Generate new optimized set
                    new_sets = self.optimize_sets(strategy)
                    if new_sets:
                        optimal_set = new_sets[0]
                        numbers = optimal_set['numbers']
                        confidence = optimal_set['combined_score']
                    else:
                        # Fallback to random selection
                        numbers = sorted(np.random.choice(
                            range(self.number_range[0], self.number_range[1] + 1),
                            size=self.numbers_per_draw,
                            replace=False
                        ).tolist())
                        confidence = 0.3
                
                predictions.append({
                    'numbers': numbers,
                    'confidence': min(0.95, confidence),
                    'strategy': strategy,
                    'optimization_scores': {
                        'coverage': self._calculate_coverage_score(numbers),
                        'hit_rate': self._calculate_hit_rate_score(numbers),
                        'diversity': self._calculate_diversity_score(numbers)
                    },
                    'generated_at': datetime.now().isoformat(),
                    'engine': 'set_optimizer'
                })
            
            logger.info(f"🎯 Generated {num_predictions} optimized predictions using {strategy} strategy")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            raise
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization summary.
        
        Returns:
            Dictionary containing optimization summary
        """
        if not self.trained:
            return {'error': 'Optimizer not trained'}
        
        return {
            'engine': 'set_optimizer',
            'total_optimal_sets': len(self.optimal_sets),
            'strategies_available': ['balanced', 'coverage', 'hit_rate', 'diversity'],
            'coverage_patterns': {
                'total_combinations_analyzed': len(self.combination_patterns.get('pairs', {})),
                'total_numbers': len(self.number_frequencies)
            },
            'optimization_metrics': {
                'max_coverage_score': max([s['coverage_score'] for s in self.optimal_sets]) if self.optimal_sets else 0,
                'max_hit_rate_score': max([s['hit_rate_score'] for s in self.optimal_sets]) if self.optimal_sets else 0,
                'max_diversity_score': max([s['diversity_score'] for s in self.optimal_sets]) if self.optimal_sets else 0
            },
            'clustering_enabled': self.clustering_enabled,
            'wheeling_system': self.wheeling_system
        }
    
    def get_set_analysis(self, numbers: List[int]) -> Dict[str, Any]:
        """
        Get detailed analysis for a specific number set.
        
        Args:
            numbers: List of numbers to analyze
            
        Returns:
            Dictionary containing detailed set analysis
        """
        if not self.trained:
            return {'error': 'Optimizer not trained'}
        
        return {
            'numbers': sorted(numbers),
            'coverage_score': self._calculate_coverage_score(numbers),
            'hit_rate_score': self._calculate_hit_rate_score(numbers),
            'diversity_score': self._calculate_diversity_score(numbers),
            'individual_frequencies': {num: self.number_frequencies.get(num, {}) for num in numbers},
            'pair_analysis': self._analyze_pairs_in_set(numbers),
            'optimization_rank': self._get_set_optimization_rank(numbers)
        }
    
    def _analyze_pairs_in_set(self, numbers: List[int]) -> Dict[str, Any]:
        """Analyze pairs within a number set."""
        pairs_in_set = list(combinations(numbers, 2))
        pair_analysis = {
            'total_pairs': len(pairs_in_set),
            'pair_frequencies': {},
            'strong_pairs': [],
            'weak_pairs': []
        }
        
        for pair in pairs_in_set:
            pair_key = tuple(sorted(pair))
            frequency = self.combination_patterns.get('pairs', {}).get(pair_key, 0)
            pair_analysis['pair_frequencies'][pair_key] = frequency
            
            if frequency > 0.1:  # Arbitrary threshold
                pair_analysis['strong_pairs'].append(pair_key)
            elif frequency < 0.05:
                pair_analysis['weak_pairs'].append(pair_key)
        
        return pair_analysis
    
    def _get_set_optimization_rank(self, numbers: List[int]) -> int:
        """Get the optimization rank of a number set."""
        if not self.optimal_sets:
            return -1
        
        set_score = (self._calculate_coverage_score(numbers) + 
                    self._calculate_hit_rate_score(numbers) + 
                    self._calculate_diversity_score(numbers)) / 3
        
        # Count how many optimal sets have higher scores
        rank = 1
        for optimal_set in self.optimal_sets:
            if optimal_set['combined_score'] > set_score:
                rank += 1
        
        return rank

    def optimize_prediction_sets_advanced(self, base_predictions: List[List[int]], 
                                        target_sets: int = 10) -> Dict[str, Any]:
        """Perform advanced set optimization using sophisticated strategies"""
        try:
            logger.info("🚀 Performing advanced set optimization with multiple strategies...")
            
            # Prepare game config for sophisticated optimizer
            game_config = {
                'max_number': self.number_range[1],
                'numbers_per_set': self.numbers_per_draw,
                'has_bonus': self.has_bonus
            }
            
            # Run sophisticated optimization
            optimization_result = self.ultra_optimizer.optimize_prediction_sets(
                base_predictions, game_config, target_sets
            )
            
            # Store optimized sets for future reference
            self.optimal_sets = optimization_result.get('optimized_sets', [])
            
            logger.info(f"✅ Advanced optimization complete. Generated {len(self.optimal_sets)} optimized sets")
            logger.info(f"📊 Overall quality score: {optimization_result.get('overall_quality', {}).get('overall_score', 0):.3f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"❌ Error in advanced set optimization: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'optimized_sets': [],
                'overall_quality': {'overall_score': 0.0}
            }

    def generate_set_optimization_insights(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from set optimization results"""
        try:
            insights = {
                'optimization_insights': [],
                'strategy_performance': {},
                'coverage_analysis': {},
                'recommendations': []
            }
            
            # Extract optimization data
            optimized_sets = optimization_result.get('optimized_sets', [])
            strategy_results = optimization_result.get('strategy_results', {})
            overall_quality = optimization_result.get('overall_quality', {})
            
            # Analyze strategy performance
            for strategy, results in strategy_results.items():
                avg_score = results.get('average_score', 0)
                sets_count = results.get('sets_generated', 0)
                
                insights['strategy_performance'][strategy] = {
                    'average_score': avg_score,
                    'sets_generated': sets_count,
                    'performance_level': 'High' if avg_score > 0.7 else 'Medium' if avg_score > 0.4 else 'Low'
                }
                
                # Generate strategy-specific insights
                if strategy == 'coverage' and avg_score > 0.6:
                    insights['optimization_insights'].append("🎯 Coverage optimization performed well - good number spread achieved")
                elif strategy == 'complementary' and avg_score > 0.6:
                    insights['optimization_insights'].append("🔄 Complementary sets provide good gap filling")
                elif strategy == 'balanced' and avg_score > 0.6:
                    insights['optimization_insights'].append("⚖️ Balanced distribution strategies showing strong performance")
            
            # Coverage analysis
            coverage = overall_quality.get('coverage', 0)
            diversity = overall_quality.get('diversity', 0)
            coverage_improvement = overall_quality.get('coverage_improvement', 0)
            
            insights['coverage_analysis'] = {
                'total_coverage': float(coverage),
                'set_diversity': float(diversity),
                'improvement_over_base': float(coverage_improvement),
                'coverage_quality': 'Excellent' if coverage > 0.8 else 'Good' if coverage > 0.6 else 'Moderate'
            }
            
            # Generate recommendations
            overall_score = overall_quality.get('overall_score', 0)
            if overall_score > 0.7:
                insights['recommendations'].append("🌟 High-quality optimization achieved - consider using these sets")
            elif overall_score > 0.4:
                insights['recommendations'].append("✅ Moderate optimization quality - combine with other analysis methods")
            else:
                insights['recommendations'].append("⚠️ Consider additional optimization parameters or more base predictions")
            
            if coverage_improvement > 0.1:
                insights['recommendations'].append("📈 Significant coverage improvement - optimization adds value")
            
            if diversity > 0.8:
                insights['recommendations'].append("🎲 High set diversity achieved - good risk distribution")
            
            # Best performing sets analysis
            if optimized_sets:
                top_sets = sorted(optimized_sets, key=lambda x: x.get('ranking_score', 0), reverse=True)[:3]
                insights['top_performing_sets'] = [
                    {
                        'numbers': s['numbers'],
                        'strategy': s.get('optimization_strategy', 'unknown'),
                        'score': s.get('ranking_score', 0)
                    }
                    for s in top_sets
                ]
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generating set optimization insights: {e}")
            return {
                'optimization_insights': ["Error generating insights"],
                'strategy_performance': {},
                'coverage_analysis': {},
                'recommendations': ["Unable to generate recommendations"]
            }

    def get_set_optimizer_insights(self, base_predictions: List[List[int]], 
                                 target_sets: int = 10) -> Dict[str, Any]:
        """Get comprehensive set optimizer insights with advanced optimization"""
        try:
            logger.info("🔧 Generating comprehensive set optimization insights...")
            
            # Perform advanced optimization
            optimization_result = self.optimize_prediction_sets_advanced(base_predictions, target_sets)
            
            # Generate insights
            optimization_insights = self.generate_set_optimization_insights(optimization_result)
            
            # Combine all insights
            comprehensive_insights = {
                'optimization_result': optimization_result,
                'optimization_insights': optimization_insights,
                'analysis_metadata': {
                    'base_predictions_analyzed': len(base_predictions),
                    'target_sets': target_sets,
                    'optimization_quality': optimization_result.get('overall_quality', {}).get('overall_score', 0.0),
                    'strategies_used': list(optimization_result.get('strategy_results', {}).keys()),
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Set optimization insights generated with quality score: {comprehensive_insights['analysis_metadata']['optimization_quality']:.3f}")
            
            return comprehensive_insights
            
        except Exception as e:
            logger.error(f"❌ Error getting set optimizer insights: {e}")
            return {
                'optimization_result': {'optimized_sets': [], 'overall_quality': {'overall_score': 0.0}},
                'optimization_insights': {'optimization_insights': ["Error generating insights"]},
                'error': str(e),
                'analysis_metadata': {
                    'base_predictions_analyzed': 0,
                    'optimization_quality': 0.0,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }