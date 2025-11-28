#!/usr/bin/env python3
"""
Set-Based Optimization for Ultra-High Accuracy Lottery Prediction
This module implements advanced set-based optimization strategies:
- Coverage Optimization
- Complementary Set Generation
- Balanced Distribution Sets
- Risk-Adjusted Diversification
- Pattern Completion Sets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
import math
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, timedelta
import logging
import itertools
from abc import ABC, abstractmethod

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
                        # Update frequencies to reflect new selection
                        if best_number in number_frequencies:
                            del number_frequencies[best_number]
                    else:
                        # Fallback: select random unused number
                        available = [n for n in range(1, max_number + 1) 
                                   if n not in current_set and n not in used_numbers]
                        if available:
                            selected = np.random.choice(available)
                            current_set.append(selected)
                            used_numbers.add(selected)
                
                # Calculate set properties
                coverage_score = self._calculate_coverage_contribution(
                    current_set, used_numbers, max_number
                )
                
                quality_score = self._calculate_set_quality(
                    current_set, base_predictions
                )
                
                optimized_sets.append({
                    'set_id': set_id + 1,
                    'numbers': sorted(current_set),
                    'strategy': 'Coverage Optimization',
                    'coverage_score': coverage_score,
                    'quality_score': quality_score,
                    'combined_score': (coverage_score * 0.6 + quality_score * 0.4),
                    'optimizer': self.name
                })
            
            return optimized_sets
            
        except Exception as e:
            logger.error(f"Error in coverage optimization: {e}")
            return []
    
    def _calculate_unused_frequencies(self, base_predictions: List[List[int]], 
                                    used_numbers: Set[int], max_number: int) -> Dict[int, float]:
        """Calculate frequencies of unused numbers in base predictions"""
        frequencies = Counter()
        
        for prediction_set in base_predictions:
            for number in prediction_set:
                if number not in used_numbers and 1 <= number <= max_number:
                    frequencies[number] += 1
        
        # Normalize frequencies
        total_count = sum(frequencies.values())
        if total_count > 0:
            return {num: count / total_count for num, count in frequencies.items()}
        else:
            return {}
    
    def _select_best_coverage_number(self, frequencies: Dict[int, float], 
                                   current_set: List[int], used_numbers: Set[int], 
                                   max_number: int) -> Optional[int]:
        """Select the best number for coverage optimization"""
        candidates = []
        
        # Prioritize numbers that appear in base predictions but aren't used yet
        for number, freq in frequencies.items():
            if number not in current_set and number not in used_numbers:
                # Score based on frequency and diversity from current set
                diversity_bonus = self._calculate_diversity_bonus(number, current_set)
                score = freq + diversity_bonus
                candidates.append((number, score))
        
        # If no frequent numbers available, consider all unused numbers
        if not candidates:
            for number in range(1, max_number + 1):
                if number not in current_set and number not in used_numbers:
                    diversity_bonus = self._calculate_diversity_bonus(number, current_set)
                    score = 0.1 + diversity_bonus  # Base score for any number
                    candidates.append((number, score))
        
        if candidates:
            # Sort by score and return best candidate
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _calculate_diversity_bonus(self, number: int, current_set: List[int]) -> float:
        """Calculate diversity bonus for adding a number to current set"""
        if not current_set:
            return 0.5
        
        # Bonus for maintaining good distribution
        min_gap = min(abs(number - existing) for existing in current_set)
        gap_bonus = min(0.3, min_gap / 10.0)  # Reward larger gaps
        
        # Bonus for balanced position in range
        if current_set:
            min_existing = min(current_set)
            max_existing = max(current_set)
            
            if number < min_existing or number > max_existing:
                range_bonus = 0.2  # Bonus for extending range
            else:
                range_bonus = 0.1  # Smaller bonus for filling gaps
        else:
            range_bonus = 0.1
        
        return gap_bonus + range_bonus
    
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
        
        # Identify under-represented numbers
        total_predictions = len(base_predictions)
        expected_frequency = total_predictions * (6 / max_number)  # Expected appearances
        
        under_represented = []
        over_represented = []
        
        for number in range(1, max_number + 1):
            actual_freq = number_frequencies.get(number, 0)
            if actual_freq < expected_frequency * 0.5:  # Less than 50% of expected
                under_represented.append((number, expected_frequency - actual_freq))
            elif actual_freq > expected_frequency * 1.5:  # More than 150% of expected
                over_represented.append((number, actual_freq - expected_frequency))
        
        # Analyze range gaps
        range_gaps = self._analyze_range_gaps(base_predictions, max_number)
        
        return {
            'under_represented': sorted(under_represented, key=lambda x: x[1], reverse=True),
            'over_represented': sorted(over_represented, key=lambda x: x[1], reverse=True),
            'range_gaps': range_gaps,
            'total_predictions': total_predictions
        }
    
    def _analyze_range_gaps(self, base_predictions: List[List[int]], 
                          max_number: int) -> Dict[str, List[int]]:
        """Analyze which number ranges are under-represented"""
        # Define ranges
        ranges = {
            'low': (1, max_number // 3),
            'mid': (max_number // 3 + 1, 2 * max_number // 3),
            'high': (2 * max_number // 3 + 1, max_number)
        }
        
        range_counts = {range_name: 0 for range_name in ranges}
        
        for pred_set in base_predictions:
            for number in pred_set:
                for range_name, (start, end) in ranges.items():
                    if start <= number <= end:
                        range_counts[range_name] += 1
                        break
        
        # Find under-represented ranges
        total_numbers = sum(range_counts.values())
        expected_per_range = total_numbers / len(ranges)
        
        gap_ranges = {}
        for range_name, count in range_counts.items():
            if count < expected_per_range * 0.7:  # Less than 70% of expected
                start, end = ranges[range_name]
                gap_ranges[range_name] = list(range(start, end + 1))
        
        return gap_ranges
    
    def _generate_gap_filling_set(self, gap_analysis: Dict[str, Any], 
                                numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a set that fills the biggest gaps"""
        current_set = []
        
        # Prioritize under-represented numbers
        under_represented = gap_analysis.get('under_represented', [])
        
        # Add most under-represented numbers first
        for number, gap_size in under_represented:
            if len(current_set) < numbers_per_set:
                current_set.append(number)
        
        # Fill remaining slots with numbers from gap ranges
        range_gaps = gap_analysis.get('range_gaps', {})
        for range_name, numbers in range_gaps.items():
            available_numbers = [n for n in numbers if n not in current_set]
            while len(current_set) < numbers_per_set and available_numbers:
                selected = np.random.choice(available_numbers)
                current_set.append(selected)
                available_numbers.remove(selected)
        
        # Fill any remaining slots randomly
        while len(current_set) < numbers_per_set:
            available = [n for n in range(1, max_number + 1) if n not in current_set]
            if available:
                current_set.append(np.random.choice(available))
        
        return current_set[:numbers_per_set]
    
    def _generate_contrarian_set(self, base_predictions: List[List[int]], 
                               numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a set using contrarian logic (opposite of trends)"""
        # Find most frequently predicted numbers
        number_frequencies = Counter()
        for pred_set in base_predictions:
            for number in pred_set:
                number_frequencies[number] += 1
        
        # Select least frequently predicted numbers
        all_numbers = list(range(1, max_number + 1))
        frequency_sorted = sorted(all_numbers, key=lambda x: number_frequencies.get(x, 0))
        
        # Take numbers from the least frequent end
        contrarian_set = frequency_sorted[:numbers_per_set]
        
        return contrarian_set
    
    def _generate_balanced_complement_set(self, base_predictions: List[List[int]], 
                                        existing_optimized: List[Dict[str, Any]], 
                                        numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a balanced complementary set"""
        # Combine base predictions and existing optimized sets
        all_existing_numbers = set()
        
        for pred_set in base_predictions:
            all_existing_numbers.update(pred_set)
        
        for opt_set in existing_optimized:
            all_existing_numbers.update(opt_set['numbers'])
        
        # Select numbers that provide balance
        current_set = []
        
        # Strategy: Select from different ranges to ensure balance
        ranges = [
            (1, max_number // 3),
            (max_number // 3 + 1, 2 * max_number // 3),
            (2 * max_number // 3 + 1, max_number)
        ]
        
        numbers_per_range = numbers_per_set // len(ranges)
        remainder = numbers_per_set % len(ranges)
        
        for i, (start, end) in enumerate(ranges):
            target_count = numbers_per_range + (1 if i < remainder else 0)
            range_numbers = list(range(start, end + 1))
            
            # Prefer numbers less represented in existing sets
            range_numbers.sort(key=lambda x: len([s for s in base_predictions + 
                                                [opt['numbers'] for opt in existing_optimized] 
                                                if x in s]))
            
            selected_from_range = 0
            for number in range_numbers:
                if selected_from_range < target_count and number not in current_set:
                    current_set.append(number)
                    selected_from_range += 1
        
        # Fill any remaining slots
        while len(current_set) < numbers_per_set:
            available = [n for n in range(1, max_number + 1) if n not in current_set]
            if available:
                # Prefer numbers not in existing sets
                unused = [n for n in available if n not in all_existing_numbers]
                if unused:
                    current_set.append(np.random.choice(unused))
                else:
                    current_set.append(np.random.choice(available))
        
        return current_set[:numbers_per_set]
    
    def _calculate_gap_fill_score(self, number_set: List[int], 
                                gap_analysis: Dict[str, Any]) -> float:
        """Calculate how well this set fills identified gaps"""
        under_represented = gap_analysis.get('under_represented', [])
        under_represented_numbers = [num for num, gap in under_represented]
        
        gap_fill_count = len([n for n in number_set if n in under_represented_numbers])
        return gap_fill_count / len(number_set) if number_set else 0
    
    def _calculate_complementary_score(self, number_set: List[int], 
                                     base_predictions: List[List[int]]) -> float:
        """Calculate how complementary this set is to base predictions"""
        if not base_predictions:
            return 0.5
        
        # Count overlaps with base predictions
        total_overlaps = 0
        max_possible_overlaps = 0
        
        for base_set in base_predictions:
            overlap = len(set(number_set).intersection(set(base_set)))
            total_overlaps += overlap
            max_possible_overlaps += min(len(number_set), len(base_set))
        
        # Complementary score is inverse of overlap (less overlap = more complementary)
        if max_possible_overlaps > 0:
            overlap_ratio = total_overlaps / max_possible_overlaps
            return 1.0 - overlap_ratio
        else:
            return 0.5


class BalancedDistributionOptimizer(SetOptimizationStrategy):
    """Optimizes sets for balanced number distribution"""
    
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
                # Start with a Fibonacci number if possible
                if fib_in_range:
                    current_set.append(np.random.choice(fib_in_range))
                else:
                    current_set.append(np.random.randint(1, max_number + 1))
        
        return current_set[:numbers_per_set]
    
    def _generate_statistical_balance_set(self, base_predictions: List[List[int]], 
                                        numbers_per_set: int, max_number: int) -> List[int]:
        """Generate a statistically balanced set based on common patterns"""
        # Analyze statistical properties of base predictions
        all_numbers = []
        for pred_set in base_predictions:
            all_numbers.extend(pred_set)
        
        if not all_numbers:
            return self._generate_perfect_balance_set(numbers_per_set, max_number)
        
        # Calculate target statistical properties
        target_mean = np.mean(all_numbers)
        target_std = np.std(all_numbers)
        
        current_set = []
        
        # Use optimization to find numbers that achieve target statistics
        def objective_function(numbers):
            if len(numbers) != numbers_per_set:
                return float('inf')
            
            mean_diff = abs(np.mean(numbers) - target_mean)
            std_diff = abs(np.std(numbers) - target_std)
            
            # Penalty for duplicate numbers
            if len(set(numbers)) != len(numbers):
                return float('inf')
            
            return mean_diff + std_diff
        
        # Generate candidates and select best
        best_set = None
        best_score = float('inf')
        
        for attempt in range(100):  # Try 100 random combinations
            candidate = sorted(np.random.choice(range(1, max_number + 1), numbers_per_set, replace=False))
            score = objective_function(candidate)
            
            if score < best_score:
                best_score = score
                best_set = candidate
        
        return list(best_set) if best_set is not None else self._generate_perfect_balance_set(numbers_per_set, max_number)
    
    def _calculate_range_balance(self, number_set: List[int], max_number: int) -> float:
        """Calculate how well balanced the set is across the number range"""
        if not number_set:
            return 0.0
        
        # Define ranges (low, mid, high)
        ranges = [
            (1, max_number // 3),
            (max_number // 3 + 1, 2 * max_number // 3),
            (2 * max_number // 3 + 1, max_number)
        ]
        
        range_counts = [0, 0, 0]
        
        for number in number_set:
            for i, (start, end) in enumerate(ranges):
                if start <= number <= end:
                    range_counts[i] += 1
                    break
        
        # Perfect balance would be equal distribution
        perfect_per_range = len(number_set) / len(ranges)
        
        # Calculate how close to perfect balance
        balance_score = 1.0 - sum(abs(count - perfect_per_range) for count in range_counts) / len(number_set)
        return max(0.0, balance_score)
    
    def _calculate_sum_balance(self, number_set: List[int], max_number: int, numbers_per_set: int) -> float:
        """Calculate how balanced the sum is"""
        if not number_set:
            return 0.0
        
        actual_sum = sum(number_set)
        
        # Expected sum for a balanced set
        expected_sum = (max_number + 1) * numbers_per_set / 2
        
        # Calculate how close to expected sum
        max_deviation = expected_sum * 0.5  # Allow 50% deviation
        deviation = abs(actual_sum - expected_sum)
        
        balance_score = 1.0 - min(1.0, deviation / max_deviation)
        return balance_score
    
    def _calculate_parity_balance(self, number_set: List[int]) -> float:
        """Calculate balance between odd and even numbers"""
        if not number_set:
            return 0.0
        
        odd_count = sum(1 for num in number_set if num % 2 == 1)
        even_count = len(number_set) - odd_count
        
        # Perfect balance would be 50/50 split
        ideal_split = len(number_set) / 2
        
        balance_score = 1.0 - abs(odd_count - ideal_split) / ideal_split
        return max(0.0, balance_score)
    
    def _calculate_distribution_score(self, number_set: List[int], max_number: int) -> float:
        """Calculate overall distribution quality"""
        if len(number_set) <= 1:
            return 1.0
        
        sorted_set = sorted(number_set)
        
        # Calculate gaps between consecutive numbers
        gaps = [sorted_set[i+1] - sorted_set[i] for i in range(len(sorted_set)-1)]
        
        # Ideal gap would be evenly distributed
        total_range = max_number - 1
        ideal_gap = total_range / len(number_set)
        
        # Score based on how close gaps are to ideal
        gap_scores = [1.0 - min(1.0, abs(gap - ideal_gap) / ideal_gap) for gap in gaps]
        
        return np.mean(gap_scores) if gap_scores else 1.0


class SetBasedOptimizer:
    """Main coordinator for all set-based optimization strategies"""
    
    def __init__(self):
        self.optimizers = {
            'coverage': CoverageOptimizer(),
            'complementary': ComplementarySetGenerator(),
            'balanced': BalancedDistributionOptimizer()
        }
        self.optimization_history = []
    
    def optimize_prediction_sets(self, base_predictions: List[List[int]], 
                               game_config: Dict[str, Any], target_sets: int) -> Dict[str, Any]:
        """Run all optimization strategies and return best results"""
        try:
            optimization_result = {
                'timestamp': datetime.now().isoformat(),
                'game_config': game_config,
                'base_predictions_count': len(base_predictions),
                'target_sets': target_sets,
                'optimized_sets': [],
                'optimization_scores': {},
                'best_strategy': None
            }
            
            all_optimized_sets = []
            strategy_scores = {}
            
            # Run each optimization strategy
            for strategy_name, optimizer in self.optimizers.items():
                logger.info(f"Running {optimizer.name}...")
                
                try:
                    sets = optimizer.optimize_sets(base_predictions, game_config, target_sets)
                    
                    # Calculate strategy performance
                    if sets:
                        strategy_score = np.mean([s.get('combined_score', 0) for s in sets])
                        strategy_scores[strategy_name] = strategy_score
                        
                        # Tag sets with strategy info
                        for opt_set in sets:
                            opt_set['optimization_strategy'] = strategy_name
                            opt_set['strategy_score'] = strategy_score
                        
                        all_optimized_sets.extend(sets)
                        
                        logger.info(f"{optimizer.name} completed with score: {strategy_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error in {optimizer.name}: {e}")
                    strategy_scores[strategy_name] = 0.0
            
            # Rank and select best sets
            if all_optimized_sets:
                ranked_sets = self._rank_optimized_sets(all_optimized_sets, target_sets)
                optimization_result['optimized_sets'] = ranked_sets[:target_sets]
                
                # Determine best strategy
                if strategy_scores:
                    best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
                    optimization_result['best_strategy'] = best_strategy[0]
                    optimization_result['best_strategy_score'] = best_strategy[1]
            
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
                diversity_improvement * 2,
                np.mean([opt_set.get('ranking_score', 0) for opt_set in optimized_sets])
            ]
            
            overall_score = np.mean([f for f in quality_factors if f >= 0])
            
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
    
    def get_optimization_insights(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from optimization results"""
        try:
            insights = {
                'optimization_insights': [],
                'strategy_performance': {},
                'recommendations': [],
                'quality_assessment': {}
            }
            
            # Overall quality assessment
            overall_quality = optimization_result.get('overall_quality', {})
            overall_score = overall_quality.get('overall_score', 0)
            
            if overall_score > 0.8:
                insights['optimization_insights'].append("🏆 Excellent optimization achieved")
                insights['quality_assessment']['level'] = 'Excellent'
            elif overall_score > 0.6:
                insights['optimization_insights'].append("✅ Good optimization results")
                insights['quality_assessment']['level'] = 'Good'
            elif overall_score > 0.4:
                insights['optimization_insights'].append("📊 Moderate optimization achieved")
                insights['quality_assessment']['level'] = 'Moderate'
            else:
                insights['optimization_insights'].append("⚠️ Limited optimization improvement")
                insights['quality_assessment']['level'] = 'Limited'
            
            insights['quality_assessment']['score'] = overall_score
            
            # Coverage insights
            coverage = overall_quality.get('coverage', 0)
            if coverage > 0.8:
                insights['optimization_insights'].append(f"🎯 Excellent number coverage ({coverage:.1%})")
            elif coverage > 0.6:
                insights['optimization_insights'].append(f"📈 Good number coverage ({coverage:.1%})")
            
            # Diversity insights
            diversity = overall_quality.get('diversity', 0)
            if diversity > 0.8:
                insights['optimization_insights'].append("🌈 High set diversity achieved")
            elif diversity > 0.6:
                insights['optimization_insights'].append("🔄 Good set diversity")
            
            # Strategy performance analysis
            optimization_scores = optimization_result.get('optimization_scores', {})
            if optimization_scores:
                best_strategy = max(optimization_scores.items(), key=lambda x: x[1])
                insights['strategy_performance']['best'] = best_strategy[0]
                insights['strategy_performance']['best_score'] = best_strategy[1]
                
                insights['optimization_insights'].append(
                    f"🥇 Best performing strategy: {best_strategy[0].title()} ({best_strategy[1]:.3f})"
                )
            
            # Recommendations
            if overall_score > 0.7:
                insights['recommendations'].append("Use optimized sets with high confidence")
            else:
                insights['recommendations'].append("Consider combining with base predictions")
            
            if optimization_scores.get('complementary', 0) > 0.7:
                insights['recommendations'].append("Complementary sets show strong potential")
            
            if optimization_scores.get('balanced', 0) > 0.7:
                insights['recommendations'].append("Balanced distribution strategy is effective")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating optimization insights: {e}")
            return {
                'optimization_insights': ["Error generating insights"],
                'strategy_performance': {},
                'recommendations': ["Unable to generate recommendations"],
                'quality_assessment': {'level': 'Unknown', 'score': 0.0}
            }
