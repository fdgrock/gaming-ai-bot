"""
Enhanced feature engineering for advanced lottery prediction models.
Includes 4-Phase Ultra-High Accuracy Intelligence Integration.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import math
from datetime import datetime, timedelta


def create_ultra_high_accuracy_features(draws: List[List[int]], 
                                       draw_dates: Optional[List] = None,
                                       game_type: str = 'lotto_649',
                                       pool_size: int = 50,
                                       include_4phase: bool = True,
                                       include_traditional: bool = True) -> np.ndarray:
    """
    Create ultra-high accuracy feature matrix using 4-phase intelligence.
    
    Args:
        draws: List of lottery draws (each draw is a list of numbers)
        draw_dates: Optional list of draw dates for temporal analysis
        game_type: Game type ('lotto_649' or 'lotto_max')
        pool_size: Maximum lottery number (49 or 50)
        include_4phase: Include 4-phase enhanced features
        include_traditional: Include traditional features for compatibility
    
    Returns:
        Enhanced feature matrix (n_draws, n_enhanced_features)
    """
    if len(draws) < 10:
        raise ValueError("Need at least 10 draws for ultra-high accuracy features")
    
    features = []
    
    # Initialize 4-phase engines if requested
    math_engine = None
    expert_ensemble = None
    set_optimizer = None
    temporal_engine = None
    
    if include_4phase:
        try:
            from ai_lottery_bot.mathematical_engine import AdvancedMathematicalEngine
            from ai_lottery_bot.expert_ensemble import SpecializedExpertEnsemble
            from ai_lottery_bot.set_optimizer import SetBasedOptimizer
            from ai_lottery_bot.temporal_engine import AdvancedTemporalEngine
            
            math_engine = AdvancedMathematicalEngine()
            expert_ensemble = SpecializedExpertEnsemble()
            set_optimizer = SetBasedOptimizer()
            temporal_engine = AdvancedTemporalEngine()
            
        except ImportError as e:
            print(f"Warning: 4-phase engines not available: {e}")
            include_4phase = False
    
    for i in range(len(draws)):
        draw = draws[i]
        feature_vector = []
        
        # Traditional features (if requested)
        if include_traditional:
            feature_vector.extend(_extract_basic_features(draw, pool_size))
            
            # Add traditional frequency and pattern features
            if i > 0:
                freq_features = _extract_frequency_features(draws[:i], draw, pool_size)
                normalized_freq = freq_features[:30] if len(freq_features) >= 30 else freq_features + [0] * (30 - len(freq_features))
                feature_vector.extend(normalized_freq)
            else:
                feature_vector.extend([0] * 30)
        
        # 4-Phase Enhanced Features
        if include_4phase and i > 5:  # Need sufficient history
            
            # Phase 1: Mathematical Intelligence Features
            if math_engine:
                math_features = _extract_mathematical_intelligence_features(
                    draws[:i+1], draw, math_engine, game_type
                )
                feature_vector.extend(math_features)
            else:
                feature_vector.extend([0] * 15)  # Placeholder
            
            # Phase 2: Expert Ensemble Features
            if expert_ensemble:
                expert_features = _extract_expert_ensemble_features(
                    draws[:i+1], draw, expert_ensemble, game_type
                )
                feature_vector.extend(expert_features)
            else:
                feature_vector.extend([0] * 12)  # Placeholder
            
            # Phase 3: Set Optimization Quality Features
            if set_optimizer:
                optimization_features = _extract_optimization_quality_features(
                    draws[:i+1], draw, set_optimizer, game_type
                )
                feature_vector.extend(optimization_features)
            else:
                feature_vector.extend([0] * 8)  # Placeholder
            
            # Phase 4: Temporal Intelligence Features
            if temporal_engine and draw_dates and i < len(draw_dates):
                temporal_features = _extract_temporal_intelligence_features(
                    draws[:i+1], draw_dates[:i+1], draw, temporal_engine, game_type
                )
                feature_vector.extend(temporal_features)
            else:
                feature_vector.extend([0] * 10)  # Placeholder
        
        else:
            # Insufficient history for 4-phase analysis
            if include_4phase:
                feature_vector.extend([0] * (15 + 12 + 8 + 10))  # Total 4-phase features
        
        features.append(feature_vector)
    
    return np.array(features, dtype=np.float32)


def create_advanced_lottery_features(draws: List[List[int]], 
                                   pool_size: int = 50,
                                   window_sizes: List[int] = [3, 5, 10],  # Reduced for testing
                                   include_basic: bool = True,
                                   include_frequency: bool = True,
                                   include_patterns: bool = True,
                                   include_temporal: bool = True,
                                   include_sequence: bool = True,
                                   draw_dates: Optional[List] = None,
                                   game_type: str = 'lotto_649',
                                   use_4phase: bool = False) -> np.ndarray:
    """
    Create comprehensive feature matrix for lottery prediction.
    Enhanced to optionally include 4-Phase Ultra-High Accuracy features.
    
    Args:
        draws: List of lottery draws (each draw is a list of numbers)
        pool_size: Maximum lottery number (49 or 50)
        window_sizes: Different lookback windows for rolling features
        include_basic: Include basic statistical features
        include_frequency: Include frequency analysis features
        include_patterns: Include pattern detection features
        include_temporal: Include temporal/cyclical features
        include_sequence: Include sequence-based features
        draw_dates: Optional dates for enhanced temporal analysis
        game_type: Game type for 4-phase analysis
        use_4phase: Enable 4-Phase Ultra-High Accuracy features
    
    Returns:
        Feature matrix (n_draws, n_features)
    """
    # Use enhanced feature generation if 4-phase is requested
    if use_4phase:
        return create_ultra_high_accuracy_features(
            draws=draws,
            draw_dates=draw_dates,
            game_type=game_type,
            pool_size=pool_size,
            include_4phase=True,
            include_traditional=True
        )
    
    # Traditional feature generation (existing functionality)
    if len(draws) < max(window_sizes) + 1:
        raise ValueError(f"Need at least {max(window_sizes) + 1} draws for features")
    
    features = []
    
    for i in range(len(draws)):
        draw = draws[i]
        feature_vector = []
        
        # Basic draw features
        if include_basic:
            feature_vector.extend(_extract_basic_features(draw, pool_size))
        
        # Historical frequency features
        if include_frequency:
            if i > 0:
                freq_features = _extract_frequency_features(draws[:i], draw, pool_size)
                # Normalize frequency features to fixed size (take first 60 or pad)
                normalized_freq = freq_features[:60] if len(freq_features) >= 60 else freq_features + [0] * (60 - len(freq_features))
                feature_vector.extend(normalized_freq)
            else:
                feature_vector.extend([0] * 60)  # Fixed padding for first draw
        
        # Rolling window features (pattern detection)
        if include_patterns:
            for window in window_sizes:
                if i >= window:
                    window_draws = draws[max(0, i-window):i]
                    feature_vector.extend(_extract_window_features(window_draws, draw, pool_size))
                else:
                    feature_vector.extend([0] * 15)  # Padding for insufficient history
        
        # Temporal features
        if include_temporal:
            feature_vector.extend(_extract_temporal_features(i, len(draws)))
        
        # Pattern features (additional pattern detection)
        if include_patterns:
            feature_vector.extend(_extract_pattern_features(draw, pool_size))
        
        # Sequence features (if enough history)
        if include_sequence:
            if i >= 3:
                recent_draws = draws[max(0, i-3):i]
                feature_vector.extend(_extract_sequence_features(recent_draws, draw, pool_size))
            else:
                feature_vector.extend([0] * 20)  # Padding
        
        features.append(feature_vector)
    
    # Validate feature vector consistency before creating array
    if not features:
        raise ValueError("No features were generated")
    
    # Check if all feature vectors have the same length
    feature_lengths = [len(f) for f in features]
    expected_length = feature_lengths[0]
    
    if not all(length == expected_length for length in feature_lengths):
        # Find the most common length
        from collections import Counter
        length_counts = Counter(feature_lengths)
        most_common_length = length_counts.most_common(1)[0][0]
        
        # Filter or pad features to have consistent length
        consistent_features = []
        for i, feature_vector in enumerate(features):
            if len(feature_vector) == most_common_length:
                consistent_features.append(feature_vector)
            elif len(feature_vector) < most_common_length:
                # Pad with zeros
                padded = feature_vector + [0] * (most_common_length - len(feature_vector))
                consistent_features.append(padded)
            else:
                # Truncate
                truncated = feature_vector[:most_common_length]
                consistent_features.append(truncated)
        
        features = consistent_features
        print(f"Warning: Adjusted {len(feature_lengths) - len(consistent_features)} feature vectors to consistent length {most_common_length}")
    
    try:
        return np.array(features, dtype=np.float64)
    except Exception as e:
        # Fallback: ensure all elements are numeric
        numeric_features = []
        for feature_vector in features:
            numeric_vector = []
            for val in feature_vector:
                try:
                    numeric_vector.append(float(val))
                except (ValueError, TypeError):
                    numeric_vector.append(0.0)
            numeric_features.append(numeric_vector)
        return np.array(numeric_features, dtype=np.float64)


def _extract_basic_features(draw: List[int], pool_size: int) -> List[float]:
    """Extract basic statistical features from a single draw."""
    if not draw:
        return [0] * 20
    
    features = []
    sorted_draw = sorted(draw)
    
    # Basic statistics
    features.append(len(draw))  # Number count
    features.append(sum(draw))  # Sum
    features.append(np.mean(draw))  # Mean
    features.append(np.median(draw))  # Median
    features.append(np.std(draw))  # Standard deviation
    features.append(min(draw))  # Min
    features.append(max(draw))  # Max
    features.append(max(draw) - min(draw))  # Range
    
    # Distribution features
    features.append(sum(1 for x in draw if x <= pool_size // 3))  # Low numbers
    features.append(sum(1 for x in draw if pool_size // 3 < x <= 2 * pool_size // 3))  # Mid numbers
    features.append(sum(1 for x in draw if x > 2 * pool_size // 3))  # High numbers
    
    # Parity features
    features.append(sum(1 for x in draw if x % 2 == 0))  # Even numbers
    features.append(sum(1 for x in draw if x % 2 == 1))  # Odd numbers
    
    # Consecutive numbers
    consecutive_count = 0
    for i in range(len(sorted_draw) - 1):
        if sorted_draw[i+1] - sorted_draw[i] == 1:
            consecutive_count += 1
    features.append(consecutive_count)
    
    # Gaps between numbers
    gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
    features.append(np.mean(gaps) if gaps else 0)  # Average gap
    features.append(np.std(gaps) if gaps else 0)   # Gap standard deviation
    
    # Divisibility features
    features.append(sum(1 for x in draw if x % 3 == 0))  # Divisible by 3
    features.append(sum(1 for x in draw if x % 5 == 0))  # Divisible by 5
    features.append(sum(1 for x in draw if x % 7 == 0))  # Divisible by 7
    
    # Digit sum features
    digit_sums = [sum(int(d) for d in str(x)) for x in draw]
    features.append(np.mean(digit_sums))  # Average digit sum
    
    # Ensure exactly 20 features are returned
    while len(features) < 20:
        features.append(0.0)
    
    # Ensure all features are numeric
    validated_features = []
    for val in features[:20]:  # Take only first 20
        try:
            validated_features.append(float(val))
        except (ValueError, TypeError):
            validated_features.append(0.0)
    
    return validated_features


def _extract_frequency_features(historical_draws: List[List[int]], 
                               current_draw: List[int], 
                               pool_size: int) -> List[float]:
    """Extract frequency-based features."""
    # Count occurrences of each number
    number_counts = Counter()
    for draw in historical_draws:
        number_counts.update(draw)
    
    total_draws = len(historical_draws)
    features = []
    
    # Number frequencies (how often each number appeared)
    for num in range(1, pool_size + 1):
        frequency = number_counts[num] / total_draws if total_draws > 0 else 0
        features.append(frequency)
    
    # Features for current draw numbers
    if current_draw:
        current_frequencies = [number_counts[num] / total_draws if total_draws > 0 else 0 
                             for num in current_draw]
        features.extend([
            np.mean(current_frequencies),  # Average frequency of drawn numbers
            np.std(current_frequencies),   # Std of frequencies
            min(current_frequencies),      # Min frequency
            max(current_frequencies),      # Max frequency
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # Hot and cold numbers
    if number_counts:
        max_count = max(number_counts.values())
        min_count = min(number_counts.values())
        hot_threshold = max_count * 0.8
        cold_threshold = min_count * 1.2
        
        hot_numbers = sum(1 for num in current_draw if number_counts[num] >= hot_threshold)
        cold_numbers = sum(1 for num in current_draw if number_counts[num] <= cold_threshold)
        
        features.extend([hot_numbers, cold_numbers])
    else:
        features.extend([0, 0])
    
    # Recency features (when numbers last appeared)
    recency_scores = []
    for num in current_draw:
        last_seen = -1
        for i in range(len(historical_draws) - 1, -1, -1):
            if num in historical_draws[i]:
                last_seen = len(historical_draws) - 1 - i
                break
        recency_scores.append(last_seen if last_seen >= 0 else len(historical_draws))
    
    if recency_scores:
        features.extend([
            np.mean(recency_scores),  # Average recency
            np.std(recency_scores),   # Recency variability
            min(recency_scores),      # Most recent
            max(recency_scores),      # Least recent
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    return features


def _extract_window_features(window_draws: List[List[int]], 
                           current_draw: List[int], 
                           pool_size: int) -> List[float]:
    """Extract rolling window features."""
    if not window_draws:
        return [0] * 15
    
    features = []
    
    # Window statistics
    all_numbers = [num for draw in window_draws for num in draw]
    window_counter = Counter(all_numbers)
    
    # Most/least frequent in window
    if window_counter:
        most_frequent = window_counter.most_common(1)[0][1]
        least_frequent = window_counter.most_common()[-1][1]
        features.extend([most_frequent, least_frequent])
    else:
        features.extend([0, 0])
    
    # Coverage (how many unique numbers appeared)
    unique_numbers = len(set(all_numbers))
    features.append(unique_numbers / pool_size)
    
    # Trend features
    draw_sums = [sum(draw) for draw in window_draws]
    if len(draw_sums) > 1:
        # Simple trend (last - first)
        trend = (draw_sums[-1] - draw_sums[0]) / len(draw_sums)
        features.append(trend)
        
        # Volatility (coefficient of variation)
        volatility = np.std(draw_sums) / (np.mean(draw_sums) + 1e-8)
        features.append(volatility)
    else:
        features.extend([0, 0])
    
    # Momentum features
    if len(window_draws) >= 4:
        recent_half = window_draws[len(window_draws)//2:]
        older_half = window_draws[:len(window_draws)//2]
        
        recent_avg = np.mean([sum(draw) for draw in recent_half])
        older_avg = np.mean([sum(draw) for draw in older_half])
        momentum = (recent_avg - older_avg) / (older_avg + 1e-8)
        features.append(momentum)
    else:
        features.append(0)
    
    # Pattern consistency
    patterns = []
    for draw in window_draws:
        # Pattern: [low, mid, high] distribution
        low = sum(1 for x in draw if x <= pool_size // 3)
        mid = sum(1 for x in draw if pool_size // 3 < x <= 2 * pool_size // 3)
        high = sum(1 for x in draw if x > 2 * pool_size // 3)
        patterns.append([low, mid, high])
    
    if patterns:
        pattern_matrix = np.array(patterns)
        pattern_consistency = np.mean(np.std(pattern_matrix, axis=0))
        features.append(pattern_consistency)
    else:
        features.append(0)
    
    # Gap analysis
    all_gaps = []
    for draw in window_draws:
        sorted_draw = sorted(draw)
        gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(len(sorted_draw) - 1)]
        all_gaps.extend(gaps)
    
    if all_gaps:
        features.extend([
            np.mean(all_gaps),
            np.std(all_gaps),
            min(all_gaps),
            max(all_gaps)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # Repeat analysis (numbers that appeared multiple times in window)
    repeats = sum(1 for count in window_counter.values() if count > 1)
    features.append(repeats)
    
    # Ensure exactly 15 features are returned
    while len(features) < 15:
        features.append(0.0)
    
    # Ensure all features are numeric and take only first 15
    validated_features = []
    for val in features[:15]:
        try:
            validated_features.append(float(val))
        except (ValueError, TypeError):
            validated_features.append(0.0)
    
    return validated_features


def _extract_temporal_features(current_index: int, total_draws: int) -> List[float]:
    """Extract temporal/positional features."""
    features = []
    
    # Position in dataset
    features.append(current_index / total_draws)
    
    # Cyclical features (assuming weekly draws)
    # Week cycle
    week_position = (current_index % 52) / 52
    features.append(np.sin(2 * np.pi * week_position))
    features.append(np.cos(2 * np.pi * week_position))
    
    # Month cycle
    month_position = (current_index % 12) / 12
    features.append(np.sin(2 * np.pi * month_position))
    features.append(np.cos(2 * np.pi * month_position))
    
    # Quarter cycle
    quarter_position = (current_index % 4) / 4
    features.append(np.sin(2 * np.pi * quarter_position))
    features.append(np.cos(2 * np.pi * quarter_position))
    
    # Ensure exactly 8 features are returned
    while len(features) < 8:
        features.append(0.0)
    
    # Validate numeric values
    validated_features = []
    for val in features[:8]:
        try:
            validated_features.append(float(val))
        except (ValueError, TypeError):
            validated_features.append(0.0)
    
    return validated_features


def _extract_pattern_features(draw: List[int], pool_size: int) -> List[float]:
    """Extract pattern-based features."""
    if not draw:
        return [0] * 10
    
    features = []
    sorted_draw = sorted(draw)
    
    # Arithmetic progression detection
    is_arithmetic = True
    if len(sorted_draw) > 2:
        diff = sorted_draw[1] - sorted_draw[0]
        for i in range(2, len(sorted_draw)):
            if sorted_draw[i] - sorted_draw[i-1] != diff:
                is_arithmetic = False
                break
    else:
        is_arithmetic = False
    features.append(float(is_arithmetic))
    
    # Geometric-like progression (ratios)
    ratios = []
    for i in range(1, len(sorted_draw)):
        if sorted_draw[i-1] > 0:
            ratios.append(sorted_draw[i] / sorted_draw[i-1])
    
    if ratios:
        ratio_consistency = 1.0 - (np.std(ratios) / (np.mean(ratios) + 1e-8))
        features.append(max(0, ratio_consistency))
    else:
        features.append(0)
    
    # Symmetry around middle
    middle = pool_size / 2
    symmetry_score = 0
    for num in draw:
        distance_from_middle = abs(num - middle)
        symmetry_score += 1.0 / (1.0 + distance_from_middle)
    features.append(symmetry_score / len(draw))
    
    # Clustering (how spread out numbers are)
    if len(sorted_draw) > 1:
        total_span = sorted_draw[-1] - sorted_draw[0]
        max_possible_span = pool_size - 1
        clustering = 1.0 - (total_span / max_possible_span)
        features.append(clustering)
    else:
        features.append(0)
    
    # Prime numbers
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    prime_count = sum(1 for num in draw if is_prime(num))
    features.append(prime_count / len(draw))
    
    # Perfect squares
    perfect_squares = sum(1 for num in draw if int(np.sqrt(num))**2 == num)
    features.append(perfect_squares / len(draw))
    
    # Fibonacci-like numbers (basic check)
    fib_numbers = {1, 1, 2, 3, 5, 8, 13, 21, 34}
    fib_count = sum(1 for num in draw if num in fib_numbers)
    features.append(fib_count / len(draw))
    
    # Ending digit distribution
    ending_digits = [num % 10 for num in draw]
    unique_endings = len(set(ending_digits))
    features.append(unique_endings / len(draw))
    
    # Sum modulo patterns
    draw_sum = sum(draw)
    features.append(draw_sum % 10)  # Last digit of sum
    features.append(draw_sum % 7)   # Sum mod 7
    
    # Ensure exactly 10 features are returned
    while len(features) < 10:
        features.append(0.0)
    
    # Validate numeric values
    validated_features = []
    for val in features[:10]:
        try:
            validated_features.append(float(val))
        except (ValueError, TypeError):
            validated_features.append(0.0)
    
    return validated_features


def _extract_sequence_features(recent_draws: List[List[int]], 
                             current_draw: List[int], 
                             pool_size: int) -> List[float]:
    """Extract features based on recent sequence patterns."""
    features = []
    
    if len(recent_draws) < 2:
        return [0] * 20
    
    # Carry-over analysis (numbers that repeat from previous draws)
    carryovers = []
    for i in range(1, len(recent_draws)):
        prev_draw = set(recent_draws[i-1])
        curr_draw = set(recent_draws[i])
        carryover = len(prev_draw.intersection(curr_draw))
        carryovers.append(carryover)
    
    if carryovers:
        features.extend([
            np.mean(carryovers),
            np.std(carryovers),
            max(carryovers),
            min(carryovers)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # Alternating patterns
    all_numbers = [num for draw in recent_draws for num in draw]
    alternation_score = 0
    for i in range(1, len(all_numbers)):
        if (all_numbers[i] % 2) != (all_numbers[i-1] % 2):
            alternation_score += 1
    features.append(alternation_score / max(1, len(all_numbers) - 1))
    
    # Increasing/decreasing trends in individual positions
    if len(recent_draws) >= 3:
        position_trends = []
        max_len = max(len(draw) for draw in recent_draws)
        
        for pos in range(max_len):
            position_values = []
            for draw in recent_draws:
                if pos < len(draw):
                    position_values.append(sorted(draw)[pos])
            
            if len(position_values) >= 3:
                # Simple trend: is it generally increasing/decreasing?
                increasing = sum(1 for i in range(1, len(position_values)) 
                               if position_values[i] > position_values[i-1])
                trend_strength = increasing / (len(position_values) - 1)
                position_trends.append(trend_strength)
        
        if position_trends:
            features.extend([
                np.mean(position_trends),
                np.std(position_trends)
            ])
        else:
            features.extend([0, 0])
    else:
        features.extend([0, 0])
    
    # Sum progression
    sums = [sum(draw) for draw in recent_draws]
    if len(sums) >= 2:
        sum_differences = [sums[i] - sums[i-1] for i in range(1, len(sums))]
        features.extend([
            np.mean(sum_differences),
            np.std(sum_differences)
        ])
        
        # Is sum increasing/decreasing?
        sum_trend = sum(1 for d in sum_differences if d > 0) / len(sum_differences)
        features.append(sum_trend)
    else:
        features.extend([0, 0, 0])
    
    # Range progression
    ranges = [max(draw) - min(draw) for draw in recent_draws if draw]
    if len(ranges) >= 2:
        range_differences = [ranges[i] - ranges[i-1] for i in range(1, len(ranges))]
        features.extend([
            np.mean(range_differences),
            np.std(range_differences)
        ])
    else:
        features.extend([0, 0])
    
    # Pattern repetition detection
    pattern_similarity = 0
    for i in range(len(recent_draws) - 1):
        draw1 = set(recent_draws[i])
        draw2 = set(recent_draws[i + 1])
        jaccard = len(draw1.intersection(draw2)) / len(draw1.union(draw2))
        pattern_similarity += jaccard
    
    if len(recent_draws) > 1:
        pattern_similarity /= (len(recent_draws) - 1)
    features.append(pattern_similarity)
    
    # Decade distribution consistency
    decade_patterns = []
    for draw in recent_draws:
        decades = [0] * 5  # 0-9, 10-19, 20-29, 30-39, 40-49
        for num in draw:
            decade_idx = min(num // 10, 4)
            decades[decade_idx] += 1
        decade_patterns.append(decades)
    
    if len(decade_patterns) > 1:
        decade_matrix = np.array(decade_patterns)
        decade_consistency = np.mean(np.std(decade_matrix, axis=0))
        features.append(decade_consistency)
    else:
        features.append(0)
    
    # Remaining features to reach 20
    while len(features) < 20:
        features.append(0.0)
    
    # Validate numeric values
    validated_features = []
    for val in features[:20]:
        try:
            validated_features.append(float(val))
        except (ValueError, TypeError):
            validated_features.append(0.0)
    
    return validated_features  # Ensure exactly 20 features


def build_sequences_from_features(features: np.ndarray, 
                                window: int = 10, 
                                use_one_hot: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequence data from feature matrix for LSTM/Transformer training.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        window: Sequence window size
        use_one_hot: Whether to include one-hot encoding
    
    Returns:
        X: Input sequences (n_sequences, window, features)
        y: Target sequences (n_sequences, features)
    """
    if len(features) < window + 1:
        raise ValueError(f"Need at least {window + 1} samples")
    
    X_sequences = []
    y_sequences = []
    
    for i in range(window, len(features)):
        X_sequences.append(features[i-window:i])
        y_sequences.append(features[i])
    
    return np.array(X_sequences), np.array(y_sequences)


def create_sequence_features_for_lstm(draws: List[List[int]], 
                                     pool_size: int = 49,
                                     sequence_length: int = 10,
                                     draw_dates: List = None,
                                     game_type: str = 'lotto_649',
                                     use_4phase: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create LSTM sequence features from lottery draws.
    
    Args:
        draws: List of lottery draws
        pool_size: Maximum number in pool
        sequence_length: Length of input sequences
        draw_dates: Optional draw dates for temporal features
        game_type: Game type for 4-phase analysis
        use_4phase: Enable 4-Phase Ultra-High Accuracy features
    
    Returns:
        X: Input sequences (n_sequences, sequence_length, features)
        y: Target sequences (n_sequences, features)
    """
    try:
        # Generate features using advanced feature engineering
        if use_4phase:
            features = create_ultra_high_accuracy_features(
                draws=draws,
                draw_dates=draw_dates,
                game_type=game_type,
                pool_size=pool_size,
                include_4phase=True,
                include_traditional=True
            )
        else:
            features = create_advanced_lottery_features(
                draws=draws,
                pool_size=pool_size,
                draw_dates=draw_dates,
                game_type=game_type,
                use_4phase=False
            )
        
        # Build sequences from features
        X, y = build_sequences_from_features(features, window=sequence_length)
        
        print(f"LSTM sequences created: X={X.shape}, y={y.shape}")
        return X, y
        
    except Exception as e:
        print(f"Error creating LSTM sequences: {e}")
        # Fallback: create basic sequences from raw draws
        features = []
        for draw in draws:
            # Create basic feature vector from draw
            feature_vector = [0] * pool_size
            for num in draw:
                if 1 <= num <= pool_size:
                    feature_vector[num-1] = 1
            features.append(feature_vector)
        
        features = np.array(features)
        X, y = build_sequences_from_features(features, window=sequence_length)
        return X, y


def create_sequence_features_for_transformer(draws: List[List[int]], 
                                           pool_size: int = 49,
                                           sequence_length: int = 15,
                                           draw_dates: List = None,
                                           game_type: str = 'lotto_649',
                                           use_4phase: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Transformer sequence features from lottery draws.
    
    Args:
        draws: List of lottery draws
        pool_size: Maximum number in pool
        sequence_length: Length of input sequences
        draw_dates: Optional draw dates for temporal features
        game_type: Game type for 4-phase analysis
        use_4phase: Enable 4-Phase Ultra-High Accuracy features
    
    Returns:
        X: Input sequences (n_sequences, sequence_length, features)
        y: Target sequences (n_sequences, features)
    """
    try:
        # Generate features using advanced feature engineering
        if use_4phase:
            features = create_ultra_high_accuracy_features(
                draws=draws,
                draw_dates=draw_dates,
                game_type=game_type,
                pool_size=pool_size,
                include_4phase=True,
                include_traditional=True
            )
        else:
            features = create_advanced_lottery_features(
                draws=draws,
                pool_size=pool_size,
                draw_dates=draw_dates,
                game_type=game_type,
                use_4phase=False
            )
        
        # Build sequences from features for transformer
        X, y = build_sequences_from_features(features, window=sequence_length)
        
        print(f"Transformer sequences created: X={X.shape}, y={y.shape}")
        return X, y
        
    except Exception as e:
        print(f"Error creating Transformer sequences: {e}")
        # Fallback: create basic sequences from raw draws
        features = []
        for draw in draws:
            # Create basic feature vector from draw
            feature_vector = [0] * pool_size
            for num in draw:
                if 1 <= num <= pool_size:
                    feature_vector[num-1] = 1
            features.append(feature_vector)
        
        features = np.array(features)
        X, y = build_sequences_from_features(features, window=sequence_length)
        return X, y


# ==================== 4-PHASE ULTRA-HIGH ACCURACY FEATURE EXTRACTORS ====================

def _extract_mathematical_intelligence_features(draws: List[List[int]], 
                                               current_draw: List[int], 
                                               math_engine, 
                                               game_type: str) -> List[float]:
    """Extract Phase 1 mathematical intelligence features."""
    features = []
    
    try:
        # Use the actual methods from mathematical engine
        analysis = math_engine.analyze_deep_patterns(draws, game_type)
        
        # Extract mathematical confidence and insights
        confidence = analysis.get('confidence', 0.0)
        pattern_strength = analysis.get('pattern_strength', 0.0)
        
        # Prime pattern features from engine analysis
        features.extend([
            confidence,
            pattern_strength,
            analysis.get('mathematical_score', 0.0),
            analysis.get('prime_analysis', {}).get('prime_density', 0.0)
        ])
        
        # Number theory features
        features.extend([
            analysis.get('sum_analysis', {}).get('sum_optimality', 0.0),
            analysis.get('distribution_analysis', {}).get('balance_score', 0.0),
            analysis.get('graph_analysis', {}).get('connectivity', 0.0)
        ])
        
        # Advanced mathematical features
        features.extend([
            analysis.get('range_analysis', {}).get('range_quality', 0.0),
            analysis.get('gap_analysis', {}).get('gap_score', 0.0),
            analysis.get('pattern_analysis', {}).get('pattern_quality', 0.0)
        ])
        
        # Combinatorial features
        features.extend([
            analysis.get('combinatorial_analysis', {}).get('diversity', 0.0),
            analysis.get('clustering_analysis', {}).get('cluster_score', 0.0),
            analysis.get('symmetry_analysis', {}).get('symmetry_score', 0.0)
        ])
        
        # Overall mathematical confidence
        features.extend([
            confidence,
            analysis.get('recommendation_strength', 0.0)
        ])
        
    except Exception as e:
        print(f"Mathematical feature extraction error: {e}")
        features = [0.0] * 15
    
    # Ensure exactly 15 features
    while len(features) < 15:
        features.append(0.0)
    
    return features[:15]


def _extract_expert_ensemble_features(draws: List[List[int]], 
                                     current_draw: List[int], 
                                     expert_ensemble, 
                                     game_type: str) -> List[float]:
    """Extract Phase 2 expert ensemble features."""
    features = []
    
    try:
        # First get ensemble analysis
        ensemble_analysis = expert_ensemble.analyze_all_patterns(draws, game_type)
        
        # Extract features from ensemble analysis
        ensemble_confidence = ensemble_analysis.get('ensemble_confidence', 0.0)
        confidence_variance = ensemble_analysis.get('confidence_variance', 0.0)
        
        # Mathematical specialist features
        math_analysis = ensemble_analysis.get('specialist_analyses', {}).get('mathematical', {})
        features.extend([
            math_analysis.get('confidence', ensemble_confidence * 0.8),
            math_analysis.get('pattern_score', ensemble_confidence * 0.7)
        ])
        
        # Temporal specialist features
        temporal_analysis = ensemble_analysis.get('specialist_analyses', {}).get('temporal', {})
        features.extend([
            temporal_analysis.get('confidence', ensemble_confidence * 0.6),
            temporal_analysis.get('cyclical_score', ensemble_confidence * 0.5)
        ])
        
        # Frequency specialist features (use ensemble confidence as proxy)
        features.extend([
            ensemble_confidence * 0.9,  # Frequency confidence proxy
            confidence_variance  # Balance score proxy
        ])
        
        # Cross-validation features
        features.extend([
            ensemble_confidence * 0.7,  # Cross-validation score proxy
            ensemble_confidence * 0.8   # Specialist agreement proxy
        ])
        
        # Ensemble quality metrics
        features.extend([
            ensemble_confidence,
            confidence_variance * 0.6  # Prediction stability proxy
        ])
        
        # Overall ensemble assessment
        features.extend([
            ensemble_confidence,
            min(1.0, ensemble_confidence + (1.0 - confidence_variance))  # Overall quality
        ])
        
    except Exception as e:
        print(f"Expert ensemble feature extraction error: {e}")
        features = [0.0] * 12
    
    # Ensure exactly 12 features
    while len(features) < 12:
        features.append(0.0)
    
    return features[:12]


def _extract_optimization_quality_features(draws: List[List[int]], 
                                          current_draw: List[int], 
                                          set_optimizer, 
                                          game_type: str) -> List[float]:
    """Extract Phase 3 set optimization quality features."""
    features = []
    
    try:
        # Analyze set optimization using actual method
        max_number = 49 if '649' in game_type else 50
        numbers_per_set = 6 if '649' in game_type else 7
        
        game_config = {
            'max_number': max_number,
            'numbers_per_set': numbers_per_set,
            'game_type': game_type
        }
        
        optimization_result = set_optimizer.optimize_prediction_sets(
            [current_draw], game_config, 3
        )
        
        # Coverage optimization features
        overall_quality = optimization_result.get('overall_quality', {})
        features.extend([
            overall_quality.get('coverage_score', 0.0),
            overall_quality.get('diversity_score', 0.0)
        ])
        
        # Complementary set features
        features.extend([
            overall_quality.get('optimization_score', 0.0),
            overall_quality.get('balance_score', 0.0)
        ])
        
        # Balanced strategy features
        optimized_sets = optimization_result.get('optimized_sets', [])
        avg_score = np.mean([s.get('optimization_score', 0.0) for s in optimized_sets]) if optimized_sets else 0.0
        features.extend([
            avg_score,
            overall_quality.get('quality_score', 0.0)
        ])
        
        # Set optimization metrics
        features.extend([
            optimization_result.get('overall_optimization_score', avg_score),
            overall_quality.get('confidence_score', 0.0)
        ])
        
    except Exception as e:
        print(f"Set optimization feature extraction error: {e}")
        features = [0.0] * 8
    
    # Ensure exactly 8 features
    while len(features) < 8:
        features.append(0.0)
    
    return features[:8]


def _extract_temporal_intelligence_features(draws: List[List[int]], 
                                          draw_dates: List, 
                                          current_draw: List[int], 
                                          temporal_engine, 
                                          game_type: str) -> List[float]:
    """Extract Phase 4 temporal intelligence features."""
    features = []
    
    try:
        # Get temporal analysis using actual method
        temporal_analysis = temporal_engine.analyze_temporal_patterns(
            draws, draw_dates, game_type
        )
        
        # Seasonal pattern features
        seasonal_analysis = temporal_analysis.get('seasonal_analysis', {})
        features.extend([
            seasonal_analysis.get('seasonal_strength', 0.0),
            seasonal_analysis.get('monthly_pattern', 0.0),
            seasonal_analysis.get('weekly_pattern', 0.0)
        ])
        
        # Cyclical intelligence features
        cyclical_analysis = temporal_analysis.get('cyclical_analysis', {})
        features.extend([
            cyclical_analysis.get('cyclical_strength', 0.0),
            cyclical_analysis.get('frequency_cycles', 0.0)
        ])
        
        # Trend analysis features
        trend_analysis = temporal_analysis.get('trend_analysis', {})
        features.extend([
            trend_analysis.get('trend_strength', 0.0),
            trend_analysis.get('momentum_score', 0.0)
        ])
        
        # Temporal prediction quality
        features.extend([
            temporal_analysis.get('temporal_confidence', 0.0),
            temporal_analysis.get('prediction_stability', 0.0),
            temporal_analysis.get('confidence', 0.0)
        ])
        
    except Exception as e:
        print(f"Temporal intelligence feature extraction error: {e}")
        features = [0.0] * 10
    
    # Ensure exactly 10 features
    while len(features) < 10:
        features.append(0.0)
    
    return features[:10]


def _extract_basic_features(draw: List[int], pool_size: int) -> List[float]:
    """Extract basic statistical features from a draw."""
    if not draw:
        return [0.0] * 10
    
    features = []
    
    # Basic statistics
    features.extend([
        np.mean(draw),
        np.std(draw),
        np.min(draw),
        np.max(draw),
        np.max(draw) - np.min(draw),  # range
        len(draw)
    ])
    
    # Number properties
    even_count = sum(1 for x in draw if x % 2 == 0)
    features.extend([
        even_count / len(draw),  # even ratio
        sum(draw),  # sum
        np.median(draw),  # median
        len(set(draw)) / len(draw)  # uniqueness ratio
    ])
    
    return features


def _extract_frequency_features(historical_draws: List[List[int]], 
                               current_draw: List[int], 
                               pool_size: int) -> List[float]:
    """Extract frequency-based features."""
    if not historical_draws:
        return [0.0] * 30
    
    # Count frequencies
    frequencies = Counter()
    for draw in historical_draws:
        frequencies.update(draw)
    
    features = []
    
    # Hot/cold number analysis
    if frequencies:
        max_freq = max(frequencies.values())
        min_freq = min(frequencies.values())
        
        hot_numbers = [num for num, freq in frequencies.items() if freq >= max_freq * 0.8]
        cold_numbers = [num for num, freq in frequencies.items() if freq <= min_freq * 1.2]
        
        # Features for current draw
        hot_in_draw = sum(1 for num in current_draw if num in hot_numbers)
        cold_in_draw = sum(1 for num in current_draw if num in cold_numbers)
        
        features.extend([
            hot_in_draw / len(current_draw),
            cold_in_draw / len(current_draw),
            len(hot_numbers) / pool_size,
            len(cold_numbers) / pool_size
        ])
        
        # Frequency statistics for current draw
        current_freqs = [frequencies.get(num, 0) for num in current_draw]
        features.extend([
            np.mean(current_freqs),
            np.std(current_freqs),
            np.min(current_freqs),
            np.max(current_freqs)
        ])
    else:
        features.extend([0.0] * 8)
    
    # Pad to 30 features
    while len(features) < 30:
        features.append(0.0)
    
    return features[:30]
