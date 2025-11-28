"""
Phase 1: Mathematical Engine for lottery prediction system.

This engine implements statistical and mathematical analysis methods for
lottery number prediction, including frequency analysis, gap analysis,
hot/cold number detection, and statistical pattern recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from collections import Counter, defaultdict
import math
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PrimePatternAnalyzer:
    """Advanced analysis of prime number patterns in lottery draws"""
    
    def __init__(self):
        self.primes_cache = self._sieve_of_eratosthenes(100)
    
    def _sieve_of_eratosthenes(self, limit: int) -> set:
        """Generate prime numbers up to limit using Sieve of Eratosthenes"""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return {i for i, is_prime in enumerate(sieve) if is_prime}
    
    def _ensure_integer_data(self, draw):
        """Ensure draw data is converted to integers"""
        if isinstance(draw, list) and len(draw) > 0 and isinstance(draw[0], str):
            return [int(num) for num in draw if str(num).isdigit()]
        return draw

    def analyze_distribution(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze prime number distribution patterns in historical draws"""
        try:
            prime_counts = []
            prime_ratios = []
            prime_positions = []
            
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                primes_in_draw = [num for num in draw if num in self.primes_cache]
                prime_counts.append(len(primes_in_draw))
                
                if len(draw) > 0:
                    prime_ratios.append(len(primes_in_draw) / len(draw))
                    
                # Analyze positional distribution
                sorted_draw = sorted(draw)
                for i, num in enumerate(sorted_draw):
                    if num in self.primes_cache:
                        prime_positions.append(i / len(sorted_draw))
            
            analysis = {
                'average_prime_count': float(np.mean(prime_counts)) if prime_counts else 0.0,
                'prime_count_variance': float(np.var(prime_counts)) if prime_counts else 0.0,
                'average_prime_ratio': float(np.mean(prime_ratios)) if prime_ratios else 0.0,
                'positional_bias': float(np.mean(prime_positions)) if prime_positions else 0.5,
                'clustering_analysis': self._analyze_prime_clusters(historical_data),
                'gap_analysis': self._analyze_prime_gaps(historical_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in prime distribution analysis: {e}")
            return {}
    
    def _analyze_prime_clusters(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze clustering patterns of prime numbers in draws"""
        try:
            cluster_scores = []
            
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                primes_in_draw = sorted([num for num in draw if num in self.primes_cache])
                
                if len(primes_in_draw) >= 2:
                    # Calculate clustering score based on proximity
                    gaps = [primes_in_draw[i+1] - primes_in_draw[i] for i in range(len(primes_in_draw)-1)]
                    avg_gap = np.mean(gaps)
                    cluster_score = 1.0 / (1.0 + avg_gap/10.0)  # Inverse relationship with gap size
                    cluster_scores.append(cluster_score)
            
            return {
                'average_clustering': float(np.mean(cluster_scores)) if cluster_scores else 0.0,
                'clustering_variance': float(np.var(cluster_scores)) if cluster_scores else 0.0,
                'cluster_strength': float(len(cluster_scores) / len(historical_data)) if historical_data else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in prime cluster analysis: {e}")
            return {'average_clustering': 0.0, 'clustering_variance': 0.0, 'cluster_strength': 0.0}
    
    def _analyze_prime_gaps(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze gaps between prime numbers in draws"""
        try:
            all_gaps = []
            
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                primes_in_draw = sorted([num for num in draw if num in self.primes_cache])
                
                if len(primes_in_draw) >= 2:
                    gaps = [primes_in_draw[i+1] - primes_in_draw[i] for i in range(len(primes_in_draw)-1)]
                    all_gaps.extend(gaps)
            
            if all_gaps:
                return {
                    'average_gap': float(np.mean(all_gaps)),
                    'median_gap': float(np.median(all_gaps)),
                    'gap_variance': float(np.var(all_gaps)),
                    'most_common_gap': float(Counter(all_gaps).most_common(1)[0][0])
                }
            else:
                return {'average_gap': 0.0, 'median_gap': 0.0, 'gap_variance': 0.0, 'most_common_gap': 0.0}
                
        except Exception as e:
            logger.error(f"Error in prime gap analysis: {e}")
            return {'average_gap': 0.0, 'median_gap': 0.0, 'gap_variance': 0.0, 'most_common_gap': 0.0}


class NumberTheoryEngine:
    """Advanced number theory analysis for lottery patterns"""
    
    def __init__(self):
        self.fibonacci_cache = self._generate_fibonacci(100)
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def _generate_fibonacci(self, limit: int) -> List[int]:
        """Generate Fibonacci numbers up to limit"""
        fib = [1, 1]
        while fib[-1] < limit:
            fib.append(fib[-1] + fib[-2])
        return [f for f in fib if f <= limit]
    
    def _ensure_integer_data(self, draw):
        """Ensure draw data is converted to integers"""
        if isinstance(draw, list) and len(draw) > 0 and isinstance(draw[0], str):
            return [int(num) for num in draw if str(num).isdigit()]
        return draw

    def find_modular_patterns(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Find modular arithmetic patterns in lottery draws"""
        try:
            modular_patterns = {}
            
            # Analyze patterns for different moduli
            for mod in [3, 5, 7, 11, 13]:
                mod_distribution = defaultdict(int)
                mod_sequences = []
                
                for draw in historical_data:
                    draw = self._ensure_integer_data(draw)
                    
                    # Count numbers by modular class
                    for num in draw:
                        mod_distribution[num % mod] += 1
                    
                    # Create modular sequence for this draw
                    mod_sequence = sorted([num % mod for num in draw])
                    mod_sequences.append(mod_sequence)
                
                # Analyze distribution uniformity
                total_numbers = sum(mod_distribution.values())
                expected_per_class = total_numbers / mod
                
                chi_square = sum((count - expected_per_class) ** 2 / expected_per_class 
                               for count in mod_distribution.values())
                
                modular_patterns[f'mod_{mod}'] = {
                    'distribution': dict(mod_distribution),
                    'chi_square': float(chi_square),
                    'uniformity_score': 1.0 / (1.0 + chi_square / mod)
                }
            
            return modular_patterns
            
        except Exception as e:
            logger.error(f"Error in modular pattern analysis: {e}")
            return {}


class NumberGraphAnalyzer:
    """Graph theory analysis of number relationships"""
    
    def __init__(self):
        self.adjacency_matrix = None
        self.node_weights = None
    
    def _ensure_integer_data(self, draw):
        """Ensure draw data is converted to integers"""
        if isinstance(draw, list) and len(draw) > 0 and isinstance(draw[0], str):
            return [int(num) for num in draw if str(num).isdigit()]
        return draw

    def build_number_dependency_graph(self, historical_data: List[List[int]], 
                                    max_number: int = 50) -> Dict[str, Any]:
        """Build a graph of number co-occurrence relationships"""
        try:
            # Initialize adjacency matrix and node weights
            self.adjacency_matrix = np.zeros((max_number, max_number))
            self.node_weights = np.zeros(max_number)
            
            # Build graph from historical data
            for draw in historical_data:
                draw = self._ensure_integer_data(draw)
                
                # Update node weights (frequency)
                for num in draw:
                    if 1 <= num <= max_number:
                        self.node_weights[num - 1] += 1
                
                # Update edge weights (co-occurrence)
                for i, num1 in enumerate(draw):
                    for j, num2 in enumerate(draw):
                        if i != j and 1 <= num1 <= max_number and 1 <= num2 <= max_number:
                            self.adjacency_matrix[num1 - 1][num2 - 1] += 1
            
            # Normalize by number of draws
            if len(historical_data) > 0:
                self.adjacency_matrix /= len(historical_data)
                self.node_weights /= len(historical_data)
            
            # Calculate graph metrics
            graph_metrics = self._calculate_graph_metrics()
            
            # Find number clusters
            clusters = self._find_number_clusters()
            
            # Calculate centrality measures
            centrality = self._calculate_centrality_measures()
            
            return {
                'adjacency_matrix': self.adjacency_matrix.tolist(),
                'node_weights': self.node_weights.tolist(),
                'graph_metrics': graph_metrics,
                'number_clusters': clusters,
                'centrality_measures': centrality,
                'connectivity_score': float(np.sum(self.adjacency_matrix > 0) / (max_number * (max_number - 1))),
                'average_degree': float(np.mean(np.sum(self.adjacency_matrix > 0, axis=1)))
            }
            
        except Exception as e:
            logger.error(f"Error building dependency graph: {e}")
            return {}
    
    def _calculate_graph_metrics(self) -> Dict[str, float]:
        """Calculate various graph theory metrics"""
        try:
            if self.adjacency_matrix is None:
                return {}
            
            # Density
            n = self.adjacency_matrix.shape[0]
            total_possible_edges = n * (n - 1)
            actual_edges = np.sum(self.adjacency_matrix > 0)
            density = actual_edges / total_possible_edges if total_possible_edges > 0 else 0
            
            # Average clustering coefficient
            clustering_coeffs = []
            for i in range(n):
                neighbors = np.where(self.adjacency_matrix[i] > 0)[0]
                if len(neighbors) >= 2:
                    possible_connections = len(neighbors) * (len(neighbors) - 1)
                    actual_connections = 0
                    for j in neighbors:
                        for k in neighbors:
                            if j != k and self.adjacency_matrix[j][k] > 0:
                                actual_connections += 1
                    clustering_coeffs.append(actual_connections / possible_connections)
            
            avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0
            
            return {
                'density': float(density),
                'average_clustering': float(avg_clustering),
                'max_degree': float(np.max(np.sum(self.adjacency_matrix > 0, axis=1))),
                'min_degree': float(np.min(np.sum(self.adjacency_matrix > 0, axis=1)))
            }
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
            return {}
    
    def _find_number_clusters(self) -> List[List[int]]:
        """Find clusters of frequently co-occurring numbers"""
        try:
            if self.adjacency_matrix is None:
                return []
            
            # Simple clustering based on edge weights
            clusters = []
            used_nodes = set()
            
            for i in range(len(self.adjacency_matrix)):
                if i not in used_nodes:
                    cluster = [i + 1]  # Convert back to 1-based numbering
                    used_nodes.add(i)
                    
                    # Find strongly connected neighbors
                    for j in range(len(self.adjacency_matrix)):
                        if j not in used_nodes and self.adjacency_matrix[i][j] > 0.5:
                            cluster.append(j + 1)
                            used_nodes.add(j)
                    
                    if len(cluster) > 1:
                        clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error finding clusters: {e}")
            return []
    
    def _calculate_centrality_measures(self) -> Dict[str, List[float]]:
        """Calculate centrality measures for each node"""
        try:
            if self.adjacency_matrix is None:
                return {}
            
            n = len(self.adjacency_matrix)
            
            # Degree centrality
            degree_centrality = [sum(self.adjacency_matrix[i] > 0) / (n - 1) for i in range(n)]
            
            # Weighted degree centrality
            weighted_degree = [sum(self.adjacency_matrix[i]) for i in range(n)]
            max_weighted = max(weighted_degree) if weighted_degree else 1
            weighted_centrality = [w / max_weighted for w in weighted_degree]
            
            return {
                'degree_centrality': [float(x) for x in degree_centrality],
                'weighted_centrality': [float(x) for x in weighted_centrality]
            }
            
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return {}


class CombinatorialOptimizer:
    """Advanced combinatorial optimization for number selection"""
    
    def optimize_number_combinations(self, historical_data: List[List[int]], 
                                   target_size: int = 6, max_number: int = 49) -> Dict[str, Any]:
        """Optimize number combinations using mathematical principles"""
        try:
            # Analyze number frequencies
            frequency_map = defaultdict(int)
            for draw in historical_data:
                for num in draw:
                    if isinstance(num, str):
                        num = int(num) if num.isdigit() else 0
                    if 1 <= num <= max_number:
                        frequency_map[num] += 1
            
            # Calculate optimization metrics
            total_draws = len(historical_data)
            expected_frequency = total_draws * target_size / max_number
            
            optimization_scores = {}
            for num in range(1, max_number + 1):
                freq = frequency_map.get(num, 0)
                deviation = abs(freq - expected_frequency)
                optimization_scores[num] = 1.0 / (1.0 + deviation / expected_frequency)
            
            # Find optimal combinations
            sorted_numbers = sorted(optimization_scores.keys(), 
                                  key=lambda x: optimization_scores[x], reverse=True)
            
            return {
                'frequency_map': dict(frequency_map),
                'optimization_scores': optimization_scores,
                'recommended_numbers': sorted_numbers[:target_size],
                'diversity_score': self._calculate_diversity_score(sorted_numbers[:target_size])
            }
            
        except Exception as e:
            logger.error(f"Error in combinatorial optimization: {e}")
            return {}
    
    def _calculate_diversity_score(self, numbers: List[int]) -> float:
        """Calculate diversity score for a set of numbers"""
        try:
            if len(numbers) < 2:
                return 0.0
            
            # Range diversity
            range_score = (max(numbers) - min(numbers)) / max(numbers)
            
            # Gap uniformity
            sorted_nums = sorted(numbers)
            gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
            gap_variance = np.var(gaps) if gaps else 0
            gap_score = 1.0 / (1.0 + gap_variance)
            
            return float((range_score + gap_score) / 2)
            
        except Exception as e:
            logger.error(f"Error calculating diversity score: {e}")
            return 0.0


class MathematicalEngine:
    """
    Mathematical analysis engine for lottery predictions.
    
    Implements various statistical methods including:
    - Frequency analysis
    - Gap analysis  
    - Hot/cold number detection
    - Statistical distribution analysis
    - Regression analysis
    - Moving averages
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Mathematical Engine.
        
        Args:
            config: Configuration dictionary containing engine parameters
        """
        self.config = config
        self.game_config = config.get('game_config', {})
        self.analysis_config = config.get('analysis_config', {})
        
        # Game parameters
        self.number_range = self.game_config.get('number_range', [1, 49])
        self.numbers_per_draw = self.game_config.get('numbers_per_draw', 6)
        self.has_bonus = self.game_config.get('has_bonus', False)
        
        # Analysis parameters
        self.hot_cold_threshold = self.analysis_config.get('hot_cold_threshold', 0.2)
        self.gap_analysis_window = self.analysis_config.get('gap_analysis_window', 50)
        self.frequency_weight = self.analysis_config.get('frequency_weight', 0.3)
        self.gap_weight = self.analysis_config.get('gap_weight', 0.3)
        self.trend_weight = self.analysis_config.get('trend_weight', 0.4)
        
        # Initialize sophisticated analysis components
        self.prime_analyzer = PrimePatternAnalyzer()
        self.number_theory_engine = NumberTheoryEngine()
        self.graph_analyzer = NumberGraphAnalyzer()
        self.combinatorial_optimizer = CombinatorialOptimizer()
        
        # Analysis cache for performance
        self.analysis_cache = {}
        
        logger.info("Enhanced Mathematical Engine initialized with sophisticated analysis components")
        
        # Initialize data structures
        self.historical_data = None
        self.frequency_stats = {}
        self.gap_stats = {}
        self.trend_stats = {}
        self.last_analysis_time = None
        
        # Statistical models
        self.scaler = StandardScaler()
        self.trained = False
        
        logger.info("🔢 Mathematical Engine initialized")
    
    def load_data(self, historical_data: pd.DataFrame) -> None:
        """
        Load historical lottery data for analysis.
        
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
            
            logger.info(f"📊 Loaded {len(self.historical_data)} historical draws")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def _process_numbers_format(self) -> None:
        """Process numbers column to ensure consistent format."""
        processed_numbers = []
        
        for _, row in self.historical_data.iterrows():
            numbers = row['numbers']
            
            if isinstance(numbers, str):
                # Parse string format "1,2,3,4,5,6"
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
    
    def analyze_frequency(self) -> Dict[str, Any]:
        """
        Perform frequency analysis on historical data.
        
        Returns:
            Dictionary containing frequency statistics
        """
        try:
            all_numbers = []
            for numbers in self.historical_data['processed_numbers']:
                all_numbers.extend(numbers)
            
            # Count frequencies
            frequency_counter = Counter(all_numbers)
            total_draws = len(self.historical_data)
            
            # Calculate statistics for each number
            frequency_stats = {}
            for num in range(self.number_range[0], self.number_range[1] + 1):
                count = frequency_counter.get(num, 0)
                frequency = count / total_draws if total_draws > 0 else 0
                
                frequency_stats[num] = {
                    'count': count,
                    'frequency': frequency,
                    'percentage': frequency * 100,
                    'expected_frequency': self.numbers_per_draw / (self.number_range[1] - self.number_range[0] + 1)
                }
                
                # Calculate deviation from expected
                expected = frequency_stats[num]['expected_frequency']
                frequency_stats[num]['deviation'] = frequency - expected
                frequency_stats[num]['z_score'] = (frequency - expected) / math.sqrt(expected * (1 - expected) / total_draws) if total_draws > 0 else 0
            
            # Identify hot and cold numbers
            frequencies = [stats['frequency'] for stats in frequency_stats.values()]
            mean_freq = np.mean(frequencies)
            std_freq = np.std(frequencies)
            
            hot_threshold = mean_freq + (self.hot_cold_threshold * std_freq)
            cold_threshold = mean_freq - (self.hot_cold_threshold * std_freq)
            
            hot_numbers = [num for num, stats in frequency_stats.items() if stats['frequency'] > hot_threshold]
            cold_numbers = [num for num, stats in frequency_stats.items() if stats['frequency'] < cold_threshold]
            
            self.frequency_stats = {
                'number_stats': frequency_stats,
                'hot_numbers': hot_numbers,
                'cold_numbers': cold_numbers,
                'mean_frequency': mean_freq,
                'std_frequency': std_freq,
                'total_draws': total_draws
            }
            
            logger.info(f"📊 Frequency analysis complete: {len(hot_numbers)} hot, {len(cold_numbers)} cold numbers")
            return self.frequency_stats
            
        except Exception as e:
            logger.error(f"❌ Frequency analysis failed: {e}")
            raise
    
    def analyze_gaps(self) -> Dict[str, Any]:
        """
        Perform gap analysis to identify patterns in number appearances.
        
        Returns:
            Dictionary containing gap analysis results
        """
        try:
            gap_stats = {}
            
            for num in range(self.number_range[0], self.number_range[1] + 1):
                appearances = []
                
                # Find all appearances of this number
                for idx, numbers in enumerate(self.historical_data['processed_numbers']):
                    if num in numbers:
                        appearances.append(idx)
                
                if len(appearances) < 2:
                    gap_stats[num] = {
                        'gaps': [],
                        'mean_gap': 0,
                        'std_gap': 0,
                        'last_seen': -1,
                        'current_gap': len(self.historical_data),
                        'predicted_next': len(self.historical_data) + 1
                    }
                    continue
                
                # Calculate gaps between appearances
                gaps = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
                
                gap_stats[num] = {
                    'gaps': gaps,
                    'mean_gap': np.mean(gaps),
                    'std_gap': np.std(gaps) if len(gaps) > 1 else 0,
                    'min_gap': min(gaps),
                    'max_gap': max(gaps),
                    'last_seen': appearances[-1],
                    'current_gap': len(self.historical_data) - 1 - appearances[-1],
                    'total_appearances': len(appearances)
                }
                
                # Predict next appearance based on average gap
                gap_stats[num]['predicted_next'] = appearances[-1] + gap_stats[num]['mean_gap']
                
                # Calculate gap score (how overdue is this number)
                if gap_stats[num]['mean_gap'] > 0:
                    gap_stats[num]['overdue_score'] = gap_stats[num]['current_gap'] / gap_stats[num]['mean_gap']
                else:
                    gap_stats[num]['overdue_score'] = 0
            
            # Identify overdue numbers
            overdue_scores = [stats['overdue_score'] for stats in gap_stats.values()]
            mean_overdue = np.mean(overdue_scores)
            std_overdue = np.std(overdue_scores)
            
            overdue_threshold = mean_overdue + std_overdue
            overdue_numbers = [num for num, stats in gap_stats.items() if stats['overdue_score'] > overdue_threshold]
            
            self.gap_stats = {
                'number_stats': gap_stats,
                'overdue_numbers': overdue_numbers,
                'mean_overdue_score': mean_overdue,
                'overdue_threshold': overdue_threshold
            }
            
            logger.info(f"📈 Gap analysis complete: {len(overdue_numbers)} overdue numbers identified")
            return self.gap_stats
            
        except Exception as e:
            logger.error(f"❌ Gap analysis failed: {e}")
            raise
    
    def analyze_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in number appearances over time.
        
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            trend_stats = {}
            window_size = min(self.gap_analysis_window, len(self.historical_data) // 4)
            
            if window_size < 10:
                logger.warning("⚠️ Insufficient data for trend analysis")
                return {}
            
            for num in range(self.number_range[0], self.number_range[1] + 1):
                # Create time series of appearances
                appearances = np.zeros(len(self.historical_data))
                
                for idx, numbers in enumerate(self.historical_data['processed_numbers']):
                    if num in numbers:
                        appearances[idx] = 1
                
                # Calculate rolling averages
                rolling_freq = pd.Series(appearances).rolling(window=window_size, min_periods=1).mean()
                
                # Calculate trend direction
                recent_freq = rolling_freq.iloc[-window_size:].mean()
                earlier_freq = rolling_freq.iloc[:-window_size].mean() if len(rolling_freq) > window_size else recent_freq
                
                trend_direction = recent_freq - earlier_freq
                
                # Linear regression for trend
                x = np.arange(len(appearances))
                if np.sum(appearances) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, appearances)
                else:
                    slope = intercept = r_value = p_value = std_err = 0
                
                trend_stats[num] = {
                    'recent_frequency': recent_freq,
                    'earlier_frequency': earlier_freq,
                    'trend_direction': trend_direction,
                    'trend_slope': slope,
                    'trend_correlation': r_value,
                    'trend_significance': p_value,
                    'rolling_frequency': rolling_freq.tolist()
                }
                
                # Classify trend
                if abs(trend_direction) < 0.01:
                    trend_stats[num]['trend_class'] = 'stable'
                elif trend_direction > 0:
                    trend_stats[num]['trend_class'] = 'increasing'
                else:
                    trend_stats[num]['trend_class'] = 'decreasing'
            
            # Identify trending numbers
            increasing_numbers = [num for num, stats in trend_stats.items() if stats['trend_class'] == 'increasing']
            decreasing_numbers = [num for num, stats in trend_stats.items() if stats['trend_class'] == 'decreasing']
            
            self.trend_stats = {
                'number_stats': trend_stats,
                'increasing_numbers': increasing_numbers,
                'decreasing_numbers': decreasing_numbers,
                'analysis_window': window_size
            }
            
            logger.info(f"📈 Trend analysis complete: {len(increasing_numbers)} increasing, {len(decreasing_numbers)} decreasing")
            return self.trend_stats
            
        except Exception as e:
            logger.error(f"❌ Trend analysis failed: {e}")
            raise
    
    def train(self) -> None:
        """Train the mathematical engine on historical data."""
        try:
            if self.historical_data is None:
                raise ValueError("No historical data loaded")
            
            logger.info("🎓 Training Mathematical Engine...")
            
            # Perform all analyses
            self.analyze_frequency()
            self.analyze_gaps()
            self.analyze_trends()
            
            # Create feature matrix for ML components
            self._create_feature_matrix()
            
            self.trained = True
            self.last_analysis_time = datetime.now()
            
            logger.info("✅ Mathematical Engine training complete")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def _create_feature_matrix(self) -> None:
        """Create feature matrix for additional ML analysis."""
        try:
            features = []
            
            for num in range(self.number_range[0], self.number_range[1] + 1):
                feature_vector = [
                    self.frequency_stats['number_stats'][num]['frequency'],
                    self.frequency_stats['number_stats'][num]['deviation'],
                    self.frequency_stats['number_stats'][num]['z_score'],
                    self.gap_stats['number_stats'][num]['mean_gap'],
                    self.gap_stats['number_stats'][num]['current_gap'],
                    self.gap_stats['number_stats'][num]['overdue_score'],
                    self.trend_stats['number_stats'][num]['trend_direction'],
                    self.trend_stats['number_stats'][num]['trend_slope'],
                    self.trend_stats['number_stats'][num]['trend_correlation']
                ]
                features.append(feature_vector)
            
            self.feature_matrix = np.array(features)
            self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
            
        except Exception as e:
            logger.error(f"❌ Feature matrix creation failed: {e}")
            raise
    
    def predict(self, num_predictions: int = 1, strategy: str = 'balanced') -> List[Dict[str, Any]]:
        """
        Generate predictions using mathematical analysis.
        
        Args:
            num_predictions: Number of prediction sets to generate
            strategy: Prediction strategy ('balanced', 'hot', 'cold', 'overdue', 'trending')
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if not self.trained:
                raise ValueError("Engine not trained. Call train() first.")
            
            predictions = []
            
            for i in range(num_predictions):
                if strategy == 'balanced':
                    prediction = self._predict_balanced()
                elif strategy == 'hot':
                    prediction = self._predict_hot_numbers()
                elif strategy == 'cold':
                    prediction = self._predict_cold_numbers()
                elif strategy == 'overdue':
                    prediction = self._predict_overdue_numbers()
                elif strategy == 'trending':
                    prediction = self._predict_trending_numbers()
                else:
                    prediction = self._predict_balanced()
                
                predictions.append({
                    'numbers': prediction['numbers'],
                    'confidence': prediction['confidence'],
                    'strategy': strategy,
                    'analysis_scores': prediction['scores'],
                    'generated_at': datetime.now().isoformat(),
                    'engine': 'mathematical'
                })
            
            logger.info(f"🎯 Generated {num_predictions} mathematical predictions using {strategy} strategy")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            raise
    
    def _predict_balanced(self) -> Dict[str, Any]:
        """Generate balanced prediction using all analysis methods."""
        scores = {}
        
        for num in range(self.number_range[0], self.number_range[1] + 1):
            # Frequency score
            freq_score = self.frequency_stats['number_stats'][num]['frequency'] * self.frequency_weight
            
            # Gap score
            gap_score = self.gap_stats['number_stats'][num]['overdue_score'] * self.gap_weight
            
            # Trend score
            trend_score = max(0, self.trend_stats['number_stats'][num]['trend_direction']) * self.trend_weight
            
            # Combined score
            scores[num] = freq_score + gap_score + trend_score
        
        # Select top numbers
        sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_numbers = [num for num, score in sorted_numbers[:self.numbers_per_draw]]
        
        # Calculate confidence
        top_scores = [score for _, score in sorted_numbers[:self.numbers_per_draw]]
        confidence = min(0.95, np.mean(top_scores) / max(scores.values()) if max(scores.values()) > 0 else 0.5)
        
        return {
            'numbers': sorted(selected_numbers),
            'confidence': confidence,
            'scores': {num: scores[num] for num in selected_numbers}
        }
    
    def _predict_hot_numbers(self) -> Dict[str, Any]:
        """Generate prediction based on hot numbers."""
        hot_numbers = self.frequency_stats['hot_numbers']
        
        if len(hot_numbers) >= self.numbers_per_draw:
            # Select from hot numbers
            hot_scores = {num: self.frequency_stats['number_stats'][num]['frequency'] 
                         for num in hot_numbers}
            sorted_hot = sorted(hot_scores.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in sorted_hot[:self.numbers_per_draw]]
        else:
            # Supplement with high frequency numbers
            all_scores = {num: self.frequency_stats['number_stats'][num]['frequency'] 
                         for num in range(self.number_range[0], self.number_range[1] + 1)}
            sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in sorted_all[:self.numbers_per_draw]]
        
        confidence = 0.7  # Moderate confidence for hot number strategy
        
        return {
            'numbers': sorted(selected_numbers),
            'confidence': confidence,
            'scores': {num: self.frequency_stats['number_stats'][num]['frequency'] 
                      for num in selected_numbers}
        }
    
    def _predict_cold_numbers(self) -> Dict[str, Any]:
        """Generate prediction based on cold numbers (due for appearance)."""
        cold_numbers = self.frequency_stats['cold_numbers']
        
        if len(cold_numbers) >= self.numbers_per_draw:
            # Select from cold numbers with highest overdue scores
            cold_scores = {num: self.gap_stats['number_stats'][num]['overdue_score'] 
                          for num in cold_numbers}
            sorted_cold = sorted(cold_scores.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in sorted_cold[:self.numbers_per_draw]]
        else:
            # Supplement with most overdue numbers
            all_overdue = {num: self.gap_stats['number_stats'][num]['overdue_score'] 
                          for num in range(self.number_range[0], self.number_range[1] + 1)}
            sorted_overdue = sorted(all_overdue.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in sorted_overdue[:self.numbers_per_draw]]
        
        confidence = 0.6  # Lower confidence for cold number strategy
        
        return {
            'numbers': sorted(selected_numbers),
            'confidence': confidence,
            'scores': {num: self.gap_stats['number_stats'][num]['overdue_score'] 
                      for num in selected_numbers}
        }
    
    def _predict_overdue_numbers(self) -> Dict[str, Any]:
        """Generate prediction based on overdue analysis."""
        overdue_numbers = self.gap_stats['overdue_numbers']
        
        if len(overdue_numbers) >= self.numbers_per_draw:
            overdue_scores = {num: self.gap_stats['number_stats'][num]['overdue_score'] 
                            for num in overdue_numbers}
            sorted_overdue = sorted(overdue_scores.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in sorted_overdue[:self.numbers_per_draw]]
        else:
            # Get all numbers sorted by overdue score
            all_overdue = {num: self.gap_stats['number_stats'][num]['overdue_score'] 
                          for num in range(self.number_range[0], self.number_range[1] + 1)}
            sorted_overdue = sorted(all_overdue.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in sorted_overdue[:self.numbers_per_draw]]
        
        confidence = 0.65
        
        return {
            'numbers': sorted(selected_numbers),
            'confidence': confidence,
            'scores': {num: self.gap_stats['number_stats'][num]['overdue_score'] 
                      for num in selected_numbers}
        }
    
    def _predict_trending_numbers(self) -> Dict[str, Any]:
        """Generate prediction based on trending analysis."""
        increasing_numbers = self.trend_stats['increasing_numbers']
        
        if len(increasing_numbers) >= self.numbers_per_draw:
            trend_scores = {num: self.trend_stats['number_stats'][num]['trend_direction'] 
                           for num in increasing_numbers}
            sorted_trending = sorted(trend_scores.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in sorted_trending[:self.numbers_per_draw]]
        else:
            # Get all numbers sorted by trend direction
            all_trends = {num: self.trend_stats['number_stats'][num]['trend_direction'] 
                         for num in range(self.number_range[0], self.number_range[1] + 1)}
            sorted_trends = sorted(all_trends.items(), key=lambda x: x[1], reverse=True)
            selected_numbers = [num for num, _ in sorted_trends[:self.numbers_per_draw]]
        
        confidence = 0.6
        
        return {
            'numbers': sorted(selected_numbers),
            'confidence': confidence,
            'scores': {num: self.trend_stats['number_stats'][num]['trend_direction'] 
                      for num in selected_numbers}
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis summary.
        
        Returns:
            Dictionary containing analysis summary
        """
        if not self.trained:
            return {'error': 'Engine not trained'}
        
        return {
            'engine': 'mathematical',
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'data_points': len(self.historical_data) if self.historical_data is not None else 0,
            'frequency_analysis': {
                'hot_numbers': self.frequency_stats.get('hot_numbers', []),
                'cold_numbers': self.frequency_stats.get('cold_numbers', []),
                'total_draws': self.frequency_stats.get('total_draws', 0)
            },
            'gap_analysis': {
                'overdue_numbers': self.gap_stats.get('overdue_numbers', []),
                'overdue_threshold': self.gap_stats.get('overdue_threshold', 0)
            },
            'trend_analysis': {
                'increasing_numbers': self.trend_stats.get('increasing_numbers', []),
                'decreasing_numbers': self.trend_stats.get('decreasing_numbers', [])
            },
            'available_strategies': ['balanced', 'hot', 'cold', 'overdue', 'trending']
        }
    
    def get_number_analysis(self, number: int) -> Dict[str, Any]:
        """
        Get detailed analysis for a specific number.
        
        Args:
            number: The lottery number to analyze
            
        Returns:
            Dictionary containing detailed number analysis
        """
        if not self.trained:
            return {'error': 'Engine not trained'}
        
        if not (self.number_range[0] <= number <= self.number_range[1]):
            return {'error': f'Number {number} outside valid range {self.number_range}'}
        
        return {
            'number': number,
            'frequency_stats': self.frequency_stats['number_stats'].get(number, {}),
            'gap_stats': self.gap_stats['number_stats'].get(number, {}),
            'trend_stats': self.trend_stats['number_stats'].get(number, {}),
            'classifications': {
                'is_hot': number in self.frequency_stats.get('hot_numbers', []),
                'is_cold': number in self.frequency_stats.get('cold_numbers', []),
                'is_overdue': number in self.gap_stats.get('overdue_numbers', []),
                'is_trending_up': number in self.trend_stats.get('increasing_numbers', []),
                'is_trending_down': number in self.trend_stats.get('decreasing_numbers', [])
            }
        }

    def _ensure_integer_data(self, historical_data: List[List]) -> List[List[int]]:
        """Ensure all data is converted to integers"""
        processed_data = []
        for draw in historical_data:
            if not isinstance(draw, (list, tuple)):
                continue
                
            validated_draw = []
            for num in draw:
                try:
                    if isinstance(num, str):
                        if num.strip().isdigit():
                            validated_draw.append(int(num))
                    elif isinstance(num, (int, float)):
                        validated_draw.append(int(num))
                except (ValueError, TypeError):
                    continue
            
            if validated_draw:  # Only add non-empty draws
                processed_data.append(validated_draw)
        return processed_data

    def analyze_deep_patterns(self, historical_data: List[List[int]], game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Perform comprehensive mathematical analysis of historical lottery data"""
        try:
            # Ensure data is in correct format
            historical_data = self._ensure_integer_data(historical_data)
            
            # Create cache key
            cache_key = f"{game_type}_{len(historical_data)}_{hash(str(historical_data[-10:]))}"
            
            if cache_key in self.analysis_cache:
                logger.info("Using cached mathematical analysis")
                return self.analysis_cache[cache_key]
            
            logger.info("Performing deep mathematical pattern analysis...")
            
            # Determine game parameters
            max_number = self.number_range[1]
            
            # Prime number analysis
            logger.info("Analyzing prime number patterns...")
            prime_patterns = self.prime_analyzer.analyze_distribution(historical_data)
            
            # Number theory analysis
            logger.info("Analyzing number theory patterns...")
            modular_patterns = self.number_theory_engine.find_modular_patterns(historical_data)
            
            # Graph analysis
            logger.info("Building number dependency graph...")
            graph_analysis = self.graph_analyzer.build_number_dependency_graph(historical_data, max_number)
            
            # Combinatorial optimization
            logger.info("Optimizing number combinations...")
            target_size = self.numbers_per_draw
            optimization_results = self.combinatorial_optimizer.optimize_number_combinations(
                historical_data, target_size, max_number
            )
            
            # Combine all analyses
            comprehensive_analysis = {
                'prime_patterns': prime_patterns,
                'modular_patterns': modular_patterns,
                'graph_analysis': graph_analysis,
                'optimization_results': optimization_results,
                'overall_confidence': self._calculate_overall_confidence(
                    prime_patterns, modular_patterns, graph_analysis, optimization_results
                ),
                'analysis_timestamp': datetime.now().isoformat(),
                'game_type': game_type,
                'data_size': len(historical_data)
            }
            
            # Cache result
            self.analysis_cache[cache_key] = comprehensive_analysis
            
            logger.info("Mathematical analysis complete. Overall confidence: {:.3f}".format(
                comprehensive_analysis['overall_confidence']
            ))
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error in deep pattern analysis: {e}")
            return {'error': str(e), 'overall_confidence': 0.0}
    
    def _calculate_overall_confidence(self, prime_patterns: Dict, modular_patterns: Dict,
                                    graph_analysis: Dict, optimization_results: Dict) -> float:
        """Calculate overall confidence score from all analyses"""
        try:
            confidence_components = []
            
            # Prime pattern confidence
            if prime_patterns and 'average_prime_count' in prime_patterns:
                prime_confidence = min(1.0, prime_patterns.get('average_prime_count', 0) / 3.0)
                confidence_components.append(prime_confidence)
            
            # Modular pattern confidence
            if modular_patterns:
                avg_uniformity = np.mean([
                    pattern.get('uniformity_score', 0) 
                    for pattern in modular_patterns.values()
                ])
                confidence_components.append(avg_uniformity)
            
            # Graph analysis confidence
            if graph_analysis and 'graph_metrics' in graph_analysis:
                graph_confidence = graph_analysis['graph_metrics'].get('density', 0)
                confidence_components.append(graph_confidence)
            
            # Optimization confidence
            if optimization_results and 'diversity_score' in optimization_results:
                opt_confidence = optimization_results['diversity_score']
                confidence_components.append(opt_confidence)
            
            # Calculate weighted average
            if confidence_components:
                overall_confidence = np.mean(confidence_components)
                return float(max(0.0, min(1.0, overall_confidence)))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.0

    def generate_mathematical_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from mathematical analysis"""
        try:
            insights = {
                'high_confidence_numbers': [],
                'pattern_recommendations': [],
                'risk_assessment': 'medium',
                'strategy_suggestions': []
            }
            
            # Extract high-confidence numbers from optimization results
            if 'optimization_results' in analysis_results:
                opt_results = analysis_results['optimization_results']
                if 'recommended_numbers' in opt_results:
                    insights['high_confidence_numbers'] = opt_results['recommended_numbers'][:10]
            
            # Generate pattern-based recommendations
            if 'prime_patterns' in analysis_results:
                prime_data = analysis_results['prime_patterns']
                avg_primes = prime_data.get('average_prime_count', 0)
                if avg_primes > 2:
                    insights['pattern_recommendations'].append("Consider including 2-3 prime numbers")
            
            # Assess risk based on overall confidence
            confidence = analysis_results.get('overall_confidence', 0)
            if confidence > 0.7:
                insights['risk_assessment'] = 'low'
            elif confidence < 0.3:
                insights['risk_assessment'] = 'high'
            
            # Generate strategy suggestions
            if 'graph_analysis' in analysis_results:
                graph_data = analysis_results['graph_analysis']
                if graph_data.get('connectivity_score', 0) > 0.5:
                    insights['strategy_suggestions'].append("Focus on historically connected number pairs")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating mathematical insights: {e}")
            return {'error': str(e)}

    def get_mathematical_insights(self, historical_data: List[List[int]], game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Get mathematical insights by analyzing historical data and generating actionable recommendations"""
        try:
            logger.info("Generating mathematical insights for lottery predictions...")
            
            # Perform deep pattern analysis
            analysis_results = self.analyze_deep_patterns(historical_data, game_type)
            
            # Generate actionable insights from the analysis
            insights = self.generate_mathematical_insights(analysis_results)
            
            # Add analysis metadata
            insights['analysis_metadata'] = {
                'confidence_level': analysis_results.get('overall_confidence', 0.0),
                'data_points_analyzed': len(historical_data),
                'game_type': game_type,
                'analysis_timestamp': analysis_results.get('analysis_timestamp', datetime.now().isoformat())
            }
            
            logger.info(f"Mathematical insights generated with confidence: {insights['analysis_metadata']['confidence_level']:.3f}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting mathematical insights: {e}")
            return {
                'high_confidence_numbers': [],
                'pattern_recommendations': [],
                'risk_assessment': 'high',
                'strategy_suggestions': [],
                'error': str(e),
                'analysis_metadata': {
                    'confidence_level': 0.0,
                    'data_points_analyzed': 0,
                    'game_type': game_type,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }