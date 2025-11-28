#!/usr/bin/env python3
"""
Advanced Mathematical Engine for Ultra-High Accuracy Lottery Prediction

This module implements sophisticated mathematical analysis including:
- Prime number distribution patterns
- Number theory applications
- Advanced probability modeling
- Mathematical relationship discovery
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import math
from scipy import stats
from scipy.special import comb
import itertools
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PrimePatternAnalyzer:
    """Analyzes prime number distribution patterns in lottery draws"""
    
    def __init__(self):
        self.primes_cache = self._generate_primes(100)  # Cache primes up to 100
    
    def _generate_primes(self, limit: int) -> List[int]:
        """Generate prime numbers up to limit using Sieve of Eratosthenes"""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
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


class AdvancedMathematicalEngine:
    """Main engine coordinating all mathematical analysis components"""
    
    def __init__(self):
        self.prime_analyzer = PrimePatternAnalyzer()
        self.number_theory_engine = NumberTheoryEngine()
        self.graph_analyzer = NumberGraphAnalyzer()
        self.combinatorial_optimizer = CombinatorialOptimizer()
        self.analysis_cache = {}
    
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
            max_number = 50 if 'max' in game_type.lower() else 49
            
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
            target_size = 7 if 'max' in game_type.lower() else 6
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
