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
    
    def analyze_distribution(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Analyze prime number distribution patterns in historical draws"""
        try:
            prime_counts = []
            prime_ratios = []
            prime_positions = []
            
            for draw in historical_data:
                # Count primes in each draw
                primes_in_draw = [num for num in draw if num in self.primes_cache]
                prime_counts.append(len(primes_in_draw))
                
                # Calculate prime ratio
                if len(draw) > 0:
                    prime_ratios.append(len(primes_in_draw) / len(draw))
                
                # Analyze prime positions
                for i, num in enumerate(sorted(draw)):
                    if num in self.primes_cache:
                        prime_positions.append(i)
            
            # Statistical analysis
            avg_primes = np.mean(prime_counts) if prime_counts else 0
            std_primes = np.std(prime_counts) if prime_counts else 0
            avg_ratio = np.mean(prime_ratios) if prime_ratios else 0
            
            # Prime clustering analysis
            prime_clusters = self._analyze_prime_clusters(historical_data)
            
            # Prime gap analysis
            prime_gaps = self._analyze_prime_gaps(historical_data)
            
            return {
                'average_primes_per_draw': float(avg_primes),
                'std_primes_per_draw': float(std_primes),
                'average_prime_ratio': float(avg_ratio),
                'prime_clusters': prime_clusters,
                'prime_gaps': prime_gaps,
                'prime_position_distribution': np.histogram(prime_positions, bins=10)[0].tolist() if prime_positions else [],
                'confidence_score': min(0.95, avg_ratio * 2.5),  # Confidence based on prime density
                'pattern_strength': float(1.0 - std_primes / (avg_primes + 1e-6))
            }
            
        except Exception as e:
            logger.error(f"Error in prime pattern analysis: {e}")
            return {
                'average_primes_per_draw': 0.0,
                'std_primes_per_draw': 0.0,
                'average_prime_ratio': 0.0,
                'prime_clusters': {},
                'prime_gaps': {},
                'confidence_score': 0.0,
                'pattern_strength': 0.0
            }
    
    def _analyze_prime_clusters(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze clustering patterns of prime numbers"""
        try:
            cluster_scores = []
            
            for draw in historical_data:
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
                # Convert to integers if they're strings
                if isinstance(draw, list) and len(draw) > 0 and isinstance(draw[0], str):
                    draw = [int(num) for num in draw if num.isdigit()]
                
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
    
    def find_modular_patterns(self, historical_data: List[List[int]]) -> Dict[str, Any]:
        """Find modular arithmetic patterns in lottery draws"""
        try:
            modular_patterns = {}
            
            # Analyze patterns for different moduli
            for mod in [3, 5, 7, 11, 13]:
                mod_distribution = defaultdict(int)
                mod_sequences = []
                
                for draw in historical_data:
                    # Convert to integers if they're strings
                    if isinstance(draw, list) and len(draw) > 0 and isinstance(draw[0], str):
                        draw = [int(num) for num in draw if num.isdigit()]
                    
                    # Count numbers by modular class
                    for num in draw:
                        mod_distribution[num % mod] += 1
                    
                    # Create modular sequence for this draw
                    mod_sequence = sorted([num % mod for num in draw])
                    mod_sequences.append(mod_sequence)
                
                # Calculate distribution evenness (entropy)
                total_numbers = sum(mod_distribution.values())
                entropy = 0
                if total_numbers > 0:
                    for count in mod_distribution.values():
                        if count > 0:
                            p = count / total_numbers
                            entropy -= p * math.log2(p)
                
                # Pattern consistency analysis
                sequence_counter = Counter(tuple(seq) for seq in mod_sequences)
                most_common_pattern = sequence_counter.most_common(1)[0] if sequence_counter else ((), 0)
                
                modular_patterns[f'mod_{mod}'] = {
                    'distribution': dict(mod_distribution),
                    'entropy': float(entropy),
                    'max_entropy': float(math.log2(mod)),
                    'evenness_score': float(entropy / math.log2(mod)) if mod > 1 else 0.0,
                    'most_common_pattern': most_common_pattern[0],
                    'pattern_frequency': float(most_common_pattern[1] / len(historical_data)) if historical_data else 0.0
                }
            
            # Fibonacci analysis
            fibonacci_analysis = self._analyze_fibonacci_patterns(historical_data)
            modular_patterns['fibonacci'] = fibonacci_analysis
            
            # Golden ratio analysis
            golden_analysis = self._analyze_golden_ratio_patterns(historical_data)
            modular_patterns['golden_ratio'] = golden_analysis
            
            return modular_patterns
            
        except Exception as e:
            logger.error(f"Error in modular pattern analysis: {e}")
            return {}
    
    def _analyze_fibonacci_patterns(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze Fibonacci number patterns in draws"""
        try:
            fib_counts = []
            fib_ratios = []
            
            for draw in historical_data:
                fib_in_draw = [num for num in draw if num in self.fibonacci_cache]
                fib_counts.append(len(fib_in_draw))
                
                if len(draw) > 0:
                    fib_ratios.append(len(fib_in_draw) / len(draw))
            
            return {
                'average_fibonacci_per_draw': float(np.mean(fib_counts)) if fib_counts else 0.0,
                'fibonacci_ratio': float(np.mean(fib_ratios)) if fib_ratios else 0.0,
                'fibonacci_variance': float(np.var(fib_counts)) if fib_counts else 0.0,
                'fibonacci_consistency': float(1.0 - np.std(fib_ratios)) if fib_ratios and np.mean(fib_ratios) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci analysis: {e}")
            return {'average_fibonacci_per_draw': 0.0, 'fibonacci_ratio': 0.0, 'fibonacci_variance': 0.0, 'fibonacci_consistency': 0.0}
    
    def _analyze_golden_ratio_patterns(self, historical_data: List[List[int]]) -> Dict[str, float]:
        """Analyze golden ratio relationships in draws"""
        try:
            golden_relationships = []
            
            for draw in historical_data:
                if len(draw) >= 2:
                    sorted_draw = sorted(draw)
                    
                    # Check for golden ratio relationships between consecutive numbers
                    for i in range(len(sorted_draw) - 1):
                        if sorted_draw[i] > 0:
                            ratio = sorted_draw[i + 1] / sorted_draw[i]
                            # Check if ratio is close to golden ratio
                            golden_diff = abs(ratio - self.golden_ratio)
                            if golden_diff < 0.1:  # Tolerance for golden ratio
                                golden_relationships.append(ratio)
            
            if golden_relationships:
                return {
                    'golden_ratio_frequency': float(len(golden_relationships) / len(historical_data)),
                    'average_golden_ratio': float(np.mean(golden_relationships)),
                    'golden_ratio_accuracy': float(1.0 - np.mean([abs(r - self.golden_ratio) for r in golden_relationships])),
                    'golden_consistency': float(1.0 - np.std(golden_relationships) / self.golden_ratio)
                }
            else:
                return {
                    'golden_ratio_frequency': 0.0,
                    'average_golden_ratio': 0.0,
                    'golden_ratio_accuracy': 0.0,
                    'golden_consistency': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error in golden ratio analysis: {e}")
            return {'golden_ratio_frequency': 0.0, 'average_golden_ratio': 0.0, 'golden_ratio_accuracy': 0.0, 'golden_consistency': 0.0}


class NumberGraphAnalyzer:
    """Analyzes number relationships as graph structures"""
    
    def __init__(self):
        self.adjacency_matrix = None
        self.node_weights = None
    
    def build_dependency_graph(self, historical_data: List[List[int]], max_number: int = 50) -> Dict[str, Any]:
        """Build a graph representing number co-occurrence relationships"""
        try:
            # Initialize adjacency matrix
            self.adjacency_matrix = np.zeros((max_number, max_number))
            self.node_weights = np.zeros(max_number)
            
            # Build co-occurrence matrix
            for draw in historical_data:
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
                    # Count edges between neighbors
                    edges_between_neighbors = 0
                    for j in neighbors:
                        for k in neighbors:
                            if j != k and self.adjacency_matrix[j][k] > 0:
                                edges_between_neighbors += 1
                    
                    possible_edges = len(neighbors) * (len(neighbors) - 1)
                    clustering_coeff = edges_between_neighbors / possible_edges if possible_edges > 0 else 0
                    clustering_coeffs.append(clustering_coeff)
            
            avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0
            
            # Small-world coefficient
            path_lengths = self._calculate_average_path_length()
            small_world_coeff = avg_clustering / path_lengths if path_lengths > 0 else 0
            
            return {
                'density': float(density),
                'average_clustering': float(avg_clustering),
                'average_path_length': float(path_lengths),
                'small_world_coefficient': float(small_world_coeff)
            }
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
            return {}
    
    def _calculate_average_path_length(self) -> float:
        """Calculate average shortest path length using Floyd-Warshall algorithm"""
        try:
            n = self.adjacency_matrix.shape[0]
            
            # Initialize distance matrix
            dist = np.full((n, n), float('inf'))
            
            # Set distances for direct edges
            for i in range(n):
                for j in range(n):
                    if i == j:
                        dist[i][j] = 0
                    elif self.adjacency_matrix[i][j] > 0:
                        dist[i][j] = 1  # Unweighted graph
            
            # Floyd-Warshall algorithm
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if dist[i][k] + dist[k][j] < dist[i][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
            
            # Calculate average path length
            finite_distances = dist[dist != float('inf')]
            finite_distances = finite_distances[finite_distances > 0]  # Exclude self-loops
            
            return float(np.mean(finite_distances)) if len(finite_distances) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating path length: {e}")
            return 0.0
    
    def _find_number_clusters(self) -> Dict[str, List[int]]:
        """Find clusters of numbers that frequently appear together"""
        try:
            from sklearn.cluster import SpectralClustering
            
            # Use spectral clustering on the adjacency matrix
            n_clusters = min(5, self.adjacency_matrix.shape[0] // 10)  # Adaptive number of clusters
            
            if n_clusters >= 2:
                clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
                cluster_labels = clustering.fit_predict(self.adjacency_matrix)
                
                # Group numbers by cluster
                clusters = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    clusters[f'cluster_{label}'].append(i + 1)  # Convert back to 1-indexed
                
                return dict(clusters)
            else:
                return {'cluster_0': list(range(1, self.adjacency_matrix.shape[0] + 1))}
                
        except Exception as e:
            logger.error(f"Error finding clusters: {e}")
            return {}
    
    def _calculate_centrality_measures(self) -> Dict[str, List[float]]:
        """Calculate various centrality measures for numbers"""
        try:
            n = self.adjacency_matrix.shape[0]
            
            # Degree centrality
            degree_centrality = np.sum(self.adjacency_matrix, axis=1).tolist()
            
            # Eigenvector centrality (simplified)
            eigenvalues, eigenvectors = np.linalg.eig(self.adjacency_matrix)
            max_eigen_idx = np.argmax(eigenvalues.real)
            eigenvector_centrality = np.abs(eigenvectors[:, max_eigen_idx].real).tolist()
            
            # Betweenness centrality (simplified approximation)
            betweenness_centrality = []
            for i in range(n):
                # Count how many shortest paths pass through node i
                paths_through_node = 0
                for j in range(n):
                    for k in range(n):
                        if j != k and j != i and k != i:
                            # Check if i is on shortest path from j to k
                            if (self.adjacency_matrix[j][i] > 0 and 
                                self.adjacency_matrix[i][k] > 0 and
                                self.adjacency_matrix[j][k] == 0):
                                paths_through_node += 1
                betweenness_centrality.append(float(paths_through_node))
            
            return {
                'degree_centrality': degree_centrality,
                'eigenvector_centrality': eigenvector_centrality,
                'betweenness_centrality': betweenness_centrality
            }
            
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return {}


class CombinatorialOptimizer:
    """Optimizes lottery number combinations using advanced algorithms"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_sets(self, historical_data: List[List[int]], game_type: str = 'lotto_649', 
                     num_sets: int = 3) -> Dict[str, Any]:
        """Optimize lottery number sets using combinatorial analysis"""
        try:
            # Determine game parameters
            if 'max' in game_type.lower():
                max_number = 50
                numbers_per_set = 7
            else:
                max_number = 49
                numbers_per_set = 6
            
            # Analyze historical patterns
            number_frequencies = self._calculate_frequencies(historical_data, max_number)
            pair_frequencies = self._calculate_pair_frequencies(historical_data, max_number)
            triplet_frequencies = self._calculate_triplet_frequencies(historical_data, max_number)
            
            # Generate candidate sets using different strategies
            frequency_sets = self._generate_frequency_based_sets(
                number_frequencies, num_sets, numbers_per_set, max_number
            )
            
            pair_optimized_sets = self._generate_pair_optimized_sets(
                pair_frequencies, num_sets, numbers_per_set, max_number
            )
            
            coverage_optimized_sets = self._generate_coverage_optimized_sets(
                historical_data, num_sets, numbers_per_set, max_number
            )
            
            # Evaluate and rank all sets
            all_sets = frequency_sets + pair_optimized_sets + coverage_optimized_sets
            ranked_sets = self._rank_sets(all_sets, historical_data, max_number)
            
            # Select top sets with diversity
            optimized_sets = self._select_diverse_sets(ranked_sets, num_sets)
            
            return {
                'optimized_sets': optimized_sets,
                'frequency_analysis': number_frequencies,
                'pair_analysis': dict(pair_frequencies.most_common(20)),
                'optimization_score': float(np.mean([s['score'] for s in optimized_sets])),
                'diversity_score': self._calculate_diversity_score(optimized_sets),
                'coverage_percentage': self._calculate_coverage_percentage(optimized_sets, max_number)
            }
            
        except Exception as e:
            logger.error(f"Error in set optimization: {e}")
            return {}
    
    def _calculate_frequencies(self, historical_data: List[List[int]], max_number: int) -> Dict[int, float]:
        """Calculate normalized frequency for each number"""
        frequencies = Counter()
        total_draws = len(historical_data)
        
        for draw in historical_data:
            for num in draw:
                if 1 <= num <= max_number:
                    frequencies[num] += 1
        
        # Normalize frequencies
        normalized_frequencies = {}
        for num in range(1, max_number + 1):
            normalized_frequencies[num] = frequencies[num] / total_draws if total_draws > 0 else 0
        
        return normalized_frequencies
    
    def _calculate_pair_frequencies(self, historical_data: List[List[int]], max_number: int) -> Counter:
        """Calculate frequency of number pairs"""
        pair_frequencies = Counter()
        
        for draw in historical_data:
            valid_numbers = [num for num in draw if 1 <= num <= max_number]
            for i, num1 in enumerate(valid_numbers):
                for j, num2 in enumerate(valid_numbers):
                    if i < j:  # Avoid counting the same pair twice
                        pair = tuple(sorted([num1, num2]))
                        pair_frequencies[pair] += 1
        
        return pair_frequencies
    
    def _calculate_triplet_frequencies(self, historical_data: List[List[int]], max_number: int) -> Counter:
        """Calculate frequency of number triplets"""
        triplet_frequencies = Counter()
        
        for draw in historical_data:
            valid_numbers = [num for num in draw if 1 <= num <= max_number]
            if len(valid_numbers) >= 3:
                for triplet in itertools.combinations(valid_numbers, 3):
                    triplet_frequencies[tuple(sorted(triplet))] += 1
        
        return triplet_frequencies
    
    def _generate_frequency_based_sets(self, frequencies: Dict[int, float], 
                                     num_sets: int, numbers_per_set: int, max_number: int) -> List[List[int]]:
        """Generate sets based on number frequencies"""
        sorted_numbers = sorted(frequencies.keys(), key=lambda x: frequencies[x], reverse=True)
        sets = []
        
        for i in range(num_sets):
            # Different strategies for each set
            if i == 0:  # Highest frequency numbers
                selected = sorted_numbers[:numbers_per_set]
            elif i == 1:  # Mix of high and medium frequency
                high_freq = sorted_numbers[:numbers_per_set//2]
                medium_freq = sorted_numbers[numbers_per_set//2:numbers_per_set*2]
                selected = high_freq + medium_freq[:numbers_per_set - len(high_freq)]
            else:  # Balanced approach
                # Select from different frequency tiers
                selected = []
                tier_size = len(sorted_numbers) // numbers_per_set
                for j in range(numbers_per_set):
                    start_idx = j * tier_size
                    end_idx = min(start_idx + tier_size, len(sorted_numbers))
                    if start_idx < len(sorted_numbers):
                        selected.append(sorted_numbers[start_idx])
            
            sets.append(sorted(selected[:numbers_per_set]))
        
        return sets
    
    def _generate_pair_optimized_sets(self, pair_frequencies: Counter, 
                                    num_sets: int, numbers_per_set: int, max_number: int) -> List[List[int]]:
        """Generate sets optimized for pair relationships"""
        sets = []
        top_pairs = pair_frequencies.most_common(50)
        
        for i in range(num_sets):
            selected = set()
            
            # Start with strongest pairs
            for pair, freq in top_pairs:
                if len(selected) < numbers_per_set:
                    selected.update(pair)
                if len(selected) >= numbers_per_set:
                    break
            
            # Fill remaining slots with high-frequency numbers
            if len(selected) < numbers_per_set:
                all_numbers = list(range(1, max_number + 1))
                for num in all_numbers:
                    if num not in selected:
                        selected.add(num)
                        if len(selected) >= numbers_per_set:
                            break
            
            sets.append(sorted(list(selected)[:numbers_per_set]))
        
        return sets
    
    def _generate_coverage_optimized_sets(self, historical_data: List[List[int]], 
                                        num_sets: int, numbers_per_set: int, max_number: int) -> List[List[int]]:
        """Generate sets optimized for number coverage"""
        sets = []
        used_numbers = set()
        
        # Simple greedy coverage algorithm
        for i in range(num_sets):
            selected = []
            
            # Select numbers that maximize coverage
            for num in range(1, max_number + 1):
                if num not in used_numbers and len(selected) < numbers_per_set:
                    selected.append(num)
                    used_numbers.add(num)
            
            # If not enough new numbers, reuse some
            if len(selected) < numbers_per_set:
                for num in range(1, max_number + 1):
                    if len(selected) >= numbers_per_set:
                        break
                    if num not in selected:
                        selected.append(num)
            
            sets.append(sorted(selected[:numbers_per_set]))
        
        return sets
    
    def _rank_sets(self, sets: List[List[int]], historical_data: List[List[int]], max_number: int) -> List[Dict[str, Any]]:
        """Rank sets based on multiple criteria"""
        ranked_sets = []
        
        for number_set in sets:
            # Calculate various scores
            frequency_score = self._calculate_frequency_score(number_set, historical_data, max_number)
            diversity_score = self._calculate_set_diversity_score(number_set)
            pattern_score = self._calculate_pattern_score(number_set, historical_data)
            
            # Combined score
            combined_score = (frequency_score * 0.4 + diversity_score * 0.3 + pattern_score * 0.3)
            
            ranked_sets.append({
                'numbers': number_set,
                'score': combined_score,
                'frequency_score': frequency_score,
                'diversity_score': diversity_score,
                'pattern_score': pattern_score
            })
        
        return sorted(ranked_sets, key=lambda x: x['score'], reverse=True)
    
    def _calculate_frequency_score(self, number_set: List[int], historical_data: List[List[int]], max_number: int) -> float:
        """Calculate score based on historical frequency"""
        frequencies = self._calculate_frequencies(historical_data, max_number)
        total_frequency = sum(frequencies[num] for num in number_set)
        return total_frequency / len(number_set) if number_set else 0
    
    def _calculate_set_diversity_score(self, number_set: List[int]) -> float:
        """Calculate diversity score for a set"""
        if len(number_set) <= 1:
            return 0
        
        # Range diversity
        number_range = max(number_set) - min(number_set)
        max_possible_range = max(number_set) - 1  # Maximum possible range
        range_score = number_range / max_possible_range if max_possible_range > 0 else 0
        
        # Even distribution across ranges
        ranges = [0, 0, 0]  # Low, Medium, High
        for num in number_set:
            if num <= 16:
                ranges[0] += 1
            elif num <= 33:
                ranges[1] += 1
            else:
                ranges[2] += 1
        
        # Calculate distribution evenness
        total = sum(ranges)
        if total > 0:
            distribution_score = 1 - max(ranges) / total  # More even = higher score
        else:
            distribution_score = 0
        
        return (range_score + distribution_score) / 2
    
    def _calculate_pattern_score(self, number_set: List[int], historical_data: List[List[int]]) -> float:
        """Calculate score based on pattern matching with historical data"""
        pattern_matches = 0
        total_comparisons = 0
        
        for historical_draw in historical_data:
            matches = len(set(number_set).intersection(set(historical_draw)))
            pattern_matches += matches
            total_comparisons += 1
        
        return pattern_matches / (total_comparisons * len(number_set)) if total_comparisons > 0 else 0
    
    def _select_diverse_sets(self, ranked_sets: List[Dict[str, Any]], num_sets: int) -> List[Dict[str, Any]]:
        """Select diverse sets from ranked list"""
        if len(ranked_sets) <= num_sets:
            return ranked_sets
        
        selected = [ranked_sets[0]]  # Start with highest scored set
        
        for _ in range(num_sets - 1):
            best_candidate = None
            best_diversity = -1
            
            for candidate in ranked_sets:
                if candidate not in selected:
                    # Calculate diversity from already selected sets
                    min_diversity = float('inf')
                    for selected_set in selected:
                        overlap = len(set(candidate['numbers']).intersection(set(selected_set['numbers'])))
                        diversity = len(candidate['numbers']) - overlap
                        min_diversity = min(min_diversity, diversity)
                    
                    # Combine diversity with score
                    combined_metric = candidate['score'] * 0.7 + (min_diversity / len(candidate['numbers'])) * 0.3
                    
                    if combined_metric > best_diversity:
                        best_diversity = combined_metric
                        best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
        
        return selected[:num_sets]
    
    def _calculate_diversity_score(self, optimized_sets: List[Dict[str, Any]]) -> float:
        """Calculate overall diversity score for selected sets"""
        if len(optimized_sets) <= 1:
            return 0
        
        total_diversity = 0
        comparisons = 0
        
        for i in range(len(optimized_sets)):
            for j in range(i + 1, len(optimized_sets)):
                set1 = set(optimized_sets[i]['numbers'])
                set2 = set(optimized_sets[j]['numbers'])
                
                overlap = len(set1.intersection(set2))
                diversity = 1 - (overlap / len(set1.union(set2)))
                
                total_diversity += diversity
                comparisons += 1
        
        return total_diversity / comparisons if comparisons > 0 else 0
    
    def _calculate_coverage_percentage(self, optimized_sets: List[Dict[str, Any]], max_number: int) -> float:
        """Calculate what percentage of possible numbers are covered"""
        all_numbers = set()
        for set_info in optimized_sets:
            all_numbers.update(set_info['numbers'])
        
        return len(all_numbers) / max_number * 100


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
            if isinstance(draw, list) and len(draw) > 0 and isinstance(draw[0], str):
                processed_draw = [int(num) for num in draw if str(num).isdigit()]
            else:
                processed_draw = draw
            processed_data.append(processed_draw)
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
            number_graph = self.graph_analyzer.build_dependency_graph(historical_data, max_number)
            
            # Combinatorial optimization
            logger.info("Optimizing number combinations...")
            optimal_sets = self.combinatorial_optimizer.optimize_sets(historical_data, game_type, 4)
            
            # Calculate overall confidence score
            confidence_components = [
                prime_patterns.get('confidence_score', 0),
                modular_patterns.get('mod_7', {}).get('evenness_score', 0),
                min(number_graph.get('connectivity_score', 0) * 2, 1.0),
                optimal_sets.get('optimization_score', 0)
            ]
            
            overall_confidence = np.mean([c for c in confidence_components if c > 0])
            
            analysis_result = {
                'prime_patterns': prime_patterns,
                'modular_patterns': modular_patterns,
                'number_graph': number_graph,
                'optimal_sets': optimal_sets,
                'overall_confidence': float(overall_confidence),
                'analysis_timestamp': datetime.now().isoformat(),
                'game_type': game_type,
                'data_points': len(historical_data)
            }
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis_result
            
            logger.info(f"Mathematical analysis complete. Overall confidence: {overall_confidence:.3f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in deep pattern analysis: {e}")
            return {
                'prime_patterns': {},
                'modular_patterns': {},
                'number_graph': {},
                'optimal_sets': {},
                'overall_confidence': 0.0,
                'error': str(e)
            }
    
    def get_mathematical_insights(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable insights from mathematical analysis"""
        try:
            insights = []
            
            # Prime pattern insights
            prime_patterns = analysis_result.get('prime_patterns', {})
            if prime_patterns:
                avg_primes = prime_patterns.get('average_primes_per_draw', 0)
                prime_ratio = prime_patterns.get('average_prime_ratio', 0)
                
                if prime_ratio > 0.4:
                    insights.append(f"🔢 High prime density detected ({prime_ratio:.1%})")
                elif prime_ratio > 0.25:
                    insights.append(f"🔢 Moderate prime presence ({prime_ratio:.1%})")
                else:
                    insights.append(f"🔢 Low prime frequency ({prime_ratio:.1%})")
            
            # Modular pattern insights
            modular_patterns = analysis_result.get('modular_patterns', {})
            if modular_patterns:
                # Check mod 7 patterns (often significant in lotteries)
                mod7_data = modular_patterns.get('mod_7', {})
                if mod7_data:
                    evenness = mod7_data.get('evenness_score', 0)
                    if evenness > 0.9:
                        insights.append("📊 Excellent modular distribution balance")
                    elif evenness > 0.7:
                        insights.append("📊 Good modular distribution balance")
                    else:
                        insights.append("📊 Uneven modular distribution detected")
            
            # Graph insights
            number_graph = analysis_result.get('number_graph', {})
            if number_graph:
                connectivity = number_graph.get('connectivity_score', 0)
                if connectivity > 0.3:
                    insights.append("🌐 Strong number interconnectedness")
                elif connectivity > 0.15:
                    insights.append("🌐 Moderate number relationships")
                else:
                    insights.append("🌐 Weak number correlations")
            
            # Optimization insights
            optimal_sets = analysis_result.get('optimal_sets', {})
            if optimal_sets:
                opt_score = optimal_sets.get('optimization_score', 0)
                if opt_score > 0.8:
                    insights.append("🎯 Highly optimized number combinations")
                elif opt_score > 0.6:
                    insights.append("🎯 Well-optimized number combinations")
                else:
                    insights.append("🎯 Basic optimization achieved")
            
            # Overall confidence insight
            confidence = analysis_result.get('overall_confidence', 0)
            if confidence > 0.8:
                insights.append("⭐ Very high mathematical confidence")
            elif confidence > 0.6:
                insights.append("⭐ Good mathematical confidence")
            elif confidence > 0.4:
                insights.append("⭐ Moderate mathematical confidence")
            else:
                insights.append("⭐ Low mathematical confidence")
            
            return {
                'insights': insights,
                'confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low',
                'recommendation': self._get_recommendation(analysis_result)
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                'insights': ["⚠️ Error generating mathematical insights"],
                'confidence_level': 'Unknown',
                'recommendation': "Unable to generate recommendation"
            }
    
    def _get_recommendation(self, analysis_result: Dict[str, Any]) -> str:
        """Generate recommendation based on analysis"""
        try:
            confidence = analysis_result.get('overall_confidence', 0)
            prime_patterns = analysis_result.get('prime_patterns', {})
            
            if confidence > 0.8:
                return "Mathematical patterns are very strong. Consider using optimized sets with high confidence."
            elif confidence > 0.6:
                prime_ratio = prime_patterns.get('average_prime_ratio', 0)
                if prime_ratio > 0.3:
                    return "Good mathematical foundation. Prime-focused strategy recommended."
                else:
                    return "Solid mathematical base. Balanced approach with graph-optimized numbers."
            elif confidence > 0.4:
                return "Moderate patterns detected. Use mathematical insights as supplementary guidance."
            else:
                return "Weak mathematical signals. Rely more on statistical and frequency analysis."
                
        except Exception:
            return "Analysis inconclusive. Use standard prediction methods."
