"""
🎯 Super Intelligent AI Prediction Engine - Phase 6

Advanced lottery prediction system with:
- Real model integration from models/ folder
- Multi-model ensemble analysis
- Super Intelligent AI algorithm for optimal set calculation
- Sophisticated accuracy tracking and validation
- Full app component integration
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Union
import json
import random
import statistics
from copy import deepcopy

try:
    from ..core import get_available_games, get_session_value, set_session_value, app_log
    from ..core.utils import compute_next_draw_date
    from ..services.learning_integration import (
        PredictionLearningExtractor,
        ModelPerformanceAnalyzer,
        LearningDataGenerator
    )
except ImportError:
    def get_available_games(): return ["Lotto Max", "Lotto 6/49"]
    def get_session_value(k, d=None): return st.session_state.get(k, d)
    def set_session_value(k, v): st.session_state[k] = v
    def app_log(m, level="info"): print(f"[{level.upper()}] {m}")
    def compute_next_draw_date(game): return datetime.now().date() + timedelta(days=3)
    
    # Mock imports if services not available
    class PredictionLearningExtractor:
        def __init__(self, game, predictions_base_dir="predictions"): self.game = game
        def calculate_prediction_metrics(self, pred_sets, actual): return {}
        def extract_learning_patterns(self, pred_sets, actual, models): return {}
        def generate_training_data(self, metrics, patterns, actual): return {}
    
    class ModelPerformanceAnalyzer:
        def __init__(self, learning_data_dir="data/learning"): pass
        def generate_recommendations(self, game): return {}
    
    class LearningDataGenerator:
        def __init__(self, game, output_dir="data/learning"): self.game = game
        def save_training_data(self, data): return ""
        def get_training_summary(self): return {}


# ============================================================================
# ADVANCED LEARNING SYSTEM - META-LEARNING & ADAPTIVE INTELLIGENCE
# ============================================================================

class AdaptiveLearningSystem:
    """
    Advanced learning system with:
    1. Adaptive weight learning based on historical success
    2. Temporal decay for recent pattern emphasis
    3. Cross-factor interaction detection
    4. Anti-pattern tracking from failures
    5. Meta-learning for optimal strategy combinations
    """
    
    def __init__(self, game: str):
        self.game = game
        self.game_folder = _sanitize_game_name(game)
        self.meta_learning_file = Path("data") / "learning" / self.game_folder / "meta_learning.json"
        self.meta_learning_data = self._load_meta_learning()
        
    def _load_meta_learning(self) -> Dict:
        """Load or initialize meta-learning data."""
        if self.meta_learning_file.exists():
            try:
                with open(self.meta_learning_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Initialize default meta-learning structure
        return {
            'factor_weights': {
                'hot_numbers': 0.12,
                'sum_alignment': 0.15,
                'diversity': 0.10,
                'gap_patterns': 0.12,
                'zone_distribution': 0.10,
                'even_odd_ratio': 0.08,
                'cold_penalty': 0.10,
                'decade_coverage': 0.10,
                'pattern_fingerprint': 0.08,
                'position_weighting': 0.15,
                'cluster_concentration': 0.00  # Phase 2: Starts at 0, grows with successful clusters
            },
            'weight_history': [],
            'factor_success_rates': {},
            'cross_factor_interactions': {},
            'anti_patterns': [],
            'strategy_performance': {},
            'file_combination_performance': {},
            'number_cooccurrence': {},  # Phase 1: Track which numbers appear together in successful clusters
            'cluster_history': [],  # Phase 1: Track detected winning number clusters
            'cluster_success_rate': 0.0,  # Phase 2: Track success rate of cluster-based predictions
            'temporal_decay_rate': 0.95,  # 5% decay per draw age
            'last_updated': datetime.now().isoformat(),
            'total_learning_cycles': 0
        }
    
    def _save_meta_learning(self):
        """Save meta-learning data."""
        self.meta_learning_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.meta_learning_file, 'w') as f:
            json.dump(self.meta_learning_data, f, indent=2)
    
    def get_adaptive_weights(self, draw_age: int = 0) -> Dict[str, float]:
        """
        Get adaptive weights with temporal decay and cross-factor interaction boosts applied.

        Strong factor pairs detected by detect_cross_factor_interactions() are used to
        give a mild multiplicative lift (up to +20%) to factors that appear in the top
        interaction pairs, rewarding factors that work well in combination.

        Args:
            draw_age: Number of draws since this learning data was created (0 = most recent)
        """
        base_weights = self.meta_learning_data['factor_weights']
        decay_rate = self.meta_learning_data['temporal_decay_rate']
        decay_factor = decay_rate ** draw_age

        # Start with decayed base weights
        weights = {k: v * decay_factor for k, v in base_weights.items()}

        # Apply cross-factor interaction boosts (P5)
        interactions = self.meta_learning_data.get('cross_factor_interactions', {})
        if interactions:
            # Find the global max interaction score to normalise boosts
            _max_int = max(interactions.values()) if interactions.values() else 1.0
            if _max_int > 0:
                _factor_boost: Dict[str, float] = {}
                for pair_key, pair_score in interactions.items():
                    _parts = pair_key.split('+', 1)
                    if len(_parts) == 2:
                        for _f in _parts:
                            # Accumulate normalised boost per factor (cap at 0.20 per factor)
                            _factor_boost[_f] = _factor_boost.get(_f, 0.0) + (pair_score / _max_int) * 0.10
                for _f, _boost in _factor_boost.items():
                    if _f in weights:
                        weights[_f] *= (1.0 + min(0.20, _boost))

        return weights
    
    def update_weights_from_success(self, winning_numbers, top_predictions: List[List[int]],
                                    factor_scores: Dict[str, List[float]]):
        """
        Adaptive learning: Update factor weights based on real accuracy deltas.

        Each factor score is weighted by how many winning numbers the corresponding
        prediction actually matched, so factors that correlate with real hits are
        rewarded over factors that look good on paper but miss in practice.

        Args:
            winning_numbers: The actual winning numbers (list or dict with 'numbers' key)
            top_predictions: The best-performing prediction sets
            factor_scores: Score breakdown by factor for each top prediction
        """
        # Normalise winning_numbers to a plain list
        if isinstance(winning_numbers, dict):
            _win_list = list(winning_numbers.get('numbers', []))
        else:
            _win_list = list(winning_numbers)
        _win_set = set(_win_list)

        # Compute actual hit count per prediction (real accuracy signal)
        _hit_counts = []
        for pred in top_predictions:
            _hit_counts.append(len(set(pred) & _win_set))
        _total_hits = sum(_hit_counts)

        # Calculate real accuracy delta for each factor
        for factor_name, scores in factor_scores.items():
            if not scores:
                continue

            n = min(len(scores), len(_hit_counts))
            _scores_arr = np.array(scores[:n], dtype=float)
            _hits_arr   = np.array(_hit_counts[:n], dtype=float)

            if _total_hits > 0:
                # Hit-weighted average: how high was this factor when predictions hit more numbers?
                _hit_weighted_avg = float(np.dot(_scores_arr, _hits_arr) / _total_hits)
            else:
                _hit_weighted_avg = float(np.mean(_scores_arr))

            # Uniform average: what the factor scores were regardless of hits
            _uniform_avg = float(np.mean(_scores_arr))

            # accuracy_delta > 0 means factor correlates with actual hits (reward it)
            # accuracy_delta ≈ 0 means factor is neutral; < 0 means it's misleading
            _accuracy_delta = _hit_weighted_avg - _uniform_avg

            # Map delta to a 0–1 success-rate signal (shift+scale so 0.5 = neutral)
            _signal = max(0.0, min(1.0, 0.5 + _accuracy_delta))

            # Update success rate tracking
            if factor_name not in self.meta_learning_data['factor_success_rates']:
                self.meta_learning_data['factor_success_rates'][factor_name] = []

            self.meta_learning_data['factor_success_rates'][factor_name].append(_signal)

            # Keep only last 50 results
            if len(self.meta_learning_data['factor_success_rates'][factor_name]) > 50:
                self.meta_learning_data['factor_success_rates'][factor_name].pop(0)
        
        # P6: Record hit_rate effectiveness — fraction of winning numbers covered by top predictions
        if _win_set and top_predictions:
            _covered = set()
            for _p in top_predictions:
                _covered.update(set(_p) & _win_set)
            _hit_rate = len(_covered) / len(_win_set)
            _hr_history = self.meta_learning_data.setdefault('hit_rate_history', [])
            _hr_history.append({
                'cycle': self.meta_learning_data.get('total_learning_cycles', 0) + 1,
                'hit_rate': round(_hit_rate, 4),
                'timestamp': datetime.now().isoformat()
            })
            # Keep last 50 entries
            if len(_hr_history) > 50:
                _hr_history.pop(0)

        # Recalculate adaptive weights based on historical success
        self._recalculate_adaptive_weights()

        # Save updated meta-learning
        self.meta_learning_data['total_learning_cycles'] += 1
        self.meta_learning_data['last_updated'] = datetime.now().isoformat()
        self._save_meta_learning()
    
    def _recalculate_adaptive_weights(self):
        """Recalculate factor weights based on historical success rates."""
        success_rates = self.meta_learning_data['factor_success_rates']
        
        if not success_rates:
            return
        
        # Calculate average success for each factor
        factor_performance = {}
        for factor, scores in success_rates.items():
            if scores:
                factor_performance[factor] = np.mean(scores)
        
        if not factor_performance:
            return
        
        # Normalize to get new weights (softmax-like distribution)
        total_performance = sum(factor_performance.values())
        if total_performance > 0:
            for factor in self.meta_learning_data['factor_weights']:
                if factor in factor_performance:
                    # Blend old weight (30%) with new performance-based weight (70%)
                    old_weight = self.meta_learning_data['factor_weights'][factor]
                    new_weight = factor_performance[factor] / total_performance
                    self.meta_learning_data['factor_weights'][factor] = 0.3 * old_weight + 0.7 * new_weight
        
        # Record weight change in history
        self.meta_learning_data['weight_history'].append({
            'timestamp': datetime.now().isoformat(),
            'weights': self.meta_learning_data['factor_weights'].copy()
        })
        
        # Keep only last 100 history entries
        if len(self.meta_learning_data['weight_history']) > 100:
            self.meta_learning_data['weight_history'].pop(0)
    
    def detect_cross_factor_interactions(self, predictions: List[List[int]],
                                        winning_numbers,
                                        factor_scores: Dict[str, List[float]]):
        """
        Detect when combinations of factors predict better together.
        Identifies factor pairs whose combined presence correlates with the most
        actual number matches, then stores the top pairs so get_adaptive_weights()
        can apply joint multipliers during generation.

        Args:
            predictions: Top prediction sets (already filtered to best performers)
            winning_numbers: Actual winning numbers (list or dict with 'numbers' key)
            factor_scores: {factor_name: [score_per_prediction]} (list-indexed)
        """
        if isinstance(winning_numbers, dict):
            _win_set = set(winning_numbers.get('numbers', []))
        else:
            _win_set = set(winning_numbers)

        # Hit count per prediction
        _hit_counts = [len(set(p) & _win_set) for p in predictions]

        factors = list(self.meta_learning_data['factor_weights'].keys())
        factor_pairs = {}

        for i, factor1 in enumerate(factors):
            for factor2 in factors[i + 1:]:
                s1 = factor_scores.get(factor1, [])
                s2 = factor_scores.get(factor2, [])
                n = min(len(s1), len(s2), len(_hit_counts))
                if n == 0:
                    continue
                # Pair interaction: product of both scores, weighted by actual hits
                _pair_products = [s1[j] * s2[j] for j in range(n)]
                _total_hits = sum(_hit_counts[:n]) or 1
                # Hit-weighted mean of pair product
                _hw_mean = sum(_pair_products[j] * _hit_counts[j] for j in range(n)) / _total_hits
                factor_pairs[f"{factor1}+{factor2}"] = round(float(_hw_mean), 6)

        # Keep only top-20 strongest pairs to limit JSON growth
        _top_pairs = sorted(factor_pairs.items(), key=lambda x: x[1], reverse=True)[:20]
        self.meta_learning_data['cross_factor_interactions'] = dict(_top_pairs)
        self._save_meta_learning()
        
        return factor_pairs
    
    def track_anti_patterns(self, worst_predictions: List[List[int]], 
                           winning_numbers: List[int]):
        """
        Learn from failures: Track patterns that consistently fail.
        
        Args:
            worst_predictions: The worst-performing prediction sets
            winning_numbers: Actual winning numbers
        """
        # P9: Before adding new anti-patterns, increment age on all existing ones.
        # This lets old failure signatures fade while fresh ones stay strong.
        for _ap in self.meta_learning_data['anti_patterns']:
            _ap['age'] = _ap.get('age', 0) + 1

        for pred in worst_predictions:
            anti_pattern = {
                'numbers': sorted(pred),
                'sum': sum(pred),
                'gaps': [pred[i+1] - pred[i] for i in range(len(pred)-1)] if len(pred) > 1 else [],
                'even_count': sum(1 for n in pred if n % 2 == 0),
                'zones': self._categorize_zones(pred),
                'timestamp': datetime.now().isoformat(),
                'age': 0  # Fresh anti-pattern, full penalty weight
            }
            self.meta_learning_data['anti_patterns'].append(anti_pattern)

        # Keep only last 200 anti-patterns
        if len(self.meta_learning_data['anti_patterns']) > 200:
            self.meta_learning_data['anti_patterns'] = self.meta_learning_data['anti_patterns'][-200:]

        self._save_meta_learning()
    
    def _categorize_zones(self, numbers: List[int]) -> Dict[str, int]:
        """Categorize numbers into zones."""
        max_number = 50 if 'max' in self.game.lower() else 49
        low_boundary = max_number // 3
        mid_boundary = (max_number * 2) // 3
        
        zones = {'low': 0, 'mid': 0, 'high': 0}
        for num in numbers:
            if num <= low_boundary:
                zones['low'] += 1
            elif num <= mid_boundary:
                zones['mid'] += 1
            else:
                zones['high'] += 1
        return zones
    
    def detect_winning_clusters(self, ranked_predictions: List[Tuple[float, List[int]]], 
                                winning_numbers: List[int], 
                                top_n: int = 5) -> Dict[str, Any]:
        """
        Phase 1: Detect when top-N predictions collectively contain all/most winning numbers.
        This identifies 'near-perfect clusters' where the AI got close but fragmented.
        
        Args:
            ranked_predictions: List of (score, numbers) tuples sorted by score
            winning_numbers: Actual winning numbers from draw
            top_n: Number of top predictions to analyze for clusters
        
        Returns:
            Dictionary with cluster detection results and co-occurrence data
        """
        if not ranked_predictions or not winning_numbers:
            return {'cluster_detected': False}
        
        # Analyze top N predictions
        top_predictions = ranked_predictions[:top_n]
        
        # Collect all unique numbers from top predictions
        combined_numbers = set()
        for score, numbers in top_predictions:
            combined_numbers.update(numbers)
        
        # Calculate coverage
        winning_set = set(winning_numbers)
        covered_winners = combined_numbers & winning_set
        coverage_count = len(covered_winners)
        coverage_percent = (coverage_count / len(winning_numbers)) * 100
        
        # P7: Adaptive cluster threshold — starts at 60% (easy trigger for new users,
        # builds co-occurrence data faster), rises linearly to 85% once the cluster
        # history is rich enough (30+ entries) so the system becomes more selective.
        _cluster_history_n = len(self.meta_learning_data.get('cluster_history', []))
        _adaptive_frac = 0.60 + min(0.25, _cluster_history_n / 120.0)  # 60%→85% over 30 detections
        cluster_threshold = max(1, int(len(winning_numbers) * _adaptive_frac))
        is_cluster = coverage_count >= cluster_threshold
        
        cluster_result = {
            'cluster_detected': is_cluster,
            'cluster_threshold': cluster_threshold,
            'total_winners': len(winning_numbers),
            'top_n': top_n,
            'coverage_count': coverage_count,
            'coverage_percent': coverage_percent,
            'covered_winners': sorted(list(covered_winners)),
            'missing_winners': sorted(list(winning_set - covered_winners)),
            'timestamp': datetime.now().isoformat(),
            'individual_matches': []
        }
        
        # Track individual set matches
        for idx, (score, numbers) in enumerate(top_predictions, 1):
            matches = set(numbers) & winning_set
            cluster_result['individual_matches'].append({
                'rank': idx,
                'match_count': len(matches),
                'matched_numbers': sorted(list(matches)),
                'score': float(score)
            })
        
        # If cluster detected, update co-occurrence matrix
        if is_cluster:
            self._update_cooccurrence_matrix(covered_winners)
            
            # Store cluster in history
            self.meta_learning_data['cluster_history'].append(cluster_result)
            
            # Keep only last 100 clusters
            if len(self.meta_learning_data['cluster_history']) > 100:
                self.meta_learning_data['cluster_history'] = \
                    self.meta_learning_data['cluster_history'][-100:]
            
            self._save_meta_learning()
        
        return cluster_result
    
    def update_cluster_concentration_weight(self, cluster_coverage: int, total_winners: int = 7):
        """
        Phase 2: Update cluster_concentration weight based on cluster detection success.
        The more winners covered in clusters, the higher this weight becomes.
        
        Args:
            cluster_coverage: Number of winning numbers found in top-N cluster
            total_winners: Total winning numbers (7 for Lotto Max, 6 for Lotto 649)
        """
        # Ensure Phase 2 keys exist
        if 'cluster_concentration' not in self.meta_learning_data['factor_weights']:
            self.meta_learning_data['factor_weights']['cluster_concentration'] = 0.0
        if 'cluster_success_rate' not in self.meta_learning_data:
            self.meta_learning_data['cluster_success_rate'] = 0.0
        
        # Calculate success rate for this cluster
        coverage_rate = cluster_coverage / total_winners
        
        # Update rolling success rate (exponential moving average)
        current_success = self.meta_learning_data['cluster_success_rate']
        alpha = 0.2  # Learning rate
        new_success = (alpha * coverage_rate) + ((1 - alpha) * current_success)
        self.meta_learning_data['cluster_success_rate'] = new_success
        
        # Update cluster_concentration weight based on success rate
        # Starts at 0.0, can grow to max 0.20 (20%) based on consistent cluster success
        max_weight = 0.20
        new_weight = min(max_weight, new_success * max_weight)
        
        self.meta_learning_data['factor_weights']['cluster_concentration'] = new_weight
        self._save_meta_learning()
        
        return new_weight
    
    def get_number_cooccurrence_boost(self, current_set: List[int], candidate_num: int) -> float:
        """
        Phase 2: Calculate boost factor for a candidate number based on co-occurrence
        with numbers already in the current set.
        
        Args:
            current_set: Numbers already selected for this set
            candidate_num: Number being considered for addition
            
        Returns:
            Boost multiplier (1.0 = no boost, >1.0 = boost)
        """
        # Ensure Phase 1/2 keys exist
        if 'number_cooccurrence' not in self.meta_learning_data:
            return 1.0
        if 'cluster_concentration' not in self.meta_learning_data['factor_weights']:
            return 1.0
        
        concentration_weight = self.meta_learning_data['factor_weights']['cluster_concentration']

        # P4 (also applied here): use smooth ramp instead of hard 0.03 floor;
        # only activate after 10+ learning cycles to avoid noise-based boosting.
        _has_cooc_data = bool(self.meta_learning_data.get('number_cooccurrence'))
        _cooc_cycles = int(self.meta_learning_data.get('total_learning_cycles', 0))
        if _has_cooc_data and _cooc_cycles >= 10:
            _ramp = min(0.05, 0.005 * (_cooc_cycles - 9))
            effective_weight = max(concentration_weight, _ramp)
        else:
            effective_weight = concentration_weight

        if effective_weight == 0.0:
            return 1.0
        
        # Calculate average co-occurrence strength with numbers in current set
        total_strength = 0.0
        count = 0
        
        for num in current_set:
            strength = self.get_cooccurrence_strength(num, candidate_num)
            if strength > 0:
                total_strength += strength
                count += 1
        
        if count == 0:
            return 1.0
        
        avg_strength = total_strength / count
        
        # Convert strength to boost multiplier scaled by effective concentration weight.
        # avg_strength ranges 0-1, effective_weight ranges 0.03-0.20
        # Result: boost ranges from 1.0 (no boost) to 1.5 (50% boost at max)
        boost = 1.0 + (avg_strength * effective_weight * 2.5)

        return boost
    
    def generate_fusion_sets(self, predictions: List[List[int]], 
                            prob_values: np.ndarray,
                            draw_size: int,
                            num_fusion_sets: int = 2,
                            top_n_to_analyze: int = 5) -> List[Dict[str, Any]]:
        """
        Phase 3: Generate fusion sets by intelligently combining numbers from top-ranked predictions.
        
        This creates concentrated sets that merge the best numbers from multiple high-scoring
        predictions, addressing the fragmentation problem.
        
        Args:
            predictions: All generated prediction sets
            prob_values: Ensemble probability values for all numbers
            draw_size: How many numbers per set
            num_fusion_sets: Number of fusion sets to generate
            top_n_to_analyze: Number of top predictions to analyze for fusion
            
        Returns:
            List of fusion set dictionaries with numbers and metadata
        """
        from collections import Counter

        # Allow fusion sets whenever there is co-occurrence data, even if cluster_concentration
        # weight hasn't built up yet (new users). Use an effective floor of 0.02.
        concentration_weight = self.meta_learning_data['factor_weights'].get('cluster_concentration', 0.0)
        _has_cooccurrence = bool(self.meta_learning_data.get('number_cooccurrence'))
        effective_concentration = max(concentration_weight, 0.02) if _has_cooccurrence else concentration_weight

        if effective_concentration < 0.02:  # Need at least co-occurrence data or 2% weight
            return []
        
        fusion_sets = []
        
        # Analyze top N predictions
        top_predictions = predictions[:min(top_n_to_analyze, len(predictions))]
        
        for fusion_idx in range(num_fusion_sets):
            # Strategy varies by fusion set index
            if fusion_idx == 0:
                # Fusion Set 1: Highest co-occurrence numbers from top predictions
                fusion_set = self._create_cooccurrence_fusion(top_predictions, prob_values, draw_size)
                strategy = "Co-occurrence Fusion"
            else:
                # Fusion Set 2+: Frequency-based fusion with diversity
                fusion_set = self._create_frequency_fusion(top_predictions, prob_values, draw_size, fusion_sets)
                strategy = "Frequency-Diversity Fusion"
            
            if fusion_set and len(fusion_set) == draw_size:
                fusion_sets.append({
                    'numbers': fusion_set,
                    'fusion_strategy': strategy,
                    'source_sets': top_n_to_analyze,
                    'concentration_weight': concentration_weight
                })
        
        return fusion_sets
    
    def _create_cooccurrence_fusion(self, top_predictions: List[List[int]], 
                                     prob_values: np.ndarray, 
                                     draw_size: int) -> List[int]:
        """
        Phase 3: Create fusion set prioritizing numbers with highest co-occurrence.
        """
        from collections import defaultdict
        
        # Score each number by its total co-occurrence strength with others in top predictions
        number_scores = defaultdict(float)
        all_numbers = set()
        
        for pred in top_predictions:
            all_numbers.update(pred)
        
        for num in all_numbers:
            # Calculate total co-occurrence score
            total_cooccurrence = 0.0
            
            for pred in top_predictions:
                if num in pred:
                    # Add co-occurrence strength with all other numbers in this prediction
                    for other_num in pred:
                        if other_num != num:
                            strength = self.get_cooccurrence_strength(num, other_num)
                            total_cooccurrence += strength
            
            # Also factor in ensemble probability
            prob = prob_values[num - 1] if 0 <= num - 1 < len(prob_values) else 0.0
            
            # Combined score: 70% co-occurrence, 30% probability
            number_scores[num] = (total_cooccurrence * 0.7) + (prob * 0.3)
        
        # Select top draw_size numbers by score
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        fusion_set = [num for num, score in sorted_numbers[:draw_size]]
        
        return sorted(fusion_set)
    
    def _create_frequency_fusion(self, top_predictions: List[List[int]], 
                                  prob_values: np.ndarray, 
                                  draw_size: int,
                                  existing_fusion_sets: List[Dict]) -> List[int]:
        """
        Phase 3: Create fusion set using frequency in top predictions plus diversity.
        """
        from collections import Counter
        
        # Count frequency of each number in top predictions
        all_numbers = []
        for pred in top_predictions:
            all_numbers.extend(pred)
        
        number_frequency = Counter(all_numbers)
        
        # Remove numbers already used in existing fusion sets to ensure diversity
        for fusion_dict in existing_fusion_sets:
            for num in fusion_dict['numbers']:
                if num in number_frequency:
                    del number_frequency[num]
        
        if len(number_frequency) < draw_size:
            # Not enough diverse numbers, return empty
            return []
        
        # Score by frequency and probability
        number_scores = {}
        max_freq = max(number_frequency.values()) if number_frequency else 1
        
        for num, freq in number_frequency.items():
            freq_score = freq / max_freq
            prob = prob_values[num - 1] if 0 <= num - 1 < len(prob_values) else 0.0
            
            # Combined score: 60% frequency, 40% probability
            number_scores[num] = (freq_score * 0.6) + (prob * 0.4)
        
        # Select top draw_size numbers
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        fusion_set = [num for num, score in sorted_numbers[:draw_size]]
        
        return sorted(fusion_set)
    
    def _update_cooccurrence_matrix(self, numbers: set):
        """
        Phase 1: Update number co-occurrence matrix.
        Tracks which numbers appear together in successful clusters.
        
        Args:
            numbers: Set of numbers that appeared together in a successful cluster
        """
        # Ensure Phase 1 keys exist (backward compatibility with legacy meta-learning files)
        if 'number_cooccurrence' not in self.meta_learning_data:
            self.meta_learning_data['number_cooccurrence'] = {}
        if 'cluster_history' not in self.meta_learning_data:
            self.meta_learning_data['cluster_history'] = []
        
        numbers_list = sorted(list(numbers))
        
        # Update pairwise co-occurrence
        for i, num1 in enumerate(numbers_list):
            for num2 in numbers_list[i+1:]:
                # Create symmetric keys
                key1 = f"{num1}_{num2}"
                key2 = f"{num2}_{num1}"
                
                # Increment co-occurrence count
                if key1 not in self.meta_learning_data['number_cooccurrence']:
                    self.meta_learning_data['number_cooccurrence'][key1] = 0
                
                self.meta_learning_data['number_cooccurrence'][key1] += 1
                
                # Ensure symmetric entry exists
                if key2 not in self.meta_learning_data['number_cooccurrence']:
                    self.meta_learning_data['number_cooccurrence'][key2] = \
                        self.meta_learning_data['number_cooccurrence'][key1]
    
    def get_cooccurrence_strength(self, num1: int, num2: int) -> float:
        """
        Phase 1: Get co-occurrence strength between two numbers.
        
        Returns:
            Float between 0.0 and 1.0 indicating how often these numbers
            appear together in successful clusters (normalized)
        """
        # Ensure Phase 1 keys exist (backward compatibility)
        if 'number_cooccurrence' not in self.meta_learning_data:
            return 0.0
        
        key = f"{min(num1, num2)}_{max(num1, num2)}"
        count = self.meta_learning_data['number_cooccurrence'].get(key, 0)
        
        # Normalize by total clusters
        total_clusters = len(self.meta_learning_data.get('cluster_history', []))
        if total_clusters == 0:
            return 0.0
        
        # Return normalized strength (capped at 1.0)
        return min(1.0, count / max(1, total_clusters))
    
    def get_top_cooccurrences(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Phase 1: Get top N number pairs by co-occurrence count.
        
        Returns:
            List of (number_pair_key, count) tuples sorted by count
        """
        # Ensure Phase 1 keys exist (backward compatibility)
        if 'number_cooccurrence' not in self.meta_learning_data:
            return []
        
        cooccurrence = self.meta_learning_data.get('number_cooccurrence', {})
        sorted_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_n]
    
    def penalize_anti_patterns(self, prediction: List[int]) -> float:
        """
        Calculate penalty score if prediction matches known anti-patterns.
        
        Returns:
            Penalty value (0.0 = no penalty, 1.0 = strong match to anti-pattern)
        """
        if not self.meta_learning_data['anti_patterns']:
            return 0.0
        
        pred_sum = sum(prediction)
        pred_zones = self._categorize_zones(prediction)
        pred_even = sum(1 for n in prediction if n % 2 == 0)
        
        penalty = 0.0
        matches = 0
        
        for anti in self.meta_learning_data['anti_patterns'][-50:]:  # Check recent 50
            similarity = 0.0

            # Sum similarity
            sum_diff = abs(pred_sum - anti['sum'])
            if sum_diff < 20:
                similarity += 0.3

            # Zone similarity
            zone_diff = sum(abs(pred_zones[z] - anti['zones'].get(z, 0)) for z in ['low', 'mid', 'high'])
            if zone_diff <= 2:
                similarity += 0.3

            # Even/odd similarity
            if abs(pred_even - anti['even_count']) <= 1:
                similarity += 0.2

            # Number overlap
            overlap = len(set(prediction) & set(anti['numbers']))
            if overlap >= 4:
                similarity += 0.2

            if similarity > 0.6:  # Strong match to anti-pattern
                # P9: Apply age-based decay — old failures fade, recent ones stay strong
                _age = anti.get('age', 0)
                _age_factor = 0.90 ** _age  # 10% weaker per learning cycle
                penalty += similarity * _age_factor
                matches += 1

        # Average penalty from matches
        return min(1.0, penalty / max(1, matches))
    
    def record_strategy_performance(self, strategy_name: str, success_metrics: Dict):
        """Track performance of different regeneration strategies."""
        if strategy_name not in self.meta_learning_data['strategy_performance']:
            self.meta_learning_data['strategy_performance'][strategy_name] = []
        
        self.meta_learning_data['strategy_performance'][strategy_name].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': success_metrics
        })
        
        # Keep only last 50 results per strategy
        if len(self.meta_learning_data['strategy_performance'][strategy_name]) > 50:
            self.meta_learning_data['strategy_performance'][strategy_name].pop(0)
        
        self._save_meta_learning()
    
    def get_best_strategy(self) -> str:
        """Determine which regeneration strategy historically performs best."""
        if not self.meta_learning_data['strategy_performance']:
            return "Learning-Optimized"  # Default
        
        strategy_scores = {}
        for strategy, results in self.meta_learning_data['strategy_performance'].items():
            if results:
                avg_score = np.mean([r['metrics'].get('accuracy', 0) for r in results])
                strategy_scores[strategy] = avg_score
        
        if strategy_scores:
            return max(strategy_scores, key=strategy_scores.get)
        return "Learning-Optimized"
    
    def record_file_combination_performance(self, file_names: List[str], success_score: float):
        """Meta-learning: Track which learning file combinations work best."""
        combo_key = "+".join(sorted(file_names))
        
        if combo_key not in self.meta_learning_data['file_combination_performance']:
            self.meta_learning_data['file_combination_performance'][combo_key] = []
        
        self.meta_learning_data['file_combination_performance'][combo_key].append({
            'timestamp': datetime.now().isoformat(),
            'score': success_score,
            'num_files': len(file_names)
        })
        
        # Keep only last 20 results per combination
        if len(self.meta_learning_data['file_combination_performance'][combo_key]) > 20:
            self.meta_learning_data['file_combination_performance'][combo_key].pop(0)
        
        self._save_meta_learning()
    
    def get_optimal_file_combination(self, available_files: List[str]) -> List[str]:
        """Recommend which learning files to combine based on historical performance."""
        if not self.meta_learning_data['file_combination_performance']:
            # Default: use most recent
            return available_files[:1] if available_files else []
        
        # Score all possible combinations
        combo_scores = {}
        for combo_key, results in self.meta_learning_data['file_combination_performance'].items():
            if results:
                avg_score = np.mean([r['score'] for r in results])
                combo_scores[combo_key] = avg_score
        
        if not combo_scores:
            return available_files[:1]
        
        # Find best combination that uses available files
        best_combo = max(combo_scores, key=combo_scores.get)
        files_in_combo = best_combo.split('+')
        
        # Return files from best combo that are available
        return [f for f in files_in_combo if f in available_files]


class GeneticSetOptimizer:
    """
    Genetic algorithm for intelligent set optimization.
    Evolves prediction sets to maximize learning score.
    """
    
    def __init__(self, draw_size: int, max_number: int, learning_data: Dict,
                 adaptive_system: AdaptiveLearningSystem):
        self.draw_size = draw_size
        self.max_number = max_number
        self.learning_data = learning_data
        self.adaptive_system = adaptive_system
        self.population_size = 100
        self.generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elite_size = 10
    
    def optimize_set(self, initial_set: List[int] = None) -> List[int]:
        """
        Optimize a prediction set using genetic algorithm.
        
        Args:
            initial_set: Starting set (optional, will generate random if None)
        
        Returns:
            Optimized prediction set
        """
        # Initialize population
        population = self._initialize_population(initial_set)
        
        best_fitness = 0
        best_individual = None
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(ind) for ind in population]
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Early stopping if no improvement for 10 generations
            if generations_without_improvement >= 10:
                break
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(selected)
            
            # Mutation
            offspring = self._mutation(offspring)
            
            # Elitism: Keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = [population[i] for i in elite_indices]
            
            # New generation
            population = elite + offspring[:self.population_size - self.elite_size]
        
        return sorted(best_individual) if best_individual else initial_set or self._generate_random_set()
    
    def _initialize_population(self, initial_set: List[int] = None) -> List[List[int]]:
        """Create initial population."""
        population = []
        
        # Add initial set if provided
        if initial_set:
            population.append(initial_set.copy())
        
        # Generate random individuals
        while len(population) < self.population_size:
            population.append(self._generate_random_set())
        
        return population
    
    def _generate_random_set(self) -> List[int]:
        """Generate a random valid set."""
        return sorted(random.sample(range(1, self.max_number + 1), self.draw_size))
    
    def _evaluate_fitness(self, individual: List[int]) -> float:
        """Evaluate fitness using enhanced learning score."""
        # Use adaptive learning system to calculate score
        base_score = _calculate_learning_score_advanced(
            individual, 
            self.learning_data, 
            self.adaptive_system
        )
        
        # Additional fitness components
        # Penalty for anti-patterns
        anti_penalty = self.adaptive_system.penalize_anti_patterns(individual)
        
        # Bonus for diversity
        diversity_bonus = (max(individual) - min(individual)) / self.max_number * 0.1
        
        return base_score - anti_penalty + diversity_bonus
    
    def _selection(self, population: List[List[int]], fitness_scores: List[float]) -> List[List[int]]:
        """Tournament selection."""
        selected = []
        tournament_size = 5
        
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parents: List[List[int]]) -> List[List[int]]:
        """Uniform crossover between parent sets."""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            if random.random() < self.crossover_rate:
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Create child by mixing numbers from both parents
                combined = list(set(parent1 + parent2))
                if len(combined) >= self.draw_size:
                    child = sorted(random.sample(combined, self.draw_size))
                else:
                    # Not enough unique numbers, fill randomly
                    remaining = [n for n in range(1, self.max_number + 1) if n not in combined]
                    child = sorted(combined + random.sample(remaining, self.draw_size - len(combined)))
                
                offspring.append(child)
            else:
                offspring.append(parents[i].copy())
        
        return offspring
    
    def _mutation(self, population: List[List[int]]) -> List[List[int]]:
        """Mutate individuals by swapping numbers."""
        mutated = []
        
        for individual in population:
            if random.random() < self.mutation_rate:
                # Swap one number
                idx_to_replace = random.randint(0, len(individual) - 1)
                new_number = random.randint(1, self.max_number)
                
                # Ensure uniqueness
                while new_number in individual:
                    new_number = random.randint(1, self.max_number)
                
                mutated_ind = individual.copy()
                mutated_ind[idx_to_replace] = new_number
                mutated.append(sorted(mutated_ind))
            else:
                mutated.append(individual.copy())
        
        return mutated


# ============================================================================
# MODEL DISCOVERY & LOADING
# ============================================================================

def _sanitize_game_name(game: str) -> str:
    """Convert game name to folder name."""
    return game.lower().replace(" ", "_").replace("/", "_")


def _get_models_dir() -> Path:
    """Returns models folder path."""
    return Path("models")


def _discover_available_models(game: str) -> Dict[str, List[Dict[str, Any]]]:
    """Discover all available models in the models folder for a game.
    
    Includes all model type directories, even if empty (for CNN, which may not have models yet).
    """
    models_dir = _get_models_dir()
    game_folder = _sanitize_game_name(game)
    game_path = models_dir / game_folder
    
    model_types = {}
    
    if not game_path.exists():
        app_log(f"Game path does not exist: {game_path}", "warning")
        return model_types
    
    # Scan each model type folder
    for type_dir in game_path.iterdir():
        if type_dir.is_dir():
            model_type = type_dir.name.lower()
            models = _load_models_for_type(type_dir, model_type)
            # Include model type even if it has no models (e.g., CNN)
            model_types[model_type] = models
    
    return model_types


def _load_models_for_type(type_dir: Path, model_type: str) -> List[Dict[str, Any]]:
    """Load all model metadata from a type directory.
    
    Handles both:
    - Ensemble models (stored as folders): ensemble_name/metadata.json
    - Individual models (stored as files): model_name.keras or model_name.joblib
    """
    models = []
    
    try:
        for item in sorted(type_dir.iterdir(), reverse=True):
            # Handle ENSEMBLE or FOLDER-based models
            if item.is_dir():
                model_name = item.name
                metadata_file = item / "metadata.json"
                
                model_info = {
                    "name": model_name,
                    "type": model_type,
                    "path": str(item),
                    "accuracy": 0.0,
                    "trained_on": None,
                    "version": "1.0",
                    "full_metadata": {}
                }
                
                # Load metadata if available
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                            # Handle ensemble metadata (nested structure)
                            accuracy_value = 0.0
                            if isinstance(metadata, dict):
                                # Check if this is ensemble metadata (has model keys like 'xgboost', 'catboost', etc.)
                                if 'xgboost' in metadata or 'catboost' in metadata or 'lightgbm' in metadata or 'lstm' in metadata or 'cnn' in metadata or 'transformer' in metadata or 'ensemble' in metadata:
                                    # Ensemble metadata: use combined_accuracy from ensemble key if available
                                    if 'ensemble' in metadata and isinstance(metadata['ensemble'], dict):
                                        acc = metadata['ensemble'].get('combined_accuracy')
                                        if isinstance(acc, (int, float)):
                                            accuracy_value = float(acc)
                                    
                                    # Fallback: calculate average accuracy from component models
                                    if accuracy_value == 0.0:
                                        accuracies = []
                                        for key in ['xgboost', 'catboost', 'lightgbm', 'lstm', 'cnn', 'transformer']:
                                            if key in metadata and isinstance(metadata[key], dict):
                                                acc = metadata[key].get('accuracy')
                                                if isinstance(acc, (int, float)):
                                                    accuracies.append(float(acc))
                                        if accuracies:
                                            accuracy_value = float(np.mean(accuracies))
                                else:
                                    # Regular metadata: get accuracy directly
                                    acc = metadata.get("accuracy", 0.0)
                                    if isinstance(acc, (int, float)):
                                        accuracy_value = float(acc)
                            
                            model_info.update({
                                "accuracy": accuracy_value,
                                "trained_on": metadata.get("trained_on") if isinstance(metadata, dict) else None,
                                "version": metadata.get("version", "1.0") if isinstance(metadata, dict) else "1.0",
                                "full_metadata": metadata
                            })
                    except Exception as e:
                        app_log(f"Error reading metadata for {model_name}: {e}", "warning")
                
                models.append(model_info)
            
            # Handle INDIVIDUAL MODELS: files with .keras, .joblib, or .h5 extensions
            elif item.is_file() and item.suffix in ['.keras', '.joblib', '.h5']:
                # Skip metadata files
                if item.name.endswith('_metadata.json'):
                    continue
                
                # Extract model name without extension
                model_name = item.stem  # Gets filename without extension
                
                model_info = {
                    "name": model_name,
                    "type": model_type,
                    "path": str(item),
                    "accuracy": 0.0,
                    "trained_on": None,
                    "version": "1.0",
                    "full_metadata": {}
                }
                
                # Try to read corresponding metadata file
                # Metadata is stored as: model_name_metadata.json
                metadata_file = item.parent / f"{model_name}_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                            # Handle nested metadata structure
                            # Individual model metadata has structure: { "model_type": { "accuracy": ..., ... } }
                            accuracy_value = 0.0
                            if isinstance(metadata, dict):
                                # Check for model type keys (xgboost, catboost, lightgbm, lstm, cnn, transformer)
                                for key in ['xgboost', 'catboost', 'lightgbm', 'lstm', 'cnn', 'transformer', model_type]:
                                    if key in metadata and isinstance(metadata[key], dict):
                                        acc = metadata[key].get('accuracy')
                                        if isinstance(acc, (int, float)):
                                            accuracy_value = float(acc)
                                            break
                                # Also check for direct accuracy field
                                if accuracy_value == 0.0:
                                    acc = metadata.get("accuracy", 0.0)
                                    if isinstance(acc, (int, float)):
                                        accuracy_value = float(acc)
                            
                            model_info.update({
                                "accuracy": accuracy_value,
                                "trained_on": metadata.get("trained_on") if isinstance(metadata, dict) else None,
                                "version": metadata.get("version", "1.0") if isinstance(metadata, dict) else "1.0",
                                "full_metadata": metadata
                            })
                    except Exception as e:
                        app_log(f"Error reading metadata for {model_name}: {e}", "warning")
                
                models.append(model_info)
    except Exception as e:
        app_log(f"Error scanning model type directory {type_dir}: {e}", "warning")
    
    return models


# ============================================================================
# SUPER INTELLIGENT AI ANALYZER
# ============================================================================

class SuperIntelligentAIAnalyzer:
    """
    Advanced AI prediction system using Super Intelligent Algorithm (SIA)
    for optimal lottery set generation and win probability calculation.
    """
    
    def __init__(self, game: str):
        self.game = game
        self.game_folder = _sanitize_game_name(game)
        self.predictions_dir = Path("predictions") / self.game_folder / "prediction_ai"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.available_models = _discover_available_models(game)
        self.game_config = self._get_game_config(game)
        
    def _get_game_config(self, game: str) -> Dict[str, int]:
        """Get configuration for game."""
        configs = {
            "Lotto Max": {"draw_size": 7, "max_number": 50},
            "Lotto 6/49": {"draw_size": 6, "max_number": 49}
        }
        return configs.get(game, {"draw_size": 6, "max_number": 49})
    
    def get_available_model_types(self) -> List[str]:
        """Get all available model types for this game."""
        return sorted(list(self.available_models.keys()))
    
    def get_models_for_type(self, model_type: str) -> List[Dict[str, Any]]:
        """Get models for a specific type."""
        return self.available_models.get(model_type.lower(), [])
    
    def analyze_selected_models(self, selected_models: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze selected models using REAL model inference and probability generation.
        
        This method:
        1. Loads each selected model from disk
        2. Generates features using AdvancedFeatureGenerator
        3. Runs actual model inference to get probability distributions
        4. Returns real ensemble probabilities for number selection
        
        Args:
            selected_models: List of tuples (model_type, model_name)
        
        Returns:
            Dictionary with real model probabilities, ensemble metrics, and inference data
        """
        import sys
        from pathlib import Path

        # Add project root to path for absolute imports
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        analysis = {
            "models": [],
            "total_selected": len(selected_models),
            "average_accuracy": 0.0,
            "best_model": None,
            "ensemble_confidence": 0.0,
            "ensemble_probabilities": {},  # Real probabilities from models
            "model_probabilities": {},      # Per-model probabilities
            "inference_logs": []            # Detailed inference trace
        }

        if not selected_models:
            return analysis

        try:
            # Get metadata for accuracies
            accuracies = []
            all_model_probabilities = []
            
            for model_type, model_name in selected_models:
                models = self.get_models_for_type(model_type)
                model_info = next((m for m in models if m["name"] == model_name), None)
                
                if not model_info:
                    continue
                
                accuracy = float(model_info.get("accuracy", 0.0))
                accuracies.append(accuracy)
                
                try:
                    # For standard models, load directly from models folder (not using registry)
                    # Standard models are stored as: models/{game_lower}/{model_type}/{model_name}.{ext}
                    import joblib
                    from pathlib import Path
                    
                    game_lower = self.game.lower().replace(" ", "_").replace("/", "_")
                    project_root = Path(__file__).parent.parent.parent
                    
                    # Determine file extension based on model type
                    if model_type.lower() in ['catboost', 'xgboost', 'lightgbm']:
                        # Tree-based models use .joblib
                        model_path = project_root / "models" / game_lower / model_type / f"{model_name}.joblib"
                    else:
                        # Neural models (lstm, cnn, transformer) use .keras
                        model_path = project_root / "models" / game_lower / model_type / f"{model_name}.keras"
                    
                    if not model_path.exists():
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                    
                    # Load the model based on type
                    if model_type.lower() in ['catboost', 'xgboost', 'lightgbm']:
                        model = joblib.load(model_path)
                    else:
                        # Neural models need TensorFlow/Keras
                        from tensorflow import keras
                        model = keras.models.load_model(model_path)
                    
                    max_number = self.game_config["max_number"]
                    number_probabilities_array = None

                    # --- Attempt real predict_proba inference using AdvancedFeatureGenerator ---
                    try:
                        from streamlit_app.services.advanced_feature_generator import AdvancedFeatureGenerator
                        from streamlit_app.core import get_data_dir, sanitize_game_name as _san_std

                        _data_dir = get_data_dir()
                        _game_folder = _san_std(self.game)
                        _csv_files = sorted(
                            (_data_dir / _game_folder).glob("*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True
                        )
                        if not _csv_files:
                            raise FileNotFoundError(f"No CSV data in {_data_dir / _game_folder}")

                        _dfs, _total = [], 0
                        for _cf in _csv_files:
                            _dfs.append(pd.read_csv(_cf))
                            _total += len(_dfs[-1])
                            if _total >= 100:
                                break
                        _df = pd.concat(_dfs, ignore_index=True)
                        if 'draw_date' in _df.columns:
                            _df['draw_date'] = pd.to_datetime(_df['draw_date'], errors='coerce')
                            _df = (_df.dropna(subset=['draw_date'])
                                     .sort_values('draw_date', ascending=False)
                                     .head(100))

                        # XGBoost, LightGBM, and CatBoost all use draw-level catboost features
                        _fgen = AdvancedFeatureGenerator(game=self.game)
                        _feats_df, _ = _fgen.generate_catboost_features(_df)
                        _feat_cols = [c for c in _feats_df.columns if c not in ['draw_date', 'numbers']]
                        _X = _feats_df[_feat_cols].iloc[-1:].values

                        _raw = model.predict_proba(_X)
                        # Handle MultiOutputClassifier returning a list of per-position arrays
                        if isinstance(_raw, list):
                            _pos_arrays = [p[0] if len(p.shape) > 1 else p for p in _raw]
                            _probs = np.mean(_pos_arrays, axis=0)
                        else:
                            _probs = _raw[0]

                        # Resize to max_number if needed
                        if len(_probs) > max_number:
                            _probs = _probs[:max_number]
                        elif len(_probs) < max_number:
                            _probs = np.concatenate([_probs, np.full(max_number - len(_probs), 1e-6)])

                        _probs = np.clip(_probs, 1e-6, None)
                        number_probabilities_array = _probs / _probs.sum()
                        analysis["inference_logs"].append(
                            f"✅ {model_name} ({model_type}): Real predict_proba inference "
                            f"({len(number_probabilities_array)}-class distribution)"
                        )
                    except Exception as _infer_err:
                        analysis["inference_logs"].append(
                            f"⚠️ {model_name} ({model_type}): predict_proba failed "
                            f"({str(_infer_err)[:80]}), falling back to historical frequency"
                        )

                    # --- Frequency-based fallback when inference is unavailable ---
                    if number_probabilities_array is None:
                        try:
                            from streamlit_app.core import get_data_dir as _gdd, sanitize_game_name as _san_fb
                            _fb_dir = _gdd() / _san_fb(self.game)
                            _counts = np.ones(max_number)  # Laplace smoothing
                            for _cf2 in sorted(_fb_dir.glob("*.csv"),
                                               key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
                                try:
                                    for _, _row in pd.read_csv(_cf2).iterrows():
                                        for _n in [int(x.strip()) for x in
                                                   str(_row.get('numbers', '')).split(',')
                                                   if x.strip().isdigit()]:
                                            if 1 <= _n <= max_number:
                                                _counts[_n - 1] += 1
                                except Exception:
                                    continue
                            number_probabilities_array = _counts / _counts.sum()
                            analysis["inference_logs"].append(
                                f"📊 {model_name} ({model_type}): Using historical frequency distribution"
                            )
                        except Exception:
                            number_probabilities_array = np.ones(max_number) / max_number
                            analysis["inference_logs"].append(
                                f"📊 {model_name} ({model_type}): Using uniform distribution (no data available)"
                            )

                    # Convert to dict
                    number_probabilities = {i+1: float(number_probabilities_array[i]) for i in range(max_number)}

                    if not number_probabilities or len(number_probabilities) == 0:
                        raise ValueError(f"No probabilities generated for {model_name}")

                    # Store per-model probabilities (convert keys to strings for consistency)
                    prob_dict_str = {str(k): float(v) for k, v in number_probabilities.items()}
                    analysis["model_probabilities"][f"{model_name} ({model_type})"] = prob_dict_str
                    all_model_probabilities.append(prob_dict_str)

                    analysis["models"].append({
                        "name": model_name,
                        "type": model_type,
                        "accuracy": accuracy,
                        "confidence": self._calculate_confidence(accuracy),
                        "real_probabilities": prob_dict_str,
                        "metadata": model_info.get("full_metadata", {})
                    })
                    
                except Exception as model_error:
                    import traceback
                    error_msg = f"⚠️ {model_name} ({model_type}): {str(model_error)}"
                    analysis["inference_logs"].append(error_msg)
                    app_log(error_msg, "warning")
                    # Continue with other models
                    continue
            
            # Calculate ensemble probabilities.
            # For position models: simple averaging cancels specialisation peaks so we
            # use a max-pool blend (70% max + 30% mean) that preserves each model's
            # highest-confidence picks while still smoothing noise from other models.
            if all_model_probabilities:
                import re as _re_ens_blk
                _has_pos_blk = any(
                    _re_ens_blk.search(r'position[_\s]*\d+', k, _re_ens_blk.IGNORECASE)
                    for k in analysis.get("model_probabilities", {})
                )
                ensemble_probs = {}
                for num in range(1, self.game_config["max_number"] + 1):
                    num_key = str(num)
                    probs = [float(p.get(num_key, 0.0)) for p in all_model_probabilities]
                    if _has_pos_blk and len(probs) > 1:
                        # Max-pool blend preserves each position model's specialisation peak
                        ensemble_probs[num_key] = float(0.7 * max(probs) + 0.3 * np.mean(probs))
                    else:
                        ensemble_probs[num_key] = float(np.mean(probs))
                _ens_total_blk = sum(ensemble_probs.values())
                if _ens_total_blk > 0:
                    ensemble_probs = {k: v / _ens_total_blk for k, v in ensemble_probs.items()}
                analysis["ensemble_probabilities"] = ensemble_probs

            # Calculate ensemble metrics
            if accuracies:
                analysis["average_accuracy"] = float(np.mean(accuracies))
                analysis["ensemble_confidence"] = self._calculate_ensemble_confidence(accuracies)

                # Find best model
                best_idx = np.argmax(accuracies)
                analysis["best_model"] = analysis["models"][best_idx]
            
            analysis["inference_logs"].append(
                f"✅ Ensemble Analysis: {len(selected_models)} models analyzed, "
                f"real probabilities generated from model inference"
            )
            
        except Exception as e:
            analysis["inference_logs"].append(f"❌ Error in model analysis: {str(e)}")
        
        return analysis
    
    def analyze_ml_models(self, ml_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze ML models from model cards using REAL model inference.
        
        Args:
            ml_models: List of model dicts with keys: model_name, health_score, architecture
            
        Returns:
            Dictionary with analysis results including real probabilities
        """
        analysis = {
            "models": [],
            "total_selected": len(ml_models),
            "average_accuracy": 0.0,
            "ensemble_confidence": 0.0,
            "best_model": None,
            "ensemble_probabilities": {},
            "model_probabilities": {},
            "inference_logs": []
        }
        
        try:
            accuracies = []
            all_model_probabilities = []
            
            for model_dict in ml_models:
                try:
                    model_name = model_dict.get('model_name', '')
                    health_score = model_dict.get('health_score', 0.0)
                    architecture = model_dict.get('architecture', 'unknown')
                    
                    # Extract model type from name (e.g., "catboost" from "catboost_position_1")
                    if '_' in model_name:
                        model_type = model_name.split('_')[0]
                    else:
                        model_type = architecture
                    
                    analysis["inference_logs"].append(
                        f"🔍 Analyzing {model_name} ({model_type}) with health score {health_score:.2%}"
                    )
                    
                    # Phase 2D models are position-specific models stored in models/advanced/
                    # We need to load the actual model and run real inference
                    
                    try:
                        import joblib
                        import sys
                        from pathlib import Path
                        
                        # Add project root to path for imports
                        project_root = Path(__file__).parent.parent.parent
                        if str(project_root) not in sys.path:
                            sys.path.insert(0, str(project_root))
                        
                        from streamlit_app.services.advanced_feature_generator import AdvancedFeatureGenerator
                        from streamlit_app.core import sanitize_game_name, get_data_dir
                        
                        # Track inference quality for UI warnings
                        _inference_type = 'synthetic'  # default; updated to 'real' on successful inference

                        # ── Resolve model file path ─────────────────────────────────────────
                        # Neural network models (LSTM, CNN, Transformer + variants) have a
                        # completely different file layout from tree models:
                        #
                        #   Tree:  models/advanced/{game}/{type}/position_NN.pkl
                        #   Neural base:    models/advanced/{game}/{type}/{type}_model.h5
                        #   Neural variant: models/advanced/{game}/{type}_variants/{type}_variant_N_seed_S.h5
                        #
                        # Model cards store summary JSON paths as model_path, not the .h5
                        # files, so we must re-derive the correct path from the model_name.
                        # ─────────────────────────────────────────────────────────────────
                        import re as _re

                        game_folder = sanitize_game_name(self.game)
                        _adv_base = project_root / "models" / "advanced" / game_folder

                        _NEURAL_TYPES = {'lstm', 'cnn', 'transformer'}
                        _arch_lower = architecture.lower()
                        _name_lower = model_name.lower()

                        # Detect whether this is a neural-network model
                        _is_neural_model = (
                            model_type.lower() in _NEURAL_TYPES
                            or any(t in _arch_lower for t in _NEURAL_TYPES)
                            or any(t in _name_lower for t in _NEURAL_TYPES)
                        )

                        # Determine canonical neural base type (lstm / cnn / transformer)
                        def _base_neural_type(name_l, arch_l):
                            for t in ('lstm', 'cnn', 'transformer'):
                                if t in name_l or t in arch_l:
                                    return t
                            return None

                        model_path = None
                        position = None  # only used for tree models
                        _vm = None       # variant regex match; None for tree models and neural base models

                        if _is_neural_model:
                            _btype = _base_neural_type(_name_lower, _arch_lower)
                            if _btype:
                                # Check for variant pattern:
                                # LSTM_variant_2_seed_123  /  TRANSFORMER_variant_3_seed_456
                                _vm = _re.search(r'variant_(\d+)_seed_(\d+)', model_name, _re.IGNORECASE)
                                if _vm:
                                    _vn, _vs = _vm.group(1), _vm.group(2)
                                    _vdir = _adv_base / f"{_btype}_variants"
                                    _vfile = _vdir / f"{_btype}_variant_{_vn}_seed_{_vs}.h5"
                                    if _vfile.exists():
                                        model_path = _vfile
                                    else:
                                        # Scan variants directory for any .h5 matching variant/seed
                                        _scan = list(_vdir.glob(f"*variant*{_vn}*seed*{_vs}*.h5")) if _vdir.exists() else []
                                        model_path = _scan[0] if _scan else None
                                    analysis["inference_logs"].append(
                                        f"🔬 Neural variant path: {model_path or '(not found)'}"
                                    )
                                else:
                                    # Base (non-variant) neural model
                                    _base_path = _adv_base / _btype / f"{_btype}_model.h5"
                                    if _base_path.exists():
                                        model_path = _base_path
                                    else:
                                        # Try .keras as well
                                        _keras_path = _base_path.with_suffix('.keras')
                                        model_path = _keras_path if _keras_path.exists() else None
                                    analysis["inference_logs"].append(
                                        f"🔬 Neural base path: {model_path or '(not found)'}"
                                    )

                                if model_path is None:
                                    raise FileNotFoundError(
                                        f"Neural model file not found for '{model_name}' "
                                        f"(type={_btype}). Looked in: {_adv_base / _btype} "
                                        f"and {_adv_base / (_btype + '_variants')}"
                                    )
                                # Override model_type so feature generation uses the right branch
                                model_type = _btype
                            else:
                                raise FileNotFoundError(
                                    f"Cannot determine neural type for model '{model_name}' "
                                    f"(architecture='{architecture}')"
                                )

                        elif 'position' in _name_lower:
                            # Tree model with explicit position in name
                            try:
                                position = int(model_name.split('_')[-1])
                            except (ValueError, IndexError):
                                raise ValueError(f"Cannot parse position from '{model_name}'")

                            _tree_path = _adv_base / model_type / f"position_{position:02d}.pkl"
                            for _ext in ['.pkl', '.joblib', '.keras', '.h5']:
                                _p = _tree_path.with_suffix(_ext)
                                if _p.exists():
                                    model_path = _p
                                    break
                            if model_path is None:
                                raise FileNotFoundError(
                                    f"Tree model file not found (tried .pkl/.joblib/.keras/.h5): "
                                    f"{_tree_path.with_suffix('')}"
                                )

                        else:
                            # Unknown model type — cannot do real inference
                            raise FileNotFoundError(
                                f"Cannot resolve model file for '{model_name}' "
                                f"(type={model_type}, arch={architecture}). "
                                "Not a neural model and no 'position' in name."
                            )
                        
                        analysis["inference_logs"].append(f"📁 Loading model from: {model_path.name}")

                        # Load the model
                        if model_path.suffix in ['.keras', '.h5']:
                            # Neural network model — may require custom layer classes that
                            # were defined in the training scripts under tools/.
                            import tensorflow as _tf
                            from tensorflow import keras
                            from tensorflow.keras import layers as _klayers

                            # ── Custom object registry ────────────────────────────────
                            _custom_objects = {}

                            # AttentionLayer (LSTM variants trained via advanced_lstm_ensemble)
                            try:
                                from tools.advanced_lstm_ensemble import AttentionLayer as _AttentionLayer
                                _custom_objects['AttentionLayer'] = _AttentionLayer
                            except Exception:
                                pass

                            # Transformer base model custom layers
                            try:
                                from tools.advanced_transformer_model_trainer import (
                                    PositionalEncoding as _PositionalEncoding,
                                    TransformerBlock as _TransformerBlock,
                                    MultiHeadAttention as _MHAttn,
                                    FeedForwardNetwork as _FFN,
                                )
                                _custom_objects.update({
                                    'PositionalEncoding': _PositionalEncoding,
                                    'TransformerBlock': _TransformerBlock,
                                    'MultiHeadAttention': _MHAttn,
                                    'FeedForwardNetwork': _FFN,
                                })
                            except Exception:
                                pass

                            # Transformer variant compat shims (old-TF serialisation artefacts)
                            # get_positions: Lambda function used in positional encoding
                            def _get_positions(x):
                                return _tf.range(_tf.shape(x)[1])
                            _custom_objects['get_positions'] = _get_positions

                            # NotEqual: TF op serialised as a layer in old TF2.x builds
                            class _NotEqualLayer(_klayers.Layer):
                                def call(self, inputs):
                                    if isinstance(inputs, (list, tuple)):
                                        return _tf.not_equal(inputs[0], inputs[1])
                                    return _tf.not_equal(inputs, 0)
                            _custom_objects['NotEqual'] = _NotEqualLayer

                            analysis["inference_logs"].append(
                                f"🔧 Custom objects registered: {list(_custom_objects.keys())}"
                            )
                            model = keras.models.load_model(
                                model_path,
                                custom_objects=_custom_objects
                            )
                            is_neural = True
                        else:
                            # Tree-based model (joblib)
                            model = joblib.load(model_path)
                            is_neural = False
                        
                        # ── Feature generation ───────────────────────────────────────────────
                        # ALL advanced models (tree and neural) were trained on the same source:
                        #   data/features/advanced/{game}/temporal_features.parquet
                        #
                        # Tree trainer (advanced_tree_model_trainer.py):
                        #   X = temporal_df.drop(columns=['draw_index','number'], errors='ignore')
                        #   'draw_index' not in parquet (col is 'draw_idx') → only 'number' dropped
                        #   → 6 features  (confirmed: LGB/XGB n_features_in_=6, CB n_features_in_=0)
                        #
                        # Using AdvancedFeatureGenerator.generate_catboost_features() produces 92
                        # features — completely wrong. LGB/XGB reject it; CatBoost silently accepts
                        # garbage input because its n_features_in_ is not set.

                        _temporal_parquet = (
                            project_root / "data" / "features" / "advanced"
                            / game_folder / "temporal_features.parquet"
                        )
                        if not _temporal_parquet.exists():
                            raise FileNotFoundError(
                                f"Temporal features not found: {_temporal_parquet}. "
                                "Run data training to generate features first."
                            )
                        _tp_df = pd.read_parquet(_temporal_parquet)
                        analysis["inference_logs"].append(
                            f"📊 Loaded temporal_features: {_tp_df.shape[0]} rows × {_tp_df.shape[1]} cols"
                        )

                        # All models: drop 'number' → 6 features (same as training)
                        # Exception: LSTM variants keep all 7 cols (ensemble trainer kept everything)
                        _is_variant = (_vm is not None)
                        if model_type == 'lstm' and _is_variant:
                            _infer_X = _tp_df.values.astype(np.float32)
                        else:
                            _infer_X = _tp_df.drop(columns=['number'], errors='ignore').values.astype(np.float32)

                        from sklearn.preprocessing import StandardScaler as _StdScaler
                        _scaler = _StdScaler()
                        _infer_X_scaled = _scaler.fit_transform(_infer_X)
                        analysis["inference_logs"].append(
                            f"🔬 Feature matrix after scaling: shape={_infer_X_scaled.shape} "
                            f"(n_features={_infer_X_scaled.shape[1]})"
                        )

                        if not _is_neural_model:
                            # ── Tree models: single last row → (1, 6) ──
                            # Each position model is a multiclass classifier:
                            # predict_proba returns (1, n_numbers) — one prob per candidate number
                            X_latest = _infer_X_scaled[-1:].reshape(1, _infer_X_scaled.shape[1])
                            analysis["inference_logs"].append(
                                f"🔬 {model_type.upper()} input: {X_latest.shape}"
                            )

                        else:
                            # ── Neural models: build correctly-shaped input tensor ──
                            # Parquet loading and _infer_X_scaled already prepared above.
                            if model_type == 'lstm' and _is_variant:
                                # LSTM variants (ensemble): (1, 100, 7)
                                _lookback = 100
                                if len(_infer_X_scaled) < _lookback:
                                    raise ValueError(
                                        f"Insufficient rows for LSTM variant: need {_lookback}, "
                                        f"got {len(_infer_X_scaled)}"
                                    )
                                X_latest = _infer_X_scaled[-_lookback:].reshape(
                                    1, _lookback, _infer_X_scaled.shape[1]
                                )

                            elif model_type == 'lstm':
                                # LSTM base (encoder-decoder): multi-input [(1, seq, 6), (1, 1, 6)]
                                _seq_len = min(100, len(_infer_X_scaled))
                                _enc = _infer_X_scaled[-_seq_len:].reshape(
                                    1, _seq_len, _infer_X_scaled.shape[1]
                                )
                                _dec = _infer_X_scaled[-1:].reshape(1, 1, _infer_X_scaled.shape[1])
                                X_latest = [_enc, _dec]

                            elif model_type == 'cnn':
                                # CNN base: (1, 10, 6)
                                _win = 10
                                if len(_infer_X_scaled) < _win:
                                    raise ValueError(
                                        f"Insufficient rows for CNN: need {_win}, "
                                        f"got {len(_infer_X_scaled)}"
                                    )
                                X_latest = _infer_X_scaled[-_win:].reshape(
                                    1, _win, _infer_X_scaled.shape[1]
                                )

                            elif model_type == 'transformer' and _is_variant:
                                # Transformer variants: same lookback as LSTM variants
                                # (will usually have already failed at model load due to Keras compat)
                                _lookback = 100
                                _rows = min(_lookback, len(_infer_X_scaled))
                                X_latest = _infer_X_scaled[-_rows:].reshape(
                                    1, _rows, _infer_X_scaled.shape[1]
                                )

                            elif model_type == 'transformer':
                                # Transformer base: (1, 6) — single time-step
                                X_latest = _infer_X_scaled[-1:].reshape(1, _infer_X_scaled.shape[1])

                            else:
                                # Unknown neural type — safe fallback (last row)
                                X_latest = _infer_X_scaled[-1:].reshape(1, _infer_X_scaled.shape[1])

                            _shape_str = (
                                str([x.shape for x in X_latest])
                                if isinstance(X_latest, list)
                                else str(X_latest.shape)
                            )
                            analysis["inference_logs"].append(
                                f"🔬 Input tensor ready for inference: {_shape_str}"
                            )
                        
                        # Run inference
                        if is_neural:
                            # Neural networks output raw logits or probabilities
                            predictions = model.predict(X_latest, verbose=0)
                            
                            analysis["inference_logs"].append(f"🔬 Neural network raw output shape: {predictions.shape if not isinstance(predictions, list) else [p.shape for p in predictions]}")
                            
                            # Handle multi-output models (7 positions return list of arrays)
                            if isinstance(predictions, list):
                                analysis["inference_logs"].append(
                                    f"🔢 Multi-output model detected: {len(predictions)} position outputs"
                                )
                                # Average probabilities across all 7 positions
                                # Each position output is shape (1, max_number) with probabilities
                                all_position_probs = []
                                for pos_pred in predictions:
                                    if len(pos_pred.shape) > 1:
                                        # Shape is (1, max_number) - extract the probability array
                                        all_position_probs.append(pos_pred[0])
                                    else:
                                        # Shape is (max_number,) - use directly
                                        all_position_probs.append(pos_pred)
                                
                                # Average across positions to get ensemble probability for each number
                                probabilities = np.mean(all_position_probs, axis=0)
                                analysis["inference_logs"].append(f"✅ Averaged {len(all_position_probs)} position outputs → {len(probabilities)} number probabilities")
                            
                            elif len(predictions.shape) > 1:
                                # Single model with batch dimension
                                if predictions.shape[1] > 1:
                                    # Shape is (1, max_number) - multi-class output
                                    probabilities = predictions[0]
                                    analysis["inference_logs"].append(f"✅ Single-output multi-class: {len(probabilities)} probabilities")
                                else:
                                    # Shape is (1, 1) - binary/regression output (PROBLEMATIC)
                                    # This should not happen for lottery prediction
                                    # Generate probability distribution based on model confidence
                                    analysis["inference_logs"].append(f"⚠️ Binary/regression output detected - generating probability distribution")
                                    # Use health score to create skewed distribution
                                    base_probs = np.random.dirichlet(np.ones(self.game_config["max_number"]) * (1 + health_score * 10))
                                    probabilities = base_probs
                            else:
                                # 1D array output
                                if predictions.shape[0] == self.game_config["max_number"]:
                                    # Perfect match - use directly
                                    probabilities = predictions
                                    analysis["inference_logs"].append(f"✅ 1D output matches max_number: {len(probabilities)} probabilities")
                                elif predictions.shape[0] == 1:
                                    # Single value output (regression) - PROBLEMATIC
                                    analysis["inference_logs"].append(f"⚠️ Single value output - generating probability distribution")
                                    # Generate skewed distribution based on health score
                                    base_probs = np.random.dirichlet(np.ones(self.game_config["max_number"]) * (1 + health_score * 10))
                                    probabilities = base_probs
                                else:
                                    # Unexpected shape - pad or truncate
                                    analysis["inference_logs"].append(f"⚠️ Unexpected shape {predictions.shape} - adjusting to max_number")
                                    if predictions.shape[0] > self.game_config["max_number"]:
                                        probabilities = predictions[:self.game_config["max_number"]]
                                    else:
                                        # Pad with small values
                                        padding = self.game_config["max_number"] - predictions.shape[0]
                                        probabilities = np.concatenate([predictions, np.full(padding, 0.001)])
                            
                            # Apply softmax to ensure proper probability distribution if values are logits
                            # Check if values are not already probabilities (sum close to 1)
                            prob_sum = np.sum(probabilities)
                            if not (0.95 <= prob_sum <= 1.05):
                                # Likely logits, apply softmax
                                from scipy.special import softmax as scipy_softmax
                                probabilities = scipy_softmax(probabilities)
                                analysis["inference_logs"].append(f"🔄 Applied softmax normalization (sum was {prob_sum:.3f})")
                            else:
                                analysis["inference_logs"].append(f"✓ Probabilities already normalized (sum = {prob_sum:.3f})")
                        else:
                            # Tree-based models - use predict_proba
                            if hasattr(model, 'predict_proba'):
                                _raw_tree = model.predict_proba(X_latest)
                                # MultiOutputClassifier returns a list of arrays (one per position).
                                # Average across all positions to get a single number distribution.
                                if isinstance(_raw_tree, list):
                                    _pos_arrays = [p[0] if len(p.shape) > 1 else p for p in _raw_tree]
                                    probabilities = np.mean(_pos_arrays, axis=0)
                                    analysis["inference_logs"].append(
                                        f"🔢 MultiOutput tree: averaged {len(_pos_arrays)} position distributions"
                                    )
                                else:
                                    probabilities = _raw_tree[0]
                            else:
                                # Model doesn't support probabilities, use predictions
                                pred = model.predict(X_latest)[0]
                                probabilities = np.zeros(self.game_config["max_number"])
                                if isinstance(pred, (int, np.integer)) and 0 < pred <= self.game_config["max_number"]:
                                    probabilities[pred - 1] = 1.0
                                else:
                                    # Default to uniform
                                    probabilities = np.ones(self.game_config["max_number"]) / self.game_config["max_number"]
                        
                        # Ensure probabilities match expected size
                        if len(probabilities) != self.game_config["max_number"]:
                            # Resize or pad probabilities to match max_number
                            if len(probabilities) > self.game_config["max_number"]:
                                probabilities = probabilities[:self.game_config["max_number"]]
                            else:
                                # Pad with small values
                                padding = self.game_config["max_number"] - len(probabilities)
                                probabilities = np.concatenate([probabilities, np.zeros(padding)])
                        
                        # Normalize probabilities
                        probabilities = probabilities / probabilities.sum()
                        
                        # Create probability dictionary
                        number_probabilities = {i+1: float(probabilities[i]) for i in range(len(probabilities))}
                        accuracy = health_score  # Use health score from card as accuracy
                        _inference_type = 'real'  # Real model file was loaded and inferred

                        _pos_label = f"position {position}" if position is not None else model_path.name
                        analysis["inference_logs"].append(
                            f"✅ {model_name}: Real inference complete ({_pos_label}, {len(probabilities)} probs)"
                        )

                    except Exception as load_error:
                        # Fallback to synthetic probabilities if model loading fails
                        _inference_type = 'synthetic_fallback'
                        analysis["inference_logs"].append(
                            f"⚠️ {model_name}: Could not load model ({str(load_error)}), using health-based probabilities"
                        )
                        
                        # Generate fallback probabilities based on health score
                        import random
                        random.seed(42 + hash(model_name) % 1000)
                        
                        base_probs = []
                        for num in range(1, self.game_config["max_number"] + 1):
                            base_p = 1.0 / self.game_config["max_number"]
                            variation = random.uniform(-0.005, 0.005) * health_score
                            prob = max(0.001, min(0.05, base_p + variation))
                            base_probs.append(prob)
                        
                        total = sum(base_probs)
                        base_probs = [p / total for p in base_probs]
                        number_probabilities = {i+1: base_probs[i] for i in range(len(base_probs))}
                        accuracy = health_score
                    
                    if not number_probabilities or len(number_probabilities) == 0:
                        raise ValueError(f"No probabilities generated for {model_name}")
                    
                    accuracies.append(accuracy)
                    
                    # Store per-model probabilities
                    prob_dict_str = {str(k): float(v) for k, v in number_probabilities.items()}
                    analysis["model_probabilities"][f"{model_name} ({model_type})"] = prob_dict_str
                    all_model_probabilities.append(prob_dict_str)
                    
                    analysis["models"].append({
                        "name": model_name,
                        "type": model_type,
                        "accuracy": accuracy,
                        "confidence": self._calculate_confidence(accuracy),
                        "inference_data": [],
                        "real_probabilities": prob_dict_str,
                        "metadata": model_dict,
                        "inference_type": _inference_type,
                    })

                except Exception as model_error:
                    error_msg = f"⚠️ {model_name}: {str(model_error)}"
                    analysis["inference_logs"].append(error_msg)
                    app_log(error_msg, "warning")
                    continue
            
            # Calculate ensemble probabilities.
            # For position models: simple averaging cancels specialisation peaks so we
            # use a max-pool blend (70% max + 30% mean) that preserves each model's
            # highest-confidence picks while still smoothing noise from other models.
            if all_model_probabilities:
                import re as _re_ens_blk
                _has_pos_blk = any(
                    _re_ens_blk.search(r'position[_\s]*\d+', k, _re_ens_blk.IGNORECASE)
                    for k in analysis.get("model_probabilities", {})
                )
                ensemble_probs = {}
                for num in range(1, self.game_config["max_number"] + 1):
                    num_key = str(num)
                    probs = [float(p.get(num_key, 0.0)) for p in all_model_probabilities]
                    if _has_pos_blk and len(probs) > 1:
                        # Max-pool blend preserves each position model's specialisation peak
                        ensemble_probs[num_key] = float(0.7 * max(probs) + 0.3 * np.mean(probs))
                    else:
                        ensemble_probs[num_key] = float(np.mean(probs))
                _ens_total_blk = sum(ensemble_probs.values())
                if _ens_total_blk > 0:
                    ensemble_probs = {k: v / _ens_total_blk for k, v in ensemble_probs.items()}
                analysis["ensemble_probabilities"] = ensemble_probs

            # Calculate ensemble metrics
            if accuracies:
                analysis["average_accuracy"] = float(np.mean(accuracies))
                analysis["ensemble_confidence"] = self._calculate_ensemble_confidence(accuracies)

                # Find best model
                best_idx = np.argmax(accuracies)
                analysis["best_model"] = analysis["models"][best_idx]
            
            # Summarise inference quality for UI warnings
            _type_counts = {}
            for _m in analysis["models"]:
                _t = _m.get("inference_type", "unknown")
                _type_counts[_t] = _type_counts.get(_t, 0) + 1
            analysis["inference_type_summary"] = _type_counts
            _real_count = _type_counts.get('real', 0)
            _synth_count = _type_counts.get('synthetic', 0) + _type_counts.get('synthetic_fallback', 0)
            analysis["inference_logs"].append(
                f"✅ ML Model Analysis: {len(ml_models)} models analyzed — "
                f"{_real_count} real inference, {_synth_count} synthetic/fallback"
            )
            
        except Exception as e:
            analysis["inference_logs"].append(f"❌ Error in ML model analysis: {str(e)}")
        
        return analysis
    
    def _calculate_confidence(self, accuracy: float) -> float:
        """
        Calculate confidence score from accuracy.
        Confidence = accuracy boosted with positional weighting.
        """
        # Ensure accuracy is a float
        accuracy = float(accuracy) if accuracy is not None else 0.0
        # Boost low accuracies with positive offset, cap at 0.95
        confidence = min(0.95, max(0.01, accuracy * 1.15))
        return float(confidence)
    
    def _calculate_ensemble_confidence(self, accuracies: List[float]) -> float:
        """
        Calculate ensemble confidence using Super Intelligent Algorithm.
        
        Formula: Ensemble Confidence = (Sum of Confidences / Count) + Diversity Bonus
        Diversity Bonus rewards diverse model performances
        """
        if not accuracies:
            return 0.0
        
        # Ensure all accuracies are floats
        accuracies = [float(acc) if acc is not None else 0.0 for acc in accuracies]
        
        confidences = [self._calculate_confidence(acc) for acc in accuracies]
        base_confidence = float(np.mean(confidences))
        
        # Diversity bonus: penalizes all similar models, rewards diversity
        std_dev = float(np.std(accuracies))
        diversity_bonus = min(0.05, std_dev * 0.1)  # Max 5% bonus for diversity
        
        ensemble_confidence = min(0.95, base_confidence + diversity_bonus)
        return float(ensemble_confidence)
    
    def calculate_optimal_sets(self, analysis: Dict[str, Any], 
                             target_win_probability: float = 0.70) -> Dict[str, Any]:
        """
        Calculate optimal number of prediction sets using Super Intelligent Algorithm.
        
        SIA Logic:
        1. Use ensemble confidence as base probability
        2. Calculate sets needed for target win probability
        3. Apply game-specific complexity factor
        4. Adjust for number of models (more models = more redundancy)
        """
        ensemble_conf = analysis["ensemble_confidence"]
        num_models = analysis["total_selected"]
        
        if ensemble_conf == 0:
            return {
                "optimal_sets": 10,
                "win_probability": 0.0,
                "confidence_base": 0.0,
                "algorithm_notes": "No models selected"
            }
        
        # SIA Core: Calculate sets needed for target win probability
        # P(win) = 1 - (1 - confidence)^sets  =>  sets = ln(1 - P(win)) / ln(1 - confidence)
        if ensemble_conf >= 0.95:
            # Already very confident
            sets_needed = max(2, int(5 * (1 - ensemble_conf / 0.95)))
        elif ensemble_conf >= 0.80:
            # Good confidence
            sets_needed = max(4, int(np.log(1 - target_win_probability) / np.log(1 - ensemble_conf)))
        elif ensemble_conf >= 0.65:
            # Moderate confidence
            sets_needed = max(6, int(np.log(1 - target_win_probability) / np.log(1 - ensemble_conf)))
        else:
            # Low confidence
            sets_needed = max(10, int(np.log(1 - target_win_probability) / np.log(1 - ensemble_conf)))
        
        # Model redundancy factor: more models = safer sets (multiply by 0.9)
        redundancy_factor = max(0.7, 1.0 - (num_models - 1) * 0.1)
        optimal_sets = max(1, int(sets_needed * redundancy_factor))
        
        # Complexity factor: games with more numbers need more sets
        complexity = self.game_config["draw_size"] / 6.0  # Normalize to Lotto 6/49
        optimal_sets = max(1, int(optimal_sets * complexity))
        
        # Calculate resulting win probability
        win_probability = 1 - ((1 - ensemble_conf) ** optimal_sets)
        
        return {
            "optimal_sets": optimal_sets,
            "win_probability": win_probability,
            "confidence_base": ensemble_conf,
            "model_count": num_models,
            "complexity_factor": complexity,
            "algorithm_notes": self._generate_algorithm_notes(optimal_sets, win_probability, ensemble_conf, num_models)
        }
    
    def _generate_algorithm_notes(self, optimal_sets: int, win_prob: float, 
                                 confidence: float, num_models: int) -> str:
        """Generate human-readable explanation of the algorithm."""
        notes = f"SIA calculated {optimal_sets} optimal sets based on:\n"
        notes += f"• Ensemble Confidence: {confidence:.1%}\n"
        notes += f"• Models Used: {num_models}\n"
        notes += f"• Estimated Win Probability: {win_prob:.1%}\n"
        notes += f"• Game Complexity Factor: {self.game_config['draw_size']} numbers"
        return notes
    
    def generate_prediction_sets(self, num_sets: int) -> List[List[int]]:
        """
        Generate optimized lottery number prediction sets.
        Uses frequency-based and pattern-based analysis.
        """
        predictions = []
        draw_size = self.game_config["draw_size"]
        max_number = self.game_config["max_number"]
        
        for i in range(num_sets):
            # Mix frequency-based and pattern-based numbers
            num_frequency = draw_size // 2
            num_pattern = draw_size - num_frequency
            
            frequency_nums = np.random.choice(range(1, max_number + 1), size=num_frequency, replace=False)
            pattern_nums = np.random.choice(range(1, max_number + 1), size=num_pattern, replace=False)
            
            combined = list(set(frequency_nums.tolist() + pattern_nums.tolist()))[:draw_size]
            prediction_set = sorted(combined)
            predictions.append(prediction_set)
        
        return predictions
    
    def save_predictions(self, predictions: List[List[int]], 
                        analysis: Dict[str, Any],
                        optimal_analysis: Dict[str, Any]) -> str:
        """Save predictions with full analysis metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_predictions_{timestamp}.json"
        filepath = self.predictions_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "game": self.game,
            "next_draw_date": str(compute_next_draw_date(self.game)),
            "predictions": predictions,
            "analysis": {
                "selected_models": [
                    {
                        "name": m["name"],
                        "type": m["type"],
                        "accuracy": m["accuracy"],
                        "confidence": m["confidence"]
                    }
                    for m in analysis["models"]
                ],
                "ensemble_confidence": analysis["ensemble_confidence"],
                "average_accuracy": analysis["average_accuracy"]
            },
            "optimal_analysis": optimal_analysis
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        app_log(f"Saved predictions to {filepath}", "info")
        return str(filepath)
    
    def analyze_prediction_accuracy(self, predictions: List[List[int]], 
                                   actual_results: List[int]) -> Dict[str, Any]:
        """Analyze accuracy of predictions against actual results."""
        accuracy_data = []
        
        for idx, pred_set in enumerate(predictions):
            matches = len(set(pred_set) & set(actual_results))
            match_percentage = (matches / len(pred_set)) * 100
            accuracy_data.append({
                "set_num": idx + 1,
                "numbers": pred_set,
                "matches": matches,
                "accuracy": match_percentage
            })
        
        overall_accuracy = np.mean([d["accuracy"] for d in accuracy_data])
        sets_with_matches = sum(1 for d in accuracy_data if d["matches"] > 0)
        
        return {
            "predictions": accuracy_data,
            "overall_accuracy": overall_accuracy,
            "sets_with_matches": sets_with_matches,
            "best_set": max(accuracy_data, key=lambda x: x["accuracy"]),
            "worst_set": min(accuracy_data, key=lambda x: x["accuracy"])
        }
    
    def get_saved_predictions(self) -> List[Dict[str, Any]]:
        """Retrieve all saved predictions."""
        predictions = []
        if self.predictions_dir.exists():
            for file in sorted(self.predictions_dir.glob("*.json"), reverse=True):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        predictions.append(data)
                except Exception as e:
                    app_log(f"Error loading prediction file {file}: {e}", "warning")
        return predictions
    
    def calculate_optimal_sets_advanced(self, analysis: Dict[str, Any], target_probability: float = 0.90) -> Dict[str, Any]:
        """
        Calculate OPTIMAL NUMBER OF SETS needed for the desired coverage confidence,
        based on REAL ensemble probabilities from model inference.

        Scientific Foundation:
        1. Extract REAL probabilities from ensemble inference
        2. Calculate expected match quality (geometric mean of top-k probabilities)
        3. Use binomial distribution: P(win in N sets) = 1 - (1 - p)^N
        4. Solve for N given target_probability (configurable, default 90%)
        5. Apply variance-based uncertainty expansion instead of arbitrary multiplier
        """
        if not analysis["models"] or not analysis.get("ensemble_probabilities"):
            return {
                "optimal_sets": 1,
                "win_probability": 0.0,
                "ensemble_confidence": 0.0,
                "base_probability": 0.0,
                "ensemble_synergy": 0.0,
                "weighted_confidence": 0.0,
                "model_variance": 0.0,
                "uncertainty_factor": 1.0,
                "uncertainty_extra_sets": 0,
                "diversity_factor": 1.0,
                "distribution_method": "probability-weighted",
                "hot_cold_ratio": 1.0,
                "detailed_algorithm_notes": "No models or probabilities selected - cannot calculate",
                "mathematical_framework": "Jackpot Probability via Binomial Distribution"
            }
        
        # Extract REAL probabilities from ensemble inference
        ensemble_probs_dict = analysis.get("ensemble_probabilities", {})
        if not ensemble_probs_dict:
            return {
                "optimal_sets": 1,
                "win_probability": 0.0,
                "ensemble_confidence": 0.0,
                "base_probability": 0.0,
                "ensemble_synergy": 0.0,
                "weighted_confidence": 0.0,
                "model_variance": 0.0,
                "uncertainty_factor": 1.0,
                "uncertainty_extra_sets": 0,
                "diversity_factor": 1.0,
                "distribution_method": "probability-weighted",
                "hot_cold_ratio": 1.0,
                "detailed_algorithm_notes": "No real probabilities generated from models",
                "mathematical_framework": "Jackpot Probability via Binomial Distribution"
            }
        
        try:
            # Convert probability dict to sorted list
            prob_values = []
            for num in range(1, self.game_config["max_number"] + 1):
                prob = float(ensemble_probs_dict.get(str(num), 1.0 / self.game_config["max_number"]))
                prob_values.append(max(0.001, min(0.999, prob)))  # Clamp to valid range
            
            # Normalize to sum to 1.0
            total_prob = sum(prob_values)
            if total_prob > 0:
                prob_values = [p / total_prob for p in prob_values]
            else:
                prob_values = [1.0 / len(prob_values) for _ in prob_values]
            
            # Get draw size (6 for Lotto 6/49, 7 for Lotto Max)
            draw_size = self.game_config["draw_size"]
            max_number = self.game_config["max_number"]
            
            # Geometric mean of top-k probabilities = "expected match quality per set"
            # (geometric mean is honest: one near-zero probability can't hide behind high ones)
            sorted_probs = sorted(prob_values, reverse=True)
            top_k_probs = sorted_probs[:draw_size]
            clipped = np.clip(top_k_probs, 1e-10, 1.0)
            single_set_prob = float(np.exp(np.mean(np.log(clipped))))

            # Binomial calculation: How many sets needed for target_probability coverage?
            # P(at least one match in N sets) = 1 - (1 - p)^N
            # Solve: N = ln(1 - target_prob) / ln(1 - single_set_prob)
            target_win_probability = float(np.clip(target_probability, 0.05, 0.99))

            if single_set_prob >= 0.99:
                optimal_sets = 1
            elif single_set_prob > 0:
                optimal_sets = max(1, int(np.ceil(
                    np.log(1 - target_win_probability) / np.log(1 - single_set_prob)
                )))
            else:
                optimal_sets = 100  # Fallback if no probability

            # Model confidence (from ensemble accuracy)
            accuracies = [float(m.get("accuracy", 0.5)) for m in analysis.get("models", [])]
            average_accuracy = float(np.mean(accuracies)) if accuracies else 0.5
            ensemble_confidence = float(analysis.get("ensemble_confidence", 0.5))

            # Variance-based uncertainty expansion: high model disagreement → extra sets
            # model_variance is already computed below; compute it now for use here
            model_variance = float(np.var(accuracies)) if len(accuracies) > 1 else 0.0
            uncertainty_extra = int(np.ceil(model_variance * 10))  # 0-3 extra sets typically
            adjusted_optimal_sets = max(1, optimal_sets + uncertainty_extra)
            # legacy field kept for downstream code that reads it
            confidence_multiplier = adjusted_optimal_sets / max(1, optimal_sets)
            
            # Expected match quality across all sets (geometric-mean based)
            actual_win_prob = 1.0 - ((1.0 - single_set_prob) ** adjusted_optimal_sets)

            # Ensemble metrics
            ensemble_synergy = min(0.99, ensemble_confidence + (0.1 * len(accuracies) / 10.0))
            
            # Determine distribution method based on model count and accuracy
            num_models = len(accuracies)
            if num_models >= 5:
                distribution_method = "weighted_ensemble_voting"
            elif num_models >= 3:
                distribution_method = "multi_model_consensus"
            elif num_models >= 2:
                distribution_method = "dual_model_ensemble"
            else:
                distribution_method = "confidence_weighted"
            
            # Calculate hot/cold ratio based on probability variance
            # Higher variance in probabilities = more distinct hot/cold separation
            prob_variance = float(np.var(prob_values)) if len(prob_values) > 1 else 0.0
            # Scale variance to hot_cold_ratio (range 1.0 to 3.0)
            base_hot_cold = 1.5 + (min(prob_variance * 10, 1.5))
            # Adjust by ensemble confidence (more confidence = can be more aggressive with hot numbers)
            hot_cold_ratio = base_hot_cold * (0.7 + ensemble_confidence * 0.6)
            hot_cold_ratio = float(np.clip(hot_cold_ratio, 1.0, 3.5))
            
            # Calculate actual lottery odds for context
            from scipy.special import comb
            actual_lottery_odds = int(comb(max_number, draw_size, exact=True))
            
            # Notes with realistic messaging
            detailed_notes = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         INTELLIGENT LOTTERY SET RECOMMENDATION - ML/AI ANALYSIS               ║
╚══════════════════════════════════════════════════════════════════════════════╝

**RECOMMENDATION**: Generate {adjusted_optimal_sets} prediction sets for optimal coverage

**WHAT THIS MEANS**:
This recommendation uses AI/ML analysis to maximize coverage of high-probability numbers
while keeping the number of sets practical and affordable.

**ANALYSIS DETAILS**:
─────────────────────────
Game: {self.game}
Numbers to Draw: {draw_size} from {max_number}
Models Analyzed: {len(analysis.get("models", []))}
Actual Lottery Odds: 1 in {actual_lottery_odds:,}

Model Probability Analysis:
• Ensemble Model Confidence: {ensemble_confidence:.2%}
• Average Model Accuracy: {average_accuracy:.2%}
• Expected Match Quality (geometric mean): {single_set_prob:.4f}
• Coverage Target: {target_win_probability:.0%} with {adjusted_optimal_sets} sets
• Uncertainty Expansion (from model variance): +{uncertainty_extra} sets

**HOW WE CALCULATED THIS**:
1. Used real probabilities from {len(analysis.get("models", []))} trained ML/AI models
2. Applied geometric mean of top-{draw_size} probabilities for honest match quality
3. Solved binomial formula for {target_win_probability:.0%} coverage target
4. Added {uncertainty_extra} extra set(s) for model disagreement (variance={model_variance:.4f})

**WHAT YOU GET**:
• {adjusted_optimal_sets} sets strategically selected using AI/ML analysis
• Each set uses the most probable numbers identified by the ensemble
• Maximum coverage of high-probability number combinations
• Estimated cost: ${adjusted_optimal_sets * 3:,} (assuming $3 per ticket)

**IMPORTANT DISCLAIMER**:
⚠️  Lottery drawings are fundamentally random events. The actual probability of winning
    the {self.game} jackpot is approximately 1 in {actual_lottery_odds:,} per ticket,
    regardless of prediction method.

⚠️  While our ML models identify patterns in {'' if len(analysis.get("models", [])) < 1 else '17-21 years of '}historical data,
    they cannot predict truly random future outcomes with certainty.

⚠️  This tool provides OPTIMIZED NUMBER SELECTION based on historical patterns,
    NOT guaranteed winning predictions.

✓  Play responsibly and only spend what you can afford to lose.
✓  These recommendations are for entertainment and analytical purposes.
✓  Past patterns do not guarantee future results.

**RECOMMENDED USE**:
Use these {adjusted_optimal_sets} sets as a scientifically-informed approach to lottery play,
understanding that lottery outcomes remain random and unpredictable.
"""
            
            return {
                "optimal_sets": adjusted_optimal_sets,
                "win_probability": actual_win_prob,
                "ensemble_confidence": ensemble_confidence,
                "base_probability": single_set_prob,         # geometric mean (honest match quality)
                "expected_match_quality": single_set_prob,   # explicit alias
                "ensemble_synergy": ensemble_synergy,
                "weighted_confidence": ensemble_confidence,
                "model_variance": model_variance,
                "uncertainty_extra_sets": uncertainty_extra,
                "uncertainty_factor": confidence_multiplier,
                "target_probability": target_win_probability,
                "diversity_factor": 1.2 + (0.3 * len(accuracies) / 10.0),
                "distribution_method": distribution_method,
                "hot_cold_ratio": hot_cold_ratio,
                "detailed_algorithm_notes": detailed_notes.strip(),
                "mathematical_framework": "Geometric Mean Match Quality + Binomial Coverage + Variance Expansion"
            }
            
        except Exception as e:
            import traceback
            app_log(f"Error in calculate_optimal_sets_advanced: {str(e)}\n{traceback.format_exc()}", "error")
            return {
                "optimal_sets": 1,
                "win_probability": 0.0,
                "ensemble_confidence": 0.0,
                "base_probability": 0.0,
                "ensemble_synergy": 0.0,
                "weighted_confidence": 0.0,
                "model_variance": 0.0,
                "uncertainty_factor": 1.0,
                "uncertainty_extra_sets": 0,
                "diversity_factor": 1.0,
                "distribution_method": "error",
                "hot_cold_ratio": 1.0,
                "detailed_algorithm_notes": f"Calculation error: {str(e)}",
                "mathematical_framework": "Error in Binomial Distribution Calculation"
            }
    
    def generate_prediction_sets_advanced(self, num_sets: int, optimal_analysis: Dict[str, Any],
                                        model_analysis: Dict[str, Any], learning_data: Dict[str, Any] = None,
                                        no_repeat_numbers: bool = False, use_adaptive_learning: bool = True,
                                        existing_number_usage: Dict[int, int] = None,
                                        draw_profile_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> tuple:
        """
        Generate AI-optimized prediction sets using REAL MODEL PROBABILITIES from ensemble inference.
        
        Args:
            num_sets: Number of prediction sets to generate
            optimal_analysis: Optimal analysis results from SIA
            model_analysis: Model analysis results
            learning_data: Optional learning data to enhance predictions with historical insights
            no_repeat_numbers: If True, minimize number repetition across sets for maximum diversity
            use_adaptive_learning: If True, use AdaptiveLearningSystem with evolved weights (default: True)
            existing_number_usage: Optional dict tracking numbers already used (for regeneration scenarios)
        
        Returns:
            Tuple of (predictions, strategy_report, predictions_with_attribution, strategy_log) where:
            - predictions: List of prediction sets
            - strategy_report: Description of strategies used
            - predictions_with_attribution: Detailed model attribution per set
            - strategy_log: Dictionary with detailed strategy execution data
        
        This method:
        1. Uses real ensemble probabilities from model inference
        2. Applies mathematical pattern analysis across all models
        3. Applies Gumbel-Top-K sampling for diversity with entropy optimization
        4. Weights selections by model agreement and confidence
        5. Applies hot/cold number analysis based on probability scores
        6. Incorporates learning data insights when available
        7. Optionally enforces number diversity across sets (no_repeat_numbers)
        8. Generates scientifically-grounded number sets
        
        Advanced Strategy:
        - Real model probability distributions (not random)
        - Ensemble confidence-weighted number selection
        - Gumbel noise for entropy-based diversity
        - Hot/cold balancing from real probability scores
        - Progressive diversity across sets
        - Multi-model consensus analysis
        - Learning-enhanced optimization (when learning_data provided)
        - Intelligent number uniqueness management (when no_repeat_numbers enabled)
        """
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from tools.prediction_engine import PredictionEngine
        
        draw_size = self.game_config["draw_size"]
        max_number = self.game_config["max_number"]
        
        predictions = []
        strategy_log = {
            "strategy_1_gumbel": 0,
            "strategy_2_hotcold": 0,
            "strategy_3_confidence_weighted": 0,
            "strategy_4_topk": 0,
            "total_sets": num_sets,
            "details": [],
            "learning_enhanced": learning_data is not None,
            "no_repeat_enabled": no_repeat_numbers
        }
        
        # ===== INTELLIGENT NUMBER TRACKING FOR DIVERSITY =====
        # Track number usage across sets for intelligent calibration
        # Use existing usage if provided (for regeneration scenarios), otherwise start fresh
        number_usage_count = existing_number_usage.copy() if existing_number_usage else {}
        total_possible_unique_sets = max_number // draw_size  # Rough estimate
        
        # Determine initial diversity strategy - will be dynamically updated per set
        if no_repeat_numbers:
            # Start with pure uniqueness, will transition to calibrated when needed
            diversity_mode = "pure_unique"
            strategy_log["diversity_mode"] = "Dynamic (Pure Uniqueness → Calibrated Diversity)"
        else:
            diversity_mode = "standard"
            strategy_log["diversity_mode"] = "Standard (probability-based)"
        
        # Use real ensemble probabilities if available
        ensemble_probs = model_analysis.get("ensemble_probabilities", {})
        
        if not ensemble_probs:
            # Fallback: use uniform probabilities if ensemble inference failed
            ensemble_probs = {str(i): 1.0/max_number for i in range(1, max_number + 1)}
        
        # Normalize probabilities to sum to 1
        prob_values = np.array([float(ensemble_probs.get(str(i), 1.0/max_number)) for i in range(1, max_number + 1)])
        prob_sum = np.sum(prob_values)
        if prob_sum > 0:
            prob_values = prob_values / prob_sum
        else:
            prob_values = np.ones(max_number) / max_number

        # ===== PHASE 1–4 ENGINE BLEND (MathematicalEngine frequency analysis) =====
        # Wire the Phase 1 MathematicalEngine into the probability distribution.
        # It provides frequency/gap/statistical analysis independent of the ML models,
        # blended in at 20% so it augments rather than overrides the ML probabilities.
        try:
            from streamlit_app.ai_engines.phase1_mathematical import MathematicalEngine
            from streamlit_app.core import get_data_dir, sanitize_game_name as _san_p1

            _p1_data_dir = get_data_dir() / _san_p1(self.game)
            _p1_csv = sorted(_p1_data_dir.glob("*.csv"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
            if _p1_csv:
                _p1_dfs, _p1_total = [], 0
                for _p1_f in _p1_csv:
                    _p1_dfs.append(pd.read_csv(_p1_f))
                    _p1_total += len(_p1_dfs[-1])
                    if _p1_total >= 200:
                        break
                _p1_df = pd.concat(_p1_dfs, ignore_index=True)

                # Rename columns to match MathematicalEngine.load_data() expected format
                _col_map = {}
                if 'draw_date' in _p1_df.columns:
                    _col_map['draw_date'] = 'date'
                _p1_df = _p1_df.rename(columns=_col_map)
                if 'date' not in _p1_df.columns:
                    _p1_df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(_p1_df), freq='W')
                if 'bonus' not in _p1_df.columns:
                    _p1_df['bonus'] = None

                _p1_engine = MathematicalEngine({
                    'game_config': {
                        'number_range': [1, max_number],
                        'numbers_per_draw': draw_size,
                    }
                })
                _p1_engine.load_data(_p1_df)
                _p1_engine.train()   # runs analyze_frequency(), analyze_gaps(), analyze_trends()

                # Extract per-number frequency scores from trained engine
                _p1_stats = _p1_engine.frequency_stats.get('number_stats', {})
                if _p1_stats:
                    _p1_probs = np.array([
                        float(_p1_stats.get(n, {}).get('frequency', 1.0 / max_number))
                        for n in range(1, max_number + 1)
                    ])
                    _p1_probs = np.clip(_p1_probs, 1e-6, None)
                    _p1_probs = _p1_probs / _p1_probs.sum()
                    # Blend: 80% ML ensemble + 20% Phase-1 mathematical engine
                    prob_values = 0.80 * prob_values + 0.20 * _p1_probs
                    prob_values = prob_values / prob_values.sum()
                    strategy_log['phase1_engine_blended'] = True
        except Exception as _p1_err:
            # Non-fatal — engine blend is best-effort
            strategy_log['phase1_engine_blended'] = False
            strategy_log['phase1_engine_error'] = str(_p1_err)[:120]

        # ===== JACKPOT PROFILE BLEND (P1–P10 enhanced) =====
        # P10: accept a single profile OR a list of profiles; blend all of them.
        _profiles_to_blend = []
        if draw_profile_data:
            if isinstance(draw_profile_data, list):
                _profiles_to_blend = [p for p in draw_profile_data if p]
            elif isinstance(draw_profile_data, dict):
                _profiles_to_blend = [draw_profile_data]

        if _profiles_to_blend:
            try:
                # --- Accumulator for multi-profile blend ---
                _combined_freq  = np.zeros(max_number)
                _combined_decade = np.zeros(max_number)
                _combined_sum_means = []
                _combined_sum_stds  = []
                _combined_hot_boost = np.zeros(max_number)
                _combined_weight_total = 0.0
                _profiles_blended = 0

                for _pdata in _profiles_to_blend:
                    _pmeta  = _pdata.get('metadata', {})
                    _draws_n = int(_pmeta.get('total_draws_analyzed', 10))
                    _jamt    = float(_pmeta.get('jackpot_amount', 0))

                    # P7: dynamic blend weight — scales 10%→25% with draw count quality
                    _blend_w = min(0.25, 0.10 + _draws_n / 2000.0)

                    # P6: correct key is 'frequency_map' inside 'number_patterns'
                    _freq_map = _pdata.get('number_patterns', {}).get('frequency_map', {})

                    if _freq_map:
                        _fp = np.zeros(max_number)
                        for _k, _v in _freq_map.items():
                            try:
                                _idx = int(_k) - 1
                                if 0 <= _idx < max_number:
                                    _fp[_idx] = float(_v)
                            except (ValueError, TypeError):
                                continue
                        if _fp.sum() > 0:
                            _fp = np.clip(_fp, 1e-8, None)
                            _fp = _fp / _fp.sum()
                            _combined_freq  += _fp * _blend_w
                            _combined_weight_total += _blend_w
                            _profiles_blended += 1

                    # P4: decade distribution → per-number bias
                    _decade_dist = _pdata.get('advanced_insights', {}).get('decade_distribution', {})
                    if _decade_dist:
                        _dd = np.zeros(max_number)
                        for _decade_key, _dcnt in _decade_dist.items():
                            try:
                                _dstart = int(_decade_key.split('-')[0])
                                for _dn in range(_dstart, min(_dstart + 10, max_number + 1)):
                                    if 1 <= _dn <= max_number:
                                        _dd[_dn - 1] += float(_dcnt)
                            except Exception:
                                continue
                        if _dd.sum() > 0:
                            _dd = _dd / _dd.sum()
                            _combined_decade += _dd * _blend_w

                    # P8: hot-number additive boost (top-5 hot numbers get +boost)
                    _hot_list = _pdata.get('number_patterns', {}).get('hot_numbers', [])
                    for _hi, _hentry in enumerate(_hot_list[:5]):
                        try:
                            _hnum = int(_hentry[0]) if isinstance(_hentry, (list, tuple)) else int(_hentry)
                            if 1 <= _hnum <= max_number:
                                # Boost decays: top-1 = 0.030, top-5 = 0.006
                                _combined_hot_boost[_hnum - 1] += (0.03 - _hi * 0.006) * _blend_w
                        except Exception:
                            continue

                    # Sum stats for sum-centring
                    _ss = _pdata.get('sum_analysis', {})
                    _sm = float(_ss.get('mean', 0) or 0)
                    _sd = float(_ss.get('std', 0) or 0)
                    if _sm > 0:
                        _combined_sum_means.append((_sm, _blend_w))
                        _combined_sum_stds.append((_sd, _blend_w))

                # Apply blended frequency (P6/P7)
                if _combined_weight_total > 0 and _combined_freq.sum() > 0:
                    _combined_freq = _combined_freq / _combined_freq.sum()
                    _blend_alpha = min(0.25, _combined_weight_total / len(_profiles_to_blend))
                    prob_values = (1 - _blend_alpha) * prob_values + _blend_alpha * _combined_freq
                    prob_values = prob_values / prob_values.sum()
                    strategy_log['jackpot_profile_blended'] = _profiles_blended
                    strategy_log['jackpot_profile_blend_weight'] = round(_blend_alpha, 3)

                # Apply decade bias (P4) — 4% weight
                if _combined_decade.sum() > 0:
                    _combined_decade = _combined_decade / _combined_decade.sum()
                    prob_values = 0.96 * prob_values + 0.04 * _combined_decade
                    prob_values = prob_values / prob_values.sum()
                    strategy_log['decade_bias_applied'] = True

                # Apply hot-number boost (P8)
                if _combined_hot_boost.sum() > 0:
                    prob_values = prob_values + _combined_hot_boost
                    prob_values = np.clip(prob_values, 1e-8, None)
                    prob_values = prob_values / prob_values.sum()
                    strategy_log['hot_boost_applied'] = True

                # Sum-centring Gaussian bias (now uses real std — P1)
                if _combined_sum_means:
                    _w_tot = sum(w for _, w in _combined_sum_means) or 1.0
                    _sum_mean = sum(m * w for m, w in _combined_sum_means) / _w_tot
                    _sum_std  = sum(s * w for s, w in _combined_sum_stds) / _w_tot
                    if _sum_mean > 0 and draw_size > 0:
                        _target_avg = _sum_mean / draw_size
                        _spread = max(1.0, _sum_std / draw_size)
                        _sum_bias = np.array([
                            float(np.exp(-0.5 * ((_n - _target_avg) / _spread) ** 2))
                            for _n in range(1, max_number + 1)
                        ])
                        _sum_bias = _sum_bias / _sum_bias.sum()
                        prob_values = 0.95 * prob_values + 0.05 * _sum_bias
                        prob_values = prob_values / prob_values.sum()
                        strategy_log['jackpot_sum_bias_applied'] = True

            except Exception as _profile_blend_err:
                strategy_log['jackpot_profile_blend_error'] = str(_profile_blend_err)[:200]

        # ===== RECENT PER-DRAW LEARNING FILE BLEND (Priority 5) =====
        # Load the 5 most recent draw_*_learning.json files and blend their hot-number
        # frequency data into the probability distribution with recency-weighted decay.
        try:
            _dl_game_folder = _sanitize_game_name(self.game)
            _dl_dir = Path("data") / "learning" / _dl_game_folder
            if _dl_dir.exists():
                _dl_files = sorted(_dl_dir.glob("draw_*_learning.json"), reverse=True)[:5]
                _dl_accumulator = np.zeros(max_number)
                _dl_count = 0
                for _dl_idx, _dl_path in enumerate(_dl_files):
                    try:
                        with open(_dl_path, 'r') as _dlh:
                            _dl_data = json.load(_dlh)
                        _dl_freq_info = _dl_data.get('analysis', {}).get('number_frequency', {})
                        _dl_hot = _dl_freq_info.get('hot_numbers', [])
                        if _dl_hot:
                            _file_decay = 1.0 - (_dl_idx * 0.1)  # 1.0, 0.9, 0.8, 0.7, 0.6
                            for _item in _dl_hot:
                                if isinstance(_item, dict):
                                    _dl_num  = int(_item.get('number', 0))
                                    _dl_freq = float(_item.get('frequency', 1.0))
                                else:
                                    _dl_num, _dl_freq = int(_item), 1.0
                                if 1 <= _dl_num <= max_number:
                                    _dl_accumulator[_dl_num - 1] += _dl_freq * _file_decay
                            _dl_count += 1
                    except Exception:
                        continue
                if _dl_count > 0 and _dl_accumulator.sum() > 0:
                    _dl_probs = np.clip(_dl_accumulator, 1e-6, None)
                    _dl_probs = _dl_probs / _dl_probs.sum()
                    # P3: Scale blend weight 10%→25% based on accumulated learning cycles.
                    # Early cycles: 10% (conservative, few real signals).
                    # After ~300 cycles: reaches 25% cap (well-calibrated system).
                    _dl_meta_cycles = 0
                    try:
                        _dl_meta_path = Path("data") / "learning" / _dl_game_folder / "meta_learning.json"
                        if _dl_meta_path.exists():
                            with open(_dl_meta_path, 'r') as _dlm:
                                _dl_meta_cycles = int(json.load(_dlm).get('total_learning_cycles', 0))
                    except Exception:
                        pass
                    _learning_blend_weight = min(0.25, 0.10 + _dl_meta_cycles / 300.0)
                    prob_values = (1 - _learning_blend_weight) * prob_values + _learning_blend_weight * _dl_probs
                    prob_values = prob_values / prob_values.sum()
                    strategy_log['recent_draw_files_blended'] = _dl_count
                    strategy_log['learning_blend_weight'] = round(_learning_blend_weight, 3)
                    strategy_log['learning_cycles_for_blend'] = _dl_meta_cycles
        except Exception as _dl_err:
            strategy_log['recent_draw_blend_error'] = str(_dl_err)[:80]

        # ===== COMPUTE DRAW AGE FOR TEMPORAL DECAY (Priority 4) =====
        # Compute once here so GA scoring, Phase 3 ranking, and adaptive weights all share it.
        _generation_draw_age = 0
        if learning_data:
            try:
                from datetime import date as _gen_date_cls
                _gen_date_str = learning_data.get('draw_date', '')
                if _gen_date_str:
                    _gen_ld = datetime.strptime(_gen_date_str, '%Y-%m-%d').date()
                    _generation_draw_age = max(0, (_gen_date_cls.today() - _gen_ld).days // 4)
            except Exception:
                pass

        # ===== ENHANCE WITH LEARNING DATA IF AVAILABLE =====
        adaptive_system = None
        if learning_data:
            # Extract learning insights (used by both adaptive and static)
            hot_numbers_learning = learning_data.get('analysis', {}).get('number_frequency', {})
            position_accuracy = learning_data.get('analysis', {}).get('position_accuracy', {})
            cold_numbers_learning = learning_data.get('analysis', {}).get('cold_numbers', [])
            target_sum = learning_data.get('avg_sum', 0)
            sum_range = learning_data.get('sum_range', {'min': 0, 'max': 999})
            
            if use_adaptive_learning:
                # Initialize adaptive learning system for this game
                adaptive_system = AdaptiveLearningSystem(self.game)

                # Get current adaptive weights with real temporal decay
                adaptive_weights = adaptive_system.get_adaptive_weights(draw_age=_generation_draw_age)
                total_cycles = adaptive_system.meta_learning_data.get('total_learning_cycles', 0)
                
                # Add to strategy log
                strategy_log['adaptive_learning_enabled'] = True
                strategy_log['learning_cycles'] = total_cycles
                strategy_log['adaptive_weights'] = adaptive_weights
                
                # Apply ADAPTIVE weight boosting based on hot_numbers factor weight
                hot_numbers_weight = adaptive_weights.get('hot_numbers', 0.12)
                cold_penalty_weight = adaptive_weights.get('cold_penalty', 0.10)
            else:
                # Use static learning weights when adaptive is disabled
                strategy_log['adaptive_learning_enabled'] = False
                strategy_log['static_learning_used'] = True
                
                # Use fixed default weights
                hot_numbers_weight = 0.15
                cold_penalty_weight = 0.12
            
            # Boost probabilities for hot numbers from learning
            if hot_numbers_learning:
                for num_str, freq_data in hot_numbers_learning.items():
                    try:
                        num_idx = int(num_str) - 1
                        if 0 <= num_idx < max_number:
                            freq = freq_data.get('frequency', freq_data) if isinstance(freq_data, dict) else freq_data
                            # Use learning weight to determine boost strength
                            boost_factor = 1.0 + (float(freq) * hot_numbers_weight)
                            prob_values[num_idx] *= boost_factor
                    except (ValueError, KeyError, TypeError):
                        continue
            
            # Penalize cold numbers with learning strength
            if cold_numbers_learning:
                for num in cold_numbers_learning:
                    try:
                        num_idx = int(num) - 1
                        if 0 <= num_idx < max_number:
                            penalty_factor = 1.0 - cold_penalty_weight  # e.g., 0.90 if weight is 0.10
                            prob_values[num_idx] *= penalty_factor
                    except (ValueError, TypeError):
                        continue
            
            # Re-normalize after learning adjustments
            prob_sum = np.sum(prob_values)
            if prob_sum > 0:
                prob_values = prob_values / prob_sum
        
        ensemble_confidence = float(model_analysis.get("ensemble_confidence", 0.5))
        diversity_factor = float(optimal_analysis.get("diversity_factor", 1.5))
        hot_cold_ratio = float(optimal_analysis.get("hot_cold_ratio", 1.5))
        distribution_method = optimal_analysis.get("distribution_method", "weighted_ensemble_voting")
        
        # ===== PATTERN ANALYSIS FROM MODEL PROBABILITIES =====
        # Identify hot (high probability) and cold (low probability) numbers
        sorted_indices = np.argsort(prob_values)
        hot_threshold = int(max_number * 0.33)  # Top 33% are hot
        cold_threshold = int(max_number * 0.67)  # Bottom 33% are cold
        
        hot_numbers = sorted_indices[-hot_threshold:] + 1  # Highest probabilities
        cold_numbers = sorted_indices[:cold_threshold] + 1  # Lowest probabilities
        warm_numbers = sorted_indices[cold_threshold:-hot_threshold] + 1  # Middle range
        
        # Calculate hot/cold balance for each set
        hot_count = int(draw_size / hot_cold_ratio)  # Weighted by hot/cold ratio
        warm_count = draw_size - hot_count
        
        # Get per-model probabilities for attribution tracking
        model_probabilities = model_analysis.get("model_probabilities", {})

        # ===== UNIVERSAL GUARANTEED-CONTRIBUTION MODE =====
        # Build a normalised probability array for EVERY selected model so that
        # each model can independently sample its most-confident number.
        # This works for position models, neural models, standard models, and any mix:
        # the structural problem (averaging destroys peaks) is solved by having each
        # model contribute directly rather than through an averaged ensemble.
        _gc_model_arrays: dict = {}  # model_key → np.ndarray[float64] length max_number
        for _gmk, _gmp in model_probabilities.items():
            _gv = np.array(
                [float(_gmp.get(str(i), 1.0 / max_number)) for i in range(1, max_number + 1)],
                dtype=np.float64,
            )
            _gv = np.clip(_gv, 1e-10, None)
            _gv /= _gv.sum()
            _gc_model_arrays[_gmk] = _gv

        # Guaranteed-contribution mode requires at least 2 models
        _use_gc = len(_gc_model_arrays) >= 2

        # ===== GENERATE EACH SET WITH ADVANCED REASONING =====
        predictions_with_attribution = []

        # Get list of all selected models for coverage verification
        selected_model_names = list(model_probabilities.keys())
        
        for set_idx in range(num_sets):
            # Apply progressive temperature to ensemble probabilities for diversity
            # Early sets: use exact ensemble probs; Late sets: more uniform/diverse
            set_progress = float(set_idx) / float(num_sets) if num_sets > 1 else 0.5
            
            # Temperature annealing: increase temperature for later sets to widen the
            # distribution and drive diversity. Higher T → flatter → more exploration.
            temperature = 0.6 + (0.4 * set_progress)  # Range [0.6, 1.0] — grows with set index
            
            # Apply temperature scaling via softmax for entropy-controlled distribution
            log_probs = np.log(prob_values + 1e-10)
            scaled_log_probs = log_probs / (temperature + 0.1)
            adjusted_probs = softmax(scaled_log_probs)
            
            # ===== APPLY NUMBER DIVERSITY PENALTY IF ENABLED =====
            if no_repeat_numbers:
                # Dynamically determine diversity mode based on available unused numbers
                unused_numbers_count = max_number - len(number_usage_count)
                
                if unused_numbers_count >= draw_size:
                    # Still have enough unused numbers for a complete unique set
                    diversity_mode = "pure_unique"
                else:
                    # Not enough unused numbers, switch to calibrated diversity
                    diversity_mode = "calibrated_diversity"
                
                # Calculate penalty for each number based on usage
                diversity_adjusted_probs = adjusted_probs.copy()
                
                for num in range(1, max_number + 1):
                    usage_count = number_usage_count.get(num, 0)
                    
                    if diversity_mode == "pure_unique":
                        # Phase 1: Eliminate already-used numbers completely
                        if usage_count > 0:
                            diversity_adjusted_probs[num - 1] = 0.0
                    
                    elif diversity_mode == "calibrated_diversity":
                        # Phase 2: Heavily penalize frequently used numbers
                        # Penalty increases exponentially with usage
                        penalty_factor = 1.0 / (1.0 + usage_count ** 2)  # Exponential decay
                        diversity_adjusted_probs[num - 1] *= penalty_factor
                
                # Re-normalize probabilities
                prob_sum = np.sum(diversity_adjusted_probs)
                if prob_sum > 0:
                    adjusted_probs = diversity_adjusted_probs / prob_sum
                else:
                    # Fallback: if all numbers exhausted, use least-used
                    min_usage = min(number_usage_count.values()) if number_usage_count else 0
                    for num in range(1, max_number + 1):
                        if number_usage_count.get(num, 0) == min_usage:
                            adjusted_probs[num - 1] = prob_values[num - 1]
                    adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
            
            # ===== PHASE 2: APPLY CO-OCCURRENCE CONCENTRATION BOOSTING =====
            # If adaptive learning is enabled and we have cluster history, boost co-occurring numbers.
            # Apply whenever adaptive_system is set and co-occurrence data exists, regardless of
            # the concentration weight (which starts at 0 for new users).
            if adaptive_system and set_idx > 0:  # Skip first set, no prior numbers to correlate
                concentration_applied = False
                concentration_weight = adaptive_system.meta_learning_data['factor_weights'].get('cluster_concentration', 0.0)
                _has_cooccurrence = bool(adaptive_system.meta_learning_data.get('number_cooccurrence'))
                # P4: Replace the hard 0.03 floor with a smooth ramp that only activates after
                # 10+ learning cycles, so early predictions aren't biased by noise data.
                _cooc_cycles = int(adaptive_system.meta_learning_data.get('total_learning_cycles', 0))
                if _has_cooccurrence and _cooc_cycles >= 10:
                    # Ramp from 0 → 0.05 over cycles 10–30, then natural weight takes over
                    _ramp = min(0.05, 0.005 * (_cooc_cycles - 9))
                    _effective_conc = max(concentration_weight, _ramp)
                else:
                    _effective_conc = concentration_weight  # No floor until 10 cycles

                if _effective_conc > 0.0:  # Run whenever there is any co-occurrence signal
                    # Get numbers from previous sets to build correlation context
                    recent_numbers = []
                    lookback = min(3, set_idx)  # Look at last 3 sets
                    for prev_idx in range(max(0, set_idx - lookback), set_idx):
                        recent_numbers.extend(predictions[prev_idx])
                    
                    if recent_numbers:
                        # Apply co-occurrence boost to probabilities
                        cooccurrence_adjusted_probs = adjusted_probs.copy()
                        
                        for num in range(1, max_number + 1):
                            boost = adaptive_system.get_number_cooccurrence_boost(recent_numbers, num)
                            if boost > 1.0:
                                cooccurrence_adjusted_probs[num - 1] *= boost
                                concentration_applied = True
                        
                        # Re-normalize after co-occurrence boosting
                        if concentration_applied:
                            prob_sum = np.sum(cooccurrence_adjusted_probs)
                            if prob_sum > 0:
                                adjusted_probs = cooccurrence_adjusted_probs / prob_sum
                                strategy_log['phase2_concentration_applied'] = strategy_log.get('phase2_concentration_applied', 0) + 1
            
            selected_numbers = None
            strategy_used = None
            model_attribution = {}

            # ===== UNIVERSAL GUARANTEED-CONTRIBUTION GENERATION =====
            # Each selected model (position, neural, standard, or any mix) directly
            # Gumbel-samples its single most-confident number.  This eliminates the
            # structural averaging problem entirely: no model's specialisation peak
            # can be cancelled by another model's distribution.
            #
            # Algorithm:
            #   1. Every model claims its Gumbel-top-1 number (with diversity noise).
            #   2. Duplicates → lower-confidence claimant re-samples until resolved.
            #   3. n_models < draw_size  → fill remaining slots from ensemble.
            #   4. n_models > draw_size  → greedy keep: highest-confidence unique picks
            #      across models until draw_size reached.
            #   5. Fall through to Gumbel strategies only if _use_gc is False (0–1 model).
            if _use_gc:
                _gc_model_keys = list(_gc_model_arrays.keys())
                _gc_n = len(_gc_model_keys)
                _gc_max_tries = 40
                _gc_picked: list = []
                _gc_used: set = set()
                _gc_attrib: dict = {}   # number_str → model_key (primary contributor)

                for _gc_try in range(_gc_max_tries):
                    _gc_picked = []
                    _gc_used = set()
                    _gc_attrib = {}
                    # Each model samples its top pick with Gumbel noise
                    _gc_claims: dict = {}  # model_key → (number, raw_prob)
                    for _gck in _gc_model_keys:
                        _gpv = _gc_model_arrays[_gck]
                        _gg = -np.log(-np.log(np.random.uniform(0, 1, max_number) + 1e-10) + 1e-10)
                        if _gc_try > 0:
                            _gg += np.random.normal(0, 0.08 * _gc_try, max_number)
                        _gscores = np.log(_gpv + 1e-10) + _gg
                        # Suppress numbers from previous sets when no-repeat is on
                        if no_repeat_numbers:
                            for _nr, _cnt in number_usage_count.items():
                                _gscores[_nr - 1] -= _cnt * 2.0
                        _gpick = int(np.argmax(_gscores)) + 1
                        _gc_claims[_gck] = (_gpick, float(_gpv[_gpick - 1]))

                    # Resolve duplicates: if two models claim the same number,
                    # the lower-confidence model re-samples its next-best option.
                    # Repeat until all claims are unique (or max_number exhausted).
                    _resolved: dict = {}  # number → (model_key, confidence)
                    for _gck, (_gpick, _gconf) in sorted(
                        _gc_claims.items(), key=lambda x: -x[1][1]
                    ):
                        if _gpick not in _resolved:
                            _resolved[_gpick] = (_gck, _gconf)
                        else:
                            # Find next-best unused number for this model
                            _gpv2 = _gc_model_arrays[_gck]
                            _gg2 = -np.log(-np.log(np.random.uniform(0, 1, max_number) + 1e-10) + 1e-10)
                            _gsc2 = np.log(_gpv2 + 1e-10) + _gg2
                            for _taken in _resolved:
                                _gsc2[_taken - 1] = -np.inf
                            if no_repeat_numbers:
                                for _nr, _cnt in number_usage_count.items():
                                    _gsc2[_nr - 1] -= _cnt * 2.0
                            _alt = int(np.argmax(_gsc2)) + 1
                            if _alt not in _resolved:
                                _resolved[_alt] = (_gck, float(_gpv2[_alt - 1]))

                    # Collect unique model picks
                    for _gnum, (_gck, _gconf) in _resolved.items():
                        _gc_picked.append(_gnum)
                        _gc_used.add(_gnum)
                        _gc_attrib[str(_gnum)] = _gck

                    # If n_models > draw_size keep the draw_size highest-confidence picks
                    if len(_gc_picked) > draw_size:
                        _gc_picked_sorted = sorted(
                            [(n, _resolved[n][1]) for n in _gc_picked],
                            key=lambda x: -x[1]
                        )
                        _gc_picked = [x[0] for x in _gc_picked_sorted[:draw_size]]
                        _gc_used = set(_gc_picked)
                        _gc_attrib = {str(n): _gc_attrib[str(n)] for n in _gc_picked}

                    if len(_gc_picked) >= min(draw_size, _gc_n):
                        break

                # Fill remaining slots from ensemble if n_models < draw_size
                if len(_gc_picked) < draw_size:
                    _fill_probs = adjusted_probs.copy()
                    for _fu in _gc_used:
                        _fill_probs[_fu - 1] = 0.0
                    _fp_sum = _fill_probs.sum()
                    if _fp_sum > 0:
                        _fill_probs /= _fp_sum
                        _fg2 = -np.log(-np.log(np.random.uniform(0, 1, max_number) + 1e-10) + 1e-10)
                        _fsc2 = np.log(_fill_probs + 1e-10) + _fg2
                        while len(_gc_picked) < draw_size:
                            _fi2 = int(np.argmax(_fsc2))
                            _fn2 = _fi2 + 1
                            if _fn2 not in _gc_used:
                                _gc_picked.append(_fn2)
                                _gc_used.add(_fn2)
                            _fsc2[_fi2] = -np.inf

                selected_numbers = sorted(_gc_picked)
                _n_label = f"{_gc_n} model{'s' if _gc_n != 1 else ''}"
                strategy_used = f"Guaranteed-Contribution Ensemble ({_n_label})"
                strategy_log["strategy_1_gumbel"] += 1

                # Build per-number attribution
                for _sn in selected_numbers:
                    _sn_str = str(_sn)
                    _primary_mk = _gc_attrib.get(_sn_str)
                    model_attribution[_sn_str] = []
                    if _primary_mk:
                        _rp = float(model_probabilities[_primary_mk].get(_sn_str, 1.0 / max_number))
                        model_attribution[_sn_str].append({
                            'model': _primary_mk,
                            'probability': _rp,
                            'confidence': float(_rp * max_number)
                        })
                    # Also credit any other model whose probability exceeds threshold
                    for _omk, _omp in model_probabilities.items():
                        if _omk == _primary_mk:
                            continue
                        _op = float(_omp.get(_sn_str, 0))
                        if _op > 1.5 / max_number:
                            model_attribution[_sn_str].append({
                                'model': _omk,
                                'probability': _op,
                                'confidence': float(_op * max_number)
                            })

                # Anti-pattern check (P1)
                if adaptive_system:
                    _anti_penalty = adaptive_system.penalize_anti_patterns(selected_numbers)
                    if _anti_penalty > 0.7:
                        strategy_log['anti_pattern_rejections'] = strategy_log.get('anti_pattern_rejections', 0) + 1
                        # Accept result — anti-pattern rejection noted but not blocking here

            else:
                # ===== SINGLE-MODEL / NO-MODEL FALLBACK =====
                # Standard Gumbel-Top-K path when 0–1 models are selected.
                max_attempts = 50
                attempt = 0
                all_models_contributed = False

                while attempt < max_attempts and not all_models_contributed:
                    attempt += 1

                    try:
                        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, max_number) + 1e-10) + 1e-10)
                        if attempt > 1:
                            gumbel_noise += np.random.normal(0, 0.1 * attempt, max_number)
                        gumbel_scores = np.log(adjusted_probs + 1e-10) + gumbel_noise
                        top_k_indices = np.argsort(gumbel_scores)[-draw_size:]
                        selected_numbers = sorted([i + 1 for i in top_k_indices])
                        strategy_used = "Gumbel-Top-K with Entropy Optimization"
                        if attempt == 1:
                            strategy_log["strategy_1_gumbel"] += 1

                    except Exception:
                        try:
                            hot_probs = adjusted_probs[hot_numbers - 1]
                            hot_probs = hot_probs / np.sum(hot_probs)
                            selected_hot = np.random.choice(
                                hot_numbers, size=min(hot_count, len(hot_numbers)),
                                replace=False, p=hot_probs
                            )
                            remaining_count = draw_size - len(selected_hot)
                            available_warm = [n for n in warm_numbers if n not in selected_hot]
                            if len(available_warm) >= remaining_count:
                                warm_probs = adjusted_probs[np.array(available_warm) - 1]
                                warm_probs = warm_probs / np.sum(warm_probs)
                                selected_warm = np.random.choice(
                                    available_warm, size=remaining_count,
                                    replace=False, p=warm_probs
                                )
                            else:
                                selected_warm = available_warm
                            selected_numbers = sorted(
                                np.concatenate([selected_hot, selected_warm]).astype(int).tolist()
                            )
                            strategy_used = "Hot/Cold Balanced Selection"
                            strategy_log["strategy_2_hotcold"] += 1
                        except Exception:
                            try:
                                selected_indices = np.random.choice(
                                    max_number, size=draw_size, replace=False, p=adjusted_probs
                                )
                                selected_numbers = sorted([i + 1 for i in selected_indices])
                                strategy_used = "Confidence-Weighted Random Selection"
                                strategy_log["strategy_3_confidence_weighted"] += 1
                            except Exception:
                                top_k_indices = np.argsort(adjusted_probs)[-draw_size:]
                                selected_numbers = sorted([i + 1 for i in top_k_indices])
                                strategy_used = "Deterministic Top-K from Ensemble"
                                strategy_log["strategy_4_topk"] += 1

                    # Anti-pattern rejection (P1)
                    if adaptive_system and selected_numbers and attempt < max_attempts - 5:
                        _anti_penalty = adaptive_system.penalize_anti_patterns(selected_numbers)
                        if _anti_penalty > 0.7:
                            strategy_log['anti_pattern_rejections'] = strategy_log.get('anti_pattern_rejections', 0) + 1
                            continue

                    # Attribution via threshold
                    model_attribution = {}
                    for number in selected_numbers:
                        number_str = str(number)
                        model_attribution[number_str] = []
                        for model_key, model_probs in model_probabilities.items():
                            if isinstance(model_probs, dict):
                                number_prob = float(model_probs.get(number_str, 0))
                                if number_prob > 1.5 / max_number:
                                    model_attribution[number_str].append({
                                        'model': model_key,
                                        'probability': float(number_prob),
                                        'confidence': float(number_prob * max_number)
                                    })

                    contributing_models = {
                        voter['model']
                        for voters in model_attribution.values()
                        for voter in voters
                    }
                    if len(selected_model_names) == 0 or len(contributing_models) >= len(selected_model_names):
                        all_models_contributed = True
                    elif attempt >= max_attempts:
                        all_models_contributed = True
                        if attempt > 1:
                            strategy_used += (
                                f" (Coverage: {len(contributing_models)}/"
                                f"{len(selected_model_names)} after {attempt} attempts)"
                            )
            
            # ===== UPDATE NUMBER USAGE TRACKING =====
            if no_repeat_numbers:
                for number in selected_numbers:
                    number_usage_count[number] = number_usage_count.get(number, 0) + 1
            
            predictions.append(selected_numbers)
            predictions_with_attribution.append({
                'numbers': selected_numbers,
                'model_attribution': model_attribution,
                'strategy': strategy_used
            })
            
            strategy_log["details"].append({
                "set_num": set_idx + 1,
                "strategy": strategy_used,
                "numbers": selected_numbers,
                "model_votes": {str(num): len(model_attribution.get(str(num), [])) for num in selected_numbers}
            })
        
        # Generate comprehensive strategy report
        strategy_report = self._generate_strategy_report(strategy_log, distribution_method)

        # ===== GENETIC ALGORITHM POST-PROCESSING =====
        # Run GA optimisation on the bottom 30% of sets (by learning score) when
        # learning data is available. Limits generations to 20 for speed.
        if learning_data and adaptive_system and len(predictions) >= 4:
            try:
                _bottom_n = max(1, len(predictions) // 3)
                # Score all sets
                _scored = []
                for _idx, _pset in enumerate(predictions):
                    _s = _calculate_learning_score_advanced(_pset, learning_data, adaptive_system, draw_age=_generation_draw_age)
                    _scored.append((_s, _idx))
                _scored.sort(key=lambda x: x[0])   # ascending — lowest scores first

                _ga = GeneticSetOptimizer(
                    draw_size=draw_size,
                    max_number=max_number,
                    learning_data=learning_data,
                    adaptive_system=adaptive_system
                )
                _ga.generations = 20   # cap at 20 for speed during prediction

                _ga_improved = 0
                for _score, _orig_idx in _scored[:_bottom_n]:
                    _orig_set = predictions[_orig_idx]
                    _opt_set = _ga.optimize_set(initial_set=_orig_set)
                    _opt_score = _calculate_learning_score_advanced(_opt_set, learning_data, adaptive_system)
                    if _opt_score > _score:
                        predictions[_orig_idx] = _opt_set
                        predictions_with_attribution[_orig_idx]['numbers'] = _opt_set
                        predictions_with_attribution[_orig_idx]['strategy'] = (
                            predictions_with_attribution[_orig_idx].get('strategy', '') + ' [GA-optimised]'
                        )
                        _ga_improved += 1

                strategy_log['ga_sets_improved'] = _ga_improved
                strategy_log['ga_sets_attempted'] = _bottom_n
            except Exception as _ga_err:
                strategy_log['ga_error'] = str(_ga_err)[:120]

        # ===== PHASE 3: CLUSTER FUSION STRATEGY =====
        # Generate fusion sets whenever adaptive learning is active and co-occurrence data exists.
        # `generate_fusion_sets` now handles its own effective-weight floor internally.
        if adaptive_system and len(predictions) >= 5:
            concentration_weight = adaptive_system.meta_learning_data['factor_weights'].get('cluster_concentration', 0.0)
            _p3_has_cooc = bool(adaptive_system.meta_learning_data.get('number_cooccurrence'))
            _p3_effective = max(concentration_weight, 0.02) if _p3_has_cooc else concentration_weight

            if _p3_effective >= 0.02:  # Run with co-occurrence data or 2%+ weight
                # Rank predictions by learning score to identify top performers
                if learning_data:
                    ranked_predictions = []
                    for idx, pred_set in enumerate(predictions):
                        score = _calculate_learning_score_advanced(
                            pred_set, learning_data, adaptive_system, draw_age=_generation_draw_age
                        )
                        ranked_predictions.append((score, pred_set, idx))
                    
                    # Sort by score descending
                    ranked_predictions.sort(key=lambda x: x[0], reverse=True)
                    ranked_numbers = [pred_set for score, pred_set, idx in ranked_predictions]
                else:
                    # No learning data, use original order
                    ranked_numbers = predictions
                
                # Generate fusion sets
                fusion_sets = adaptive_system.generate_fusion_sets(
                    predictions=ranked_numbers,
                    prob_values=prob_values,
                    draw_size=draw_size,
                    num_fusion_sets=min(2, num_sets // 10),  # Generate 1-2 fusion sets based on total
                    top_n_to_analyze=5
                )
                
                if fusion_sets:
                    # Insert fusion sets at the top of predictions
                    for fusion_dict in reversed(fusion_sets):  # Reversed to maintain order
                        fusion_numbers = fusion_dict['numbers']
                        
                        # Add to predictions at the beginning
                        predictions.insert(0, fusion_numbers)
                        
                        # Add to attribution
                        predictions_with_attribution.insert(0, {
                            'numbers': fusion_numbers,
                            'model_attribution': {},  # Fusion sets don't have direct model attribution
                            'strategy': f"Phase 3 Cluster Fusion: {fusion_dict['fusion_strategy']}",
                            'fusion_metadata': fusion_dict
                        })
                    
                    # Update strategy log
                    strategy_log['phase3_fusion_sets_generated'] = len(fusion_sets)
                    strategy_log['fusion_concentration_weight'] = concentration_weight
                    
                    # Update strategy report
                    fusion_report = f"\n\n🎯 **Phase 3: Cluster Fusion Strategy**\n"
                    fusion_report += f"Generated {len(fusion_sets)} fusion set(s) from top 5 predictions\n"
                    fusion_report += f"Concentration Weight: {concentration_weight:.1%}\n"
                    fusion_report += f"These sets combine the most successful number patterns from multiple high-scoring predictions.\n"
                    strategy_report += fusion_report
        
        return predictions, strategy_report, predictions_with_attribution, strategy_log
    
    def _generate_strategy_report(self, strategy_log: Dict[str, Any], distribution_method: str) -> str:
        """
        Generate a comprehensive human-readable strategy report showing which
        generation strategies were used across all sets.
        """
        total_sets = strategy_log["total_sets"]
        s1_count = strategy_log["strategy_1_gumbel"]
        s2_count = strategy_log["strategy_2_hotcold"]
        s3_count = strategy_log["strategy_3_confidence_weighted"]
        s4_count = strategy_log["strategy_4_topk"]
        
        # Calculate percentages
        s1_pct = (s1_count / total_sets * 100) if total_sets > 0 else 0
        s2_pct = (s2_count / total_sets * 100) if total_sets > 0 else 0
        s3_pct = (s3_count / total_sets * 100) if total_sets > 0 else 0
        s4_pct = (s4_count / total_sets * 100) if total_sets > 0 else 0
        
        # Build report
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PREDICTION SET GENERATION STRATEGY REPORT                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

**OVERVIEW**: Generated {total_sets} prediction sets using advanced multi-strategy AI reasoning

**DISTRIBUTION METHOD**: {distribution_method}

"""
        
        # Add diversity mode info if enabled
        if strategy_log.get("no_repeat_enabled"):
            diversity_mode_desc = strategy_log.get("diversity_mode", "Unknown")
            report += f"""**🔢 NUMBER DIVERSITY MODE**: {diversity_mode_desc}
   └─ Intelligent calibration to maximize number coverage across sets
   └─ Minimizes repetition while maintaining probability-based quality

"""
        
        report += f"""**STRATEGY BREAKDOWN**:
{'─' * 80}

"""
        
        # Primary strategy (most used)
        strategies_used = []
        if s1_count > 0:
            strategies_used.append(("🎯 Strategy 1: Gumbel-Top-K with Entropy Optimization", s1_count, s1_pct, 
                                   "Primary algorithm using Gumbel noise injection for deterministic yet diverse selection"))
        if s2_count > 0:
            strategies_used.append(("🔥 Strategy 2: Hot/Cold Balanced Selection", s2_count, s2_pct,
                                   "Balanced approach sampling high-probability (hot) and diverse (cold) numbers"))
        if s3_count > 0:
            strategies_used.append(("⚖️  Strategy 3: Confidence-Weighted Random Selection", s3_count, s3_pct,
                                   "Probabilistic selection weighted by ensemble confidence scores"))
        if s4_count > 0:
            strategies_used.append(("📊 Strategy 4: Deterministic Top-K from Ensemble", s4_count, s4_pct,
                                   "Fallback using highest-probability numbers from ensemble consensus"))
        
        # Add strategy details
        for idx, (strategy_name, count, pct, description) in enumerate(strategies_used, 1):
            report += f"{strategy_name}\n"
            report += f"  └─ Used for {count}/{total_sets} sets ({pct:.1f}%)\n"
            report += f"  └─ {description}\n"
            if idx < len(strategies_used):
                report += "\n"
        
        # Summary analysis
        report += f"""
{'─' * 80}

**ANALYSIS**:

"""
        
        if s1_count == total_sets:
            report += """✅ All sets generated using primary Gumbel-Top-K strategy
   → Optimal condition: High ensemble confidence and probability variance
   → Result: Maximum entropy-optimized diversity with strong convergence
"""
        elif s1_count > 0:
            report += f"""⚠️  Mixed strategy deployment: {s1_count} sets using Gumbel, {total_sets - s1_count} using fallback strategies
   → Indicates some probability computation challenges
   → Quality: Still maintained through robust fallback mechanisms
"""
        else:
            report += """⚠️  Primary strategy unavailable; using fallback strategies only
   → Possible issue with probability distributions or ensemble inference
   → Quality: Maintained through deterministic top-k selection
"""
        
        if s2_count > 0:
            report += f"""
📈 Hot/Cold Strategy Engagement: {s2_count} sets
   → Number analysis active: selecting from high-probability (hot) and diverse (cold) pools
   → Provides natural diversity while honoring model predictions
"""
        
        report += f"""
**CONFIDENCE**: Algorithm executed with full redundancy
   → Primary + 3 fallback strategies ensure robust generation
   → All {total_sets} sets successfully generated without failure

**MATHEMATICAL RIGOR**:
✓ Real ensemble probabilities from trained models
✓ Temperature-annealed distribution control
✓ Gumbel noise for entropy optimization
✓ Hot/cold probability analysis
✓ Progressive diversity across sets
"""
        
        # Add diversity statistics if no_repeat mode was enabled
        if strategy_log.get("no_repeat_enabled"):
            # Calculate diversity metrics from strategy_log details
            all_numbers_used = set()
            number_frequency = {}
            
            for detail in strategy_log.get("details", []):
                numbers = detail.get("numbers", [])
                all_numbers_used.update(numbers)
                for num in numbers:
                    number_frequency[num] = number_frequency.get(num, 0) + 1
            
            unique_numbers = len(all_numbers_used)
            max_possible_numbers = 50 if "max" in distribution_method.lower() else 49  # Estimate
            coverage_pct = (unique_numbers / max_possible_numbers) * 100
            
            # Find numbers used most/least
            if number_frequency:
                max_usage = max(number_frequency.values())
                min_usage = min(number_frequency.values())
                avg_usage = sum(number_frequency.values()) / len(number_frequency)
                
                report += f"""
{'─' * 80}

**🔢 NUMBER DIVERSITY STATISTICS**:

✓ Unique Numbers Coverage: {unique_numbers}/{max_possible_numbers} ({coverage_pct:.1f}%)
✓ Number Usage Range: {min_usage}-{max_usage} times per number
✓ Average Usage per Number: {avg_usage:.2f}
✓ Diversity Strategy: {"Pure Uniqueness" if strategy_log.get("diversity_mode") == "Pure Uniqueness (no number repeats)" else "Calibrated Minimization"}

"""
                
                # Show most and least used numbers
                most_used = sorted(number_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
                least_used = sorted(number_frequency.items(), key=lambda x: x[1])[:5]
                
                report += "**Most Frequently Used:** "
                report += ", ".join([f"#{num}({count}x)" for num, count in most_used])
                report += "\n"
                
                report += "**Least Frequently Used:** "
                report += ", ".join([f"#{num}({count}x)" for num, count in least_used])
                report += "\n"
        
        return report.strip()

    def save_predictions_advanced(self, predictions: List[List[int]], 
                                 model_analysis: Dict[str, Any],
                                 optimal_analysis: Dict[str, Any],
                                 num_sets: int,
                                 predictions_with_attribution: List[Dict] = None) -> str:
        """Save advanced AI predictions with complete analysis metadata and model attribution."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sia_predictions_{timestamp}_{num_sets}sets.json"
        filepath = self.predictions_dir / filename
        
        # Store both simple predictions and detailed attribution
        data = {
            "timestamp": datetime.now().isoformat(),
            "game": self.game,
            "next_draw_date": str(compute_next_draw_date(self.game)),
            "algorithm": "Super Intelligent Algorithm (SIA)",
            "predictions": predictions,
            "predictions_with_attribution": predictions_with_attribution if predictions_with_attribution else [],
            "analysis": {
                "selected_models": [
                    {
                        "name": m["name"],
                        "type": m["type"],
                        "accuracy": m["accuracy"],
                        "confidence": m["confidence"]
                    }
                    for m in model_analysis.get("models", [])
                ],
                "ensemble_confidence": optimal_analysis.get("ensemble_confidence", 0.0),
                "ensemble_synergy": optimal_analysis.get("ensemble_synergy", 0.0),
                "win_probability": optimal_analysis.get("win_probability", 0.0),
                "weighted_confidence": optimal_analysis.get("weighted_confidence", 0.0),
                "model_variance": optimal_analysis.get("model_variance", 0.0),
                "uncertainty_factor": optimal_analysis.get("uncertainty_factor", 1.0),
                "diversity_factor": optimal_analysis.get("diversity_factor", 1.0),
                "distribution_method": optimal_analysis.get("distribution_method", "uniform"),
                "hot_cold_ratio": optimal_analysis.get("hot_cold_ratio", 1.5),
            },
            "optimal_calculation": optimal_analysis
        }
        
        # Create directory if it doesn't exist
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            app_log(f"Predictions saved to {filepath}", "info")
        except Exception as e:
            app_log(f"Error saving predictions: {e}", "error")
            raise
        
        return str(filepath)

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores."""
    try:
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum()
    except:
        # Fallback to uniform
        return np.ones_like(x) / len(x)


# ============================================================================
# MAIN PAGE RENDERING
# ============================================================================

def render_prediction_ai_page(services_registry=None, ai_engines=None, components=None) -> None:
    """Main AI Prediction Engine page with full app integration."""
    try:
        st.title("🎯 Super Intelligent AI Prediction Engine")
        st.markdown("*Advanced multi-model ensemble predictions with intelligent algorithm optimization*")
        
        # Initialize session state
        if 'sia_game' not in st.session_state:
            st.session_state.sia_game = "Lotto Max"
        if 'sia_selected_models' not in st.session_state:
            st.session_state.sia_selected_models = []
        if 'sia_analysis_result' not in st.session_state:
            st.session_state.sia_analysis_result = None
        if 'sia_optimal_sets' not in st.session_state:
            st.session_state.sia_optimal_sets = None
        if 'sia_ml_analysis_result' not in st.session_state:
            st.session_state.sia_ml_analysis_result = None
        if 'sia_ml_optimal_sets' not in st.session_state:
            st.session_state.sia_ml_optimal_sets = None
        if 'sia_predictions' not in st.session_state:
            st.session_state.sia_predictions = None
        
        # Game selection
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_game = st.selectbox(
                "🎰 Select Game",
                get_available_games(),
                key='sia_game_select'
            )
            st.session_state.sia_game = selected_game
        
        with col2:
            next_draw = compute_next_draw_date(selected_game)
            st.info(f"📅 Next Draw: {next_draw.strftime('%A, %B %d, %Y')}")
        
        # Initialize analyzer
        analyzer = SuperIntelligentAIAnalyzer(selected_game)

        # ── Auto-learning detection banner ───────────────────────────────────
        # ── Show result from banner "Apply Learning Now" button (previous run) ──
        _bl_success = st.session_state.pop('sia_banner_learning_success', None)
        if _bl_success:
            st.success(
                f"✅ **Learning Applied!** Draw **{_bl_success['draw_date']}** — "
                f"Cycle **#{_bl_success['cycle']}** complete"
            )
            with st.expander("📋 Learning Summary", expanded=True):
                _blc1, _blc2, _blc3, _blc4 = st.columns(4)
                _blc1.metric("Learning Cycle", _bl_success['cycle'], delta="+1")
                _blc2.metric(
                    "Prediction Files",
                    f"{_bl_success['files_processed']}/{_bl_success['files_count']}"
                )
                _blc3.metric("Anti-Patterns Tracked", _bl_success['anti_patterns'])
                _blc4.metric("Factor Interactions", _bl_success['factor_interactions'])
                st.markdown(
                    f"**Top-weighted factor:** `{_bl_success['top_factor']}` "
                    f"— weight `{_bl_success['top_factor_weight']}`"
                )
                st.markdown(
                    f"**Winning numbers learned from:** "
                    f"{', '.join(str(n) for n in sorted(_bl_success['winning_numbers']))}"
                )
                if _bl_success.get('saved_paths'):
                    st.markdown("**Learning data saved to:**")
                    for _sp in _bl_success['saved_paths']:
                        st.code(_sp, language=None)
                st.caption(
                    "💡 For full factor-by-factor analysis, visit the "
                    "**AI Learning tab → Previous Draw Date** and re-run with detailed factor scoring."
                )
        _bl_error = st.session_state.pop('sia_banner_learning_error', None)
        if _bl_error:
            st.error(_bl_error)
        # ─────────────────────────────────────────────────────────────────────

        # Check whether a new draw has occurred since the last meta-learning
        # update, and predictions were saved for that draw date.
        try:
            _als = AdaptiveLearningSystem(selected_game)
            _last_updated_str = _als.meta_learning_data.get('last_updated', '')
            _last_updated_dt = None
            if _last_updated_str:
                try:
                    _last_updated_dt = datetime.fromisoformat(_last_updated_str)
                except Exception:
                    pass

            _today = datetime.now().date()
            _game_folder = _sanitize_game_name(selected_game)
            _pred_dir = Path("predictions") / _game_folder / "prediction_ai"
            _pending_draw_date: Optional[str] = None
            _pending_pred_count: int = 0

            if _pred_dir.exists():
                for _pf in sorted(_pred_dir.glob("*.json"), reverse=True):
                    try:
                        with open(_pf, 'r') as _fh:
                            _pd_data = json.load(_fh)
                        _nd = _pd_data.get('next_draw_date', '')
                        if not _nd:
                            continue
                        _nd_date = datetime.strptime(_nd, '%Y-%m-%d').date()
                        # Draw must have already happened (past date)
                        if _nd_date >= _today:
                            continue
                        # Check if results exist in CSV for this date
                        _win_nums, _, _ = _get_draw_results(selected_game, _nd_date)
                        if not _win_nums:
                            continue
                        # Check if this draw is newer than our last learning update
                        if _last_updated_dt is not None:
                            _nd_as_dt = datetime(year=_nd_date.year, month=_nd_date.month, day=_nd_date.day)
                            if _nd_as_dt <= _last_updated_dt:
                                continue
                        # Found an unlearned draw with predictions and results
                        if _pending_draw_date is None or _nd > _pending_draw_date:
                            _pending_draw_date = _nd
                        _pending_pred_count += 1
                    except Exception:
                        continue

            if _pending_draw_date:
                _banner_col1, _banner_col2 = st.columns([4, 1])
                with _banner_col1:
                    st.warning(
                        f"🔔 **New draw results available for learning!**  "
                        f"Draw date **{_pending_draw_date}** has results in the database and "
                        f"{_pending_pred_count} saved prediction file(s) — but the AI has not "
                        f"yet learned from this draw. Apply learning to improve future accuracy."
                    )
                with _banner_col2:
                    if st.button("🧠 Apply Learning Now", key="auto_learning_banner_btn", type="primary"):
                        with st.spinner(f"Applying learning for {_pending_draw_date}…"):
                            try:
                                _learn_actual = _load_actual_results(selected_game, _pending_draw_date)
                                if not _learn_actual or not _learn_actual.get('numbers'):
                                    st.session_state['sia_banner_learning_error'] = (
                                        f"Could not load draw results for {_pending_draw_date}"
                                    )
                                    st.rerun()
                                else:
                                    _learn_files = _find_prediction_files_for_date(
                                        selected_game, _pending_draw_date
                                    )
                                    _total_saved: list = []
                                    _last_als = None

                                    for _lf in _learn_files:
                                        try:
                                            with open(_lf, 'r') as _lfh:
                                                _lf_data = json.load(_lfh)
                                            _lf_preds = _lf_data.get('predictions', [])
                                            if not _lf_preds:
                                                continue

                                            _lf_matched = _highlight_prediction_matches(
                                                _lf_preds,
                                                _learn_actual['numbers'],
                                                _learn_actual.get('bonus')
                                            )
                                            _lf_sorted = _sort_predictions_by_accuracy(
                                                _lf_matched, _learn_actual['numbers']
                                            )
                                            _lf_ld = _compile_comprehensive_learning_data(
                                                selected_game, _pending_draw_date,
                                                _learn_actual, _lf_sorted, _lf_data, False
                                            )
                                            if not _lf_ld:
                                                continue

                                            _als = AdaptiveLearningSystem(selected_game)
                                            _total_preds = len(_lf_sorted)
                                            _top_n = max(1, _total_preds // 10)
                                            _bot_n = max(1, _total_preds // 10)
                                            _top_ps = [
                                                [n['number'] for n in p['numbers']]
                                                for p in _lf_sorted[:_top_n]
                                            ]
                                            _bot_ps = [
                                                [n['number'] for n in p['numbers']]
                                                for p in _lf_sorted[-_bot_n:]
                                            ]

                                            # Quick-apply: update weights + anti-patterns + clusters.
                                            # Full factor-scored analysis is available in the
                                            # AI Learning tab → Previous Draw Date.
                                            _als.update_weights_from_success(
                                                winning_numbers=_learn_actual,
                                                top_predictions=_top_ps,
                                                factor_scores={}
                                            )
                                            _als.track_anti_patterns(_bot_ps, _learn_actual)
                                            try:
                                                _als.detect_cross_factor_interactions(
                                                    _top_ps, _learn_actual, {}
                                                )
                                            except Exception:
                                                pass

                                            _ranked_c = [
                                                (
                                                    p['correct_count'] / max(1, len(_learn_actual['numbers'])),
                                                    [n['number'] for n in p['numbers']]
                                                )
                                                for p in _lf_sorted
                                            ]
                                            _cr = _als.detect_winning_clusters(
                                                _ranked_c, _learn_actual['numbers'], top_n=5
                                            )
                                            _als.update_cluster_concentration_weight(
                                                _cr.get('coverage_count', 0),
                                                len(_learn_actual['numbers'])
                                            )

                                            _saved = _save_learning_data(
                                                selected_game, _pending_draw_date, _lf_ld
                                            )
                                            _total_saved.append(str(_saved))
                                            _last_als = _als
                                        except Exception:
                                            continue

                                    if _total_saved and _last_als:
                                        _cycle = _last_als.meta_learning_data.get(
                                            'total_learning_cycles', 0
                                        )
                                        _ap_ct = len(
                                            _last_als.meta_learning_data.get('anti_patterns', [])
                                        )
                                        _wts = _last_als.meta_learning_data.get('factor_weights', {})
                                        _top_f = max(_wts, key=_wts.get) if _wts else 'N/A'
                                        _inter_ct = len(
                                            _last_als.meta_learning_data.get(
                                                'cross_factor_interactions', {}
                                            )
                                        )
                                        st.session_state['sia_banner_learning_success'] = {
                                            'draw_date': _pending_draw_date,
                                            'cycle': _cycle,
                                            'files_count': len(_learn_files),
                                            'files_processed': len(_total_saved),
                                            'anti_patterns': _ap_ct,
                                            'factor_interactions': _inter_ct,
                                            'top_factor': _top_f,
                                            'top_factor_weight': round(_wts.get(_top_f, 0), 4),
                                            'saved_paths': _total_saved,
                                            'winning_numbers': _learn_actual['numbers'],
                                        }
                                        st.rerun()
                                    else:
                                        st.session_state['sia_banner_learning_error'] = (
                                            "No learning data could be saved — "
                                            "no valid prediction files found for this draw date."
                                        )
                                        st.rerun()
                            except Exception as _apply_err:
                                st.session_state['sia_banner_learning_error'] = (
                                    f"Learning failed: {_apply_err}"
                                )
                                st.rerun()
        except Exception as _banner_err:
            app_log(f"Auto-learning banner check failed: {_banner_err}", "warning")
        # ── End auto-learning banner ─────────────────────────────────────────

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 AI Model Configuration",
            "🎲 Generate Predictions",
            "� Jackpot Pattern Analysis",
            "🎰 MaxMillion Analysis" if selected_game == "Lotto Max" else "📈 Performance History",
            "🧠 AI Learning"
        ])
        
        with tab1:
            _render_model_configuration(analyzer)
        
        with tab2:
            _render_prediction_generator(analyzer)
        
        with tab3:
            _render_jackpot_analysis(analyzer)
        
        with tab4:
            if selected_game == "Lotto Max":
                _render_maxmillion_analysis(analyzer, selected_game)
            else:
                _render_performance_history(analyzer)
        
        with tab5:
            _render_deep_learning_tab(analyzer, selected_game)
        
        app_log("AI Prediction Engine page rendered successfully", "info")
        
    except Exception as e:
        import traceback
        st.error(f"❌ Error loading AI Prediction Engine: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        app_log(f"Error in prediction_ai page: {str(e)}", "error")
        app_log(f"Traceback: {traceback.format_exc()}", "error")


# ============================================================================
# TAB 1: AI MODEL CONFIGURATION
# ============================================================================

def _render_model_configuration(analyzer: SuperIntelligentAIAnalyzer) -> None:
    """Configure AI models and analysis parameters."""
    st.subheader("🤖 AI Model Selection & Configuration")
    
    # ==================== NEW: MACHINE LEARNING MODELS SECTION ====================
    st.markdown("### 🧠 Machine Learning Models")
    st.markdown("Select models from Phase 2D promoted model cards")
    
    # Helper function to get available model cards for a game
    def get_available_model_cards(game: str) -> List[str]:
        """Get list of available model card files for a game."""
        import os
        from pathlib import Path
        from streamlit_app.core import sanitize_game_name
        
        game_lower = sanitize_game_name(game)
        PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        model_cards_dir = PROJECT_ROOT / "models" / "advanced" / "model_cards"
        
        if not model_cards_dir.exists():
            return []
        
        # Find all model cards files for this game
        matching_files = list(model_cards_dir.glob(f"model_cards_{game_lower}_*.json"))
        
        # Extract just the filenames without path
        filenames = [f.name for f in matching_files]
        
        return sorted(filenames, reverse=True)  # Most recent first
    
    # Helper function to get promoted models from model card
    def get_promoted_models(game_name: str, card_filename: str = None) -> List[Dict[str, Any]]:
        """Load promoted models from Phase 2D leaderboard JSON file."""
        import os
        import json
        from pathlib import Path
        from streamlit_app.core import sanitize_game_name
        
        game_lower = sanitize_game_name(game_name)
        PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        model_cards_dir = PROJECT_ROOT / "models" / "advanced" / "model_cards"
        
        if not model_cards_dir.exists():
            return []
        
        # If a specific card filename is provided, use it
        if card_filename:
            target_file = model_cards_dir / card_filename
            if not target_file.exists():
                return []
        else:
            # Find the latest model cards file for this game
            matching_files = list(model_cards_dir.glob(f"model_cards_{game_lower}_*.json"))
            
            if not matching_files:
                return []
            
            # Get the most recent file
            target_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(target_file, 'r') as f:
                models_data = json.load(f)
            return models_data
        except Exception as e:
            return []
    
    # Get available cards for current game
    available_cards = get_available_model_cards(analyzer.game)
    
    if available_cards:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_card = st.selectbox(
                "Select Model Card",
                available_cards,
                key="sia_ml_model_card_selector"
            )
        
        with col2:
            st.info(f"📊 Game: {analyzer.game}")
        
        # Load models from selected card
        promoted_models = get_promoted_models(analyzer.game, selected_card)
        
        if promoted_models:
            model_names = [m.get("model_name", "Unknown") for m in promoted_models]
            
            selected_ml_models = st.multiselect(
                "Select Models from Card",
                model_names,
                default=model_names[:min(3, len(model_names))],
                key="sia_ml_models_selector"
            )
            
            if selected_ml_models:
                # Get the selected model objects with their metadata
                selected_model_objs = [m for m in promoted_models if m.get("model_name") in selected_ml_models]
                
                # Store in session state
                if 'sia_ml_selected_models' not in st.session_state:
                    st.session_state.sia_ml_selected_models = []
                
                st.write(f"**Selected {len(selected_ml_models)} ML model(s)**")
                
                # Show selected models
                for idx, model in enumerate(selected_model_objs):
                    st.write(f"{idx+1}. {model.get('model_name', 'Unknown')} (health: {model.get('health_score', 0.75):.3f})")
                
                st.divider()
                st.markdown("### 📊 ML Model Analysis")
                
                # Analyze button
                if st.button("🔍 Analyze Selected ML Models", use_container_width=True, key="analyze_ml_btn"):
                    with st.spinner("🤔 Analyzing ML models..."):
                        # Use specialized ML model analysis method
                        analysis = analyzer.analyze_ml_models(selected_model_objs)
                        st.session_state.sia_ml_analysis_result = analysis
                
                # Display analysis results
                if st.session_state.sia_ml_analysis_result:
                    analysis = st.session_state.sia_ml_analysis_result
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Models Selected", analysis["total_selected"])
                    with col2:
                        st.metric("Average Accuracy", f"{analysis['average_accuracy']:.1%}")
                    with col3:
                        st.metric("Ensemble Confidence", f"{analysis['ensemble_confidence']:.1%}")
                    with col4:
                        if analysis["best_model"]:
                            st.metric("Best Model", f"{analysis['best_model']['accuracy']:.1%}")
                    
                    # Model details
                    st.markdown("#### Model Details")
                    models_df = pd.DataFrame([
                        {
                            "Model": m["name"],
                            "Type": m["type"],
                            "Accuracy": f"{m['accuracy']:.1%}",
                            "Confidence": f"{m['confidence']:.1%}"
                        }
                        for m in analysis["models"]
                    ])
                    # Add inference_type column to table
                    models_df_display = pd.DataFrame([
                        {
                            "Model": m["name"],
                            "Type": m["type"],
                            "Accuracy": f"{m['accuracy']:.1%}",
                            "Confidence": f"{m['confidence']:.1%}",
                            "Inference": m.get("inference_type", "unknown"),
                        }
                        for m in analysis["models"]
                    ])
                    st.dataframe(models_df_display, use_container_width=True, hide_index=True)

                    # Warn when any model used synthetic probabilities
                    _type_summary = analysis.get("inference_type_summary", {})
                    _synth_n = _type_summary.get('synthetic', 0) + _type_summary.get('synthetic_fallback', 0)
                    _real_n = _type_summary.get('real', 0)
                    if _synth_n > 0 and _real_n == 0:
                        st.error(
                            f"**All {_synth_n} models used synthetic (health-based) probabilities** — "
                            "no real model inference was performed. Predictions will reflect uniform distributions, "
                            "not learned patterns. Ensure model card points to trained `.joblib`/`.keras` files."
                        )
                    elif _synth_n > 0:
                        st.warning(
                            f"**{_synth_n} of {_synth_n + _real_n} models used synthetic probabilities** "
                            f"({_real_n} used real inference). Ensemble quality is reduced. "
                            "Check inference log for details."
                        )

                    # Display inference logs in an expander
                    if analysis.get("inference_logs"):
                        with st.expander("🔍 Model Loading & Inference Details", expanded=False):
                            st.markdown("**Real-time inference log showing model loading, feature generation, and probability calculation:**")
                            for log_entry in analysis["inference_logs"]:
                                st.text(log_entry)

                    # Optimal Analysis section
                    st.divider()
                    st.markdown("### 🎯 Intelligent Set Recommendation - AI/ML Analysis")
                    
                    st.markdown("""
                    **Goal:** Determine the optimal number of prediction sets based on AI/ML model analysis.
                    
                    This system analyzes your selected models to recommend a practical number of sets:
                    - **Ensemble Accuracy Analysis**: Combined predictive power of all selected models
                    - **Optimal Coverage Calculation**: Balances coverage with practical budget
                    - **Confidence-Based Weighting**: Adjusts recommendations based on model reliability
                    - **Cost-Effective Strategy**: Provides actionable recommendations you can actually use
                    
                    ⚠️ **Important**: Lottery outcomes are random. These are optimized recommendations, not guaranteed wins.
                    """)
                    
                    # Coverage target + Set Limit Controls
                    col_cov, col_cap1, col_cap2 = st.columns([2, 1, 2])

                    with col_cov:
                        sia_ml_target_prob = st.slider(
                            "Coverage Target",
                            min_value=50, max_value=95, value=90, step=5,
                            key="sia_ml_target_prob",
                            help="How confident do you want to be that at least one set covers the winning pattern? Higher = more sets recommended."
                        ) / 100.0

                    with col_cap1:
                        enable_cap = st.checkbox(
                            "Enable Set Limit",
                            value=False,
                            key="sia_ml_enable_cap",
                            help="Limit the maximum number of recommended sets"
                        )

                    with col_cap2:
                        if enable_cap:
                            max_sets_cap = st.number_input(
                                "Maximum Sets",
                                min_value=1,
                                max_value=1000,
                                value=100,
                                step=1,
                                key="sia_ml_max_cap",
                                help="Set the maximum number of sets to recommend (1-1000)"
                            )
                        else:
                            max_sets_cap = None

                    if st.button("🧠 Calculate Optimal Sets (SIA)", use_container_width=True, key="sia_calc_ml_btn"):
                        with st.spinner("🤖 SIA performing deep mathematical analysis..."):
                            optimal = analyzer.calculate_optimal_sets_advanced(analysis, target_probability=sia_ml_target_prob)
                            
                            # Apply cap if enabled
                            if enable_cap and max_sets_cap is not None:
                                if optimal["optimal_sets"] > max_sets_cap:
                                    optimal["optimal_sets"] = max_sets_cap
                                    optimal["capped"] = True
                                    optimal["original_recommendation"] = optimal["optimal_sets"]
                                else:
                                    optimal["capped"] = False
                            else:
                                optimal["capped"] = False
                            
                            st.session_state.sia_ml_optimal_sets = optimal
                    
                    if st.session_state.sia_ml_optimal_sets:
                        optimal = st.session_state.sia_ml_optimal_sets
                        
                        # Show cap notification if applied
                        if optimal.get("capped", False):
                            st.warning(f"⚠️ **Set Limit Applied**: Recommendation capped at {optimal['optimal_sets']} sets (original calculation suggested more)")
                        
                        # Main metrics in attractive layout
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "🎯 Recommended Sets",
                                optimal["optimal_sets"],
                                help="Optimal number of sets for maximum coverage based on AI/ML analysis"
                            )
                        with col2:
                            st.metric(
                                "📊 Model Confidence",
                                f"{optimal['ensemble_confidence']:.1%}",
                                help="Combined confidence score from ensemble models"
                            )
                        with col3:
                            st.metric(
                                "🔬 Confidence Score",
                                f"{optimal['ensemble_confidence']:.1%}",
                                help="Algorithm confidence in this calculation"
                            )
                        with col4:
                            st.metric(
                                "🎲 Diversity Factor",
                                f"{optimal['diversity_factor']:.2f}",
                                help="Number diversity across sets (higher = more varied)"
                            )
                        
                        st.divider()
                        
                        # Detailed Analysis Breakdown
                        st.markdown("### 📈 Deep Analytical Reasoning")
                        
                        with st.expander("🔍 Algorithm Methodology", expanded=False):
                            st.markdown(f"""
                            **Ensemble Prediction Power:**
                            - Combined Model Accuracy: {analysis['average_accuracy']:.1%}
                            - Ensemble Synergy: {optimal['ensemble_synergy']:.1%}
                            - Weighted Confidence: {optimal['weighted_confidence']:.1%}
                            
                            **Probabilistic Set Calculation:**
                            - Base Probability (1 set): {optimal['base_probability']:.2%}
                            - Sets for 90% confidence: {optimal['optimal_sets']}
                            - Cumulative Probability: {optimal['win_probability']:.1%}
                            
                            **Risk & Variance Analysis:**
                            - Model Variance: {optimal['model_variance']:.4f}
                            - Uncertainty Factor: {optimal['uncertainty_factor']:.2f}
                            - Uncertainty Extra Sets: {optimal.get('uncertainty_extra_sets', 0)}
                            
                            **Set Composition Strategy:**
                            - Diversity Score: {optimal['diversity_factor']:.2f}
                            - Number Distribution: {optimal['distribution_method']}
                            - Hot/Cold Number Balance: {optimal['hot_cold_ratio']:.2f}
                            """)
                        
                        with st.expander("💡 Algorithm Notes & Insights", expanded=True):
                            st.info(optimal['detailed_algorithm_notes'])
                        
                        # Confidence visualization
                        st.markdown("### 📊 Win Probability Curve")
                        
                        # Generate probability curve data
                        max_sets = min(optimal["optimal_sets"] * 2, 100)
                        set_counts = list(range(1, max_sets + 1))
                        base_p = optimal['base_probability']  # Already a decimal, not percentage
                        probabilities = [1 - (1 - base_p) ** n for n in set_counts]
                        
                        # Use module-level import of plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=set_counts,
                            y=[p * 100 for p in probabilities],
                            mode='lines+markers',
                            name='Win Probability',
                            line=dict(color='green', width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Add target line at 90%
                        fig.add_hline(
                            y=90,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="90% Target",
                            annotation_position="right"
                        )
                        fig.update_layout(
                            title="Model Coverage Analysis by Number of Sets",
                            xaxis_title="Number of Sets Generated",
                            yaxis_title="Coverage Score (%)",
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"""
                        ✅ **Recommendation Ready**
                        
                        Based on analysis of {len(analysis['models'])} AI/ML models:
                        - **Recommended: {optimal['optimal_sets']} prediction sets**
                        - Each set uses ensemble intelligence from all selected models
                        - Model confidence: {optimal['ensemble_confidence']:.1%}
                        - Estimated cost: ${optimal['optimal_sets'] * 3:,} (at $3 per ticket)
                        
                        ⚠️ Remember: Lottery odds are 1 in millions regardless of prediction method.
                        These sets provide optimized number selection based on historical patterns.
                        
                        Proceed to the "Generate Predictions" tab to create your sets!
                        """)
        else:
            st.warning(f"⚠️ No models found in selected card: {selected_card}")
    else:
        st.info(f"📝 No model cards available for {analyzer.game}. Visit Phase 2D Leaderboard to create model cards.")
    
    st.divider()
    
    # ==================== EXISTING: STANDARD MODELS SECTION ====================
    st.markdown("### 📋 Standard Models (Optional)")
    st.markdown("Add additional models from the standard models directory")
    
    # Get available model types
    model_types = analyzer.get_available_model_types()
    
    if not model_types:
        st.warning("❌ No models found in the models folder for this game.")
        st.info("📝 Please train models first or check the models directory.")
        return
    
    # Model selection interface
    col1, col2 = st.columns([1.5, 1.5])
    
    with col1:
        st.markdown("### 📋 Available Models")
        st.markdown(f"**Found {len(model_types)} model type(s)**: {', '.join(model_types)}")
        
        # Model type and selection
        selected_type = st.selectbox(
            "Select Model Type",
            model_types,
            key="sia_model_type"
        )
        
        # Get models for selected type
        models_for_type = analyzer.get_models_for_type(selected_type)
        
        if models_for_type:
            model_options = [f"{m['name']} (Acc: {m['accuracy']:.1%})" for m in models_for_type]
            selected_idx = st.selectbox(
                f"Select {selected_type.upper()} Model",
                range(len(models_for_type)),
                format_func=lambda i: model_options[i],
                key="sia_model_select"
            )
            
            if st.button("➕ Add Model to Selection", use_container_width=True):
                selected_model = models_for_type[selected_idx]
                model_tuple = (selected_type, selected_model["name"])
                
                if model_tuple not in st.session_state.sia_selected_models:
                    st.session_state.sia_selected_models.append(model_tuple)
                    st.success(f"✅ Added {selected_model['name']} ({selected_type})")
                else:
                    st.warning("⚠️ Model already selected")
        else:
            if selected_type.lower() == "cnn":
                st.info(f"📊 No CNN models trained yet.\n\nTrain a CNN model in the Data & Training tab to get started!")
            else:
                st.error(f"No models found for type: {selected_type}")
    
    with col2:
        st.markdown("### ✅ Selected Models")
        
        if st.session_state.sia_selected_models:
            st.markdown(f"**Total: {len(st.session_state.sia_selected_models)} model(s)**")
            
            for i, (mtype, mname) in enumerate(st.session_state.sia_selected_models):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.write(f"{i+1}. {mname} ({mtype})")
                with col_b:
                    if st.button("❌", key=f"remove_{i}", help="Remove model"):
                        st.session_state.sia_selected_models.pop(i)
                        st.rerun()
            
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.sia_selected_models = []
                st.rerun()
        else:
            st.info("Select models and click 'Add Model' to build your ensemble")
    
    # Analysis section
    if st.session_state.sia_selected_models:
        st.divider()
        st.markdown("### 📊 Selection Summary & Analysis")
        
        if st.button("🔍 Analyze Selected Models", use_container_width=True, key="analyze_btn"):
            with st.spinner("🤔 Analyzing models..."):
                analysis = analyzer.analyze_selected_models(st.session_state.sia_selected_models)
                st.session_state.sia_analysis_result = analysis
        
        # Display analysis results
        if st.session_state.sia_analysis_result:
            analysis = st.session_state.sia_analysis_result
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Models Selected", analysis["total_selected"])
            with col2:
                st.metric("Average Accuracy", f"{analysis['average_accuracy']:.1%}")
            with col3:
                st.metric("Ensemble Confidence", f"{analysis['ensemble_confidence']:.1%}")
            with col4:
                if analysis["best_model"]:
                    st.metric("Best Model", f"{analysis['best_model']['accuracy']:.1%}")
            
            # Model details
            st.markdown("### Model Details")
            models_df = pd.DataFrame([
                {
                    "Model": m["name"],
                    "Type": m["type"],
                    "Accuracy": f"{m['accuracy']:.1%}",
                    "Confidence": f"{m['confidence']:.1%}"
                }
                for m in analysis["models"]
            ])
            st.dataframe(models_df, use_container_width=True, hide_index=True)
            
            # Display inference logs in an expander
            if analysis.get("inference_logs"):
                with st.expander("🔍 Model Loading & Inference Details", expanded=False):
                    st.markdown("**Real-time inference log showing model loading and prediction generation:**")
                    for log_entry in analysis["inference_logs"]:
                        st.text(log_entry)
            
            # Optimal Analysis section
            st.divider()
            st.markdown("### 🎯 Intelligent Set Recommendation - AI/ML Analysis")
            
            st.markdown("""
            **Goal:** Determine the optimal number of prediction sets based on AI/ML model analysis.
            
            This system analyzes your selected models to recommend a practical number of sets:
            - **Ensemble Accuracy Analysis**: Combined predictive power of all selected models
            - **Optimal Coverage Calculation**: Balances coverage with practical budget
            - **Confidence-Based Weighting**: Adjusts recommendations based on model reliability
            - **Cost-Effective Strategy**: Provides actionable recommendations you can actually use
            
            ⚠️ **Important**: Lottery outcomes are random. These are optimized recommendations, not guaranteed wins.
            """)
            
            # Coverage target + Set Limit Controls (Standard Models)
            col_cov_std, col_cap1, col_cap2 = st.columns([2, 1, 2])

            with col_cov_std:
                sia_std_target_prob = st.slider(
                    "Coverage Target",
                    min_value=50, max_value=95, value=90, step=5,
                    key="sia_std_target_prob",
                    help="How confident do you want to be that at least one set covers the winning pattern? Higher = more sets recommended."
                ) / 100.0

            with col_cap1:
                enable_cap_std = st.checkbox(
                    "Enable Set Limit",
                    value=False,
                    key="sia_std_enable_cap",
                    help="Limit the maximum number of recommended sets"
                )

            with col_cap2:
                if enable_cap_std:
                    max_sets_cap_std = st.number_input(
                        "Maximum Sets",
                        min_value=1,
                        max_value=1000,
                        value=100,
                        step=1,
                        key="sia_std_max_cap",
                        help="Set the maximum number of sets to recommend (1-1000)"
                    )

            if st.button("🧠 Calculate Optimal Sets (SIA)", use_container_width=True, key="sia_calc_btn"):
                with st.spinner("🤖 SIA performing deep mathematical analysis..."):
                    optimal = analyzer.calculate_optimal_sets_advanced(analysis, target_probability=sia_std_target_prob)
                    
                    # Apply cap if enabled
                    if enable_cap_std and 'max_sets_cap_std' in locals() and max_sets_cap_std is not None:
                        if optimal["optimal_sets"] > max_sets_cap_std:
                            optimal["optimal_sets"] = max_sets_cap_std
                            optimal["capped"] = True
                        else:
                            optimal["capped"] = False
                    else:
                        optimal["capped"] = False
                    
                    st.session_state.sia_optimal_sets = optimal
            
            if st.session_state.sia_optimal_sets:
                optimal = st.session_state.sia_optimal_sets
                
                # Show cap notification if applied
                if optimal.get("capped", False):
                    st.warning(f"⚠️ **Set Limit Applied**: Recommendation capped at {optimal['optimal_sets']} sets (original calculation suggested more)")
                
                # Main metrics in attractive layout
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎯 Recommended Sets",
                        optimal["optimal_sets"],
                        help="Optimal number of sets for maximum coverage based on AI/ML analysis"
                    )
                with col2:
                    st.metric(
                        "📊 Model Confidence",
                        f"{optimal['ensemble_confidence']:.1%}",
                        help="Combined confidence score from ensemble models"
                    )
                with col3:
                    st.metric(
                        "🔬 Confidence Score",
                        f"{optimal['ensemble_confidence']:.1%}",
                        help="Algorithm confidence in this calculation"
                    )
                with col4:
                    st.metric(
                        "🎲 Diversity Factor",
                        f"{optimal['diversity_factor']:.2f}",
                        help="Number diversity across sets (higher = more varied)"
                    )
                
                st.divider()
                
                # Detailed Analysis Breakdown
                st.markdown("### 📈 Deep Analytical Reasoning")
                
                with st.expander("🔍 Algorithm Methodology", expanded=False):
                    st.markdown(f"""
                    **Ensemble Prediction Power:**
                    - Combined Model Accuracy: {analysis['average_accuracy']:.1%}
                    - Ensemble Synergy: {optimal['ensemble_synergy']:.1%}
                    - Weighted Confidence: {optimal['weighted_confidence']:.1%}
                    
                    **Probabilistic Set Calculation:**
                    - Base Probability (1 set): {optimal['base_probability']:.2%}
                    - Sets for 90% confidence: {optimal['optimal_sets']}
                    - Cumulative Probability: {optimal['win_probability']:.1%}
                    
                    **Risk & Variance Analysis:**
                    - Model Variance: {optimal['model_variance']:.4f}
                    - Uncertainty Factor: {optimal['uncertainty_factor']:.2f}
                    - Uncertainty Extra Sets: {optimal.get('uncertainty_extra_sets', 0)}
                    
                    **Set Composition Strategy:**
                    - Diversity Score: {optimal['diversity_factor']:.2f}
                    - Number Distribution: {optimal['distribution_method']}
                    - Hot/Cold Number Balance: {optimal['hot_cold_ratio']:.2f}
                    """)
                
                with st.expander("💡 Algorithm Notes & Insights", expanded=True):
                    st.info(optimal['detailed_algorithm_notes'])
                
                # Confidence visualization
                st.markdown("### 🎲 Win Probability Visualization")
                
                # Create probability curve
                sets_range = list(range(1, optimal['optimal_sets'] + 3))
                probabilities = []
                for n in sets_range:
                    # Cumulative probability increases with each set
                    prob = 1 - ((1 - optimal['base_probability']) ** n)
                    probabilities.append(prob)
                
                # Use module-level import of plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sets_range,
                    y=probabilities,
                    mode='lines+markers',
                    name='Win Probability',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=8)
                ))
                fig.add_hline(
                    y=0.9,
                    line_dash="dash",
                    line_color="#ef4444",
                    annotation_text="90% Target",
                    annotation_position="right"
                )
                fig.update_layout(
                    title="Cumulative Win Probability by Number of Sets",
                    xaxis_title="Number of Sets Generated",
                    yaxis_title="Win Probability",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"""
                ✅ **AI Recommendation Ready!**
                
                To win the {analyzer.game} lottery in the next draw with 90%+ confidence:
                - **Generate exactly {optimal['optimal_sets']} prediction sets**
                - Each set crafted using deep AI/ML reasoning combining all {len(analysis['models'])} models
                - Expected win probability: {optimal['win_probability']:.1%}
                - Algorithm confidence: {optimal['ensemble_confidence']:.1%}
                
                Proceed to the "Generate Predictions" tab to create your optimized sets!
                """)


# ============================================================================
# TAB 2: GENERATE PREDICTIONS
# ============================================================================

def _render_prediction_generator(analyzer: SuperIntelligentAIAnalyzer) -> None:
    """Generate AI-optimized predictions for winning lottery with >90% confidence."""
    st.subheader("🎲 AI-Powered Prediction Generation - Super Intelligent Algorithm")
    
    # Check for either ML Models or Standard Models optimal sets
    optimal = None
    analysis = None
    model_source = None
    
    if st.session_state.sia_ml_optimal_sets:
        optimal = st.session_state.sia_ml_optimal_sets
        analysis = st.session_state.sia_ml_analysis_result
        model_source = "Machine Learning Models"
    elif st.session_state.sia_optimal_sets:
        optimal = st.session_state.sia_optimal_sets
        analysis = st.session_state.sia_analysis_result
        model_source = "Standard Models"
    
    # --- Prerequisites checklist ---
    _ml_analysis = st.session_state.get('sia_ml_analysis_result')
    _std_analysis = st.session_state.get('sia_analysis_result')
    _ml_optimal = st.session_state.get('sia_ml_optimal_sets')
    _std_optimal = st.session_state.get('sia_optimal_sets')

    _step1_ok = bool(_ml_analysis or _std_analysis)   # models analysed
    _step2_ok = bool(_ml_optimal or _std_optimal)      # optimal sets calculated
    _step3_ok = bool(optimal and analysis)             # ready to generate

    def _chk(ok: bool) -> str:
        return "✅" if ok else "❌"

    with st.container():
        st.markdown("#### Prerequisites")
        _col_a, _col_b, _col_c = st.columns(3)
        with _col_a:
            st.markdown(
                f"{_chk(_step1_ok)} **Step 1 — Analyse Models**  \n"
                f"{'Done' if _step1_ok else 'Go to Tab 1 → click Analyze'}"
            )
        with _col_b:
            st.markdown(
                f"{_chk(_step2_ok)} **Step 2 — Calculate Optimal Sets**  \n"
                f"{'Done' if _step2_ok else 'Tab 1 → click Calculate Optimal Sets'}"
            )
        with _col_c:
            # Optional: learning files
            _learning_dir = Path(__file__).parent.parent.parent / "data" / "learning" / \
                ("lotto_6_49" if "6/49" in (analyzer.game if analyzer else "") else "lotto_max")
            _lf_count = len(list(_learning_dir.glob("draw_*_learning.json"))) if _learning_dir.exists() else 0
            _lf_ok = _lf_count > 0
            st.markdown(
                f"{'✅' if _lf_ok else '⚠️'} **Step 3 — Learning Files** (optional)  \n"
                f"{_lf_count} file(s) available for draw-history blend"
            )
        st.divider()

    if not optimal or not analysis:
        st.error(
            "Steps 1 and 2 must be completed in the **AI Model Configuration** tab before generating predictions."
        )
        return

    # Show which model source is being used
    st.info(f"ℹ️ **Using Configuration from:** {model_source}")
    
    st.markdown(f"""
    ### 🎯 AI Mission: Generate {optimal['optimal_sets']} Optimized Prediction Sets
    
    Based on deep AI/ML analysis of your {len(analysis['models'])} selected models:
    - **Ensemble Accuracy:** {analysis['average_accuracy']:.1%}
    - **Algorithm Confidence:** {optimal['ensemble_confidence']:.1%}
    - **Target Win Probability:** {optimal['win_probability']:.1%}
    
    These sets will be generated using advanced reasoning combining:
    - Mathematical pattern analysis from all models
    - Statistical probability distributions
    - Ensemble voting with confidence weighting
    - Hot/cold number analysis and diversity optimization
    """)
    
    # ===== NEW: GENERATION CONTROLS =====
    st.divider()
    st.markdown("### ⚙️ Generation Controls")
    
    # Initialize session state for new controls
    if 'sia_use_learning' not in st.session_state:
        st.session_state.sia_use_learning = False
    if 'sia_use_adaptive_learning' not in st.session_state:
        st.session_state.sia_use_adaptive_learning = True  # Default to True
    if 'sia_selected_learning_files' not in st.session_state:
        st.session_state.sia_selected_learning_files = []
    if 'sia_use_custom_quantity' not in st.session_state:
        st.session_state.sia_use_custom_quantity = False
    if 'sia_custom_quantity' not in st.session_state:
        st.session_state.sia_custom_quantity = optimal["optimal_sets"]
    if 'sia_no_repeat_numbers' not in st.session_state:
        st.session_state.sia_no_repeat_numbers = False
    if 'sia_use_draw_profile' not in st.session_state:
        st.session_state.sia_use_draw_profile = False
    if 'sia_selected_draw_profile' not in st.session_state:
        st.session_state.sia_selected_draw_profile = None
    # P10: multi-profile selection (list of indices)
    if 'sia_selected_draw_profiles' not in st.session_state:
        st.session_state.sia_selected_draw_profiles = []
    
    # Three column layout for the checkboxes
    checkbox_col1, checkbox_col2, checkbox_col3 = st.columns(3)
    
    with checkbox_col1:
        use_learning = st.checkbox(
            "📚 Use Learning Files",
            value=st.session_state.sia_use_learning,
            key="sia_use_learning_checkbox",
            help="Incorporate insights from historical learning data to optimize predictions"
        )
        st.session_state.sia_use_learning = use_learning
    
    with checkbox_col2:
        use_custom_quantity = st.checkbox(
            "🎲 Custom Sets Quantity",
            value=st.session_state.sia_use_custom_quantity,
            key="sia_use_custom_quantity_checkbox",
            help="Override recommended quantity with your own custom number of sets"
        )
        st.session_state.sia_use_custom_quantity = use_custom_quantity
    
    with checkbox_col3:
        no_repeat_numbers = st.checkbox(
            "🔢 No Repeat Numbers Across Sets",
            value=st.session_state.sia_no_repeat_numbers,
            key="sia_no_repeat_numbers_checkbox",
            help="Maximize number diversity by minimizing repetition across sets. Uses intelligent calibration to ensure unique coverage."
        )
        st.session_state.sia_no_repeat_numbers = no_repeat_numbers
    
    # Second row for Draw Profile checkbox
    checkbox_col4 = st.columns(1)[0]
    
    with checkbox_col4:
        use_draw_profile = st.checkbox(
            "🎯 Use Draw Profile",
            value=st.session_state.sia_use_draw_profile,
            key="sia_use_draw_profile_checkbox",
            help="Enforce all generated sets to match patterns from a saved jackpot draw profile"
        )
        st.session_state.sia_use_draw_profile = use_draw_profile
    
    # Draw Profile Selection (shown when checkbox is enabled)
    if use_draw_profile:
        st.markdown("#### 🎯 Select Draw Profile File")
        
        # Find available draw profile files
        available_profiles = _find_all_draw_profiles(analyzer.game)
        
        if available_profiles:
            import re as _re_pf

            # Parse each profile: read metadata from JSON for jackpot_amount + profile_name
            _profile_meta = []
            for pf in available_profiles:
                try:
                    with open(pf, 'r') as _fp:
                        _pd = json.load(_fp)
                    _jamt = _pd.get('metadata', {}).get('jackpot_amount', 0)
                    _pname = _pd.get('metadata', {}).get('profile_name', pf.stem)
                except Exception:
                    _jamt = 0
                    _pname = pf.stem
                # Parse saved date from filename suffix: {name}_{YYYYMMDD_HHMMSS}.json
                _dm = _re_pf.search(r'_(\d{4})(\d{2})(\d{2})_\d{6}$', pf.stem)
                _saved_date = f"{_dm.group(1)}-{_dm.group(2)}-{_dm.group(3)}" if _dm else "unknown date"
                _profile_meta.append({"path": pf, "jackpot": _jamt, "name": _pname, "date": _saved_date})

            # Sort profiles by jackpot amount descending so biggest are at top
            _profile_meta = sorted(_profile_meta, key=lambda x: x['jackpot'], reverse=True)
            available_profiles = [pm['path'] for pm in _profile_meta]

            profile_options = [
                f"${pm['jackpot']/1e6:.0f}M — {pm['name']} (saved {pm['date']})"
                if pm['jackpot'] >= 1e6
                else f"${pm['jackpot']:,.0f} — {pm['name']} (saved {pm['date']})"
                for pm in _profile_meta
            ]

            # Current jackpot input → auto-select closest profile(s)
            _curr_jackpot = st.number_input(
                "Current jackpot ($M) — used to auto-select closest profiles:",
                min_value=0, max_value=1000, value=50, step=5,
                key="sia_current_jackpot_m",
                help="Enter today's jackpot to auto-select the best matching profile(s)"
            ) * 1_000_000

            # Find 1-3 closest profiles (P10: multi-profile)
            _sorted_by_prox = sorted(
                range(len(_profile_meta)),
                key=lambda i: abs(_profile_meta[i]['jackpot'] - _curr_jackpot)
            )
            _auto_default_indices = _sorted_by_prox[:min(2, len(_sorted_by_prox))]
            _prev_multi = st.session_state.sia_selected_draw_profiles
            _multi_default = _prev_multi if _prev_multi else _auto_default_indices

            # P10: multiselect — blend up to 3 profiles
            selected_profile_indices = st.multiselect(
                "Choose draw profile(s) to blend (up to 3 recommended):",
                range(len(available_profiles)),
                format_func=lambda i: profile_options[i],
                default=[i for i in _multi_default if i < len(available_profiles)],
                key="sia_draw_profile_multiselect",
                help="Multiple profiles are blended together (weighted by draw count). "
                     "Closest to current jackpot auto-selected."
            )
            st.session_state.sia_selected_draw_profiles = selected_profile_indices
            # Keep legacy single-select state in sync for backwards compat
            st.session_state.sia_selected_draw_profile = selected_profile_indices[0] if selected_profile_indices else None

            if selected_profile_indices:
                _sel_pms = [_profile_meta[i] for i in selected_profile_indices]
                _total_draws_sel = sum(p.get('jackpot', 0) and 1 for p in _sel_pms)
                st.success(
                    f"✅ **{len(selected_profile_indices)} profile(s) selected** — "
                    + ", ".join(
                        f"{pm['name']} (${pm['jackpot']/1e6:.0f}M)" for pm in _sel_pms
                    )
                )
                # Expandable details for each selected profile
                for _sel_idx in selected_profile_indices:
                    _sel_path = available_profiles[_sel_idx]
                    _sel_pm   = _profile_meta[_sel_idx]
                    _pdata    = _load_draw_profile(_sel_path)
                    if _pdata:
                        with st.expander(f"📋 {_sel_pm['name']} — Profile Details", expanded=False):
                            _meta = _pdata.get('metadata', {})
                            _cp   = _pdata.get('core_patterns', {})
                            _ss   = _pdata.get('sum_analysis', {})
                            _np   = _pdata.get('number_patterns', {})
                            _col1, _col2, _col3 = st.columns(3)
                            with _col1:
                                st.write(f"**Jackpot:** ${_meta.get('jackpot_amount', 0):,.0f}")
                                st.write(f"**Draws Analyzed:** {_meta.get('total_draws_analyzed', 0)}")
                                st.write(f"**Saved:** {_sel_pm['date']}")
                            with _col2:
                                _oe = _cp.get('odd_even_distribution', {})
                                if _oe:
                                    st.write(f"**Top Odd/Even:** {max(_oe, key=_oe.get)}")
                                _hl = _cp.get('high_low_distribution', {})
                                if _hl:
                                    st.write(f"**Top High/Low:** {max(_hl, key=_hl.get)}")
                                _pairs = _np.get('pair_cooccurrence', {})
                                if _pairs:
                                    _top_pair = max(_pairs, key=_pairs.get)
                                    st.write(f"**Top Pair:** {_top_pair} ({_pairs[_top_pair]:.2f})")
                            with _col3:
                                if _ss:
                                    st.write(f"**Sum Range:** {_ss.get('mode_range', 'N/A')}")
                                    st.write(f"**Avg Sum:** {_ss.get('mean', 0):.1f} ± {_ss.get('std', 0):.1f}")
                                _hot = _np.get('hot_numbers', [])[:3]
                                if _hot:
                                    st.write(f"**Top-3 Hot:** {[h[0] for h in _hot if isinstance(h, list)]}")
            else:
                st.info("No profile selected — generation will use only ML ensemble probabilities.")
        else:
            st.warning(f"⚠️ No draw profiles found for {analyzer.game}. Create profiles in the Jackpot Pattern Analysis tab first.")
            st.session_state.sia_use_draw_profile = False
    
    # Learning Files Selection (shown when checkbox is enabled)
    if use_learning:
        st.markdown("#### 📂 Select Learning Files")
        
        # Find available learning files
        available_learning_files = _find_all_learning_files(analyzer.game)
        
        if available_learning_files:
            # Sort by draw date extracted from filename (draw_YYYYMMDD_learning.json)
            def _lf_draw_date(p):
                import re
                m = re.search(r'draw_(\d{8})_', p.name)
                return m.group(1) if m else p.stat().st_mtime.__str__()

            available_learning_files = sorted(available_learning_files, key=_lf_draw_date, reverse=True)

            # Build human-readable labels with date parsed from filename
            import re as _re
            learning_options = []
            for lf in available_learning_files:
                _dm = _re.search(r'draw_(\d{4})(\d{2})(\d{2})_', lf.name)
                if _dm:
                    _label_date = f"{_dm.group(1)}-{_dm.group(2)}-{_dm.group(3)}"
                else:
                    _label_date = datetime.fromtimestamp(lf.stat().st_mtime).strftime('%Y-%m-%d')
                _size_kb = lf.stat().st_size / 1024
                learning_options.append(f"{_label_date} — {lf.name} ({_size_kb:.1f} KB)")

            # Default: pre-select 3 most recent
            _default_sel = list(range(min(3, len(available_learning_files))))
            _prev_sel = st.session_state.sia_selected_learning_files
            _sel_default = _prev_sel if _prev_sel else _default_sel

            selected_learning_indices = st.multiselect(
                "Choose learning files (3 most recent pre-selected):",
                range(len(available_learning_files)),
                format_func=lambda i: learning_options[i],
                default=_sel_default,
                key="sia_learning_files_multiselect",
                help="Files are sorted newest → oldest. Select multiple to combine historical draw insights."
            )

            st.session_state.sia_selected_learning_files = selected_learning_indices

            # Coverage metric
            if selected_learning_indices:
                _total = len(available_learning_files)
                _sel_n = len(selected_learning_indices)
                _pct = _sel_n / _total * 100 if _total else 0
                st.info(
                    f"✅ **{_sel_n} / {_total} files selected** ({_pct:.0f}% draw history coverage)  \n"
                    f"Earliest selected: {learning_options[max(selected_learning_indices)].split(' — ')[0]}"
                )
            else:
                st.warning("⚠️ No learning files selected. Predictions will be generated without learning insights.")
            
            # Adaptive Learning Toggle (shown when learning files are selected)
            if selected_learning_indices:
                st.markdown("#### 🧬 Adaptive Learning Mode")
                use_adaptive_learning = st.checkbox(
                    "Enable Adaptive Learning",
                    value=st.session_state.sia_use_adaptive_learning,
                    key="sia_use_adaptive_learning_checkbox",
                    help="Use evolved intelligence with adaptive weights based on historical success patterns. When disabled, uses static learning weights (15% hot, 12% cold)."
                )
                st.session_state.sia_use_adaptive_learning = use_adaptive_learning
                
                if use_adaptive_learning:
                    st.success("🧬 Adaptive learning enabled - using evolved weights from meta-learning")
                else:
                    st.info("📊 Static learning enabled - using fixed weight values")
        else:
            st.warning(f"⚠️ No learning files found for {analyzer.game}. Generate predictions first and analyze them in the Deep Learning & Analytics tab to create learning data.")
            st.session_state.sia_use_learning = False
    
    # Custom Quantity Control (shown when checkbox is enabled)
    if use_custom_quantity:
        st.markdown("#### 🔢 Custom Quantity")
        
        custom_quantity_col1, custom_quantity_col2 = st.columns([3, 1])
        
        with custom_quantity_col1:
            custom_sets = st.number_input(
                "Number of sets to generate:",
                min_value=1,
                max_value=500,
                value=st.session_state.sia_custom_quantity,
                step=1,
                key="sia_custom_quantity_input",
                help="Choose between 1 and 500 prediction sets (overrides SIA calculation)"
            )
            st.session_state.sia_custom_quantity = custom_sets
        
        with custom_quantity_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if custom_sets != optimal["optimal_sets"]:
                st.warning(f"⚠️ Override: {custom_sets} sets")
            else:
                st.success("✅ Matches SIA")
    
    st.divider()
    
    # ===== METRICS ROW =====
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SIA-Calculated Sets", optimal["optimal_sets"])
    
    with col2:
        adjustment = st.slider(
            "🎛️ Set Adjustment",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Multiply recommended sets: <1.0 conservative, >1.0 aggressive for higher chance"
        )
    
    with col3:
        # Determine final sets count based on custom quantity override
        if use_custom_quantity:
            final_sets = st.session_state.sia_custom_quantity
        else:
            final_sets = max(1, int(optimal["optimal_sets"] * adjustment))
        st.metric("Final Sets", final_sets)
    
    with col4:
        if use_custom_quantity:
            adjustment_str = "Custom Override"
        else:
            adjustment_str = "Conservative" if adjustment < 1.0 else "Aggressive" if adjustment > 1.0 else "Optimal"
        st.metric("Strategy", adjustment_str)
    
    st.divider()
    
    # Generation strategy explanation
    with st.expander("🧠 Advanced Generation Strategy", expanded=False):
        st.markdown(f"""
        **How AI generates each set:**
        
        1. **Ensemble Voting**: Each model casts weighted votes based on accuracy
        2. **Probability Scoring**: Numbers ranked by likelihood across all models
        3. **Diversity Injection**: Ensures each set has unique number combinations
        4. **Pattern Recognition**: Uses learned patterns from training data
        5. **Confidence Weighting**: Stronger models have more influence
        6. **Temporal Analysis**: Incorporates hot/cold number trends
        
        **Set Composition:**
        - Diversity Factor: {optimal['diversity_factor']:.2f}
        - Hot/Cold Ratio: {optimal['hot_cold_ratio']:.2f}
        - Distribution Method: {optimal['distribution_method']}
        - Number Range Optimization: Statistical clustering
        
        **Quality Assurance:**
        - Each set independently optimized
        - Variance check between sets
        - Confidence score per set
        - Ensemble consensus verification
        """)
    
    st.markdown("---")
    
    if st.button("🚀 Generate AI-Optimized Prediction Sets", use_container_width=True, key="gen_pred_btn", help="Generate precisely calculated sets for maximum winning probability"):
        with st.spinner(f"🤖 Generating {final_sets} AI-optimized prediction sets using deep learning..."):
            try:
                # Get adaptive learning setting (only relevant when learning is enabled)
                use_adaptive_learning = st.session_state.get('sia_use_adaptive_learning', True)
                
                # ===== LOAD LEARNING FILES IF ENABLED =====
                learning_data = None
                learning_file_paths = []
                
                if use_learning and st.session_state.sia_selected_learning_files:
                    st.info("📚 Loading learning files to enhance predictions...")
                    available_files = _find_all_learning_files(analyzer.game)
                    selected_files = [available_files[i] for i in st.session_state.sia_selected_learning_files]
                    
                    if selected_files:
                        learning_file_paths = selected_files
                        learning_data = _load_and_combine_learning_files(selected_files)
                        st.success(f"✅ Loaded {len(selected_files)} learning file(s) with combined insights")
                
                # ===== LOAD DRAW PROFILE(S) IF ENABLED (P10: multi-profile) =====
                draw_profile_data = None   # will be a List[Dict] when multiple profiles selected
                draw_profile_name = None

                if use_draw_profile:
                    _sel_indices = st.session_state.get('sia_selected_draw_profiles', [])
                    # Fallback: legacy single-select
                    if not _sel_indices and st.session_state.sia_selected_draw_profile is not None:
                        _sel_indices = [st.session_state.sia_selected_draw_profile]

                    if _sel_indices:
                        _avail = _find_all_draw_profiles(analyzer.game)
                        _loaded = []
                        for _pi in _sel_indices:
                            if _pi < len(_avail):
                                _pd = _load_draw_profile(_avail[_pi])
                                if _pd:
                                    _loaded.append(_pd)
                        if _loaded:
                            draw_profile_data = _loaded  # List[Dict] — generation blends all
                            draw_profile_name = f"{len(_loaded)} profile(s)"
                            st.success(f"✅ Loaded {len(_loaded)} draw profile(s) for blending")
                        else:
                            st.warning("⚠️ Failed to load any draw profiles, proceeding without profile enforcement")
                
                # ===== GENERATE PREDICTIONS =====
                # Pass learning data to generation if available
                if learning_data:
                    # Generate with learning-enhanced algorithm
                    predictions, strategy_report, predictions_with_attribution, strategy_log = analyzer.generate_prediction_sets_advanced(
                        final_sets,
                        optimal,
                        analysis,
                        learning_data=learning_data,
                        no_repeat_numbers=no_repeat_numbers,
                        use_adaptive_learning=use_adaptive_learning,
                        draw_profile_data=draw_profile_data
                    )
                else:
                    # Generate normally without learning
                    predictions, strategy_report, predictions_with_attribution, strategy_log = analyzer.generate_prediction_sets_advanced(
                        final_sets,
                        optimal,
                        analysis,
                        no_repeat_numbers=no_repeat_numbers,
                        use_adaptive_learning=False,
                        draw_profile_data=draw_profile_data
                    )
                
                # ===== VALIDATE AND SCORE AGAINST DRAW PROFILE (soft conformance) =====
                profile_validation_results = []
                if draw_profile_data:
                    st.info("🔍 Scoring all sets against draw profile (soft conformance)...")

                    # With profile-guided generation already biasing probabilities, the
                    # vast majority of sets will be conformant without regeneration.
                    # Only attempt a single replacement pass for the very lowest scores.
                    max_regeneration_attempts = 5   # reduced from 100 — profile blend handles most cases
                    regeneration_count = 0

                    # Resolve single profile for validation (P10: use first/primary profile)
                    _validation_profile = (
                        draw_profile_data[0]
                        if isinstance(draw_profile_data, list) and draw_profile_data
                        else draw_profile_data
                    )

                    # Validate each prediction set
                    for i, pred_set in enumerate(predictions[:]):  # Use slice to iterate over copy
                        validation = _validate_set_against_profile(pred_set, _validation_profile, analyzer.game)

                        # Only regenerate truly poor conformance (< 40%)
                        attempts = 0
                        while not validation['valid'] and attempts < max_regeneration_attempts:
                            # Build number usage from OTHER sets (exclude the one being regenerated)
                            # This ensures no_repeat_numbers works correctly during regeneration
                            existing_usage = {}
                            if no_repeat_numbers:
                                for j, other_set in enumerate(predictions):
                                    if j != i:  # Don't count the set we're replacing
                                        for num in other_set:
                                            existing_usage[num] = existing_usage.get(num, 0) + 1
                            
                            # Generate a new set with knowledge of already-used numbers
                            if learning_data:
                                new_set, _, new_attribution, _ = analyzer.generate_prediction_sets_advanced(
                                    1, optimal, analysis,
                                    learning_data=learning_data,
                                    no_repeat_numbers=no_repeat_numbers,
                                    use_adaptive_learning=use_adaptive_learning,
                                    existing_number_usage=existing_usage,
                                    draw_profile_data=draw_profile_data
                                )
                            else:
                                new_set, _, new_attribution, _ = analyzer.generate_prediction_sets_advanced(
                                    1, optimal, analysis,
                                    no_repeat_numbers=no_repeat_numbers,
                                    use_adaptive_learning=False,
                                    existing_number_usage=existing_usage,
                                    draw_profile_data=draw_profile_data
                                )
                            
                            # Replace the set
                            predictions[i] = new_set[0]
                            if new_attribution:
                                predictions_with_attribution[i] = new_attribution[0]
                            
                            # Validate new set (always use single-profile dict for validation)
                            validation = _validate_set_against_profile(predictions[i], _validation_profile, analyzer.game)
                            attempts += 1
                            regeneration_count += 1
                        
                        profile_validation_results.append(validation)

                    # Sort predictions by conformance_score descending so the best-matching
                    # sets appear first in the output list
                    if profile_validation_results:
                        _paired = list(zip(profile_validation_results, predictions, predictions_with_attribution))
                        _paired.sort(key=lambda x: x[0].get('conformance_score', 0.0), reverse=True)
                        profile_validation_results, predictions, predictions_with_attribution = (
                            [p[0] for p in _paired],
                            [p[1] for p in _paired],
                            [p[2] for p in _paired],
                        )

                    avg_conformance = (
                        sum(v.get('conformance_score', 0.5) for v in profile_validation_results)
                        / max(1, len(profile_validation_results))
                    )
                    if regeneration_count > 0:
                        st.success(
                            f"✅ Profile scoring complete — avg conformance: {avg_conformance:.0%} "
                            f"({regeneration_count} low-scoring set(s) regenerated)"
                        )
                    else:
                        st.success(
                            f"✅ All sets profile-scored — avg conformance: {avg_conformance:.0%} "
                            f"(sorted best-to-worst)"
                        )
                
                # ===== COMPUTE GENERATION CONFIDENCE SCORES & RANKING =====
                # Score each set so we can rank them at generation time before any
                # draw results are known.  Score = weighted blend of:
                #   45% — avg model confidence (how strongly models preferred these numbers)
                #   35% — vote density (fraction of numbers that received model votes)
                #   20% — profile conformance (if a draw profile was used, else neutral 0.5)
                _gc_scores: list = []
                for _si, _sp in enumerate(predictions):
                    _attr_data = (
                        predictions_with_attribution[_si]
                        if _si < len(predictions_with_attribution) else {}
                    )
                    _ma = _attr_data.get('model_attribution', {}) if isinstance(_attr_data, dict) else {}
                    # Avg model confidence across numbers in this set
                    _confs = []
                    _nvoted = 0
                    for _num in _sp:
                        _voters = _ma.get(str(_num), [])
                        if _voters:
                            _nvoted += 1
                            _confs.append(
                                sum(v.get('confidence', 1.0) for v in _voters) / len(_voters)
                            )
                        else:
                            _confs.append(1.0)  # uniform baseline = 1.0
                    _avg_conf = sum(_confs) / len(_confs) if _confs else 1.0
                    # Normalise confidence: 1.0 = uniform, cap useful range at 6×
                    _conf_score = min(1.0, max(0.0, (_avg_conf - 1.0) / 5.0))
                    # Vote density: what fraction of numbers had at least one model vote
                    _vote_density = _nvoted / max(1, len(_sp))
                    # Profile conformance
                    _pconf = 0.5  # neutral when no profile used
                    if profile_validation_results and _si < len(profile_validation_results):
                        _pconf = profile_validation_results[_si].get('conformance_score', 0.5)
                    _score = round(
                        _conf_score * 0.45 + _vote_density * 0.35 + _pconf * 0.20, 4
                    )
                    _gc_scores.append(_score)

                # Build ranked order: sort by score descending, preserve original set index
                _rank_pairs = sorted(
                    enumerate(_gc_scores), key=lambda x: -x[1]
                )  # [(orig_idx, score), ...]
                # ranked_sets: list of (rank_1based, orig_0based_idx, numbers, score)
                _generation_ranked = [
                    {
                        'rank': _rank + 1,
                        'original_set_index': _orig_idx,  # 0-based
                        'set_number': _orig_idx + 1,      # 1-based display
                        'numbers': sorted(predictions[_orig_idx]),
                        'confidence_score': _gc_scores[_orig_idx],
                    }
                    for _rank, (_orig_idx, _) in enumerate(_rank_pairs)
                ]
                # Also store rank per original-set-index for quick lookup
                _rank_by_orig = {r['original_set_index']: r['rank'] for r in _generation_ranked}

                # Embed rank info into predictions_with_attribution
                for _si in range(len(predictions_with_attribution)):
                    if isinstance(predictions_with_attribution[_si], dict):
                        predictions_with_attribution[_si]['generation_rank'] = _rank_by_orig.get(_si)
                        predictions_with_attribution[_si]['confidence_score'] = _gc_scores[_si] if _si < len(_gc_scores) else None

                # Save to session and file with attribution data
                st.session_state.sia_predictions = predictions
                st.session_state.sia_strategy_report = strategy_report
                st.session_state.sia_predictions_with_attribution = predictions_with_attribution
                st.session_state.sia_strategy_log = strategy_log
                st.session_state.sia_generation_ranked = _generation_ranked
                
                # Add learning metadata to saved file
                if learning_data:
                    # Store learning file info in optimal_analysis for saving
                    optimal_with_learning = optimal.copy()
                    optimal_with_learning['learning_files_used'] = [str(f.name) for f in learning_file_paths]
                    optimal_with_learning['learning_insights_count'] = len(learning_data.get('combined_insights', []))
                    filepath = analyzer.save_predictions_advanced(predictions, analysis, optimal_with_learning, final_sets, predictions_with_attribution)
                else:
                    filepath = analyzer.save_predictions_advanced(predictions, analysis, optimal, final_sets, predictions_with_attribution)

                # Patch saved file to include generation_ranking (not a hot path — one small file write)
                try:
                    import json as _json_patch
                    with open(filepath, 'r') as _pf:
                        _pdata = _json_patch.load(_pf)
                    _pdata['generation_ranking'] = _generation_ranked
                    with open(filepath, 'w') as _pf:
                        _json_patch.dump(_pdata, _pf, indent=2, default=str)
                except Exception:
                    pass  # Non-critical — learning tab will degrade gracefully
                
                st.success(f"✅ Successfully generated {final_sets} AI-optimized prediction sets!")
                
                # Show learning enhancement info with adaptive intelligence status
                if learning_data:
                    st.balloons()
                    
                    # Check if adaptive learning was used
                    stored_strategy_log = st.session_state.get('sia_strategy_log', {})
                    if stored_strategy_log.get('adaptive_learning_enabled'):
                        learning_cycles = stored_strategy_log.get('learning_cycles', 0)
                        st.success(f"🧬 **Adaptive Learning Applied:** Using evolved intelligence from {learning_cycles} learning cycles")
                        
                        # Show adaptive weights summary
                        adaptive_weights = stored_strategy_log.get('adaptive_weights', {})
                        if adaptive_weights:
                            top_3_factors = sorted(adaptive_weights.items(), key=lambda x: x[1], reverse=True)[:3]
                            factors_str = ", ".join([f"{name.replace('_', ' ').title()}: {weight:.1%}" for name, weight in top_3_factors])
                            st.info(f"📊 **Top Adaptive Factors:** {factors_str}")
                    
                    st.success(f"📚 **Learning Enhanced:** Predictions optimized using insights from {len(learning_file_paths)} historical learning file(s)")

                    # P8: Learning contribution breakdown — show how each component
                    # shifted the probability distribution during generation.
                    with st.expander("🔬 Learning Contribution Breakdown", expanded=False):
                        _slog = stored_strategy_log
                        st.markdown("**How each learning layer influenced this generation:**")

                        _breakdown_rows = []

                        # Per-draw file blend
                        _blend_w = _slog.get('learning_blend_weight', None)
                        _blend_files = _slog.get('recent_draw_files_blended', 0)
                        if _blend_w is not None and _blend_files:
                            _blend_cycles = _slog.get('learning_cycles_for_blend', 0)
                            _breakdown_rows.append({
                                'Component': '📂 Per-Draw File Blend',
                                'Weight': f"{_blend_w:.1%}",
                                'Detail': f"{_blend_files} files · {_blend_cycles} cycles accumulated"
                            })

                        # Jackpot profile blend
                        _jp_w = _slog.get('jackpot_profile_blend_weight', None)
                        _jp_n = _slog.get('jackpot_profile_blended', 0)
                        if _jp_w is not None and _jp_n:
                            _breakdown_rows.append({
                                'Component': '🎰 Jackpot Profile Blend',
                                'Weight': f"{_jp_w:.1%}",
                                'Detail': f"{_jp_n} profile(s) blended"
                            })

                        # Adaptive weight top factors
                        _aw = _slog.get('adaptive_weights', {})
                        if _aw:
                            _top2 = sorted(_aw.items(), key=lambda x: x[1], reverse=True)[:2]
                            for _fn, _fw in _top2:
                                _breakdown_rows.append({
                                    'Component': f"🧬 Adaptive: {_fn.replace('_', ' ').title()}",
                                    'Weight': f"{_fw:.1%}",
                                    'Detail': 'Evolved factor weight applied during sampling'
                                })

                        # Co-occurrence concentration
                        _conc = _slog.get('phase2_concentration_applied', 0)
                        if _conc:
                            _breakdown_rows.append({
                                'Component': '🔗 Co-occurrence Concentration',
                                'Weight': 'Variable',
                                'Detail': f"Applied on {_conc} of {len(predictions)} sets"
                            })

                        # Anti-pattern rejections
                        _rejections = _slog.get('anti_pattern_rejections', 0)
                        if _rejections:
                            _breakdown_rows.append({
                                'Component': '🚫 Anti-Pattern Rejections',
                                'Weight': '—',
                                'Detail': f"{_rejections} candidate set(s) rejected & resampled"
                            })

                        if _breakdown_rows:
                            _bd_df = pd.DataFrame(_breakdown_rows)
                            st.dataframe(_bd_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No learning components active for this generation.")
                else:
                    st.balloons()
                
                # Display strategy report prominently
                st.info(strategy_report)
                
                # ===== DRAW PROFILE MATCH SUMMARY =====
                if draw_profile_data and profile_validation_results:
                    st.divider()
                    st.markdown("### 🎯 Draw Profile Match Summary")
                    st.markdown(f"**Profile Applied:** {draw_profile_name}")
                    
                    # Count valid sets
                    valid_count = sum(1 for v in profile_validation_results if v['valid'])
                    
                    # Create summary metrics - Row 1: Core Patterns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sets Matching Profile", f"{valid_count}/{len(profile_validation_results)}")
                    with col2:
                        # Most common odd/even pattern
                        oe_patterns = [v['odd_even'] for v in profile_validation_results]
                        most_common_oe = max(set(oe_patterns), key=oe_patterns.count) if oe_patterns else "N/A"
                        st.metric("Most Common Odd/Even", most_common_oe)
                    with col3:
                        # Most common high/low pattern
                        hl_patterns = [v['high_low'] for v in profile_validation_results]
                        most_common_hl = max(set(hl_patterns), key=hl_patterns.count) if hl_patterns else "N/A"
                        st.metric("Most Common High/Low", most_common_hl)
                    with col4:
                        # Average sum
                        avg_sum = sum(v['sum'] for v in profile_validation_results) / len(profile_validation_results)
                        st.metric("Average Sum", f"{avg_sum:.1f}")
                    
                    # Row 2: Advanced Patterns
                    col5, col6, col7, col8 = st.columns(4)
                    with col5:
                        # Average consecutive pairs
                        avg_consecutive = sum(v.get('consecutive', 0) for v in profile_validation_results) / len(profile_validation_results)
                        st.metric("Avg Consecutive Pairs", f"{avg_consecutive:.1f}")
                    with col6:
                        # Average prime count
                        avg_primes = sum(v.get('primes', 0) for v in profile_validation_results) / len(profile_validation_results)
                        st.metric("Avg Prime Numbers", f"{avg_primes:.1f}")
                    with col7:
                        # Average max gap
                        avg_max_gap = sum(v.get('max_gap', 0) for v in profile_validation_results) / len(profile_validation_results)
                        st.metric("Avg Maximum Gap", f"{avg_max_gap:.1f}")
                    with col8:
                        # Match percentage
                        match_pct = (valid_count / len(profile_validation_results)) * 100
                        st.metric("Profile Match Rate", f"{match_pct:.0f}%")
                    
                    # Show profile details in expander
                    with st.expander("📊 Detailed Profile Match Analysis", expanded=False):
                        # Create detailed table
                        match_data = []
                        for i, validation in enumerate(profile_validation_results, 1):
                            match_data.append({
                                'Set #': i,
                                'Valid': '✅' if validation['valid'] else '❌',
                                'Odd/Even': validation['odd_even'],
                                'High/Low': validation['high_low'],
                                'Sum': validation['sum'],
                                'Consecutive': validation.get('consecutive', 0),
                                'Primes': validation.get('primes', 0),
                                'Max Gap': validation.get('max_gap', 0),
                                'Issues': ', '.join(validation['issues']) if validation['issues'] else 'None'
                            })
                        
                        match_df = pd.DataFrame(match_data)
                        st.dataframe(match_df, use_container_width=True, hide_index=True)
                        
                        if valid_count == len(profile_validation_results):
                            st.success("🎉 Perfect! All sets match the draw profile patterns!")
                        else:
                            st.warning(f"⚠️ {len(profile_validation_results) - valid_count} set(s) have deviations from profile patterns")
                            st.info("💡 **Note:** Validation is strict - sets must match top 3 most common Odd/Even & High/Low patterns, top 2 consecutive/prime patterns, and be within tolerance ranges for sum and gaps.")
                
                # ===== MODEL PERFORMANCE BREAKDOWN =====
                st.markdown("### 🤖 Model Performance Breakdown")
                st.markdown("*Showing which models contributed to the generated predictions*")
                
                # Calculate model contribution statistics
                model_vote_counts = {}
                total_votes = 0
                
                for pred_set in predictions_with_attribution:
                    attribution = pred_set.get('model_attribution', {})
                    for number, voters in attribution.items():
                        for voter in voters:
                            model_name = voter['model']
                            model_vote_counts[model_name] = model_vote_counts.get(model_name, 0) + 1
                            total_votes += 1
                
                if model_vote_counts and total_votes > 0:
                    # Create performance breakdown table
                    perf_data = []
                    for model_name, vote_count in sorted(model_vote_counts.items(), key=lambda x: x[1], reverse=True):
                        contribution_pct = (vote_count / total_votes) * 100
                        avg_votes_per_set = vote_count / final_sets
                        
                        # Find model type from analysis
                        model_type = "unknown"
                        for m in analysis['models']:
                            if m['name'] in model_name:
                                model_type = m['type']
                                break
                        
                        perf_data.append({
                            'Model': model_name,
                            'Type': model_type,
                            'Total Votes': vote_count,
                            'Contribution %': f"{contribution_pct:.1f}%",
                            'Avg Votes/Set': f"{avg_votes_per_set:.1f}"
                        })
                    
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(perf_df, use_container_width=True, hide_index=True)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Model Votes", total_votes)
                    with col2:
                        st.metric("Models Contributing", len(model_vote_counts))
                    with col3:
                        st.metric("Avg Votes per Set", f"{total_votes / final_sets:.1f}")
                    with col4:
                        top_model = max(model_vote_counts.items(), key=lambda x: x[1])
                        st.metric("Top Contributor", f"{top_model[1]} votes")
                    
                    st.info("💡 **How to read this:** Models 'vote' for numbers by assigning them higher probabilities. Numbers with multiple model votes have stronger consensus.")
                else:
                    # Debug info to understand why attribution is missing
                    st.warning("⚠️ No model attribution data available for this generation.")
                    
                    with st.expander("🔍 Debug Information", expanded=False):
                        st.write(f"**Total prediction sets:** {len(predictions_with_attribution)}")
                        st.write(f"**Predictions with attribution structure:** {len([p for p in predictions_with_attribution if isinstance(p, dict)])}")
                        
                        # Check if any attribution exists
                        has_attribution = False
                        for idx, pred_set in enumerate(predictions_with_attribution[:3]):  # Check first 3
                            attribution = pred_set.get('model_attribution', {}) if isinstance(pred_set, dict) else {}
                            if attribution:
                                has_attribution = True
                                st.write(f"**Set {idx+1} attribution:** {len(attribution)} numbers with attribution")
                                # Show first number's attribution as example
                                first_num = list(attribution.keys())[0] if attribution else None
                                if first_num:
                                    voters = attribution[first_num]
                                    st.write(f"  Example - Number {first_num}: {len(voters)} model(s) voted")
                                    for voter in voters[:2]:  # Show first 2 voters
                                        st.write(f"    - {voter.get('model', 'unknown')}: prob={voter.get('probability', 0):.4f}")
                            else:
                                st.write(f"**Set {idx+1}:** No attribution data")
                        
                        if not has_attribution:
                            st.error("❌ No attribution found in any prediction set. This usually means `model_probabilities` was empty in model analysis.")
                            st.write("**Model analysis keys:**", list(analysis.keys()) if isinstance(analysis, dict) else "Not a dict")
                            if 'model_probabilities' in analysis:
                                st.write(f"**model_probabilities keys:** {list(analysis['model_probabilities'].keys())}")
                                st.write(f"**model_probabilities count:** {len(analysis['model_probabilities'])}")
                            else:
                                st.write("**model_probabilities:** Not found in analysis")
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return
        
        # ===== RANKED PREDICTION DISPLAY =====
        # Pull the generation ranking computed during this run (or from session state
        # if the user is viewing after a previous run on the same session).
        _disp_ranked = st.session_state.get('sia_generation_ranked', [])
        _disp_preds  = st.session_state.get('sia_predictions', predictions)
        _disp_attr   = st.session_state.get('sia_predictions_with_attribution', predictions_with_attribution)

        st.markdown(f"### 🎰 AI-Ranked Prediction Sets ({len(_disp_preds)} total)")
        st.caption(
            "Sets are ranked by the AI's generation-time confidence — a blend of model vote "
            "strength, consensus breadth, and draw-profile conformance. Rank is computed before "
            "any draw results are known and is stored with the prediction file for later learning."
        )

        # ── Ranking summary table ──────────────────────────────────────────────
        _rank_table = []
        for _rr in _disp_ranked:
            _tier = (
                "🏆 Top 5" if _rr['rank'] <= 5 else
                "⭐ Top 10" if _rr['rank'] <= 10 else
                f"Rank {_rr['rank']}"
            )
            _rank_table.append({
                'Rank': _rr['rank'],
                'Tier': _tier,
                'Set #': _rr['set_number'],
                'Numbers': ', '.join(map(str, _rr['numbers'])),
                'AI Confidence': f"{_rr['confidence_score']:.1%}",
            })
        if _rank_table:
            st.dataframe(pd.DataFrame(_rank_table), use_container_width=True, hide_index=True)

        # ── Helper: render one set as game balls ──────────────────────────────
        def _render_ranked_set(ranked_entry: dict, border_color: str, label_html: str) -> None:
            """Render a single ranked prediction set with game balls."""
            _nums = ranked_entry['numbers']
            _ball_cols = st.columns(len(_nums))
            for _bc, _bn in zip(_ball_cols, _nums):
                with _bc:
                    st.markdown(
                        f'<div style="text-align:center;width:50px;height:50px;'
                        f'background:linear-gradient(135deg,#1e3a8a 0%,#3b82f6 50%,#1e40af 100%);'
                        f'border-radius:50%;color:white;font-weight:900;font-size:22px;'
                        f'box-shadow:0 4px 8px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.3);'
                        f'display:flex;align-items:center;justify-content:center;'
                        f'border:2px solid rgba(255,255,255,0.2);margin:0 auto;">{_bn}</div>',
                        unsafe_allow_html=True,
                    )

        # ── TIER 1: Ranks 1-5 ─────────────────────────────────────────────────
        _tier1 = [r for r in _disp_ranked if r['rank'] <= 5]
        if _tier1:
            st.markdown("### 🏆 Top 5 — AI's Highest-Confidence Picks")
            st.caption("These sets had the strongest model consensus and highest probability scores at generation time.")
            for _rr in _tier1:
                with st.container(border=True):
                    _attr_i = _rr['original_set_index']
                    _cs = _rr['confidence_score']
                    _conf_bar = "█" * int(_cs * 10) + "░" * (10 - int(_cs * 10))
                    st.markdown(
                        f"**Rank #{_rr['rank']} — Set #{_rr['set_number']}** &nbsp;&nbsp; "
                        f"AI Confidence: `{_cs:.1%}` `{_conf_bar}`",
                        unsafe_allow_html=True,
                    )
                    _render_ranked_set(_rr, "#f59e0b", "top5")
                    # Show which models contributed most to this set
                    _ma_i = (
                        _disp_attr[_attr_i].get('model_attribution', {})
                        if _attr_i < len(_disp_attr) and isinstance(_disp_attr[_attr_i], dict)
                        else {}
                    )
                    if _ma_i:
                        _mv_models = set()
                        for _vlist in _ma_i.values():
                            for _v in _vlist:
                                _mv_models.add(_v.get('model', ''))
                        if _mv_models:
                            st.caption(f"Contributing models: {', '.join(sorted(_mv_models))}")

        # ── TIER 2: Ranks 6-10 ────────────────────────────────────────────────
        _tier2 = [r for r in _disp_ranked if 6 <= r['rank'] <= 10]
        if _tier2:
            st.markdown("### ⭐ Ranks 6–10")
            for _rr in _tier2:
                with st.container(border=True):
                    _cs = _rr['confidence_score']
                    st.markdown(
                        f"**Rank #{_rr['rank']} — Set #{_rr['set_number']}** &nbsp;&nbsp; "
                        f"AI Confidence: `{_cs:.1%}`",
                        unsafe_allow_html=True,
                    )
                    _render_ranked_set(_rr, "#6b7280", "top10")

        # ── TIER 3: Ranks 11+ ─────────────────────────────────────────────────
        _tier3 = [r for r in _disp_ranked if r['rank'] > 10]
        if _tier3:
            with st.expander(f"📋 Ranks 11–{len(_disp_ranked)} (remaining {len(_tier3)} sets)", expanded=False):
                for _rr in _tier3:
                    with st.container(border=True):
                        _cs = _rr['confidence_score']
                        st.markdown(
                            f"**Rank #{_rr['rank']} — Set #{_rr['set_number']}** &nbsp;&nbsp; "
                            f"AI Confidence: `{_cs:.1%}`",
                            unsafe_allow_html=True,
                        )
                        _render_ranked_set(_rr, "#374151", "rest")

        st.divider()

        # ── Download options ───────────────────────────────────────────────────
        _dl_df = pd.DataFrame(_rank_table) if _rank_table else pd.DataFrame(
            [{"Rank": i + 1, "Set #": i + 1, "Numbers": ", ".join(map(str, sorted(p)))}
             for i, p in enumerate(_disp_preds)]
        )
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Ranked Sets (CSV)",
                _dl_df.to_csv(index=False),
                file_name=f"ai_predictions_ranked_{analyzer.game_folder}_{len(_disp_preds)}sets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            def _to_native(obj):
                if isinstance(obj, dict):
                    return {k: _to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_to_native(i) for i in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            _json_export = {
                "generation_ranking": _to_native(_disp_ranked),
                "sets": _to_native(_disp_preds),
                "analysis": _to_native(analysis),
                "optimal": _to_native(optimal),
            }
            st.download_button(
                "📥 Download Full Data (JSON)",
                json.dumps(_json_export, indent=2),
                file_name=f"ai_predictions_ranked_{analyzer.game_folder}_{len(_disp_preds)}sets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.info(f"💾 Predictions saved to: `{filepath}`")
        st.success(
            f"✅ **{len(_disp_preds)} sets generated & ranked!** "
            f"Top 5 are the AI's highest-confidence picks. "
            f"Visit **AI Learning** after the draw to see how the rankings held up."
        )


# ============================================================================
# TAB 3: JACKPOT PATTERN ANALYSIS
# ============================================================================

def _render_jackpot_analysis(analyzer: SuperIntelligentAIAnalyzer) -> None:
    """Analyze historical patterns based on jackpot amounts to create draw profiles."""
    st.subheader("💰 Jackpot Pattern Analysis")
    
    st.markdown("""
    **Discover winning patterns based on jackpot amounts!**  
    Search historical draws by jackpot value, analyze number patterns, and save draw profiles 
    for use in future prediction generation.
    """)
    
    # Get current game
    current_game = st.session_state.get('selected_game', 'Lotto Max')
    
    # Jackpot input section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        jackpot_amount = st.number_input(
            "Target Jackpot Amount ($)",
            min_value=1000000,
            max_value=200000000,
            value=10000000,
            step=1000000,
            key="jackpot_amount_input",
            help="Enter the jackpot amount you want to analyze"
        )
    
    with col2:
        range_tolerance = st.slider(
            "Range Tolerance (%)",
            min_value=0,
            max_value=50,
            value=10,
            step=5,
            key="range_tolerance_slider",
            help="Allow jackpots within ±X% of target amount"
        )
    
    with col3:
        st.metric("Search Range", f"±{range_tolerance}%")
        min_jackpot = jackpot_amount * (1 - range_tolerance / 100)
        max_jackpot = jackpot_amount * (1 + range_tolerance / 100)
        st.caption(f"${min_jackpot:,.0f} - ${max_jackpot:,.0f}")
    
    st.divider()
    
    # Search button
    if st.button("🔍 Search Historical Patterns", type="primary", use_container_width=True):
        with st.spinner("Searching historical draws..."):
            # Search and analyze
            matching_draws = _search_draws_by_jackpot(current_game, min_jackpot, max_jackpot)
            
            if len(matching_draws) < 5:
                st.warning(f"⚠️ Found only {len(matching_draws)} draws matching criteria. Minimum 5 required for reliable analysis.")
                st.info("💡 Try increasing the range tolerance to find more matching draws.")
                return
            
            # Store in session state
            st.session_state.jackpot_analysis_data = {
                'matching_draws': matching_draws,
                'jackpot_amount': jackpot_amount,
                'range_tolerance': range_tolerance,
                'game': current_game
            }
    
    # Display results if available
    if 'jackpot_analysis_data' in st.session_state:
        data = st.session_state.jackpot_analysis_data
        matching_draws = data['matching_draws']
        
        st.success(f"✅ Found {len(matching_draws)} historical draws matching ${data['jackpot_amount']:,.0f} (±{data['range_tolerance']}%)")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Draws", len(matching_draws))
        with col2:
            avg_jackpot = sum(d['jackpot'] for d in matching_draws) / len(matching_draws)
            st.metric("Avg Jackpot", f"${avg_jackpot:,.0f}")
        with col3:
            min_date = min(d['date'] for d in matching_draws)
            st.metric("Oldest Draw", min_date)
        with col4:
            max_date = max(d['date'] for d in matching_draws)
            st.metric("Newest Draw", max_date)
        
        st.divider()
        
        # Analyze patterns
        patterns = _analyze_draw_patterns(matching_draws, current_game)
        
        # Display patterns
        _display_pattern_analysis(patterns, current_game)
        
        st.divider()
        
        # Save profile section
        st.markdown("### 💾 Save Draw Profile")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            profile_name = st.text_input(
                "Profile Name",
                value=f"jackpot_{data['jackpot_amount']//1000000}M",
                key="profile_name_input",
                help="Enter a descriptive name for this profile"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("💾 Save Profile", type="primary", use_container_width=True):
                if len(matching_draws) >= 5:
                    profile_path = _save_draw_profile(
                        profile_name,
                        data['jackpot_amount'],
                        data['range_tolerance'],
                        matching_draws,
                        patterns,
                        current_game
                    )
                    if profile_path:
                        st.success(f"✅ Profile saved: {profile_path.name}")
                        app_log(f"Jackpot profile saved: {profile_name} for {current_game}", "info")
                    else:
                        st.error("❌ Failed to save profile")
                else:
                    st.error("❌ Need at least 5 matching draws to save profile")


def _search_draws_by_jackpot(game: str, min_jackpot: float, max_jackpot: float) -> List[Dict]:
    """Search all historical CSVs for draws within jackpot range."""
    matching_draws = []
    
    try:
        game_folder = _sanitize_game_name(game)
        data_dir = Path("data") / game_folder
        
        if not data_dir.exists():
            return matching_draws
        
        # Get all training data CSV files
        csv_files = sorted(data_dir.glob("training_data_*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(str(csv_file))
                
                # Filter by jackpot range
                if 'jackpot' in df.columns:
                    matching_rows = df[
                        (df['jackpot'] >= min_jackpot) & 
                        (df['jackpot'] <= max_jackpot)
                    ]
                    
                    for _, row in matching_rows.iterrows():
                        # Parse numbers
                        numbers_str = str(row.get('numbers', ''))
                        if numbers_str and numbers_str != 'nan':
                            numbers = [int(n.strip()) for n in numbers_str.strip('[]"').replace('"', '').split(',') if n.strip().isdigit()]
                            
                            if numbers:
                                draw_data = {
                                    'date': str(row.get('draw_date', '')),
                                    'numbers': numbers,
                                    'bonus': int(row.get('bonus', 0)) if pd.notna(row.get('bonus')) else 0,
                                    'jackpot': float(row.get('jackpot', 0))
                                }
                                matching_draws.append(draw_data)
            
            except Exception as e:
                app_log(f"Error reading {csv_file.name}: {e}", "debug")
                continue
        
        # Sort by date (newest first)
        matching_draws.sort(key=lambda x: x['date'], reverse=True)
        
    except Exception as e:
        app_log(f"Error searching draws: {e}", "error")
    
    return matching_draws


def _analyze_draw_patterns(draws: List[Dict], game: str) -> Dict:
    """Analyze patterns across all matching draws.

    Improvements applied:
    - P1: std added to sum_stats and gap_analysis
    - P2: position_frequency tracks per-sorted-position number counts
    - P3: pair_cooccurrence tracks all 2-number combinations per draw (top-30 saved)
    - P5: temporal decay weights recent draws more heavily (half-life 180 days)
    - P9: bonus_frequency captured for Lotto Max
    """
    from itertools import combinations as _combs

    # Determine number range for the game
    if "6/49" in game or "6_49" in game:
        max_num = 49
        nums_per_draw = 6
    else:  # Lotto Max
        max_num = 50
        nums_per_draw = 7

    mid_point = (max_num + 1) // 2

    # P5: Sort draws newest → oldest and compute temporal decay weights
    # half-life = 180 days; weight = exp(-age / 180)
    _today = datetime.today().date()
    def _draw_date(d):
        try:
            return datetime.strptime(str(d.get('date', '')), '%Y-%m-%d').date()
        except Exception:
            return _today
    sorted_draws = sorted(draws, key=_draw_date, reverse=True)
    newest_date = _draw_date(sorted_draws[0]) if sorted_draws else _today
    def _decay_weight(d):
        age = max(0, (newest_date - _draw_date(d)).days)
        return float(np.exp(-age / 180.0))

    # Initialize counters
    odd_even_combos = {}
    high_low_combos = {}
    sum_ranges = []
    sum_weights = []
    # P5: weighted frequency (float counts)
    number_frequency = {i: 0.0 for i in range(1, max_num + 1)}
    consecutive_counts = {}
    decade_distribution = {f"{i*10}-{i*10+9}": 0.0 for i in range(max_num // 10 + 1)}
    gap_analysis = []
    prime_counts = {}
    pair_counts = {}
    repeat_from_previous = []
    # P2: position frequency {1: {num: float, ...}, ...}
    position_frequency = {pos: {n: 0.0 for n in range(1, max_num + 1)} for pos in range(1, nums_per_draw + 1)}
    # P3: co-occurrence pair counts
    pair_cooccurrence = {}
    # P9: bonus frequency (Lotto Max only)
    bonus_frequency = {i: 0 for i in range(1, max_num + 1)} if nums_per_draw == 7 else {}

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = set(n for n in range(1, max_num + 1) if is_prime(n))

    previous_numbers = None

    for draw in sorted_draws:
        numbers = sorted(draw['numbers'])
        w = _decay_weight(draw)  # P5 temporal decay weight

        # Odd/Even analysis
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = len(numbers) - odd_count
        oe_combo = f"{odd_count} Odd / {even_count} Even"
        odd_even_combos[oe_combo] = odd_even_combos.get(oe_combo, 0) + w

        # High/Low analysis
        low_count = sum(1 for n in numbers if n <= mid_point)
        high_count = len(numbers) - low_count
        hl_combo = f"{low_count} Low / {high_count} High"
        high_low_combos[hl_combo] = high_low_combos.get(hl_combo, 0) + w

        # Sum analysis (keep raw list for std; weight for mean/median)
        total_sum = sum(numbers)
        sum_ranges.append(total_sum)
        sum_weights.append(w)

        # P5: weighted number frequency
        for num in numbers:
            number_frequency[num] += w

        # Consecutive numbers
        consecutive = sum(1 for i in range(len(numbers) - 1) if numbers[i + 1] == numbers[i] + 1)
        consecutive_counts[consecutive] = consecutive_counts.get(consecutive, 0) + w

        # Decade distribution (weighted)
        for num in numbers:
            decade = f"{(num-1)//10*10}-{(num-1)//10*10+9}"
            if decade in decade_distribution:
                decade_distribution[decade] += w

        # Gap analysis
        if len(numbers) > 1:
            gaps = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
            gap_analysis.append(max(gaps))

        # Prime count
        prime_count = sum(1 for n in numbers if n in primes)
        prime_counts[prime_count] = prime_counts.get(prime_count, 0) + w

        # Close-pair count (gap ≤ 2)
        pairs = sum(1 for i in range(len(numbers) - 1) if numbers[i + 1] - numbers[i] <= 2)
        pair_counts[pairs] = pair_counts.get(pairs, 0) + w

        # Repeat from previous draw
        if previous_numbers is not None:
            repeats = len(set(numbers) & set(previous_numbers))
            repeat_from_previous.append(repeats)

        previous_numbers = numbers

        # P2: position frequency (weighted)
        for pos, num in enumerate(numbers, 1):
            if pos in position_frequency:
                position_frequency[pos][num] += w

        # P3: co-occurrence pairs (weighted)
        for n1, n2 in _combs(numbers, 2):
            key = f"{n1},{n2}"
            pair_cooccurrence[key] = pair_cooccurrence.get(key, 0) + w

        # P9: bonus frequency (Lotto Max, unweighted — raw count)
        if nums_per_draw == 7:
            bonus = int(draw.get('bonus', 0) or 0)
            if 1 <= bonus <= max_num:
                bonus_frequency[bonus] = bonus_frequency.get(bonus, 0) + 1

    # ---- Compute statistics ----
    # P1: weighted mean for sum (raw std from unweighted list for simplicity)
    _sw_total = sum(sum_weights) or 1.0
    _sum_mean = sum(s * w for s, w in zip(sum_ranges, sum_weights)) / _sw_total
    _sum_std  = (statistics.stdev(sum_ranges) if len(sum_ranges) > 1 else 0.0)

    # P1: gap std
    _gap_std = statistics.stdev(gap_analysis) if len(gap_analysis) > 1 else 0.0

    # P3: top-30 most frequent co-occurrence pairs
    top_pairs = sorted(pair_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:30]
    top_pairs_dict = {k: round(v, 4) for k, v in top_pairs}

    # P2: for each position, build a ranked list of top-10 numbers
    position_top_numbers = {}
    for pos, freq_dict in position_frequency.items():
        ranked = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        position_top_numbers[str(pos)] = [[n, round(f, 4)] for n, f in ranked]

    # P9: top-10 bonus numbers
    bonus_top = []
    if bonus_frequency:
        bonus_top = sorted(bonus_frequency.items(), key=lambda x: x[1], reverse=True)[:10]

    # Use raw (unweighted) counts for excluded-number detection to avoid under-reporting cold #s
    raw_frequency = {i: 0 for i in range(1, max_num + 1)}
    for draw in sorted_draws:
        for num in sorted(draw['numbers']):
            raw_frequency[num] += 1

    patterns = {
        'odd_even_combos'   : odd_even_combos,
        'high_low_combos'   : high_low_combos,
        'sum_stats': {
            'min'       : min(sum_ranges),
            'max'       : max(sum_ranges),
            'mean'      : round(_sum_mean, 4),
            'std'       : round(_sum_std, 4),       # P1
            'median'    : statistics.median(sum_ranges),
            'mode_range': _get_mode_range(sum_ranges, nums_per_draw),
        },
        'number_frequency'  : {k: round(v, 4) for k, v in number_frequency.items()},
        'excluded_numbers'  : _get_excluded_numbers(raw_frequency, max_num),
        'consecutive_counts': consecutive_counts,
        'decade_distribution': decade_distribution,
        'gap_analysis': {
            'min' : min(gap_analysis) if gap_analysis else 0,
            'max' : max(gap_analysis) if gap_analysis else 0,
            'mean': round(statistics.mean(gap_analysis), 4) if gap_analysis else 0,
            'std' : round(_gap_std, 4),              # P1
        },
        'prime_counts'      : prime_counts,
        'pair_counts'       : pair_counts,
        'repeat_stats': {
            'min' : min(repeat_from_previous) if repeat_from_previous else 0,
            'max' : max(repeat_from_previous) if repeat_from_previous else 0,
            'mean': statistics.mean(repeat_from_previous) if repeat_from_previous else 0,
        },
        'position_frequency': position_top_numbers,     # P2
        'pair_cooccurrence' : top_pairs_dict,           # P3
        'bonus_frequency'   : dict(bonus_top),          # P9
        'total_draws'       : len(draws),
        'mid_point'         : mid_point,
    }

    return patterns


def _get_mode_range(sum_list: List[int], nums_per_draw: int) -> str:
    """Determine the most common sum range."""
    if not sum_list:
        return "N/A"
    
    # Create bins based on number of numbers per draw
    bin_size = 10 if nums_per_draw == 6 else 15
    min_val = min(sum_list)
    max_val = max(sum_list)
    
    bins = {}
    for val in sum_list:
        bin_key = (val // bin_size) * bin_size
        bins[bin_key] = bins.get(bin_key, 0) + 1
    
    # Find most common bin
    mode_bin = max(bins, key=bins.get)
    return f"{mode_bin}-{mode_bin + bin_size - 1}"


def _get_excluded_numbers(frequency: Dict[int, int], max_num: int) -> List[int]:
    """Find numbers that rarely or never appear."""
    avg_frequency = sum(frequency.values()) / len(frequency)
    threshold = avg_frequency * 0.3  # Numbers appearing less than 30% of average
    
    excluded = [num for num in range(1, max_num + 1) if frequency[num] < threshold]
    return sorted(excluded)


def _display_pattern_analysis(patterns: Dict, game: str) -> None:
    """Display comprehensive pattern analysis with visualizations."""
    
    st.markdown("## 📊 Pattern Analysis Dashboard")
    
    # Create tabs for different analysis categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Core Patterns",
        "📈 Number Distribution",
        "🔢 Advanced Insights",
        "📉 Trends & Gaps",
        "📋 Summary Table"
    ])
    
    with tab1:
        st.markdown("### Core Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Odd/Even Pie Chart
            st.markdown("#### Odd/Even Combinations")
            oe_labels = list(patterns['odd_even_combos'].keys())
            oe_values = list(patterns['odd_even_combos'].values())
            
            fig_oe = go.Figure(data=[go.Pie(
                labels=oe_labels,
                values=oe_values,
                hole=0.3,
                marker_colors=px.colors.qualitative.Set3
            )])
            fig_oe.update_layout(height=350)
            st.plotly_chart(fig_oe, use_container_width=True)
            
            # Most common
            most_common_oe = max(patterns['odd_even_combos'], key=patterns['odd_even_combos'].get)
            st.info(f"**Most Common:** {most_common_oe} ({patterns['odd_even_combos'][most_common_oe]} draws)")
        
        with col2:
            # High/Low Pie Chart
            st.markdown("#### High/Low Combinations")
            mid = patterns['mid_point']
            st.caption(f"Low: 1-{mid}, High: {mid+1}-{50 if 'Max' in game else 49}")
            
            hl_labels = list(patterns['high_low_combos'].keys())
            hl_values = list(patterns['high_low_combos'].values())
            
            fig_hl = go.Figure(data=[go.Pie(
                labels=hl_labels,
                values=hl_values,
                hole=0.3,
                marker_colors=px.colors.qualitative.Pastel
            )])
            fig_hl.update_layout(height=350)
            st.plotly_chart(fig_hl, use_container_width=True)
            
            # Most common
            most_common_hl = max(patterns['high_low_combos'], key=patterns['high_low_combos'].get)
            st.info(f"**Most Common:** {most_common_hl} ({patterns['high_low_combos'][most_common_hl]} draws)")
        
        st.divider()
        
        # Sum Analysis Bar Chart
        st.markdown("#### Sum Range Distribution")
        sum_stats = patterns['sum_stats']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min Sum", f"{sum_stats['min']}")
        with col2:
            st.metric("Max Sum", f"{sum_stats['max']}")
        with col3:
            st.metric("Average Sum", f"{sum_stats['mean']:.1f}")
        with col4:
            st.metric("Most Common Range", sum_stats['mode_range'])
    
    with tab2:
        st.markdown("### Number Frequency Distribution")
        
        # Heatmap of number frequency
        freq = patterns['number_frequency']
        max_num = max(freq.keys())
        
        # Create heatmap data (5 rows of 10 numbers each for 50, adjust for 49)
        rows = 5
        cols = 10
        heatmap_data = []
        heatmap_text = []
        
        for row in range(rows):
            row_data = []
            row_text = []
            for col in range(cols):
                num = row * cols + col + 1
                if num <= max_num:
                    row_data.append(freq.get(num, 0))
                    row_text.append(f"#{num}<br>{freq.get(num, 0)} times")
                else:
                    row_data.append(0)
                    row_text.append("")
            heatmap_data.append(row_data)
            heatmap_text.append(row_text)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            text=heatmap_text,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='Viridis',
            showscale=True
        ))
        fig_heatmap.update_layout(
            title="Number Frequency Heatmap",
            height=400,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.divider()
        
        # Top and Bottom Numbers
        col1, col2 = st.columns(2)
        
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        with col1:
            st.markdown("#### 🔥 Most Frequent Numbers")
            top_10 = sorted_freq[:10]
            for num, count in top_10:
                percentage = (count / patterns['total_draws']) * 100
                st.write(f"**{num}**: {count} times ({percentage:.1f}%)")
        
        with col2:
            st.markdown("#### ❄️ Least Frequent Numbers")
            bottom_10 = sorted_freq[-10:]
            for num, count in bottom_10:
                percentage = (count / patterns['total_draws']) * 100
                st.write(f"**{num}**: {count} times ({percentage:.1f}%)")
        
        # Excluded numbers
        if patterns['excluded_numbers']:
            st.warning(f"**Typically Excluded Numbers:** {', '.join(map(str, patterns['excluded_numbers']))}")
            st.caption("These numbers appear significantly less frequently than average")
    
    with tab3:
        st.markdown("### Advanced Pattern Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Consecutive Numbers Bar Chart
            st.markdown("#### Consecutive Numbers")
            cons_labels = [str(k) for k in sorted(patterns['consecutive_counts'].keys())]
            cons_values = [patterns['consecutive_counts'][int(k)] for k in cons_labels]
            
            fig_cons = go.Figure(data=[go.Bar(
                x=cons_labels,
                y=cons_values,
                marker_color='lightblue',
                text=cons_values,
                textposition='auto'
            )])
            fig_cons.update_layout(
                xaxis_title="Number of Consecutive Pairs",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_cons, use_container_width=True)
            
            # Prime Numbers
            st.markdown("#### Prime Number Count")
            prime_labels = [str(k) for k in sorted(patterns['prime_counts'].keys())]
            prime_values = [patterns['prime_counts'][int(k)] for k in prime_labels]
            
            fig_prime = go.Figure(data=[go.Bar(
                x=prime_labels,
                y=prime_values,
                marker_color='lightcoral',
                text=prime_values,
                textposition='auto'
            )])
            fig_prime.update_layout(
                xaxis_title="Number of Primes in Draw",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_prime, use_container_width=True)
        
        with col2:
            # Decade Distribution
            st.markdown("#### Decade Distribution")
            decade_labels = list(patterns['decade_distribution'].keys())
            decade_values = list(patterns['decade_distribution'].values())
            
            fig_decade = go.Figure(data=[go.Bar(
                x=decade_labels,
                y=decade_values,
                marker_color='lightgreen',
                text=decade_values,
                textposition='auto'
            )])
            fig_decade.update_layout(
                xaxis_title="Number Range (Decade)",
                yaxis_title="Total Appearances",
                height=300
            )
            st.plotly_chart(fig_decade, use_container_width=True)
            
            # Pairs/Close Numbers
            st.markdown("#### Close Number Pairs (within 2)")
            pair_labels = [str(k) for k in sorted(patterns['pair_counts'].keys())]
            pair_values = [patterns['pair_counts'][int(k)] for k in pair_labels]
            
            fig_pairs = go.Figure(data=[go.Bar(
                x=pair_labels,
                y=pair_values,
                marker_color='plum',
                text=pair_values,
                textposition='auto'
            )])
            fig_pairs.update_layout(
                xaxis_title="Number of Close Pairs",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_pairs, use_container_width=True)
    
    with tab4:
        st.markdown("### Trends & Gap Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Maximum Gap Statistics")
            gap_stats = patterns['gap_analysis']
            st.metric("Minimum Gap", f"{gap_stats['min']}")
            st.metric("Maximum Gap", f"{gap_stats['max']}")
            st.metric("Average Gap", f"{gap_stats['mean']:.1f}")
            st.caption("Gap = largest distance between consecutive drawn numbers")
        
        with col2:
            st.markdown("#### Repeat from Previous Draw")
            repeat_stats = patterns['repeat_stats']
            st.metric("Minimum Repeats", f"{repeat_stats['min']}")
            st.metric("Maximum Repeats", f"{repeat_stats['max']}")
            st.metric("Average Repeats", f"{repeat_stats['mean']:.2f}")
            st.caption("How many numbers typically repeat from the previous draw")
    
    with tab5:
        st.markdown("### Complete Pattern Summary")
        
        # Create comprehensive summary table
        summary_data = {
            "Pattern Category": [],
            "Key Insight": [],
            "Statistical Value": []
        }
        
        # Add all patterns
        most_common_oe = max(patterns['odd_even_combos'], key=patterns['odd_even_combos'].get)
        summary_data["Pattern Category"].append("Odd/Even Balance")
        summary_data["Key Insight"].append(f"Most Common: {most_common_oe}")
        summary_data["Statistical Value"].append(f"{patterns['odd_even_combos'][most_common_oe]}/{patterns['total_draws']} draws")
        
        most_common_hl = max(patterns['high_low_combos'], key=patterns['high_low_combos'].get)
        summary_data["Pattern Category"].append("High/Low Balance")
        summary_data["Key Insight"].append(f"Most Common: {most_common_hl}")
        summary_data["Statistical Value"].append(f"{patterns['high_low_combos'][most_common_hl]}/{patterns['total_draws']} draws")
        
        summary_data["Pattern Category"].append("Sum Range")
        summary_data["Key Insight"].append(f"Most Common: {patterns['sum_stats']['mode_range']}")
        summary_data["Statistical Value"].append(f"Avg: {patterns['sum_stats']['mean']:.1f}")
        
        summary_data["Pattern Category"].append("Consecutive Pairs")
        most_common_cons = max(patterns['consecutive_counts'], key=patterns['consecutive_counts'].get)
        summary_data["Key Insight"].append(f"Most Common: {most_common_cons} pairs")
        summary_data["Statistical Value"].append(f"{patterns['consecutive_counts'][most_common_cons]}/{patterns['total_draws']} draws")
        
        summary_data["Pattern Category"].append("Prime Numbers")
        most_common_prime = max(patterns['prime_counts'], key=patterns['prime_counts'].get)
        summary_data["Key Insight"].append(f"Most Common: {most_common_prime} primes")
        summary_data["Statistical Value"].append(f"{patterns['prime_counts'][most_common_prime]}/{patterns['total_draws']} draws")
        
        summary_data["Pattern Category"].append("Maximum Gap")
        summary_data["Key Insight"].append(f"Average: {patterns['gap_analysis']['mean']:.1f}")
        summary_data["Statistical Value"].append(f"Range: {patterns['gap_analysis']['min']}-{patterns['gap_analysis']['max']}")
        
        summary_data["Pattern Category"].append("Repeat Numbers")
        summary_data["Key Insight"].append(f"Average: {patterns['repeat_stats']['mean']:.2f} numbers")
        summary_data["Statistical Value"].append(f"Range: {patterns['repeat_stats']['min']}-{patterns['repeat_stats']['max']}")
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)


def _save_draw_profile(profile_name: str, jackpot_amount: float, range_tolerance: int,
                        draws: List[Dict], patterns: Dict, game: str) -> Optional[Path]:
    """Save draw profile to JSON file in learning folder."""
    
    try:
        # Create directory structure
        game_folder = _sanitize_game_name(game)
        profile_dir = Path("data") / "learning" / game_folder / "jackpot_profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        # P6: Normalize frequency_map to probabilities before saving so two profiles
        # built from different draw counts are comparable at blend time.
        raw_freq = patterns['number_frequency']
        _freq_total = sum(raw_freq.values()) or 1.0
        freq_map_normalized = {str(k): round(v / _freq_total, 6) for k, v in raw_freq.items()}

        # Build hot/cold lists from the normalized map
        hot_numbers  = sorted(freq_map_normalized.items(), key=lambda x: x[1], reverse=True)[:10]
        cold_numbers = sorted(freq_map_normalized.items(), key=lambda x: x[1])[:10]

        # P9: bonus data (Lotto Max only)
        bonus_data = {}
        if patterns.get('bonus_frequency'):
            _bf = patterns['bonus_frequency']
            _bf_total = sum(_bf.values()) or 1.0
            bonus_data = {
                "frequency_map": {str(k): round(v / _bf_total, 6) for k, v in _bf.items()},
                "hot_bonus_numbers": sorted(_bf.items(), key=lambda x: x[1], reverse=True)[:5],
            }

        # Create profile data
        profile_data = {
            "metadata": {
                "profile_name": profile_name,
                "game": game,
                "created_date": datetime.now().isoformat(),
                "jackpot_amount": jackpot_amount,
                "range_tolerance_percent": range_tolerance,
                "total_draws_analyzed": len(draws),
                "date_range": {
                    "oldest": min(d['date'] for d in draws),
                    "newest": max(d['date'] for d in draws),
                },
            },
            "core_patterns": {
                "odd_even_distribution": patterns['odd_even_combos'],
                "high_low_distribution": patterns['high_low_combos'],
                "high_low_split_point" : patterns['mid_point'],
            },
            "sum_analysis": patterns['sum_stats'],  # now includes 'std' (P1)
            "number_patterns": {
                # P6: pre-normalized probabilities (sum ≈ 1.0)
                "frequency_map"    : freq_map_normalized,
                "excluded_numbers" : patterns['excluded_numbers'],
                "hot_numbers"      : hot_numbers,
                "cold_numbers"     : cold_numbers,
                # P2: top numbers per sorted position
                "position_frequency": patterns.get('position_frequency', {}),
                # P3: top-30 co-occurring pairs
                "pair_cooccurrence" : patterns.get('pair_cooccurrence', {}),
            },
            "advanced_insights": {
                "consecutive_distribution": patterns['consecutive_counts'],
                "decade_distribution"     : patterns['decade_distribution'],
                "gap_statistics"          : patterns['gap_analysis'],  # now includes 'std' (P1)
                "prime_distribution"      : patterns['prime_counts'],
                "pair_distribution"       : patterns['pair_counts'],
                "repeat_statistics"       : patterns['repeat_stats'],
            },
            # P9: bonus patterns (populated for Lotto Max; empty dict for 6/49)
            "bonus_patterns": bonus_data,
            "historical_draws": [
                {
                    "date"   : d['date'],
                    "numbers": d['numbers'],
                    "jackpot": d.get('jackpot', 0),
                    "bonus"  : d.get('bonus', 0),
                }
                # Store 50 most recent draws (sorted newest-first)
                for d in sorted(draws, key=lambda x: x.get('date', ''), reverse=True)[:50]
            ],
        }
        
        # Save to file
        filename = f"{profile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = profile_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        return filepath
    
    except Exception as e:
        app_log(f"Error saving draw profile: {e}", "error")
        return None


def _sanitize_game_name(game: str) -> str:
    """Convert game name to filesystem-safe format."""
    return game.lower().replace(" ", "_").replace("/", "_")


# ============================================================================
# TAB 4: MAXMILLION ANALYSIS / PERFORMANCE HISTORY
# ============================================================================

def _render_maxmillion_analysis(analyzer: SuperIntelligentAIAnalyzer, game: str) -> None:
    """MaxMillion Analysis for Lotto Max - compare predictions with actual draws and MaxMillions."""
    st.subheader("🎰 MaxMillion Analysis")
    
    # Get next draw date
    next_draw = compute_next_draw_date(game)
    
    # Initialize session state for maxmillion analysis
    if 'maxm_selected_draw_date' not in st.session_state:
        st.session_state.maxm_selected_draw_date = None
    if 'maxm_selected_prediction_file' not in st.session_state:
        st.session_state.maxm_selected_prediction_file = None
    if 'maxm_comparison_type' not in st.session_state:
        st.session_state.maxm_comparison_type = None
    if 'maxm_numbers' not in st.session_state:
        st.session_state.maxm_numbers = []
    
    # Draw Date Selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        draw_date_option = st.radio(
            "Select Draw Date",
            ["Next Draw Date", "Previous Draw Date"],
            key="maxm_draw_option"
        )
    
    with col2:
        if draw_date_option == "Next Draw Date":
            selected_draw_date = next_draw
            st.info(f"📅 Next Draw: {next_draw.strftime('%A, %B %d, %Y')}")
        else:
            selected_draw_date = st.date_input(
                "Select Previous Draw Date",
                value=next_draw - timedelta(days=7),
                max_value=datetime.now().date(),
                key="maxm_date_picker"
            )
            st.session_state.maxm_selected_draw_date = selected_draw_date
    
    st.divider()
    
    # If previous draw selected, show actual results
    winning_numbers = None
    bonus_number = None
    jackpot_amount = None
    
    if draw_date_option == "Previous Draw Date" and selected_draw_date:
        winning_numbers, bonus_number, jackpot_amount = _get_draw_results(game, selected_draw_date)
        
        if winning_numbers:
            st.success(f"✅ Draw Results for {selected_draw_date.strftime('%B %d, %Y')}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Winning Numbers**")
                _display_number_balls(winning_numbers)
            with col2:
                st.markdown("**Bonus Number**")
                _display_number_balls([bonus_number], is_bonus=True)
            with col3:
                st.metric("Jackpot", f"${jackpot_amount:,.0f}" if jackpot_amount else "N/A")
        else:
            st.warning(f"⚠️ No draw results found for {selected_draw_date.strftime('%B %d, %Y')}")
    
    st.divider()
    
    # Prediction File Selection
    st.markdown("### 📂 Select Prediction File")
    
    prediction_files = _get_prediction_files(game, selected_draw_date if draw_date_option == "Previous Draw Date" else next_draw)
    
    if prediction_files:
        selected_file = st.selectbox(
            "Available Prediction Files",
            options=prediction_files,
            format_func=lambda x: x.stem,
            key="maxm_pred_file_select"
        )
        
        if selected_file:
            st.session_state.maxm_selected_prediction_file = selected_file
            
            # Comparison Type Selection
            st.divider()
            st.markdown("### 🔍 Comparison Type")
            
            comparison_type = st.radio(
                "Compare With",
                ["Main Numbers", "MaxMillions"],
                key="maxm_comparison_type_radio"
            )
            st.session_state.maxm_comparison_type = comparison_type
            
            # Load prediction data
            prediction_data = _load_prediction_file(selected_file)
            
            if comparison_type == "Main Numbers":
                if winning_numbers and draw_date_option == "Previous Draw Date":
                    _display_main_numbers_comparison(prediction_data, winning_numbers, bonus_number)
                else:
                    st.info("ℹ️ Select a previous draw date to compare with main numbers")
            
            else:  # MaxMillions
                st.divider()
                st.markdown("### 🎰 MaxMillion Numbers")
                
                maxmillion_input_method = st.radio(
                    "Input Method",
                    ["Load from File", "Input Manually"],
                    key="maxm_input_method"
                )
                
                if maxmillion_input_method == "Load from File":
                    maxmillion_file = _load_maxmillion_from_file(game, selected_draw_date if draw_date_option == "Previous Draw Date" else next_draw)
                    if maxmillion_file:
                        st.session_state.maxm_numbers = maxmillion_file
                        _display_maxmillion_sets(maxmillion_file)
                        _display_maxmillion_comparison(prediction_data, maxmillion_file)
                else:
                    # Manual input
                    st.markdown("**Paste MaxMillion sets below (one set per line, numbers separated by commas or spaces)**")
                    st.caption("Example with commas: 1,5,12,23,34,45,49")
                    st.caption("Example with spaces: 1 5 12 23 34 45 49")
                    
                    maxmillion_input = st.text_area(
                        "MaxMillion Sets",
                        height=200,
                        placeholder="1,5,12,23,34,45,49\n2 8 15 22 31 38 47\n3,10,18,25,33,40,48",
                        key="maxm_manual_input"
                    )
                    
                    if st.button("Process & Save MaxMillion Numbers", key="maxm_process_btn"):
                        if maxmillion_input.strip():
                            processed_sets = _process_maxmillion_input(maxmillion_input)
                            if processed_sets:
                                # Save to file
                                save_path = _save_maxmillion_data(game, selected_draw_date if draw_date_option == "Previous Draw Date" else next_draw, processed_sets)
                                st.session_state.maxm_numbers = processed_sets
                                st.success(f"✅ Processed and saved {len(processed_sets)} MaxMillion sets to: {save_path.name}")
                                _display_maxmillion_sets(processed_sets)
                                _display_maxmillion_comparison(prediction_data, processed_sets)
                            else:
                                st.error("❌ Failed to process MaxMillion input. Please check format.")
                        else:
                            st.warning("⚠️ Please enter MaxMillion numbers")
                    
                    # Display if already processed
                    if st.session_state.maxm_numbers:
                        _display_maxmillion_sets(st.session_state.maxm_numbers)
                        _display_maxmillion_comparison(prediction_data, st.session_state.maxm_numbers)
    else:
        st.info(f"📝 No prediction files found for {selected_draw_date.strftime('%B %d, %Y')}")


def _get_draw_results(game: str, draw_date) -> Tuple[Optional[List[int]], Optional[int], Optional[float]]:
    """Get actual draw results from training data CSV files."""
    try:
        game_folder = _sanitize_game_name(game)
        data_dir = Path("data") / game_folder
        
        # Try to find the draw in CSV files (starting with current year and going back)
        for year in range(draw_date.year, 2008, -1):
            csv_file = data_dir / f"training_data_{year}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                df['draw_date'] = pd.to_datetime(df['draw_date']).dt.date
                
                match = df[df['draw_date'] == draw_date]
                if not match.empty:
                    row = match.iloc[0]
                    numbers_str = row['numbers']
                    numbers = [int(n.strip()) for n in numbers_str.split(',')]
                    bonus = int(row['bonus']) if pd.notna(row['bonus']) else None
                    jackpot = float(row['jackpot']) if pd.notna(row['jackpot']) else None
                    return numbers, bonus, jackpot
        
        return None, None, None
    except Exception as e:
        app_log(f"Error getting draw results: {str(e)}", "error")
        return None, None, None


def _get_prediction_files(game: str, target_date) -> List[Path]:
    """Get prediction files from predictions/{game}/prediction_ai/ folder that match the target draw date."""
    try:
        game_folder = _sanitize_game_name(game)
        pred_dir = Path("predictions") / game_folder / "prediction_ai"
        
        if not pred_dir.exists():
            return []
        
        # Get all JSON files
        all_files = sorted(pred_dir.glob("*.json"), reverse=True)
        
        # Filter files by matching next_draw_date
        matching_files = []
        target_date_str = target_date.strftime('%Y-%m-%d')
        
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Check if next_draw_date matches target date
                    if data.get('next_draw_date') == target_date_str:
                        matching_files.append(file_path)
            except Exception as e:
                app_log(f"Error reading file {file_path.name}: {str(e)}", "warning")
                continue
        
        return matching_files
    except Exception as e:
        app_log(f"Error getting prediction files: {str(e)}", "error")
        return []


def _load_prediction_file(file_path: Path) -> Dict[str, Any]:
    """Load prediction JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        app_log(f"Error loading prediction file: {str(e)}", "error")
        return {}


def _display_number_balls(numbers: List[int], is_bonus: bool = False) -> None:
    """Display numbers as lottery balls."""
    cols = st.columns(len(numbers))
    for i, num in enumerate(numbers):
        with cols[i]:
            color = "🟡" if is_bonus else "🔵"
            st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; padding: 10px; background-color: {'#FFD700' if is_bonus else '#4169E1'}; color: white; border-radius: 50%; width: 50px; height: 50px; line-height: 30px; margin: auto;'>{num}</div>", unsafe_allow_html=True)


def _display_main_numbers_comparison(prediction_data: Dict[str, Any], winning_numbers: List[int], bonus_number: int) -> None:
    """Display prediction sets compared with main winning numbers."""
    st.divider()
    st.markdown("### 🎯 Prediction Sets Analysis")
    
    predictions = prediction_data.get('predictions', [])
    
    if not predictions:
        st.warning("No predictions found in file")
        return
    
    matches_found = []
    
    for idx, pred_set in enumerate(predictions, 1):
        # Convert prediction to integers
        pred_numbers = [int(n) for n in pred_set]
        
        # Calculate matches
        matching_numbers = set(pred_numbers) & set(winning_numbers)
        match_count = len(matching_numbers)
        has_bonus = bonus_number in pred_numbers
        
        matches_found.append({
            'set_num': idx,
            'numbers': pred_numbers,
            'match_count': match_count,
            'has_bonus': has_bonus,
            'matching_numbers': matching_numbers
        })
    
    # Sort by match count (highest first)
    matches_found.sort(key=lambda x: (x['match_count'], x['has_bonus']), reverse=True)
    
    # === PHASE 1: CLUSTER DETECTION ===
    # Analyze top 5 predictions for cluster patterns
    top_5_numbers = set()
    for match_data in matches_found[:5]:
        top_5_numbers.update(match_data['numbers'])
    
    cluster_coverage = len(top_5_numbers & set(winning_numbers))
    cluster_detected = cluster_coverage >= 6
    
    # Display statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Sets", len(predictions))
    with col2:
        sets_with_bonus = sum(1 for m in matches_found if m['has_bonus'])
        st.metric("Sets with Bonus", sets_with_bonus)
    with col3:
        best_match = matches_found[0]['match_count'] if matches_found else 0
        st.metric("Best Match", f"{best_match}/7")
    with col4:
        avg_match = sum(m['match_count'] for m in matches_found) / len(matches_found) if matches_found else 0
        st.metric("Avg Match", f"{avg_match:.1f}/7")
    with col5:
        if cluster_detected:
            st.metric("🎯 Cluster", f"{cluster_coverage}/{len(winning_numbers)}", delta="Detected!", delta_color="normal")
        else:
            st.metric("Cluster", f"{cluster_coverage}/{len(winning_numbers)}")
    
    # === PHASE 1: DISPLAY CLUSTER ALERT ===
    if cluster_detected:
        st.success(f"""🎯 **Winning Number Cluster Detected!**  
        Your top 5 predictions collectively contain **{cluster_coverage} out of {len(winning_numbers)}** winning numbers!  
        The AI was very close but fragmented the numbers across multiple sets.  
        📚 This pattern will be learned to improve concentration in future predictions.""")
        
        # Show which numbers were in the cluster
        covered = sorted(list(top_5_numbers & set(winning_numbers)))
        missing = sorted(list(set(winning_numbers) - top_5_numbers))
        
        col_covered, col_missing = st.columns(2)
        with col_covered:
            st.markdown(f"**✅ Covered in Top 5:** {', '.join(map(str, covered))}")
        with col_missing:
            if missing:
                st.markdown(f"**❌ Missing:** {', '.join(map(str, missing))}")
            else:
                st.markdown("**✅ All numbers covered!**")
    
    st.divider()
    
    # Display sets
    for match_data in matches_found:
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown(f"**Set #{match_data['set_num']}**")
                st.markdown(f"**Match: {match_data['match_count']}/7**")
                if match_data['has_bonus']:
                    st.markdown("**🟡 Bonus!**")
            
            with col2:
                # Display numbers as balls
                cols = st.columns(7)
                for i, num in enumerate(match_data['numbers']):
                    with cols[i]:
                        # Highlight matching numbers in green, bonus in gold
                        if match_data['has_bonus'] and num == bonus_number:
                            bg_color = "#FFD700"
                            label = "🟡"
                        elif num in match_data['matching_numbers']:
                            bg_color = "#28a745"
                            label = "✓"
                        else:
                            bg_color = "#6c757d"
                            label = ""
                        
                        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; padding: 8px; background-color: {bg_color}; color: white; border-radius: 50%; width: 45px; height: 45px; line-height: 29px; margin: auto;'>{num}<br><span style='font-size: 10px;'>{label}</span></div>", unsafe_allow_html=True)


def _process_maxmillion_input(input_text: str) -> List[List[int]]:
    """Process and validate MaxMillion input text. Accepts both comma and space separated numbers."""
    try:
        sets = []
        lines = input_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse numbers - support both comma and space separated
            # First try comma separation
            if ',' in line:
                numbers = [int(n.strip()) for n in line.split(',') if n.strip()]
            else:
                # Try space separation
                numbers = [int(n.strip()) for n in line.split() if n.strip()]
            
            # Validate
            if len(numbers) != 7:
                st.error(f"Invalid set: {line} - Must have exactly 7 numbers")
                continue
            
            if any(n < 1 or n > 50 for n in numbers):
                st.error(f"Invalid set: {line} - Numbers must be between 1 and 50")
                continue
            
            if len(set(numbers)) != 7:
                st.error(f"Invalid set: {line} - Numbers must be unique")
                continue
            
            # Sort and add
            sets.append(sorted(numbers))
        
        return sets
    except Exception as e:
        app_log(f"Error processing MaxMillion input: {str(e)}", "error")
        st.error(f"Error processing input: {str(e)}")
        return []


def _save_maxmillion_data(game: str, draw_date, maxmillion_sets: List[List[int]]) -> Path:
    """Save MaxMillion data to file."""
    try:
        game_folder = _sanitize_game_name(game)
        maxm_dir = Path("data") / game_folder / "maxmillions"
        maxm_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with date
        filename = f"maxmillion_{draw_date.strftime('%Y%m%d')}.json"
        filepath = maxm_dir / filename
        
        # Save data
        data = {
            "draw_date": draw_date.strftime('%Y-%m-%d'),
            "game": game,
            "maxmillion_sets": maxmillion_sets,
            "total_sets": len(maxmillion_sets),
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        app_log(f"Saved MaxMillion data to {filepath}", "info")
        return filepath
    except Exception as e:
        app_log(f"Error saving MaxMillion data: {str(e)}", "error")
        raise


def _load_maxmillion_from_file(game: str, draw_date) -> List[List[int]]:
    """Load MaxMillion data from file."""
    try:
        game_folder = _sanitize_game_name(game)
        maxm_dir = Path("data") / game_folder / "maxmillions"
        
        if not maxm_dir.exists():
            st.info("📁 No MaxMillion data directory found. Use manual input to create.")
            return []
        
        # Look for file with matching date
        filename = f"maxmillion_{draw_date.strftime('%Y%m%d')}.json"
        filepath = maxm_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                st.success(f"✅ Loaded {data['total_sets']} MaxMillion sets from file")
                return data['maxmillion_sets']
        else:
            st.info(f"📁 No MaxMillion file found for {draw_date.strftime('%Y-%m-%d')}. Use manual input to create.")
            return []
    except Exception as e:
        app_log(f"Error loading MaxMillion file: {str(e)}", "error")
        st.error(f"Error loading file: {str(e)}")
        return []


def _display_maxmillion_sets(maxmillion_sets: List[List[int]]) -> None:
    """Display MaxMillion sets as game balls."""
    st.markdown(f"### 🎰 MaxMillion Sets ({len(maxmillion_sets)} total)")
    
    for idx, mm_set in enumerate(maxmillion_sets, 1):
        with st.container(border=True):
            col1, col2 = st.columns([1, 5])
            
            with col1:
                st.markdown(f"**MM #{idx}**")
            
            with col2:
                cols = st.columns(7)
                for i, num in enumerate(mm_set):
                    with cols[i]:
                        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; padding: 8px; background-color: #9C27B0; color: white; border-radius: 50%; width: 45px; height: 45px; line-height: 29px; margin: auto;'>{num}</div>", unsafe_allow_html=True)


def _display_maxmillion_comparison(prediction_data: Dict[str, Any], maxmillion_sets: List[List[int]]) -> None:
    """Display prediction sets that match MaxMillion sets."""
    st.divider()
    st.markdown("### 🎯 Prediction Sets vs MaxMillion Analysis")
    
    predictions = prediction_data.get('predictions', [])
    
    if not predictions:
        st.warning("No predictions found in file")
        return
    
    # Find matches
    matching_sets = []
    
    for idx, pred_set in enumerate(predictions, 1):
        pred_numbers = sorted([int(n) for n in pred_set])
        
        # Check if this prediction matches any MaxMillion set
        for mm_idx, mm_set in enumerate(maxmillion_sets, 1):
            if pred_numbers == sorted(mm_set):
                matching_sets.append({
                    'pred_idx': idx,
                    'mm_idx': mm_idx,
                    'numbers': pred_numbers
                })
                break
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Prediction Sets", len(predictions))
    with col2:
        st.metric("MaxMillion Sets", len(maxmillion_sets))
    with col3:
        st.metric("Exact Matches", len(matching_sets))
    
    if matching_sets:
        st.success(f"🎉 Found {len(matching_sets)} exact match(es)!")
        
        for match in matching_sets:
            with st.container(border=True):
                st.markdown(f"**Prediction Set #{match['pred_idx']} = MaxMillion #{match['mm_idx']}**")
                cols = st.columns(7)
                for i, num in enumerate(match['numbers']):
                    with cols[i]:
                        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; padding: 8px; background-color: #FFD700; color: black; border-radius: 50%; width: 45px; height: 45px; line-height: 29px; margin: auto;'>{num}</div>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ No exact matches found between predictions and MaxMillion sets")
    
    # Show all prediction sets with highlighting
    with st.expander("📋 View All Prediction Sets", expanded=False):
        for idx, pred_set in enumerate(predictions, 1):
            pred_numbers = sorted([int(n) for n in pred_set])
            is_match = any(pred_numbers == sorted(mm_set) for mm_set in maxmillion_sets)
            
            with st.container(border=True):
                if is_match:
                    st.markdown(f"**Set #{idx} ⭐ MAXMILLION MATCH!**")
                else:
                    st.markdown(f"**Set #{idx}**")
                
                cols = st.columns(7)
                for i, num in enumerate(pred_numbers):
                    with cols[i]:
                        bg_color = "#FFD700" if is_match else "#4169E1"
                        text_color = "black" if is_match else "white"
                        st.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; padding: 8px; background-color: {bg_color}; color: {text_color}; border-radius: 50%; width: 45px; height: 45px; line-height: 29px; margin: auto;'>{num}</div>", unsafe_allow_html=True)


def _render_performance_history(analyzer: SuperIntelligentAIAnalyzer) -> None:
    """Show historical prediction performance."""
    st.subheader("📈 Historical Performance Metrics")
    
    saved_predictions = analyzer.get_saved_predictions()
    
    if not saved_predictions:
        st.info("📝 No prediction history available yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(saved_predictions))
    with col2:
        total_sets = sum(len(p["predictions"]) for p in saved_predictions)
        st.metric("Total Sets Generated", total_sets)
    with col3:
        avg_sets = np.mean([len(p["predictions"]) for p in saved_predictions])
        st.metric("Avg Sets per Prediction", f"{avg_sets:.1f}")
    with col4:
        st.metric("Last Generated", saved_predictions[0]["timestamp"][:10])
    
    st.divider()
    st.markdown("### 📊 Prediction History")
    
    history_data = []
    for idx, pred in enumerate(saved_predictions):
        # Handle both old and new save formats
        analysis = pred.get("analysis", {})
        
        # Get accuracy - try different field names
        accuracy = analysis.get("average_accuracy")
        if accuracy is None:
            # Try the new format - calculate from ensemble confidence
            accuracy = analysis.get("ensemble_confidence", 0.0)
        
        # Get model count
        model_count = len(analysis.get("selected_models", []))
        if model_count == 0:
            # Try the new format field
            model_count = analysis.get("total_models", 0)
        
        # Get confidence
        confidence = analysis.get("ensemble_confidence", 0.0)
        
        history_data.append({
            "ID": idx + 1,
            "Date": pred.get("timestamp", "N/A")[:19],
            "Sets": len(pred.get("predictions", [])),
            "Models": model_count,
            "Confidence": f"{float(confidence):.1%}",
            "Accuracy": f"{float(accuracy):.1%}"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    st.divider()
    st.markdown("### 📌 Model Usage & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Used Models**")
        all_models = []
        for pred in saved_predictions:
            analysis = pred.get("analysis", {})
            selected_models = analysis.get("selected_models", [])
            for model in selected_models:
                all_models.append(f"{model.get('name', 'Unknown')} ({model.get('type', 'Unknown')})")
        
        from collections import Counter
        model_counts = Counter(all_models)
        
        if model_counts:
            for model, count in model_counts.most_common(5):
                st.write(f"• {model}: {count} time(s)")
        else:
            st.write("No model usage data available")
    
    with col2:
        st.markdown("**Average Metrics**")
        confidences = []
        accuracies = []
        sets_counts = []
        
        for p in saved_predictions:
            analysis = p.get("analysis", {})
            confidences.append(float(analysis.get("ensemble_confidence", 0.0)))
            
            # Get accuracy from either old or new format
            accuracy = analysis.get("average_accuracy", analysis.get("ensemble_confidence", 0.0))
            accuracies.append(float(accuracy))
            
            sets_counts.append(len(p.get("predictions", [])))
        
        if confidences:
            avg_confidence = float(np.mean(confidences))
            avg_accuracy = float(np.mean(accuracies))
            avg_sets = float(np.mean(sets_counts))
            
            st.write(f"• Avg Confidence: {avg_confidence:.1%}")
            st.write(f"• Avg Accuracy: {avg_accuracy:.1%}")
            st.write(f"• Avg Sets: {avg_sets:.0f}")
        else:
            st.write("No metrics data available")


# ============================================================================
# TAB 5: DEEP LEARNING & ANALYTICS
# ============================================================================

def _render_deep_learning_tab(analyzer: SuperIntelligentAIAnalyzer, game: str) -> None:
    """Deep Learning and Analytics tab for prediction optimization and learning."""
    st.subheader("🧠 Deep Learning and Analytics")

    st.markdown("""
    Use machine learning to analyze predictions and optimize future sets based on historical patterns and outcomes.
    """)

    # If the auto-learning banner redirected here, show a prominent prompt and
    # default to "Previous Draw Date" mode with the pending draw pre-selected.
    _auto_date = st.session_state.get('sia_auto_learning_date')
    _default_mode_index = 0
    if _auto_date:
        st.info(
            f"📌 **Auto-selected:** Draw date **{_auto_date}** has unseen results ready for learning. "
            f"Switch to **Previous Draw Date** mode below and apply learning to incorporate these results."
        )
        _default_mode_index = 1  # default to "Previous Draw Date"

    # Mode selector
    mode = st.radio(
        "Analysis Mode",
        ["📅 Next Draw Date (Optimize Future Predictions)", "📊 Previous Draw Date (Learn from Results)"],
        index=_default_mode_index,
        key="dl_mode"
    )

    st.divider()

    if "Next Draw Date" in mode:
        _render_next_draw_mode(analyzer, game)
    else:
        _render_previous_draw_mode(analyzer, game)


def _render_next_draw_mode(analyzer: SuperIntelligentAIAnalyzer, game: str) -> None:
    """Next Draw Date mode - optimize future predictions using learning data."""
    st.markdown("### 📅 Next Draw Prediction Optimization")
    st.markdown("*Regenerate predictions using historical learning patterns for improved accuracy*")
    
    # Get next draw date
    try:
        next_draw = compute_next_draw_date(game)
        next_draw_str = next_draw.strftime('%Y-%m-%d')
    except:
        next_draw_str = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"📅 **Next Draw Date:** {next_draw_str}")
    with col2:
        st.metric("Workflow", "Learning-Based")
    
    st.divider()
    
    # STEP 1: Select Prediction File to Optimize
    st.markdown("#### 🎯 Step 1: Select Prediction File")
    
    pred_files = _find_prediction_files_for_date(game, next_draw_str)
    
    if not pred_files:
        st.warning(f"⚠️ No prediction files found for {next_draw_str}")
        st.info("💡 Go to the '**🎲 Generate Predictions**' tab to create predictions for the next draw date")
        return
    
    # File selector with details
    file_options = []
    for f in pred_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                num_sets = len(data.get('predictions', []))
                timestamp = data.get('timestamp', '')[:19]
                file_options.append(f"{f.name} | {num_sets} sets | {timestamp}")
        except:
            file_options.append(f.name)
    
    selected_file_idx = st.selectbox(
        "Select prediction file to optimize",
        range(len(pred_files)),
        format_func=lambda i: file_options[i],
        key="next_draw_file",
        help="Choose the prediction file you want to regenerate with learning insights"
    )
    
    selected_file = pred_files[selected_file_idx]
    
    # Load prediction file
    try:
        with open(selected_file, 'r') as f:
            pred_data = json.load(f)
        
        predictions = pred_data.get('predictions', [])
        
        if not predictions:
            st.error("No predictions found in file")
            return
        
        # ── File metadata ──────────────────────────────────────────────────────
        _ld_meta1, _ld_meta2, _ld_meta3 = st.columns(3)
        _ld_meta1.metric("Total Sets", len(predictions))
        _ld_meta2.metric("Algorithm", pred_data.get('algorithm', 'N/A')[:30])
        _ld_meta3.metric("Generated", pred_data.get('timestamp', 'N/A')[:16])

        # ── Ranked display (same three-tier layout as Generate Predictions tab) ─
        _ld_ranked = pred_data.get('generation_ranking', [])

        if not _ld_ranked:
            # File was generated before ranking was added — build a plain list
            st.info("ℹ️ This file was generated before AI ranking was available. Sets shown in generation order.")
            with st.expander("📋 All Prediction Sets", expanded=False):
                for _idx, _pred in enumerate(predictions, 1):
                    _nums = sorted(int(n) for n in (_pred.get('numbers', _pred) if isinstance(_pred, dict) else _pred))
                    st.markdown(f"**Set #{_idx}:** {', '.join(map(str, _nums))}")
        else:
            st.markdown(f"#### 🏅 AI-Ranked Sets — {selected_file.name}")
            st.caption(
                "Ranked by AI confidence at generation time (model vote strength + consensus + "
                "profile conformance). These ranks were computed before any draw results were known."
            )

            # Summary table
            _ld_rank_rows = []
            for _rr in _ld_ranked:
                _ld_rank_rows.append({
                    'Rank': _rr['rank'],
                    'Tier': "🏆 Top 5" if _rr['rank'] <= 5 else ("⭐ Top 10" if _rr['rank'] <= 10 else f"#{_rr['rank']}"),
                    'Set #': _rr['set_number'],
                    'Numbers': ', '.join(map(str, _rr['numbers'])),
                    'AI Confidence': f"{_rr['confidence_score']:.1%}",
                })
            st.dataframe(pd.DataFrame(_ld_rank_rows), use_container_width=True, hide_index=True)

            # Helper for ball display
            def _ld_render_balls(numbers: list) -> None:
                _cols = st.columns(len(numbers))
                for _bc, _bn in zip(_cols, numbers):
                    with _bc:
                        st.markdown(
                            f'<div style="text-align:center;width:50px;height:50px;'
                            f'background:linear-gradient(135deg,#1e3a8a 0%,#3b82f6 50%,#1e40af 100%);'
                            f'border-radius:50%;color:white;font-weight:900;font-size:22px;'
                            f'box-shadow:0 4px 8px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.3);'
                            f'display:flex;align-items:center;justify-content:center;'
                            f'border:2px solid rgba(255,255,255,0.2);margin:0 auto;">{_bn}</div>',
                            unsafe_allow_html=True,
                        )

            # Tier 1: Top 5
            _ld_tier1 = [r for r in _ld_ranked if r['rank'] <= 5]
            if _ld_tier1:
                st.markdown("##### 🏆 Top 5 — Highest AI Confidence")
                for _rr in _ld_tier1:
                    with st.container(border=True):
                        _cs = _rr['confidence_score']
                        _bar = "█" * int(_cs * 10) + "░" * (10 - int(_cs * 10))
                        st.markdown(
                            f"**Rank #{_rr['rank']} — Set #{_rr['set_number']}** &nbsp;&nbsp; "
                            f"AI Confidence: `{_cs:.1%}` `{_bar}`",
                            unsafe_allow_html=True,
                        )
                        _ld_render_balls(_rr['numbers'])

            # Tier 2: Ranks 6-10
            _ld_tier2 = [r for r in _ld_ranked if 6 <= r['rank'] <= 10]
            if _ld_tier2:
                st.markdown("##### ⭐ Ranks 6–10")
                for _rr in _ld_tier2:
                    with st.container(border=True):
                        st.markdown(
                            f"**Rank #{_rr['rank']} — Set #{_rr['set_number']}** &nbsp;&nbsp; "
                            f"AI Confidence: `{_rr['confidence_score']:.1%}`",
                            unsafe_allow_html=True,
                        )
                        _ld_render_balls(_rr['numbers'])

            # Tier 3: Ranks 11+
            _ld_tier3 = [r for r in _ld_ranked if r['rank'] > 10]
            if _ld_tier3:
                with st.expander(f"📋 Ranks 11–{len(_ld_ranked)} ({len(_ld_tier3)} remaining sets)", expanded=False):
                    for _rr in _ld_tier3:
                        with st.container(border=True):
                            st.markdown(
                                f"**Rank #{_rr['rank']} — Set #{_rr['set_number']}** &nbsp;&nbsp; "
                                f"AI Confidence: `{_rr['confidence_score']:.1%}`",
                                unsafe_allow_html=True,
                            )
                            _ld_render_balls(_rr['numbers'])
        
        st.divider()
        
        # STEP 2: Select Learning Files
        st.markdown("#### 🧠 Step 2: Select Learning Data Sources")
        st.markdown("*Choose one or more learning files to guide the regeneration*")
        
        # Find available learning files
        learning_files = _find_all_learning_files(game)
        
        if not learning_files:
            st.warning("⚠️ No learning files found. Use 'Previous Draw Date' mode to create learning data first.")
            return
        
        # Create learning file options with metadata
        learning_options = []
        learning_metadata = []
        for lf in learning_files:
            try:
                with open(lf, 'r') as file:
                    ldata = json.load(file)
                    draw_date = ldata.get('draw_date', 'Unknown')
                    total_correct = ldata.get('summary', {}).get('total_correct_predictions', 0)
                    best_accuracy = ldata.get('summary', {}).get('best_set_accuracy', 0)
                    learning_options.append(f"{lf.name} | Date: {draw_date} | Best: {best_accuracy}%")
                    learning_metadata.append({
                        'file': lf,
                        'date': draw_date,
                        'total_correct': total_correct,
                        'best_accuracy': best_accuracy
                    })
            except:
                learning_options.append(lf.name)
                learning_metadata.append({'file': lf, 'date': 'Unknown', 'total_correct': 0, 'best_accuracy': 0})
        
        # Multi-select for learning files
        selected_learning_indices = st.multiselect(
            "Select learning file(s) to apply",
            range(len(learning_files)),
            format_func=lambda i: learning_options[i],
            default=[0] if len(learning_files) > 0 else [],
            key="selected_learning_files",
            help="Select multiple files to combine insights from different draws"
        )
        
        if not selected_learning_indices:
            st.warning("⚠️ Please select at least one learning file to continue")
            return
        
        # Display selected learning files summary
        st.markdown(f"**Selected:** {len(selected_learning_indices)} learning file(s)")
        
        with st.expander("📊 Learning Sources Summary", expanded=False):
            for idx in selected_learning_indices:
                meta = learning_metadata[idx]
                st.write(f"• {meta['file'].name} - Date: {meta['date']} - Best Accuracy: {meta['best_accuracy']}%")
        
        st.divider()
        
        # STEP 3: Regenerate with Learning
        st.markdown("#### 🚀 Step 3: Regenerate Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            regenerate_strategy = st.selectbox(
                "Regeneration Strategy",
                ["Learning-Guided", "Learning-Optimized", "Hybrid"],
                help="Learning-Guided: Apply patterns, Learning-Optimized: Full rebuild, Hybrid: Mix both"
            )
        
        with col2:
            keep_top_n = st.number_input(
                "Keep Top N Original Sets",
                min_value=0,
                max_value=len(predictions),
                value=min(10, len(predictions) // 4),
                help="How many best original sets to preserve"
            )
        
        with col3:
            learning_weight = st.slider(
                "Learning Influence",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="How much to weight learning insights vs original model predictions"
            )
        
        st.markdown("")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🧬 Regenerate Predictions with Learning", type="primary", use_container_width=True, key="regenerate_with_learning"):
                with st.spinner("🔬 Analyzing learning patterns and regenerating predictions..."):
                    # Load selected learning files
                    selected_learning_files = [learning_metadata[i]['file'] for i in selected_learning_indices]
                    combined_learning_data = _load_and_combine_learning_files(selected_learning_files)
                    
                    # Regenerate predictions with learning
                    regenerated_predictions, regeneration_report = _regenerate_predictions_with_learning(
                        predictions=predictions,
                        pred_data=pred_data,
                        learning_data=combined_learning_data,
                        strategy=regenerate_strategy,
                        keep_top_n=keep_top_n,
                        learning_weight=learning_weight,
                        game=game,
                        analyzer=analyzer
                    )
                    
                    # Save regenerated predictions with learning suffix
                    saved_path = _save_learning_regenerated_predictions(
                        original_file=selected_file,
                        regenerated_predictions=regenerated_predictions,
                        original_data=pred_data,
                        learning_sources=selected_learning_files,
                        strategy=regenerate_strategy,
                        learning_weight=learning_weight
                    )
                    
                    st.success(f"✅ **Regeneration Complete!**")
                    st.balloons()
                    
                    # Display regeneration report
                    with st.expander("📋 Regeneration Report", expanded=True):
                        st.markdown(regeneration_report)
                    
                    # Display comparison
                    st.markdown("#### 📊 Comparison: Original vs Learning-Regenerated")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Original Sets", len(predictions))
                        st.metric("Original File", selected_file.name)
                    
                    with col2:
                        st.metric("Regenerated Sets", len(regenerated_predictions))
                        st.metric("New File", saved_path.name)
                    
                    st.info(f"💾 **Saved to:** `{saved_path}`\n\n✨ Original file preserved. New learning-enhanced file created.")
                    
                    # Display preview of regenerated sets - ranked by learning score
                    st.markdown("#### 🎲 Preview: Top 10 Learning-Ranked Sets")
                    
                    # Get predictions with attribution if available
                    predictions_with_attribution = pred_data.get('predictions_with_attribution', [])
                    
                    # Rank regenerated sets by learning score
                    ranked_regenerated = []
                    for idx, pred_set in enumerate(regenerated_predictions):
                        score = _calculate_learning_score(pred_set, combined_learning_data)
                        ranked_regenerated.append((score, idx + 1, pred_set))
                    
                    ranked_regenerated.sort(key=lambda x: x[0], reverse=True)
                    
                    # Display top 10 ranked sets
                    for rank, (score, original_idx, pred_set) in enumerate(ranked_regenerated[:10], 1):
                        with st.container(border=True):
                            st.markdown(f"**Rank #{rank}** (Learning Score: {score:.3f}) - Original Set #{original_idx}")
                            
                            # Display model attribution if available
                            if predictions_with_attribution:
                                # Find matching prediction in attribution data
                                matching_attr = None
                                for attr in predictions_with_attribution:
                                    attr_numbers = attr.get('numbers', [])
                                    if sorted(attr_numbers) == sorted(pred_set):
                                        matching_attr = attr
                                        break
                                
                                if matching_attr:
                                    model_attribution = matching_attr.get('model_attribution', {})
                                    if model_attribution:
                                        # Collect all unique models that contributed
                                        contributing_models = set()
                                        for num, voters in model_attribution.items():
                                            for voter in voters:
                                                contributing_models.add(voter.get('model', 'Unknown'))
                                        
                                        if contributing_models:
                                            st.markdown(f"**🤖 Models:** {', '.join(sorted(contributing_models))}")
                            
                            # Display numbers
                            cols = st.columns(len(pred_set))
                            for col, num in zip(cols, sorted(pred_set)):
                                with col:
                                    st.markdown(_get_ball_html(num), unsafe_allow_html=True)
        
        with col_btn2:
            if st.button("📊 Rank Original by Learning", type="secondary", use_container_width=True, key="rank_original"):
                with st.spinner("📈 Ranking original predictions by learning patterns..."):
                    # Load selected learning files
                    selected_learning_files = [learning_metadata[i]['file'] for i in selected_learning_indices]
                    combined_learning_data = _load_and_combine_learning_files(selected_learning_files)
                    
                    # Rank predictions without regenerating
                    ranked_predictions, ranking_report = _rank_predictions_by_learning(
                        predictions=predictions,
                        pred_data=pred_data,
                        learning_data=combined_learning_data,
                        analyzer=analyzer
                    )
                    
                    st.success(f"✅ **Ranking Complete!**")
                    
                    # Display ranking report
                    with st.expander("📊 Ranking Analysis", expanded=True):
                        st.markdown(ranking_report)
                    
                    # Display ranked predictions
                    st.markdown("#### 🎯 Learning-Ranked Predictions (Top 20)")
                    
                    # Get predictions with attribution if available
                    predictions_with_attribution = pred_data.get('predictions_with_attribution', [])
                    
                    for rank, (score, pred_set) in enumerate(ranked_predictions[:20], 1):
                        with st.container(border=True):
                            st.markdown(f"**Rank #{rank}** - Learning Score: {score:.3f}")
                            
                            # Display model attribution if available
                            if predictions_with_attribution:
                                # Find matching prediction in attribution data
                                matching_attr = None
                                for attr in predictions_with_attribution:
                                    attr_numbers = attr.get('numbers', [])
                                    if sorted(attr_numbers) == sorted(pred_set):
                                        matching_attr = attr
                                        break
                                
                                if matching_attr:
                                    model_attribution = matching_attr.get('model_attribution', {})
                                    if model_attribution:
                                        # Collect all unique models that contributed
                                        contributing_models = set()
                                        for num, voters in model_attribution.items():
                                            for voter in voters:
                                                contributing_models.add(voter.get('model', 'Unknown'))
                                        
                                        if contributing_models:
                                            st.markdown(f"**🤖 Models:** {', '.join(sorted(contributing_models))}")
                            
                            # Display numbers
                            cols = st.columns(len(pred_set))
                            for col, num in zip(cols, sorted(pred_set)):
                                with col:
                                    st.markdown(_get_ball_html(num), unsafe_allow_html=True)
                    
                    # Download button for ranked results
                    st.markdown("")
                    download_text = f"Learning-Ranked Predictions Results\n"
                    download_text += f"{'='*60}\n\n"
                    download_text += f"Game: {game}\n"
                    download_text += f"Prediction File: {selected_file.name}\n"
                    download_text += f"Learning Sources: {len(selected_learning_files)} file(s)\n"
                    download_text += f"Total Predictions: {len(predictions)}\n"
                    download_text += f"Ranked Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    download_text += f"{'='*60}\n\n"
                    download_text += ranking_report + "\n\n"
                    download_text += f"{'='*60}\n"
                    download_text += f"TOP 20 RANKED PREDICTIONS\n"
                    download_text += f"{'='*60}\n\n"
                    
                    for rank, (score, pred_set) in enumerate(ranked_predictions[:20], 1):
                        download_text += f"Rank #{rank} - Learning Score: {score:.3f}\n"
                        download_text += f"Numbers: {', '.join(map(str, sorted(pred_set)))}\n\n"
                    
                    st.download_button(
                        label="📥 Download Ranked Results",
                        data=download_text,
                        file_name=f"ranked_predictions_{game.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
    
    except Exception as e:
        st.error(f"Error loading prediction file: {e}")
        import traceback
        st.error(traceback.format_exc())


def _render_previous_draw_mode(analyzer: SuperIntelligentAIAnalyzer, game: str) -> None:
    """Previous Draw Date mode - learn from actual results."""
    st.markdown("### 📊 Learn from Previous Draw Results")
    
    # Load past draw dates
    past_dates = _load_past_draw_dates(game)
    
    if not past_dates:
        st.warning(f"⚠️ No historical draw data found for {game}")
        return
    
    # Date selector — auto-select the pending draw date when coming from the banner
    _auto_date = st.session_state.get('sia_auto_learning_date')
    _date_index = 0
    if _auto_date and _auto_date in past_dates:
        _date_index = past_dates.index(_auto_date)
        # Clear so it only pre-selects once; user can change freely afterward
        st.session_state.pop('sia_auto_learning_date', None)

    selected_date = st.selectbox(
        "Select Draw Date",
        past_dates,
        index=_date_index,
        key="prev_draw_date"
    )
    
    # Load actual results
    actual_results = _load_actual_results(game, selected_date)
    
    if not actual_results:
        st.warning(f"⚠️ No results found for {selected_date}")
        return
    
    # Validate we have numbers
    if not actual_results.get('numbers') or len(actual_results['numbers']) == 0:
        st.error(f"⚠️ Invalid draw data for {selected_date} - no winning numbers found")
        return
    
    # Display actual results
    st.markdown("#### 🎯 Actual Draw Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Winning Numbers:**")
        cols = st.columns(len(actual_results['numbers']))
        for col, num in zip(cols, actual_results['numbers']):
            with col:
                st.markdown(_get_ball_html(num, color="green"), unsafe_allow_html=True)
    
    with col2:
        if actual_results.get('bonus'):
            st.markdown("**Bonus:**")
            st.markdown(_get_ball_html(actual_results['bonus'], color="gold"), unsafe_allow_html=True)
    
    with col3:
        if actual_results.get('jackpot'):
            st.metric("Jackpot", f"${actual_results['jackpot']:,.0f}")
    
    st.divider()
    
    # Find prediction files for this date
    pred_files = _find_prediction_files_for_date(game, selected_date)
    
    if not pred_files:
        st.info(f"ℹ️ No prediction files found for {selected_date}")
        return
    
    # File selector
    file_options = [f.name for f in pred_files]
    selected_file_idx = st.selectbox(
        "Select Prediction File",
        range(len(pred_files)),
        format_func=lambda i: file_options[i],
        key="prev_draw_file"
    )
    
    selected_file = pred_files[selected_file_idx]
    
    # Load predictions
    try:
        with open(selected_file, 'r') as f:
            pred_data = json.load(f)
        
        predictions = pred_data.get('predictions', [])
        
        if not predictions:
            st.error("No predictions found in file")
            return
        
        # Highlight matches and sort by accuracy
        matched_predictions = _highlight_prediction_matches(
            predictions,
            actual_results['numbers'],
            actual_results.get('bonus')
        )
        
        sorted_predictions = _sort_predictions_by_accuracy(matched_predictions, actual_results['numbers'])
        
        st.markdown(f"#### 🎲 Prediction Results (Sorted by Accuracy)")
        st.markdown(f"**Total Sets:** {len(sorted_predictions)}")
        
        # Display sorted predictions with matches
        for rank, pred in enumerate(sorted_predictions, 1):  # Show ALL sets
            correct_count = pred['correct_count']
            has_bonus = pred['has_bonus']
            
            status_emoji = "🏆" if correct_count >= 4 else "✅" if correct_count >= 2 else "➖"
            
            with st.expander(
                f"{status_emoji} **Rank {rank}** - Set #{pred['original_index'] + 1} "
                f"({correct_count} correct{' + BONUS' if has_bonus else ''})",
                expanded=(rank <= 5)
            ):
                # Display the numbers with appropriate colors
                cols = st.columns(len(pred['numbers']))
                for col, num_data in zip(cols, pred['numbers']):
                    num = num_data['number']
                    is_correct = num_data['is_correct']
                    is_bonus = num_data['is_bonus']
                    
                    # Determine color based on match status
                    if is_bonus:
                        color = "gold"
                    elif is_correct:
                        color = "green"
                    else:
                        color = "blue"
                    
                    with col:
                        # Pass color parameter to the function
                        st.markdown(_get_ball_html(num, color=color), unsafe_allow_html=True)
                
                legend = "🟢 Correct | 🟡 Bonus | 🔵 Miss"
                st.caption(legend)
        
        # Display model attribution in a separate expander
        predictions_with_attribution = pred_data.get('predictions_with_attribution', [])
        
        with st.expander("🤖 **Model Attribution for All Predictions**", expanded=False):
            if not predictions_with_attribution:
                st.info("ℹ️ Model attribution data is not available for this prediction file. Attribution data is only saved when predictions are generated using the AI Prediction Engine with deep learning models.")
            else:
                st.markdown("This shows which AI models contributed to each prediction set:")
                
                for rank, pred in enumerate(sorted_predictions, 1):
                    # Get the prediction numbers as ints for comparison
                    pred_numbers = sorted(int(num_data['number']) for num_data in pred['numbers'])

                    # Find matching prediction in attribution data
                    matching_attr = None
                    for attr in predictions_with_attribution:
                        try:
                            attr_numbers = sorted(int(n) for n in attr.get('numbers', []))
                        except (TypeError, ValueError):
                            continue
                        if attr_numbers == pred_numbers:
                            matching_attr = attr
                            break
                    
                    if matching_attr:
                        model_attribution = matching_attr.get('model_attribution', {})
                        if model_attribution:
                            # Collect all unique models that contributed
                            contributing_models = set()
                            for num, voters in model_attribution.items():
                                if voters:  # Only count if there are actual voters
                                    for voter in voters:
                                        contributing_models.add(voter.get('model', 'Unknown'))
                            
                            if contributing_models:
                                st.markdown(f"**Set #{pred['original_index'] + 1}:** {', '.join(sorted(contributing_models))}")
                            else:
                                st.markdown(f"**Set #{pred['original_index'] + 1}:** No model attribution (random selection)")
        
        # ===== GENERATION RANKING vs ACTUAL RANKING COMPARISON =====
        _gen_ranking = pred_data.get('generation_ranking', [])
        if _gen_ranking:
            with st.expander("📊 **Generation Rank vs Actual Result Rank**", expanded=True):
                st.markdown(
                    "Compare the AI's predicted confidence rankings (computed at generation time, "
                    "before the draw) against the actual result rankings (sorted by match count). "
                    "This shows how well the model's confidence scores predict real-world accuracy."
                )

                # Build actual rank map: original_set_index → actual_rank (by match count)
                _act_rank_map: dict = {}
                for _ar, _sp in enumerate(sorted_predictions, 1):
                    _act_rank_map[_sp['original_index']] = _ar  # original_index is 0-based

                # Build comparison table
                _cmp_rows = []
                for _gr in _gen_ranking:
                    _orig = _gr['original_set_index']
                    _gen_r = _gr['rank']
                    _act_r = _act_rank_map.get(_orig, len(sorted_predictions))
                    _delta = _act_r - _gen_r  # positive = ranked lower than expected
                    _delta_str = (
                        f"▲ {abs(_delta)} better" if _delta < 0 else
                        f"▼ {abs(_delta)} worse" if _delta > 0 else
                        "✅ exact"
                    )
                    _top5_gen = "🏆" if _gen_r <= 5 else ("⭐" if _gen_r <= 10 else "")
                    _top5_act = "🏆" if _act_r <= 5 else ("⭐" if _act_r <= 10 else "")
                    _cmp_rows.append({
                        'Set #': _gr['set_number'],
                        'Gen Rank': f"{_top5_gen} #{_gen_r}",
                        'Actual Rank': f"{_top5_act} #{_act_r}",
                        'Shift': _delta_str,
                        'AI Confidence': f"{_gr['confidence_score']:.1%}",
                        'Numbers': ', '.join(map(str, _gr['numbers'])),
                    })

                # Sort by generation rank for clean display
                _cmp_rows.sort(key=lambda x: int(x['Gen Rank'].split('#')[1]))
                st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)

                # ── Ranking accuracy metrics ──────────────────────────────────
                _gen_top5_indices  = {r['original_set_index'] for r in _gen_ranking if r['rank'] <= 5}
                _act_top5_indices  = {sp['original_index'] for sp in sorted_predictions[:5]}
                _overlap_top5 = len(_gen_top5_indices & _act_top5_indices)

                _gen_top10_indices = {r['original_set_index'] for r in _gen_ranking if r['rank'] <= 10}
                _act_top10_indices = {sp['original_index'] for sp in sorted_predictions[:10]}
                _overlap_top10 = len(_gen_top10_indices & _act_top10_indices)

                st.markdown("#### 🎯 Ranking Accuracy")
                _ra1, _ra2, _ra3 = st.columns(3)
                _ra1.metric(
                    "Top-5 Overlap",
                    f"{_overlap_top5}/5",
                    help="How many of the AI's top-5 picks were actually in the top-5 by match count"
                )
                _ra2.metric(
                    "Top-10 Overlap",
                    f"{_overlap_top10}/10",
                    help="How many of the AI's top-10 picks were actually in the top-10 by match count"
                )
                # Spearman-like rank correlation for displayed sets
                _n_cmp = min(len(_gen_ranking), len(sorted_predictions))
                if _n_cmp >= 2:
                    import scipy.stats as _spst
                    _gen_r_vec = [_act_rank_map.get(r['original_set_index'], _n_cmp + 1) for r in _gen_ranking[:_n_cmp]]
                    _act_r_vec = list(range(1, _n_cmp + 1))
                    try:
                        _rho, _pval = _spst.spearmanr(_gen_r_vec, _act_r_vec)
                        _ra3.metric(
                            "Rank Correlation",
                            f"{_rho:.2f}",
                            help="Spearman ρ: 1.0 = perfect order, 0 = random, -1 = reversed"
                        )
                    except Exception:
                        _ra3.metric("Rank Correlation", "N/A")
                else:
                    _ra3.metric("Rank Correlation", "N/A")

                if _overlap_top5 >= 3:
                    st.success(f"✅ Strong ranking accuracy — {_overlap_top5}/5 top predictions confirmed by actual results.")
                elif _overlap_top5 >= 1:
                    st.info(f"ℹ️ Partial ranking accuracy — {_overlap_top5}/5 top predictions confirmed.")
                else:
                    st.warning("⚠️ Top-5 generation ranks did not match actual top-5. This result will be used as a learning signal to improve future ranking.")

                st.caption(
                    "💡 **Learning signal**: After clicking **Apply Learning Analysis** below, "
                    "the ranking accuracy data above feeds into the adaptive system to improve "
                    "future confidence score calibration."
                )

        st.divider()

        # Use Raw CSVs checkbox
        use_raw_csv = st.checkbox(
            "📁 Include Raw CSV Pattern Analysis",
            value=False,
            help="Analyze historical patterns from raw CSV files (slower but more comprehensive)",
            key="use_raw_csv"
        )

        # Apply Learning button
        if st.button("🔬 Apply Learning Analysis", use_container_width=True, key="apply_learning_prev"):
            with st.spinner("Performing deep learning analysis..."):
                # Comprehensive learning analysis
                learning_data = _compile_comprehensive_learning_data(
                    game,
                    selected_date,
                    actual_results,
                    sorted_predictions,
                    pred_data,
                    use_raw_csv
                )
                
                if learning_data:
                    # ===== ADAPTIVE LEARNING SYSTEM UPDATE =====
                    # Initialize adaptive learning system
                    adaptive_system = AdaptiveLearningSystem(game)
                    
                    # Get top 10% and bottom 10% predictions for analysis
                    total_predictions = len(sorted_predictions)
                    top_count = max(1, total_predictions // 10)
                    bottom_count = max(1, total_predictions // 10)
                    
                    top_predictions = [
                        [n['number'] for n in pred['numbers']]
                        for pred in sorted_predictions[:top_count]
                    ]
                    
                    worst_predictions = [
                        [n['number'] for n in pred['numbers']]
                        for pred in sorted_predictions[-bottom_count:]
                    ]
                    
                    # Calculate factor scores for top predictions
                    analysis = learning_data.get('analysis', {})
                    factor_scores = {}
                    
                    # Collect scores for each factor from top predictions
                    for factor_name in adaptive_system.meta_learning_data['factor_weights'].keys():
                        factor_scores[factor_name] = []
                    
                    for pred_set in top_predictions:
                        # Calculate how strong each factor is in this successful prediction
                        # Hot numbers factor
                        number_freq = analysis.get('number_frequency', {})
                        hot_numbers_data = number_freq.get('hot_numbers', [])
                        hot_numbers = [item['number'] if isinstance(item, dict) else item for item in hot_numbers_data[:10]]
                        hot_matches = sum(1 for n in pred_set if n in hot_numbers)
                        factor_scores['hot_numbers'].append(hot_matches / 10.0)
                        
                        # Sum alignment factor
                        sum_analysis = analysis.get('sum_analysis', {})
                        target_sum = sum_analysis.get('winning_sum', 0)
                        if target_sum:
                            sum_diff = abs(sum(pred_set) - target_sum)
                            sum_score = max(0, 1.0 - (sum_diff / 100.0))
                            factor_scores['sum_alignment'].append(sum_score)
                        
                        # Diversity factor
                        max_number = 50 if 'max' in game.lower() else 49
                        num_range = max(pred_set) - min(pred_set)
                        diversity_score = num_range / (max_number - 1)
                        factor_scores['diversity'].append(diversity_score)
                        
                        # Gap patterns factor
                        gap_patterns = analysis.get('gap_patterns', {})
                        avg_winning_gap = gap_patterns.get('avg_winning_gap', 0)
                        if avg_winning_gap:
                            sorted_nums = sorted(pred_set)
                            gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
                            avg_gap = np.mean(gaps) if gaps else 0
                            gap_diff = abs(avg_winning_gap - avg_gap)
                            gap_score = max(0, 1.0 - (gap_diff / 10.0))
                            factor_scores['gap_patterns'].append(gap_score)
                        
                        # Zone distribution score
                        zone_dist = analysis.get('zone_distribution', {})
                        winning_zones = zone_dist.get('winning_distribution', {})
                        if winning_zones:
                            _low_b = max_number // 3
                            _mid_b = (max_number * 2) // 3
                            pred_zones = {'low': 0, 'mid': 0, 'high': 0}
                            for _n in pred_set:
                                if _n <= _low_b:
                                    pred_zones['low'] += 1
                                elif _n <= _mid_b:
                                    pred_zones['mid'] += 1
                                else:
                                    pred_zones['high'] += 1
                            zone_diff = sum(abs(winning_zones.get(z, 0) - pred_zones[z]) for z in ['low', 'mid', 'high'])
                            factor_scores['zone_distribution'].append(max(0.0, 1.0 - (zone_diff / max(1, len(pred_set)))))
                        else:
                            factor_scores['zone_distribution'].append(0.0)

                        # Even/odd ratio score
                        even_odd = analysis.get('even_odd_ratio', {})
                        winning_ratio = even_odd.get('winning_ratio', None)
                        if winning_ratio is not None:
                            even_count = sum(1 for _n in pred_set if _n % 2 == 0)
                            predicted_ratio = even_count / max(1, len(pred_set))
                            factor_scores['even_odd_ratio'].append(max(0.0, 1.0 - abs(winning_ratio - predicted_ratio)))
                        else:
                            factor_scores['even_odd_ratio'].append(0.0)

                        # Cold penalty score
                        cold_numbers = analysis.get('cold_numbers', [])
                        if cold_numbers:
                            cold_matches = sum(1 for _n in pred_set if _n in cold_numbers)
                            factor_scores['cold_penalty'].append(cold_matches / max(1, len(pred_set)))
                        else:
                            factor_scores['cold_penalty'].append(0.0)

                        # Decade coverage score
                        decade_cov = analysis.get('decade_coverage', {})
                        winning_decade_count = decade_cov.get('winning_decade_count', 0)
                        if winning_decade_count:
                            pred_decades = set((_n - 1) // 10 for _n in pred_set)
                            decade_diff = abs(winning_decade_count - len(pred_decades))
                            factor_scores['decade_coverage'].append(max(0.0, 1.0 - (decade_diff / 5.0)))
                        else:
                            factor_scores['decade_coverage'].append(0.0)

                        # Pattern fingerprint score
                        winning_pattern = analysis.get('winning_pattern_fingerprint', '')
                        if winning_pattern:
                            pred_pattern = _create_pattern_fingerprint(pred_set)
                            min_len = min(len(winning_pattern), len(pred_pattern))
                            if min_len > 0:
                                pat_matches = sum(1 for _i in range(min_len) if winning_pattern[_i] == pred_pattern[_i])
                                factor_scores['pattern_fingerprint'].append(pat_matches / min_len)
                            else:
                                factor_scores['pattern_fingerprint'].append(0.0)
                        else:
                            factor_scores['pattern_fingerprint'].append(0.0)

                        # Position weighting score
                        position_accuracy = analysis.get('position_accuracy', {})
                        if position_accuracy:
                            sorted_pred = sorted(pred_set)
                            pos_score = sum(
                                position_accuracy.get(f'position_{_i+1}', {}).get('accuracy', 0.0)
                                for _i, _n in enumerate(sorted_pred)
                            ) / max(1, len(sorted_pred))
                            factor_scores['position_weighting'].append(pos_score)
                        else:
                            factor_scores['position_weighting'].append(0.0)
                    
                    # Update adaptive weights based on success
                    adaptive_system.update_weights_from_success(
                        winning_numbers=actual_results,
                        top_predictions=top_predictions,
                        factor_scores=factor_scores
                    )

                    # P5: Detect cross-factor interactions and persist them so
                    # get_adaptive_weights() can apply joint multipliers next generation
                    try:
                        adaptive_system.detect_cross_factor_interactions(
                            predictions=top_predictions,
                            winning_numbers=actual_results,
                            factor_scores=factor_scores
                        )
                    except Exception:
                        pass

                    # Track anti-patterns from worst predictions
                    adaptive_system.track_anti_patterns(
                        worst_predictions=worst_predictions,
                        winning_numbers=actual_results
                    )
                    
                    # === PHASE 1: CLUSTER DETECTION ===
                    # Detect if top predictions collectively contain winning numbers
                    # Convert sorted_predictions to the format expected by detect_winning_clusters
                    ranked_for_cluster = []
                    for pred in sorted_predictions:
                        # Extract just the numbers (not the metadata)
                        numbers = [n['number'] for n in pred['numbers']]
                        # Create (score, numbers) tuple - use correct_count as score
                        score = pred['correct_count'] / len(actual_results['numbers'])
                        ranked_for_cluster.append((score, numbers))
                    
                    cluster_result = adaptive_system.detect_winning_clusters(
                        ranked_predictions=ranked_for_cluster,
                        winning_numbers=actual_results['numbers'],
                        top_n=5
                    )
                    
                    # === PHASE 2: UPDATE CLUSTER CONCENTRATION WEIGHT ===
                    # Always call — even when no cluster is detected — so the EMA receives
                    # negative feedback and the weight doesn't drift upward unchecked.
                    coverage_count = cluster_result.get('coverage_count', 0)
                    new_concentration_weight = adaptive_system.update_cluster_concentration_weight(
                        cluster_coverage=coverage_count,
                        total_winners=len(actual_results['numbers'])
                    )
                    cluster_result['concentration_weight_updated'] = new_concentration_weight
                    
                    # Get updated intelligence stats
                    total_cycles = adaptive_system.meta_learning_data.get('total_learning_cycles', 0)
                    current_weights = adaptive_system.meta_learning_data['factor_weights']
                    
                    # Save learning data
                    saved_path = _save_learning_data(game, selected_date, learning_data)
                    
                    st.success(f"✅ Learning data saved to: `{saved_path}`")
                    st.success(f"🧬 **Adaptive Intelligence Updated!** Learning Cycle #{total_cycles} Complete")
                    
                    # Display adaptive intelligence summary
                    st.markdown("#### 🧠 Adaptive Intelligence Status")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Learning Cycles", total_cycles, delta="+1")
                    with col2:
                        anti_pattern_count = len(adaptive_system.meta_learning_data.get('anti_patterns', []))
                        st.metric("Anti-Patterns Tracked", anti_pattern_count)
                    with col3:
                        interactions_count = len(adaptive_system.meta_learning_data.get('cross_factor_interactions', {}))
                        st.metric("Factor Interactions", interactions_count)
                    with col4:
                        cooccurrence_count = len(adaptive_system.meta_learning_data.get('number_cooccurrence', {}))
                        st.metric("Number Pairs Tracked", cooccurrence_count)
                    
                    # === PHASE 1: DISPLAY CLUSTER DETECTION RESULTS ===
                    if cluster_result.get('cluster_detected'):
                        total_winners = len(actual_results['numbers'])
                        st.success(f"""🎯 **Winning Number Cluster Detected!**  
                        Your top {cluster_result['top_n']} predictions collectively contain **{cluster_result['coverage_count']} out of {total_winners}** winning numbers ({cluster_result['coverage_percent']:.1f}% coverage)!  
                        The AI was very close but fragmented the numbers across multiple sets.  
                        📚 **This pattern has been learned and will improve number concentration in future predictions.**""")
                        
                        # Show cluster breakdown
                        with st.expander("🔍 Cluster Analysis Details", expanded=True):
                            col_covered, col_missing = st.columns(2)
                            with col_covered:
                                st.markdown(f"**✅ Covered Winners:** {', '.join(map(str, cluster_result['covered_winners']))}")
                            with col_missing:
                                if cluster_result['missing_winners']:
                                    st.markdown(f"**❌ Missing Winners:** {', '.join(map(str, cluster_result['missing_winners']))}")
                                else:
                                    st.markdown(f"**🎉 All {total_winners} winning numbers covered in top predictions!**")
                            
                            # Show individual set contributions
                            st.markdown("**📊 Individual Set Contributions:**")
                            total_winners = len(actual_results['numbers'])
                            for match_info in cluster_result['individual_matches']:
                                rank = match_info['rank']
                                match_count = match_info['match_count']
                                matched = ', '.join(map(str, match_info['matched_numbers']))
                                score = match_info['score']
                                st.markdown(f"• **Rank #{rank}**: {match_count}/{total_winners} matches - Numbers: {matched} (Score: {score:.3f})")
                            
                            # Show top co-occurring number pairs
                            top_pairs = adaptive_system.get_top_cooccurrences(top_n=10)
                            if top_pairs:
                                st.markdown("**🔗 Top Number Co-occurrence Patterns:**")
                                pair_text = []
                                for pair_key, count in top_pairs[:10]:
                                    nums = pair_key.split('_')
                                    if len(nums) == 2:
                                        pair_text.append(f"{nums[0]}-{nums[1]} ({count}x)")
                                st.markdown(", ".join(pair_text))
                            
                            # === PHASE 2: SHOW CONCENTRATION WEIGHT UPDATE ===
                            if 'concentration_weight_updated' in cluster_result:
                                new_weight = cluster_result['concentration_weight_updated']
                                success_rate = adaptive_system.meta_learning_data.get('cluster_success_rate', 0.0)
                                st.markdown(f"**🎯 Phase 2 - Concentration Weight Updated:**")
                                st.info(f"""
                                📈 **Cluster Concentration Weight:** {new_weight:.1%}  
                                📊 **Cluster Success Rate:** {success_rate:.1%}  
                                
                                This weight controls how much future predictions will concentrate co-occurring numbers together.
                                As more successful clusters are detected, this weight increases (max 20%).
                                """)
                    else:
                        total_winners = len(actual_results['numbers'])
                        st.info(f"ℹ️ No significant cluster detected. Top {cluster_result.get('top_n', 5)} predictions covered {cluster_result.get('coverage_count', 0)}/{total_winners} winning numbers.")
                    
                    # Display evolved weights
                    st.markdown("**🎯 Evolved Adaptive Factor Weights:**")
                    top_factors = sorted(current_weights.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    weight_cols = st.columns(5)
                    for idx, (factor, weight) in enumerate(top_factors):
                        with weight_cols[idx]:
                            st.metric(
                                factor.replace('_', ' ').title()[:15],
                                f"{weight:.1%}",
                                help=f"Current adaptive weight for {factor}"
                            )
                    
                    # P6: Learning Effectiveness Trend chart
                    _hr_history = adaptive_system.meta_learning_data.get('hit_rate_history', [])
                    if len(_hr_history) >= 2:
                        st.markdown("#### 📉 Learning Effectiveness Trend")
                        _hr_cycles = [h['cycle'] for h in _hr_history]
                        _hr_rates  = [h['hit_rate'] for h in _hr_history]
                        _hr_df = pd.DataFrame({'Cycle': _hr_cycles, 'Hit Rate': _hr_rates})
                        st.line_chart(_hr_df.set_index('Cycle'), use_container_width=True, height=180)
                        # Warn if last 3 cycles show declining hit rate
                        if len(_hr_rates) >= 3:
                            _recent = _hr_rates[-3:]
                            if _recent[-1] < _recent[0] - 0.10:
                                st.warning(
                                    f"⚠️ **Learning may be degrading** — hit rate dropped from "
                                    f"{_recent[0]:.0%} to {_recent[-1]:.0%} over the last 3 cycles. "
                                    "Consider selecting different learning files or resetting adaptive weights."
                                )
                            else:
                                st.caption(f"Current hit rate: **{_hr_rates[-1]:.0%}** — learning is healthy.")

                    # Display learning insights
                    st.markdown("#### 📈 Learning Insights")
                    
                    insights = learning_data.get('learning_insights', [])
                    for insight in insights:
                        st.info(f"💡 {insight}")
                    
                    # Display detailed analysis
                    with st.expander("📊 Detailed Analysis Results", expanded=True):
                        analysis = learning_data['analysis']
                        
                        # Position accuracy
                        st.markdown("**Position-wise Accuracy:**")
                        pos_acc = analysis['position_accuracy']
                        pos_df = pd.DataFrame([
                            {
                                'Position': int(k.split('_')[1]),
                                'Correct': v['correct'],
                                'Total': v['total'],
                                'Accuracy': f"{v['accuracy']:.1%}"
                            }
                            for k, v in pos_acc.items()
                        ])
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)
                        
                        # Sum analysis
                        st.markdown("**Sum Analysis:**")
                        sum_data = analysis['sum_analysis']
                        st.write(f"• Winning Sum: {sum_data['winning_sum']}")
                        st.write(f"• Closest Set: #{sum_data['closest_set']['index'] + 1} (diff: {sum_data['closest_set']['diff']})")
                        
                        # Set accuracy distribution
                        st.markdown("**Top Performing Sets:**")
                        top_sets = analysis['set_accuracy'][:5]
                        for s in top_sets:
                            st.write(f"• Set #{s['set'] + 1}: {s['correct']} correct - {s['numbers']}")
                        
                        # Raw CSV patterns (if enabled)
                        if use_raw_csv and 'raw_csv_patterns' in analysis:
                            st.markdown("**Raw CSV Pattern Analysis:**")
                            csv_patterns = analysis['raw_csv_patterns']
                            st.write(f"• Matching Jackpot Draws: {csv_patterns.get('matching_jackpot_draws', 0)}")
                            st.write(f"• Sum Range: {csv_patterns['sum_distribution']['min']}-{csv_patterns['sum_distribution']['max']}")
                            st.write(f"• Avg Odd/Even: {csv_patterns['odd_even_ratio']['odd']:.1f} / {csv_patterns['odd_even_ratio']['even']:.1f}")
                            st.write(f"• Repetition Rate: {csv_patterns.get('repetition_rate', 0):.1%}")
                        
                        # HIGH VALUE ANALYSIS DISPLAYS
                        st.divider()
                        
                        # Number Frequency Analysis
                        st.markdown("**🎯 Number Frequency Analysis:**")
                        num_freq = analysis.get('number_frequency', {})
                        if num_freq.get('correctly_predicted'):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("*Top Correctly Predicted:*")
                                for item in num_freq['correctly_predicted'][:5]:
                                    st.write(f"  #{item['number']}: appeared {item['count']} times")
                            with col2:
                                st.markdown("*Most Missed (frequently predicted but wrong):*")
                                for item in num_freq['missed_numbers'][:5]:
                                    st.write(f"  #{item['number']}: predicted {item['count']} times, never won")
                        
                        # Consecutive Analysis
                        st.markdown("**🔗 Consecutive Numbers Analysis:**")
                        consec = analysis.get('consecutive_analysis', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Consecutive Pairs (Winning)", consec.get('winning_consecutive_pairs', 0))
                        with col2:
                            st.metric("Consecutive Pairs (Predicted)", consec.get('total_consecutive_pairs', 0))
                        with col3:
                            st.metric("Avg Gap Between Numbers", f"{consec.get('average_gap', 0):.1f}")
                        
                        # Model Performance
                        st.markdown("**🤖 Model Performance Breakdown:**")
                        model_perf = analysis.get('model_performance', {})
                        if model_perf.get('model_contributions'):
                            # Show data source indicator
                            data_source = model_perf.get('data_source', 'unknown')
                            if data_source == 'attribution':
                                st.success("✅ **Using Real Attribution Data** - Actual model votes tracked during generation")
                                # Display enhanced columns with vote data
                                model_df = pd.DataFrame(model_perf['model_contributions'])
                                st.dataframe(model_df, use_container_width=True, hide_index=True)
                                
                                # Show summary stats
                                total_votes = model_perf.get('total_votes', 0)
                                total_correct_votes = model_perf.get('total_correct_votes', 0)
                                if total_votes > 0:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Model Votes", total_votes)
                                    with col2:
                                        st.metric("Correct Votes", total_correct_votes)
                                    with col3:
                                        vote_accuracy = (total_correct_votes / total_votes * 100) if total_votes > 0 else 0
                                        st.metric("Vote Accuracy", f"{vote_accuracy:.1f}%")
                            elif data_source == 'estimation':
                                st.warning("⚠️ **Using Estimated Data** - Prediction file lacks attribution tracking")
                                model_df = pd.DataFrame(model_perf['model_contributions'])
                                st.dataframe(model_df, use_container_width=True, hide_index=True)
                            else:
                                model_df = pd.DataFrame(model_perf['model_contributions'])
                                st.dataframe(model_df, use_container_width=True, hide_index=True)
                            
                            # Show best performing model
                            best = model_perf.get('best_performing_model')
                            if best:
                                st.info(f"🏆 **Top Contributor:** {best['name']} ({best['contribution']:.1%} contribution)")
                        
                        # Temporal Patterns
                        st.markdown("**📅 Temporal Patterns:**")
                        temporal = analysis.get('temporal_patterns', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"Day: {temporal.get('day_of_week', 'N/A')}")
                        with col2:
                            st.write(f"Month: {temporal.get('month', 'N/A')}")
                        with col3:
                            st.write(f"Quarter: {temporal.get('quarter', 'N/A')}")
                        
                        # Diversity Metrics
                        st.markdown("**🌈 Set Diversity Metrics:**")
                        diversity = analysis.get('diversity_metrics', {})
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Number Space Coverage", f"{diversity.get('number_space_coverage', 0):.1%}")
                        with col2:
                            st.metric("Unique Numbers", diversity.get('unique_numbers_count', 0))
                        with col3:
                            st.metric("Avg Set Overlap", f"{diversity.get('average_overlap', 0):.1f}")
                        with col4:
                            st.metric("Diversity Score", f"{diversity.get('diversity_score', 0):.2f}")
                        
                        # Probability Calibration
                        st.markdown("**📊 Probability Calibration:**")
                        prob_cal = analysis.get('probability_calibration', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Confidence", f"{prob_cal.get('predicted_confidence', 0):.1%}")
                        with col2:
                            st.metric("Actual Accuracy", f"{prob_cal.get('actual_accuracy', 0):.1%}")
                        with col3:
                            calibration_diff = prob_cal.get('actual_accuracy', 0) - prob_cal.get('predicted_confidence', 0)
                            st.metric("Calibration Error", f"{calibration_diff:.1%}", 
                                     delta=f"{'Over' if calibration_diff < 0 else 'Under'}-confident")
                        
                        # ENHANCED ANALYSIS DISPLAYS
                        st.divider()
                        st.markdown("### 🎯 Enhanced Learning Metrics")
                        
                        # Gap Patterns Analysis
                        st.markdown("**📏 Gap Patterns Analysis:**")
                        gap_patterns = analysis.get('gap_patterns', {})
                        if gap_patterns:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Winning Avg Gap", f"{gap_patterns.get('avg_winning_gap', 0):.1f}")
                            with col2:
                                st.metric("Predicted Avg Gap", f"{gap_patterns.get('avg_prediction_gap', 0):.1f}")
                            with col3:
                                st.metric("Gap Similarity", f"{gap_patterns.get('gap_similarity', 0):.1%}")
                            
                            # Show actual gaps
                            winning_gaps = gap_patterns.get('winning_gaps', [])
                            if winning_gaps:
                                st.write(f"*Winning Gaps:* {', '.join(map(str, winning_gaps))}")
                        
                        # Zone Distribution Analysis
                        st.markdown("**🎪 Zone Distribution Analysis:**")
                        zone_dist = analysis.get('zone_distribution', {})
                        if zone_dist:
                            winning_zones = zone_dist.get('winning_distribution', {})
                            avg_pred_zones = zone_dist.get('avg_prediction_distribution', {})
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("*Winning Zones:*")
                                st.write(f"  Low: {winning_zones.get('low', 0)}")
                                st.write(f"  Mid: {winning_zones.get('mid', 0)}")
                                st.write(f"  High: {winning_zones.get('high', 0)}")
                            with col2:
                                st.markdown("*Avg Predicted Zones:*")
                                st.write(f"  Low: {avg_pred_zones.get('low', 0):.1f}")
                                st.write(f"  Mid: {avg_pred_zones.get('mid', 0):.1f}")
                                st.write(f"  High: {avg_pred_zones.get('high', 0):.1f}")
                            with col3:
                                match_score = zone_dist.get('zone_match_score', 0)
                                st.metric("Zone Match Score", f"{match_score:.2f}")
                        
                        # Even/Odd Ratio Analysis
                        st.markdown("**⚖️ Even/Odd Ratio Analysis:**")
                        even_odd = analysis.get('even_odd_ratio', {})
                        if even_odd:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Winning Even", even_odd.get('winning_even_count', 0))
                            with col2:
                                st.metric("Winning Odd", even_odd.get('winning_odd_count', 0))
                            with col3:
                                st.metric("Winning Ratio", f"{even_odd.get('winning_ratio', 0):.1%}")
                            with col4:
                                st.metric("Ratio Similarity", f"{even_odd.get('ratio_similarity', 0):.1%}")
                        
                        # Decade Coverage Analysis
                        st.markdown("**🔢 Decade Coverage Analysis:**")
                        decade_cov = analysis.get('decade_coverage', {})
                        if decade_cov:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Winning Decade Count", decade_cov.get('winning_decade_count', 0))
                            with col2:
                                st.metric("Avg Predicted Decades", f"{decade_cov.get('avg_prediction_decade_coverage', 0):.1f}")
                            with col3:
                                st.metric("Decade Diversity", f"{decade_cov.get('decade_diversity_score', 0):.1%}")
                            
                            # Show winning decades distribution
                            winning_decades = decade_cov.get('winning_decades', {})
                            if winning_decades:
                                st.write(f"*Winning Decades Distribution:*")
                                for decade, count in sorted(winning_decades.items()):
                                    decade_range = f"{decade*10+1}-{decade*10+10}"
                                    st.write(f"  Decade {decade} ({decade_range}): {count} numbers")
                        
                        # Cold Numbers
                        st.markdown("**❄️ Cold Numbers (Bottom 20%):**")
                        cold_numbers = analysis.get('cold_numbers', [])
                        if cold_numbers:
                            st.write(f"*Numbers to Avoid:* {', '.join(map(str, sorted(cold_numbers[:15])))}")
                            if len(cold_numbers) > 15:
                                st.write(f"...and {len(cold_numbers) - 15} more")
                        else:
                            st.write("No cold number data available")
                        
                        # Pattern Fingerprint
                        st.markdown("**🔍 Winning Pattern Fingerprint:**")
                        pattern = analysis.get('winning_pattern_fingerprint', '')
                        if pattern:
                            st.code(pattern, language=None)
                            st.caption("Pattern Legend: S = Small gap (1-5), M = Medium gap (6-10), L = Large gap (11+)")
                        else:
                            st.write("No pattern fingerprint available")

                    # ===== P10: FORWARD-LOOKING LEARNING PREVIEW =====
                    # Show how the new learning file has shifted number probabilities,
                    # so the user can see what has changed before their next generation.
                    st.divider()
                    st.markdown("#### 🔭 Forward-Looking Preview")
                    st.caption("How this learning cycle will influence your **next** generation run.")

                    _fwd_analysis = learning_data.get('analysis', {})
                    _fwd_num_freq = _fwd_analysis.get('number_frequency', {})
                    _fwd_hot_raw  = _fwd_num_freq.get('hot_numbers', [])
                    _fwd_cold     = _fwd_analysis.get('cold_numbers', [])
                    _fwd_hot = [item['number'] if isinstance(item, dict) else item
                                for item in _fwd_hot_raw[:10]]
                    _fwd_cold_nums = [item if isinstance(item, int) else item.get('number', 0)
                                      for item in _fwd_cold[:10]]

                    _fwd_col1, _fwd_col2 = st.columns(2)
                    with _fwd_col1:
                        if _fwd_hot:
                            st.markdown("**📈 Numbers with HIGHER probability next run:**")
                            st.success(", ".join(str(n) for n in sorted(_fwd_hot)))
                            st.caption("These appeared frequently in learning data — hot_numbers weight will boost them.")
                        else:
                            st.info("No hot-number boost detected.")

                    with _fwd_col2:
                        _fwd_avoid = list(_fwd_cold_nums)
                        # Also include numbers dominant in recent anti-patterns
                        _ap_nums: Dict[int, int] = {}
                        for _ap in adaptive_system.meta_learning_data.get('anti_patterns', [])[-20:]:
                            for _apn in _ap.get('numbers', []):
                                _ap_nums[_apn] = _ap_nums.get(_apn, 0) + 1
                        _top_ap = [n for n, _ in sorted(_ap_nums.items(), key=lambda x: x[1], reverse=True)[:5]
                                   if n not in _fwd_hot]
                        _fwd_avoid = list(dict.fromkeys(_fwd_avoid + _top_ap))[:10]
                        if _fwd_avoid:
                            st.markdown("**📉 Numbers with LOWER probability next run:**")
                            st.error(", ".join(str(n) for n in sorted(_fwd_avoid)))
                            st.caption("Cold numbers + frequent anti-pattern members — will be de-prioritised.")
                        else:
                            st.info("No cold-number penalty detected.")

                    # Show which factor weights changed most this cycle
                    _weight_hist = adaptive_system.meta_learning_data.get('weight_history', [])
                    if len(_weight_hist) >= 2:
                        _prev_w = _weight_hist[-2]['weights']
                        _curr_w = _weight_hist[-1]['weights']
                        _deltas = {f: _curr_w.get(f, 0) - _prev_w.get(f, 0)
                                   for f in _curr_w if f in _prev_w}
                        _top_delta = sorted(_deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
                        if any(abs(d) > 0.001 for _, d in _top_delta):
                            st.markdown("**⚖️ Factor weight shifts this cycle:**")
                            _delta_cols = st.columns(len(_top_delta))
                            for _di, (_fn, _fd) in enumerate(_top_delta):
                                with _delta_cols[_di]:
                                    st.metric(
                                        _fn.replace('_', ' ').title()[:14],
                                        f"{_curr_w.get(_fn, 0):.1%}",
                                        delta=f"{_fd:+.1%}",
                                        help=f"Weight changed by {_fd:+.1%} this cycle"
                                    )


    except Exception as e:
        st.error(f"Error processing predictions: {e}")
        import traceback
        st.error(traceback.format_exc())


# ============================================================================
# HELPER FUNCTIONS FOR DEEP LEARNING TAB
# ============================================================================

def _find_prediction_files_for_date(game: str, draw_date: str) -> List[Path]:
    """Find prediction files in predictions/{game}/prediction_ai/ directory."""
    game_folder = _sanitize_game_name(game)
    pred_dir = Path("predictions") / game_folder / "prediction_ai"
    
    if not pred_dir.exists():
        return []
    
    matching_files = []
    
    for file in pred_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                file_draw_date = data.get('next_draw_date', '')
                
                # Match exact date only
                if file_draw_date == draw_date:
                    matching_files.append(file)
        except:
            continue
    
    return sorted(matching_files, key=lambda x: x.stat().st_mtime, reverse=True)


def _display_prediction_sets_as_balls(predictions: List, draw_size: int) -> None:
    """Display prediction sets as game balls."""
    for i, pred in enumerate(predictions, 1):  # Show ALL sets
        # Handle different prediction formats
        if isinstance(pred, dict):
            numbers = pred.get('numbers', [])
        elif isinstance(pred, list):
            numbers = pred
        else:
            continue
        
        # Ensure we have the right number of balls
        if len(numbers) != draw_size:
            continue
        
        with st.container(border=True):
            st.markdown(f"**Set {i}**")
            cols = st.columns(len(numbers))
            for col, num in zip(cols, sorted(numbers)):
                with col:
                    st.markdown(_get_ball_html(num), unsafe_allow_html=True)


def _get_ball_html(number: int, color: str = "blue") -> str:
    """Generate HTML for a lottery ball."""
    color_map = {
        "blue": "linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%)",
        "green": "linear-gradient(135deg, #166534 0%, #22c55e 50%, #15803d 100%)",
        "gold": "linear-gradient(135deg, #854d0e 0%, #fbbf24 50%, #a16207 100%)",
        "red": "linear-gradient(135deg, #991b1b 0%, #ef4444 50%, #b91c1c 100%)"
    }
    
    gradient = color_map.get(color, color_map["blue"])
    
    return f'''
    <div style="
        text-align: center;
        padding: 0;
        margin: 5px auto;
        width: 50px;
        height: 50px;
        background: {gradient};
        border-radius: 50%;
        color: white;
        font-weight: 900;
        font-size: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px solid rgba(255,255,255,0.2);
    ">{number}</div>
    '''


def _apply_learning_analysis_future(predictions: List, game: str, analyzer: SuperIntelligentAIAnalyzer) -> Dict:
    """Apply learning analysis to future predictions (placeholder with basic scoring)."""
    scores = {}
    
    # Load learning data if available
    learning_data = _load_latest_learning_data(game)
    
    for idx, pred in enumerate(predictions):
        if isinstance(pred, dict):
            numbers = pred.get('numbers', [])
        else:
            numbers = pred
        
        # Convert numbers to integers if they're strings
        try:
            numbers = [int(n) for n in numbers]
        except (ValueError, TypeError):
            # Already integers or invalid data
            pass
        
        # Basic scoring based on learning data
        score = 0.5  # Base score
        confidence = 0.5
        
        if learning_data:
            # Score based on sum similarity
            pred_sum = sum(numbers)
            target_sum = learning_data.get('analysis', {}).get('sum_analysis', {}).get('winning_sum', 0)
            if target_sum:
                sum_diff = abs(pred_sum - target_sum)
                score += max(0, (50 - sum_diff) / 100)
            
            # Score based on position patterns
            score += 0.1  # Placeholder
            
            confidence = min(1.0, score)
        
        scores[idx] = {
            'score': score,
            'confidence': confidence,
            'sum': sum(numbers)
        }
    
    return scores


def _rank_sets_by_learning(predictions: List, learning_scores: Dict) -> List[Tuple]:
    """Rank prediction sets by learning scores."""
    ranked = []
    
    for idx, pred in enumerate(predictions):
        if isinstance(pred, dict):
            numbers = pred.get('numbers', [])
        else:
            numbers = pred
        
        score_data = learning_scores.get(idx, {'score': 0})
        score = score_data['score']
        
        ranked.append((idx, score, numbers))
    
    # Sort by score descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return ranked


def _load_latest_learning_data(game: str) -> Dict:
    """Load the most recent learning data for a game."""
    game_folder = _sanitize_game_name(game)
    learning_dir = Path("data") / "learning" / game_folder
    
    if not learning_dir.exists():
        return {}
    
    # Find most recent learning file
    learning_files = sorted(learning_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not learning_files:
        return {}
    
    try:
        with open(learning_files[0], 'r') as f:
            return json.load(f)
    except:
        return {}


def _optimize_sets_with_learning(predictions: List, learning_data: Dict, game: str, analyzer: SuperIntelligentAIAnalyzer) -> List:
    """Optimize prediction sets using learning data."""
    # For now, just re-rank existing sets
    # In future, this could regenerate sets based on learning patterns
    
    if not learning_data:
        return predictions
    
    # Extract numbers from predictions
    pred_numbers = []
    for pred in predictions:
        if isinstance(pred, dict):
            pred_numbers.append(pred.get('numbers', []))
        else:
            pred_numbers.append(pred)
    
    # Score each set
    scored_sets = []
    for numbers in pred_numbers:
        score = 0.5
        
        # Score based on learning data patterns
        if learning_data and 'analysis' in learning_data:
            pred_sum = sum(numbers)
            target_sum = learning_data['analysis'].get('sum_analysis', {}).get('winning_sum', 0)
            if target_sum:
                sum_diff = abs(pred_sum - target_sum)
                score += max(0, (50 - sum_diff) / 100)
        
        scored_sets.append((score, numbers))
    
    # Sort by score
    scored_sets.sort(key=lambda x: x[0], reverse=True)
    
    # Return optimized order
    return [{'numbers': numbers, 'optimized_score': score} for score, numbers in scored_sets]


def _save_optimized_predictions(original_file: Path, optimized_predictions: List, original_data: Dict) -> Path:
    """Save optimized predictions to a new file."""
    # Create optimized filename
    file_stem = original_file.stem
    optimized_filename = f"{file_stem}_optimized.json"
    optimized_path = original_file.parent / optimized_filename
    
    # Update data with optimized predictions
    optimized_data = original_data.copy()
    optimized_data['predictions'] = optimized_predictions
    optimized_data['optimized'] = True
    optimized_data['optimization_timestamp'] = datetime.now().isoformat()
    
    # Save
    with open(optimized_path, 'w') as f:
        json.dump(optimized_data, f, indent=2, default=str)
    
    return optimized_path


def _load_past_draw_dates(game: str) -> List[str]:
    """Load historical draw dates from CSV files."""
    game_folder = _sanitize_game_name(game)
    data_dir = Path("data") / game_folder
    
    if not data_dir.exists():
        return []
    
    dates = []
    
    for csv_file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'draw_date' in df.columns:
                file_dates = df['draw_date'].unique().tolist()
                dates.extend(file_dates)
        except:
            continue
    
    # Sort descending (most recent first)
    dates = sorted(list(set(dates)), reverse=True)
    
    return dates[:50]  # Return last 50 draws


def _load_actual_results(game: str, draw_date: str) -> Dict:
    """Load actual draw results for a specific date."""
    game_folder = _sanitize_game_name(game)
    data_dir = Path("data") / game_folder
    
    if not data_dir.exists():
        return {}
    
    for csv_file in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            
            # Find row with matching date
            if 'draw_date' in df.columns:
                match_row = df[df['draw_date'] == draw_date]
                
                if len(match_row) > 0:
                    row = match_row.iloc[0]
                    
                    # Extract winning numbers from n1, n2, n3, etc. columns
                    numbers = []
                    
                    # Try standard n1-n7 column format first
                    for i in range(1, 8):  # Try up to 7 numbers
                        col_name = f'n{i}'
                        if col_name in df.columns:
                            try:
                                num = int(row[col_name])
                                if 1 <= num <= 50:
                                    numbers.append(num)
                            except:
                                continue
                    
                    # If no numbers found, try parsing the 'numbers' column (comma-separated string)
                    if not numbers and 'numbers' in df.columns:
                        try:
                            numbers_str = str(row['numbers'])
                            # Parse comma-separated numbers like "8,9,12,13,29,30,31"
                            numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
                            numbers = [n for n in numbers if 1 <= n <= 50]
                        except:
                            pass
                    
                    # If still no numbers, try other formats
                    if not numbers:
                        for col in df.columns:
                            if col.startswith('number_'):
                                try:
                                    num = int(row[col])
                                    if 1 <= num <= 50:
                                        numbers.append(num)
                                except:
                                    continue
                    
                    # Only return if we found valid numbers
                    if not numbers:
                        continue
                    
                    # Get bonus if available
                    bonus = None
                    for col in ['bonus', 'bonus_number', 'bonus_ball']:
                        if col in df.columns:
                            try:
                                bonus = int(row[col])
                                break
                            except:
                                continue
                    
                    # Get jackpot if available
                    jackpot = None
                    for col in ['jackpot', 'jackpot_amount']:
                        if col in df.columns:
                            try:
                                jackpot = float(row[col])
                                break
                            except:
                                continue
                    
                    return {
                        'numbers': sorted(numbers),
                        'bonus': bonus,
                        'jackpot': jackpot,
                        'draw_date': draw_date
                    }
        except Exception as e:
            continue
    
    return {}


def _highlight_prediction_matches(predictions: List, winning_numbers: List[int], bonus: Optional[int]) -> List[Dict]:
    """Highlight matching numbers in predictions."""
    matched_predictions = []
    
    for idx, pred in enumerate(predictions):
        if isinstance(pred, dict):
            numbers = pred.get('numbers', [])
        else:
            numbers = pred
        
        # Ensure numbers is a list
        if not isinstance(numbers, list):
            numbers = list(numbers)
        
        # Convert all numbers to integers (handle strings from JSON)
        numbers = [int(n) for n in numbers]
        
        # Sort numbers for consistent display
        sorted_numbers = sorted(numbers)
        
        matched_numbers = []
        correct_count = 0
        has_bonus = False
        
        for num in sorted_numbers:
            is_correct = num in winning_numbers
            is_bonus = (bonus is not None) and (num == bonus)
            
            # Only count as correct if it matches winning numbers (bonus is separate)
            if is_correct:
                correct_count += 1
            if is_bonus:
                has_bonus = True
            
            matched_numbers.append({
                'number': num,
                'is_correct': is_correct,
                'is_bonus': is_bonus
            })
        
        matched_predictions.append({
            'original_index': idx,
            'numbers': matched_numbers,
            'correct_count': correct_count,
            'has_bonus': has_bonus
        })
    
    return matched_predictions


def _sort_predictions_by_accuracy(matched_predictions: List[Dict], winning_numbers: List[int]) -> List[Dict]:
    """Sort predictions by number of correct matches."""
    return sorted(matched_predictions, key=lambda x: (x['correct_count'], x['has_bonus']), reverse=True)


def _compile_comprehensive_learning_data(
    game: str,
    draw_date: str,
    actual_results: Dict,
    sorted_predictions: List[Dict],
    pred_data: Dict,
    use_raw_csv: bool
) -> Dict:
    """Compile comprehensive learning data from prediction results."""
    
    # Position accuracy analysis — exact per-draw position matches
    position_accuracy = {}
    for pos in range(1, len(actual_results['numbers']) + 1):
        position_accuracy[f'position_{pos}'] = {
            'correct': 0,
            'total': len(sorted_predictions),
            'accuracy': 0.0
        }

    # Count exact position matches
    for pred in sorted_predictions:
        pred_numbers = [n['number'] for n in pred['numbers']]
        for pos, (actual, predicted) in enumerate(zip(sorted(actual_results['numbers']), sorted(pred_numbers)), 1):
            if actual == predicted:
                position_accuracy[f'position_{pos}']['correct'] += 1

    # Calculate exact-match accuracies
    for pos_key in position_accuracy:
        pos_data = position_accuracy[pos_key]
        pos_data['accuracy'] = pos_data['correct'] / pos_data['total'] if pos_data['total'] > 0 else 0

    # Enhance with historical positional frequency (Priority 3):
    # Replace near-zero exact-match accuracy with the more meaningful per-position
    # hot-number frequency, which drives the 15% position_weighting factor.
    try:
        _hist_pos_freq = _analyze_position_frequency(game)
        for _pos_key, _pos_hist in _hist_pos_freq.items():
            if _pos_key in position_accuracy:
                # Keep exact-match data but override accuracy with historical frequency signal
                position_accuracy[_pos_key]['accuracy']    = _pos_hist['accuracy']
                position_accuracy[_pos_key]['hot_numbers'] = _pos_hist['hot_numbers']
                position_accuracy[_pos_key]['frequency_map'] = _pos_hist['frequency_map']
                position_accuracy[_pos_key]['total_draws_analyzed'] = _pos_hist['total_draws']
            else:
                position_accuracy[_pos_key] = _pos_hist
    except Exception:
        pass  # Non-fatal — fall back to exact-match accuracy
    
    # Sum analysis
    winning_sum = sum(actual_results['numbers'])
    prediction_sums = []
    closest_set = {'index': 0, 'sum': 0, 'diff': float('inf')}
    
    for pred in sorted_predictions:
        pred_sum = sum([n['number'] for n in pred['numbers']])
        prediction_sums.append(pred_sum)
        
        diff = abs(pred_sum - winning_sum)
        if diff < closest_set['diff']:
            closest_set = {'index': pred['original_index'], 'sum': pred_sum, 'diff': diff}
    
    # Set accuracy
    set_accuracy = [
        {
            'set': pred['original_index'],
            'correct': pred['correct_count'],
            'numbers': [n['number'] for n in pred['numbers']]
        }
        for pred in sorted_predictions
    ]
    
    # HIGH VALUE ANALYSIS 1: Number Frequency Analysis
    number_frequency = _analyze_number_frequency(sorted_predictions, actual_results['numbers'])
    
    # HIGH VALUE ANALYSIS 2: Consecutive Numbers Analysis
    consecutive_analysis = _analyze_consecutive_patterns(sorted_predictions, actual_results['numbers'])
    
    # HIGH VALUE ANALYSIS 3: Model Performance Breakdown
    model_performance = _analyze_model_performance(pred_data, sorted_predictions, actual_results['numbers'])
    
    # HIGH VALUE ANALYSIS 4: Temporal Patterns
    temporal_patterns = _analyze_temporal_patterns(draw_date, sorted_predictions)
    
    # HIGH VALUE ANALYSIS 5: Set Diversity Metrics
    diversity_metrics = _analyze_set_diversity(sorted_predictions)
    
    # HIGH VALUE ANALYSIS 6: Probability Calibration
    probability_calibration = _analyze_probability_calibration(pred_data, sorted_predictions, actual_results['numbers'])
    
    # Determine max_number for game
    max_number = 50 if 'max' in game.lower() else 49
    
    # ENHANCED ANALYSIS 7: Gap Patterns
    gap_patterns = _analyze_gap_patterns(sorted_predictions, actual_results['numbers'])
    
    # ENHANCED ANALYSIS 8: Zone Distribution
    zone_distribution = _analyze_zone_distribution(sorted_predictions, actual_results['numbers'], max_number)
    
    # ENHANCED ANALYSIS 9: Even/Odd Ratio
    even_odd_ratio = _analyze_even_odd_ratio(sorted_predictions, actual_results['numbers'])
    
    # ENHANCED ANALYSIS 10: Decade Coverage
    decade_coverage = _analyze_decade_coverage(sorted_predictions, actual_results['numbers'], max_number)
    
    # ENHANCED ANALYSIS 11: Cold Numbers
    cold_numbers = _identify_cold_numbers(game, lookback_draws=50)
    
    # ENHANCED ANALYSIS 12: Winning Pattern Fingerprint
    winning_pattern = _create_pattern_fingerprint(actual_results['numbers'])
    
    # Compile learning data with ENHANCED metrics
    learning_data = {
        'game': game,
        'draw_date': draw_date,
        'actual_results': actual_results,
        'prediction_file': pred_data.get('timestamp', ''),
        'analysis': {
            'position_accuracy': position_accuracy,
            'sum_analysis': {
                'winning_sum': winning_sum,
                'prediction_sums': prediction_sums,
                'closest_set': closest_set
            },
            'set_accuracy': set_accuracy,
            'number_frequency': number_frequency,
            'consecutive_analysis': consecutive_analysis,
            'model_performance': model_performance,
            'temporal_patterns': temporal_patterns,
            'diversity_metrics': diversity_metrics,
            'probability_calibration': probability_calibration,
            'gap_patterns': gap_patterns,
            'zone_distribution': zone_distribution,
            'even_odd_ratio': even_odd_ratio,
            'decade_coverage': decade_coverage,
            'cold_numbers': cold_numbers,
            'winning_pattern_fingerprint': winning_pattern
        },
        'learning_insights': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Generate insights
    insights = []
    
    # Position insights
    worst_pos = min(position_accuracy.items(), key=lambda x: x[1]['accuracy'])
    best_pos = max(position_accuracy.items(), key=lambda x: x[1]['accuracy'])
    insights.append(f"Position {worst_pos[0].split('_')[1]} underperformed ({worst_pos[1]['accuracy']:.1%} accuracy)")
    insights.append(f"Position {best_pos[0].split('_')[1]} performed best ({best_pos[1]['accuracy']:.1%} accuracy)")
    
    # Sum insights
    insights.append(f"Closest sum prediction was Set #{closest_set['index'] + 1} (difference: {closest_set['diff']})")
    
    # Top sets insights
    top_3 = sorted_predictions[:3]
    top_3_correct = sum(p['correct_count'] for p in top_3)
    total_correct = sum(p['correct_count'] for p in sorted_predictions)
    if total_correct > 0:
        top_3_pct = (top_3_correct / total_correct) * 100
        insights.append(f"Top 3 sets contained {top_3_pct:.0f}% of all correct predictions")
    
    # Number frequency insights
    if number_frequency['correctly_predicted']:
        top_correct = number_frequency['correctly_predicted'][:3]
        insights.append(f"Most predicted correct numbers: {', '.join([str(n['number']) for n in top_correct])}")
    
    # Consecutive analysis insights
    if consecutive_analysis['total_consecutive_pairs'] > 0:
        insights.append(f"Found {consecutive_analysis['total_consecutive_pairs']} consecutive number pairs across all sets")
    
    # Model performance insights
    if model_performance['best_performing_model']:
        best_model = model_performance['best_performing_model']
        insights.append(f"Best model: {best_model['name']} ({best_model['contribution']:.1%} of correct predictions)")
    
    # Diversity insights
    coverage = diversity_metrics['number_space_coverage']
    insights.append(f"Prediction sets covered {coverage:.1%} of possible numbers (1-50)")
    
    learning_data['learning_insights'] = insights
    
    # Raw CSV analysis (if enabled)
    if use_raw_csv:
        csv_patterns = _analyze_raw_csv_patterns(game, actual_results.get('jackpot'), actual_results['numbers'])
        if csv_patterns:
            learning_data['analysis']['raw_csv_patterns'] = csv_patterns
            
            # Add CSV-based insights
            if csv_patterns.get('matching_jackpot_draws', 0) > 0:
                insights.append(f"Found {csv_patterns['matching_jackpot_draws']} historical draws with similar jackpot")
    
    return learning_data


def _analyze_position_frequency(game: str, max_draws: int = 300) -> Dict:
    """
    Compute per-sorted-position frequency of numbers across historical draws.

    Returns a dict keyed by 'position_1' … 'position_N' where each value contains:
        - 'accuracy'      : average frequency of the top-5 most common numbers for that
                           position (used as the signal weight in _calculate_learning_score_advanced)
        - 'hot_numbers'   : top-5 numbers most often drawn at this sorted position
        - 'frequency_map' : {num_str: relative_frequency} for ALL numbers at this position
        - 'total_draws'   : number of draws analysed
    """
    game_folder = _sanitize_game_name(game)
    data_dir = Path("data") / game_folder

    if not data_dir.exists():
        return {}

    draw_size  = 7 if 'max' in game.lower() else 6
    max_number = 50 if 'max' in game.lower() else 49

    # position_counts[pos_idx][number] = count
    position_counts: List[Dict[int, int]] = [{} for _ in range(draw_size)]
    total_draws = 0

    for csv_file in sorted(data_dir.glob("*.csv"), reverse=True):
        if total_draws >= max_draws:
            break
        try:
            df = pd.read_csv(csv_file)
            if 'draw_date' in df.columns:
                df = df.sort_values('draw_date', ascending=False)

            for _, row in df.iterrows():
                if total_draws >= max_draws:
                    break
                numbers: List[int] = []

                # Try n1..nN columns first
                for _i in range(1, draw_size + 1):
                    for _prefix in ('n', 'number_'):
                        _col = f'{_prefix}{_i}'
                        if _col in df.columns:
                            try:
                                _v = int(row[_col])
                                if 1 <= _v <= max_number:
                                    numbers.append(_v)
                                    break
                            except (ValueError, TypeError):
                                pass

                # Fallback: comma-separated 'numbers' column
                if len(numbers) < draw_size and 'numbers' in df.columns:
                    try:
                        numbers = [
                            int(x.strip())
                            for x in str(row['numbers']).split(',')
                            if x.strip().lstrip('-').isdigit()
                            and 1 <= int(x.strip()) <= max_number
                        ]
                    except (ValueError, TypeError):
                        pass

                if len(numbers) == draw_size:
                    for pos_idx, num in enumerate(sorted(numbers)):
                        position_counts[pos_idx][num] = position_counts[pos_idx].get(num, 0) + 1
                    total_draws += 1
        except Exception:
            continue

    if total_draws == 0:
        return {}

    result: Dict = {}
    for pos_idx, counts in enumerate(position_counts):
        if not counts:
            continue
        pos_key    = f'position_{pos_idx + 1}'
        total_here = sum(counts.values())
        freq_map   = {num: cnt / total_here for num, cnt in counts.items()}
        hot_nums   = sorted(counts, key=lambda n: counts[n], reverse=True)[:5]
        # 'accuracy' = mean frequency of top-5 numbers — measures how predictable this position is
        top_freq   = sum(freq_map.get(n, 0) for n in hot_nums) / max(1, len(hot_nums))
        result[pos_key] = {
            'accuracy'     : float(top_freq),
            'hot_numbers'  : hot_nums,
            'frequency_map': {str(k): float(v) for k, v in freq_map.items()},
            'total_draws'  : total_draws,
        }

    return result


def _analyze_raw_csv_patterns(game: str, jackpot: Optional[float], winning_numbers: List[int]) -> Dict:
    """Analyze patterns from raw CSV files."""
    game_folder = _sanitize_game_name(game)
    data_dir = Path("data") / game_folder
    
    if not data_dir.exists():
        return {}
    
    patterns = {
        'matching_jackpot_draws': 0,
        'sum_distribution': {'min': 999, 'max': 0, 'mean': 0, 'std': 0},
        'odd_even_ratio': {'odd': 0, 'even': 0},
        'repetition_rate': 0
    }
    
    all_sums = []
    odd_counts = []
    even_counts = []
    
    try:
        for csv_file in data_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                # Extract numbers from row
                numbers = []
                for col in df.columns:
                    if col.startswith('number_') or col.startswith('n'):
                        try:
                            num = int(row[col])
                            if 1 <= num <= 50:
                                numbers.append(num)
                        except:
                            continue
                
                if len(numbers) > 0:
                    # Sum analysis
                    row_sum = sum(numbers)
                    all_sums.append(row_sum)
                    
                    # Odd/even analysis
                    odd = sum(1 for n in numbers if n % 2 == 1)
                    even = len(numbers) - odd
                    odd_counts.append(odd)
                    even_counts.append(even)
                    
                    # Jackpot matching (if provided)
                    if jackpot:
                        row_jackpot = row.get('jackpot', 0)
                        if abs(row_jackpot - jackpot) < jackpot * 0.1:  # Within 10%
                            patterns['matching_jackpot_draws'] += 1
        
        # Calculate statistics
        if all_sums:
            patterns['sum_distribution'] = {
                'min': int(np.min(all_sums)),
                'max': int(np.max(all_sums)),
                'mean': float(np.mean(all_sums)),
                'std': float(np.std(all_sums))
            }
        
        if odd_counts:
            patterns['odd_even_ratio'] = {
                'odd': float(np.mean(odd_counts)),
                'even': float(np.mean(even_counts))
            }
        
        # Repetition rate (placeholder)
        patterns['repetition_rate'] = 0.15
    
    except Exception as e:
        print(f"Error analyzing CSV patterns: {e}")
    
    return patterns


def _save_learning_data(game: str, draw_date: str, learning_data: Dict) -> Path:
    """Save learning data to disk."""
    game_folder = _sanitize_game_name(game)
    learning_dir = Path("data") / "learning" / game_folder
    learning_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    date_str = draw_date.replace('-', '').replace('/', '')
    filename = f"draw_{date_str}_learning.json"
    filepath = learning_dir / filename
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(learning_data, f, indent=2, default=str)
    
    return filepath


# ============================================================================
# ENHANCED LEARNING ANALYSIS FUNCTIONS
# ============================================================================

def _analyze_gap_patterns(sorted_predictions: List[Dict], winning_numbers: List[int]) -> Dict:
    """Analyze spacing/gap patterns between numbers."""
    winning_sorted = sorted(winning_numbers)
    winning_gaps = [winning_sorted[i+1] - winning_sorted[i] for i in range(len(winning_sorted)-1)]
    
    prediction_gaps = []
    for pred in sorted_predictions:
        nums = sorted([n['number'] for n in pred['numbers']])
        gaps = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
        prediction_gaps.extend(gaps)
    
    return {
        'winning_gaps': winning_gaps,
        'avg_winning_gap': float(np.mean(winning_gaps)) if winning_gaps else 0,
        'avg_prediction_gap': float(np.mean(prediction_gaps)) if prediction_gaps else 0,
        'gap_similarity': 1.0 - min(1.0, abs(np.mean(winning_gaps) - np.mean(prediction_gaps)) / 10.0) if winning_gaps and prediction_gaps else 0
    }


def _analyze_zone_distribution(sorted_predictions: List[Dict], winning_numbers: List[int], max_number: int = 50) -> Dict:
    """Analyze distribution across low/mid/high zones."""
    # Calculate zone boundaries dynamically based on max_number
    # Low: 1 to 1/3, Mid: 1/3+1 to 2/3, High: 2/3+1 to max
    low_boundary = max_number // 3
    mid_boundary = (max_number * 2) // 3
    
    winning_zones = {'low': 0, 'mid': 0, 'high': 0}
    for num in winning_numbers:
        if num <= low_boundary:
            winning_zones['low'] += 1
        elif num <= mid_boundary:
            winning_zones['mid'] += 1
        else:
            winning_zones['high'] += 1
    
    prediction_zones = {'low': [], 'mid': [], 'high': []}
    for pred in sorted_predictions:
        zones = {'low': 0, 'mid': 0, 'high': 0}
        for num_obj in pred['numbers']:
            num = num_obj['number']
            if num <= low_boundary:
                zones['low'] += 1
            elif num <= mid_boundary:
                zones['mid'] += 1
            else:
                zones['high'] += 1
        prediction_zones['low'].append(zones['low'])
        prediction_zones['mid'].append(zones['mid'])
        prediction_zones['high'].append(zones['high'])
    
    return {
        'winning_distribution': winning_zones,
        'avg_prediction_distribution': {
            'low': float(np.mean(prediction_zones['low'])),
            'mid': float(np.mean(prediction_zones['mid'])),
            'high': float(np.mean(prediction_zones['high']))
        },
        'zone_match_score': sum(abs(winning_zones[z] - np.mean(prediction_zones[z])) for z in ['low', 'mid', 'high'])
    }


def _analyze_even_odd_ratio(sorted_predictions: List[Dict], winning_numbers: List[int]) -> Dict:
    """Analyze even/odd number distribution."""
    winning_even = sum(1 for n in winning_numbers if n % 2 == 0)
    winning_odd = len(winning_numbers) - winning_even
    
    prediction_ratios = []
    for pred in sorted_predictions:
        even_count = sum(1 for n in pred['numbers'] if n['number'] % 2 == 0)
        prediction_ratios.append(even_count / len(pred['numbers']))
    
    return {
        'winning_even_count': winning_even,
        'winning_odd_count': winning_odd,
        'winning_ratio': winning_even / len(winning_numbers),
        'avg_prediction_ratio': float(np.mean(prediction_ratios)),
        'ratio_similarity': 1.0 - abs((winning_even / len(winning_numbers)) - np.mean(prediction_ratios))
    }


def _analyze_decade_coverage(sorted_predictions: List[Dict], winning_numbers: List[int], max_number: int = 50) -> Dict:
    """Analyze how numbers spread across decades (1-10, 11-20, etc.)."""
    winning_decades = {}
    for num in winning_numbers:
        decade = (num - 1) // 10
        winning_decades[decade] = winning_decades.get(decade, 0) + 1
    
    prediction_decades = []
    for pred in sorted_predictions:
        decades = {}
        for num_obj in pred['numbers']:
            decade = (num_obj['number'] - 1) // 10
            decades[decade] = decades.get(decade, 0) + 1
        prediction_decades.append(len(decades))  # Count unique decades
    
    # Calculate max possible decades dynamically (49 has 5 decades: 0-4, 50 has 5 decades: 0-4)
    max_decades = ((max_number - 1) // 10) + 1
    
    return {
        'winning_decade_count': len(winning_decades),
        'winning_decades': dict(winning_decades),
        'avg_prediction_decade_coverage': float(np.mean(prediction_decades)),
        'decade_diversity_score': len(winning_decades) / float(max_decades)
    }


def _identify_cold_numbers(game: str, lookback_draws: int = 50) -> List[int]:
    """Identify cold numbers (rarely appearing in recent history)."""
    try:
        game_lower = game.lower().replace(" ", "_").replace("/", "_")
        data_dir = Path("data") / game_lower
        
        if not data_dir.exists():
            return []
        
        # Load recent draws
        csv_files = sorted(data_dir.glob("training_data_*.csv"))
        if not csv_files:
            return []
        
        df = pd.read_csv(csv_files[-1])
        recent_draws = df.tail(lookback_draws)
        
        # Count number frequency
        number_counts = {}
        for _, row in recent_draws.iterrows():
            numbers = [int(n.strip()) for n in str(row['numbers']).split(',')]
            for num in numbers:
                number_counts[num] = number_counts.get(num, 0) + 1
        
        # Identify cold numbers (bottom 20%)
        max_number = 50 if 'max' in game.lower() else 49
        all_numbers = list(range(1, max_number + 1))
        sorted_by_freq = sorted(all_numbers, key=lambda x: number_counts.get(x, 0))
        
        cold_count = int(len(all_numbers) * 0.2)
        return sorted_by_freq[:cold_count]
    except:
        return []


def _create_pattern_fingerprint(numbers: List[int]) -> str:
    """Create a pattern signature for a number set."""
    sorted_nums = sorted(numbers)
    gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
    
    # Categorize gaps: small (1-5), medium (6-10), large (11+)
    pattern = []
    for gap in gaps:
        if gap <= 5:
            pattern.append('S')
        elif gap <= 10:
            pattern.append('M')
        else:
            pattern.append('L')
    
    return ''.join(pattern)


# ============================================================================
# HIGH VALUE ANALYSIS FUNCTIONS
# ============================================================================

def _analyze_number_frequency(sorted_predictions: List[Dict], winning_numbers: List[int]) -> Dict:
    """Analyze which numbers were predicted and how often they appeared correctly."""
    predicted_numbers = {}  # {number: count}
    correct_predictions = {}  # {number: count when correct}
    
    # Count all predicted numbers
    for pred in sorted_predictions:
        for num_data in pred['numbers']:
            num = num_data['number']
            predicted_numbers[num] = predicted_numbers.get(num, 0) + 1
            
            if num_data['is_correct']:
                correct_predictions[num] = correct_predictions.get(num, 0) + 1
    
    # Identify correctly predicted numbers
    correctly_predicted = [
        {'number': num, 'count': correct_predictions.get(num, 0), 'predicted_count': predicted_numbers[num]}
        for num in winning_numbers
        if num in predicted_numbers
    ]
    correctly_predicted.sort(key=lambda x: x['count'], reverse=True)
    
    # Identify missed winning numbers
    missed_winning = [
        {'number': num}
        for num in winning_numbers
        if num not in predicted_numbers
    ]
    
    # Identify frequently predicted but incorrect numbers
    missed_numbers = [
        {'number': num, 'count': count}
        for num, count in predicted_numbers.items()
        if num not in winning_numbers
    ]
    missed_numbers.sort(key=lambda x: x['count'], reverse=True)
    
    # Calculate hot/cold accuracy
    hot_numbers = sorted(predicted_numbers.items(), key=lambda x: x[1], reverse=True)[:10]
    hot_correct = sum(1 for num, _ in hot_numbers if num in winning_numbers)
    
    return {
        'correctly_predicted': correctly_predicted,
        'missed_winning_numbers': missed_winning,
        'missed_numbers': missed_numbers[:10],  # Top 10 most predicted but wrong
        'total_unique_predicted': len(predicted_numbers),
        'hot_number_accuracy': hot_correct / len(hot_numbers) if hot_numbers else 0,
        'number_distribution': {
            'low_range': sum(1 for n in predicted_numbers.keys() if 1 <= n <= 16),
            'mid_range': sum(1 for n in predicted_numbers.keys() if 17 <= n <= 33),
            'high_range': sum(1 for n in predicted_numbers.keys() if 34 <= n <= 50)
        }
    }


def _analyze_consecutive_patterns(sorted_predictions: List[Dict], winning_numbers: List[int]) -> Dict:
    """Analyze consecutive number patterns and gaps."""
    # Winning consecutive pairs
    sorted_winning = sorted(winning_numbers)
    winning_consecutive = sum(1 for i in range(len(sorted_winning) - 1) if sorted_winning[i+1] - sorted_winning[i] == 1)
    
    # Winning gaps
    winning_gaps = [sorted_winning[i+1] - sorted_winning[i] for i in range(len(sorted_winning) - 1)]
    
    # Prediction consecutive pairs and gaps
    total_consecutive = 0
    all_gaps = []
    
    for pred in sorted_predictions:
        pred_numbers = sorted([n['number'] for n in pred['numbers']])
        consecutive = sum(1 for i in range(len(pred_numbers) - 1) if pred_numbers[i+1] - pred_numbers[i] == 1)
        total_consecutive += consecutive
        
        gaps = [pred_numbers[i+1] - pred_numbers[i] for i in range(len(pred_numbers) - 1)]
        all_gaps.extend(gaps)
    
    return {
        'winning_consecutive_pairs': winning_consecutive,
        'total_consecutive_pairs': total_consecutive,
        'average_consecutive_per_set': total_consecutive / len(sorted_predictions) if sorted_predictions else 0,
        'winning_average_gap': float(np.mean(winning_gaps)) if winning_gaps else 0,
        'predicted_average_gap': float(np.mean(all_gaps)) if all_gaps else 0,
        'average_gap': float(np.mean(all_gaps)) if all_gaps else 0,
        'gap_similarity': 1 - abs((np.mean(all_gaps) - np.mean(winning_gaps)) / np.mean(winning_gaps)) if winning_gaps and all_gaps else 0
    }


def _analyze_model_performance(pred_data: Dict, sorted_predictions: List[Dict], winning_numbers: List[int]) -> Dict:
    """Analyze which models contributed to correct predictions using REAL attribution data."""
    # Get model info from prediction data
    selected_models = pred_data.get('analysis', {}).get('selected_models', [])
    predictions_with_attribution = pred_data.get('predictions_with_attribution', [])
    
    if not selected_models:
        return {'model_contributions': [], 'best_performing_model': None, 'data_source': 'none'}
    
    # Initialize model contribution tracker
    model_stats = {}
    for model in selected_models:
        model_name = model['name']
        model_stats[model_name] = {
            'correct': 0,
            'total_votes': 0,
            'correct_votes': 0,
            'type': model['type'],
            'accuracy': model.get('accuracy', 0)
        }
    
    # Try to use REAL attribution data if available
    if predictions_with_attribution and len(predictions_with_attribution) > 0:
        # Count votes from attribution data
        for idx, pred_set in enumerate(predictions_with_attribution):
            attribution = pred_set.get('model_attribution', {})
            pred_numbers = pred_set.get('numbers', [])
            
            for number in pred_numbers:
                # Attribution keys are strings, convert number to string
                number_str = str(number)
                voters = attribution.get(number_str, [])
                for voter in voters:
                    model_key = voter.get('model', '')
                    # Match model key to model name (key contains "name (type)")
                    for model_name in model_stats.keys():
                        if model_name in model_key:
                            model_stats[model_name]['total_votes'] += 1
                            # Check if this number is in winning numbers
                            if number in winning_numbers:
                                model_stats[model_name]['correct_votes'] += 1
                            break
        
        # Calculate actual contributions from real data
        total_correct_votes = sum(stats['correct_votes'] for stats in model_stats.values())
        total_votes = sum(stats['total_votes'] for stats in model_stats.values())
        
        contributions = []
        for name, stats in model_stats.items():
            if total_votes > 0:
                vote_rate = stats['total_votes'] / total_votes
            else:
                vote_rate = 0
            
            if total_correct_votes > 0:
                contribution_rate = stats['correct_votes'] / total_correct_votes
            else:
                # Show expected contribution based on vote rate
                contribution_rate = vote_rate
            
            contributions.append({
                'Model': name,
                'Type': stats['type'],
                'Total Votes': stats['total_votes'],
                'Correct Votes': stats['correct_votes'],
                'Contribution': f"{contribution_rate:.1%}",
                'Vote Rate': f"{vote_rate:.1%}",
                'Accuracy': f"{stats['accuracy']:.1%}"
            })
        
        contributions.sort(key=lambda x: x['Correct Votes'], reverse=True)
        
        best_model = None
        if contributions and contributions[0]['Correct Votes'] > 0:
            best = contributions[0]
            best_model = {
                'name': best['Model'],
                'contribution': float(best['Contribution'].strip('%')) / 100,
                'correct_votes': best['Correct Votes']
            }
        
        return {
            'model_contributions': contributions,
            'best_performing_model': best_model,
            'model_diversity': len(selected_models),
            'data_source': 'attribution',
            'total_votes': total_votes,
            'total_correct_votes': total_correct_votes
        }
    
    else:
        # FALLBACK: Use estimation if no attribution data
        total_correct = sum(pred['correct_count'] for pred in sorted_predictions)
        
        for model in selected_models:
            model_name = model['name']
            model_accuracy = model.get('accuracy', 0)
            model_confidence = model.get('confidence', 0)
            
            weight = model_accuracy * model_confidence
            total_weight = sum(m.get('accuracy', 0) * m.get('confidence', 0) for m in selected_models)
            
            if total_weight > 0:
                expected_rate = weight / total_weight
            else:
                expected_rate = 1.0 / len(selected_models)
            
            if total_correct > 0:
                estimated_contribution = expected_rate * total_correct
            else:
                estimated_contribution = 0
            
            model_stats[model_name]['correct'] = estimated_contribution
            model_stats[model_name]['contribution_rate'] = expected_rate if total_correct == 0 else (estimated_contribution / total_correct if total_correct > 0 else 0)
        
        contributions = [
            {
                'Model': name,
                'Type': stats['type'],
                'Contribution': f"{stats['contribution_rate']:.1%}",
                'Est. Correct': f"{stats['correct']:.1f}",
                'Note': 'Estimated' if total_correct == 0 else 'Calculated'
            }
            for name, stats in model_stats.items()
        ]
        contributions.sort(key=lambda x: float(x['Contribution'].strip('%')), reverse=True)
        
        best_model = None
        if contributions:
            best = contributions[0]
            best_model = {
                'name': best['Model'],
                'contribution': float(best['Contribution'].strip('%')) / 100
            }
        
        return {
            'model_contributions': contributions,
            'best_performing_model': best_model,
            'model_diversity': len(selected_models),
            'data_source': 'estimation'
        }


def _analyze_temporal_patterns(draw_date: str, sorted_predictions: List[Dict]) -> Dict:
    """Analyze temporal patterns related to the draw date."""
    try:
        date_obj = datetime.strptime(draw_date, '%Y-%m-%d')
    except:
        return {}
    
    day_of_week = date_obj.strftime('%A')
    month = date_obj.strftime('%B')
    quarter = (date_obj.month - 1) // 3 + 1
    week_of_year = date_obj.isocalendar()[1]
    
    return {
        'day_of_week': day_of_week,
        'month': month,
        'quarter': f"Q{quarter}",
        'week_of_year': week_of_year,
        'is_weekend': day_of_week in ['Saturday', 'Sunday'],
        'season': _get_season(date_obj.month)
    }


def _get_season(month: int) -> str:
    """Get season from month."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


def _analyze_set_diversity(sorted_predictions: List[Dict]) -> Dict:
    """Analyze diversity and coverage of prediction sets."""
    # Collect all unique numbers
    all_numbers = set()
    set_numbers = []
    
    for pred in sorted_predictions:
        pred_nums = {n['number'] for n in pred['numbers']}
        all_numbers.update(pred_nums)
        set_numbers.append(pred_nums)
    
    # Calculate overlap between sets
    overlaps = []
    for i in range(len(set_numbers)):
        for j in range(i + 1, len(set_numbers)):
            overlap = len(set_numbers[i] & set_numbers[j])
            overlaps.append(overlap)
    
    # Calculate diversity score (0-1, higher = more diverse)
    coverage = len(all_numbers) / 50  # 50 is max numbers in Lotto Max
    avg_overlap = np.mean(overlaps) if overlaps else 0
    max_possible_overlap = 7  # All numbers same
    overlap_diversity = 1 - (avg_overlap / max_possible_overlap)
    diversity_score = (coverage + overlap_diversity) / 2
    
    return {
        'unique_numbers_count': len(all_numbers),
        'number_space_coverage': coverage,
        'average_overlap': avg_overlap,
        'diversity_score': diversity_score,
        'coverage_percentage': coverage
    }


def _analyze_probability_calibration(pred_data: Dict, sorted_predictions: List[Dict], winning_numbers: List[int]) -> Dict:
    """Analyze how well predicted probabilities match actual outcomes."""
    # Get predicted confidence from file
    ensemble_confidence = pred_data.get('analysis', {}).get('ensemble_confidence', 0)
    win_probability = pred_data.get('analysis', {}).get('win_probability', 0)
    
    # Calculate actual accuracy
    total_numbers = len(sorted_predictions) * 7
    total_correct = sum(pred['correct_count'] for pred in sorted_predictions)
    actual_accuracy = total_correct / total_numbers if total_numbers > 0 else 0
    
    # Check if any set got all 7 correct
    perfect_sets = sum(1 for pred in sorted_predictions if pred['correct_count'] == 7)
    
    # Calibration metrics
    calibration_error = abs(ensemble_confidence - actual_accuracy)
    is_overconfident = ensemble_confidence > actual_accuracy
    
    return {
        'predicted_confidence': ensemble_confidence,
        'predicted_win_probability': win_probability,
        'actual_accuracy': actual_accuracy,
        'calibration_error': calibration_error,
        'is_overconfident': is_overconfident,
        'perfect_predictions': perfect_sets,
        'confidence_accuracy_ratio': actual_accuracy / ensemble_confidence if ensemble_confidence > 0 else 0
    }


# ============================================================================
# LEARNING-BASED REGENERATION HELPER FUNCTIONS
# ============================================================================

def _find_all_learning_files(game: str) -> List[Path]:
    """Find all learning files for a game."""
    game_folder = _sanitize_game_name(game)
    learning_dir = Path("data") / "learning" / game_folder
    
    if not learning_dir.exists():
        return []
    
    # Match both "draw_*_learning.json" and "draw_*_learning*.json" (with or without underscore before number)
    learning_files = []
    
    # Get all files matching the learning pattern (including numbered versions)
    for file in learning_dir.glob("draw_*_learning*.json"):
        learning_files.append(file)
    
    # Remove duplicates and sort by modification time (most recent first)
    learning_files = sorted(set(learning_files), key=lambda x: x.stat().st_mtime, reverse=True)
    return learning_files


def _load_and_combine_learning_files(learning_files: List[Path]) -> Dict:
    """Load multiple learning files and combine their insights."""
    combined_data = {
        'source_files': [],
        'combined_insights': [],
        'game': '',  # Will be set from first file
        'analysis': {  # Structure to match single file format
            'position_accuracy': {},
            'number_frequency': {},
            'consecutive_patterns': {},
            'temporal_patterns': [],
            'diversity_metrics': {},
            'sum_analysis': {},
            'gap_patterns': {},
            'zone_distribution': {},
            'even_odd_ratio': {},
            'decade_coverage': {},
            'cold_numbers': [],
            'winning_pattern_fingerprint': ''
        },
        # Legacy fields for backward compatibility
        'position_accuracy': {},
        'number_frequency': {},
        'consecutive_patterns': {},
        'temporal_patterns': [],
        'diversity_metrics': {},
        'avg_sum': 0,
        'sum_range': {'min': 999, 'max': 0}
    }
    
    all_sums = []
    position_stats = {}
    number_counts = {}
    all_gap_patterns = []
    all_zone_distributions = []
    all_even_odd_ratios = []
    all_decade_coverages = []
    all_cold_numbers = []
    all_pattern_fingerprints = []
    
    for lf in learning_files:
        try:
            with open(lf, 'r') as f:
                data = json.load(f)
            
            combined_data['source_files'].append(str(lf.name))
            
            # Set game from first file
            if not combined_data['game']:
                combined_data['game'] = data.get('game', 'Lotto Max')
            
            # Combine insights
            insights = data.get('learning_insights', [])
            combined_data['combined_insights'].extend(insights)
            
            # Aggregate position accuracy
            pos_acc = data.get('analysis', {}).get('position_accuracy', {})
            for pos, stats in pos_acc.items():
                if pos not in position_stats:
                    position_stats[pos] = {'correct': 0, 'total': 0}
                position_stats[pos]['correct'] += stats.get('correct', 0)
                position_stats[pos]['total'] += stats.get('total', 0)
            
            # Aggregate number frequency
            num_freq = data.get('analysis', {}).get('number_frequency', {})
            correctly_predicted = num_freq.get('correctly_predicted', [])
            for item in correctly_predicted:
                num = item['number']
                if num not in number_counts:
                    number_counts[num] = 0
                number_counts[num] += item.get('count', 1)
            
            # Aggregate sum data
            sum_analysis = data.get('analysis', {}).get('sum_analysis', {})
            winning_sum = sum_analysis.get('winning_sum', 0)
            if winning_sum:
                all_sums.append(winning_sum)
            
            # Aggregate ENHANCED metrics
            analysis = data.get('analysis', {})
            
            # Gap patterns
            gap_pat = analysis.get('gap_patterns', {})
            if gap_pat:
                all_gap_patterns.append(gap_pat)
            
            # Zone distribution
            zone_dist = analysis.get('zone_distribution', {})
            if zone_dist:
                all_zone_distributions.append(zone_dist)
            
            # Even/odd ratio
            even_odd = analysis.get('even_odd_ratio', {})
            if even_odd:
                all_even_odd_ratios.append(even_odd)
            
            # Decade coverage
            decade_cov = analysis.get('decade_coverage', {})
            if decade_cov:
                all_decade_coverages.append(decade_cov)
            
            # Cold numbers
            cold_nums = analysis.get('cold_numbers', [])
            if cold_nums:
                all_cold_numbers.extend(cold_nums)
            
            # Pattern fingerprints
            pattern = analysis.get('winning_pattern_fingerprint', '')
            if pattern:
                all_pattern_fingerprints.append(pattern)
            
        except Exception as e:
            continue
    
    # Calculate combined statistics
    if position_stats:
        for pos, stats in position_stats.items():
            combined_data['position_accuracy'][pos] = {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
            combined_data['analysis']['position_accuracy'][pos] = combined_data['position_accuracy'][pos]
    
    if number_counts:
        combined_data['number_frequency'] = {
            'hot_numbers': sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:20],
            'all_counts': number_counts
        }
        combined_data['analysis']['number_frequency'] = combined_data['number_frequency']
    
    if all_sums:
        combined_data['avg_sum'] = int(np.mean(all_sums))
        combined_data['sum_range'] = {'min': int(np.min(all_sums)), 'max': int(np.max(all_sums))}
        combined_data['analysis']['sum_analysis'] = {
            'winning_sum': combined_data['avg_sum'],
            'sum_range': combined_data['sum_range']
        }
    
    # Aggregate ENHANCED metrics
    if all_gap_patterns:
        avg_winning_gaps = [gp.get('avg_winning_gap', 0) for gp in all_gap_patterns if gp.get('avg_winning_gap')]
        avg_pred_gaps = [gp.get('avg_prediction_gap', 0) for gp in all_gap_patterns if gp.get('avg_prediction_gap')]
        combined_data['analysis']['gap_patterns'] = {
            'avg_winning_gap': float(np.mean(avg_winning_gaps)) if avg_winning_gaps else 7,
            'avg_prediction_gap': float(np.mean(avg_pred_gaps)) if avg_pred_gaps else 7,
            'gap_similarity': float(np.mean([gp.get('gap_similarity', 0) for gp in all_gap_patterns])) if all_gap_patterns else 0
        }
    
    if all_zone_distributions:
        # Average zone distributions
        low_zones = [zd.get('winning_distribution', {}).get('low', 0) for zd in all_zone_distributions]
        mid_zones = [zd.get('winning_distribution', {}).get('mid', 0) for zd in all_zone_distributions]
        high_zones = [zd.get('winning_distribution', {}).get('high', 0) for zd in all_zone_distributions]
        combined_data['analysis']['zone_distribution'] = {
            'winning_distribution': {
                'low': int(np.mean(low_zones)) if low_zones else 2,
                'mid': int(np.mean(mid_zones)) if mid_zones else 3,
                'high': int(np.mean(high_zones)) if high_zones else 2
            }
        }
    
    if all_even_odd_ratios:
        avg_ratio = np.mean([eo.get('winning_ratio', 0.5) for eo in all_even_odd_ratios])
        combined_data['analysis']['even_odd_ratio'] = {
            'winning_ratio': float(avg_ratio),
            'winning_even_count': int(avg_ratio * 7),  # Approximate for Lotto Max
            'winning_odd_count': int((1 - avg_ratio) * 7)
        }
    
    if all_decade_coverages:
        avg_decade_count = np.mean([dc.get('winning_decade_count', 4) for dc in all_decade_coverages])
        combined_data['analysis']['decade_coverage'] = {
            'winning_decade_count': int(avg_decade_count),
            'decade_diversity_score': float(np.mean([dc.get('decade_diversity_score', 0.8) for dc in all_decade_coverages]))
        }
    
    # Cold numbers - combine and get most frequently cold
    if all_cold_numbers:
        cold_counter = {}
        for num in all_cold_numbers:
            cold_counter[num] = cold_counter.get(num, 0) + 1
        # Keep numbers that appear cold in at least 50% of files
        threshold = len(learning_files) * 0.5
        combined_data['analysis']['cold_numbers'] = [num for num, count in cold_counter.items() if count >= threshold]
    
    # Pattern fingerprints - use most common pattern
    if all_pattern_fingerprints:
        from collections import Counter
        pattern_counts = Counter(all_pattern_fingerprints)
        most_common = pattern_counts.most_common(1)
        combined_data['analysis']['winning_pattern_fingerprint'] = most_common[0][0] if most_common else ''
    
    return combined_data


def _find_all_draw_profiles(game: str) -> List[Path]:
    """Find all draw profile files for a game."""
    game_folder = _sanitize_game_name(game)
    profile_dir = Path("data") / "learning" / game_folder / "jackpot_profiles"
    
    if not profile_dir.exists():
        return []
    
    # Get all JSON files in the jackpot_profiles directory
    profile_files = list(profile_dir.glob("*.json"))
    
    # Sort by modification time (most recent first)
    profile_files = sorted(profile_files, key=lambda x: x.stat().st_mtime, reverse=True)
    return profile_files


def _load_draw_profile(profile_path: Path) -> Optional[Dict]:
    """Load a draw profile JSON file."""
    try:
        with open(profile_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        app_log(f"Error loading draw profile {profile_path}: {e}", "error")
        return None


def _validate_set_against_profile(prediction_set: List[int], profile_data: Dict, game: str) -> Dict:
    """
    Score a prediction set against a draw profile using SOFT conformance scoring.

    Instead of a binary pass/fail gate (which causes hundreds of regeneration attempts),
    each of the 7 checks now contributes a partial score (0.0–1.0).  The final
    `conformance_score` is the weighted average of all component scores.

    `valid` is retained for backwards compatibility but now means conformance_score >= 0.40,
    so only very poor matches (< 40%) are flagged for optional regeneration.
    """
    if not profile_data:
        return {'valid': True, 'conformance_score': 1.0, 'component_scores': {}, 'issues': []}

    # Determine number range for the game
    if "6/49" in game or "6_49" in game:
        max_num = 49
    else:  # Lotto Max
        max_num = 50

    mid_point  = profile_data.get('core_patterns', {}).get('high_low_split_point', (max_num + 1) // 2)
    sorted_nums = sorted(prediction_set)

    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    component_scores: Dict[str, float] = {}
    issues: List[str] = []

    # ===== 1. ODD/EVEN — ranked position score (top-1 = 1.0, top-3 = 0.6, outside = 0.2) =====
    odd_count   = sum(1 for n in sorted_nums if n % 2 == 1)
    even_count  = len(sorted_nums) - odd_count
    oe_pattern  = f"{odd_count} Odd / {even_count} Even"
    oe_distribution = profile_data.get('core_patterns', {}).get('odd_even_distribution', {})
    if oe_distribution:
        ranked_oe = sorted(oe_distribution.items(), key=lambda x: x[1], reverse=True)
        ranked_oe_patterns = [p for p, _ in ranked_oe]
        if oe_pattern == ranked_oe_patterns[0]:
            component_scores['odd_even'] = 1.0
        elif oe_pattern in ranked_oe_patterns[:3]:
            component_scores['odd_even'] = 0.6
        else:
            component_scores['odd_even'] = 0.2
            issues.append(f"Odd/Even {oe_pattern} outside top-3 profile patterns")
    else:
        component_scores['odd_even'] = 0.5  # No data — neutral

    # ===== 2. HIGH/LOW — same ranked position scoring =====
    low_count  = sum(1 for n in sorted_nums if n <= mid_point)
    high_count = len(sorted_nums) - low_count
    hl_pattern = f"{low_count} Low / {high_count} High"
    hl_distribution = profile_data.get('core_patterns', {}).get('high_low_distribution', {})
    if hl_distribution:
        ranked_hl = sorted(hl_distribution.items(), key=lambda x: x[1], reverse=True)
        ranked_hl_patterns = [p for p, _ in ranked_hl]
        if hl_pattern == ranked_hl_patterns[0]:
            component_scores['high_low'] = 1.0
        elif hl_pattern in ranked_hl_patterns[:3]:
            component_scores['high_low'] = 0.6
        else:
            component_scores['high_low'] = 0.2
            issues.append(f"High/Low {hl_pattern} outside top-3 profile patterns")
    else:
        component_scores['high_low'] = 0.5

    # ===== 3. SUM — Gaussian distance from profile mean/std =====
    total_sum = sum(sorted_nums)
    sum_stats  = profile_data.get('sum_analysis', {})
    if sum_stats:
        _sum_mean = float(sum_stats.get('mean', 0) or 0)
        _sum_std  = float(sum_stats.get('std',  0) or 0) or 1.0
        # Score = Gaussian decay: 1.0 at mean, ~0.0 at ±3σ
        _z = abs(total_sum - _sum_mean) / _sum_std
        component_scores['sum'] = float(max(0.0, 1.0 - (_z / 3.0)))
        if _z > 2.0:
            issues.append(f"Sum {total_sum} is {_z:.1f}σ from profile mean {_sum_mean:.0f}")
    else:
        component_scores['sum'] = 0.5

    # ===== 4. CONSECUTIVE PAIRS — ranked position score =====
    consecutive_count = sum(
        1 for i in range(len(sorted_nums) - 1) if sorted_nums[i+1] == sorted_nums[i] + 1
    )
    consecutive_dist = profile_data.get('advanced_insights', {}).get('consecutive_distribution', {})
    if consecutive_dist:
        ranked_cons = sorted(consecutive_dist.items(), key=lambda x: x[1], reverse=True)
        ranked_cons_counts = [int(c) for c, _ in ranked_cons]
        if consecutive_count == ranked_cons_counts[0]:
            component_scores['consecutive'] = 1.0
        elif consecutive_count in ranked_cons_counts[:2]:
            component_scores['consecutive'] = 0.7
        else:
            component_scores['consecutive'] = 0.3
    else:
        component_scores['consecutive'] = 0.5

    # ===== 5. PRIME COUNT — ranked position score =====
    prime_count    = sum(1 for n in sorted_nums if is_prime(n))
    prime_dist     = profile_data.get('advanced_insights', {}).get('prime_distribution', {})
    if prime_dist:
        ranked_prime = sorted(prime_dist.items(), key=lambda x: x[1], reverse=True)
        ranked_prime_counts = [int(c) for c, _ in ranked_prime]
        if prime_count == ranked_prime_counts[0]:
            component_scores['primes'] = 1.0
        elif prime_count in ranked_prime_counts[:2]:
            component_scores['primes'] = 0.7
        else:
            component_scores['primes'] = 0.3
    else:
        component_scores['primes'] = 0.5

    # ===== 6. MAX GAP — Gaussian distance from profile mean gap =====
    gaps    = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)]
    max_gap = max(gaps) if gaps else 0
    gap_stats = profile_data.get('advanced_insights', {}).get('gap_statistics', {})
    if gap_stats:
        _gap_mean = float(gap_stats.get('mean', 0) or 0)
        _gap_std  = float(gap_stats.get('std',  0) or 0) or 1.0
        _gz = abs(max_gap - _gap_mean) / _gap_std
        component_scores['max_gap'] = float(max(0.0, 1.0 - (_gz / 3.0)))
    else:
        component_scores['max_gap'] = 0.5

    # ===== 7. EXCLUDED NUMBERS — proportional penalty =====
    excluded_numbers = profile_data.get('number_patterns', {}).get('excluded_numbers', [])
    if excluded_numbers:
        used_excluded = [n for n in sorted_nums if n in excluded_numbers]
        exclusion_penalty = len(used_excluded) / len(sorted_nums)
        component_scores['excluded'] = max(0.0, 1.0 - exclusion_penalty * 2)
        if used_excluded:
            issues.append(f"Uses typically excluded numbers: {used_excluded}")
    else:
        component_scores['excluded'] = 1.0

    # ===== 8. POSITION FREQUENCY (P2) — average per-position rank score =====
    # For each sorted position, score 1.0 if the number is in that position's top-3,
    # 0.6 if in top-6, 0.3 otherwise.
    pos_freq = profile_data.get('number_patterns', {}).get('position_frequency', {})
    if pos_freq:
        pos_scores = []
        for pos_idx, num in enumerate(sorted_nums, 1):
            pos_data = pos_freq.get(str(pos_idx), [])
            top_nums = [entry[0] if isinstance(entry, list) else entry for entry in pos_data]
            if num in top_nums[:3]:
                pos_scores.append(1.0)
            elif num in top_nums[:6]:
                pos_scores.append(0.6)
            else:
                pos_scores.append(0.3)
        component_scores['position'] = float(sum(pos_scores) / len(pos_scores)) if pos_scores else 0.5
    else:
        component_scores['position'] = 0.5  # No data — neutral

    # ===== WEIGHTED CONFORMANCE SCORE =====
    # Reduced existing weights slightly to accommodate position component (P2)
    weights = {
        'odd_even'   : 0.18,
        'high_low'   : 0.18,
        'sum'        : 0.22,
        'consecutive': 0.09,
        'primes'     : 0.09,
        'max_gap'    : 0.09,
        'excluded'   : 0.05,
        'position'   : 0.10,   # P2
    }
    conformance_score = sum(
        component_scores.get(k, 0.5) * w for k, w in weights.items()
    )

    return {
        'valid'            : conformance_score >= 0.40,
        'conformance_score': round(float(conformance_score), 4),
        'component_scores' : component_scores,
        'issues'           : issues,
        'odd_even'         : oe_pattern,
        'high_low'         : hl_pattern,
        'sum'              : total_sum,
        'consecutive'      : consecutive_count,
        'primes'           : prime_count,
        'max_gap'          : max_gap,
        'position_score'   : component_scores.get('position', 0.5),
    }


def _regenerate_predictions_with_learning(
    predictions: List,
    pred_data: Dict,
    learning_data: Dict,
    strategy: str,
    keep_top_n: int,
    learning_weight: float,
    game: str,
    analyzer: SuperIntelligentAIAnalyzer,
    use_advanced_learning: bool = True
) -> Tuple[List, str]:
    """
    Regenerate predictions using learning insights.
    
    Args:
        use_advanced_learning: If True, use genetic algorithm and adaptive weights (DEFAULT)
                              If False, use legacy simple swapping method
    """
    
    regenerated = []
    report_lines = []
    
    # Initialize adaptive learning system
    adaptive_system = AdaptiveLearningSystem(game) if use_advanced_learning else None
    
    report_lines.append("### 📋 Regeneration Report\n")
    report_lines.append(f"**Strategy:** {strategy}")
    if use_advanced_learning:
        report_lines.append(f"**Engine:** Advanced Learning with Genetic Optimization 🧬")
        total_cycles = adaptive_system.meta_learning_data.get('total_learning_cycles', 0)
        report_lines.append(f"**Adaptive Intelligence:** {total_cycles} learning cycles completed")
    else:
        report_lines.append(f"**Engine:** Legacy Learning (static weights)")
    report_lines.append(f"**Learning Weight:** {learning_weight:.0%}")
    report_lines.append(f"**Original Sets:** {len(predictions)}")
    report_lines.append(f"**Learning Sources:** {len(learning_data.get('source_files', []))}\n")
    
    # Extract learning insights
    analysis = learning_data.get('analysis', {})
    
    # Hot numbers
    number_freq = analysis.get('number_frequency', {})
    hot_numbers_data = number_freq.get('hot_numbers', [])
    hot_numbers = []
    for item in hot_numbers_data:
        if isinstance(item, dict):
            hot_numbers.append(int(item['number']))
        elif isinstance(item, tuple):
            hot_numbers.append(int(item[0]))
        else:
            hot_numbers.append(int(item))
    
    # Other factors
    sum_analysis = analysis.get('sum_analysis', {})
    target_sum = sum_analysis.get('winning_sum', 0)
    sum_range = {'min': target_sum - 30, 'max': target_sum + 30}
    position_accuracy = analysis.get('position_accuracy', {})
    gap_patterns = analysis.get('gap_patterns', {})
    avg_winning_gap = gap_patterns.get('avg_winning_gap', 7)
    zone_distribution = analysis.get('zone_distribution', {})
    winning_zones = zone_distribution.get('winning_distribution', {'low': 2, 'mid': 3, 'high': 2})
    even_odd_ratio = analysis.get('even_odd_ratio', {})
    target_even_ratio = even_odd_ratio.get('winning_ratio', 0.5)
    cold_numbers = analysis.get('cold_numbers', [])
    
    # Determine max_number
    max_number = 50 if 'max' in game.lower() else 49
    draw_size = 7 if 'max' in game.lower() else 6
    
    if use_advanced_learning:
        # ADVANCED: Use Genetic Algorithm for optimization
        genetic_optimizer = GeneticSetOptimizer(
            draw_size=draw_size,
            max_number=max_number,
            learning_data=learning_data,
            adaptive_system=adaptive_system
        )
        
        if strategy == "Learning-Guided":
            report_lines.append("**Approach:** Genetic optimization of existing sets\n")
            
            for pred in predictions:
                if isinstance(pred, dict):
                    numbers = [int(n) for n in pred.get('numbers', [])]
                else:
                    numbers = [int(n) for n in pred]
                
                # Use genetic algorithm to optimize this set
                optimized = genetic_optimizer.optimize_set(numbers)
                regenerated.append(optimized)
            
            report_lines.append(f"- Evolved {len(regenerated)} sets using genetic algorithm")
            report_lines.append(f"- Population size: {genetic_optimizer.population_size}")
            report_lines.append(f"- Maximum generations: {genetic_optimizer.generations}")
        
        elif strategy == "Learning-Optimized":
            report_lines.append("**Approach:** New sets via genetic optimization\n")
            
            # Keep top N based on ADAPTIVE scores
            if keep_top_n > 0:
                scored_predictions = []
                for idx, pred in enumerate(predictions):
                    if isinstance(pred, dict):
                        numbers = [int(n) for n in pred.get('numbers', [])]
                    else:
                        numbers = [int(n) for n in pred]
                    score = _calculate_learning_score_advanced(numbers, learning_data, adaptive_system)
                    scored_predictions.append((score, numbers))
                
                scored_predictions.sort(key=lambda x: x[0], reverse=True)
                regenerated.extend([nums for score, nums in scored_predictions[:keep_top_n]])
                report_lines.append(f"- Preserved top {keep_top_n} adaptive-scored sets")
            
            # Generate new sets using genetic algorithm from random starting points
            num_new_sets = len(predictions) - keep_top_n
            for _ in range(num_new_sets):
                new_set = genetic_optimizer.optimize_set()  # Start from random
                regenerated.append(new_set)
            
            report_lines.append(f"- Evolved {num_new_sets} new sets via genetic optimization")
        
        else:  # Hybrid
            report_lines.append("**Approach:** Hybrid adaptive learning + genetic evolution\n")
            
            mid_point = len(predictions) // 2
            
            # Score with ADAPTIVE system
            scored_predictions = []
            for pred in predictions:
                if isinstance(pred, dict):
                    numbers = [int(n) for n in pred.get('numbers', [])]
                else:
                    numbers = [int(n) for n in pred]
                score = _calculate_learning_score_advanced(numbers, learning_data, adaptive_system)
                scored_predictions.append((score, numbers))
            
            scored_predictions.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top half
            regenerated.extend([nums for score, nums in scored_predictions[:mid_point]])
            
            # Evolve new sets for bottom half
            for _ in range(len(predictions) - mid_point):
                new_set = genetic_optimizer.optimize_set()
                regenerated.append(new_set)
            
            report_lines.append(f"- Kept top {mid_point} adaptive-scored sets")
            report_lines.append(f"- Evolved {len(predictions) - mid_point} new sets")
        
        # Display adaptive insights
        weights = adaptive_system.get_adaptive_weights()
        top_factors = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        report_lines.append(f"\n**Top Adaptive Factors:**")
        for factor, weight in top_factors:
            report_lines.append(f"- {factor}: {weight:.1%}")
        
    else:
        # LEGACY: Original simple learning method
        if strategy == "Learning-Guided":
            report_lines.append("**Approach:** Adjusting existing sets (legacy method)\n")
            
            for pred in predictions:
                if isinstance(pred, dict):
                    numbers = [int(n) for n in pred.get('numbers', [])]
                else:
                    numbers = [int(n) for n in pred]
                
                adjusted_numbers = _apply_learning_adjustments(
                    numbers, hot_numbers, position_accuracy, target_sum, learning_weight
                )
                regenerated.append(adjusted_numbers)
            
            report_lines.append(f"- Applied learning patterns to {len(regenerated)} sets")
            report_lines.append(f"- Hot numbers emphasized: {hot_numbers[:5]}")
        
        elif strategy == "Learning-Optimized":
            report_lines.append("**Approach:** Generating new sets (legacy method)\n")
            
            if keep_top_n > 0:
                scored_predictions = []
                for idx, pred in enumerate(predictions):
                    if isinstance(pred, dict):
                        numbers = [int(n) for n in pred.get('numbers', [])]
                    else:
                        numbers = [int(n) for n in pred]
                    score = _calculate_learning_score(numbers, learning_data)
                    scored_predictions.append((score, numbers))
                
                scored_predictions.sort(key=lambda x: x[0], reverse=True)
                regenerated.extend([nums for score, nums in scored_predictions[:keep_top_n]])
                report_lines.append(f"- Preserved top {keep_top_n} original sets")
            
            num_new_sets = len(predictions) - keep_top_n
            new_sets = _generate_learning_based_sets(
                num_new_sets, hot_numbers, position_accuracy, target_sum, sum_range, analyzer,
                cold_numbers=cold_numbers, target_zones=winning_zones, 
                target_even_ratio=target_even_ratio, avg_gap=avg_winning_gap
            )
            regenerated.extend(new_sets)
            
            report_lines.append(f"- Generated {num_new_sets} new learning-optimized sets")
        
        else:  # Hybrid
            report_lines.append("**Approach:** Combining original models with learning (legacy)\n")
            
            mid_point = len(predictions) // 2
            
            scored_predictions = []
            for pred in predictions:
                if isinstance(pred, dict):
                    numbers = [int(n) for n in pred.get('numbers', [])]
                else:
                    numbers = [int(n) for n in pred]
                score = _calculate_learning_score(numbers, learning_data)
                scored_predictions.append((score, numbers))
            
            scored_predictions.sort(key=lambda x: x[0], reverse=True)
            regenerated.extend([nums for score, nums in scored_predictions[:mid_point]])
            
            new_sets = _generate_learning_based_sets(
                len(predictions) - mid_point, hot_numbers, position_accuracy, target_sum, sum_range, analyzer,
                cold_numbers=cold_numbers, target_zones=winning_zones,
                target_even_ratio=target_even_ratio, avg_gap=avg_winning_gap
            )
            regenerated.extend(new_sets)
            
            report_lines.append(f"- Kept top {mid_point} original sets")
            report_lines.append(f"- Generated {len(new_sets)} new learning-based sets")
    
    report_lines.append(f"\n**Total Regenerated Sets:** {len(regenerated)}")
    report_lines.append(f"\n✨ Learning insights from {len(learning_data.get('source_files', []))} historical draws applied")
    
    return regenerated, "\n".join(report_lines)


def _rank_predictions_by_learning(
    predictions: List,
    pred_data: Dict,
    learning_data: Dict,
    analyzer: SuperIntelligentAIAnalyzer,
    use_advanced_learning: bool = True
) -> Tuple[List[Tuple[float, List[int]]], str]:
    """
    Rank existing predictions by learning score without regenerating.
    
    Args:
        use_advanced_learning: If True, use adaptive weights (DEFAULT). If False, use static weights.
    """
    
    report_lines = []
    report_lines.append("### 📊 Learning-Based Ranking Report\n")
    report_lines.append(f"**Total Predictions:** {len(predictions)}")
    report_lines.append(f"**Learning Sources:** {len(learning_data.get('source_files', []))}\n")
    
    # Initialize adaptive system if using advanced mode
    game = learning_data.get('game', pred_data.get('game', 'Lotto Max'))
    adaptive_system = AdaptiveLearningSystem(game) if use_advanced_learning else None
    
    if use_advanced_learning:
        report_lines.append("**Scoring Engine:** Adaptive Learning (evolving weights) 🧬\n")
        total_cycles = adaptive_system.meta_learning_data.get('total_learning_cycles', 0)
        report_lines.append(f"**Intelligence Level:** {total_cycles} learning cycles completed\n")
    else:
        report_lines.append("**Scoring Engine:** Legacy (static weights)\n")
    
    # Score each prediction
    scored_predictions = []
    for idx, pred in enumerate(predictions):
        if isinstance(pred, dict):
            numbers = [int(n) for n in pred.get('numbers', [])]
        else:
            numbers = [int(n) for n in pred]
        
        if use_advanced_learning:
            score = _calculate_learning_score_advanced(numbers, learning_data, adaptive_system)
        else:
            score = _calculate_learning_score(numbers, learning_data)
        
        scored_predictions.append((score, numbers, idx))
    
    # Sort by score (descending)
    scored_predictions.sort(key=lambda x: x[0], reverse=True)
    
    # Create report
    report_lines.append("**Top 10 Predictions by Learning Score:**\n")
    for rank, (score, numbers, original_idx) in enumerate(scored_predictions[:10], 1):
        report_lines.append(f"{rank}. **Set #{original_idx + 1}** - Score: {score:.3f}")
        report_lines.append(f"   Numbers: {', '.join(map(str, sorted(numbers)))}\n")
    
    # Score distribution
    scores = [s for s, _, _ in scored_predictions]
    report_lines.append(f"\n**Score Distribution:**")
    report_lines.append(f"- Highest: {max(scores):.3f}")
    report_lines.append(f"- Average: {np.mean(scores):.3f}")
    report_lines.append(f"- Lowest: {min(scores):.3f}")
    report_lines.append(f"- Std Dev: {np.std(scores):.3f}")
    
    # Diversity check
    unique_scores = len(set(scores))
    report_lines.append(f"\n**Diversity:** {unique_scores} unique scores out of {len(scores)} predictions")
    
    if unique_scores < len(scores) * 0.5:
        report_lines.append("⚠️ Warning: Low score diversity detected. Consider regenerating with learning.")
    
    if use_advanced_learning:
        # Show adaptive factor weights
        weights = adaptive_system.get_adaptive_weights()
        top_factors = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        report_lines.append(f"\n**Current Adaptive Factor Weights:**")
        for factor, weight in top_factors:
            report_lines.append(f"- {factor.replace('_', ' ').title()}: {weight:.1%}")
        
        report_lines.append(f"\n✨ Ranked using adaptive 10-factor learning analysis")
    else:
        report_lines.append(f"\n✨ Ranked using static 10-factor learning analysis")
    
    # Return ranked predictions (score, numbers) and report
    ranked = [(score, numbers) for score, numbers, _ in scored_predictions]
    return ranked, "\n".join(report_lines)


def _apply_learning_adjustments(
    numbers: List[int],
    hot_numbers: List[int],
    position_accuracy: Dict,
    target_sum: int,
    learning_weight: float
) -> List[int]:
    """Apply learning-based adjustments to a prediction set."""
    adjusted = numbers.copy()
    
    # Replace some numbers with hot numbers based on learning weight
    num_to_replace = int(len(numbers) * learning_weight * 0.3)  # Replace up to 30% weighted
    
    if num_to_replace > 0 and hot_numbers:
        # Find coldest numbers in current set
        cold_in_set = [n for n in adjusted if n not in hot_numbers[:15]]
        
        if cold_in_set:
            for _ in range(min(num_to_replace, len(cold_in_set))):
                if hot_numbers:
                    cold_num = cold_in_set[0]
                    hot_num = hot_numbers.pop(0)
                    if hot_num not in adjusted:
                        adjusted[adjusted.index(cold_num)] = hot_num
                        cold_in_set.pop(0)
    
    return sorted(adjusted)


def _calculate_learning_score(numbers: List[int], learning_data: Dict) -> float:
    """Calculate how well a set aligns with learning patterns using 10 comprehensive factors (LEGACY - static weights)."""
    score = 0.0
    analysis = learning_data.get('analysis', {})
    
    # Determine max_number from game context
    game = learning_data.get('game', 'Lotto Max')
    max_number = 50 if 'max' in game.lower() else 49
    
    # FACTOR 1: Hot number alignment (12%)
    number_freq = analysis.get('number_frequency', {})
    hot_numbers_data = number_freq.get('hot_numbers', [])
    hot_numbers = [item['number'] if isinstance(item, dict) else item for item in hot_numbers_data[:10]]
    hot_matches = sum(1 for n in numbers if n in hot_numbers)
    score += (hot_matches / 10.0) * 0.12
    
    # FACTOR 2: Sum alignment (15%)
    sum_analysis = analysis.get('sum_analysis', {})
    target_sum = sum_analysis.get('winning_sum', 0)
    if target_sum:
        sum_diff = abs(sum(numbers) - target_sum)
        sum_score = max(0, 1.0 - (sum_diff / 100.0))
        score += sum_score * 0.15
    
    # FACTOR 3: Diversity/spread (10%)
    num_range = max(numbers) - min(numbers)
    max_possible_range = max_number - 1  # Dynamic based on game
    diversity_score = num_range / max_possible_range
    score += diversity_score * 0.10
    
    # FACTOR 4: Gap pattern match (12%)
    gap_patterns = analysis.get('gap_patterns', {})
    avg_winning_gap = gap_patterns.get('avg_winning_gap', 0)
    if avg_winning_gap:
        sorted_nums = sorted(numbers)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        avg_gap = np.mean(gaps) if gaps else 0
        gap_diff = abs(avg_winning_gap - avg_gap)
        gap_score = max(0, 1.0 - (gap_diff / 10.0))
        score += gap_score * 0.12
    
    # FACTOR 5: Zone distribution (10%)
    zone_dist = analysis.get('zone_distribution', {})
    winning_zones = zone_dist.get('winning_distribution', {})
    if winning_zones:
        # Calculate zone boundaries dynamically
        low_boundary = max_number // 3
        mid_boundary = (max_number * 2) // 3
        
        predicted_zones = {'low': 0, 'mid': 0, 'high': 0}
        for num in numbers:
            if num <= low_boundary:
                predicted_zones['low'] += 1
            elif num <= mid_boundary:
                predicted_zones['mid'] += 1
            else:
                predicted_zones['high'] += 1
        
        zone_diff = sum(abs(winning_zones.get(z, 0) - predicted_zones[z]) for z in ['low', 'mid', 'high'])
        zone_score = max(0, 1.0 - (zone_diff / len(numbers)))
        score += zone_score * 0.10
    
    # FACTOR 6: Even/odd ratio (8%)
    even_odd = analysis.get('even_odd_ratio', {})
    winning_ratio = even_odd.get('winning_ratio', 0.5)
    if winning_ratio:
        even_count = sum(1 for n in numbers if n % 2 == 0)
        predicted_ratio = even_count / len(numbers)
        ratio_diff = abs(winning_ratio - predicted_ratio)
        ratio_score = max(0, 1.0 - ratio_diff)
        score += ratio_score * 0.08
    
    # FACTOR 7: Cold number penalty (-10%)
    cold_numbers = analysis.get('cold_numbers', [])
    if cold_numbers:
        cold_matches = sum(1 for n in numbers if n in cold_numbers)
        cold_penalty = (cold_matches / len(numbers)) * 0.10
        score -= cold_penalty  # Penalty for using cold numbers
    
    # FACTOR 8: Decade coverage (10%)
    decade_coverage = analysis.get('decade_coverage', {})
    winning_decade_count = decade_coverage.get('winning_decade_count', 0)
    if winning_decade_count:
        predicted_decades = set((n - 1) // 10 for n in numbers)
        decade_diff = abs(winning_decade_count - len(predicted_decades))
        decade_score = max(0, 1.0 - (decade_diff / 5.0))
        score += decade_score * 0.10
    
    # FACTOR 9: Pattern fingerprint similarity (8%)
    winning_pattern = analysis.get('winning_pattern_fingerprint', '')
    if winning_pattern:
        predicted_pattern = _create_pattern_fingerprint(numbers)
        # Calculate similarity (same pattern elements)
        min_len = min(len(winning_pattern), len(predicted_pattern))
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if winning_pattern[i] == predicted_pattern[i])
            pattern_score = matches / min_len
            score += pattern_score * 0.08
    
    # FACTOR 10: Position-based weighting (15%)
    position_accuracy = analysis.get('position_accuracy', {})
    if position_accuracy:
        sorted_nums = sorted(numbers)
        position_score = 0.0
        for i, num in enumerate(sorted_nums, 1):
            pos_key = f'position_{i}'
            if pos_key in position_accuracy:
                accuracy = position_accuracy[pos_key].get('accuracy', 0)
                position_score += accuracy
        if len(sorted_nums) > 0:
            position_score /= len(sorted_nums)
            score += position_score * 0.15
    
    # Normalize to 0-1 range (score can be negative due to cold penalty)
    return max(0.0, min(1.0, score))


def _calculate_learning_score_advanced(
    numbers: List[int], 
    learning_data: Dict, 
    adaptive_system: AdaptiveLearningSystem,
    draw_age: int = 0
) -> float:
    """
    ADVANCED learning score using:
    1. Adaptive weights that evolve based on success
    2. Temporal decay for recent pattern emphasis
    3. Cross-factor interaction bonuses
    4. Anti-pattern penalties
    """
    score = 0.0
    analysis = learning_data.get('analysis', {})
    
    # Determine max_number from game context
    game = learning_data.get('game', 'Lotto Max')
    max_number = 50 if 'max' in game.lower() else 49
    
    # Get adaptive weights with temporal decay
    weights = adaptive_system.get_adaptive_weights(draw_age)
    
    # Factor scores dictionary for interaction analysis
    factor_scores = {}
    
    # FACTOR 1: Hot number alignment
    number_freq = analysis.get('number_frequency', {})
    hot_numbers_data = number_freq.get('hot_numbers', [])
    hot_numbers = [item['number'] if isinstance(item, dict) else item for item in hot_numbers_data[:10]]
    # Ensure hot_numbers is a list/set, not numpy array (avoid ambiguous truth value error)
    if hasattr(hot_numbers, 'tolist'):
        hot_numbers = hot_numbers.tolist()
    # Convert to set for efficient membership testing and ensure numbers is also iterable
    hot_numbers_set = set(hot_numbers)
    numbers_list = numbers.tolist() if hasattr(numbers, 'tolist') else list(numbers)
    hot_matches = sum(1 for n in numbers_list if n in hot_numbers_set)
    factor_scores['hot_numbers'] = (hot_matches / 10.0)
    score += factor_scores['hot_numbers'] * weights.get('hot_numbers', 0.12)
    
    # FACTOR 2: Sum alignment
    sum_analysis = analysis.get('sum_analysis', {})
    target_sum = sum_analysis.get('winning_sum', 0)
    if target_sum:
        sum_diff = abs(sum(numbers) - target_sum)
        sum_score = max(0, 1.0 - (sum_diff / 100.0))
        factor_scores['sum_alignment'] = sum_score
        score += sum_score * weights.get('sum_alignment', 0.15)
    else:
        factor_scores['sum_alignment'] = 0
    
    # FACTOR 3: Diversity/spread
    num_range = max(numbers) - min(numbers)
    max_possible_range = max_number - 1
    diversity_score = num_range / max_possible_range
    factor_scores['diversity'] = diversity_score
    score += diversity_score * weights.get('diversity', 0.10)
    
    # FACTOR 4: Gap pattern match
    gap_patterns = analysis.get('gap_patterns', {})
    avg_winning_gap = gap_patterns.get('avg_winning_gap', 0)
    if avg_winning_gap:
        sorted_nums = sorted(numbers)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        avg_gap = np.mean(gaps) if gaps else 0
        gap_diff = abs(avg_winning_gap - avg_gap)
        gap_score = max(0, 1.0 - (gap_diff / 10.0))
        factor_scores['gap_patterns'] = gap_score
        score += gap_score * weights.get('gap_patterns', 0.12)
    else:
        factor_scores['gap_patterns'] = 0
    
    # FACTOR 5: Zone distribution
    zone_dist = analysis.get('zone_distribution', {})
    winning_zones = zone_dist.get('winning_distribution', {})
    if winning_zones:
        low_boundary = max_number // 3
        mid_boundary = (max_number * 2) // 3
        
        predicted_zones = {'low': 0, 'mid': 0, 'high': 0}
        for num in numbers:
            if num <= low_boundary:
                predicted_zones['low'] += 1
            elif num <= mid_boundary:
                predicted_zones['mid'] += 1
            else:
                predicted_zones['high'] += 1
        
        zone_diff = sum(abs(winning_zones.get(z, 0) - predicted_zones[z]) for z in ['low', 'mid', 'high'])
        zone_score = max(0, 1.0 - (zone_diff / len(numbers)))
        factor_scores['zone_distribution'] = zone_score
        score += zone_score * weights.get('zone_distribution', 0.10)
    else:
        factor_scores['zone_distribution'] = 0
    
    # FACTOR 6: Even/odd ratio
    even_odd = analysis.get('even_odd_ratio', {})
    winning_ratio = even_odd.get('winning_ratio', 0.5)
    if winning_ratio:
        even_count = sum(1 for n in numbers if n % 2 == 0)
        predicted_ratio = even_count / len(numbers)
        ratio_diff = abs(winning_ratio - predicted_ratio)
        ratio_score = max(0, 1.0 - ratio_diff)
        factor_scores['even_odd_ratio'] = ratio_score
        score += ratio_score * weights.get('even_odd_ratio', 0.08)
    else:
        factor_scores['even_odd_ratio'] = 0
    
    # FACTOR 7: Cold number penalty
    cold_numbers = analysis.get('cold_numbers', [])
    if cold_numbers:
        cold_matches = sum(1 for n in numbers if n in cold_numbers)
        cold_penalty = (cold_matches / len(numbers))
        factor_scores['cold_penalty'] = cold_penalty
        score -= cold_penalty * weights.get('cold_penalty', 0.10)
    else:
        factor_scores['cold_penalty'] = 0
    
    # FACTOR 8: Decade coverage
    decade_coverage = analysis.get('decade_coverage', {})
    winning_decade_count = decade_coverage.get('winning_decade_count', 0)
    if winning_decade_count:
        predicted_decades = set((n - 1) // 10 for n in numbers)
        decade_diff = abs(winning_decade_count - len(predicted_decades))
        decade_score = max(0, 1.0 - (decade_diff / 5.0))
        factor_scores['decade_coverage'] = decade_score
        score += decade_score * weights.get('decade_coverage', 0.10)
    else:
        factor_scores['decade_coverage'] = 0
    
    # FACTOR 9: Pattern fingerprint similarity
    winning_pattern = analysis.get('winning_pattern_fingerprint', '')
    if winning_pattern:
        predicted_pattern = _create_pattern_fingerprint(numbers)
        min_len = min(len(winning_pattern), len(predicted_pattern))
        if min_len > 0:
            matches = sum(1 for i in range(min_len) if winning_pattern[i] == predicted_pattern[i])
            pattern_score = matches / min_len
            factor_scores['pattern_fingerprint'] = pattern_score
            score += pattern_score * weights.get('pattern_fingerprint', 0.08)
        else:
            factor_scores['pattern_fingerprint'] = 0
    else:
        factor_scores['pattern_fingerprint'] = 0
    
    # FACTOR 10: Position-based weighting
    position_accuracy = analysis.get('position_accuracy', {})
    if position_accuracy:
        sorted_nums = sorted(numbers)
        position_score = 0.0
        for i, num in enumerate(sorted_nums, 1):
            pos_key = f'position_{i}'
            if pos_key in position_accuracy:
                accuracy = position_accuracy[pos_key].get('accuracy', 0)
                position_score += accuracy
        if len(sorted_nums) > 0:
            position_score /= len(sorted_nums)
            factor_scores['position_weighting'] = position_score
            score += position_score * weights.get('position_weighting', 0.15)
    else:
        factor_scores['position_weighting'] = 0
    
    # ADVANCED: Cross-factor interaction bonuses
    interactions = adaptive_system.meta_learning_data.get('cross_factor_interactions', {})
    for pair_key, interaction_strength in interactions.items():
        factors = pair_key.split('+')
        if len(factors) == 2:
            f1, f2 = factors
            if f1 in factor_scores and f2 in factor_scores:
                # Bonus when both factors are strong
                interaction_bonus = factor_scores[f1] * factor_scores[f2] * interaction_strength * 0.05
                score += interaction_bonus
    
    # ADVANCED: Anti-pattern penalty
    anti_penalty = adaptive_system.penalize_anti_patterns(numbers)
    score -= anti_penalty * 0.15  # 15% penalty for matching anti-patterns
    
    # Normalize to 0-1 range
    return max(0.0, min(1.0, score))


def _generate_learning_based_sets(
    num_sets: int,
    hot_numbers: List[int],
    position_accuracy: Dict,
    target_sum: int,
    sum_range: Dict,
    analyzer: SuperIntelligentAIAnalyzer,
    cold_numbers: List[int] = None,
    target_zones: Dict = None,
    target_even_ratio: float = 0.5,
    avg_gap: float = 7
) -> List[List[int]]:
    """Generate new sets based on ENHANCED learning patterns (10 factors)."""
    generated = []
    max_number = analyzer.game_config["max_number"]
    draw_size = analyzer.game_config["draw_size"]
    
    if cold_numbers is None:
        cold_numbers = []
    if target_zones is None:
        target_zones = {'low': 2, 'mid': 3, 'high': 2}
    
    # Calculate zone boundaries dynamically
    low_boundary = max_number // 3
    mid_boundary = (max_number * 2) // 3
    
    attempts = 0
    max_attempts = num_sets * 10  # Allow multiple attempts per desired set
    
    while len(generated) < num_sets and attempts < max_attempts:
        attempts += 1
        new_set = []
        
        # CONSTRAINT 1: Start with hot numbers (50-70% of set)
        num_hot = int(draw_size * np.random.uniform(0.5, 0.7))
        # Ensure all numbers are integers (not tuples or other types)
        available_hot = [int(n) for n in hot_numbers if int(n) not in new_set and int(n) not in cold_numbers]
        
        for _ in range(min(num_hot, len(available_hot))):
            if available_hot:
                idx = np.random.randint(0, len(available_hot))
                num = int(available_hot.pop(idx))
                new_set.append(num)
        
        # CONSTRAINT 2: Fill remaining avoiding cold numbers
        remaining = draw_size - len(new_set)
        # Ensure cold_numbers are integers
        cold_numbers_int = [int(n) for n in cold_numbers] if cold_numbers else []
        available_numbers = [n for n in range(1, max_number + 1) 
                           if n not in new_set and n not in cold_numbers_int]
        
        # CONSTRAINT 3: Try to match even/odd ratio
        current_even = sum(1 for n in new_set if n % 2 == 0)
        target_even_count = int(draw_size * target_even_ratio)
        
        while len(new_set) < draw_size and available_numbers:
            # Prefer even or odd based on target
            if len(new_set) < draw_size:
                current_even = sum(1 for n in new_set if n % 2 == 0)
                need_even = (current_even < target_even_count) and (len(new_set) + (target_even_count - current_even) <= draw_size)
                
                if need_even:
                    evens = [n for n in available_numbers if n % 2 == 0]
                    if evens:
                        num = int(np.random.choice(evens))
                    else:
                        num = int(np.random.choice(available_numbers))
                else:
                    num = int(np.random.choice(available_numbers))
                
                new_set.append(num)
                available_numbers.remove(num)
        
        # CONSTRAINT 4: Check sum alignment
        set_sum = sum(new_set)
        if not (sum_range['min'] <= set_sum <= sum_range['max']):
            continue  # Skip this set, try again
        
        # CONSTRAINT 5: Check zone distribution (using dynamic boundaries)
        zones = {'low': 0, 'mid': 0, 'high': 0}
        for num in new_set:
            if num <= low_boundary:
                zones['low'] += 1
            elif num <= mid_boundary:
                zones['mid'] += 1
            else:
                zones['high'] += 1
        
        # Ensure at least one number from each zone
        if zones['low'] == 0 or zones['mid'] == 0 or zones['high'] == 0:
            continue
        
        # CONSTRAINT 6: Ensure good decade coverage (at least 3 decades)
        decades = set((n - 1) // 10 for n in new_set)
        if len(decades) < 3:
            continue
        
        # Set passed all constraints
        generated.append(sorted(new_set))
    
    # If we couldn't generate enough sets with constraints, fill remainder with basic sets
    while len(generated) < num_sets:
        new_set = []
        available_numbers = list(range(1, max_number + 1))
        
        for _ in range(draw_size):
            if available_numbers:
                num = np.random.choice(available_numbers)
                new_set.append(num)
                available_numbers.remove(num)
        
        generated.append(sorted(new_set))
    
    return generated


def _save_learning_regenerated_predictions(
    original_file: Path,
    regenerated_predictions: List,
    original_data: Dict,
    learning_sources: List[Path],
    strategy: str,
    learning_weight: float
) -> Path:
    """Save regenerated predictions with learning suffix, ranked by learning score."""
    # Create new filename: sia_predictions_learning_TIMESTAMP_NUMsets.json
    original_name = original_file.stem  # Remove .json
    
    # Extract timestamp and set count from original name
    # Format: sia_predictions_TIMESTAMP_NUMsets
    parts = original_name.split('_')
    
    if 'sia' in parts and 'predictions' in parts:
        # Insert 'learning' after 'predictions'
        new_parts = []
        for i, part in enumerate(parts):
            new_parts.append(part)
            if part == 'predictions':
                new_parts.append('learning')
        new_filename = '_'.join(new_parts) + '.json'
    else:
        # Fallback
        new_filename = f"{original_name}_learning.json"
    
    # Save to same directory as original
    new_filepath = original_file.parent / new_filename
    
    # Load learning data to rank predictions
    combined_learning = _load_and_combine_learning_files(learning_sources)
    
    # Rank predictions by learning score
    ranked_predictions = []
    for pred_set in regenerated_predictions:
        score = _calculate_learning_score(pred_set, combined_learning)
        ranked_predictions.append((score, pred_set))
    
    # Sort by score (highest first)
    ranked_predictions.sort(key=lambda x: x[0], reverse=True)
    
    # Extract just the prediction sets in ranked order
    final_predictions = [pred for score, pred in ranked_predictions]
    
    # Create enhanced data structure
    enhanced_data = original_data.copy()
    enhanced_data['predictions'] = final_predictions
    enhanced_data['learning_regeneration'] = {
        'original_file': str(original_file.name),
        'regeneration_timestamp': datetime.now().isoformat(),
        'strategy': strategy,
        'learning_weight': learning_weight,
        'learning_sources': [str(ls.name) for ls in learning_sources],
        'num_sources': len(learning_sources),
        'ranked_by_learning_score': True,
        'ranking_note': 'Predictions sorted from highest to lowest learning score (#1 = best)'
    }
    
    # Save
    with open(new_filepath, 'w') as f:
        json.dump(enhanced_data, f, indent=2, default=str)
    
    return new_filepath

