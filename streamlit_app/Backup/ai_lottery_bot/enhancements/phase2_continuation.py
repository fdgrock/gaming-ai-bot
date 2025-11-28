#!/usr/bin/env python3
"""
PHASE 2: CROSS-GAME LEARNING INTELLIGENCE - CONTINUATION
=======================================================

This module contains the remaining Phase 2 components:
- TemporalPatternAnalyzer 
- IntelligentModelSelector
- Helper methods and utilities

Author: AI Assistant
Date: 2025-01-02  
Phase: 2 - Cross-Game Learning Intelligence (Continuation)
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Helper methods
    def _calculate_frequency_distribution(self, numbers: List[int], max_num: int) -> Dict[int, float]:
        """Calculate frequency distribution of numbers"""
        freq = {}
        total = len(numbers)
        for i in range(1, max_num + 1):
            freq[i] = numbers.count(i) / total if total > 0 else 0
        return freq
    
    def _analyze_range_preferences(self, numbers: List[int]) -> Dict[str, float]:
        """Analyze range preferences in number selection"""
        if not numbers:
            return {}
        
        max_num = max(numbers) if numbers else 49
        
        # Define ranges
        low_range = list(range(1, max_num // 3 + 1))
        mid_range = list(range(max_num // 3 + 1, 2 * max_num // 3 + 1))
        high_range = list(range(2 * max_num // 3 + 1, max_num + 1))
        
        total = len(numbers)
        return {
            'low_preference': sum(1 for n in numbers if n in low_range) / total,
            'mid_preference': sum(1 for n in numbers if n in mid_range) / total,
            'high_preference': sum(1 for n in numbers if n in high_range) / total
        }
    
    def _analyze_day_of_week_effects(self, game: str, data: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
        """Analyze day of week effects on lottery results"""
        try:
            dow_data = {}
            
            # Analyze historical results by day of week
            for entry in data:
                if 'date' in entry and 'winning_numbers' in entry:
                    try:
                        date_obj = datetime.fromisoformat(entry['date'].replace('Z', '+00:00'))
                        dow = date_obj.strftime('%A')
                        
                        if dow not in dow_data:
                            dow_data[dow] = {'numbers': [], 'count': 0}
                        
                        dow_data[dow]['numbers'].extend(entry['winning_numbers'])
                        dow_data[dow]['count'] += 1
                    except Exception:
                        continue
            
            # Analyze patterns by day
            dow_analysis = {}
            for dow, data_info in dow_data.items():
                numbers = data_info['numbers']
                if numbers:
                    dow_analysis[dow] = {
                        'average_number': np.mean(numbers),
                        'std_deviation': np.std(numbers),
                        'odd_ratio': sum(1 for n in numbers if n % 2 == 1) / len(numbers),
                        'high_number_ratio': sum(1 for n in numbers if n > 25) / len(numbers),
                        'draw_count': data_info['count']
                    }
            
            return dow_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing day of week effects: {e}")
            return {}
    
    def _analyze_monthly_trends(self, game: str, data: List[Dict]) -> Dict[str, Any]:
        """Analyze monthly trends in lottery results"""
        try:
            monthly_data = {str(i): [] for i in range(1, 13)}
            
            for entry in data:
                if 'date' in entry and 'winning_numbers' in entry:
                    try:
                        date_obj = datetime.fromisoformat(entry['date'].replace('Z', '+00:00'))
                        month = str(date_obj.month)
                        monthly_data[month].extend(entry['winning_numbers'])
                    except Exception:
                        continue
            
            # Analyze trends for each month
            monthly_analysis = {}
            for month, numbers in monthly_data.items():
                if numbers:
                    monthly_analysis[month] = {
                        'average_number': np.mean(numbers),
                        'number_range_preference': self._analyze_range_preferences(numbers),
                        'frequency_distribution': self._calculate_frequency_distribution(numbers, 50 if game == 'lotto_max' else 49),
                        'sample_size': len(numbers)
                    }
            
            return monthly_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing monthly trends: {e}")
            return {}
    
    def _analyze_yearly_cycles(self, game: str, data: List[Dict]) -> Dict[str, Any]:
        """Analyze yearly cycles in lottery results"""
        try:
            yearly_data = {}
            
            for entry in data:
                if 'date' in entry and 'winning_numbers' in entry:
                    try:
                        date_obj = datetime.fromisoformat(entry['date'].replace('Z', '+00:00'))
                        year = str(date_obj.year)
                        
                        if year not in yearly_data:
                            yearly_data[year] = []
                        yearly_data[year].extend(entry['winning_numbers'])
                    except Exception:
                        continue
            
            # Analyze cycles across years
            yearly_analysis = {}
            for year, numbers in yearly_data.items():
                if numbers and len(numbers) >= 50:  # Ensure sufficient data
                    yearly_analysis[year] = {
                        'total_draws': len(numbers) // (7 if game == 'lotto_max' else 6),
                        'average_number': np.mean(numbers),
                        'trend_direction': self._calculate_yearly_trend(numbers),
                        'volatility': np.std(numbers),
                        'unique_numbers': len(set(numbers))
                    }
            
            return yearly_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing yearly cycles: {e}")
            return {}
    
    def _calculate_yearly_trend(self, numbers: List[int]) -> str:
        """Calculate yearly trend direction"""
        try:
            if len(numbers) < 10:
                return 'insufficient_data'
            
            # Split into first and second half
            mid_point = len(numbers) // 2
            first_half_avg = np.mean(numbers[:mid_point])
            second_half_avg = np.mean(numbers[mid_point:])
            
            difference = second_half_avg - first_half_avg
            
            if difference > 2:
                return 'increasing'
            elif difference < -2:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'
    
    def _get_current_season(self, date_obj: datetime) -> str:
        """Get current season"""
        month = date_obj.month
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        else:
            return 'winter'
    
    def _calculate_temporal_confidence(self, game: str, current_date: datetime) -> Dict[str, float]:
        """Calculate confidence scores for temporal predictions"""
        try:
            confidence = {
                'seasonal_confidence': 0.5,
                'cyclical_confidence': 0.5,
                'trend_confidence': 0.5,
                'overall_confidence': 0.5
            }
            
            # Seasonal confidence based on data availability
            if game in self.seasonal_analysis:
                current_season = self._get_current_season(current_date)
                if current_season in self.seasonal_analysis[game]:
                    seasonal_data = self.seasonal_analysis[game][current_season]
                    sample_size = seasonal_data.get('sample_size', 0)
                    confidence['seasonal_confidence'] = min(sample_size / 100, 1.0)
            
            # Cyclical confidence
            if game in self.cyclical_patterns:
                cyclical_data = self.cyclical_patterns[game]
                if 'weekly_cycles' in cyclical_data:
                    cycle_count = len(cyclical_data['weekly_cycles'])
                    confidence['cyclical_confidence'] = min(cycle_count / 7, 1.0)
            
            # Overall confidence
            confidence['overall_confidence'] = np.mean([
                confidence['seasonal_confidence'],
                confidence['cyclical_confidence'],
                confidence['trend_confidence']
            ])
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating temporal confidence: {e}")
            return {'overall_confidence': 0.5}
    
    def _generate_temporal_recommendations(self, game: str, forecast_results: Dict[str, Any]) -> List[str]:
        """Generate temporal-based recommendations"""
        try:
            recommendations = []
            
            # Seasonal recommendations
            seasonal_adjustments = forecast_results.get('seasonal_adjustments', {})
            if seasonal_adjustments:
                if seasonal_adjustments.get('odd_ratio', 0.5) > 0.55:
                    recommendations.append("Favor odd numbers in current season")
                elif seasonal_adjustments.get('odd_ratio', 0.5) < 0.45:
                    recommendations.append("Favor even numbers in current season")
                
                avg_number = seasonal_adjustments.get('average_number', 25)
                if avg_number > 28:
                    recommendations.append("Target higher number ranges")
                elif avg_number < 22:
                    recommendations.append("Focus on lower number ranges")
            
            # Confidence-based recommendations
            confidence = forecast_results.get('temporal_confidence', {})
            overall_confidence = confidence.get('overall_confidence', 0.5)
            
            if overall_confidence > 0.8:
                recommendations.append("High confidence in temporal patterns - use aggressive strategies")
            elif overall_confidence < 0.4:
                recommendations.append("Low temporal confidence - use conservative approaches")
            else:
                recommendations.append("Moderate confidence - balance temporal insights with other factors")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating temporal recommendations: {e}")
            return []

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
    
    def _get_game_characteristics(self, game: str) -> Dict[str, Any]:
        """Get characteristics for the specified game"""
        try:
            # Normalize game name
            normalized_game = game.lower().replace(' ', '_')
            
            if 'max' in normalized_game:
                return {
                    'number_range': (1, 50),
                    'selection_count': 7,
                    'complexity': 'high',
                    'preferred_models': ['transformer', 'lstm', 'xgboost']
                }
            elif '649' in normalized_game or '6_49' in normalized_game:
                return {
                    'number_range': (1, 49),
                    'selection_count': 6,
                    'complexity': 'medium',
                    'preferred_models': ['xgboost', 'lstm', 'transformer']
                }
            else:
                # Default fallback
                return {
                    'number_range': (1, 50),
                    'selection_count': 7,
                    'complexity': 'high',
                    'preferred_models': ['transformer', 'lstm', 'xgboost']
                }
        except Exception as e:
            logger.error(f"Error getting game characteristics: {e}")
            return {'complexity': 'medium', 'preferred_models': ['xgboost']}
    
    def _assess_context_suitability(self, model: str, context_analysis: Dict[str, Any]) -> float:
        """Assess how suitable a model is for the current context"""
        try:
            suitability_score = 0.5  # Base score
            
            # Model-specific context suitability
            if model == 'transformer':
                # Transformers good for complex patterns
                if context_analysis.get('complexity', 'medium') == 'high':
                    suitability_score += 0.2
                if context_analysis.get('data_volume', 'medium') == 'high':
                    suitability_score += 0.1
            
            elif model == 'lstm':
                # LSTMs good for temporal sequences
                if context_analysis.get('temporal_importance', 'medium') == 'high':
                    suitability_score += 0.2
                if context_analysis.get('sequence_patterns', False):
                    suitability_score += 0.1
            
            elif model == 'xgboost':
                # XGBoost good for feature-based predictions
                if context_analysis.get('feature_richness', 'medium') == 'high':
                    suitability_score += 0.2
                if context_analysis.get('structured_data', True):
                    suitability_score += 0.1
            
            # Clamp to valid range
            return max(0.0, min(1.0, suitability_score))
            
        except Exception as e:
            logger.error(f"Error assessing context suitability: {e}")
            return 0.5
    
    def _calculate_dynamic_weights(self, models: List[str], model_scores: Dict[str, float], 
                                 context_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic weights for model selection"""
        try:
            weights = {}
            total_score = 0.0
            
            # Calculate weighted scores
            for model in models:
                base_score = model_scores.get(model, 0.5)
                context_bonus = self._assess_context_suitability(model, context_analysis)
                
                # Combine scores
                combined_score = (base_score * 0.7) + (context_bonus * 0.3)
                weights[model] = combined_score
                total_score += combined_score
            
            # Normalize weights
            if total_score > 0:
                for model in weights:
                    weights[model] /= total_score
            else:
                # Equal weights fallback
                equal_weight = 1.0 / len(models) if models else 0.0
                weights = {model: equal_weight for model in models}
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {e}")
            return {model: 1.0/len(models) for model in models} if models else {}

# Global instances for Phase 2 components (continuation)
temporal_pattern_analyzer = TemporalPatternAnalyzer()
intelligent_model_selector = IntelligentModelSelector()

logger.info("Phase 2 Cross-Game Learning Intelligence (Continuation) initialized")
