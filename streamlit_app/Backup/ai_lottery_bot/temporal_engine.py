#!/usr/bin/env python3
"""
Temporal & Cyclical Intelligence Engine for Ultra-High Accuracy Lottery Prediction
This module implements sophisticated temporal analysis including:
- Seasonal pattern detection
- Cyclical trend analysis
- Time-based frequency modeling
- Calendar-based predictions
- Long-term and short-term pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import math
from datetime import datetime, timedelta, date
import calendar
from scipy import stats
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)

class SeasonalPatternDetector:
    """Detects seasonal patterns in lottery draws"""
    
    def __init__(self):
        self.seasonal_cache = {}
    
    def analyze_seasonal_patterns(self, historical_data: List[List[int]], 
                                draw_dates: List[datetime], 
                                game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Analyze seasonal patterns in lottery draws"""
        try:
            if len(historical_data) != len(draw_dates):
                logger.warning("Mismatch between historical data and draw dates")
                return self._empty_seasonal_result()
            
            # Determine game parameters
            max_number = 50 if 'max' in game_type.lower() else 49
            numbers_per_draw = 7 if 'max' in game_type.lower() else 6
            
            # Convert dates to pandas datetime for easier manipulation
            df = pd.DataFrame({
                'date': pd.to_datetime(draw_dates),
                'numbers': historical_data
            })
            
            # Extract temporal features
            df['month'] = df['date'].dt.month
            df['season'] = df['month'].apply(self._get_season)
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Analyze monthly patterns
            monthly_patterns = self._analyze_monthly_patterns(df, max_number)
            
            # Analyze seasonal patterns
            seasonal_patterns = self._analyze_seasonal_patterns(df, max_number)
            
            # Analyze day-of-week patterns
            dow_patterns = self._analyze_day_of_week_patterns(df, max_number)
            
            # Analyze quarterly patterns
            quarterly_patterns = self._analyze_quarterly_patterns(df, max_number)
            
            # Calculate overall temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(
                monthly_patterns, seasonal_patterns, dow_patterns
            )
            
            return {
                'monthly_patterns': monthly_patterns,
                'seasonal_patterns': seasonal_patterns,
                'day_of_week_patterns': dow_patterns,
                'quarterly_patterns': quarterly_patterns,
                'temporal_consistency': temporal_consistency,
                'confidence_score': float(np.mean([
                    monthly_patterns.get('pattern_strength', 0),
                    seasonal_patterns.get('pattern_strength', 0),
                    dow_patterns.get('pattern_strength', 0)
                ])),
                'game_type': game_type,
                'max_number': max_number,
                'numbers_per_draw': numbers_per_draw
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern analysis: {e}")
            return self._empty_seasonal_result()
    
    def _get_season(self, month: int) -> str:
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _analyze_monthly_patterns(self, df: pd.DataFrame, max_number: int) -> Dict[str, Any]:
        """Analyze monthly number patterns"""
        try:
            monthly_frequencies = defaultdict(lambda: defaultdict(int))
            monthly_stats = {}
            
            for month in range(1, 13):
                month_data = df[df['month'] == month]['numbers'].tolist()
                if month_data:
                    # Count number frequencies for this month
                    for draw in month_data:
                        for num in draw:
                            if 1 <= num <= max_number:
                                monthly_frequencies[month][num] += 1
                    
                    # Calculate statistics for this month
                    total_numbers = sum(monthly_frequencies[month].values())
                    monthly_stats[month] = {
                        'total_draws': len(month_data),
                        'total_numbers': total_numbers,
                        'avg_per_number': total_numbers / max_number if max_number > 0 else 0,
                        'most_frequent': max(monthly_frequencies[month].items(), 
                                          key=lambda x: x[1]) if monthly_frequencies[month] else (0, 0),
                        'least_frequent': min(monthly_frequencies[month].items(), 
                                           key=lambda x: x[1]) if monthly_frequencies[month] else (0, 0)
                    }
            
            # Calculate pattern strength
            pattern_strength = self._calculate_monthly_pattern_strength(monthly_frequencies)
            
            # Find strongest monthly trends
            strongest_months = self._find_strongest_monthly_trends(monthly_frequencies, monthly_stats)
            
            return {
                'frequencies': dict(monthly_frequencies),
                'statistics': monthly_stats,
                'pattern_strength': pattern_strength,
                'strongest_months': strongest_months,
                'month_rankings': self._rank_months_by_activity(monthly_stats)
            }
            
        except Exception as e:
            logger.error(f"Error in monthly pattern analysis: {e}")
            return {}
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame, max_number: int) -> Dict[str, Any]:
        """Analyze seasonal number patterns"""
        try:
            seasonal_frequencies = defaultdict(lambda: defaultdict(int))
            seasonal_stats = {}
            
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                season_data = df[df['season'] == season]['numbers'].tolist()
                if season_data:
                    # Count number frequencies for this season
                    for draw in season_data:
                        for num in draw:
                            if 1 <= num <= max_number:
                                seasonal_frequencies[season][num] += 1
                    
                    # Calculate statistics for this season
                    total_numbers = sum(seasonal_frequencies[season].values())
                    seasonal_stats[season] = {
                        'total_draws': len(season_data),
                        'total_numbers': total_numbers,
                        'avg_per_number': total_numbers / max_number if max_number > 0 else 0
                    }
            
            # Calculate seasonal balance
            seasonal_balance = self._calculate_seasonal_balance(seasonal_stats)
            
            # Find seasonal preferences
            seasonal_preferences = self._find_seasonal_number_preferences(seasonal_frequencies)
            
            pattern_strength = self._calculate_seasonal_pattern_strength(seasonal_frequencies)
            
            return {
                'frequencies': dict(seasonal_frequencies),
                'statistics': seasonal_stats,
                'balance_score': seasonal_balance,
                'preferences': seasonal_preferences,
                'pattern_strength': pattern_strength
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern analysis: {e}")
            return {}
    
    def _analyze_day_of_week_patterns(self, df: pd.DataFrame, max_number: int) -> Dict[str, Any]:
        """Analyze day-of-week patterns"""
        try:
            dow_frequencies = defaultdict(lambda: defaultdict(int))
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for dow in range(7):
                dow_data = df[df['day_of_week'] == dow]['numbers'].tolist()
                if dow_data:
                    for draw in dow_data:
                        for num in draw:
                            if 1 <= num <= max_number:
                                dow_frequencies[dow_names[dow]][num] += 1
            
            # Calculate day-of-week preferences
            dow_preferences = self._calculate_dow_preferences(dow_frequencies)
            pattern_strength = self._calculate_dow_pattern_strength(dow_frequencies)
            
            return {
                'frequencies': dict(dow_frequencies),
                'preferences': dow_preferences,
                'pattern_strength': pattern_strength,
                'most_active_day': max(dow_preferences.items(), key=lambda x: x[1])[0] if dow_preferences else 'Unknown'
            }
            
        except Exception as e:
            logger.error(f"Error in day-of-week pattern analysis: {e}")
            return {}
    
    def _analyze_quarterly_patterns(self, df: pd.DataFrame, max_number: int) -> Dict[str, Any]:
        """Analyze quarterly patterns"""
        try:
            quarterly_frequencies = defaultdict(lambda: defaultdict(int))
            quarterly_stats = {}
            
            for quarter in range(1, 5):
                quarter_data = df[df['quarter'] == quarter]['numbers'].tolist()
                if quarter_data:
                    for draw in quarter_data:
                        for num in draw:
                            if 1 <= num <= max_number:
                                quarterly_frequencies[f'Q{quarter}'][num] += 1
                    
                    total_numbers = sum(quarterly_frequencies[f'Q{quarter}'].values())
                    quarterly_stats[f'Q{quarter}'] = {
                        'total_draws': len(quarter_data),
                        'total_numbers': total_numbers,
                        'avg_per_number': total_numbers / max_number if max_number > 0 else 0
                    }
            
            pattern_strength = self._calculate_quarterly_pattern_strength(quarterly_frequencies)
            
            return {
                'frequencies': dict(quarterly_frequencies),
                'statistics': quarterly_stats,
                'pattern_strength': pattern_strength
            }
            
        except Exception as e:
            logger.error(f"Error in quarterly pattern analysis: {e}")
            return {}
    
    def _calculate_monthly_pattern_strength(self, monthly_frequencies: dict) -> float:
        """Calculate strength of monthly patterns"""
        try:
            if not monthly_frequencies:
                return 0.0
            
            # Calculate variance in monthly activity
            monthly_totals = []
            for month in range(1, 13):
                total = sum(monthly_frequencies.get(month, {}).values())
                monthly_totals.append(total)
            
            if len(monthly_totals) == 0:
                return 0.0
            
            mean_total = np.mean(monthly_totals)
            if mean_total == 0:
                return 0.0
            
            coefficient_of_variation = np.std(monthly_totals) / mean_total
            
            # Convert to pattern strength (higher variation = stronger pattern)
            return min(1.0, coefficient_of_variation * 2)
            
        except Exception as e:
            logger.error(f"Error calculating monthly pattern strength: {e}")
            return 0.0
    
    def _calculate_seasonal_pattern_strength(self, seasonal_frequencies: dict) -> float:
        """Calculate strength of seasonal patterns"""
        try:
            if not seasonal_frequencies:
                return 0.0
            
            seasonal_totals = []
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                total = sum(seasonal_frequencies.get(season, {}).values())
                seasonal_totals.append(total)
            
            if len(seasonal_totals) == 0:
                return 0.0
            
            mean_total = np.mean(seasonal_totals)
            if mean_total == 0:
                return 0.0
            
            coefficient_of_variation = np.std(seasonal_totals) / mean_total
            return min(1.0, coefficient_of_variation * 1.5)
            
        except Exception as e:
            logger.error(f"Error calculating seasonal pattern strength: {e}")
            return 0.0
    
    def _calculate_dow_pattern_strength(self, dow_frequencies: dict) -> float:
        """Calculate strength of day-of-week patterns"""
        try:
            if not dow_frequencies:
                return 0.0
            
            dow_totals = [sum(freq.values()) for freq in dow_frequencies.values()]
            
            if len(dow_totals) == 0:
                return 0.0
            
            mean_total = np.mean(dow_totals)
            if mean_total == 0:
                return 0.0
            
            coefficient_of_variation = np.std(dow_totals) / mean_total
            return min(1.0, coefficient_of_variation * 1.5)
            
        except Exception as e:
            logger.error(f"Error calculating day-of-week pattern strength: {e}")
            return 0.0
    
    def _calculate_quarterly_pattern_strength(self, quarterly_frequencies: dict) -> float:
        """Calculate strength of quarterly patterns"""
        try:
            if not quarterly_frequencies:
                return 0.0
            
            quarterly_totals = [sum(freq.values()) for freq in quarterly_frequencies.values()]
            
            if len(quarterly_totals) == 0:
                return 0.0
            
            mean_total = np.mean(quarterly_totals)
            if mean_total == 0:
                return 0.0
            
            coefficient_of_variation = np.std(quarterly_totals) / mean_total
            return min(1.0, coefficient_of_variation * 1.5)
            
        except Exception as e:
            logger.error(f"Error calculating quarterly pattern strength: {e}")
            return 0.0
    
    def _find_strongest_monthly_trends(self, monthly_frequencies: dict, monthly_stats: dict) -> List[Dict[str, Any]]:
        """Find months with strongest patterns"""
        try:
            month_strengths = []
            
            for month in range(1, 13):
                if month in monthly_stats:
                    stats = monthly_stats[month]
                    # Calculate relative activity
                    activity_score = stats.get('total_numbers', 0)
                    
                    month_strengths.append({
                        'month': month,
                        'month_name': calendar.month_name[month],
                        'activity_score': activity_score,
                        'total_draws': stats.get('total_draws', 0),
                        'avg_per_number': stats.get('avg_per_number', 0)
                    })
            
            # Sort by activity score
            return sorted(month_strengths, key=lambda x: x['activity_score'], reverse=True)[:6]
            
        except Exception as e:
            logger.error(f"Error finding strongest monthly trends: {e}")
            return []
    
    def _find_seasonal_number_preferences(self, seasonal_frequencies: dict) -> Dict[str, List[int]]:
        """Find numbers that are preferred in specific seasons"""
        try:
            preferences = {}
            
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                if season in seasonal_frequencies:
                    # Find top numbers for this season
                    season_freqs = seasonal_frequencies[season]
                    top_numbers = sorted(season_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
                    preferences[season] = [num for num, freq in top_numbers]
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error finding seasonal preferences: {e}")
            return {}
    
    def _calculate_seasonal_balance(self, seasonal_stats: dict) -> float:
        """Calculate how balanced the seasons are"""
        try:
            if not seasonal_stats:
                return 0.0
            
            totals = [stats.get('total_numbers', 0) for stats in seasonal_stats.values()]
            
            if len(totals) == 0:
                return 0.0
            
            mean_total = np.mean(totals)
            if mean_total == 0:
                return 1.0  # Perfect balance when no data
            
            # Balance score: closer to 1 means more balanced
            coefficient_of_variation = np.std(totals) / mean_total
            balance_score = 1.0 / (1.0 + coefficient_of_variation)
            
            return float(balance_score)
            
        except Exception as e:
            logger.error(f"Error calculating seasonal balance: {e}")
            return 0.0
    
    def _calculate_dow_preferences(self, dow_frequencies: dict) -> Dict[str, float]:
        """Calculate day-of-week activity preferences"""
        try:
            preferences = {}
            total_activity = sum(sum(freq.values()) for freq in dow_frequencies.values())
            
            if total_activity == 0:
                return preferences
            
            for day, freq_dict in dow_frequencies.items():
                day_activity = sum(freq_dict.values())
                preferences[day] = day_activity / total_activity
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error calculating day-of-week preferences: {e}")
            return {}
    
    def _rank_months_by_activity(self, monthly_stats: dict) -> List[Dict[str, Any]]:
        """Rank months by activity level"""
        try:
            rankings = []
            
            for month, stats in monthly_stats.items():
                rankings.append({
                    'month': month,
                    'month_name': calendar.month_name[month],
                    'activity_level': stats.get('total_numbers', 0),
                    'draw_count': stats.get('total_draws', 0)
                })
            
            return sorted(rankings, key=lambda x: x['activity_level'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error ranking months: {e}")
            return []
    
    def _calculate_temporal_consistency(self, monthly_patterns: dict, 
                                      seasonal_patterns: dict, dow_patterns: dict) -> float:
        """Calculate overall temporal consistency"""
        try:
            consistencies = []
            
            # Monthly consistency
            monthly_strength = monthly_patterns.get('pattern_strength', 0)
            consistencies.append(monthly_strength)
            
            # Seasonal consistency
            seasonal_strength = seasonal_patterns.get('pattern_strength', 0)
            consistencies.append(seasonal_strength)
            
            # Day-of-week consistency
            dow_strength = dow_patterns.get('pattern_strength', 0)
            consistencies.append(dow_strength)
            
            return float(np.mean(consistencies)) if consistencies else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating temporal consistency: {e}")
            return 0.0
    
    def _empty_seasonal_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'monthly_patterns': {},
            'seasonal_patterns': {},
            'day_of_week_patterns': {},
            'quarterly_patterns': {},
            'temporal_consistency': 0.0,
            'confidence_score': 0.0,
            'game_type': 'unknown',
            'max_number': 49,
            'numbers_per_draw': 6
        }


class CyclicalTrendAnalyzer:
    """Analyzes cyclical trends and periodicities in lottery data"""
    
    def __init__(self):
        self.trend_cache = {}
    
    def analyze_cyclical_trends(self, historical_data: List[List[int]], 
                              draw_dates: List[datetime],
                              game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Analyze cyclical trends in lottery data"""
        try:
            if len(historical_data) != len(draw_dates):
                logger.warning("Mismatch between historical data and draw dates")
                return self._empty_cyclical_result()
            
            # Determine game parameters
            max_number = 50 if 'max' in game_type.lower() else 49
            numbers_per_draw = 7 if 'max' in game_type.lower() else 6
            
            # Convert to time series
            time_series = self._create_time_series(historical_data, draw_dates, max_number)
            
            # Analyze frequencies using FFT
            frequency_analysis = self._analyze_frequencies(time_series)
            
            # Detect recurring patterns
            recurring_patterns = self._detect_recurring_patterns(historical_data, draw_dates)
            
            # Analyze long-term trends
            long_term_trends = self._analyze_long_term_trends(time_series, draw_dates)
            
            # Calculate cycle strength
            cycle_strength = self._calculate_cycle_strength(frequency_analysis, recurring_patterns)
            
            return {
                'frequency_analysis': frequency_analysis,
                'recurring_patterns': recurring_patterns,
                'long_term_trends': long_term_trends,
                'cycle_strength': cycle_strength,
                'confidence_score': float(cycle_strength),
                'game_type': game_type,
                'max_number': max_number,
                'numbers_per_draw': numbers_per_draw
            }
            
        except Exception as e:
            logger.error(f"Error in cyclical trend analysis: {e}")
            return self._empty_cyclical_result()
    
    def _create_time_series(self, historical_data: List[List[int]], 
                           draw_dates: List[datetime], max_number: int) -> Dict[int, List[float]]:
        """Create time series for each number"""
        try:
            time_series = {num: [] for num in range(1, max_number + 1)}
            
            for draw, draw_date in zip(historical_data, draw_dates):
                # Create binary presence indicator for each number
                for num in range(1, max_number + 1):
                    time_series[num].append(1.0 if num in draw else 0.0)
            
            return time_series
            
        except Exception as e:
            logger.error(f"Error creating time series: {e}")
            return {}
    
    def _analyze_frequencies(self, time_series: Dict[int, List[float]]) -> Dict[str, Any]:
        """Analyze frequency patterns using FFT"""
        try:
            if not time_series:
                return {}
            
            frequency_results = {}
            dominant_frequencies = []
            
            for num, series in time_series.items():
                if len(series) > 10:  # Need sufficient data for FFT
                    # Apply FFT
                    fft_result = fft(series)
                    freqs = fftfreq(len(series))
                    
                    # Find dominant frequencies (excluding DC component)
                    magnitudes = np.abs(fft_result[1:len(fft_result)//2])
                    frequencies = freqs[1:len(freqs)//2]
                    
                    if len(magnitudes) > 0:
                        # Find peak frequency
                        peak_idx = np.argmax(magnitudes)
                        peak_frequency = frequencies[peak_idx]
                        peak_magnitude = magnitudes[peak_idx]
                        
                        # Calculate period if frequency is significant
                        if abs(peak_frequency) > 1e-6:
                            period = 1.0 / abs(peak_frequency)
                        else:
                            period = float('inf')
                        
                        frequency_results[num] = {
                            'peak_frequency': float(peak_frequency),
                            'peak_magnitude': float(peak_magnitude),
                            'period': float(period) if period != float('inf') else None,
                            'magnitude_ratio': float(peak_magnitude / np.mean(magnitudes)) if np.mean(magnitudes) > 0 else 0
                        }
                        
                        if period != float('inf') and 2 <= period <= len(series) / 3:
                            dominant_frequencies.append({
                                'number': num,
                                'period': period,
                                'strength': peak_magnitude
                            })
            
            # Sort by strength
            dominant_frequencies.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'number_frequencies': frequency_results,
                'dominant_cycles': dominant_frequencies[:10],  # Top 10
                'average_cycle_strength': float(np.mean([f['strength'] for f in dominant_frequencies])) if dominant_frequencies else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in frequency analysis: {e}")
            return {}
    
    def _detect_recurring_patterns(self, historical_data: List[List[int]], 
                                 draw_dates: List[datetime]) -> Dict[str, Any]:
        """Detect recurring patterns in the data"""
        try:
            if len(historical_data) < 20:
                return {}
            
            # Look for recurring number combinations
            pattern_frequencies = Counter()
            
            # Analyze pairs and triplets
            for draw in historical_data:
                # Pairs
                for i in range(len(draw)):
                    for j in range(i + 1, len(draw)):
                        pair = tuple(sorted([draw[i], draw[j]]))
                        pattern_frequencies[('pair', pair)] += 1
                
                # Triplets (if enough numbers)
                if len(draw) >= 3:
                    for i in range(len(draw)):
                        for j in range(i + 1, len(draw)):
                            for k in range(j + 1, len(draw)):
                                triplet = tuple(sorted([draw[i], draw[j], draw[k]]))
                                pattern_frequencies[('triplet', triplet)] += 1
            
            # Find most frequent patterns
            most_frequent = pattern_frequencies.most_common(20)
            
            # Analyze pattern timing
            pattern_timing = self._analyze_pattern_timing(historical_data, draw_dates, most_frequent)
            
            return {
                'frequent_patterns': [{'type': pattern[0][0], 'numbers': pattern[0][1], 'frequency': pattern[1]} 
                                    for pattern in most_frequent],
                'pattern_timing': pattern_timing,
                'pattern_diversity': len(set(p[0][0] for p in most_frequent))
            }
            
        except Exception as e:
            logger.error(f"Error detecting recurring patterns: {e}")
            return {}
    
    def _analyze_pattern_timing(self, historical_data: List[List[int]], 
                              draw_dates: List[datetime], 
                              frequent_patterns: List[Tuple]) -> Dict[str, Any]:
        """Analyze timing of pattern occurrences"""
        try:
            timing_analysis = {}
            
            for pattern_info, frequency in frequent_patterns[:5]:  # Analyze top 5 patterns
                pattern_type, pattern_numbers = pattern_info
                
                # Find occurrences of this pattern
                occurrences = []
                for i, draw in enumerate(historical_data):
                    if pattern_type == 'pair':
                        if all(num in draw for num in pattern_numbers):
                            occurrences.append(i)
                    elif pattern_type == 'triplet':
                        if all(num in draw for num in pattern_numbers):
                            occurrences.append(i)
                
                if len(occurrences) >= 2:
                    # Calculate intervals between occurrences
                    intervals = [occurrences[i+1] - occurrences[i] for i in range(len(occurrences)-1)]
                    
                    timing_analysis[str(pattern_numbers)] = {
                        'total_occurrences': len(occurrences),
                        'average_interval': float(np.mean(intervals)) if intervals else 0,
                        'min_interval': int(min(intervals)) if intervals else 0,
                        'max_interval': int(max(intervals)) if intervals else 0,
                        'interval_variance': float(np.var(intervals)) if intervals else 0,
                        'last_occurrence': occurrences[-1] if occurrences else 0
                    }
            
            return timing_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing pattern timing: {e}")
            return {}
    
    def _analyze_long_term_trends(self, time_series: Dict[int, List[float]], 
                                draw_dates: List[datetime]) -> Dict[str, Any]:
        """Analyze long-term trends in number appearances"""
        try:
            if not time_series or not draw_dates:
                return {}
            
            trends = {}
            
            # Analyze each number's long-term trend
            for num, series in time_series.items():
                if len(series) >= 50:  # Need sufficient data for trend analysis
                    # Calculate moving average
                    window_size = min(20, len(series) // 4)
                    moving_avg = self._calculate_moving_average(series, window_size)
                    
                    # Calculate trend slope
                    x = np.arange(len(moving_avg))
                    if len(x) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, moving_avg)
                        
                        trends[num] = {
                            'slope': float(slope),
                            'r_squared': float(r_value ** 2),
                            'p_value': float(p_value),
                            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                            'trend_strength': float(abs(slope) * len(series)),
                            'recent_activity': float(np.mean(series[-10:])) if len(series) >= 10 else 0
                        }
            
            # Find strongest trends
            strongest_trends = sorted(trends.items(), 
                                    key=lambda x: x[1]['trend_strength'], 
                                    reverse=True)[:10]
            
            return {
                'number_trends': trends,
                'strongest_trends': strongest_trends,
                'overall_trend_strength': float(np.mean([t[1]['trend_strength'] for t in strongest_trends])) if strongest_trends else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in long-term trend analysis: {e}")
            return {}
    
    def _calculate_moving_average(self, series: List[float], window_size: int) -> List[float]:
        """Calculate moving average of a series"""
        try:
            if window_size <= 0 or window_size > len(series):
                return series
            
            moving_avg = []
            for i in range(len(series) - window_size + 1):
                avg = np.mean(series[i:i + window_size])
                moving_avg.append(avg)
            
            return moving_avg
            
        except Exception as e:
            logger.error(f"Error calculating moving average: {e}")
            return series
    
    def _calculate_cycle_strength(self, frequency_analysis: dict, recurring_patterns: dict) -> float:
        """Calculate overall cyclical pattern strength"""
        try:
            strengths = []
            
            # Frequency analysis strength
            if frequency_analysis and 'average_cycle_strength' in frequency_analysis:
                strengths.append(frequency_analysis['average_cycle_strength'])
            
            # Pattern recurrence strength
            if recurring_patterns and 'frequent_patterns' in recurring_patterns:
                pattern_count = len(recurring_patterns['frequent_patterns'])
                pattern_strength = min(1.0, pattern_count / 20)  # Normalize to 0-1
                strengths.append(pattern_strength)
            
            return float(np.mean(strengths)) if strengths else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating cycle strength: {e}")
            return 0.0
    
    def _empty_cyclical_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'frequency_analysis': {},
            'recurring_patterns': {},
            'long_term_trends': {},
            'cycle_strength': 0.0,
            'confidence_score': 0.0,
            'game_type': 'unknown',
            'max_number': 49,
            'numbers_per_draw': 6
        }


class TemporalPredictionEngine:
    """Engine for making temporal-based predictions"""
    
    def __init__(self):
        self.seasonal_detector = SeasonalPatternDetector()
        self.cyclical_analyzer = CyclicalTrendAnalyzer()
    
    def generate_temporal_predictions(self, historical_data: List[List[int]], 
                                    draw_dates: List[datetime],
                                    next_draw_date: datetime,
                                    game_type: str = 'lotto_649',
                                    num_sets: int = 3) -> Dict[str, Any]:
        """Generate predictions based on temporal analysis"""
        try:
            # Determine game parameters
            max_number = 50 if 'max' in game_type.lower() else 49
            numbers_per_draw = 7 if 'max' in game_type.lower() else 6
            
            # Perform temporal analysis
            seasonal_analysis = self.seasonal_detector.analyze_seasonal_patterns(
                historical_data, draw_dates, game_type
            )
            
            cyclical_analysis = self.cyclical_analyzer.analyze_cyclical_trends(
                historical_data, draw_dates, game_type
            )
            
            # Generate predictions based on temporal patterns
            temporal_sets = self._generate_temporal_based_sets(
                seasonal_analysis, cyclical_analysis, next_draw_date,
                max_number, numbers_per_draw, num_sets
            )
            
            # Calculate confidence scores
            confidence_scores = self._calculate_temporal_confidence(
                temporal_sets, seasonal_analysis, cyclical_analysis
            )
            
            return {
                'temporal_sets': temporal_sets,
                'confidence_scores': confidence_scores,
                'seasonal_analysis': seasonal_analysis,
                'cyclical_analysis': cyclical_analysis,
                'next_draw_context': self._get_next_draw_context(next_draw_date),
                'overall_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                'game_type': game_type,
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating temporal predictions: {e}")
            return {
                'temporal_sets': [],
                'confidence_scores': [],
                'seasonal_analysis': {},
                'cyclical_analysis': {},
                'overall_confidence': 0.0,
                'error': str(e)
            }
    
    def _generate_temporal_based_sets(self, seasonal_analysis: dict, cyclical_analysis: dict,
                                    next_draw_date: datetime, max_number: int, 
                                    numbers_per_draw: int, num_sets: int) -> List[Dict[str, Any]]:
        """Generate number sets based on temporal patterns"""
        try:
            temporal_sets = []
            
            # Get temporal context for next draw
            next_month = next_draw_date.month
            next_season = self.seasonal_detector._get_season(next_month)
            next_quarter = f"Q{((next_month - 1) // 3) + 1}"
            next_dow = next_draw_date.strftime('%A')
            
            for set_idx in range(num_sets):
                # Different strategy for each set
                if set_idx == 0:
                    # Season-optimized set
                    number_set = self._generate_seasonal_optimized_set(
                        seasonal_analysis, next_season, max_number, numbers_per_draw
                    )
                elif set_idx == 1:
                    # Monthly pattern set
                    number_set = self._generate_monthly_optimized_set(
                        seasonal_analysis, next_month, max_number, numbers_per_draw
                    )
                elif set_idx == 2:
                    # Cyclical trend set
                    number_set = self._generate_cyclical_optimized_set(
                        cyclical_analysis, max_number, numbers_per_draw
                    )
                else:
                    # Balanced temporal set
                    number_set = self._generate_balanced_temporal_set(
                        seasonal_analysis, cyclical_analysis, 
                        next_season, next_month, max_number, numbers_per_draw
                    )
                
                temporal_sets.append({
                    'set_id': set_idx + 1,
                    'numbers': sorted(number_set),
                    'strategy': ['seasonal', 'monthly', 'cyclical', 'balanced'][min(set_idx, 3)],
                    'temporal_context': {
                        'target_season': next_season,
                        'target_month': next_month,
                        'target_quarter': next_quarter,
                        'target_day': next_dow
                    }
                })
            
            return temporal_sets
            
        except Exception as e:
            logger.error(f"Error generating temporal sets: {e}")
            return []
    
    def _generate_seasonal_optimized_set(self, seasonal_analysis: dict, target_season: str,
                                       max_number: int, numbers_per_draw: int) -> List[int]:
        """Generate set optimized for specific season"""
        try:
            number_set = set()
            
            # Get seasonal preferences
            seasonal_patterns = seasonal_analysis.get('seasonal_patterns', {})
            preferences = seasonal_patterns.get('preferences', {})
            
            if target_season in preferences:
                # Use preferred numbers for this season
                preferred_numbers = preferences[target_season]
                
                # Add top seasonal numbers
                for num in preferred_numbers[:min(numbers_per_draw, len(preferred_numbers))]:
                    number_set.add(num)
            
            # Fill remaining slots with balanced selection
            while len(number_set) < numbers_per_draw:
                # Add numbers that complement the set
                candidate = np.random.randint(1, max_number + 1)
                if candidate not in number_set:
                    number_set.add(candidate)
            
            return list(number_set)[:numbers_per_draw]
            
        except Exception as e:
            logger.error(f"Error generating seasonal set: {e}")
            return list(range(1, numbers_per_draw + 1))
    
    def _generate_monthly_optimized_set(self, seasonal_analysis: dict, target_month: int,
                                      max_number: int, numbers_per_draw: int) -> List[int]:
        """Generate set optimized for specific month"""
        try:
            number_set = set()
            
            # Get monthly patterns
            monthly_patterns = seasonal_analysis.get('monthly_patterns', {})
            frequencies = monthly_patterns.get('frequencies', {})
            
            if target_month in frequencies:
                # Sort numbers by frequency for this month
                month_freqs = frequencies[target_month]
                sorted_numbers = sorted(month_freqs.items(), key=lambda x: x[1], reverse=True)
                
                # Add top numbers for this month
                for num, freq in sorted_numbers[:min(numbers_per_draw, len(sorted_numbers))]:
                    if 1 <= num <= max_number:
                        number_set.add(num)
            
            # Fill remaining slots
            while len(number_set) < numbers_per_draw:
                candidate = np.random.randint(1, max_number + 1)
                if candidate not in number_set:
                    number_set.add(candidate)
            
            return list(number_set)[:numbers_per_draw]
            
        except Exception as e:
            logger.error(f"Error generating monthly set: {e}")
            return list(range(1, numbers_per_draw + 1))
    
    def _generate_cyclical_optimized_set(self, cyclical_analysis: dict, 
                                       max_number: int, numbers_per_draw: int) -> List[int]:
        """Generate set based on cyclical patterns"""
        try:
            number_set = set()
            
            # Get dominant cycles
            frequency_analysis = cyclical_analysis.get('frequency_analysis', {})
            dominant_cycles = frequency_analysis.get('dominant_cycles', [])
            
            # Add numbers with strong cyclical patterns
            for cycle_info in dominant_cycles[:min(numbers_per_draw, len(dominant_cycles))]:
                num = cycle_info.get('number')
                if num and 1 <= num <= max_number:
                    number_set.add(num)
            
            # Fill remaining slots with trend-based numbers
            long_term_trends = cyclical_analysis.get('long_term_trends', {})
            strongest_trends = long_term_trends.get('strongest_trends', [])
            
            for num, trend_info in strongest_trends:
                if len(number_set) >= numbers_per_draw:
                    break
                if 1 <= num <= max_number and trend_info.get('trend_direction') == 'increasing':
                    number_set.add(num)
            
            # Fill any remaining slots
            while len(number_set) < numbers_per_draw:
                candidate = np.random.randint(1, max_number + 1)
                if candidate not in number_set:
                    number_set.add(candidate)
            
            return list(number_set)[:numbers_per_draw]
            
        except Exception as e:
            logger.error(f"Error generating cyclical set: {e}")
            return list(range(1, numbers_per_draw + 1))
    
    def _generate_balanced_temporal_set(self, seasonal_analysis: dict, cyclical_analysis: dict,
                                      target_season: str, target_month: int,
                                      max_number: int, numbers_per_draw: int) -> List[int]:
        """Generate balanced set using all temporal insights"""
        try:
            number_set = set()
            
            # Combine insights from different temporal analyses
            
            # 1. Add seasonal favorites (25% of set)
            seasonal_count = max(1, numbers_per_draw // 4)
            seasonal_prefs = seasonal_analysis.get('seasonal_patterns', {}).get('preferences', {})
            if target_season in seasonal_prefs:
                for num in seasonal_prefs[target_season][:seasonal_count]:
                    if 1 <= num <= max_number:
                        number_set.add(num)
            
            # 2. Add monthly favorites (25% of set)
            monthly_count = max(1, numbers_per_draw // 4)
            monthly_freqs = seasonal_analysis.get('monthly_patterns', {}).get('frequencies', {})
            if target_month in monthly_freqs:
                month_nums = sorted(monthly_freqs[target_month].items(), 
                                  key=lambda x: x[1], reverse=True)
                for num, freq in month_nums[:monthly_count]:
                    if len(number_set) < numbers_per_draw and 1 <= num <= max_number:
                        number_set.add(num)
            
            # 3. Add cyclical trends (25% of set)
            cyclical_count = max(1, numbers_per_draw // 4)
            dominant_cycles = cyclical_analysis.get('frequency_analysis', {}).get('dominant_cycles', [])
            for cycle_info in dominant_cycles[:cyclical_count]:
                num = cycle_info.get('number')
                if len(number_set) < numbers_per_draw and num and 1 <= num <= max_number:
                    number_set.add(num)
            
            # 4. Fill remaining with balanced selection
            while len(number_set) < numbers_per_draw:
                candidate = np.random.randint(1, max_number + 1)
                if candidate not in number_set:
                    number_set.add(candidate)
            
            return list(number_set)[:numbers_per_draw]
            
        except Exception as e:
            logger.error(f"Error generating balanced temporal set: {e}")
            return list(range(1, numbers_per_draw + 1))
    
    def _calculate_temporal_confidence(self, temporal_sets: List[Dict[str, Any]],
                                     seasonal_analysis: dict, cyclical_analysis: dict) -> List[float]:
        """Calculate confidence scores for temporal predictions"""
        try:
            confidence_scores = []
            
            # Base confidence from temporal pattern strength
            seasonal_confidence = seasonal_analysis.get('confidence_score', 0.0)
            cyclical_confidence = cyclical_analysis.get('confidence_score', 0.0)
            base_confidence = (seasonal_confidence + cyclical_confidence) / 2
            
            for set_info in temporal_sets:
                strategy = set_info.get('strategy', 'balanced')
                
                # Strategy-specific confidence adjustments
                if strategy == 'seasonal':
                    confidence = base_confidence * 1.1  # Slight boost for seasonal
                elif strategy == 'monthly':
                    confidence = base_confidence * 1.05  # Small boost for monthly
                elif strategy == 'cyclical':
                    confidence = cyclical_confidence * 1.15  # Boost for cyclical
                else:  # balanced
                    confidence = base_confidence
                
                # Ensure confidence is within valid range
                confidence = max(0.0, min(1.0, confidence))
                confidence_scores.append(confidence)
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error calculating temporal confidence: {e}")
            return [0.5] * len(temporal_sets)
    
    def _get_next_draw_context(self, next_draw_date: datetime) -> Dict[str, Any]:
        """Get temporal context for next draw date"""
        try:
            return {
                'date': next_draw_date.strftime('%Y-%m-%d'),
                'month': next_draw_date.month,
                'month_name': calendar.month_name[next_draw_date.month],
                'season': self.seasonal_detector._get_season(next_draw_date.month),
                'quarter': f"Q{((next_draw_date.month - 1) // 3) + 1}",
                'day_of_week': next_draw_date.strftime('%A'),
                'day_of_year': next_draw_date.timetuple().tm_yday,
                'week_of_year': next_draw_date.isocalendar()[1],
                'is_weekend': next_draw_date.weekday() >= 5,
                'is_month_start': next_draw_date.day <= 7,
                'is_month_end': next_draw_date.day >= 23
            }
            
        except Exception as e:
            logger.error(f"Error getting next draw context: {e}")
            return {}


class AdvancedTemporalEngine:
    """Main engine coordinating all temporal analysis components"""
    
    def __init__(self):
        self.seasonal_detector = SeasonalPatternDetector()
        self.cyclical_analyzer = CyclicalTrendAnalyzer()
        self.prediction_engine = TemporalPredictionEngine()
        self.analysis_cache = {}
    
    def analyze_temporal_patterns(self, historical_data: List[List[int]], 
                                draw_dates: List[datetime],
                                game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Perform comprehensive temporal pattern analysis"""
        try:
            # Create cache key
            cache_key = f"{game_type}_{len(historical_data)}_{hash(str(draw_dates[-5:]))}"
            
            if cache_key in self.analysis_cache:
                logger.info("Using cached temporal analysis")
                return self.analysis_cache[cache_key]
            
            logger.info("Performing comprehensive temporal pattern analysis...")
            
            # Seasonal pattern analysis
            logger.info("Analyzing seasonal patterns...")
            seasonal_analysis = self.seasonal_detector.analyze_seasonal_patterns(
                historical_data, draw_dates, game_type
            )
            
            # Cyclical trend analysis
            logger.info("Analyzing cyclical trends...")
            cyclical_analysis = self.cyclical_analyzer.analyze_cyclical_trends(
                historical_data, draw_dates, game_type
            )
            
            # Calculate overall temporal intelligence score
            temporal_scores = [
                seasonal_analysis.get('confidence_score', 0),
                cyclical_analysis.get('confidence_score', 0),
                seasonal_analysis.get('temporal_consistency', 0)
            ]
            
            overall_temporal_intelligence = np.mean([s for s in temporal_scores if s > 0])
            
            analysis_result = {
                'seasonal_analysis': seasonal_analysis,
                'cyclical_analysis': cyclical_analysis,
                'overall_temporal_intelligence': float(overall_temporal_intelligence),
                'temporal_confidence': float(overall_temporal_intelligence),
                'analysis_timestamp': datetime.now().isoformat(),
                'game_type': game_type,
                'data_points': len(historical_data)
            }
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis_result
            
            logger.info(f"Temporal analysis complete. Intelligence score: {overall_temporal_intelligence:.3f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
            return {
                'seasonal_analysis': {},
                'cyclical_analysis': {},
                'overall_temporal_intelligence': 0.0,
                'temporal_confidence': 0.0,
                'error': str(e)
            }
    
    def generate_temporal_predictions(self, historical_data: List[List[int]], 
                                    draw_dates: List[datetime],
                                    next_draw_date: datetime,
                                    game_type: str = 'lotto_649',
                                    num_sets: int = 3) -> Dict[str, Any]:
        """Generate predictions using temporal intelligence"""
        try:
            return self.prediction_engine.generate_temporal_predictions(
                historical_data, draw_dates, next_draw_date, game_type, num_sets
            )
            
        except Exception as e:
            logger.error(f"Error generating temporal predictions: {e}")
            return {
                'temporal_sets': [],
                'confidence_scores': [],
                'overall_confidence': 0.0,
                'error': str(e)
            }
    
    def get_temporal_insights(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable insights from temporal analysis"""
        try:
            insights = []
            
            # Seasonal insights
            seasonal_analysis = analysis_result.get('seasonal_analysis', {})
            if seasonal_analysis:
                seasonal_patterns = seasonal_analysis.get('seasonal_patterns', {})
                if seasonal_patterns:
                    balance_score = seasonal_patterns.get('balance_score', 0)
                    if balance_score > 0.8:
                        insights.append("🌈 Excellent seasonal balance detected")
                    elif balance_score > 0.6:
                        insights.append("🌿 Good seasonal distribution patterns")
                    else:
                        insights.append("🍂 Uneven seasonal activity detected")
                
                monthly_patterns = seasonal_analysis.get('monthly_patterns', {})
                if monthly_patterns:
                    strongest_months = monthly_patterns.get('strongest_months', [])
                    if strongest_months:
                        top_month = strongest_months[0]
                        insights.append(f"📅 Peak activity in {top_month.get('month_name', 'Unknown')}")
            
            # Cyclical insights
            cyclical_analysis = analysis_result.get('cyclical_analysis', {})
            if cyclical_analysis:
                cycle_strength = cyclical_analysis.get('cycle_strength', 0)
                if cycle_strength > 0.7:
                    insights.append("🔄 Strong cyclical patterns identified")
                elif cycle_strength > 0.4:
                    insights.append("🔄 Moderate cyclical trends detected")
                else:
                    insights.append("🔄 Weak cyclical patterns found")
                
                frequency_analysis = cyclical_analysis.get('frequency_analysis', {})
                if frequency_analysis:
                    dominant_cycles = frequency_analysis.get('dominant_cycles', [])
                    if dominant_cycles:
                        insights.append(f"⏰ {len(dominant_cycles)} dominant cycles identified")
            
            # Overall temporal intelligence insight
            temporal_intelligence = analysis_result.get('overall_temporal_intelligence', 0)
            if temporal_intelligence > 0.8:
                insights.append("🧠 Very high temporal intelligence")
            elif temporal_intelligence > 0.6:
                insights.append("🧠 Good temporal intelligence")
            elif temporal_intelligence > 0.4:
                insights.append("🧠 Moderate temporal patterns")
            else:
                insights.append("🧠 Limited temporal intelligence")
            
            return {
                'insights': insights,
                'intelligence_level': 'High' if temporal_intelligence > 0.7 else 'Medium' if temporal_intelligence > 0.4 else 'Low',
                'recommendation': self._get_temporal_recommendation(analysis_result)
            }
            
        except Exception as e:
            logger.error(f"Error generating temporal insights: {e}")
            return {
                'insights': ["⚠️ Error generating temporal insights"],
                'intelligence_level': 'Unknown',
                'recommendation': "Unable to generate temporal recommendation"
            }
    
    def _get_temporal_recommendation(self, analysis_result: Dict[str, Any]) -> str:
        """Generate recommendation based on temporal analysis"""
        try:
            temporal_intelligence = analysis_result.get('overall_temporal_intelligence', 0)
            seasonal_analysis = analysis_result.get('seasonal_analysis', {})
            
            if temporal_intelligence > 0.8:
                return "Temporal patterns are very strong. Use seasonal and cyclical strategies with high confidence."
            elif temporal_intelligence > 0.6:
                balance_score = seasonal_analysis.get('seasonal_patterns', {}).get('balance_score', 0)
                if balance_score > 0.7:
                    return "Good temporal foundation. Focus on seasonal optimization strategies."
                else:
                    return "Solid temporal patterns. Use cyclical trend analysis for optimization."
            elif temporal_intelligence > 0.4:
                return "Moderate temporal signals. Use temporal insights as supplementary guidance."
            else:
                return "Weak temporal patterns. Rely more on other prediction methods."
                
        except Exception:
            return "Temporal analysis inconclusive. Use balanced prediction approach."
