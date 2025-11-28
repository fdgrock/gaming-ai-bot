"""
Phase 4: Temporal Engine for lottery prediction system.

This engine focuses on time-series analysis and temporal patterns in
lottery data. It implements sequential pattern recognition, trend analysis,
and time-based prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, date
import logging
import time
from collections import defaultdict, Counter
import calendar
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from collections import defaultdict, Counter
import calendar
import math
import warnings

warnings.filterwarnings('ignore')
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
        """Analyze monthly patterns in lottery draws"""
        try:
            monthly_frequencies = defaultdict(lambda: defaultdict(int))
            monthly_stats = {}
            
            for month in range(1, 13):
                month_data = df[df['month'] == month]['numbers'].tolist()
                if month_data:
                    for draw in month_data:
                        for num in draw:
                            if 1 <= num <= max_number:
                                monthly_frequencies[month][num] += 1
                    
                    total_numbers = sum(monthly_frequencies[month].values())
                    monthly_stats[month] = {
                        'total_draws': len(month_data),
                        'total_numbers': total_numbers,
                        'avg_per_number': total_numbers / max_number if max_number > 0 else 0,
                        'month_name': calendar.month_name[month]
                    }
            
            pattern_strength = self._calculate_monthly_pattern_strength(monthly_frequencies)
            
            # Find strongest and weakest months
            month_totals = [(month, stats['total_numbers']) for month, stats in monthly_stats.items()]
            strongest_months = sorted(month_totals, key=lambda x: x[1], reverse=True)[:3]
            weakest_months = sorted(month_totals, key=lambda x: x[1])[:3]
            
            return {
                'frequencies': dict(monthly_frequencies),
                'statistics': monthly_stats,
                'pattern_strength': pattern_strength,
                'strongest_months': [{'month': m, 'count': c, 'month_name': calendar.month_name[m]} for m, c in strongest_months],
                'weakest_months': [{'month': m, 'count': c, 'month_name': calendar.month_name[m]} for m, c in weakest_months]
            }
            
        except Exception as e:
            logger.error(f"Error in monthly pattern analysis: {e}")
            return {}

    def _analyze_seasonal_patterns(self, df: pd.DataFrame, max_number: int) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        try:
            seasonal_frequencies = defaultdict(lambda: defaultdict(int))
            seasonal_stats = {}
            
            for season in ['Winter', 'Spring', 'Summer', 'Fall']:
                season_data = df[df['season'] == season]['numbers'].tolist()
                if season_data:
                    for draw in season_data:
                        for num in draw:
                            if 1 <= num <= max_number:
                                seasonal_frequencies[season][num] += 1
                    
                    total_numbers = sum(seasonal_frequencies[season].values())
                    seasonal_stats[season] = {
                        'total_draws': len(season_data),
                        'total_numbers': total_numbers,
                        'avg_per_number': total_numbers / max_number if max_number > 0 else 0
                    }
            
            pattern_strength = self._calculate_seasonal_pattern_strength(seasonal_frequencies)
            
            # Calculate balance score
            season_totals = [stats['total_numbers'] for stats in seasonal_stats.values()]
            balance_score = 1 - (np.std(season_totals) / np.mean(season_totals)) if season_totals and np.mean(season_totals) > 0 else 0
            balance_score = max(0, min(1, balance_score))
            
            return {
                'frequencies': dict(seasonal_frequencies),
                'statistics': seasonal_stats,
                'pattern_strength': pattern_strength,
                'balance_score': float(balance_score)
            }
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern analysis: {e}")
            return {}

    def _analyze_day_of_week_patterns(self, df: pd.DataFrame, max_number: int) -> Dict[str, Any]:
        """Analyze day-of-week patterns"""
        try:
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_frequencies = defaultdict(lambda: defaultdict(int))
            
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
            return min(1.0, coefficient_of_variation * 2)
            
        except Exception as e:
            logger.error(f"Error calculating seasonal pattern strength: {e}")
            return 0.0

    def _calculate_dow_preferences(self, dow_frequencies: dict) -> Dict[str, float]:
        """Calculate day-of-week preferences"""
        try:
            preferences = {}
            total_activity = 0
            
            for day, frequencies in dow_frequencies.items():
                day_total = sum(frequencies.values())
                preferences[day] = day_total
                total_activity += day_total
            
            # Normalize to percentages
            if total_activity > 0:
                for day in preferences:
                    preferences[day] = preferences[day] / total_activity
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error calculating DOW preferences: {e}")
            return {}

    def _calculate_dow_pattern_strength(self, dow_frequencies: dict) -> float:
        """Calculate strength of day-of-week patterns"""
        try:
            if not dow_frequencies:
                return 0.0
            
            daily_totals = []
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                total = sum(dow_frequencies.get(day, {}).values())
                daily_totals.append(total)
            
            if len(daily_totals) == 0:
                return 0.0
            
            mean_total = np.mean(daily_totals)
            if mean_total == 0:
                return 0.0
            
            coefficient_of_variation = np.std(daily_totals) / mean_total
            return min(1.0, coefficient_of_variation * 2)
            
        except Exception as e:
            logger.error(f"Error calculating DOW pattern strength: {e}")
            return 0.0

    def _calculate_quarterly_pattern_strength(self, quarterly_frequencies: dict) -> float:
        """Calculate strength of quarterly patterns"""
        try:
            if not quarterly_frequencies:
                return 0.0
            
            quarterly_totals = []
            for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                total = sum(quarterly_frequencies.get(quarter, {}).values())
                quarterly_totals.append(total)
            
            if len(quarterly_totals) == 0:
                return 0.0
            
            mean_total = np.mean(quarterly_totals)
            if mean_total == 0:
                return 0.0
            
            coefficient_of_variation = np.std(quarterly_totals) / mean_total
            return min(1.0, coefficient_of_variation * 2)
            
        except Exception as e:
            logger.error(f"Error calculating quarterly pattern strength: {e}")
            return 0.0

    def _calculate_temporal_consistency(self, monthly_patterns: dict, seasonal_patterns: dict, 
                                      dow_patterns: dict) -> Dict[str, float]:
        """Calculate overall temporal consistency"""
        try:
            consistency_scores = []
            
            # Monthly consistency
            if monthly_patterns and 'pattern_strength' in monthly_patterns:
                consistency_scores.append(monthly_patterns['pattern_strength'])
            
            # Seasonal consistency
            if seasonal_patterns and 'pattern_strength' in seasonal_patterns:
                consistency_scores.append(seasonal_patterns['pattern_strength'])
            
            # DOW consistency
            if dow_patterns and 'pattern_strength' in dow_patterns:
                consistency_scores.append(dow_patterns['pattern_strength'])
            
            overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
            
            return {
                'overall_consistency': float(overall_consistency),
                'monthly_consistency': monthly_patterns.get('pattern_strength', 0),
                'seasonal_consistency': seasonal_patterns.get('pattern_strength', 0),
                'dow_consistency': dow_patterns.get('pattern_strength', 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating temporal consistency: {e}")
            return {'overall_consistency': 0.0}

    def _empty_seasonal_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'monthly_patterns': {},
            'seasonal_patterns': {},
            'day_of_week_patterns': {},
            'quarterly_patterns': {},
            'temporal_consistency': {'overall_consistency': 0.0},
            'confidence_score': 0.0,
            'game_type': 'unknown',
            'max_number': 49,
            'numbers_per_draw': 6
        }


class CyclicalTrendAnalyzer:
    """Analyzes cyclical trends and patterns in lottery draws"""
    
    def __init__(self):
        self.cyclical_cache = {}

    def analyze_cyclical_patterns(self, historical_data: List[List[int]], 
                                draw_dates: List[datetime], 
                                game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Analyze cyclical patterns in lottery draws"""
        try:
            if len(historical_data) != len(draw_dates):
                logger.warning("Mismatch between historical data and draw dates")
                return self._empty_cyclical_result()
            
            # Determine game parameters
            max_number = 50 if 'max' in game_type.lower() else 49
            numbers_per_draw = 7 if 'max' in game_type.lower() else 6
            
            # Create time series for each number
            time_series = self._create_number_time_series(historical_data, draw_dates, max_number)
            
            # Analyze frequency patterns
            frequency_analysis = self._analyze_frequency_patterns(time_series, draw_dates)
            
            # Detect recurring patterns
            recurring_patterns = self._detect_recurring_patterns(time_series)
            
            # Analyze long-term trends
            long_term_trends = self._analyze_long_term_trends(time_series, draw_dates)
            
            # Calculate overall cycle strength
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
            logger.error(f"Error in cyclical pattern analysis: {e}")
            return self._empty_cyclical_result()

    def _create_number_time_series(self, historical_data: List[List[int]], 
                                 draw_dates: List[datetime], max_number: int) -> Dict[int, List[float]]:
        """Create time series for each number"""
        try:
            time_series = {num: [] for num in range(1, max_number + 1)}
            
            for draw, date in zip(historical_data, draw_dates):
                # Create binary indicators for each number
                draw_indicators = {num: 0 for num in range(1, max_number + 1)}
                for num in draw:
                    if 1 <= num <= max_number:
                        draw_indicators[num] = 1
                
                # Add to time series
                for num in range(1, max_number + 1):
                    time_series[num].append(draw_indicators[num])
            
            return time_series
            
        except Exception as e:
            logger.error(f"Error creating number time series: {e}")
            return {}

    def _analyze_frequency_patterns(self, time_series: Dict[int, List[float]], 
                                  draw_dates: List[datetime]) -> Dict[str, Any]:
        """Analyze frequency domain patterns using FFT"""
        try:
            if not time_series:
                return {}
            
            dominant_cycles = []
            cycle_strengths = []
            
            # Analyze each number's frequency spectrum
            for num, series in time_series.items():
                if len(series) >= 20:  # Need sufficient data for FFT
                    # Apply FFT
                    fft_values = fft(series)
                    frequencies = fftfreq(len(series))
                    
                    # Find dominant frequencies (excluding DC component)
                    magnitude = np.abs(fft_values[1:len(fft_values)//2])
                    freq_axis = frequencies[1:len(frequencies)//2]
                    
                    if len(magnitude) > 0:
                        # Find peaks in frequency spectrum
                        peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude) * 0.3)
                        
                        for peak_idx in peaks[:3]:  # Top 3 peaks
                            if peak_idx < len(freq_axis):
                                frequency = freq_axis[peak_idx]
                                if frequency > 0:  # Avoid zero frequency
                                    cycle_length = int(1 / frequency)
                                    if 2 <= cycle_length <= len(series) // 4:  # Valid cycle lengths
                                        dominant_cycles.append({
                                            'number': num,
                                            'cycle_length': cycle_length,
                                            'strength': magnitude[peak_idx]
                                        })
            
            # Calculate average cycle strength
            if dominant_cycles:
                cycle_strengths = [cycle['strength'] for cycle in dominant_cycles]
                avg_strength = np.mean(cycle_strengths)
                max_strength = np.max(cycle_strengths)
                
                # Normalize strength
                normalized_strength = min(1.0, avg_strength / (max_strength + 1e-6))
            else:
                normalized_strength = 0.0
            
            return {
                'dominant_cycles': dominant_cycles[:10],  # Top 10 cycles
                'average_cycle_strength': float(normalized_strength),
                'total_cycles_found': len(dominant_cycles)
            }
            
        except Exception as e:
            logger.error(f"Error in frequency pattern analysis: {e}")
            return {}

    def _detect_recurring_patterns(self, time_series: Dict[int, List[float]]) -> Dict[str, Any]:
        """Detect recurring patterns in number sequences"""
        try:
            if not time_series:
                return {}
            
            pattern_lengths = [3, 4, 5, 6, 7]  # Different pattern lengths to search
            frequent_patterns = []
            
            for pattern_length in pattern_lengths:
                patterns = defaultdict(int)
                
                # Search for patterns in each number's time series
                for num, series in time_series.items():
                    if len(series) >= pattern_length * 3:  # Need enough data
                        for i in range(len(series) - pattern_length + 1):
                            pattern = tuple(series[i:i + pattern_length])
                            patterns[pattern] += 1
                
                # Find most frequent patterns
                if patterns:
                    most_frequent = Counter(patterns).most_common(5)
                    for pattern, count in most_frequent:
                        if count >= 3:  # Pattern appears at least 3 times
                            frequent_patterns.append({
                                'pattern': list(pattern),
                                'length': pattern_length,
                                'frequency': count,
                                'strength': count / len(time_series)
                            })
            
            # Sort by strength
            frequent_patterns.sort(key=lambda x: x['strength'], reverse=True)
            
            return {
                'frequent_patterns': frequent_patterns[:15],  # Top 15 patterns
                'pattern_diversity': len(set(tuple(p['pattern']) for p in frequent_patterns))
            }
            
        except Exception as e:
            logger.error(f"Error detecting recurring patterns: {e}")
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


class UltraHighAccuracyTemporalEngine:
    """Main temporal engine coordinating all temporal analysis components"""
    
    def __init__(self):
        self.seasonal_detector = SeasonalPatternDetector()
        self.cyclical_analyzer = CyclicalTrendAnalyzer()
        self.analysis_history = []
    
    def perform_comprehensive_temporal_analysis(self, historical_data: List[List[int]], 
                                              draw_dates: List[datetime], 
                                              game_type: str = 'lotto_649') -> Dict[str, Any]:
        """Perform comprehensive temporal analysis"""
        try:
            logger.info("🕐 Performing comprehensive temporal analysis...")
            
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'data_points': len(historical_data),
                'date_range': {
                    'start': min(draw_dates).isoformat() if draw_dates else None,
                    'end': max(draw_dates).isoformat() if draw_dates else None
                },
                'game_type': game_type
            }
            
            # Seasonal pattern analysis
            logger.info("📅 Analyzing seasonal patterns...")
            seasonal_analysis = self.seasonal_detector.analyze_seasonal_patterns(
                historical_data, draw_dates, game_type
            )
            analysis_result['seasonal_analysis'] = seasonal_analysis
            
            # Cyclical trend analysis
            logger.info("🔄 Analyzing cyclical trends...")
            cyclical_analysis = self.cyclical_analyzer.analyze_cyclical_patterns(
                historical_data, draw_dates, game_type
            )
            analysis_result['cyclical_analysis'] = cyclical_analysis
            
            # Calculate overall temporal intelligence
            temporal_intelligence = self._calculate_overall_temporal_intelligence(
                seasonal_analysis, cyclical_analysis
            )
            analysis_result['overall_temporal_intelligence'] = temporal_intelligence
            
            # Store analysis
            self.analysis_history.append(analysis_result)
            
            logger.info(f"✅ Temporal analysis complete. Intelligence score: {temporal_intelligence:.3f}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in comprehensive temporal analysis: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall_temporal_intelligence': 0.0
            }

    def _calculate_overall_temporal_intelligence(self, seasonal_analysis: Dict[str, Any], 
                                               cyclical_analysis: Dict[str, Any]) -> float:
        """Calculate overall temporal intelligence score"""
        try:
            intelligence_factors = []
            
            # Seasonal intelligence
            seasonal_confidence = seasonal_analysis.get('confidence_score', 0)
            if seasonal_confidence > 0:
                intelligence_factors.append(seasonal_confidence)
            
            # Cyclical intelligence
            cyclical_confidence = cyclical_analysis.get('confidence_score', 0)
            if cyclical_confidence > 0:
                intelligence_factors.append(cyclical_confidence)
            
            # Temporal consistency
            temporal_consistency = seasonal_analysis.get('temporal_consistency', {}).get('overall_consistency', 0)
            if temporal_consistency > 0:
                intelligence_factors.append(temporal_consistency)
            
            # Calculate weighted average
            if intelligence_factors:
                overall_intelligence = np.mean(intelligence_factors)
                return float(max(0.0, min(1.0, overall_intelligence)))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating overall temporal intelligence: {e}")
            return 0.0

    def generate_temporal_predictions(self, analysis_result: Dict[str, Any], 
                                    num_sets: int = 5) -> Dict[str, Any]:
        """Generate temporal-based predictions"""
        try:
            logger.info(f"🎯 Generating {num_sets} temporal prediction sets...")
            
            # Extract analysis components
            seasonal_analysis = analysis_result.get('seasonal_analysis', {})
            cyclical_analysis = analysis_result.get('cyclical_analysis', {})
            
            max_number = seasonal_analysis.get('max_number', 49)
            numbers_per_draw = seasonal_analysis.get('numbers_per_draw', 6)
            
            temporal_sets = []
            confidence_scores = []
            
            for i in range(num_sets):
                # Generate set using temporal insights
                if i == 0:
                    # Seasonal-based set
                    number_set = self._generate_seasonal_set(seasonal_analysis, numbers_per_draw, max_number)
                    source = 'seasonal'
                elif i == 1:
                    # Cyclical-based set
                    number_set = self._generate_cyclical_set(cyclical_analysis, numbers_per_draw, max_number)
                    source = 'cyclical'
                else:
                    # Hybrid temporal set
                    number_set = self._generate_hybrid_temporal_set(
                        seasonal_analysis, cyclical_analysis, numbers_per_draw, max_number
                    )
                    source = 'hybrid'
                
                # Calculate confidence
                set_confidence = self._calculate_set_confidence(number_set, analysis_result)
                confidence_scores.append(set_confidence)
                
                temporal_sets.append({
                    'set_id': i + 1,
                    'numbers': sorted(number_set),
                    'source': source,
                    'confidence': set_confidence,
                    'temporal_intelligence': analysis_result.get('overall_temporal_intelligence', 0)
                })
            
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return {
                'temporal_sets': temporal_sets,
                'confidence_scores': confidence_scores,
                'overall_confidence': float(overall_confidence),
                'temporal_intelligence': analysis_result.get('overall_temporal_intelligence', 0)
            }
            
        except Exception as e:
            logger.error(f"Error generating temporal predictions: {e}")
            return {
                'temporal_sets': [],
                'confidence_scores': [],
                'overall_confidence': 0.0,
                'error': str(e)
            }

    def _generate_seasonal_set(self, seasonal_analysis: Dict[str, Any], 
                             numbers_per_draw: int, max_number: int) -> List[int]:
        """Generate a set based on seasonal patterns"""
        try:
            current_date = datetime.now()
            current_month = current_date.month
            current_season = self.seasonal_detector._get_season(current_month)
            
            number_set = []
            
            # Use current season's patterns
            seasonal_patterns = seasonal_analysis.get('seasonal_patterns', {})
            seasonal_frequencies = seasonal_patterns.get('frequencies', {})
            current_season_freq = seasonal_frequencies.get(current_season, {})
            
            if current_season_freq:
                # Select numbers based on seasonal frequency
                season_numbers = [(num, freq) for num, freq in current_season_freq.items() if freq > 0]
                season_numbers.sort(key=lambda x: x[1], reverse=True)
                
                # Take top seasonal numbers
                seasonal_count = min(numbers_per_draw // 2, len(season_numbers))
                for i in range(seasonal_count):
                    if len(number_set) < numbers_per_draw:
                        number_set.append(season_numbers[i][0])
            
            # Fill remaining with balanced selection
            while len(number_set) < numbers_per_draw:
                available = [n for n in range(1, max_number + 1) if n not in number_set]
                if available:
                    number_set.append(np.random.choice(available))
                else:
                    break
            
            return number_set[:numbers_per_draw]
            
        except Exception as e:
            logger.error(f"Error generating seasonal set: {e}")
            return list(np.random.choice(range(1, max_number + 1), size=numbers_per_draw, replace=False))

    def _generate_cyclical_set(self, cyclical_analysis: Dict[str, Any], 
                             numbers_per_draw: int, max_number: int) -> List[int]:
        """Generate a set based on cyclical patterns"""
        try:
            number_set = []
            
            # Use long-term trends
            long_term_trends = cyclical_analysis.get('long_term_trends', {})
            number_trends = long_term_trends.get('number_trends', {})
            
            if number_trends:
                # Select numbers with positive trends and recent activity
                trending_numbers = []
                for num, trend_data in number_trends.items():
                    if (trend_data.get('trend_direction') == 'increasing' and 
                        trend_data.get('recent_activity', 0) > 0.1):
                        trending_numbers.append((num, trend_data.get('trend_strength', 0)))
                
                trending_numbers.sort(key=lambda x: x[1], reverse=True)
                
                # Include top trending numbers
                trend_count = min(numbers_per_draw // 2, len(trending_numbers))
                for i in range(trend_count):
                    if len(number_set) < numbers_per_draw:
                        number_set.append(trending_numbers[i][0])
            
            # Fill remaining with pattern-based selection
            recurring_patterns = cyclical_analysis.get('recurring_patterns', {})
            frequent_patterns = recurring_patterns.get('frequent_patterns', [])
            
            if frequent_patterns and len(number_set) < numbers_per_draw:
                # Use pattern insights for additional numbers
                pattern_numbers = set()
                for pattern_data in frequent_patterns[:3]:  # Top 3 patterns
                    pattern = pattern_data.get('pattern', [])
                    # Use pattern positions to influence selection
                    for pos, value in enumerate(pattern):
                        if value > 0.5:  # Active position
                            candidate = (pos % max_number) + 1
                            if candidate not in number_set:
                                pattern_numbers.add(candidate)
                
                for num in pattern_numbers:
                    if len(number_set) < numbers_per_draw:
                        number_set.append(num)
            
            # Fill remaining randomly
            while len(number_set) < numbers_per_draw:
                available = [n for n in range(1, max_number + 1) if n not in number_set]
                if available:
                    number_set.append(np.random.choice(available))
                else:
                    break
            
            return number_set[:numbers_per_draw]
            
        except Exception as e:
            logger.error(f"Error generating cyclical set: {e}")
            return list(np.random.choice(range(1, max_number + 1), size=numbers_per_draw, replace=False))

    def _generate_hybrid_temporal_set(self, seasonal_analysis: Dict[str, Any], 
                                    cyclical_analysis: Dict[str, Any], 
                                    numbers_per_draw: int, max_number: int) -> List[int]:
        """Generate a hybrid set using both seasonal and cyclical insights"""
        try:
            number_set = []
            
            # Mix seasonal and cyclical approaches
            seasonal_contribution = numbers_per_draw // 3
            cyclical_contribution = numbers_per_draw // 3
            
            # Get seasonal numbers
            seasonal_set = self._generate_seasonal_set(seasonal_analysis, seasonal_contribution, max_number)
            number_set.extend(seasonal_set[:seasonal_contribution])
            
            # Get cyclical numbers (avoiding duplicates)
            cyclical_set = self._generate_cyclical_set(cyclical_analysis, cyclical_contribution, max_number)
            for num in cyclical_set:
                if num not in number_set and len(number_set) < seasonal_contribution + cyclical_contribution:
                    number_set.append(num)
            
            # Fill remaining randomly
            while len(number_set) < numbers_per_draw:
                available = [n for n in range(1, max_number + 1) if n not in number_set]
                if available:
                    number_set.append(np.random.choice(available))
                else:
                    break
            
            return number_set[:numbers_per_draw]
            
        except Exception as e:
            logger.error(f"Error generating hybrid temporal set: {e}")
            return list(np.random.choice(range(1, max_number + 1), size=numbers_per_draw, replace=False))

    def _calculate_set_confidence(self, number_set: List[int], analysis_result: Dict[str, Any]) -> float:
        """Calculate confidence score for a temporal prediction set"""
        try:
            confidence_factors = []
            
            # Overall temporal intelligence
            temporal_intelligence = analysis_result.get('overall_temporal_intelligence', 0)
            confidence_factors.append(temporal_intelligence)
            
            # Seasonal alignment
            seasonal_analysis = analysis_result.get('seasonal_analysis', {})
            if seasonal_analysis:
                seasonal_confidence = seasonal_analysis.get('confidence_score', 0)
                confidence_factors.append(seasonal_confidence)
            
            # Cyclical alignment
            cyclical_analysis = analysis_result.get('cyclical_analysis', {})
            if cyclical_analysis:
                cyclical_confidence = cyclical_analysis.get('confidence_score', 0)
                confidence_factors.append(cyclical_confidence)
            
            # Calculate average
            return float(np.mean(confidence_factors)) if confidence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating set confidence: {e}")
            return 0.0


class TemporalEngine:
    """
    Temporal Engine for lottery predictions.
    
    Implements time-series analysis methods:
    - Sequential pattern recognition
    - ARIMA modeling
    - LSTM neural networks
    - Seasonal decomposition
    - Fourier analysis
    - Trend detection
    - Cyclical pattern analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Temporal Engine.
        
        Args:
            config: Configuration dictionary containing engine parameters
        """
        self.config = config
        self.game_config = config.get('game_config', {})
        self.temporal_config = config.get('temporal_config', {})
        
        # Game parameters
        self.number_range = self.game_config.get('number_range', [1, 49])
        self.numbers_per_draw = self.game_config.get('numbers_per_draw', 6)
        self.has_bonus = self.game_config.get('has_bonus', False)
        
        # Temporal analysis parameters
        self.sequence_length = self.temporal_config.get('sequence_length', 20)
        self.prediction_horizon = self.temporal_config.get('prediction_horizon', 1)
        self.seasonality_period = self.temporal_config.get('seasonality_period', 52)  # Weekly draws
        self.trend_window = self.temporal_config.get('trend_window', 10)
        self.lstm_units = self.temporal_config.get('lstm_units', 50)
        self.lstm_epochs = self.temporal_config.get('lstm_epochs', 100)
        
        # Data structures
        self.historical_data = None
        self.time_series_data = None
        self.seasonal_components = {}
        self.trend_components = {}
        self.models = {}
        self.scalers = {}
        self.sequences = {}
        self.temporal_patterns = {}
        self.trained = False
        
        # Enhanced sophisticated temporal analysis components
        self.seasonal_detector = SeasonalPatternDetector()
        self.cyclical_analyzer = CyclicalTrendAnalyzer()
        self.ultra_accuracy_engine = UltraHighAccuracyTemporalEngine()
        
        # Enhanced analysis caches
        self.temporal_intelligence_cache = {}
        self.pattern_analysis_cache = {}
        self.cyclical_cache = {}
        self.seasonal_analysis_cache = {}
        
        logger.info("⏰ Advanced Temporal Engine initialized with sophisticated analysis capabilities")
    
    def load_data(self, historical_data: pd.DataFrame) -> None:
        """
        Load historical lottery data for temporal analysis.
        
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
            
            # Process numbers and create time series
            self._process_temporal_data()
            
            logger.info(f"📊 Loaded {len(self.historical_data)} historical draws for temporal analysis")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def _process_temporal_data(self) -> None:
        """Process historical data into time series format."""
        try:
            # Extract and process numbers
            processed_numbers = []
            for _, row in self.historical_data.iterrows():
                numbers = row['numbers']
                
                if isinstance(numbers, str):
                    number_list = [int(x.strip()) for x in numbers.split(',')]
                elif isinstance(numbers, list):
                    number_list = [int(x) for x in numbers]
                else:
                    raise ValueError(f"Invalid numbers format: {numbers}")
                
                processed_numbers.append(sorted(number_list))
            
            self.historical_data['processed_numbers'] = processed_numbers
            
            # Create time series for each number position and individual numbers
            self._create_time_series()
            
        except Exception as e:
            logger.error(f"❌ Temporal data processing failed: {e}")
            raise
    
    def _create_time_series(self) -> None:
        """Create various time series representations."""
        self.time_series_data = {}
        
        # Time series for each number (binary: appeared or not)
        for num in range(self.number_range[0], self.number_range[1] + 1):
            series = []
            for numbers in self.historical_data['processed_numbers']:
                series.append(1 if num in numbers else 0)
            
            self.time_series_data[f'number_{num}'] = pd.Series(
                series, 
                index=self.historical_data['date']
            )
        
        # Time series for statistical features
        self.time_series_data['draw_sum'] = pd.Series(
            [sum(numbers) for numbers in self.historical_data['processed_numbers']],
            index=self.historical_data['date']
        )
        
        self.time_series_data['draw_mean'] = pd.Series(
            [np.mean(numbers) for numbers in self.historical_data['processed_numbers']],
            index=self.historical_data['date']
        )
        
        self.time_series_data['draw_std'] = pd.Series(
            [np.std(numbers) for numbers in self.historical_data['processed_numbers']],
            index=self.historical_data['date']
        )
        
        self.time_series_data['odd_count'] = pd.Series(
            [sum(1 for n in numbers if n % 2 == 1) for numbers in self.historical_data['processed_numbers']],
            index=self.historical_data['date']
        )
        
        self.time_series_data['high_count'] = pd.Series(
            [sum(1 for n in numbers if n > (self.number_range[0] + self.number_range[1]) // 2) 
             for numbers in self.historical_data['processed_numbers']],
            index=self.historical_data['date']
        )
    
    def analyze_seasonal_patterns(self) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in the time series data.
        
        Returns:
            Dictionary containing seasonal analysis results
        """
        try:
            seasonal_results = {}
            
            # Analyze seasonality for key metrics
            key_series = ['draw_sum', 'draw_mean', 'odd_count', 'high_count']
            
            for series_name in key_series:
                if series_name in self.time_series_data:
                    series = self.time_series_data[series_name]
                    
                    if len(series) >= self.seasonality_period * 2:
                        # Perform seasonal decomposition
                        decomposition = seasonal_decompose(
                            series,
                            model='additive',
                            period=min(self.seasonality_period, len(series) // 2)
                        )
                        
                        seasonal_results[series_name] = {
                            'trend': decomposition.trend.dropna().tolist(),
                            'seasonal': decomposition.seasonal.dropna().tolist(),
                            'residual': decomposition.resid.dropna().tolist(),
                            'seasonal_strength': self._calculate_seasonal_strength(decomposition)
                        }
                        
                        self.seasonal_components[series_name] = decomposition
                    else:
                        seasonal_results[series_name] = {'error': 'Insufficient data for seasonal analysis'}
            
            # Analyze cyclical patterns
            cyclical_patterns = self._analyze_cyclical_patterns()
            seasonal_results['cyclical_patterns'] = cyclical_patterns
            
            logger.info("📊 Seasonal pattern analysis complete")
            return seasonal_results
            
        except Exception as e:
            logger.error(f"❌ Seasonal analysis failed: {e}")
            raise
    
    def _calculate_seasonal_strength(self, decomposition) -> float:
        """Calculate the strength of seasonal component."""
        try:
            var_seasonal = np.var(decomposition.seasonal.dropna())
            var_residual = np.var(decomposition.resid.dropna())
            
            if var_seasonal + var_residual > 0:
                return var_seasonal / (var_seasonal + var_residual)
            else:
                return 0.0
        except:
            return 0.0
    
    def _analyze_cyclical_patterns(self) -> Dict[str, Any]:
        """Analyze cyclical patterns using Fourier analysis."""
        cyclical_results = {}
        
        try:
            # Analyze draw sum for cyclical patterns
            if 'draw_sum' in self.time_series_data:
                series = self.time_series_data['draw_sum'].values
                
                # Remove trend
                detrended = signal.detrend(series)
                
                # FFT analysis
                fft_values = fft(detrended)
                frequencies = fftfreq(len(series))
                
                # Find dominant frequencies
                magnitude = np.abs(fft_values)
                dominant_freq_idx = np.argsort(magnitude)[-5:]  # Top 5 frequencies
                
                cyclical_results['dominant_frequencies'] = frequencies[dominant_freq_idx].tolist()
                cyclical_results['frequency_magnitudes'] = magnitude[dominant_freq_idx].tolist()
                
                # Convert to periods
                cyclical_results['dominant_periods'] = [
                    1/abs(freq) if freq != 0 else 0 for freq in frequencies[dominant_freq_idx]
                ]
            
        except Exception as e:
            logger.warning(f"⚠️ Cyclical analysis failed: {e}")
            cyclical_results = {'error': str(e)}
        
        return cyclical_results
    
    def analyze_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in the time series data.
        
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            trend_results = {}
            
            # Analyze trends for each number
            for num in range(self.number_range[0], self.number_range[1] + 1):
                series_name = f'number_{num}'
                if series_name in self.time_series_data:
                    series = self.time_series_data[series_name]
                    
                    # Calculate rolling statistics
                    rolling_mean = series.rolling(window=self.trend_window).mean()
                    rolling_std = series.rolling(window=self.trend_window).std()
                    
                    # Calculate trend direction
                    recent_mean = rolling_mean.iloc[-self.trend_window:].mean()
                    earlier_mean = rolling_mean.iloc[:-self.trend_window].mean()
                    trend_direction = recent_mean - earlier_mean
                    
                    # Linear trend analysis
                    x = np.arange(len(series))
                    slope, intercept = np.polyfit(x, series.values, 1)
                    
                    trend_results[num] = {
                        'rolling_mean': rolling_mean.dropna().tolist(),
                        'rolling_std': rolling_std.dropna().tolist(),
                        'trend_direction': float(trend_direction),
                        'linear_slope': float(slope),
                        'linear_intercept': float(intercept),
                        'recent_frequency': float(recent_mean),
                        'overall_frequency': float(series.mean())
                    }
            
            # Analyze trends for aggregate metrics
            for metric in ['draw_sum', 'draw_mean', 'odd_count', 'high_count']:
                if metric in self.time_series_data:
                    series = self.time_series_data[metric]
                    
                    # Linear trend
                    x = np.arange(len(series))
                    slope, intercept = np.polyfit(x, series.values, 1)
                    
                    trend_results[f'{metric}_trend'] = {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'recent_value': float(series.iloc[-1]),
                        'mean_value': float(series.mean())
                    }
            
            self.trend_components = trend_results
            
            logger.info("📈 Trend analysis complete")
            return trend_results
            
        except Exception as e:
            logger.error(f"❌ Trend analysis failed: {e}")
            raise
    
    def create_sequences(self) -> None:
        """Create sequences for machine learning models."""
        try:
            self.sequences = {}
            
            # Create sequences for key time series
            for series_name, series in self.time_series_data.items():
                if len(series) >= self.sequence_length + 1:
                    X, y = self._create_sequences_from_series(series.values)
                    self.sequences[series_name] = {'X': X, 'y': y}
            
            logger.info(f"📊 Created sequences for {len(self.sequences)} time series")
            
        except Exception as e:
            logger.error(f"❌ Sequence creation failed: {e}")
            raise
    
    def _create_sequences_from_series(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences from time series."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(series)):
            X.append(series[i-self.sequence_length:i])
            y.append(series[i])
        
        return np.array(X), np.array(y)
    
    def train_arima_models(self) -> Dict[str, Any]:
        """
        Train ARIMA models for time series prediction.
        
        Returns:
            Dictionary containing ARIMA model results
        """
        try:
            arima_results = {}
            
            # Train ARIMA models for key series
            key_series = ['draw_sum', 'draw_mean', 'odd_count', 'high_count']
            
            for series_name in key_series:
                if series_name in self.time_series_data:
                    series = self.time_series_data[series_name]
                    
                    try:
                        # Find optimal ARIMA parameters
                        best_aic = float('inf')
                        best_params = (1, 1, 1)
                        
                        for p in range(0, 3):
                            for d in range(0, 2):
                                for q in range(0, 3):
                                    try:
                                        model = ARIMA(series, order=(p, d, q))
                                        fitted_model = model.fit()
                                        
                                        if fitted_model.aic < best_aic:
                                            best_aic = fitted_model.aic
                                            best_params = (p, d, q)
                                    except:
                                        continue
                        
                        # Train best model
                        final_model = ARIMA(series, order=best_params)
                        fitted_final_model = final_model.fit()
                        
                        # Make predictions
                        forecast = fitted_final_model.forecast(steps=self.prediction_horizon)
                        forecast_ci = fitted_final_model.get_forecast(steps=self.prediction_horizon).conf_int()
                        
                        arima_results[series_name] = {
                            'order': best_params,
                            'aic': best_aic,
                            'forecast': forecast.tolist(),
                            'forecast_ci_lower': forecast_ci.iloc[:, 0].tolist(),
                            'forecast_ci_upper': forecast_ci.iloc[:, 1].tolist(),
                            'fitted_values': fitted_final_model.fittedvalues.tolist()
                        }
                        
                        self.models[f'arima_{series_name}'] = fitted_final_model
                        
                    except Exception as e:
                        arima_results[series_name] = {'error': str(e)}
            
            logger.info(f"📊 Trained ARIMA models for {len(arima_results)} series")
            return arima_results
            
        except Exception as e:
            logger.error(f"❌ ARIMA training failed: {e}")
            raise
    
    def train_lstm_models(self) -> Dict[str, Any]:
        """
        Train LSTM models for time series prediction.
        
        Returns:
            Dictionary containing LSTM model results
        """
        try:
            lstm_results = {}
            
            if not self.sequences:
                self.create_sequences()
            
            # Train LSTM models for key sequences
            key_sequences = ['draw_sum', 'draw_mean', 'odd_count', 'high_count']
            
            for series_name in key_sequences:
                if series_name in self.sequences:
                    sequence_data = self.sequences[series_name]
                    X, y = sequence_data['X'], sequence_data['y']
                    
                    if len(X) < 10:  # Need minimum data
                        lstm_results[series_name] = {'error': 'Insufficient sequence data'}
                        continue
                    
                    try:
                        # Normalize data
                        scaler = MinMaxScaler()
                        X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
                        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()
                        
                        self.scalers[f'lstm_{series_name}'] = scaler
                        
                        # Split data
                        split_idx = int(0.8 * len(X_scaled))
                        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
                        
                        # Reshape for LSTM
                        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        
                        # Build LSTM model
                        model = Sequential([
                            LSTM(self.lstm_units, return_sequences=True, 
                                input_shape=(self.sequence_length, 1)),
                            Dropout(0.2),
                            LSTM(self.lstm_units // 2),
                            Dropout(0.2),
                            Dense(1)
                        ])
                        
                        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                        
                        # Train model
                        history = model.fit(
                            X_train, y_train,
                            epochs=min(self.lstm_epochs, 50),  # Reduced for efficiency
                            batch_size=32,
                            validation_data=(X_test, y_test),
                            verbose=0
                        )
                        
                        # Make predictions
                        y_pred = model.predict(X_test, verbose=0)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        # Future prediction
                        last_sequence = X_scaled[-1:].reshape(1, self.sequence_length, 1)
                        future_pred = model.predict(last_sequence, verbose=0)[0][0]
                        
                        # Inverse transform
                        future_pred_original = scaler.inverse_transform([[future_pred]])[0][0]
                        
                        lstm_results[series_name] = {
                            'mse': float(mse),
                            'mae': float(mae),
                            'future_prediction': float(future_pred_original),
                            'training_loss': history.history['loss'][-1],
                            'validation_loss': history.history['val_loss'][-1]
                        }
                        
                        self.models[f'lstm_{series_name}'] = model
                        
                    except Exception as e:
                        lstm_results[series_name] = {'error': str(e)}
            
            logger.info(f"🧠 Trained LSTM models for {len(lstm_results)} series")
            return lstm_results
            
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
            raise
    
    def analyze_sequential_patterns(self) -> Dict[str, Any]:
        """
        Analyze sequential patterns in number appearances.
        
        Returns:
            Dictionary containing sequential pattern analysis
        """
        try:
            pattern_results = {}
            
            # Analyze number sequence patterns
            for num in range(self.number_range[0], self.number_range[1] + 1):
                series_name = f'number_{num}'
                if series_name in self.time_series_data:
                    series = self.time_series_data[series_name]
                    
                    # Find consecutive appearance patterns
                    consecutive_patterns = self._find_consecutive_patterns(series.values)
                    
                    # Find gap patterns
                    gap_patterns = self._find_gap_patterns(series.values)
                    
                    pattern_results[num] = {
                        'consecutive_patterns': consecutive_patterns,
                        'gap_patterns': gap_patterns,
                        'last_appearance_index': int(np.where(series.values == 1)[0][-1]) if 1 in series.values else -1,
                        'total_appearances': int(series.sum())
                    }
            
            # Analyze cross-number patterns
            cross_patterns = self._analyze_cross_number_patterns()
            pattern_results['cross_patterns'] = cross_patterns
            
            self.temporal_patterns = pattern_results
            
            logger.info("🔄 Sequential pattern analysis complete")
            return pattern_results
            
        except Exception as e:
            logger.error(f"❌ Sequential pattern analysis failed: {e}")
            raise
    
    def _find_consecutive_patterns(self, series: np.ndarray) -> Dict[str, Any]:
        """Find consecutive appearance patterns in a series."""
        patterns = {
            'max_consecutive': 0,
            'consecutive_sequences': [],
            'average_consecutive': 0
        }
        
        current_consecutive = 0
        consecutive_sequences = []
        
        for value in series:
            if value == 1:
                current_consecutive += 1
            else:
                if current_consecutive > 0:
                    consecutive_sequences.append(current_consecutive)
                    patterns['max_consecutive'] = max(patterns['max_consecutive'], current_consecutive)
                current_consecutive = 0
        
        # Handle case where series ends with consecutive 1s
        if current_consecutive > 0:
            consecutive_sequences.append(current_consecutive)
            patterns['max_consecutive'] = max(patterns['max_consecutive'], current_consecutive)
        
        patterns['consecutive_sequences'] = consecutive_sequences
        patterns['average_consecutive'] = np.mean(consecutive_sequences) if consecutive_sequences else 0
        
        return patterns
    
    def _find_gap_patterns(self, series: np.ndarray) -> Dict[str, Any]:
        """Find gap patterns between appearances."""
        patterns = {
            'gaps': [],
            'max_gap': 0,
            'min_gap': 0,
            'average_gap': 0
        }
        
        appearances = np.where(series == 1)[0]
        
        if len(appearances) > 1:
            gaps = np.diff(appearances) - 1  # Subtract 1 to get actual gap
            patterns['gaps'] = gaps.tolist()
            patterns['max_gap'] = int(gaps.max())
            patterns['min_gap'] = int(gaps.min())
            patterns['average_gap'] = float(gaps.mean())
        
        return patterns
    
    def _analyze_cross_number_patterns(self) -> Dict[str, Any]:
        """Analyze patterns between different numbers."""
        cross_patterns = {
            'correlation_matrix': {},
            'leading_indicators': {},
            'lagging_indicators': {}
        }
        
        try:
            # Create correlation matrix
            number_series = []
            numbers = []
            
            for num in range(self.number_range[0], min(self.number_range[0] + 20, self.number_range[1] + 1)):
                series_name = f'number_{num}'
                if series_name in self.time_series_data:
                    number_series.append(self.time_series_data[series_name].values)
                    numbers.append(num)
            
            if len(number_series) > 1:
                # Calculate correlations
                correlation_matrix = np.corrcoef(number_series)
                
                for i, num1 in enumerate(numbers):
                    cross_patterns['correlation_matrix'][num1] = {}
                    for j, num2 in enumerate(numbers):
                        if i != j:
                            cross_patterns['correlation_matrix'][num1][num2] = float(correlation_matrix[i, j])
                
                # Find leading/lagging relationships (simplified)
                for i, num1 in enumerate(numbers[:5]):  # Limit for efficiency
                    for j, num2 in enumerate(numbers[:5]):
                        if i != j:
                            # Calculate cross-correlation with lag
                            series1 = number_series[i]
                            series2 = number_series[j]
                            
                            cross_corr = np.correlate(series1, series2, mode='full')
                            lags = np.arange(-len(series2) + 1, len(series1))
                            
                            max_corr_idx = np.argmax(np.abs(cross_corr))
                            best_lag = lags[max_corr_idx]
                            
                            if abs(best_lag) > 0 and abs(cross_corr[max_corr_idx]) > 0.1:
                                if best_lag > 0:
                                    cross_patterns['leading_indicators'][num1] = cross_patterns['leading_indicators'].get(num1, [])
                                    cross_patterns['leading_indicators'][num1].append({
                                        'target': num2,
                                        'lag': int(best_lag),
                                        'correlation': float(cross_corr[max_corr_idx])
                                    })
        
        except Exception as e:
            logger.warning(f"⚠️ Cross-pattern analysis failed: {e}")
        
        return cross_patterns
    
    def train(self) -> None:
        """Train the temporal engine on historical data."""
        try:
            if self.historical_data is None:
                raise ValueError("No historical data loaded")
            
            logger.info("🎓 Training Temporal Engine...")
            
            # Analyze seasonal patterns
            self.analyze_seasonal_patterns()
            
            # Analyze trends
            self.analyze_trends()
            
            # Analyze sequential patterns
            self.analyze_sequential_patterns()
            
            # Create sequences for ML models
            self.create_sequences()
            
            # Train ARIMA models
            self.train_arima_models()
            
            # Train LSTM models (if TensorFlow is available)
            try:
                self.train_lstm_models()
            except Exception as e:
                logger.warning(f"⚠️ LSTM training skipped: {e}")
            
            self.trained = True
            
            logger.info("✅ Temporal Engine training complete")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def predict(self, num_predictions: int = 1, strategy: str = 'combined') -> List[Dict[str, Any]]:
        """
        Generate predictions using temporal analysis.
        
        Args:
            num_predictions: Number of prediction sets to generate
            strategy: Prediction strategy ('combined', 'arima', 'lstm', 'pattern', 'trend')
            
        Returns:
            List of prediction dictionaries
        """
        try:
            if not self.trained:
                raise ValueError("Engine not trained. Call train() first.")
            
            predictions = []
            
            for i in range(num_predictions):
                if strategy == 'combined':
                    prediction = self._predict_combined()
                elif strategy == 'arima':
                    prediction = self._predict_arima()
                elif strategy == 'lstm':
                    prediction = self._predict_lstm()
                elif strategy == 'pattern':
                    prediction = self._predict_pattern_based()
                elif strategy == 'trend':
                    prediction = self._predict_trend_based()
                else:
                    prediction = self._predict_combined()
                
                predictions.append({
                    'numbers': prediction['numbers'],
                    'confidence': prediction['confidence'],
                    'strategy': strategy,
                    'temporal_analysis': prediction.get('analysis', {}),
                    'generated_at': datetime.now().isoformat(),
                    'engine': 'temporal'
                })
            
            logger.info(f"🎯 Generated {num_predictions} temporal predictions using {strategy} strategy")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            raise
    
    def _predict_combined(self) -> Dict[str, Any]:
        """Generate prediction using combined temporal methods."""
        try:
            # Get predictions from different methods
            arima_pred = self._predict_arima()
            pattern_pred = self._predict_pattern_based()
            trend_pred = self._predict_trend_based()
            
            # Combine predictions with weights
            all_numbers = []
            all_numbers.extend(arima_pred['numbers'])
            all_numbers.extend(pattern_pred['numbers'])
            all_numbers.extend(trend_pred['numbers'])
            
            # Count occurrences and select most frequent
            from collections import Counter
            number_counts = Counter(all_numbers)
            
            # Select top numbers
            selected_numbers = []
            for num, count in number_counts.most_common():
                if len(selected_numbers) < self.numbers_per_draw:
                    selected_numbers.append(num)
            
            # Fill remaining slots if needed
            while len(selected_numbers) < self.numbers_per_draw:
                available_numbers = [n for n in range(self.number_range[0], self.number_range[1] + 1) 
                                   if n not in selected_numbers]
                if available_numbers:
                    selected_numbers.append(np.random.choice(available_numbers))
                else:
                    break
            
            # Calculate combined confidence
            confidences = [arima_pred['confidence'], pattern_pred['confidence'], trend_pred['confidence']]
            combined_confidence = np.mean(confidences)
            
            return {
                'numbers': sorted(selected_numbers),
                'confidence': combined_confidence,
                'analysis': {
                    'arima_contribution': arima_pred['confidence'],
                    'pattern_contribution': pattern_pred['confidence'],
                    'trend_contribution': trend_pred['confidence']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Combined prediction failed: {e}")
            return self._fallback_prediction()
    
    def _predict_arima(self) -> Dict[str, Any]:
        """Generate prediction using ARIMA models."""
        try:
            predictions = {}
            
            # Get ARIMA predictions for key metrics
            for model_name, model in self.models.items():
                if model_name.startswith('arima_'):
                    series_name = model_name.replace('arima_', '')
                    forecast = model.forecast(steps=1)
                    predictions[series_name] = float(forecast.iloc[0])
            
            # Convert metric predictions to number selection
            selected_numbers = self._convert_metrics_to_numbers(predictions)
            
            confidence = 0.6  # Moderate confidence for ARIMA
            
            return {
                'numbers': selected_numbers,
                'confidence': confidence,
                'analysis': predictions
            }
            
        except Exception as e:
            logger.warning(f"⚠️ ARIMA prediction failed: {e}")
            return self._fallback_prediction()
    
    def _predict_lstm(self) -> Dict[str, Any]:
        """Generate prediction using LSTM models."""
        try:
            predictions = {}
            
            # Get LSTM predictions for key metrics
            for model_name, model in self.models.items():
                if model_name.startswith('lstm_'):
                    series_name = model_name.replace('lstm_', '')
                    
                    if series_name in self.time_series_data and f'lstm_{series_name}' in self.scalers:
                        # Prepare input sequence
                        series = self.time_series_data[series_name]
                        scaler = self.scalers[f'lstm_{series_name}']
                        
                        last_sequence = series.values[-self.sequence_length:]
                        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
                        input_seq = last_sequence_scaled.reshape(1, self.sequence_length, 1)
                        
                        # Make prediction
                        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
                        pred_original = scaler.inverse_transform([[pred_scaled]])[0][0]
                        
                        predictions[series_name] = float(pred_original)
            
            # Convert metric predictions to number selection
            selected_numbers = self._convert_metrics_to_numbers(predictions)
            
            confidence = 0.7  # Higher confidence for LSTM
            
            return {
                'numbers': selected_numbers,
                'confidence': confidence,
                'analysis': predictions
            }
            
        except Exception as e:
            logger.warning(f"⚠️ LSTM prediction failed: {e}")
            return self._fallback_prediction()
    
    def _predict_pattern_based(self) -> Dict[str, Any]:
        """Generate prediction based on sequential patterns."""
        try:
            selected_numbers = []
            
            # Select numbers based on gap patterns
            for num in range(self.number_range[0], self.number_range[1] + 1):
                if num in self.temporal_patterns:
                    pattern_data = self.temporal_patterns[num]
                    
                    # Check if number is due based on gap patterns
                    if pattern_data['gap_patterns']['gaps']:
                        avg_gap = pattern_data['gap_patterns']['average_gap']
                        last_appearance = pattern_data['last_appearance_index']
                        current_gap = len(self.historical_data) - 1 - last_appearance
                        
                        # Number is "due" if current gap exceeds average
                        if current_gap >= avg_gap * 0.8:  # 80% threshold
                            selected_numbers.append(num)
            
            # If not enough numbers, add based on frequency
            if len(selected_numbers) < self.numbers_per_draw:
                frequency_scores = []
                for num in range(self.number_range[0], self.number_range[1] + 1):
                    if num not in selected_numbers and num in self.temporal_patterns:
                        appearances = self.temporal_patterns[num]['total_appearances']
                        frequency_scores.append((num, appearances))
                
                frequency_scores.sort(key=lambda x: x[1], reverse=True)
                
                for num, _ in frequency_scores:
                    if len(selected_numbers) < self.numbers_per_draw:
                        selected_numbers.append(num)
            
            # Ensure we have enough numbers
            while len(selected_numbers) < self.numbers_per_draw:
                available = [n for n in range(self.number_range[0], self.number_range[1] + 1) 
                           if n not in selected_numbers]
                if available:
                    selected_numbers.append(np.random.choice(available))
                else:
                    break
            
            confidence = 0.55
            
            return {
                'numbers': sorted(selected_numbers[:self.numbers_per_draw]),
                'confidence': confidence,
                'analysis': {'method': 'gap_pattern_analysis'}
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Pattern-based prediction failed: {e}")
            return self._fallback_prediction()
    
    def _predict_trend_based(self) -> Dict[str, Any]:
        """Generate prediction based on trend analysis."""
        try:
            selected_numbers = []
            
            # Select numbers with positive trends
            for num in range(self.number_range[0], self.number_range[1] + 1):
                if num in self.trend_components:
                    trend_data = self.trend_components[num]
                    
                    # Check for positive trend direction and slope
                    if (trend_data.get('trend_direction', 0) > 0 and 
                        trend_data.get('linear_slope', 0) > 0):
                        selected_numbers.append(num)
            
            # Sort by trend strength and select top numbers
            if len(selected_numbers) > self.numbers_per_draw:
                trend_scores = []
                for num in selected_numbers:
                    trend_strength = (self.trend_components[num]['trend_direction'] + 
                                    self.trend_components[num]['linear_slope']) / 2
                    trend_scores.append((num, trend_strength))
                
                trend_scores.sort(key=lambda x: x[1], reverse=True)
                selected_numbers = [num for num, _ in trend_scores[:self.numbers_per_draw]]
            
            # Fill remaining slots if needed
            while len(selected_numbers) < self.numbers_per_draw:
                available = [n for n in range(self.number_range[0], self.number_range[1] + 1) 
                           if n not in selected_numbers]
                if available:
                    selected_numbers.append(np.random.choice(available))
                else:
                    break
            
            confidence = 0.5
            
            return {
                'numbers': sorted(selected_numbers),
                'confidence': confidence,
                'analysis': {'method': 'trend_analysis'}
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Trend-based prediction failed: {e}")
            return self._fallback_prediction()
    
    def _convert_metrics_to_numbers(self, metric_predictions: Dict[str, float]) -> List[int]:
        """Convert metric predictions to number selection."""
        selected_numbers = []
        
        try:
            # Use predicted sum to guide selection
            if 'draw_sum' in metric_predictions:
                target_sum = metric_predictions['draw_sum']
                
                # Select numbers that could achieve target sum
                avg_number = target_sum / self.numbers_per_draw
                
                # Create range around average
                min_num = max(self.number_range[0], int(avg_number - 10))
                max_num = min(self.number_range[1], int(avg_number + 10))
                
                candidate_numbers = list(range(min_num, max_num + 1))
                selected_numbers = np.random.choice(
                    candidate_numbers, 
                    size=min(self.numbers_per_draw, len(candidate_numbers)), 
                    replace=False
                ).tolist()
            
            # Adjust for odd/even count if predicted
            if 'odd_count' in metric_predictions and len(selected_numbers) == self.numbers_per_draw:
                target_odd_count = int(round(metric_predictions['odd_count']))
                current_odd_count = sum(1 for n in selected_numbers if n % 2 == 1)
                
                # Adjust if needed (simplified approach)
                if current_odd_count != target_odd_count:
                    # This would require more complex logic to swap numbers
                    pass
            
            # Ensure we have the right number of selections
            while len(selected_numbers) < self.numbers_per_draw:
                available = [n for n in range(self.number_range[0], self.number_range[1] + 1) 
                           if n not in selected_numbers]
                if available:
                    selected_numbers.append(np.random.choice(available))
                else:
                    break
            
        except Exception as e:
            logger.warning(f"⚠️ Metric conversion failed: {e}")
            
        # Fallback to random if conversion failed
        if len(selected_numbers) < self.numbers_per_draw:
            selected_numbers = np.random.choice(
                range(self.number_range[0], self.number_range[1] + 1),
                size=self.numbers_per_draw,
                replace=False
            ).tolist()
        
        return sorted(selected_numbers[:self.numbers_per_draw])
    
    def _fallback_prediction(self) -> Dict[str, Any]:
        """Generate fallback prediction when other methods fail."""
        selected_numbers = np.random.choice(
            range(self.number_range[0], self.number_range[1] + 1),
            size=self.numbers_per_draw,
            replace=False
        ).tolist()
        
        return {
            'numbers': sorted(selected_numbers),
            'confidence': 0.3,
            'analysis': {'method': 'fallback_random'}
        }
    
    def perform_comprehensive_temporal_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive temporal analysis using sophisticated algorithms.
        
        Returns:
            Dictionary containing comprehensive temporal analysis results
        """
        try:
            if not self.trained or self.historical_data is None:
                logger.warning("Engine not trained or no historical data available")
                return {'error': 'Engine not trained'}
            
            logger.info("🔬 Performing comprehensive temporal analysis...")
            start_time = time.time()
            
            # Prepare data for sophisticated analysis
            historical_numbers = self.historical_data['processed_numbers'].tolist()
            draw_dates = self.historical_data['date'].tolist()
            
            # Get game type from configuration
            game_type = 'lotto_max' if self.numbers_per_draw == 7 else 'lotto_649'
            
            # Perform comprehensive analysis using ultra-high accuracy engine
            comprehensive_analysis = self.ultra_accuracy_engine.perform_comprehensive_temporal_analysis(
                historical_numbers, draw_dates, game_type
            )
            
            # Cache the analysis results
            cache_key = f"comprehensive_{game_type}_{len(historical_numbers)}"
            self.temporal_intelligence_cache[cache_key] = comprehensive_analysis
            
            analysis_time = time.time() - start_time
            self.analysis_times.append(analysis_time)
            
            logger.info(f"✅ Comprehensive temporal analysis complete in {analysis_time:.3f}s")
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive temporal analysis: {e}")
            return {'error': str(e)}

    def generate_sophisticated_temporal_predictions(self, num_sets: int = 5) -> Dict[str, Any]:
        """
        Generate sophisticated temporal predictions using all analysis components.
        
        Args:
            num_sets: Number of prediction sets to generate
        
        Returns:
            Dictionary containing sophisticated temporal predictions
        """
        try:
            logger.info(f"🎯 Generating {num_sets} sophisticated temporal prediction sets...")
            
            # First perform comprehensive temporal analysis
            comprehensive_analysis = self.perform_comprehensive_temporal_analysis()
            if 'error' in comprehensive_analysis:
                logger.error("Cannot generate predictions without comprehensive analysis")
                return comprehensive_analysis
            
            # Generate temporal predictions using ultra-high accuracy engine
            temporal_predictions = self.ultra_accuracy_engine.generate_temporal_predictions(
                comprehensive_analysis, num_sets
            )
            
            # Enhanced predictions with original engine insights
            enhanced_predictions = self._enhance_with_original_insights(temporal_predictions, num_sets)
            
            # Calculate comprehensive confidence scores
            final_predictions = self._calculate_comprehensive_confidence(enhanced_predictions, comprehensive_analysis)
            
            logger.info(f"✅ Generated {len(final_predictions.get('temporal_sets', []))} sophisticated temporal prediction sets")
            
            return final_predictions
            
        except Exception as e:
            logger.error(f"Error generating sophisticated temporal predictions: {e}")
            return {'error': str(e), 'temporal_sets': []}

    def _enhance_with_original_insights(self, temporal_predictions: Dict[str, Any], num_sets: int) -> Dict[str, Any]:
        """
        Enhance temporal predictions with original engine insights.
        
        Args:
            temporal_predictions: Base temporal predictions
            num_sets: Number of sets to generate
        
        Returns:
            Enhanced temporal predictions
        """
        try:
            enhanced_sets = []
            temporal_sets = temporal_predictions.get('temporal_sets', [])
            
            for i, temporal_set in enumerate(temporal_sets):
                enhanced_set = temporal_set.copy()
                
                # Add original engine analysis
                try:
                    original_prediction = self.predict(strategy='combined')
                    
                    # Blend with original insights if available
                    if 'numbers' in original_prediction:
                        original_numbers = set(original_prediction['numbers'])
                        temporal_numbers = set(enhanced_set['numbers'])
                        
                        # Calculate overlap score
                        overlap_score = len(original_numbers & temporal_numbers) / len(temporal_numbers)
                        
                        enhanced_set['original_engine_overlap'] = overlap_score
                        enhanced_set['original_engine_confidence'] = original_prediction.get('confidence', 0)
                        
                        # Boost confidence if there's good overlap
                        if overlap_score > 0.3:
                            enhanced_set['confidence'] = min(1.0, enhanced_set['confidence'] * (1 + overlap_score))
                
                except Exception as e:
                    logger.warning(f"Could not enhance with original insights: {e}")
                
                enhanced_sets.append(enhanced_set)
            
            # Generate additional sets if needed
            while len(enhanced_sets) < num_sets:
                try:
                    additional_prediction = self.predict(strategy='combined')
                    if 'numbers' in additional_prediction:
                        enhanced_sets.append({
                            'set_id': len(enhanced_sets) + 1,
                            'numbers': sorted(additional_prediction['numbers']),
                            'source': 'original_engine',
                            'confidence': additional_prediction.get('confidence', 0.5),
                            'temporal_intelligence': 0.0
                        })
                except:
                    # Generate random set as last resort
                    random_numbers = np.random.choice(
                        range(self.number_range[0], self.number_range[1] + 1),
                        size=self.numbers_per_draw,
                        replace=False
                    ).tolist()
                    
                    enhanced_sets.append({
                        'set_id': len(enhanced_sets) + 1,
                        'numbers': sorted(random_numbers),
                        'source': 'random_fallback',
                        'confidence': 0.3,
                        'temporal_intelligence': 0.0
                    })
            
            return {
                'temporal_sets': enhanced_sets[:num_sets],
                'confidence_scores': [s.get('confidence', 0) for s in enhanced_sets[:num_sets]],
                'overall_confidence': temporal_predictions.get('overall_confidence', 0)
            }
            
        except Exception as e:
            logger.error(f"Error enhancing with original insights: {e}")
            return temporal_predictions

    def _calculate_comprehensive_confidence(self, predictions: Dict[str, Any], 
                                          comprehensive_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence scores for predictions.
        
        Args:
            predictions: Base predictions
            comprehensive_analysis: Comprehensive temporal analysis results
        
        Returns:
            Predictions with comprehensive confidence scores
        """
        try:
            temporal_intelligence = comprehensive_analysis.get('overall_temporal_intelligence', 0)
            seasonal_confidence = comprehensive_analysis.get('seasonal_analysis', {}).get('confidence_score', 0)
            cyclical_confidence = comprehensive_analysis.get('cyclical_analysis', {}).get('confidence_score', 0)
            
            # Calculate comprehensive confidence multiplier
            intelligence_multiplier = 1 + (temporal_intelligence * 0.5)
            seasonal_multiplier = 1 + (seasonal_confidence * 0.3)
            cyclical_multiplier = 1 + (cyclical_confidence * 0.2)
            
            overall_multiplier = min(2.0, intelligence_multiplier * seasonal_multiplier * cyclical_multiplier / 3)
            
            # Apply to all prediction sets
            enhanced_sets = []
            confidence_scores = []
            
            for prediction_set in predictions.get('temporal_sets', []):
                enhanced_set = prediction_set.copy()
                base_confidence = enhanced_set.get('confidence', 0)
                
                # Apply comprehensive confidence enhancement
                enhanced_confidence = min(1.0, base_confidence * overall_multiplier)
                enhanced_set['confidence'] = enhanced_confidence
                enhanced_set['comprehensive_analysis'] = {
                    'temporal_intelligence': temporal_intelligence,
                    'seasonal_confidence': seasonal_confidence,
                    'cyclical_confidence': cyclical_confidence,
                    'confidence_multiplier': overall_multiplier
                }
                
                enhanced_sets.append(enhanced_set)
                confidence_scores.append(enhanced_confidence)
            
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return {
                'temporal_sets': enhanced_sets,
                'confidence_scores': confidence_scores,
                'overall_confidence': float(overall_confidence),
                'temporal_intelligence': temporal_intelligence,
                'comprehensive_analysis': comprehensive_analysis
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive confidence: {e}")
            return predictions

    def get_temporal_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive temporal analysis summary.
        
        Returns:
            Dictionary containing temporal analysis summary
        """
        if not self.trained:
            return {'error': 'Engine not trained'}
        
        # Get sophisticated analysis summary
        sophisticated_summary = {}
        if self.temporal_intelligence_cache:
            latest_analysis = list(self.temporal_intelligence_cache.values())[-1]
            sophisticated_summary = {
                'temporal_intelligence': latest_analysis.get('overall_temporal_intelligence', 0),
                'seasonal_patterns': bool(latest_analysis.get('seasonal_analysis')),
                'cyclical_patterns': bool(latest_analysis.get('cyclical_analysis')),
                'analysis_timestamp': latest_analysis.get('timestamp')
            }
        
        return {
            'engine': 'advanced_temporal',
            'data_span': {
                'start_date': self.historical_data['date'].min().isoformat(),
                'end_date': self.historical_data['date'].max().isoformat(),
                'total_draws': len(self.historical_data)
            },
            'models_trained': list(self.models.keys()),
            'seasonal_components': len(self.seasonal_components),
            'temporal_patterns': len(self.temporal_patterns),
            'available_strategies': ['combined', 'arima', 'lstm', 'pattern', 'trend', 'sophisticated_temporal'],
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'sophisticated_analysis': sophisticated_summary,
            'performance_metrics': {
                'average_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0,
                'total_analyses': len(self.analysis_times)
            }
        }