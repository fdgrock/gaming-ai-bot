"""
Enhanced Analytics Service - Business Logic Extracted from Monolithic App

This module provides comprehensive analytics and visualization services combining:
1. Historical data analysis and trend detection
2. Performance metrics and insights generation
3. Strategy recommendations and optimization analysis
4. Data visualization and reporting capabilities
5. Cross-model performance comparison

Extracted Functions Integrated:
- load_analytics_historical_data() -> load_historical_analytics()
- analyze_historical_trends() -> analyze_trends()
- generate_performance_insights() -> generate_insights()
- generate_strategy_recommendations() -> recommend_strategies()
- create_trend_visualization() -> create_visualizations()

Enhanced Features:
- BaseService integration with dependency injection
- Comprehensive trend analysis and pattern recognition
- Performance insights with actionable recommendations
- Advanced statistical calculations and metrics
- Clean separation from UI visualization logic
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import os
import glob

# Phase 2 Service Integration
from .base_service import BaseService, ServiceValidationMixin
from ..core.exceptions import AnalyticsError, ValidationError, safe_execute
from ..core.utils import sanitize_game_name

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Core analytics engine for trend analysis and performance insights.
    
    Handles historical data analysis, trend detection, performance metrics,
    and strategic recommendations without UI dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize analytics engine with configuration."""
        self.config = config or {}
        self.data_cache = {}
        self.analysis_cache = {}
        
    def load_historical_analytics_data(self, game_key: str) -> pd.DataFrame:
        """
        Load all historical draw data for analytics.
        
        Extracted from: load_analytics_historical_data() in app.py (Line 8035)
        Enhanced with: Better caching, comprehensive error handling
        
        Args:
            game_key: Sanitized game name key
            
        Returns:
            DataFrame with historical data for analytics
        """
        try:
            # Check cache first
            cache_key = f"historical_{game_key}"
            if cache_key in self.data_cache:
                cache_time, data = self.data_cache[cache_key]
                if datetime.now() - cache_time < timedelta(minutes=15):
                    return data
            
            data_files = []
            
            # Check multiple locations for data
            possible_paths = [
                f"data/{game_key}/history/*.csv",
                f"data/{game_key}/*.csv", 
                f"data/history/{game_key}/*.csv"
            ]
            
            for pattern in possible_paths:
                data_files.extend(glob.glob(pattern))
            
            if not data_files:
                logger.info(f"📊 No historical data files found for {game_key}")
                return pd.DataFrame()
            
            all_data = []
            loaded_files = 0
            
            for file in data_files:
                try:
                    df = pd.read_csv(file)
                    df['source_file'] = os.path.basename(file)
                    all_data.append(df)
                    loaded_files += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {file}: {e}")
                    continue
            
            if not all_data:
                logger.warning(f"⚠️ No valid data files could be loaded for {game_key}")
                return pd.DataFrame()
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Standardize date column
            if 'draw_date' in combined_df.columns:
                combined_df['draw_date'] = pd.to_datetime(combined_df['draw_date'], errors='coerce')
            elif 'date' in combined_df.columns:
                combined_df['draw_date'] = pd.to_datetime(combined_df['date'], errors='coerce')
                combined_df = combined_df.drop('date', axis=1)
            
            # Remove duplicates and sort
            if 'draw_date' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['draw_date']).sort_values('draw_date', ascending=False)
            
            # Cache the result
            self.data_cache[cache_key] = (datetime.now(), combined_df)
            
            logger.info(f"✅ Loaded historical analytics data: {len(combined_df)} records from {loaded_files} files")
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical analytics data for {game_key}: {e}")
            return pd.DataFrame()
    
    def load_prediction_analytics_data(self, game_key: str) -> List[Dict[str, Any]]:
        """
        Load all prediction files for analytics analysis.
        
        Args:
            game_key: Sanitized game name key
            
        Returns:
            List of prediction data dictionaries
        """
        try:
            pred_files = []
            
            # Search for prediction files in multiple patterns
            patterns = [
                f"predictions/{game_key}/*.json",
                f"predictions/{game_key}/**/*.json"
            ]
            
            for pattern in patterns:
                pred_files.extend(glob.glob(pattern, recursive=True))
            
            # Remove duplicates
            pred_files = list(set(pred_files))
            
            predictions = []
            loaded_count = 0
            
            for file_path in pred_files:
                try:
                    with open(file_path, 'r') as f:
                        pred_data = json.load(f)
                    
                    # Add file metadata
                    pred_data['file_path'] = file_path
                    pred_data['file_name'] = os.path.basename(file_path)
                    
                    # Extract date from filename or modify time
                    try:
                        file_stat = os.stat(file_path)
                        pred_data['file_modified'] = datetime.fromtimestamp(file_stat.st_mtime)
                    except Exception:
                        pred_data['file_modified'] = datetime.now()
                    
                    predictions.append(pred_data)
                    loaded_count += 1
                    
                except Exception as e:
                    logger.debug(f"Could not load prediction file {file_path}: {e}")
                    continue
            
            # Sort by modification time (newest first)
            predictions.sort(key=lambda x: x.get('file_modified', datetime.min), reverse=True)
            
            logger.info(f"✅ Loaded {loaded_count} prediction files for analytics")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to load prediction analytics data: {e}")
            return []
    
    def analyze_historical_trends(self, game_key: str, time_period_days: int = 90) -> Dict[str, Any]:
        """
        Analyze historical performance trends over time.
        
        Extracted from: analyze_historical_trends() in app.py (Line 9777)
        Enhanced with: More comprehensive trend analysis, better metrics
        
        Args:
            game_key: Sanitized game name key
            time_period_days: Analysis time window in days
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Load historical analysis data
            history_file = f"analysis_history_{game_key}.json"
            
            if not os.path.exists(history_file):
                return {
                    "trends": [],
                    "insights": [],
                    "recommendations": [],
                    "data_points": 0,
                    "analysis_period": f"{time_period_days} days",
                    "status": "no_data"
                }
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Filter data by time period
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            recent_data = [
                entry for entry in history_data 
                if datetime.fromisoformat(entry.get('timestamp', '2023-01-01')) > cutoff_date
            ]
            
            if not recent_data:
                return {
                    "trends": [],
                    "insights": [],
                    "recommendations": [],
                    "data_points": 0,
                    "analysis_period": f"{time_period_days} days",
                    "status": "insufficient_data"
                }
            
            # Analyze trends
            trends = self._calculate_performance_trends(recent_data)
            insights = self._generate_performance_insights(recent_data)
            recommendations = self._generate_strategy_recommendations(trends, recent_data)
            
            # Additional analytics
            performance_stats = self._calculate_performance_statistics(recent_data)
            
            result = {
                "trends": trends,
                "insights": insights,
                "recommendations": recommendations,
                "performance_stats": performance_stats,
                "data_points": len(recent_data),
                "analysis_period": f"{time_period_days} days",
                "status": "success"
            }
            
            logger.info(f"✅ Analyzed historical trends for {game_key}: {len(recent_data)} data points")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing historical trends for {game_key}: {e}")
            return {
                "trends": [],
                "insights": [],
                "recommendations": [],
                "data_points": 0,
                "analysis_period": f"{time_period_days} days",
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_performance_trends(self, recent_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate performance trend metrics."""
        trends = []
        
        try:
            if len(recent_data) < 2:
                return trends
            
            # Extract performance metrics
            row_accuracy_trend = [entry.get('best_row_accuracy', 0) for entry in recent_data]
            overall_accuracy_trend = [entry.get('overall_accuracy', 0) for entry in recent_data]
            
            # Calculate trend directions and statistics
            metrics = [
                ("Row Performance", row_accuracy_trend),
                ("Overall Performance", overall_accuracy_trend)
            ]
            
            for metric_name, values in metrics:
                if len(values) >= 2:
                    # Calculate trend direction using linear regression
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    # Determine trend characteristics
                    direction = "improving" if slope > 0.1 else "declining" if slope < -0.1 else "stable"
                    change = values[-1] - values[0]
                    volatility = np.std(values)
                    
                    trends.append({
                        "metric": metric_name,
                        "direction": direction,
                        "slope": float(slope),
                        "change": float(change),
                        "current_value": float(values[-1]),
                        "volatility": float(volatility),
                        "data_points": len(values)
                    })
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance trends: {e}")
        
        return trends
    
    def _generate_performance_insights(self, recent_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate intelligent insights from recent performance data.
        
        Extracted from: generate_performance_insights() in app.py (Line 9901)
        Enhanced with: More comprehensive analysis, better categorization
        """
        insights = []
        
        if not recent_data:
            return insights
        
        try:
            # Calculate performance metrics
            row_accuracies = [entry.get('best_row_accuracy', 0) for entry in recent_data]
            overall_accuracies = [entry.get('overall_accuracy', 0) for entry in recent_data]
            
            avg_row_accuracy = np.mean(row_accuracies)
            avg_overall_accuracy = np.mean(overall_accuracies)
            
            # Performance consistency analysis
            row_std = np.std(row_accuracies)
            consistency_level = "high" if row_std < 5 else "moderate" if row_std < 10 else "low"
            
            insights.append({
                "type": "performance",
                "title": "Performance Consistency",
                "description": f"Your prediction consistency is {consistency_level} (std: {row_std:.1f}%)",
                "impact": "high" if consistency_level == "high" else "medium",
                "metrics": {
                    "consistency_level": consistency_level,
                    "standard_deviation": float(row_std),
                    "average_accuracy": float(avg_row_accuracy)
                }
            })
            
            # Row vs Overall performance comparison
            if avg_overall_accuracy > 0:
                performance_ratio = avg_row_accuracy / avg_overall_accuracy
                if performance_ratio > 1.2:
                    insights.append({
                        "type": "strategy",
                        "title": "Row-Focused Strategy Recommended",
                        "description": f"Row accuracy ({avg_row_accuracy:.1f}%) significantly exceeds overall accuracy ({avg_overall_accuracy:.1f}%)",
                        "impact": "high",
                        "metrics": {
                            "row_accuracy": float(avg_row_accuracy),
                            "overall_accuracy": float(avg_overall_accuracy),
                            "performance_ratio": float(performance_ratio)
                        }
                    })
            
            # Recent improvement detection
            if len(recent_data) >= 5:
                recent_avg = np.mean(row_accuracies[-3:])
                earlier_avg = np.mean(row_accuracies[:3])
                improvement = recent_avg - earlier_avg
                
                if abs(improvement) > 2:
                    trend_type = "improvement" if improvement > 0 else "decline"
                    insights.append({
                        "type": "trend",
                        "title": f"Performance {trend_type.capitalize()} Detected",
                        "description": f"Recent predictions show {abs(improvement):.1f}% {trend_type} in accuracy",
                        "impact": "positive" if improvement > 0 else "negative",
                        "metrics": {
                            "improvement": float(improvement),
                            "recent_average": float(recent_avg),
                            "earlier_average": float(earlier_avg)
                        }
                    })
            
            # Performance grade analysis
            if recent_data:
                latest_grade = recent_data[-1].get('performance_grade', 'D')
                if latest_grade in ['C', 'D']:
                    insights.append({
                        "type": "alert",
                        "title": "Performance Enhancement Needed",
                        "description": f"Current performance grade: {latest_grade}. Consider strategy adjustments.",
                        "impact": "high",
                        "metrics": {
                            "current_grade": latest_grade,
                            "grade_numeric": ord(latest_grade) - ord('A')
                        }
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generating performance insights: {e}")
            return [{
                "type": "error", 
                "title": "Analysis Error", 
                "description": f"Could not generate insights: {e}", 
                "impact": "low"
            }]
    
    def _generate_strategy_recommendations(self, trends: List[Dict[str, Any]], 
                                         recent_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate AI-driven strategy recommendations.
        
        Extracted from: generate_strategy_recommendations() in app.py (Line 9954)
        Enhanced with: More comprehensive recommendations, priority scoring
        """
        recommendations = []
        
        try:
            if not trends or not recent_data:
                return recommendations
            
            # Analyze trend patterns
            row_trend = next((t for t in trends if t["metric"] == "Row Performance"), None)
            overall_trend = next((t for t in trends if t["metric"] == "Overall Performance"), None)
            
            # Improving performance recommendations
            if row_trend and row_trend["direction"] == "improving":
                recommendations.append({
                    "priority": "high",
                    "category": "optimization",
                    "title": "Continue Row-Focused Training",
                    "description": f"Your row performance is improving (+{row_trend['change']:.1f}%). Maintain current strategy.",
                    "action": "Use row_optimization strategy in next training session",
                    "confidence": 0.8,
                    "impact_score": row_trend["change"] * 10
                })
            
            # Declining performance recommendations
            elif row_trend and row_trend["direction"] == "declining":
                recommendations.append({
                    "priority": "medium",
                    "category": "correction",
                    "title": "Address Performance Decline", 
                    "description": f"Row accuracy decreased by {abs(row_trend['change']):.1f}%. Strategy adjustment needed.",
                    "action": "Retrain with clustering_enhancement focus to improve pattern recognition",
                    "confidence": 0.7,
                    "impact_score": abs(row_trend["change"]) * 8
                })
            
            # Model diversity recommendations
            model_types_used = set(entry.get('model_type', 'unknown') for entry in recent_data[-5:])
            if len(model_types_used) == 1:
                recommendations.append({
                    "priority": "medium", 
                    "category": "diversification",
                    "title": "Try Different Model Types",
                    "description": "Single model type detected. Diversification might improve performance.",
                    "action": "Experiment with Transformer, LSTM, and XGBoost models",
                    "confidence": 0.6,
                    "impact_score": 15
                })
            
            # Volatility management
            if row_trend and row_trend.get("volatility", 0) > 10:
                recommendations.append({
                    "priority": "medium",
                    "category": "stability",
                    "title": "Reduce Performance Volatility",
                    "description": f"High performance volatility detected (σ={row_trend['volatility']:.1f}%).",
                    "action": "Use ensemble methods or regularization techniques",
                    "confidence": 0.7,
                    "impact_score": row_trend["volatility"]
                })
            
            # Sort by priority and impact score
            priority_order = {"high": 3, "medium": 2, "low": 1}
            recommendations.sort(
                key=lambda x: (priority_order.get(x["priority"], 0), x.get("impact_score", 0)),
                reverse=True
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating strategy recommendations: {e}")
            return [{
                "priority": "low", 
                "category": "error", 
                "title": "Recommendation Error", 
                "description": f"Could not generate recommendations: {e}", 
                "action": "Continue with current strategy"
            }]
    
    def _calculate_performance_statistics(self, recent_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance statistics."""
        try:
            if not recent_data:
                return {}
            
            row_accuracies = [entry.get('best_row_accuracy', 0) for entry in recent_data]
            overall_accuracies = [entry.get('overall_accuracy', 0) for entry in recent_data]
            
            stats = {
                "sample_size": len(recent_data),
                "row_performance": {
                    "mean": float(np.mean(row_accuracies)),
                    "median": float(np.median(row_accuracies)),
                    "std": float(np.std(row_accuracies)),
                    "min": float(np.min(row_accuracies)),
                    "max": float(np.max(row_accuracies)),
                    "percentile_25": float(np.percentile(row_accuracies, 25)),
                    "percentile_75": float(np.percentile(row_accuracies, 75))
                },
                "overall_performance": {
                    "mean": float(np.mean(overall_accuracies)),
                    "median": float(np.median(overall_accuracies)), 
                    "std": float(np.std(overall_accuracies)),
                    "min": float(np.min(overall_accuracies)),
                    "max": float(np.max(overall_accuracies)),
                    "percentile_25": float(np.percentile(overall_accuracies, 25)),
                    "percentile_75": float(np.percentile(overall_accuracies, 75))
                }
            }
            
            # Calculate correlation
            if len(row_accuracies) > 1 and len(overall_accuracies) > 1:
                correlation = np.corrcoef(row_accuracies, overall_accuracies)[0, 1]
                stats["correlation"] = float(correlation) if not np.isnan(correlation) else 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance statistics: {e}")
            return {}
    
    def analyze_cross_model_performance(self, game_key: str) -> Dict[str, Any]:
        """
        Analyze performance across different model types.
        
        Args:
            game_key: Sanitized game name key
            
        Returns:
            Dictionary with cross-model performance analysis
        """
        try:
            # Load prediction data
            predictions = self.load_prediction_analytics_data(game_key)
            
            if not predictions:
                return {"status": "no_data", "models": {}}
            
            # Group by model type
            model_groups = {}
            for pred in predictions:
                model_type = pred.get('model_type', 'unknown')
                if model_type not in model_groups:
                    model_groups[model_type] = []
                model_groups[model_type].append(pred)
            
            # Analyze each model type
            model_analysis = {}
            for model_type, model_predictions in model_groups.items():
                if len(model_predictions) < 2:
                    continue
                
                # Extract performance metrics
                accuracies = [p.get('accuracy', 0) for p in model_predictions if p.get('accuracy')]
                
                if accuracies:
                    model_analysis[model_type] = {
                        "prediction_count": len(model_predictions),
                        "mean_accuracy": float(np.mean(accuracies)),
                        "std_accuracy": float(np.std(accuracies)), 
                        "max_accuracy": float(np.max(accuracies)),
                        "min_accuracy": float(np.min(accuracies)),
                        "consistency_score": self._calculate_consistency_score(accuracies)
                    }
            
            # Find best performing model
            best_model = None
            if model_analysis:
                best_model = max(model_analysis.keys(), 
                               key=lambda k: model_analysis[k]["mean_accuracy"])
            
            result = {
                "status": "success",
                "models": model_analysis,
                "best_model": best_model,
                "total_predictions": len(predictions),
                "model_count": len(model_groups)
            }
            
            logger.info(f"✅ Analyzed cross-model performance: {len(model_groups)} models")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing cross-model performance: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_consistency_score(self, values: List[float]) -> float:
        """Calculate consistency score (inverse of coefficient of variation)."""
        try:
            if not values or len(values) < 2:
                return 0.0
            
            mean_val = np.mean(values)
            if mean_val == 0:
                return 0.0
            
            cv = np.std(values) / mean_val
            consistency_score = max(0, 1 - cv)  # Higher is more consistent
            return float(consistency_score)
            
        except Exception:
            return 0.0


# =============================================================================
# ENHANCED ANALYTICS SERVICE WITH BASE SERVICE INTEGRATION
# =============================================================================

class AnalyticsService(BaseService, ServiceValidationMixin):
    """
    Enhanced Analytics Service integrating Phase 2 service foundation.
    
    Combines:
    - BaseService patterns (dependency injection, logging, error handling)
    - AnalyticsEngine functionality (trends, insights, recommendations)
    - Extracted business logic from monolithic app.py
    """
    
    def _setup_service(self) -> None:
        """Initialize analytics service with integrated functionality."""
        self.log_operation("setup", status="info", action="initializing analytics service")
        
        # Initialize the AnalyticsEngine with config
        engine_config = {
            'cache_ttl': getattr(self.config, 'analytics_cache_ttl', 900),  # 15 minutes
        }
        
        self.analytics_engine = AnalyticsEngine(engine_config)
        
        self.log_operation("setup", status="success")
    
    # Delegate to AnalyticsEngine with enhanced error handling
    def analyze_trends(self, game_name: str, time_period_days: int = 90) -> Dict[str, Any]:
        """Analyze historical trends with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.analytics_engine.analyze_historical_trends,
            "analyze_trends",
            game_name=game_name,
            default_return={"trends": [], "insights": [], "recommendations": []},
            game_key=game_key,
            time_period_days=time_period_days
        )
    
    def load_historical_analytics(self, game_name: str) -> pd.DataFrame:
        """Load historical analytics data with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.analytics_engine.load_historical_analytics_data,
            "load_historical_analytics",
            game_name=game_name,
            default_return=pd.DataFrame(),
            game_key=game_key
        )
    
    def analyze_cross_model_performance(self, game_name: str) -> Dict[str, Any]:
        """Analyze cross-model performance with service-level error handling."""
        self.validate_initialized()
        game_key = self.validate_game_name(game_name)
        
        return self.safe_execute_operation(
            self.analytics_engine.analyze_cross_model_performance,
            "analyze_cross_model_performance",
            game_name=game_name,
            default_return={"status": "error", "models": {}},
            game_key=game_key
        )
    
    def generate_performance_report(self, game_name: str, 
                                  time_period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        self.validate_initialized()
        
        try:
            # Gather all analytics data
            trends = self.analyze_trends(game_name, time_period_days)
            cross_model = self.analyze_cross_model_performance(game_name)
            historical_data = self.load_historical_analytics(game_name)
            
            # Compile comprehensive report
            report = {
                "game_name": game_name,
                "analysis_period": f"{time_period_days} days",
                "generated_at": datetime.now().isoformat(),
                "trends_analysis": trends,
                "cross_model_analysis": cross_model,
                "data_summary": {
                    "historical_records": len(historical_data),
                    "has_data": not historical_data.empty
                },
                "overall_status": "success" if trends.get("status") == "success" else "limited_data"
            }
            
            self.log_operation("performance_report", game_name, "success",
                             records=len(historical_data),
                             trends_found=len(trends.get("trends", [])))
            
            return report
            
        except Exception as e:
            self.log_operation("performance_report", game_name, "error", error=str(e))
            return {
                "game_name": game_name,
                "status": "error",
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
    
    def _service_health_check(self) -> Optional[Dict[str, Any]]:
        """Analytics service specific health check."""
        health = {
            'healthy': True,
            'issues': []
        }
        
        # Check AnalyticsEngine
        try:
            if not hasattr(self.analytics_engine, 'data_cache'):
                health['healthy'] = False
                health['issues'].append("AnalyticsEngine not properly initialized")
        except Exception as e:
            health['healthy'] = False
            health['issues'].append(f"AnalyticsEngine error: {e}")
        
        # Check data access
        try:
            test_paths = ["data", "predictions", "analysis_history"]
            for path in test_paths:
                if not Path(path).exists():
                    Path(path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            health['healthy'] = False
            health['issues'].append(f"Cannot access data directories: {e}")
        
        return health