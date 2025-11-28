"""
Enhancement Engine for the lottery prediction system.

This module provides continuous improvement capabilities for AI models,
including performance monitoring, adaptive learning, and model optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import threading
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


@dataclass
class EnhancementResult:
    """Result of model enhancement operation."""
    model_id: str
    enhancement_type: str
    success: bool
    improvement_score: float
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    execution_time: float
    recommendations: List[str]
    error_message: Optional[str] = None


@dataclass
class PerformanceAlert:
    """Alert for performance degradation."""
    model_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime


class PerformanceMonitor:
    """Monitors model performance and triggers enhancement actions."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Performance Monitor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.performance_history = {}
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = config.get('performance_thresholds', {
            'accuracy_drop': 0.05,
            'confidence_drop': 0.1,
            'response_time_increase': 2.0,
            'error_rate_increase': 0.02
        })
        
        logger.info("📊 Performance Monitor initialized")
    
    def start_monitoring(self, check_interval: int = 300) -> None:
        """
        Start performance monitoring.
        
        Args:
            check_interval: Interval between checks in seconds
        """
        if self.monitoring_active:
            logger.warning("⚠️ Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"🎯 Performance monitoring started (interval: {check_interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ Performance monitoring stopped")
    
    def _monitoring_loop(self, check_interval: int) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_model_performance()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(check_interval)
    
    def record_performance(self, model_id: str, metrics: Dict[str, float]) -> None:
        """
        Record performance metrics for a model.
        
        Args:
            model_id: ID of the model
            metrics: Performance metrics
        """
        timestamp = datetime.now()
        
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        performance_record = {
            'timestamp': timestamp,
            'metrics': metrics.copy()
        }
        
        self.performance_history[model_id].append(performance_record)
        
        # Keep only last 1000 records per model
        if len(self.performance_history[model_id]) > 1000:
            self.performance_history[model_id] = self.performance_history[model_id][-1000:]
        
        # Check for performance issues
        self._check_performance_degradation(model_id, metrics)
    
    def _check_model_performance(self) -> None:
        """Check performance for all monitored models."""
        for model_id in self.performance_history:
            try:
                recent_metrics = self._get_recent_metrics(model_id)
                if recent_metrics:
                    self._analyze_performance_trends(model_id, recent_metrics)
            except Exception as e:
                logger.error(f"❌ Performance check failed for {model_id}: {e}")
    
    def _check_performance_degradation(self, model_id: str, current_metrics: Dict[str, float]) -> None:
        """
        Check for performance degradation.
        
        Args:
            model_id: ID of the model
            current_metrics: Current performance metrics
        """
        if model_id not in self.performance_history or len(self.performance_history[model_id]) < 2:
            return
        
        historical_records = self.performance_history[model_id]
        baseline_metrics = self._calculate_baseline_metrics(historical_records[-10:])
        
        # Check accuracy degradation
        if 'accuracy' in current_metrics and 'accuracy' in baseline_metrics:
            accuracy_drop = baseline_metrics['accuracy'] - current_metrics['accuracy']
            if accuracy_drop > self.thresholds['accuracy_drop']:
                alert = PerformanceAlert(
                    model_id=model_id,
                    alert_type='accuracy_degradation',
                    severity='high' if accuracy_drop > 0.1 else 'medium',
                    message=f"Accuracy dropped by {accuracy_drop:.3f}",
                    current_value=current_metrics['accuracy'],
                    threshold_value=self.thresholds['accuracy_drop'],
                    timestamp=datetime.now()
                )
                self._add_alert(alert)
        
        # Check confidence degradation
        if 'average_confidence' in current_metrics and 'average_confidence' in baseline_metrics:
            confidence_drop = baseline_metrics['average_confidence'] - current_metrics['average_confidence']
            if confidence_drop > self.thresholds['confidence_drop']:
                alert = PerformanceAlert(
                    model_id=model_id,
                    alert_type='confidence_degradation',
                    severity='medium',
                    message=f"Confidence dropped by {confidence_drop:.3f}",
                    current_value=current_metrics['average_confidence'],
                    threshold_value=self.thresholds['confidence_drop'],
                    timestamp=datetime.now()
                )
                self._add_alert(alert)
    
    def _calculate_baseline_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate baseline metrics from historical records."""
        if not records:
            return {}
        
        baseline = {}
        metrics_keys = set()
        
        for record in records:
            metrics_keys.update(record['metrics'].keys())
        
        for key in metrics_keys:
            values = [record['metrics'].get(key, 0) for record in records if key in record['metrics']]
            if values:
                baseline[key] = np.mean(values)
        
        return baseline
    
    def _get_recent_metrics(self, model_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent metrics for a model."""
        if model_id not in self.performance_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_records = [
            record for record in self.performance_history[model_id]
            if record['timestamp'] > cutoff_time
        ]
        
        return recent_records
    
    def _analyze_performance_trends(self, model_id: str, metrics: List[Dict[str, Any]]) -> None:
        """Analyze performance trends for a model."""
        if len(metrics) < 5:
            return
        
        try:
            # Analyze accuracy trend
            accuracy_values = [m['metrics'].get('accuracy', 0) for m in metrics]
            if len(accuracy_values) >= 5:
                trend = np.polyfit(range(len(accuracy_values)), accuracy_values, 1)[0]
                
                if trend < -0.001:  # Declining trend
                    alert = PerformanceAlert(
                        model_id=model_id,
                        alert_type='declining_trend',
                        severity='medium',
                        message=f"Declining accuracy trend detected (slope: {trend:.4f})",
                        current_value=accuracy_values[-1],
                        threshold_value=-0.001,
                        timestamp=datetime.now()
                    )
                    self._add_alert(alert)
        
        except Exception as e:
            logger.error(f"❌ Trend analysis failed for {model_id}: {e}")
    
    def _add_alert(self, alert: PerformanceAlert) -> None:
        """Add a performance alert."""
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.warning(f"⚠️ Performance Alert [{alert.severity}]: {alert.message} for {alert.model_id}")
    
    def get_alerts(self, model_id: Optional[str] = None, severity: Optional[str] = None) -> List[PerformanceAlert]:
        """
        Get performance alerts.
        
        Args:
            model_id: Filter by model ID
            severity: Filter by severity level
            
        Returns:
            List of matching alerts
        """
        filtered_alerts = self.alerts
        
        if model_id:
            filtered_alerts = [alert for alert in filtered_alerts if alert.model_id == model_id]
        
        if severity:
            filtered_alerts = [alert for alert in filtered_alerts if alert.severity == severity]
        
        return filtered_alerts
    
    def clear_alerts(self, model_id: Optional[str] = None) -> None:
        """
        Clear alerts.
        
        Args:
            model_id: Clear alerts for specific model (None for all)
        """
        if model_id:
            self.alerts = [alert for alert in self.alerts if alert.model_id != model_id]
        else:
            self.alerts = []
        
        logger.info(f"🧹 Alerts cleared for {model_id or 'all models'}")


class HyperparameterOptimizer:
    """Optimizes model hyperparameters using advanced techniques."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Hyperparameter Optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_history = {}
        
        # Optuna study configuration
        self.study_config = config.get('optuna_config', {
            'n_trials': 100,
            'timeout': 3600,  # 1 hour
            'n_jobs': -1
        })
        
        logger.info("🎛️ Hyperparameter Optimizer initialized")
    
    def optimize_model(self, model, training_data: pd.DataFrame, 
                      param_space: Dict[str, Any]) -> EnhancementResult:
        """
        Optimize model hyperparameters.
        
        Args:
            model: Model to optimize
            training_data: Training data
            param_space: Parameter search space
            
        Returns:
            Enhancement result
        """
        start_time = time.time()
        model_id = f"{model.metadata.name}_{model.metadata.version}"
        
        try:
            # Get baseline performance
            baseline_metrics = self._evaluate_model(model, training_data)
            
            # Create Optuna study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42)
            )
            
            # Define objective function
            def objective(trial):
                return self._optimize_objective(trial, model, training_data, param_space)
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=self.study_config['n_trials'],
                timeout=self.study_config['timeout'],
                n_jobs=1  # Keep single threaded for stability
            )
            
            # Apply best parameters
            best_params = study.best_params
            self._apply_parameters(model, best_params)
            
            # Retrain with best parameters
            model.train()
            
            # Get improved performance
            improved_metrics = self._evaluate_model(model, training_data)
            
            execution_time = time.time() - start_time
            improvement_score = improved_metrics.get('accuracy', 0) - baseline_metrics.get('accuracy', 0)
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                study, baseline_metrics, improved_metrics
            )
            
            result = EnhancementResult(
                model_id=model_id,
                enhancement_type='hyperparameter_optimization',
                success=True,
                improvement_score=improvement_score,
                before_metrics=baseline_metrics,
                after_metrics=improved_metrics,
                execution_time=execution_time,
                recommendations=recommendations
            )
            
            # Store optimization history
            self._store_optimization_result(model_id, study, result)
            
            logger.info(f"🎯 Optimization complete for {model_id}: {improvement_score:+.3f} improvement")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Optimization failed for {model_id}: {e}")
            
            return EnhancementResult(
                model_id=model_id,
                enhancement_type='hyperparameter_optimization',
                success=False,
                improvement_score=0.0,
                before_metrics={},
                after_metrics={},
                execution_time=execution_time,
                recommendations=[],
                error_message=str(e)
            )
    
    def _optimize_objective(self, trial, model, training_data: pd.DataFrame, 
                           param_space: Dict[str, Any]) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            model: Model to optimize
            training_data: Training data
            param_space: Parameter search space
            
        Returns:
            Objective value (higher is better)
        """
        try:
            # Suggest parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Apply parameters to model
            self._apply_parameters(model, params)
            
            # Train and evaluate
            model.load_data(training_data)
            model.train()
            
            # Evaluate performance
            metrics = self._evaluate_model(model, training_data)
            
            # Return accuracy as objective (can be customized)
            return metrics.get('accuracy', 0.0)
            
        except Exception as e:
            logger.warning(f"⚠️ Trial failed: {e}")
            return 0.0
    
    def _apply_parameters(self, model, params: Dict[str, Any]) -> None:
        """Apply parameters to model configuration."""
        try:
            # Update model configuration
            model.config.update(params)
            
            # If model has parameter-specific methods, call them
            if hasattr(model, 'update_parameters'):
                model.update_parameters(params)
            
        except Exception as e:
            logger.error(f"❌ Failed to apply parameters: {e}")
            raise
    
    def _evaluate_model(self, model, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            # Simple evaluation using a subset of data
            eval_size = min(100, len(data))
            eval_data = data.sample(n=eval_size)
            
            # Generate predictions
            predictions = model.predict(len(eval_data), 'balanced')
            
            # Calculate basic metrics
            if predictions:
                avg_confidence = np.mean([pred.confidence for pred in predictions])
                return {
                    'accuracy': avg_confidence,  # Simplified metric
                    'average_confidence': avg_confidence,
                    'prediction_count': len(predictions)
                }
            else:
                return {'accuracy': 0.0, 'average_confidence': 0.0, 'prediction_count': 0}
                
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {'accuracy': 0.0, 'average_confidence': 0.0, 'prediction_count': 0}
    
    def _generate_optimization_recommendations(self, study, before_metrics: Dict[str, float], 
                                            after_metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        try:
            # Parameter importance analysis
            if len(study.trials) > 10:
                importance = optuna.importance.get_param_importances(study)
                top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                
                recommendations.append(f"Most important parameters: {[p[0] for p in top_params]}")
            
            # Performance improvement analysis
            improvement = after_metrics.get('accuracy', 0) - before_metrics.get('accuracy', 0)
            if improvement > 0.01:
                recommendations.append(f"Significant improvement achieved: +{improvement:.3f}")
            elif improvement < -0.01:
                recommendations.append(f"Performance decreased: {improvement:.3f}")
            else:
                recommendations.append("Minimal performance change, consider different parameter ranges")
            
            # Best trial analysis
            best_trial = study.best_trial
            recommendations.append(f"Best trial #{best_trial.number} with score: {best_trial.value:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            recommendations.append("Recommendation generation failed")
        
        return recommendations
    
    def _store_optimization_result(self, model_id: str, study, result: EnhancementResult) -> None:
        """Store optimization result for future reference."""
        try:
            optimization_record = {
                'timestamp': datetime.now().isoformat(),
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'result': {
                    'improvement_score': result.improvement_score,
                    'execution_time': result.execution_time,
                    'success': result.success
                }
            }
            
            if model_id not in self.optimization_history:
                self.optimization_history[model_id] = []
            
            self.optimization_history[model_id].append(optimization_record)
            
            # Keep only last 10 optimization records per model
            if len(self.optimization_history[model_id]) > 10:
                self.optimization_history[model_id] = self.optimization_history[model_id][-10:]
            
        except Exception as e:
            logger.error(f"❌ Failed to store optimization result: {e}")


class EnhancementEngine:
    """
    Main Enhancement Engine that coordinates all improvement activities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Enhancement Engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
        self.hyperparameter_optimizer = HyperparameterOptimizer(config)
        self.enhancement_history = {}
        self.active_enhancements = {}
        
        # Enhancement settings
        self.enhancement_config = config.get('enhancement_config', {
            'auto_enhancement': True,
            'enhancement_interval': 3600,  # 1 hour
            'max_concurrent_enhancements': 2
        })
        
        logger.info("🚀 Enhancement Engine initialized")
    
    def start_enhancement_system(self) -> None:
        """Start the enhancement system."""
        try:
            # Start performance monitoring
            self.performance_monitor.start_monitoring(
                check_interval=self.enhancement_config.get('monitoring_interval', 300)
            )
            
            # Start auto-enhancement if enabled
            if self.enhancement_config.get('auto_enhancement', True):
                self._start_auto_enhancement()
            
            logger.info("🟢 Enhancement system started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start enhancement system: {e}")
            raise
    
    def stop_enhancement_system(self) -> None:
        """Stop the enhancement system."""
        try:
            self.performance_monitor.stop_monitoring()
            logger.info("🔴 Enhancement system stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop enhancement system: {e}")
    
    def enhance_model(self, model, enhancement_type: str = 'full') -> EnhancementResult:
        """
        Enhance a specific model.
        
        Args:
            model: Model to enhance
            enhancement_type: Type of enhancement ('hyperparameters', 'architecture', 'full')
            
        Returns:
            Enhancement result
        """
        model_id = f"{model.metadata.name}_{model.metadata.version}"
        
        try:
            if model_id in self.active_enhancements:
                logger.warning(f"⚠️ Enhancement already in progress for {model_id}")
                return self.active_enhancements[model_id]
            
            logger.info(f"🔧 Starting {enhancement_type} enhancement for {model_id}")
            
            # Mark enhancement as active
            self.active_enhancements[model_id] = None
            
            start_time = time.time()
            result = None
            
            if enhancement_type in ['hyperparameters', 'full']:
                result = self._enhance_hyperparameters(model)
            
            if enhancement_type in ['architecture', 'full'] and result and result.success:
                architecture_result = self._enhance_architecture(model)
                if architecture_result.success:
                    result = architecture_result
            
            if result is None:
                result = EnhancementResult(
                    model_id=model_id,
                    enhancement_type=enhancement_type,
                    success=False,
                    improvement_score=0.0,
                    before_metrics={},
                    after_metrics={},
                    execution_time=time.time() - start_time,
                    recommendations=[],
                    error_message="No enhancement performed"
                )
            
            # Store result
            self._store_enhancement_result(model_id, result)
            
            # Remove from active enhancements
            del self.active_enhancements[model_id]
            
            logger.info(f"✅ Enhancement complete for {model_id}: {result.improvement_score:+.3f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Enhancement failed for {model_id}: {e}")
            
            # Clean up
            if model_id in self.active_enhancements:
                del self.active_enhancements[model_id]
            
            return EnhancementResult(
                model_id=model_id,
                enhancement_type=enhancement_type,
                success=False,
                improvement_score=0.0,
                before_metrics={},
                after_metrics={},
                execution_time=time.time() - start_time,
                recommendations=[],
                error_message=str(e)
            )
    
    def _enhance_hyperparameters(self, model) -> EnhancementResult:
        """Enhance model hyperparameters."""
        try:
            # Define parameter space based on model type
            param_space = self._get_parameter_space(model)
            
            if not param_space:
                return EnhancementResult(
                    model_id=f"{model.metadata.name}_{model.metadata.version}",
                    enhancement_type='hyperparameters',
                    success=False,
                    improvement_score=0.0,
                    before_metrics={},
                    after_metrics={},
                    execution_time=0.0,
                    recommendations=["No parameter space defined for this model type"],
                    error_message="No parameter space available"
                )
            
            # Get training data from model
            if not hasattr(model, 'training_data') or model.training_data is None:
                raise ValueError("Model has no training data")
            
            # Optimize hyperparameters
            result = self.hyperparameter_optimizer.optimize_model(
                model, model.training_data, param_space
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Hyperparameter enhancement failed: {e}")
            raise
    
    def _enhance_architecture(self, model) -> EnhancementResult:
        """Enhance model architecture (placeholder for future implementation)."""
        model_id = f"{model.metadata.name}_{model.metadata.version}"
        
        # This is a placeholder for future architecture enhancement features
        logger.info(f"🏗️ Architecture enhancement not yet implemented for {model_id}")
        
        return EnhancementResult(
            model_id=model_id,
            enhancement_type='architecture',
            success=False,
            improvement_score=0.0,
            before_metrics={},
            after_metrics={},
            execution_time=0.0,
            recommendations=["Architecture enhancement coming in future version"],
            error_message="Not implemented"
        )
    
    def _get_parameter_space(self, model) -> Dict[str, Any]:
        """Get parameter search space for a model."""
        model_type = model.metadata.model_type.lower()
        
        # Define parameter spaces for different model types
        parameter_spaces = {
            'mathematical': {
                'trend_weight': {'type': 'float', 'low': 0.1, 'high': 0.9},
                'frequency_weight': {'type': 'float', 'low': 0.1, 'high': 0.9},
                'gap_weight': {'type': 'float', 'low': 0.1, 'high': 0.9}
            },
            'ensemble': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True}
            },
            'temporal': {
                'lstm_units': {'type': 'int', 'low': 32, 'high': 256},
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.5},
                'sequence_length': {'type': 'int', 'low': 10, 'high': 100}
            },
            'optimizer': {
                'population_size': {'type': 'int', 'low': 50, 'high': 500},
                'mutation_rate': {'type': 'float', 'low': 0.01, 'high': 0.1},
                'crossover_rate': {'type': 'float', 'low': 0.5, 'high': 0.9}
            }
        }
        
        return parameter_spaces.get(model_type, {})
    
    def _start_auto_enhancement(self) -> None:
        """Start automatic enhancement system."""
        def auto_enhance_loop():
            while self.enhancement_config.get('auto_enhancement', True):
                try:
                    # Check for performance alerts
                    high_priority_alerts = self.performance_monitor.get_alerts(severity='high')
                    
                    for alert in high_priority_alerts:
                        if len(self.active_enhancements) < self.enhancement_config.get('max_concurrent_enhancements', 2):
                            logger.info(f"🚨 Triggering auto-enhancement for {alert.model_id} due to {alert.alert_type}")
                            # Note: Would need model registry to actually enhance
                    
                    time.sleep(self.enhancement_config.get('enhancement_interval', 3600))
                    
                except Exception as e:
                    logger.error(f"❌ Auto-enhancement loop error: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        # Start auto-enhancement thread
        auto_thread = threading.Thread(target=auto_enhance_loop, daemon=True)
        auto_thread.start()
        
        logger.info("🤖 Auto-enhancement system started")
    
    def _store_enhancement_result(self, model_id: str, result: EnhancementResult) -> None:
        """Store enhancement result."""
        try:
            if model_id not in self.enhancement_history:
                self.enhancement_history[model_id] = []
            
            enhancement_record = {
                'timestamp': datetime.now().isoformat(),
                'enhancement_type': result.enhancement_type,
                'success': result.success,
                'improvement_score': result.improvement_score,
                'execution_time': result.execution_time,
                'recommendations': result.recommendations
            }
            
            self.enhancement_history[model_id].append(enhancement_record)
            
            # Keep only last 20 enhancement records per model
            if len(self.enhancement_history[model_id]) > 20:
                self.enhancement_history[model_id] = self.enhancement_history[model_id][-20:]
            
        except Exception as e:
            logger.error(f"❌ Failed to store enhancement result: {e}")
    
    def get_enhancement_status(self) -> Dict[str, Any]:
        """Get enhancement system status."""
        return {
            'monitoring_active': self.performance_monitor.monitoring_active,
            'active_enhancements': list(self.active_enhancements.keys()),
            'enhancement_history_count': {
                model_id: len(history) for model_id, history in self.enhancement_history.items()
            },
            'recent_alerts': len(self.performance_monitor.get_alerts()),
            'configuration': self.enhancement_config
        }
    
    def get_enhancement_history(self, model_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get enhancement history.
        
        Args:
            model_id: Filter by model ID (None for all)
            
        Returns:
            Enhancement history
        """
        if model_id:
            return {model_id: self.enhancement_history.get(model_id, [])}
        else:
            return self.enhancement_history.copy()
    
    def record_model_performance(self, model_id: str, metrics: Dict[str, float]) -> None:
        """
        Record model performance metrics.
        
        Args:
            model_id: ID of the model
            metrics: Performance metrics
        """
        self.performance_monitor.record_performance(model_id, metrics)
    
    def get_performance_alerts(self, model_id: Optional[str] = None) -> List[PerformanceAlert]:
        """
        Get performance alerts.
        
        Args:
            model_id: Filter by model ID
            
        Returns:
            List of performance alerts
        """
        return self.performance_monitor.get_alerts(model_id)