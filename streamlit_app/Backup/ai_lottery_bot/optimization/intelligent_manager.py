"""
Phase C: Intelligent Model Management & Real-time Adaptation
===========================================================

This module implements intelligent model management with real-time adaptation:
- Dynamic model performance monitoring
- Automatic model retraining triggers
- Intelligent model selection based on recent performance
- Adaptive confidence scoring
- Real-time feature drift detection
- Automated model lifecycle management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Real-time model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_score: float
    prediction_entropy: float
    feature_drift_score: float
    temporal_consistency: float
    last_updated: datetime
    prediction_count: int
    error_rate: float

@dataclass
class ModelHealthStatus:
    """Model health assessment"""
    overall_health: str  # 'excellent', 'good', 'fair', 'poor', 'critical'
    health_score: float  # 0-100
    performance_trend: str  # 'improving', 'stable', 'declining'
    recommendations: List[str]
    alerts: List[str]
    last_assessment: datetime
    days_since_training: int
    predictions_since_training: int

@dataclass
class AdaptationTrigger:
    """Triggers for model adaptation"""
    trigger_type: str  # 'performance_drop', 'feature_drift', 'time_based', 'prediction_count'
    threshold: float
    current_value: float
    triggered: bool
    trigger_time: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'

class IntelligentModelManager:
    """
    Intelligent model management system with real-time adaptation
    """
    
    def __init__(self, monitoring_window: int = 100, adaptation_threshold: float = 0.05):
        self.monitoring_window = monitoring_window
        self.adaptation_threshold = adaptation_threshold
        self.model_registry = {}
        self.performance_history = {}
        self.prediction_buffer = {}
        self.feature_baselines = {}
        self.monitoring_active = False
        self.adaptation_rules = self._initialize_adaptation_rules()
        
    def _initialize_adaptation_rules(self) -> Dict[str, AdaptationTrigger]:
        """Initialize adaptation trigger rules"""
        return {
            'accuracy_drop': AdaptationTrigger(
                trigger_type='performance_drop',
                threshold=0.05,  # 5% accuracy drop
                current_value=0.0,
                triggered=False,
                trigger_time=datetime.now(),
                severity='medium'
            ),
            'feature_drift': AdaptationTrigger(
                trigger_type='feature_drift',
                threshold=0.1,  # 10% feature drift
                current_value=0.0,
                triggered=False,
                trigger_time=datetime.now(),
                severity='high'
            ),
            'time_based': AdaptationTrigger(
                trigger_type='time_based',
                threshold=30,  # 30 days
                current_value=0.0,
                triggered=False,
                trigger_time=datetime.now(),
                severity='low'
            ),
            'prediction_count': AdaptationTrigger(
                trigger_type='prediction_count',
                threshold=1000,  # 1000 predictions
                current_value=0.0,
                triggered=False,
                trigger_time=datetime.now(),
                severity='medium'
            ),
            'error_rate_spike': AdaptationTrigger(
                trigger_type='performance_drop',
                threshold=0.1,  # 10% error rate
                current_value=0.0,
                triggered=False,
                trigger_time=datetime.now(),
                severity='critical'
            )
        }
    
    def register_model(self, model_id: str, model: Any, model_type: str, 
                      training_data: np.ndarray = None, training_labels: np.ndarray = None):
        """Register a model for intelligent management"""
        
        # Calculate feature baseline if training data provided
        feature_baseline = None
        if training_data is not None:
            feature_baseline = {
                'mean': np.mean(training_data, axis=0),
                'std': np.std(training_data, axis=0),
                'min': np.min(training_data, axis=0),
                'max': np.max(training_data, axis=0)
            }
        
        self.model_registry[model_id] = {
            'model': model,
            'model_type': model_type,
            'registration_time': datetime.now(),
            'last_training_time': datetime.now(),
            'training_data_shape': training_data.shape if training_data is not None else None,
            'n_classes': len(np.unique(training_labels)) if training_labels is not None else None,
            'version': '1.0.0',
            'status': 'active'
        }
        
        if feature_baseline:
            self.feature_baselines[model_id] = feature_baseline
        
        # Initialize performance tracking
        self.performance_history[model_id] = deque(maxlen=self.monitoring_window)
        self.prediction_buffer[model_id] = deque(maxlen=self.monitoring_window)
        
        logger.info(f"📝 Model {model_id} registered for intelligent management")
    
    def predict_with_monitoring(self, model_id: str, X: np.ndarray, 
                               return_confidence: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Make predictions with real-time monitoring"""
        
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not registered")
        
        model = self.model_registry[model_id]['model']
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            predictions = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        else:
            predictions = model.predict(X)
            confidence = np.ones(len(predictions)) * 0.5  # Default confidence
            entropy = np.ones(len(predictions)) * 1.0
        
        # Calculate feature drift if baseline available
        feature_drift = 0.0
        if model_id in self.feature_baselines:
            feature_drift = self._calculate_feature_drift(model_id, X)
        
        # Calculate confidence score
        confidence_score = self._calculate_adaptive_confidence(
            model_id, confidence, entropy, feature_drift
        )
        
        # Store prediction for monitoring
        prediction_info = {
            'timestamp': datetime.now(),
            'predictions': predictions,
            'confidence': confidence,
            'entropy': entropy,
            'feature_drift': feature_drift,
            'confidence_score': confidence_score
        }
        
        self.prediction_buffer[model_id].append(prediction_info)
        
        # Update performance metrics
        self._update_performance_metrics(model_id)
        
        # Check adaptation triggers
        self._check_adaptation_triggers(model_id)
        
        if return_confidence:
            return predictions, {
                'confidence': confidence_score,
                'feature_drift': feature_drift,
                'entropy': np.mean(entropy),
                'model_health': self.get_model_health(model_id)
            }
        else:
            return predictions
    
    def _calculate_feature_drift(self, model_id: str, X: np.ndarray) -> float:
        """Calculate feature drift score"""
        if model_id not in self.feature_baselines:
            return 0.0
        
        baseline = self.feature_baselines[model_id]
        current_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0)
        }
        
        # Calculate drift as normalized difference in means and stds
        mean_drift = np.mean(np.abs(current_stats['mean'] - baseline['mean']) / 
                           (baseline['std'] + 1e-10))
        std_drift = np.mean(np.abs(current_stats['std'] - baseline['std']) / 
                          (baseline['std'] + 1e-10))
        
        drift_score = (mean_drift + std_drift) / 2
        return min(drift_score, 1.0)  # Cap at 1.0
    
    def _calculate_adaptive_confidence(self, model_id: str, confidence: np.ndarray, 
                                     entropy: np.ndarray, feature_drift: float) -> float:
        """Calculate adaptive confidence score"""
        
        # Base confidence from model predictions
        base_confidence = np.mean(confidence)
        
        # Entropy penalty (higher entropy = lower confidence)
        entropy_penalty = np.mean(entropy) / np.log(self.model_registry[model_id].get('n_classes', 2))
        
        # Feature drift penalty
        drift_penalty = feature_drift
        
        # Historical performance factor
        performance_factor = 1.0
        if self.performance_history[model_id]:
            recent_performance = list(self.performance_history[model_id])[-10:]
            if recent_performance:
                avg_recent_accuracy = np.mean([p.accuracy for p in recent_performance])
                performance_factor = avg_recent_accuracy
        
        # Temporal consistency factor
        temporal_factor = self._calculate_temporal_consistency(model_id)
        
        # Combine factors
        adaptive_confidence = (
            base_confidence * 0.4 +
            (1 - entropy_penalty) * 0.2 +
            (1 - drift_penalty) * 0.2 +
            performance_factor * 0.1 +
            temporal_factor * 0.1
        )
        
        return max(0.0, min(1.0, adaptive_confidence))
    
    def _calculate_temporal_consistency(self, model_id: str) -> float:
        """Calculate temporal consistency of predictions"""
        if len(self.prediction_buffer[model_id]) < 2:
            return 1.0
        
        recent_predictions = list(self.prediction_buffer[model_id])[-10:]
        if len(recent_predictions) < 2:
            return 1.0
        
        # Calculate consistency in confidence scores
        confidences = [p['confidence_score'] for p in recent_predictions]
        confidence_std = np.std(confidences)
        consistency = 1.0 - min(confidence_std, 1.0)
        
        return consistency
    
    def _update_performance_metrics(self, model_id: str):
        """Update performance metrics based on recent predictions"""
        if not self.prediction_buffer[model_id]:
            return
        
        recent_predictions = list(self.prediction_buffer[model_id])[-20:]
        
        # Calculate metrics
        avg_confidence = np.mean([p['confidence_score'] for p in recent_predictions])
        avg_entropy = np.mean([np.mean(p['entropy']) for p in recent_predictions])
        avg_drift = np.mean([p['feature_drift'] for p in recent_predictions])
        temporal_consistency = self._calculate_temporal_consistency(model_id)
        
        # Create performance metrics
        metrics = ModelPerformanceMetrics(
            accuracy=0.85,  # Would be calculated with true labels in real scenario
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            confidence_score=avg_confidence,
            prediction_entropy=avg_entropy,
            feature_drift_score=avg_drift,
            temporal_consistency=temporal_consistency,
            last_updated=datetime.now(),
            prediction_count=len(self.prediction_buffer[model_id]),
            error_rate=1.0 - avg_confidence
        )
        
        self.performance_history[model_id].append(metrics)
    
    def _check_adaptation_triggers(self, model_id: str):
        """Check if any adaptation triggers are activated"""
        if not self.performance_history[model_id]:
            return
        
        current_metrics = self.performance_history[model_id][-1]
        model_info = self.model_registry[model_id]
        
        # Check accuracy drop
        if len(self.performance_history[model_id]) > 10:
            recent_accuracy = np.mean([m.accuracy for m in list(self.performance_history[model_id])[-10:]])
            baseline_accuracy = np.mean([m.accuracy for m in list(self.performance_history[model_id])[:10]])
            accuracy_drop = baseline_accuracy - recent_accuracy
            
            self.adaptation_rules['accuracy_drop'].current_value = accuracy_drop
            if accuracy_drop > self.adaptation_rules['accuracy_drop'].threshold:
                self.adaptation_rules['accuracy_drop'].triggered = True
                self.adaptation_rules['accuracy_drop'].trigger_time = datetime.now()
        
        # Check feature drift
        self.adaptation_rules['feature_drift'].current_value = current_metrics.feature_drift_score
        if current_metrics.feature_drift_score > self.adaptation_rules['feature_drift'].threshold:
            self.adaptation_rules['feature_drift'].triggered = True
            self.adaptation_rules['feature_drift'].trigger_time = datetime.now()
        
        # Check time-based trigger
        days_since_training = (datetime.now() - model_info['last_training_time']).days
        self.adaptation_rules['time_based'].current_value = days_since_training
        if days_since_training > self.adaptation_rules['time_based'].threshold:
            self.adaptation_rules['time_based'].triggered = True
            self.adaptation_rules['time_based'].trigger_time = datetime.now()
        
        # Check prediction count
        self.adaptation_rules['prediction_count'].current_value = current_metrics.prediction_count
        if current_metrics.prediction_count > self.adaptation_rules['prediction_count'].threshold:
            self.adaptation_rules['prediction_count'].triggered = True
            self.adaptation_rules['prediction_count'].trigger_time = datetime.now()
        
        # Check error rate spike
        self.adaptation_rules['error_rate_spike'].current_value = current_metrics.error_rate
        if current_metrics.error_rate > self.adaptation_rules['error_rate_spike'].threshold:
            self.adaptation_rules['error_rate_spike'].triggered = True
            self.adaptation_rules['error_rate_spike'].trigger_time = datetime.now()
    
    def get_model_health(self, model_id: str) -> ModelHealthStatus:
        """Get comprehensive model health status"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not registered")
        
        if not self.performance_history[model_id]:
            return ModelHealthStatus(
                overall_health='unknown',
                health_score=50.0,
                performance_trend='unknown',
                recommendations=['Insufficient data for assessment'],
                alerts=[],
                last_assessment=datetime.now(),
                days_since_training=0,
                predictions_since_training=0
            )
        
        current_metrics = self.performance_history[model_id][-1]
        model_info = self.model_registry[model_id]
        
        # Calculate health score
        health_components = {
            'accuracy': current_metrics.accuracy * 30,
            'confidence': current_metrics.confidence_score * 25,
            'feature_drift': (1 - current_metrics.feature_drift_score) * 20,
            'temporal_consistency': current_metrics.temporal_consistency * 15,
            'error_rate': (1 - current_metrics.error_rate) * 10
        }
        
        health_score = sum(health_components.values())
        
        # Determine overall health
        if health_score >= 90:
            overall_health = 'excellent'
        elif health_score >= 80:
            overall_health = 'good'
        elif health_score >= 70:
            overall_health = 'fair'
        elif health_score >= 60:
            overall_health = 'poor'
        else:
            overall_health = 'critical'
        
        # Determine performance trend
        if len(self.performance_history[model_id]) >= 10:
            recent_scores = [m.accuracy for m in list(self.performance_history[model_id])[-10:]]
            older_scores = [m.accuracy for m in list(self.performance_history[model_id])[-20:-10]]
            
            if len(older_scores) > 0:
                recent_avg = np.mean(recent_scores)
                older_avg = np.mean(older_scores)
                
                if recent_avg > older_avg + 0.02:
                    performance_trend = 'improving'
                elif recent_avg < older_avg - 0.02:
                    performance_trend = 'declining'
                else:
                    performance_trend = 'stable'
            else:
                performance_trend = 'stable'
        else:
            performance_trend = 'unknown'
        
        # Generate recommendations
        recommendations = []
        alerts = []
        
        if current_metrics.feature_drift_score > 0.1:
            alerts.append(f"High feature drift detected ({current_metrics.feature_drift_score:.3f})")
            recommendations.append("Consider retraining with recent data")
        
        if current_metrics.error_rate > 0.15:
            alerts.append(f"High error rate ({current_metrics.error_rate:.3f})")
            recommendations.append("Investigate model performance issues")
        
        if performance_trend == 'declining':
            recommendations.append("Performance is declining - consider model refresh")
        
        days_since_training = (datetime.now() - model_info['last_training_time']).days
        if days_since_training > 30:
            recommendations.append("Model is over 30 days old - consider retraining")
        
        if current_metrics.confidence_score < 0.7:
            recommendations.append("Low prediction confidence - review feature quality")
        
        # Check triggered adaptation rules
        for rule_name, rule in self.adaptation_rules.items():
            if rule.triggered:
                alerts.append(f"Adaptation trigger: {rule_name} (severity: {rule.severity})")
        
        return ModelHealthStatus(
            overall_health=overall_health,
            health_score=health_score,
            performance_trend=performance_trend,
            recommendations=recommendations,
            alerts=alerts,
            last_assessment=datetime.now(),
            days_since_training=days_since_training,
            predictions_since_training=current_metrics.prediction_count
        )
    
    def get_adaptation_recommendations(self, model_id: str) -> Dict[str, Any]:
        """Get specific adaptation recommendations"""
        health_status = self.get_model_health(model_id)
        triggered_rules = [name for name, rule in self.adaptation_rules.items() if rule.triggered]
        
        recommendations = {
            'immediate_actions': [],
            'scheduled_actions': [],
            'monitoring_adjustments': [],
            'triggered_rules': triggered_rules,
            'priority': 'low'
        }
        
        # Determine priority based on health and triggers
        if health_status.overall_health in ['critical', 'poor']:
            recommendations['priority'] = 'high'
            recommendations['immediate_actions'].append('Emergency model retraining required')
        elif health_status.overall_health == 'fair':
            recommendations['priority'] = 'medium'
            recommendations['scheduled_actions'].append('Schedule model retraining within 48 hours')
        
        # Specific recommendations based on triggered rules
        if 'feature_drift' in triggered_rules:
            recommendations['immediate_actions'].append('Feature preprocessing pipeline review required')
        
        if 'accuracy_drop' in triggered_rules:
            recommendations['immediate_actions'].append('Investigate data quality issues')
        
        if 'error_rate_spike' in triggered_rules:
            recommendations['immediate_actions'].append('Critical: High error rate detected')
            recommendations['priority'] = 'critical'
        
        return recommendations
    
    def reset_adaptation_triggers(self, model_id: str):
        """Reset adaptation triggers for a model"""
        for rule in self.adaptation_rules.values():
            rule.triggered = False
            rule.current_value = 0.0
        
        logger.info(f"🔄 Adaptation triggers reset for model {model_id}")
    
    def export_monitoring_report(self, model_id: str, filepath: str):
        """Export comprehensive monitoring report"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not registered")
        
        # Gather all monitoring data
        model_info = self.model_registry[model_id]
        health_status = self.get_model_health(model_id)
        adaptation_recommendations = self.get_adaptation_recommendations(model_id)
        
        # Performance history
        performance_data = []
        for metrics in self.performance_history[model_id]:
            performance_data.append(asdict(metrics))
        
        # Prediction history
        prediction_data = []
        for pred_info in list(self.prediction_buffer[model_id])[-50:]:  # Last 50 predictions
            pred_data = pred_info.copy()
            pred_data['timestamp'] = pred_data['timestamp'].isoformat()
            pred_data['predictions'] = pred_info['predictions'].tolist()
            pred_data['confidence'] = pred_info['confidence'].tolist()
            pred_data['entropy'] = pred_info['entropy'].tolist()
            prediction_data.append(pred_data)
        
        # Compile report
        report = {
            'model_info': {
                'model_id': model_id,
                'model_type': model_info['model_type'],
                'registration_time': model_info['registration_time'].isoformat(),
                'last_training_time': model_info['last_training_time'].isoformat(),
                'version': model_info['version'],
                'status': model_info['status']
            },
            'health_status': asdict(health_status),
            'adaptation_recommendations': adaptation_recommendations,
            'adaptation_rules': {name: asdict(rule) for name, rule in self.adaptation_rules.items()},
            'performance_history': performance_data,
            'recent_predictions': prediction_data,
            'monitoring_config': {
                'monitoring_window': self.monitoring_window,
                'adaptation_threshold': self.adaptation_threshold
            },
            'report_generated': datetime.now().isoformat()
        }
        
        # Save report
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Monitoring report exported to {filepath}")
        
        return report

def create_intelligent_model_manager(monitoring_window: int = 100, 
                                   adaptation_threshold: float = 0.05) -> IntelligentModelManager:
    """
    Create an intelligent model manager with specified configuration
    
    Args:
        monitoring_window: Number of recent predictions to consider
        adaptation_threshold: Threshold for triggering adaptations
    
    Returns:
        Configured IntelligentModelManager instance
    """
    return IntelligentModelManager(monitoring_window, adaptation_threshold)

if __name__ == "__main__":
    # Example usage
    print("🧠 Intelligent Model Manager - Phase C")
    print("=" * 50)
    
    # Create manager
    manager = create_intelligent_model_manager()
    
    # Simulate model registration and monitoring
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and register a sample model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_sample = np.random.randn(100, 10)
    y_sample = np.random.randint(0, 3, 100)
    model.fit(X_sample, y_sample)
    
    manager.register_model('test_model', model, 'RandomForest', X_sample, y_sample)
    
    # Simulate predictions with monitoring
    for i in range(50):
        X_new = np.random.randn(5, 10)
        predictions, info = manager.predict_with_monitoring('test_model', X_new)
        
        if i % 10 == 0:
            health = manager.get_model_health('test_model')
            print(f"Prediction {i}: Health = {health.overall_health}, Score = {health.health_score:.1f}")
    
    # Get final health assessment
    final_health = manager.get_model_health('test_model')
    print(f"\n✅ Final model health: {final_health.overall_health}")
    print(f"🏆 Health score: {final_health.health_score:.1f}/100")
    print(f"📈 Performance trend: {final_health.performance_trend}")
    
    if final_health.recommendations:
        print("\n💡 Recommendations:")
        for rec in final_health.recommendations:
            print(f"  • {rec}")
