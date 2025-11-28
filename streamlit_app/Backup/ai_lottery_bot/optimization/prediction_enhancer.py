"""
Phase C: Real-time Prediction Enhancement & Confidence Calibration
================================================================

This module implements real-time prediction enhancement with advanced confidence calibration:
- Dynamic prediction ensemble with real-time weight adjustment
- Confidence calibration using Platt scaling and isotonic regression
- Uncertainty quantification with prediction intervals
- Real-time prediction quality assessment
- Adaptive prediction fusion strategies
- Advanced prediction post-processing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import json
import os
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionQuality:
    """Prediction quality assessment"""
    confidence_score: float
    uncertainty_score: float
    calibration_score: float
    ensemble_agreement: float
    prediction_interval: Tuple[float, float]
    quality_grade: str  # 'A', 'B', 'C', 'D', 'F'
    reliability_score: float
    sharpness_score: float

@dataclass
class EnhancedPrediction:
    """Enhanced prediction with comprehensive metadata"""
    predictions: np.ndarray
    probabilities: np.ndarray
    confidence_scores: np.ndarray
    uncertainty_intervals: np.ndarray
    quality_assessment: PredictionQuality
    ensemble_weights: np.ndarray
    calibrated_probabilities: np.ndarray
    prediction_metadata: Dict[str, Any]

class ConfidenceCalibrator:
    """Advanced confidence calibration system"""
    
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray):
        """Fit calibrator on validation data"""
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            # Use max probability as confidence score
            confidence_scores = np.max(probabilities, axis=1)
            # Binary calibration: correct prediction = 1, incorrect = 0
            predictions = np.argmax(probabilities, axis=1)
            correct = (predictions == true_labels).astype(int)
            self.calibrator.fit(confidence_scores, correct)
        
        elif self.method == 'platt':
            # Implement Platt scaling for multiclass
            from sklearn.linear_model import LogisticRegression
            confidence_scores = np.max(probabilities, axis=1).reshape(-1, 1)
            predictions = np.argmax(probabilities, axis=1)
            correct = (predictions == true_labels).astype(int)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(confidence_scores, correct)
        
        self.is_fitted = True
        logger.info(f"✅ Confidence calibrator fitted using {self.method} method")
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Calibrate confidence scores"""
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning original probabilities")
            return probabilities
        
        confidence_scores = np.max(probabilities, axis=1)
        
        if self.method == 'isotonic':
            calibrated_confidence = self.calibrator.predict(confidence_scores)
        elif self.method == 'platt':
            calibrated_confidence = self.calibrator.predict_proba(
                confidence_scores.reshape(-1, 1)
            )[:, 1]  # Probability of being correct
        
        # Adjust probabilities while maintaining relative ordering
        calibrated_probs = probabilities.copy()
        max_indices = np.argmax(probabilities, axis=1)
        
        for i in range(len(probabilities)):
            # Scale the max probability by calibrated confidence
            scaling_factor = calibrated_confidence[i] / confidence_scores[i]
            calibrated_probs[i] *= scaling_factor
            # Renormalize
            calibrated_probs[i] /= np.sum(calibrated_probs[i])
        
        return calibrated_probs

class UncertaintyQuantifier:
    """Uncertainty quantification system"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def calculate_prediction_intervals(self, predictions: np.ndarray, 
                                     uncertainties: np.ndarray) -> np.ndarray:
        """Calculate prediction intervals"""
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        intervals = np.zeros((len(predictions), 2))
        intervals[:, 0] = predictions - z_score * uncertainties  # Lower bound
        intervals[:, 1] = predictions + z_score * uncertainties  # Upper bound
        
        return intervals
    
    def calculate_entropy_uncertainty(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate predictive uncertainty using entropy"""
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        # Normalize by maximum possible entropy
        max_entropy = np.log(probabilities.shape[1])
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    def calculate_variance_uncertainty(self, ensemble_predictions: List[np.ndarray]) -> np.ndarray:
        """Calculate uncertainty from ensemble variance"""
        if len(ensemble_predictions) < 2:
            return np.zeros(len(ensemble_predictions[0]))
        
        stacked_predictions = np.stack(ensemble_predictions)
        variance = np.var(stacked_predictions, axis=0)
        uncertainty = np.sqrt(variance)
        return uncertainty

class RealTimePredictionEnhancer:
    """
    Real-time prediction enhancement system with advanced calibration and uncertainty quantification
    """
    
    def __init__(self, calibration_method: str = 'isotonic', confidence_level: float = 0.95):
        self.calibrator = ConfidenceCalibrator(calibration_method)
        self.uncertainty_quantifier = UncertaintyQuantifier(confidence_level)
        self.ensemble_weights = None
        self.quality_thresholds = {
            'A': 0.9,   # Excellent
            'B': 0.8,   # Good
            'C': 0.7,   # Fair
            'D': 0.6,   # Poor
            'F': 0.0    # Fail
        }
        
    def fit_calibrator(self, validation_probabilities: np.ndarray, validation_labels: np.ndarray):
        """Fit confidence calibrator on validation data"""
        self.calibrator.fit(validation_probabilities, validation_labels)
    
    def enhance_predictions(self, 
                          model_predictions: List[np.ndarray],
                          model_probabilities: List[np.ndarray],
                          model_weights: Optional[np.ndarray] = None,
                          return_detailed: bool = True) -> Union[np.ndarray, EnhancedPrediction]:
        """
        Enhance predictions using ensemble, calibration, and uncertainty quantification
        
        Args:
            model_predictions: List of prediction arrays from different models
            model_probabilities: List of probability arrays from different models
            model_weights: Optional weights for ensemble (auto-calculated if None)
            return_detailed: Whether to return detailed enhancement results
        
        Returns:
            Enhanced predictions with comprehensive metadata
        """
        
        if len(model_predictions) == 0:
            raise ValueError("No model predictions provided")
        
        # Ensure we have weights
        if model_weights is None:
            model_weights = np.ones(len(model_predictions)) / len(model_predictions)
        else:
            model_weights = np.array(model_weights)
            model_weights = model_weights / np.sum(model_weights)  # Normalize
        
        # Calculate ensemble predictions
        ensemble_probabilities = self._calculate_ensemble_probabilities(
            model_probabilities, model_weights
        )
        ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
        
        # Calculate calibrated probabilities
        calibrated_probabilities = self.calibrator.calibrate(ensemble_probabilities)
        
        # Calculate confidence scores
        confidence_scores = np.max(calibrated_probabilities, axis=1)
        
        # Calculate uncertainty
        entropy_uncertainty = self.uncertainty_quantifier.calculate_entropy_uncertainty(
            calibrated_probabilities
        )
        
        # Calculate ensemble agreement
        ensemble_agreement = self._calculate_ensemble_agreement(model_predictions)
        
        # Calculate prediction intervals
        prediction_intervals = self.uncertainty_quantifier.calculate_prediction_intervals(
            ensemble_predictions, entropy_uncertainty
        )
        
        # Assess prediction quality
        quality_assessment = self._assess_prediction_quality(
            confidence_scores, entropy_uncertainty, ensemble_agreement, calibrated_probabilities
        )
        
        if not return_detailed:
            return ensemble_predictions
        
        # Create enhanced prediction object
        enhanced_prediction = EnhancedPrediction(
            predictions=ensemble_predictions,
            probabilities=ensemble_probabilities,
            confidence_scores=confidence_scores,
            uncertainty_intervals=prediction_intervals,
            quality_assessment=quality_assessment,
            ensemble_weights=model_weights,
            calibrated_probabilities=calibrated_probabilities,
            prediction_metadata={
                'n_models': len(model_predictions),
                'calibration_method': self.calibrator.method,
                'confidence_level': self.uncertainty_quantifier.confidence_level,
                'timestamp': datetime.now().isoformat(),
                'enhancement_version': '3.0.0'
            }
        )
        
        return enhanced_prediction
    
    def _calculate_ensemble_probabilities(self, model_probabilities: List[np.ndarray], 
                                        weights: np.ndarray) -> np.ndarray:
        """Calculate weighted ensemble probabilities"""
        if len(model_probabilities) == 1:
            return model_probabilities[0]
        
        # Stack probabilities and apply weights
        stacked_probs = np.stack(model_probabilities)
        weighted_probs = np.average(stacked_probs, axis=0, weights=weights)
        
        # Ensure probabilities sum to 1
        weighted_probs = weighted_probs / np.sum(weighted_probs, axis=1, keepdims=True)
        
        return weighted_probs
    
    def _calculate_ensemble_agreement(self, model_predictions: List[np.ndarray]) -> np.ndarray:
        """Calculate agreement between ensemble models"""
        if len(model_predictions) == 1:
            return np.ones(len(model_predictions[0]))
        
        stacked_predictions = np.stack(model_predictions)
        
        # Calculate agreement as fraction of models that agree with majority vote
        majority_vote = stats.mode(stacked_predictions, axis=0)[0].flatten()
        agreement_scores = np.mean(stacked_predictions == majority_vote, axis=0)
        
        return agreement_scores
    
    def _assess_prediction_quality(self, confidence_scores: np.ndarray, 
                                 uncertainty_scores: np.ndarray,
                                 ensemble_agreement: np.ndarray,
                                 probabilities: np.ndarray) -> PredictionQuality:
        """Assess overall prediction quality"""
        
        # Calculate reliability score (how well calibrated)
        reliability_score = np.mean(confidence_scores)
        
        # Calculate sharpness score (how decisive the predictions are)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        max_entropy = np.log(probabilities.shape[1])
        sharpness_score = 1.0 - np.mean(entropy) / max_entropy
        
        # Calculate calibration score (based on confidence and uncertainty consistency)
        calibration_score = 1.0 - np.mean(np.abs(confidence_scores - (1 - uncertainty_scores)))
        
        # Overall quality score
        quality_components = {
            'confidence': np.mean(confidence_scores) * 0.3,
            'uncertainty': (1 - np.mean(uncertainty_scores)) * 0.25,
            'agreement': np.mean(ensemble_agreement) * 0.25,
            'calibration': calibration_score * 0.2
        }
        
        overall_score = sum(quality_components.values())
        
        # Assign quality grade
        quality_grade = 'F'
        for grade, threshold in sorted(self.quality_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                quality_grade = grade
                break
        
        # Calculate prediction intervals
        avg_uncertainty = np.mean(uncertainty_scores)
        prediction_interval = (
            np.mean(confidence_scores) - 1.96 * avg_uncertainty,
            np.mean(confidence_scores) + 1.96 * avg_uncertainty
        )
        
        return PredictionQuality(
            confidence_score=np.mean(confidence_scores),
            uncertainty_score=np.mean(uncertainty_scores),
            calibration_score=calibration_score,
            ensemble_agreement=np.mean(ensemble_agreement),
            prediction_interval=prediction_interval,
            quality_grade=quality_grade,
            reliability_score=reliability_score,
            sharpness_score=sharpness_score
        )
    
    def optimize_ensemble_weights_realtime(self, 
                                         model_probabilities: List[np.ndarray],
                                         recent_performance: List[float]) -> np.ndarray:
        """Optimize ensemble weights based on recent performance"""
        
        if len(recent_performance) != len(model_probabilities):
            # Use equal weights if performance data doesn't match
            return np.ones(len(model_probabilities)) / len(model_probabilities)
        
        # Convert performance to weights (higher performance = higher weight)
        performance_array = np.array(recent_performance)
        
        # Apply softmax to convert to normalized weights
        exp_performance = np.exp(performance_array - np.max(performance_array))
        weights = exp_performance / np.sum(exp_performance)
        
        # Smooth weights to prevent extreme values
        min_weight = 0.1 / len(weights)
        weights = np.maximum(weights, min_weight)
        weights = weights / np.sum(weights)  # Renormalize
        
        self.ensemble_weights = weights
        return weights
    
    def evaluate_prediction_quality(self, enhanced_prediction: EnhancedPrediction) -> Dict[str, Any]:
        """Evaluate and explain prediction quality"""
        
        quality = enhanced_prediction.quality_assessment
        
        evaluation = {
            'overall_grade': quality.quality_grade,
            'overall_score': (
                quality.confidence_score * 0.3 +
                (1 - quality.uncertainty_score) * 0.25 +
                quality.ensemble_agreement * 0.25 +
                quality.calibration_score * 0.2
            ),
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
            'detailed_scores': {
                'confidence': quality.confidence_score,
                'uncertainty': quality.uncertainty_score,
                'calibration': quality.calibration_score,
                'agreement': quality.ensemble_agreement,
                'reliability': quality.reliability_score,
                'sharpness': quality.sharpness_score
            }
        }
        
        # Identify strengths
        if quality.confidence_score > 0.8:
            evaluation['strengths'].append("High prediction confidence")
        if quality.ensemble_agreement > 0.8:
            evaluation['strengths'].append("Strong model agreement")
        if quality.calibration_score > 0.8:
            evaluation['strengths'].append("Well-calibrated predictions")
        if quality.sharpness_score > 0.7:
            evaluation['strengths'].append("Decisive predictions")
        
        # Identify weaknesses
        if quality.confidence_score < 0.6:
            evaluation['weaknesses'].append("Low prediction confidence")
        if quality.uncertainty_score > 0.4:
            evaluation['weaknesses'].append("High prediction uncertainty")
        if quality.ensemble_agreement < 0.6:
            evaluation['weaknesses'].append("Poor model agreement")
        if quality.calibration_score < 0.6:
            evaluation['weaknesses'].append("Poor calibration")
        
        # Generate recommendations
        if quality.uncertainty_score > 0.3:
            evaluation['recommendations'].append("Consider collecting more training data")
        if quality.ensemble_agreement < 0.7:
            evaluation['recommendations'].append("Review model diversity in ensemble")
        if quality.calibration_score < 0.7:
            evaluation['recommendations'].append("Retrain calibration on recent data")
        if quality.confidence_score < 0.7:
            evaluation['recommendations'].append("Investigate feature quality")
        
        return evaluation
    
    def save_enhancement_results(self, enhanced_prediction: EnhancedPrediction, filepath: str):
        """Save enhancement results to file"""
        
        # Convert to serializable format
        results = {
            'predictions': enhanced_prediction.predictions.tolist(),
            'probabilities': enhanced_prediction.probabilities.tolist(),
            'confidence_scores': enhanced_prediction.confidence_scores.tolist(),
            'uncertainty_intervals': enhanced_prediction.uncertainty_intervals.tolist(),
            'ensemble_weights': enhanced_prediction.ensemble_weights.tolist(),
            'calibrated_probabilities': enhanced_prediction.calibrated_probabilities.tolist(),
            'quality_assessment': {
                'confidence_score': enhanced_prediction.quality_assessment.confidence_score,
                'uncertainty_score': enhanced_prediction.quality_assessment.uncertainty_score,
                'calibration_score': enhanced_prediction.quality_assessment.calibration_score,
                'ensemble_agreement': enhanced_prediction.quality_assessment.ensemble_agreement,
                'prediction_interval': enhanced_prediction.quality_assessment.prediction_interval,
                'quality_grade': enhanced_prediction.quality_assessment.quality_grade,
                'reliability_score': enhanced_prediction.quality_assessment.reliability_score,
                'sharpness_score': enhanced_prediction.quality_assessment.sharpness_score
            },
            'metadata': enhanced_prediction.prediction_metadata
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Enhancement results saved to {filepath}")

def create_prediction_enhancer(calibration_method: str = 'isotonic', 
                             confidence_level: float = 0.95) -> RealTimePredictionEnhancer:
    """
    Create a real-time prediction enhancer with specified configuration
    
    Args:
        calibration_method: 'isotonic' or 'platt'
        confidence_level: Confidence level for prediction intervals
    
    Returns:
        Configured RealTimePredictionEnhancer instance
    """
    return RealTimePredictionEnhancer(calibration_method, confidence_level)

if __name__ == "__main__":
    # Example usage
    print("🚀 Real-time Prediction Enhancer - Phase C")
    print("=" * 50)
    
    # Create enhancer
    enhancer = create_prediction_enhancer()
    
    # Simulate ensemble predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    # Create sample predictions from 3 models
    model1_probs = np.random.dirichlet(np.ones(n_classes), n_samples)
    model2_probs = np.random.dirichlet(np.ones(n_classes), n_samples)
    model3_probs = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    model1_preds = np.argmax(model1_probs, axis=1)
    model2_preds = np.argmax(model2_probs, axis=1)
    model3_preds = np.argmax(model3_probs, axis=1)
    
    # Simulate validation data for calibration
    val_probs = np.random.dirichlet(np.ones(n_classes), 50)
    val_labels = np.random.randint(0, n_classes, 50)
    enhancer.fit_calibrator(val_probs, val_labels)
    
    # Enhance predictions
    enhanced = enhancer.enhance_predictions(
        [model1_preds, model2_preds, model3_preds],
        [model1_probs, model2_probs, model3_probs],
        model_weights=np.array([0.4, 0.35, 0.25])
    )
    
    # Evaluate quality
    evaluation = enhancer.evaluate_prediction_quality(enhanced)
    
    print(f"✅ Enhanced {len(enhanced.predictions)} predictions")
    print(f"🏆 Quality Grade: {enhanced.quality_assessment.quality_grade}")
    print(f"📊 Confidence: {enhanced.quality_assessment.confidence_score:.3f}")
    print(f"🎯 Ensemble Agreement: {enhanced.quality_assessment.ensemble_agreement:.3f}")
    print(f"⚖️ Calibration Score: {enhanced.quality_assessment.calibration_score:.3f}")
    
    if evaluation['strengths']:
        print(f"\n💪 Strengths: {', '.join(evaluation['strengths'])}")
    if evaluation['recommendations']:
        print(f"💡 Recommendations: {', '.join(evaluation['recommendations'])}")
