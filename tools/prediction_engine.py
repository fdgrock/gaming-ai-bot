"""
🎯 Advanced Prediction Engine
Generates lottery predictions using single model or ensemble approaches with mathematical safeguards.
Implements bias correction, Gumbel-Top-K sampling, and KL divergence monitoring.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime
import logging
import sys
from scipy.special import softmax
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class TraceLog:
    """Detailed trace log for prediction generation."""
    logs: List[Dict] = field(default_factory=list)
    
    def log(self, level: str, category: str, message: str, data: Optional[Dict] = None):
        """Add a log entry."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'category': category,
            'message': message,
            'data': data or {}
        }
        self.logs.append(entry)
    
    def get_formatted_logs(self) -> str:
        """Get formatted log output for display."""
        output = []
        output.append("=" * 100)
        output.append("PREDICTION GENERATION LOG - ADVANCED ENGINE")
        output.append("=" * 100)
        
        for log in self.logs:
            level_icon = "ℹ️" if log['level'] == 'INFO' else "⚠️" if log['level'] == 'WARNING' else "❌"
            output.append(f"{level_icon}  [{log['category']:<20}] {log['message']}")
            
            if log.get('data'):
                for key, value in log['data'].items():
                    if isinstance(value, (list, dict)):
                        output.append(f"        ├─ {key}: {json.dumps(value)}")
                    else:
                        output.append(f"        ├─ {key}: {value}")
        
        output.append("=" * 100)
        return "\n".join(output)


@dataclass
class PredictionResult:
    """Result of a single prediction row."""
    numbers: List[int]
    probabilities: Dict[int, float]
    model_name: str
    prediction_type: str  # 'single' or 'ensemble'
    reasoning: str
    confidence: float
    generated_at: str
    game: str
    trace_log: Optional[TraceLog] = None


class ProbabilityGenerator:
    """Generates raw probability distributions from models."""
    
    def __init__(self, game: str):
        """Initialize with game configuration."""
        self.game = game
        self.game_lower = game.lower().replace(" ", "_").replace("/", "_")
        
        # Game configuration
        self.game_config = {
            "lotto_6_49": {"main_numbers": 6, "number_range": (1, 49), "bonus": 1},
            "lotto_max": {"main_numbers": 7, "number_range": (1, 50), "bonus": 1}
        }
        
        if self.game_lower not in self.game_config:
            raise ValueError(f"Unknown game: {game}")
        
        self.config = self.game_config[self.game_lower]
        self.num_numbers = self.config["number_range"][1]
    
    def generate_uniform_distribution(self) -> np.ndarray:
        """Generate uniform historical baseline distribution."""
        # Uniform distribution - equal probability for all numbers
        return np.ones(self.num_numbers) / self.num_numbers
    
    def generate_mock_model_probabilities(self, model_name: str, seed: int = None) -> np.ndarray:
        """
        Generate mock probabilities from a model.
        In production, this would load the actual model and run inference.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create a mock distribution with some bias towards lower numbers
        probs = np.random.dirichlet(np.ones(self.num_numbers))
        
        # Add slight bias for demonstration
        if "xgboost" in model_name.lower() or "catboost" in model_name.lower():
            # Tree models might have slight bias
            bias = np.linspace(1.0, 0.8, self.num_numbers)
            probs = probs * bias
        elif "lstm" in model_name.lower():
            # LSTM might have different pattern
            bias = np.linspace(0.9, 1.1, self.num_numbers)
            probs = probs * bias
        
        # Normalize to valid probability distribution
        probs = probs / probs.sum()
        return probs
    
    def apply_bias_correction(
        self,
        model_probs: np.ndarray,
        model_health_score: float,
        historical_probs: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply bias correction to model probabilities.
        
        P_corrected(number) = α * P_model(number) + (1-α) * P_historical(number)
        
        Where α (confidence factor) is derived from health_score:
        - Low health score (0.3) → α = 0.3 (more reliance on historical distribution)
        - High health score (0.9) → α = 0.9 (more reliance on model)
        """
        if historical_probs is None:
            historical_probs = self.generate_uniform_distribution()
        
        # Health score directly becomes confidence factor
        # Range [0.0, 1.0] maps to confidence [0.3, 0.9]
        alpha = 0.3 + (0.6 * model_health_score)  # Maps [0, 1] to [0.3, 0.9]
        
        # Weighted combination
        corrected_probs = alpha * model_probs + (1 - alpha) * historical_probs
        
        # Ensure valid probability distribution
        corrected_probs = corrected_probs / corrected_probs.sum()
        return corrected_probs
    
    def enforce_range(self, probs: np.ndarray, num_range: Tuple[int, int] = None) -> np.ndarray:
        """Enforce that only valid numbers have non-zero probability."""
        if num_range is None:
            num_range = self.config["number_range"]
        
        min_num, max_num = num_range
        enforced_probs = np.zeros_like(probs)
        
        # Only valid numbers (1 to max_num)
        for i in range(min_num - 1, max_num):
            enforced_probs[i] = probs[i]
        
        # Renormalize
        if enforced_probs.sum() > 0:
            enforced_probs = enforced_probs / enforced_probs.sum()
        
        return enforced_probs


class SamplingStrategy:
    """Advanced sampling techniques for probability distributions."""
    
    @staticmethod
    def gumbel_top_k(probs: np.ndarray, k: int, seed: int = None) -> Tuple[List[int], np.ndarray]:
        """
        Gumbel-Top-K Trick: Sample k unique items from a probability distribution.
        
        Each item's probability of being selected is proportional to its probability.
        This is the correct way to sample unique items from a categorical distribution.
        
        Algorithm:
        1. For each number, compute: g_i = log(p_i) - log(-log(u_i))
           where u_i is uniform random in (0,1)
        2. Select the k numbers with highest g_i values
        3. These k numbers are guaranteed to be unique
        
        Returns:
            tuple: (sampled_numbers, sorted_probabilities_of_selected)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Ensure probabilities are valid
        probs = np.array(probs).astype(float)
        probs = probs / probs.sum()
        
        # Gumbel noise
        uniform_noise = np.random.uniform(0, 1, len(probs))
        gumbel_noise = -np.log(-np.log(uniform_noise + 1e-10) + 1e-10)
        
        # Log probabilities + Gumbel noise
        log_probs = np.log(probs + 1e-10)
        scores = log_probs + gumbel_noise
        
        # Get top-k indices (numbers)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Convert from 0-indexed to 1-indexed (lottery numbers)
        sampled_numbers = [int(idx + 1) for idx in top_k_indices]
        
        # Get probabilities of selected numbers
        selected_probs = probs[top_k_indices]
        
        return sampled_numbers, selected_probs
    
    @staticmethod
    def multinomial_sampling(probs: np.ndarray, k: int, seed: int = None) -> List[int]:
        """
        Alternative: Multinomial sampling (slower but simpler).
        Useful for validation and comparison.
        """
        if seed is not None:
            np.random.seed(seed)
        
        probs = np.array(probs).astype(float)
        probs = probs / probs.sum()
        
        # Sample k unique indices
        sampled_indices = np.random.choice(
            len(probs),
            size=k,
            replace=False,
            p=probs
        )
        
        return [int(idx + 1) for idx in sampled_indices]


class EnsembleWeighting:
    """Handles ensemble weighting and probability fusion."""
    
    @staticmethod
    def fuse_probabilities(
        model_probs_list: List[Tuple[np.ndarray, float]],
        method: str = "weighted_average"
    ) -> np.ndarray:
        """
        Fuse multiple model probability distributions.
        
        Args:
            model_probs_list: List of (probabilities, weight) tuples
            method: Fusion method ('weighted_average', 'geometric_mean', etc.)
        
        Returns:
            Fused probability distribution
        """
        if not model_probs_list:
            raise ValueError("No models provided for fusion")
        
        if method == "weighted_average":
            # P_ensemble = sum(w_i * P_i) / sum(w_i)
            total_weight = sum(weight for _, weight in model_probs_list)
            
            if total_weight == 0:
                raise ValueError("Total weight is zero")
            
            fused_probs = np.zeros_like(model_probs_list[0][0])
            
            for probs, weight in model_probs_list:
                fused_probs += weight * probs
            
            fused_probs = fused_probs / total_weight
            
        elif method == "geometric_mean":
            # Product of weighted probabilities
            fused_probs = np.ones_like(model_probs_list[0][0])
            
            for probs, weight in model_probs_list:
                # Raise each probability to its weight power
                fused_probs *= np.power(probs + 1e-10, weight)
            
            # Normalize
            fused_probs = fused_probs / fused_probs.sum()
        
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        return fused_probs
    
    @staticmethod
    def check_divergence(
        ensemble_probs: np.ndarray,
        historical_probs: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[float, bool, float]:
        """
        Check KL divergence between ensemble and historical distribution.
        If divergence is too high, nudge ensemble towards historical.
        
        Returns:
            tuple: (kl_divergence, needs_correction, correction_strength)
        """
        # Calculate KL divergence
        kl_div = entropy(ensemble_probs, historical_probs)
        
        needs_correction = kl_div > threshold
        
        # Correction strength: how much to nudge towards historical
        # Higher divergence = stronger correction
        if needs_correction:
            correction_strength = min(0.5, (kl_div - threshold) / 2.0)
        else:
            correction_strength = 0.0
        
        return kl_div, needs_correction, correction_strength
    
    @staticmethod
    def apply_divergence_correction(
        ensemble_probs: np.ndarray,
        historical_probs: np.ndarray,
        correction_strength: float
    ) -> np.ndarray:
        """
        Gently nudge ensemble towards historical distribution.
        """
        corrected_probs = (
            (1 - correction_strength) * ensemble_probs +
            correction_strength * historical_probs
        )
        
        # Normalize
        corrected_probs = corrected_probs / corrected_probs.sum()
        return corrected_probs


class PredictionEngine:
    """Main engine for generating lottery predictions."""
    
    def __init__(self, game: str):
        """Initialize prediction engine."""
        self.game = game
        self.game_lower = game.lower().replace(" ", "_").replace("/", "_")
        self.prob_gen = ProbabilityGenerator(game)
        self.sampling = SamplingStrategy()
        self.ensemble = EnsembleWeighting()
        
        self.game_config = self.prob_gen.game_config[self.game_lower]
        self.num_numbers = self.game_config["number_range"][1]
    
    def predict_single_model(
        self,
        model_name: str,
        health_score: float,
        num_predictions: int = 1,
        seed: int = None
    ) -> List[PredictionResult]:
        """
        Generate predictions using a single model with detailed logging.
        
        Process:
        1. Generate raw probabilities from model
        2. Apply bias correction based on health score
        3. Enforce range constraints
        4. Sample using Gumbel-Top-K
        """
        results = []
        trace = TraceLog()
        
        trace.log('INFO', 'STARTED', f'Single model prediction: {model_name}, health_score={health_score:.3f}')
        trace.log('INFO', 'CONFIG', f'Game: {self.game}, Predictions: {num_predictions}')
        
        for pred_idx in range(num_predictions):
            # Use different seed for each prediction
            current_seed = None if seed is None else seed + pred_idx
            trace.log('INFO', 'SET_START', f'Generating prediction set {pred_idx + 1}/{num_predictions}', {'seed': current_seed})
            
            # 1. Generate raw probabilities
            model_probs = self.prob_gen.generate_mock_model_probabilities(
                model_name,
                seed=current_seed
            )
            top_indices = np.argsort(model_probs)[-5:]
            trace.log('INFO', 'MODEL_PROBS', f'Generated model probabilities', {
                'top_5_numbers': (top_indices + 1).tolist(),
                'top_5_probs': model_probs[top_indices].tolist()
            })
            
            # 2. Apply bias correction
            historical_probs = self.prob_gen.generate_uniform_distribution()
            corrected_probs = self.prob_gen.apply_bias_correction(
                model_probs,
                health_score,
                historical_probs
            )
            alpha = 0.3 + (0.6 * health_score)
            trace.log('INFO', 'BIAS_CORRECTION', f'Applied bias correction', {
                'alpha': float(f'{alpha:.3f}'),
                'health_score': float(f'{health_score:.3f}')
            })
            
            # 3. Enforce range
            safeguarded_probs = self.prob_gen.enforce_range(corrected_probs)
            trace.log('INFO', 'RANGE_ENFORCE', f'Enforced number range [1, {self.prob_gen.num_numbers}]')
            
            # 4. Sample using Gumbel-Top-K
            sampled_numbers, selected_probs = self.sampling.gumbel_top_k(
                safeguarded_probs,
                k=self.game_config["main_numbers"],
                seed=current_seed
            )
            
            trace.log('INFO', 'SAMPLING', f'Gumbel-Top-K sampling', {
                'numbers': sorted(sampled_numbers),
                'probabilities': [float(f'{p:.6f}') for p in selected_probs],
                'method': 'Gumbel-Top-K'
            })
            
            # Calculate confidence (average probability of selected numbers)
            confidence = float(np.mean(selected_probs))
            trace.log('INFO', 'CONFIDENCE', f'Calculated confidence score', {
                'confidence': float(f'{confidence:.6f}'),
                'calculation': 'mean(selected_probabilities)'
            })
            
            # Create reasoning
            reasoning = (
                f"This prediction was generated by {model_name} "
                f"(health score: {health_score:.3f}). "
                f"A post-processing correction was applied (α={alpha:.2f}) "
                f"to balance model predictions with historical distribution, "
                f"ensuring all numbers have a fair chance of being selected."
            )
            
            # Create probability dict
            prob_dict = {
                i + 1: float(safeguarded_probs[i]) 
                for i in range(self.prob_gen.num_numbers)
            }
            
            result = PredictionResult(
                numbers=sorted(sampled_numbers),
                probabilities=prob_dict,
                model_name=model_name,
                prediction_type="single",
                reasoning=reasoning,
                confidence=confidence,
                generated_at=datetime.now().isoformat(),
                game=self.game,
                trace_log=trace
            )
            
            trace.log('INFO', 'SET_COMPLETE', f'Prediction set {pred_idx + 1} complete', {
                'numbers': sorted(sampled_numbers),
                'confidence': float(f'{confidence:.6f}')
            })
            
            results.append(result)
        
        trace.log('INFO', 'COMPLETED', f'Generated {num_predictions} predictions successfully')
        return results
    
    def predict_ensemble(
        self,
        model_weights: Dict[str, float],  # {model_name: health_score}
        num_predictions: int = 1,
        seed: int = None
    ) -> List[PredictionResult]:
        """
        Generate predictions using an ensemble of models with detailed logging.
        
        Process:
        1. Generate probabilities from each model
        2. Fuse using weighted average (weights = health scores)
        3. Check KL divergence
        4. Apply divergence correction if needed
        5. Sample using Gumbel-Top-K
        """
        results = []
        trace = TraceLog()
        
        if not model_weights:
            raise ValueError("No models provided for ensemble")
        
        trace.log('INFO', 'STARTED', f'Ensemble prediction with {len(model_weights)} models')
        trace.log('INFO', 'CONFIG', f'Game: {self.game}, Predictions: {num_predictions}', {
            'models': list(model_weights.keys()),
            'health_scores': {k: float(f'{v:.3f}') for k, v in model_weights.items()}
        })
        
        for pred_idx in range(num_predictions):
            current_seed = None if seed is None else seed + pred_idx
            trace.log('INFO', 'SET_START', f'Generating ensemble prediction set {pred_idx + 1}/{num_predictions}', {'seed': current_seed})
            
            # 1. Generate probabilities from each model
            model_probs_list = []
            model_logs = {}
            
            for model_name, health_score in model_weights.items():
                model_probs = self.prob_gen.generate_mock_model_probabilities(
                    model_name,
                    seed=current_seed
                )
                
                # Apply bias correction to each model
                historical_probs = self.prob_gen.generate_uniform_distribution()
                corrected_probs = self.prob_gen.apply_bias_correction(
                    model_probs,
                    health_score,
                    historical_probs
                )
                
                top_indices = np.argsort(corrected_probs)[-3:]
                model_logs[model_name] = {
                    'health_score': float(f'{health_score:.3f}'),
                    'top_3_numbers': (top_indices + 1).tolist(),
                    'top_3_probs': [float(f'{corrected_probs[i]:.6f}') for i in top_indices]
                }
                
                model_probs_list.append((corrected_probs, health_score))
            
            trace.log('INFO', 'MODEL_CONTRIBUTIONS', f'Individual model predictions', model_logs)
            
            # 2. Fuse probabilities
            ensemble_probs = self.ensemble.fuse_probabilities(
                model_probs_list,
                method="weighted_average"
            )
            trace.log('INFO', 'FUSION', f'Fused model probabilities using weighted average')
            
            # 3. Check divergence
            historical_probs = self.prob_gen.generate_uniform_distribution()
            kl_div, needs_correction, correction_strength = self.ensemble.check_divergence(
                ensemble_probs,
                historical_probs,
                threshold=0.5
            )
            
            trace.log('INFO', 'DIVERGENCE_CHECK', f'KL divergence monitoring', {
                'kl_divergence': float(f'{kl_div:.6f}'),
                'threshold': 0.5,
                'needs_correction': needs_correction,
                'correction_strength': float(f'{correction_strength:.6f}') if needs_correction else 0.0
            })
            
            # 4. Apply divergence correction if needed
            if needs_correction:
                ensemble_probs = self.ensemble.apply_divergence_correction(
                    ensemble_probs,
                    historical_probs,
                    correction_strength
                )
                trace.log('INFO', 'CORRECTION_APPLIED', f'Applied divergence correction to ensemble')
            
            # 5. Enforce range
            safeguarded_probs = self.prob_gen.enforce_range(ensemble_probs)
            trace.log('INFO', 'RANGE_ENFORCE', f'Enforced number range [1, {self.prob_gen.num_numbers}]')
            
            # 6. Sample using Gumbel-Top-K
            sampled_numbers, selected_probs = self.sampling.gumbel_top_k(
                safeguarded_probs,
                k=self.game_config["main_numbers"],
                seed=current_seed
            )
            
            trace.log('INFO', 'SAMPLING', f'Gumbel-Top-K sampling', {
                'numbers': sorted(sampled_numbers),
                'probabilities': [float(f'{p:.6f}') for p in selected_probs]
            })
            
            # Calculate confidence
            confidence = float(np.mean(selected_probs))
            trace.log('INFO', 'CONFIDENCE', f'Calculated ensemble confidence', {
                'confidence': float(f'{confidence:.6f}'),
                'method': 'mean(selected_probabilities)'
            })
            
            # Get top contributing models
            sorted_models = sorted(
                model_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_models = [m[0] for m in sorted_models[:3]]
            
            # Create reasoning
            reasoning = (
                f"This prediction is a consensus from our AI ensemble of {len(model_weights)} models. "
                f"The most influential models were: {', '.join(top_models)}. "
                f"The numbers were chosen based on weighted model predictions "
                f"with statistical safeguards applied "
                f"(KL divergence: {kl_div:.4f}, correction applied: {needs_correction}). "
                f"Ensemble confidence: {confidence:.3f}"
            )
            
            # Create probability dict
            prob_dict = {
                i + 1: float(safeguarded_probs[i]) 
                for i in range(self.prob_gen.num_numbers)
            }
            
            result = PredictionResult(
                numbers=sorted(sampled_numbers),
                probabilities=prob_dict,
                model_name=f"Ensemble ({len(model_weights)} models)",
                prediction_type="ensemble",
                reasoning=reasoning,
                confidence=confidence,
                generated_at=datetime.now().isoformat(),
                game=self.game,
                trace_log=trace
            )
            
            trace.log('INFO', 'SET_COMPLETE', f'Ensemble prediction set {pred_idx + 1} complete', {
                'numbers': sorted(sampled_numbers),
                'confidence': float(f'{confidence:.6f}')
            })
            
            results.append(result)
        
        trace.log('INFO', 'COMPLETED', f'Generated {num_predictions} ensemble predictions successfully')
        return results


def main():
    """Example usage."""
    # Test single model prediction
    engine = PredictionEngine("Lotto Max")
    
    print("\n" + "="*80)
    print("SINGLE MODEL PREDICTION")
    print("="*80)
    
    results = engine.predict_single_model(
        model_name="catboost_position_5",
        health_score=0.85,
        num_predictions=2,
        seed=42
    )
    
    for i, result in enumerate(results):
        print(f"\nPrediction {i+1}:")
        print(f"  Numbers: {result.numbers}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Reasoning: {result.reasoning}")
    
    # Test ensemble prediction
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTION")
    print("="*80)
    
    ensemble_weights = {
        "catboost_position_5": 0.85,
        "transformer_lotto_max": 0.75,
        "lstm_variant_1": 0.80
    }
    
    results = engine.predict_ensemble(
        model_weights=ensemble_weights,
        num_predictions=2,
        seed=42
    )
    
    for i, result in enumerate(results):
        print(f"\nPrediction {i+1}:")
        print(f"  Numbers: {result.numbers}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Reasoning: {result.reasoning}")


if __name__ == "__main__":
    main()
