from typing import Any, List, Dict, Optional, Tuple
import os
import json
from datetime import date, datetime
import random
import logging

# Import the advanced prediction engine
try:
    from .advanced_engine import PredictionEngine, PredictionConfig, PredictionResult, PredictionMode
except ImportError:
    # Fallback for direct imports
    try:
        from advanced_engine import PredictionEngine, PredictionConfig, PredictionResult, PredictionMode
    except ImportError:
        # If advanced engine not available, we'll use fallback methods
        PredictionEngine = None
        PredictionConfig = None
        PredictionResult = None
        PredictionMode = None

logger = logging.getLogger(__name__)


def predict_batch(model: Any, X):
    """Return probabilities if available, otherwise direct predictions."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X)


class Predictor:
    """Enhanced predictor with advanced engine integration"""
    
    def __init__(self):
        """Initialize predictor with advanced engine if available"""
        self.advanced_engine = PredictionEngine() if PredictionEngine else None
        self.fallback_active = False
    
    def predict_with_config(self, config) -> Tuple[Any, bool]:
        """
        Predict using advanced configuration
        Returns: (PredictionResult, is_existing)
        """
        if self.advanced_engine and config:
            try:
                return self.advanced_engine.generate_prediction(config)
            except Exception as e:
                logger.error(f"Advanced engine failed: {e}")
                self.fallback_active = True
        
        # Fallback to simple prediction
        return self._fallback_predict_with_config(config)
    
    def regenerate_with_config(self, config, save_separately: bool = False) -> Tuple[Any, str]:
        """
        Regenerate prediction with advanced options
        Returns: (PredictionResult, action_taken)
        """
        if self.advanced_engine and config:
            try:
                return self.advanced_engine.regenerate_prediction(config, save_separately)
            except Exception as e:
                logger.error(f"Advanced regeneration failed: {e}")
                self.fallback_active = True
        
        # Fallback regeneration
        result, _ = self._fallback_predict_with_config(config)
        action = "regenerated_fallback_separately" if save_separately else "regenerated_fallback"
        return result, action
    
    def _fallback_predict_with_config(self, config) -> Tuple[Any, bool]:
        """Fallback prediction method when advanced engine unavailable"""
        try:
            # Extract basic parameters from config
            game = config.game if hasattr(config, 'game') else 'lotto_max'
            draw_date = config.draw_date if hasattr(config, 'draw_date') else date.today()
            num_sets = config.num_sets if hasattr(config, 'num_sets') else 3
            
            # Use the existing predict method
            prediction_sets = self.predict(game, draw_date, num_sets)
            
            # Create a simple result object
            result = {
                'sets': prediction_sets,
                'confidence_scores': [0.5 + i * 0.1 for i in range(len(prediction_sets))],
                'metadata': {'fallback': True, 'method': 'simple_prediction'},
                'model_info': getattr(config, 'model_info', {}),
                'generation_time': datetime.now(),
                'file_path': ''
            }
            
            return result, False
        
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            # Return minimal fallback
            max_num = 50 if num_sets == 4 else 49
            numbers_per_set = 7 if num_sets == 4 else 6
            
            sets = []
            for i in range(num_sets):
                numbers = sorted(random.sample(range(1, max_num + 1), numbers_per_set))
                sets.append(numbers)
            
            result = {
                'sets': sets,
                'confidence_scores': [0.3] * len(sets),
                'metadata': {'fallback': True, 'method': 'random_fallback'},
                'model_info': {},
                'generation_time': datetime.now(),
                'file_path': ''
            }
            
            return result, False

    def predict(self, game: str, draw_date: date, num_sets: int = 3) -> List[List[int]]:
        """Predict several candidate sets for a given game and draw date.

        num_sets controls how many candidate sets to return (default 3).

        This is a lightweight placeholder implementation. If a champion model is
        registered in models/registry.json, this will attempt to load it and use
        predict or predict_proba to build sets. Otherwise it returns randomized
        candidate sets (deterministic via seed derived from draw_date).
        """
        # Check cache first
        cache_dir = os.path.join("predictions", game)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{draw_date}_{num_sets}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)

        # Try to load champion model
        registry_file = os.path.join("models", "registry.json")
        champion = None
        if os.path.exists(registry_file):
            try:
                with open(registry_file, "r") as f:
                    champ = json.load(f).get("champion")
                    if champ:
                        from ai_lottery_bot.model_manager.manager import load_model
                        try:
                            champion = load_model(champ)
                        except Exception:
                            champion = None
            except Exception:
                champion = None

        # Seed deterministic randomness from draw_date
        seed = int(str(draw_date).replace('-', '')) if isinstance(draw_date, (str,)) else int(draw_date.strftime('%Y%m%d'))
        random.seed(seed)

        predictions = []
        if champion is not None:
            # If model supports predict_proba or predict, create candidate sets by taking top-K features
            try:
                # This is a best-effort placeholder; real implementation needs access to feature mapping
                preds = []
                # If champion has feature importances, sample by importance
                if hasattr(champion, 'feature_importances_'):
                    import numpy as _np
                    fi = champion.feature_importances_
                    idx = _np.argsort(fi)[::-1]
                    top_features = idx[:7]
                    base = [int((i % 50) + 1) for i in top_features]
                    preds.append(base)
                # fallback random sets
                while len(preds) < num_sets:
                    preds.append(sorted(random.sample(range(1, 51), 7)))
                predictions = preds
            except Exception:
                predictions = [sorted(random.sample(range(1, 51), 7)) for _ in range(num_sets)]
        else:
            # No model: return num_sets randomized candidate sets
            predictions = [sorted(random.sample(range(1, 51), 7)) for _ in range(num_sets)]

        # cache results
        with open(cache_file, "w") as f:
            json.dump(predictions, f)

        return predictions

    def cache_predictions(self, game: str, draw_date: date, predictions: list, num_sets: int = 3) -> None:
        """Cache predictions to a JSON file."""
        cache_dir = os.path.join("predictions", game)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{draw_date}_{num_sets}.json")
        with open(cache_file, "w") as f:
            json.dump(predictions, f)
