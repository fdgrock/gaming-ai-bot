"""
🏆 Phase 2D - Model Leaderboard & Analysis
Comprehensive evaluation of all trained models: Tree Models, Neural Networks, and Ensemble Variants.
Ranks them and generates model cards for production deployment.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ModelCard:
    """Model card for production deployment."""
    model_name: str
    model_type: str
    game: str
    phase: str
    architecture: str
    composite_score: float
    top_5_accuracy: float
    top_10_accuracy: float
    kl_divergence: float
    strength: str
    known_bias: str
    recommended_use: str
    health_score: float
    ensemble_weight: float
    created_at: str
    model_path: str
    variant_index: Optional[int] = None
    seed: Optional[int] = None
    accuracy: Optional[float] = None
    total_samples: Optional[int] = None


class Phase2DLeaderboard:
    """Comprehensive evaluation and ranking of all trained models."""
    
    def __init__(self):
        self.models_root = MODELS_DIR
        self.advanced_models_dir = MODELS_DIR / "advanced"
        
    def _calculate_composite_score(self, top_5_acc: float, kl_div: float) -> float:
        """Calculate composite score: (0.6 * Top-5 Acc) + (0.4 * (1 - KL-Div))."""
        kl_component = max(0, 1 - kl_div)  # Ensure non-negative
        return (0.6 * top_5_acc) + (0.4 * kl_component)
    
    def _get_model_strengths_and_biases(self, model_type: str, game: str, 
                                        metrics: Dict, top_5_acc: float) -> Tuple[str, str]:
        """Generate detailed model strengths and known biases based on metrics."""
        
        model_type_lower = model_type.lower()
        kl_div = metrics.get('kl_divergence', 0.0)
        
        # STRENGTH: Based on accuracy and model type
        if top_5_acc > 0.80:
            strength = f"⭐ Exceptional top-5 accuracy ({top_5_acc:.1%}). Highly reliable number predictions."
        elif top_5_acc > 0.75:
            strength = f"⭐ Excellent top-5 accuracy ({top_5_acc:.1%}). Strong pattern recognition."
        elif top_5_acc > 0.65:
            strength = f"Good top-5 accuracy ({top_5_acc:.1%}). Solid ensemble contributor."
        elif top_5_acc > 0.55:
            strength = f"Moderate accuracy ({top_5_acc:.1%}). Useful in diversified ensemble."
        else:
            strength = f"Limited accuracy ({top_5_acc:.1%}). Best used with stronger models."
        
        # Add model architecture specific strengths
        if model_type_lower in ["xgboost", "catboost", "lightgbm"]:
            tree_name = model_type_lower.upper()
            strength += f" {tree_name} efficiently learns feature interactions and handles non-linear relationships."
        elif model_type_lower == "lstm":
            strength += " LSTM captures temporal dependencies and sequential patterns in historical draws."
        elif model_type_lower == "transformer":
            strength += " Transformer provides self-attention mechanism for multi-scale pattern recognition."
        elif model_type_lower == "cnn":
            strength += " CNN excels at local pattern detection and spatial relationships in number sequences."
        
        # Add calibration quality
        if kl_div < 0.2:
            strength += " Well-calibrated probability distribution."
        elif kl_div < 0.5:
            strength += " Reasonably calibrated predictions."
        
        # KNOWN BIAS/LIMITATIONS: What the model might struggle with
        if top_5_acc < 0.50:
            bias = f"⚠️ Limited predictive power ({top_5_acc:.1%} top-5). Should only be used in large ensembles."
        elif top_5_acc > 0.75:
            bias = "⚠️ May overfit to recent draw patterns. Requires periodic retraining with fresh data."
        else:
            bias = f"⚠️ Moderate performance ({top_5_acc:.1%}). Potential drift with changing number distributions."
        
        # Add calibration issues
        if kl_div > 0.8:
            bias += " Poorly calibrated probabilities - confidence scores may be unreliable."
        elif kl_div > 0.5:
            bias += " Slight probability miscalibration detected."
        
        # Add type-specific potential biases
        if model_type_lower in ["xgboost", "catboost", "lightgbm"]:
            bias += " Tree models may struggle with unseen feature combinations."
        elif model_type_lower == "lstm":
            bias += " LSTM may have vanishing gradient issues on very long sequences."
        elif model_type_lower == "transformer":
            bias += " Transformer attention may focus too heavily on recent draws."
        elif model_type_lower == "cnn":
            bias += " CNN has limited receptive field - may miss long-range dependencies."
        
        return strength, bias
    
    def _get_recommended_use(self, top_5_acc: float, model_type: str, num_variants: int = 1) -> str:
        """Determine optimal usage recommendation."""
        model_type_lower = model_type.lower()
        
        if num_variants > 1:
            return f"🎯 Best used in ensemble. {num_variants} variants available for voting/averaging."
        
        if top_5_acc > 0.75:
            if model_type_lower in ["transformer", "lstm"]:
                return "🎯 Reliable single model or strong ensemble contributor. Neural network excels at complex patterns."
            else:
                return "🎯 Can be used as reliable single model or in ensemble for robustness."
        elif top_5_acc > 0.65:
            return "🎯 Recommended for ensemble use with complementary models (e.g., tree + neural)."
        elif top_5_acc > 0.55:
            return "🎯 Should be combined with at least 2-3 other models in ensemble."
        else:
            return "🎯 Recommended only as part of large ensemble (5+ models) for diversity."
    
    def evaluate_tree_models(self, game: str = None) -> List[Dict]:
        """Evaluate all tree models (Phase 2A) from advanced/{game}/ folder.
        
        Tree models are stored in:
        - models/advanced/{game}/catboost/
        - models/advanced/{game}/lightgbm/
        - models/advanced/{game}/xgboost/
        
        Metadata is stored flat in models/advanced/{game}/ as training_summary.json
        """
        results = []
        
        logger.info("\n🌳 Scanning Tree Models (Phase 2A)...")
        
        # Look in advanced models folder
        for game_folder in self.advanced_models_dir.iterdir():
            if not game_folder.is_dir():
                continue
            
            if game and game.lower().replace(" ", "_") != game_folder.name:
                continue
            
            # Look for training_summary.json in the game folder (flat structure)
            summary_file = game_folder / "training_summary.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    # Parse the new structure: architectures -> {xgboost/catboost/lightgbm} -> models array
                    architectures = summary_data.get('architectures', {})
                    
                    for arch_name, arch_data in architectures.items():
                        if arch_name.lower() not in ['xgboost', 'catboost', 'lightgbm']:
                            continue
                        
                        models_list = arch_data.get('models', [])
                        
                        # If models is a list, process each model
                        if isinstance(models_list, list):
                            for model_data in models_list:
                                try:
                                    metrics = model_data.get('metrics', {})
                                    top_5_acc = float(metrics.get('top_5_accuracy', 0.0))
                                    top_10_acc = float(metrics.get('top_10_accuracy', 0.0))
                                    kl_div = float(metrics.get('kl_divergence', 0.0))
                                    
                                    composite = self._calculate_composite_score(top_5_acc, kl_div)
                                    
                                    strength, bias = self._get_model_strengths_and_biases(
                                        arch_name, game_folder.name, metrics, top_5_acc
                                    )
                                    
                                    position = model_data.get('position', '?')
                                    model_name = f"{arch_name}_position_{position}"
                                    
                                    results.append({
                                        'phase': '2A',
                                        'game': game_folder.name,
                                        'model_name': model_name,
                                        'model_type': arch_name,
                                        'architecture': arch_name.upper(),
                                        'composite_score': composite,
                                        'top_5_accuracy': top_5_acc,
                                        'top_10_accuracy': top_10_acc,
                                        'kl_divergence': kl_div,
                                        'strength': strength,
                                        'known_bias': bias,
                                        'recommended_use': self._get_recommended_use(top_5_acc, arch_name),
                                        'health_score': composite,
                                        'ensemble_weight': max(0.0, composite),
                                        'created_at': summary_data.get('training_timestamp', datetime.now().isoformat()),
                                        'model_path': str(summary_file),
                                        'accuracy': metrics.get('accuracy', 0.0),
                                        'total_samples': summary_data.get('training_samples', 0),
                                        'variant_index': None,
                                        'seed': None
                                    })
                                except Exception as e:
                                    logger.warning(f"Failed to process model in {summary_file}: {e}")
                
                except Exception as e:
                    logger.warning(f"Failed to read {summary_file}: {e}")
        
        logger.info(f"✅ Found {len(results)} tree models")
        return results
    
    def evaluate_neural_models(self, game: str = None) -> List[Dict]:
        """Evaluate all neural network models (Phase 2B) from advanced/{game}/ folder.
        
        Neural networks are stored in:
        - models/advanced/{game}/lstm/
        - models/advanced/{game}/transformer/
        - models/advanced/{game}/cnn/
        
        Metadata is stored flat in models/advanced/{game}/ as:
        - training_summary_lstm.json
        - training_summary_transformer.json
        - training_summary_cnn.json
        """
        results = []
        
        logger.info("\n🧠 Scanning Neural Network Models (Phase 2B)...")
        
        # Look in advanced models folder
        for game_folder in self.advanced_models_dir.iterdir():
            if not game_folder.is_dir():
                continue
            
            if game and game.lower().replace(" ", "_") != game_folder.name:
                continue
            
            # Look for training_summary_*.json files in the game folder (flat structure)
            metadata_files = list(game_folder.glob("training_summary_*.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract model type from filename (training_summary_lstm.json -> lstm)
                    model_type = metadata_file.stem.replace('training_summary_', '')
                    
                    # Handle nested structure where metrics might be under a key or at top level
                    metrics_data = metadata
                    if 'metrics' in metadata:
                        metrics_data = metadata['metrics']
                    
                    top_5_acc = float(metrics_data.get('top_5_accuracy', 0.0))
                    top_10_acc = float(metrics_data.get('top_10_accuracy', 0.0))
                    kl_div = float(metrics_data.get('kl_divergence', 0.0))
                    
                    composite = self._calculate_composite_score(top_5_acc, kl_div)
                    
                    strength, bias = self._get_model_strengths_and_biases(
                        model_type, game_folder.name, metrics_data, top_5_acc
                    )
                    
                    model_name = metadata.get('model_name', f"{model_type}_{game_folder.name}")
                    
                    results.append({
                        'phase': '2B',
                        'game': game_folder.name,
                        'model_name': model_name,
                        'model_type': model_type,
                        'architecture': model_type.upper(),
                        'composite_score': composite,
                        'top_5_accuracy': top_5_acc,
                        'top_10_accuracy': top_10_acc,
                        'kl_divergence': kl_div,
                        'strength': strength,
                        'known_bias': bias,
                        'recommended_use': self._get_recommended_use(top_5_acc, model_type),
                        'health_score': composite,
                        'ensemble_weight': max(0.0, composite),
                        'created_at': metadata.get('training_timestamp', datetime.now().isoformat()),
                        'model_path': str(metadata_file),
                        'accuracy': metrics_data.get('accuracy', 0.0),
                        'total_samples': metrics_data.get('training_samples', 0),
                        'variant_index': None,
                        'seed': None
                    })
                
                except Exception as e:
                    logger.warning(f"Failed to read {metadata_file}: {e}")
        
        logger.info(f"✅ Found {len(results)} neural network models")
        return results
    
    def evaluate_ensemble_variants(self, game: str = None) -> List[Dict]:
        """Evaluate all ensemble variants (Phase 2C) from variant folders.
        
        Variants are stored in:
        - models/advanced/{game}/lstm_variants/ (with lstm_ensemble_summary.json)
        - models/advanced/{game}/transformer_variants/ (with metadata.json)
        
        Metrics for variants come from the summary files in the variant folders.
        """
        results = []
        
        logger.info("\n🎯 Scanning Ensemble Variants (Phase 2C)...")
        
        # Look in advanced game folders for variants
        for game_folder in self.advanced_models_dir.iterdir():
            if not game_folder.is_dir():
                continue
            
            if game and game.lower().replace(" ", "_") != game_folder.name:
                continue
            
            # Look for *_variants folders (lstm_variants, transformer_variants)
            for variant_dir in game_folder.glob("*_variants"):
                if not variant_dir.is_dir():
                    continue
                
                architecture = variant_dir.name.replace("_variants", "").upper()
                architecture_lower = architecture.lower()
                
                # Check for LSTM variants (lstm_ensemble_summary.json)
                if architecture_lower == "lstm":
                    lstm_summary_file = variant_dir / "lstm_ensemble_summary.json"
                    
                    if lstm_summary_file.exists():
                        try:
                            with open(lstm_summary_file, 'r') as f:
                                summary_data = json.load(f)
                            
                            # Get the game-specific variants
                            game_name = game_folder.name
                            games_data = summary_data.get('games', {})
                            
                            if game_name in games_data:
                                game_variants = games_data[game_name].get('variants', [])
                                num_variants = summary_data.get('num_variants', len(game_variants))
                                
                                for variant_data in game_variants:
                                    var_idx = variant_data.get('variant_index', 0)
                                    metrics = variant_data.get('metrics', {})
                                    
                                    top_5_acc = float(metrics.get('top_5_accuracy', 0.0))
                                    top_10_acc = float(metrics.get('top_10_accuracy', 0.0))
                                    kl_div = float(metrics.get('kl_divergence', 0.0))
                                    
                                    composite = self._calculate_composite_score(top_5_acc, kl_div)
                                    
                                    strength, bias = self._get_model_strengths_and_biases(
                                        architecture_lower, game_folder.name, metrics, top_5_acc
                                    )
                                    
                                    seed = variant_data.get('seed', None)
                                    model_name = f"{architecture}_variant_{var_idx + 1}"
                                    if seed:
                                        model_name += f"_seed_{seed}"
                                    
                                    results.append({
                                        'phase': '2C',
                                        'game': game_folder.name,
                                        'model_name': model_name,
                                        'model_type': architecture_lower,
                                        'architecture': f"{architecture} Ensemble",
                                        'composite_score': composite,
                                        'top_5_accuracy': top_5_acc,
                                        'top_10_accuracy': top_10_acc,
                                        'kl_divergence': kl_div,
                                        'strength': strength,
                                        'known_bias': bias,
                                        'recommended_use': self._get_recommended_use(top_5_acc, architecture_lower, num_variants),
                                        'health_score': composite,
                                        'ensemble_weight': max(0.0, composite),
                                        'created_at': variant_data.get('created_at', datetime.now().isoformat()),
                                        'model_path': str(lstm_summary_file),
                                        'accuracy': metrics.get('accuracy', 0.0),
                                        'total_samples': None,
                                        'variant_index': var_idx,
                                        'seed': seed
                                    })
                        
                        except Exception as e:
                            logger.warning(f"Failed to read LSTM variants from {lstm_summary_file}: {e}")
                
                # Check for Transformer variants (metadata.json)
                else:
                    metadata_file = variant_dir / "metadata.json"
                    
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                variant_metadata = json.load(f)
                            
                            # Get metrics from corresponding training_summary_*.json file
                            metrics_file = game_folder / f"training_summary_{architecture_lower}.json"
                            metrics_data = {}
                            
                            if metrics_file.exists():
                                try:
                                    with open(metrics_file, 'r') as f:
                                        metrics_file_data = json.load(f)
                                        metrics_data = metrics_file_data.get('metrics', {})
                                except Exception as e:
                                    logger.warning(f"Failed to read metrics from {metrics_file}: {e}")
                            
                            # Process individual variants from metadata
                            if 'variants' in variant_metadata:
                                num_variants = variant_metadata.get('num_variants', len(variant_metadata['variants']))
                                
                                for var_idx, variant_data in enumerate(variant_metadata['variants']):
                                    # Use metrics from training_summary file for all variants
                                    top_5_acc = float(metrics_data.get('top_5_accuracy', 0.0))
                                    top_10_acc = float(metrics_data.get('top_10_accuracy', 0.0))
                                    kl_div = float(metrics_data.get('kl_divergence', 0.0))
                                    
                                    composite = self._calculate_composite_score(top_5_acc, kl_div)
                                    
                                    strength, bias = self._get_model_strengths_and_biases(
                                        architecture_lower, game_folder.name, metrics_data, top_5_acc
                                    )
                                    
                                    seed = variant_data.get('seed', None)
                                    model_name = f"{architecture}_variant_{var_idx + 1}"
                                    if seed:
                                        model_name += f"_seed_{seed}"
                                    
                                    results.append({
                                        'phase': '2C',
                                        'game': game_folder.name,
                                        'model_name': model_name,
                                        'model_type': architecture_lower,
                                        'architecture': f"{architecture} Ensemble",
                                        'composite_score': composite,
                                        'top_5_accuracy': top_5_acc,
                                        'top_10_accuracy': top_10_acc,
                                        'kl_divergence': kl_div,
                                        'strength': strength,
                                        'known_bias': bias,
                                        'recommended_use': self._get_recommended_use(top_5_acc, architecture_lower, num_variants),
                                        'health_score': composite,
                                        'ensemble_weight': max(0.0, composite),
                                        'created_at': variant_data.get('created_at', datetime.now().isoformat()),
                                        'model_path': str(metadata_file),
                                        'accuracy': metrics_data.get('accuracy', 0.0),
                                        'total_samples': None,
                                        'variant_index': var_idx,
                                        'seed': seed
                                    })
                        
                        except Exception as e:
                            logger.warning(f"Failed to read variant metadata from {metadata_file}: {e}")
        
        logger.info(f"✅ Found {len(results)} ensemble variants")
        return results
    
    def generate_leaderboard(self, game: str = None) -> pd.DataFrame:
        """Generate comprehensive leaderboard of ALL models (Phase 2A + 2B + 2C)."""
        logger.info("\n" + "=" * 100)
        logger.info("🏆 GENERATING COMPREHENSIVE MODEL LEADERBOARD")
        logger.info("=" * 100)
        
        # Evaluate all model types
        tree_models = self.evaluate_tree_models(game)
        neural_models = self.evaluate_neural_models(game)
        ensemble_variants = self.evaluate_ensemble_variants(game)
        
        all_models = tree_models + neural_models + ensemble_variants
        
        if not all_models:
            logger.warning("⚠️ No models found for leaderboard")
            return pd.DataFrame()
        
        # Create DataFrame and sort by composite score
        df = pd.DataFrame(all_models)
        df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        # Reorder columns for display
        display_cols = ['rank', 'phase', 'model_name', 'model_type', 'game', 
                       'composite_score', 'top_5_accuracy', 'top_10_accuracy', 'health_score']
        df_display = df[display_cols].copy()
        
        logger.info(f"\n✅ LEADERBOARD GENERATED - {len(df)} Total Models")
        logger.info("-" * 100)
        logger.info(f"  {'Rank':<5} {'Phase':<8} {'Model':<30} {'Score':<8} {'Top-5':<8} {'Game':<15}")
        logger.info("-" * 100)
        
        for idx, row in df.head(20).iterrows():
            logger.info(f"  {row['rank']:<5} {row['phase']:<8} {row['model_name']:<30} "
                       f"{row['composite_score']:<8.4f} {row['top_5_accuracy']:<8.1%} {row['game']:<15}")
        
        logger.info("-" * 100)
        
        # Summary statistics
        logger.info("\n📊 LEADERBOARD STATISTICS:")
        logger.info(f"  Total Models: {len(df)}")
        logger.info(f"    - Phase 2A (Trees): {len(df[df['phase'] == '2A'])}")
        logger.info(f"    - Phase 2B (Neural): {len(df[df['phase'] == '2B'])}")
        logger.info(f"    - Phase 2C (Variants): {len(df[df['phase'] == '2C'])}")
        logger.info(f"  Average Score: {df['composite_score'].mean():.4f}")
        logger.info(f"  Top Score: {df['composite_score'].max():.4f}")
        logger.info(f"  Score Range: {df['composite_score'].min():.4f} - {df['composite_score'].max():.4f}")
        
        return df
    
    def generate_model_cards(self, leaderboard_df: pd.DataFrame, top_n: int = 15) -> List[ModelCard]:
        """Generate detailed model cards for top performers."""
        logger.info(f"\n{'=' * 100}")
        logger.info(f"🎫 GENERATING MODEL CARDS FOR TOP {top_n} PERFORMERS")
        logger.info(f"{'=' * 100}")
        
        model_cards = []
        
        for idx, row in leaderboard_df.head(top_n).iterrows():
            card = ModelCard(
                model_name=row['model_name'],
                model_type=row['model_type'],
                game=row['game'],
                phase=row['phase'],
                architecture=row['architecture'],
                composite_score=float(row['composite_score']),
                top_5_accuracy=float(row['top_5_accuracy']),
                top_10_accuracy=float(row['top_10_accuracy']),
                kl_divergence=float(row['kl_divergence']),
                strength=row['strength'],
                known_bias=row['known_bias'],
                recommended_use=row['recommended_use'],
                health_score=float(row['health_score']),
                ensemble_weight=float(row['ensemble_weight']),
                created_at=row['created_at'],
                model_path=row['model_path'],
                variant_index=int(row['variant_index']) if pd.notna(row['variant_index']) else None,
                seed=int(row['seed']) if pd.notna(row['seed']) else None,
                accuracy=float(row['accuracy']) if pd.notna(row['accuracy']) else None,
                total_samples=int(row['total_samples']) if pd.notna(row['total_samples']) else None
            )
            model_cards.append(card)
            
            logger.info(f"\n✅ Card {idx + 1}: {card.model_name}")
            logger.info(f"   Phase: {card.phase} | Type: {card.model_type.upper()} | Game: {card.game}")
            logger.info(f"   Score: {card.composite_score:.4f} | Top-5: {card.top_5_accuracy:.2%} | Weight: {card.ensemble_weight:.4f}")
            logger.info(f"   Recommended: {card.recommended_use.split('🎯')[1].strip()}")
        
        logger.info(f"\n{'=' * 100}")
        return model_cards
    
    def save_leaderboard(self, leaderboard_df: pd.DataFrame, game: str = "all"):
        """Save leaderboard to JSON."""
        output_dir = self.advanced_models_dir / "leaderboards"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"leaderboard_{game}_{timestamp}.json"
        
        # Convert to serializable format
        leaderboard_data = leaderboard_df.to_dict('records')
        
        with open(filename, 'w') as f:
            json.dump(leaderboard_data, f, indent=2, default=str)
        
        logger.info(f"\n✅ Leaderboard saved to: {filename}")
        return filename
    
    def save_model_cards(self, model_cards: List[ModelCard], game: str = "all"):
        """Save model cards to JSON."""
        output_dir = self.advanced_models_dir / "model_cards"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"model_cards_{game}_{timestamp}.json"
        
        cards_data = [asdict(card) for card in model_cards]
        
        with open(filename, 'w') as f:
            json.dump(cards_data, f, indent=2, default=str)
        
        logger.info(f"✅ Model cards saved to: {filename}")
        return filename


def main():
    """Generate leaderboard and model cards."""
    leaderboard = Phase2DLeaderboard()
    
    # Generate for all games
    df = leaderboard.generate_leaderboard()
    
    if not df.empty:
        # Save leaderboard
        leaderboard.save_leaderboard(df, "all")
        
        # Generate and save model cards
        cards = leaderboard.generate_model_cards(df, top_n=15)
        leaderboard.save_model_cards(cards, "all")
        
        logger.info("\n" + "=" * 100)
        logger.info("✅ PHASE 2D LEADERBOARD & MODEL CARDS GENERATION COMPLETE!")
        logger.info("=" * 100)
    else:
        logger.warning("No models available for leaderboard generation")


if __name__ == "__main__":
    main()
