"""
Learning Integration Service - Bridges Predictions and Incremental Learning

Responsibilities:
1. Fetch saved predictions from prediction_ai.py
2. Extract learning-relevant data from predictions and draw results
3. Generate training data for model retraining
4. Calculate model performance metrics
5. Generate learning insights and recommendations
6. Create learning events with structured data
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import hashlib


class PredictionLearningExtractor:
    """Extract learning data from predictions and actual results"""
    
    def __init__(self, game: str, predictions_base_dir: str = "predictions"):
        self.game = game
        self.game_folder = game.lower().replace(" ", "_").replace("/", "_")
        self.predictions_dir = Path(predictions_base_dir) / self.game_folder / "prediction_ai"
        
    def get_all_saved_predictions(self) -> List[Dict[str, Any]]:
        """Load all saved predictions for a game"""
        predictions = []
        if self.predictions_dir.exists():
            for file in sorted(self.predictions_dir.glob("*.json"), reverse=True):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        predictions.append({
                            'file': file.name,
                            'filepath': str(file),
                            **data
                        })
                except Exception as e:
                    print(f"Error loading prediction file {file}: {e}")
        return predictions
    
    def get_predictions_by_date(self, target_date: str) -> List[Dict[str, Any]]:
        """Get predictions for a specific date (YYYY-MM-DD format)"""
        all_preds = self.get_all_saved_predictions()
        filtered = []
        
        for pred in all_preds:
            try:
                pred_date = pred.get('next_draw_date', '')
                if pred_date == target_date:
                    filtered.append(pred)
            except:
                continue
        
        return filtered
    
    def calculate_prediction_metrics(self, prediction_sets: List[List[int]], 
                                    actual_results: List[int]) -> Dict[str, Any]:
        """Calculate accuracy metrics for prediction sets vs actual results"""
        metrics = {
            'total_sets': len(prediction_sets),
            'sets_data': [],
            'overall_accuracy_percent': 0,
            'best_match_count': 0,
            'worst_match_count': len(prediction_sets[0]) if prediction_sets else 0,
            'sets_with_matches': 0,
            'average_matches_per_set': 0,
            'accuracy_distribution': {}
        }
        
        if not prediction_sets or not actual_results:
            return metrics
        
        all_matches = []
        
        for idx, pred_set in enumerate(prediction_sets):
            matches = len(set(pred_set) & set(actual_results))
            match_percent = (matches / len(pred_set) * 100) if len(pred_set) > 0 else 0
            
            metrics['sets_data'].append({
                'set_num': idx + 1,
                'numbers': pred_set,
                'matches': matches,
                'accuracy_percent': match_percent,
                'correct_numbers': list(set(pred_set) & set(actual_results)),
                'incorrect_numbers': list(set(pred_set) - set(actual_results))
            })
            
            all_matches.append(matches)
            
            # Track best/worst
            if matches > metrics['best_match_count']:
                metrics['best_match_count'] = matches
            if matches < metrics['worst_match_count']:
                metrics['worst_match_count'] = matches
            
            # Count sets with at least one match
            if matches > 0:
                metrics['sets_with_matches'] += 1
            
            # Track accuracy distribution
            accuracy_bucket = f"{int(match_percent // 10) * 10}-{int(match_percent // 10) * 10 + 10}%"
            metrics['accuracy_distribution'][accuracy_bucket] = metrics['accuracy_distribution'].get(accuracy_bucket, 0) + 1
        
        # Calculate overall metrics
        if all_matches:
            metrics['overall_accuracy_percent'] = np.mean([m / len(prediction_sets[0]) * 100 for m in all_matches])
            metrics['average_matches_per_set'] = np.mean(all_matches)
        
        return metrics
    
    def extract_learning_patterns(self, prediction_sets: List[List[int]], 
                                 actual_results: List[int], 
                                 models_used: List[str]) -> Dict[str, Any]:
        """Extract learning patterns from predictions"""
        patterns = {
            'timestamp': datetime.now().isoformat(),
            'models_used': models_used,
            'prediction_analysis': {
                'total_unique_predicted_numbers': len(set(num for pred_set in prediction_sets for num in pred_set)),
                'most_predicted_numbers': [],
                'least_predicted_numbers': [],
                'predicted_number_frequency': {}
            },
            'match_analysis': {
                'matched_numbers': list(set(num for pred_set in prediction_sets for num in pred_set) & set(actual_results)),
                'missed_numbers': list(set(num for pred_set in prediction_sets for num in pred_set) - set(actual_results)),
                'actual_unmatched_numbers': list(set(actual_results) - set(num for pred_set in prediction_sets for num in pred_set))
            },
            'model_performance': {},
            'features_for_learning': []
        }
        
        # Analyze predicted number frequency
        all_predicted = [num for pred_set in prediction_sets for num in pred_set]
        num_freq = Counter(all_predicted)
        
        patterns['prediction_analysis']['most_predicted_numbers'] = [
            {'number': num, 'frequency': count} for num, count in num_freq.most_common(5)
        ]
        patterns['prediction_analysis']['least_predicted_numbers'] = [
            {'number': num, 'frequency': count} for num, count in num_freq.most_common()[-5:]
        ]
        patterns['prediction_analysis']['predicted_number_frequency'] = dict(num_freq)
        
        # Initialize per-model performance tracking
        for model in models_used:
            patterns['model_performance'][model] = {
                'accuracy_contribution': 0,
                'reliability_score': 0
            }
        
        return patterns
    
    def generate_training_data(self, prediction_metrics: Dict[str, Any],
                              learning_patterns: Dict[str, Any],
                              actual_results: List[int]) -> Dict[str, Any]:
        """Generate structured training data for model retraining"""
        training_data = {
            'timestamp': datetime.now().isoformat(),
            'data_type': 'learning_event',
            'metadata': {
                'source': 'prediction_analysis',
                'game': self.game,
                'models_used': learning_patterns['models_used']
            },
            'features': {
                'overall_accuracy_percent': prediction_metrics['overall_accuracy_percent'],
                'sets_with_matches': prediction_metrics['sets_with_matches'],
                'total_sets': prediction_metrics['total_sets'],
                'best_match_count': prediction_metrics['best_match_count'],
                'average_matches_per_set': prediction_metrics['average_matches_per_set'],
                'unique_predicted_numbers': learning_patterns['prediction_analysis']['total_unique_predicted_numbers'],
                'num_matched': len(learning_patterns['match_analysis']['matched_numbers']),
                'num_missed': len(learning_patterns['match_analysis']['missed_numbers']),
                'num_unmatched_in_draw': len(learning_patterns['match_analysis']['actual_unmatched_numbers'])
            },
            'labels': {
                'matched_numbers': learning_patterns['match_analysis']['matched_numbers'],
                'missed_numbers': learning_patterns['match_analysis']['missed_numbers'],
                'actual_results': actual_results
            },
            'set_details': prediction_metrics['sets_data'],
            'accuracy_distribution': prediction_metrics['accuracy_distribution']
        }
        
        return training_data


class ModelPerformanceAnalyzer:
    """Analyze model performance from learning events"""
    
    def __init__(self, learning_data_dir: str = "data/learning"):
        self.learning_dir = Path(learning_data_dir)
        
    def load_learning_events(self, game: str, days: int = 30) -> pd.DataFrame:
        """Load learning events for a game"""
        game_normalized = game.lower().replace(" ", "_").replace("/", "_")
        log_file = self.learning_dir / game_normalized / "learning_log.csv"
        
        if not log_file.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(log_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        cutoff = datetime.now() - timedelta(days=days)
        return df[df['timestamp'] >= cutoff]
    
    def calculate_model_performance(self, game: str) -> Dict[str, Dict[str, float]]:
        """Calculate per-model performance metrics"""
        events = self.load_learning_events(game, days=90)
        
        if events.empty:
            return {}
        
        performance = {}
        
        for model in events['model'].unique():
            model_events = events[events['model'] == model]
            
            performance[model] = {
                'total_predictions': len(model_events),
                'avg_accuracy_delta': float(model_events['accuracy_delta'].mean()),
                'std_accuracy_delta': float(model_events['accuracy_delta'].std()),
                'max_accuracy_gain': float(model_events['accuracy_delta'].max()),
                'min_accuracy_gain': float(model_events['accuracy_delta'].min()),
                'consistency_score': float(1 - (model_events['accuracy_delta'].std() / (model_events['accuracy_delta'].mean() + 0.001))),
                'trend': self._calculate_trend(model_events)
            }
        
        return performance
    
    def _calculate_trend(self, model_events: pd.DataFrame) -> str:
        """Determine if performance is improving or declining"""
        if len(model_events) < 2:
            return "insufficient_data"
        
        model_events = model_events.sort_values('timestamp')
        recent = model_events.tail(5)
        previous = model_events.head(5)
        
        recent_avg = recent['accuracy_delta'].mean()
        previous_avg = previous['accuracy_delta'].mean()
        
        if recent_avg > previous_avg * 1.1:
            return "improving"
        elif recent_avg < previous_avg * 0.9:
            return "declining"
        else:
            return "stable"
    
    def generate_recommendations(self, game: str) -> Dict[str, Any]:
        """Generate learning and retraining recommendations"""
        performance = self.calculate_model_performance(game)
        events = self.load_learning_events(game, days=30)
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'game': game,
            'summary': {
                'learning_activity_30d': len(events),
                'models_tracked': len(performance),
                'avg_improvement': float(events['accuracy_delta'].mean()) if not events.empty else 0
            },
            'per_model_recommendations': {},
            'overall_recommendations': [],
            'retrain_urgency': 'normal',
            'knowledge_base_update_needed': False
        }
        
        # Analyze each model
        for model_name, metrics in performance.items():
            trend = metrics['trend']
            consistency = metrics['consistency_score']
            avg_gain = metrics['avg_accuracy_delta']
            
            model_recs = {
                'current_trend': trend,
                'consistency_score': consistency,
                'avg_improvement': avg_gain,
                'actions': []
            }
            
            # Generate model-specific recommendations
            if trend == 'declining':
                model_recs['actions'].append("URGENT: Model performance declining - schedule retraining")
                recommendations['retrain_urgency'] = 'urgent'
            elif trend == 'improving':
                model_recs['actions'].append("Good: Model showing improvement - continue current strategy")
            
            if consistency < 0.3:
                model_recs['actions'].append("WARNING: Low consistency - may need parameter tuning")
            
            if avg_gain > 0.05:
                model_recs['actions'].append("Excellent: Strong learning - knowledge base is being effectively utilized")
                recommendations['knowledge_base_update_needed'] = True
            
            recommendations['per_model_recommendations'][model_name] = model_recs
        
        # Overall recommendations
        if len(events) > 50:
            recommendations['overall_recommendations'].append("High volume of learning data available - optimal time for batch retraining")
        
        if recommendations['knowledge_base_update_needed']:
            recommendations['overall_recommendations'].append("Knowledge base should be updated with new learned patterns")
        
        if recommendations['retrain_urgency'] == 'urgent':
            recommendations['overall_recommendations'].insert(0, "PRIORITY: Execute emergency model retraining immediately")
        
        return recommendations


class LearningDataGenerator:
    """Generate training data files for model retraining"""
    
    def __init__(self, game: str, output_dir: str = "data/learning"):
        self.game = game
        self.game_normalized = game.lower().replace(" ", "_").replace("/", "_")
        self.output_dir = Path(output_dir) / self.game_normalized
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_training_data(self, training_data: Dict[str, Any]) -> str:
        """Save training data to JSON and CSV formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = self.output_dir / f"training_data_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Save as CSV for easier analysis
        csv_file = self.output_dir / f"training_data_{timestamp}.csv"
        csv_data = {
            'timestamp': [training_data['timestamp']],
            'models_used': [','.join(training_data['metadata']['models_used'])],
            'overall_accuracy_percent': [training_data['features']['overall_accuracy_percent']],
            'sets_with_matches': [training_data['features']['sets_with_matches']],
            'total_sets': [training_data['features']['total_sets']],
            'best_match_count': [training_data['features']['best_match_count']],
            'average_matches_per_set': [training_data['features']['average_matches_per_set']]
        }
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        
        return str(json_file)
    
    def aggregate_training_data(self, days: int = 7) -> Tuple[pd.DataFrame, int]:
        """Aggregate training data for batch retraining"""
        json_files = sorted(self.output_dir.glob("training_data_*.json"), reverse=True)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        aggregated_data = []
        
        for json_file in json_files:
            try:
                file_timestamp = datetime.fromtimestamp(json_file.stat().st_mtime)
                if file_timestamp < cutoff_date:
                    continue
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    aggregated_data.append(data['features'])
            except:
                continue
        
        if aggregated_data:
            df = pd.DataFrame(aggregated_data)
            return df, len(aggregated_data)
        
        return pd.DataFrame(), 0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of available training data"""
        json_files = list(self.output_dir.glob("training_data_*.json"))
        csv_files = list(self.output_dir.glob("training_data_*.csv"))
        
        total_size_mb = sum(f.stat().st_size for f in json_files + csv_files) / (1024 * 1024)
        
        # Get latest timestamp
        latest_timestamp = None
        if json_files:
            latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, 'r') as f:
                data = json.load(f)
                latest_timestamp = data.get('timestamp')
        
        return {
            'total_json_files': len(json_files),
            'total_csv_files': len(csv_files),
            'total_size_mb': round(total_size_mb, 2),
            'latest_update': latest_timestamp,
            'ready_for_retraining': len(json_files) >= 5
        }
