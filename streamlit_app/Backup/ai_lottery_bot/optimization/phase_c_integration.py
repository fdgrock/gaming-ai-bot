"""
Phase C: Advanced Model Optimization Integration
==============================================

This module integrates Phase C optimization capabilities into the main training pipeline:
- Hyperparameter optimization during training
- Intelligent model management
- Real-time prediction enhancement
- Comprehensive optimization orchestration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
from datetime import datetime
import logging
import joblib

# Import Phase C components
from .advanced_optimizer import (
    AdvancedModelOptimizer, OptimizationConfig, OptimizationResult,
    optimize_model_comprehensive
)
from .intelligent_manager import (
    IntelligentModelManager, ModelPerformanceMetrics, ModelHealthStatus,
    create_intelligent_model_manager
)
from .prediction_enhancer import (
    RealTimePredictionEnhancer, EnhancedPrediction, PredictionQuality,
    create_prediction_enhancer
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhaseC_IntegratedOptimizer:
    """
    Integrated Phase C optimization system that combines all advanced capabilities
    """
    
    def __init__(self, 
                 optimization_config: OptimizationConfig = None,
                 monitoring_window: int = 100,
                 calibration_method: str = 'isotonic'):
        
        # Initialize components
        self.optimizer = AdvancedModelOptimizer(optimization_config or OptimizationConfig())
        self.manager = create_intelligent_model_manager(monitoring_window)
        self.enhancer = create_prediction_enhancer(calibration_method)
        
        # Configuration
        self.optimization_config = optimization_config or OptimizationConfig()
        self.monitoring_window = monitoring_window
        self.calibration_method = calibration_method
        
        # Results storage
        self.optimization_results = {}
        self.model_registry = {}
        
    def optimize_and_train_model(self, 
                                model_type: str,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray = None,
                                y_val: np.ndarray = None,
                                model_id: str = None) -> Tuple[Any, OptimizationResult]:
        """
        Perform comprehensive model optimization and training
        
        Args:
            model_type: 'xgboost', 'lstm', or 'transformer'
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_id: Unique identifier for the model
        
        Returns:
            Tuple of (trained_model, optimization_result)
        """
        
        if model_id is None:
            model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🚀 Starting Phase C optimization for {model_type} model: {model_id}")
        
        # Step 1: Feature optimization
        if self.optimization_config.feature_selection:
            logger.info("🔍 Optimizing feature selection...")
            X_train_optimized, feature_importance = self.optimizer.optimize_feature_selection(
                X_train, y_train
            )
            
            if X_val is not None and self.optimizer.feature_selector is not None:
                X_val_optimized = self.optimizer.feature_selector.transform(X_val)
            else:
                X_val_optimized = X_val
        else:
            X_train_optimized = X_train
            X_val_optimized = X_val
            feature_importance = {}
        
        # Step 2: Hyperparameter optimization
        logger.info("⚙️ Performing hyperparameter optimization...")
        optimization_result = optimize_model_comprehensive(
            model_type, X_train_optimized, y_train, X_val_optimized, y_val, 
            self.optimization_config
        )
        
        # Step 3: Train final model with optimized parameters
        logger.info("🏋️ Training final optimized model...")
        final_model = self._train_final_model(
            model_type, optimization_result.best_params, 
            X_train_optimized, y_train, X_val_optimized, y_val
        )
        
        # Step 4: Register model with intelligent manager
        logger.info("📝 Registering model with intelligent manager...")
        self.manager.register_model(
            model_id, final_model, model_type, X_train_optimized, y_train
        )
        
        # Step 5: Prepare calibration data for prediction enhancer
        if X_val_optimized is not None and y_val is not None:
            logger.info("⚖️ Calibrating prediction enhancer...")
            val_probabilities = self._get_model_probabilities(final_model, X_val_optimized)
            self.enhancer.fit_calibrator(val_probabilities, y_val)
        
        # Store results
        self.optimization_results[model_id] = optimization_result
        self.model_registry[model_id] = {
            'model': final_model,
            'model_type': model_type,
            'feature_selector': self.optimizer.feature_selector,
            'optimization_result': optimization_result,
            'registration_time': datetime.now()
        }
        
        logger.info(f"✅ Phase C optimization completed for {model_id}")
        logger.info(f"🏆 Best score: {optimization_result.best_score:.4f}")
        
        return final_model, optimization_result
    
    def _train_final_model(self, model_type: str, best_params: Dict, 
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None) -> Any:
        """Train final model with optimized parameters"""
        
        if model_type.lower() == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBClassifier(**best_params)
            
            if X_val is not None and y_val is not None:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
        
        elif model_type.lower() == 'lstm':
            try:
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
                
                # Build LSTM model from optimized parameters
                model = self._build_lstm_from_params(best_params, X_train.shape)
                
                # Train model
                if X_val is not None and y_val is not None:
                    model.fit(X_train, y_train, validation_data=(X_val, y_val),
                             epochs=50, verbose=0)
                else:
                    model.fit(X_train, y_train, epochs=50, verbose=0)
                    
            except ImportError:
                logger.warning("TensorFlow not available, using dummy model")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**{'n_estimators': 100, 'random_state': 42})
                model.fit(X_train, y_train)
        
        elif model_type.lower() == 'transformer':
            try:
                import tensorflow as tf
                
                # Build Transformer model from optimized parameters
                model = self._build_transformer_from_params(best_params, X_train.shape)
                
                # Train model
                if X_val is not None and y_val is not None:
                    model.fit(X_train, y_train, validation_data=(X_val, y_val),
                             epochs=30, verbose=0)
                else:
                    model.fit(X_train, y_train, epochs=30, verbose=0)
                    
            except ImportError:
                logger.warning("TensorFlow not available, using dummy model")
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**{'n_estimators': 100, 'random_state': 42})
                model.fit(X_train, y_train)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def _build_lstm_from_params(self, params: Dict, input_shape: Tuple) -> Any:
        """Build LSTM model from optimized parameters"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
            
            model = Sequential()
            
            # Extract parameters
            n_layers = params.get('n_layers', 2)
            use_batch_norm = params.get('use_batch_norm', False)
            
            # Add LSTM layers
            for i in range(n_layers):
                units = params.get(f'lstm_units_{i}', 128)
                dropout = params.get(f'dropout_{i}', 0.2)
                return_sequences = i < n_layers - 1
                
                if i == 0:
                    model.add(LSTM(units, return_sequences=return_sequences,
                                 input_shape=(input_shape[1], input_shape[2])))
                else:
                    model.add(LSTM(units, return_sequences=return_sequences))
                
                if use_batch_norm:
                    model.add(BatchNormalization())
                if dropout > 0:
                    model.add(Dropout(dropout))
            
            # Add dense layers
            n_dense = params.get('n_dense', 1)
            for i in range(n_dense):
                units = params.get(f'dense_units_{i}', 64)
                dropout = params.get(f'dense_dropout_{i}', 0.2)
                
                model.add(Dense(units, activation='relu'))
                if use_batch_norm:
                    model.add(BatchNormalization())
                if dropout > 0:
                    model.add(Dropout(dropout))
            
            # Output layer
            n_classes = params.get('n_classes', 2)
            model.add(Dense(n_classes, activation='softmax'))
            
            # Compile model
            optimizer_name = params.get('optimizer', 'adam')
            learning_rate = params.get('learning_rate', 0.001)
            
            if optimizer_name == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            else:
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
            
            return model
            
        except ImportError:
            raise ImportError("TensorFlow required for LSTM models")
    
    def _build_transformer_from_params(self, params: Dict, input_shape: Tuple) -> Any:
        """Build Transformer model from optimized parameters"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
            
            # Extract parameters
            d_model = params.get('d_model', 128)
            n_heads = params.get('n_heads', 8)
            n_layers = params.get('n_layers', 2)
            dff = params.get('dff', 512)
            dropout_rate = params.get('dropout_rate', 0.1)
            
            # Build model
            inputs = Input(shape=(input_shape[1], input_shape[2]))
            x = inputs
            
            # Transformer blocks
            for _ in range(n_layers):
                # Multi-head attention
                attn_output = MultiHeadAttention(
                    num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout_rate
                )(x, x)
                x = LayerNormalization(epsilon=1e-6)(x + attn_output)
                
                # Feed forward
                ffn_output = Dense(dff, activation='relu')(x)
                ffn_output = Dropout(dropout_rate)(ffn_output)
                ffn_output = Dense(d_model)(ffn_output)
                x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
            
            # Global pooling and output
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = Dropout(dropout_rate)(x)
            
            n_classes = params.get('n_classes', 2)
            outputs = Dense(n_classes, activation='softmax')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile
            learning_rate = params.get('learning_rate', 0.001)
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            
            return model
            
        except ImportError:
            raise ImportError("TensorFlow required for Transformer models")
    
    def _get_model_probabilities(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from model"""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        elif hasattr(model, 'predict'):
            try:
                # Try to get probabilities for neural networks
                probabilities = model.predict(X)
                if len(probabilities.shape) == 2:
                    return probabilities
                else:
                    # Convert single predictions to dummy probabilities
                    n_classes = len(np.unique(probabilities))
                    dummy_probs = np.zeros((len(probabilities), n_classes))
                    dummy_probs[np.arange(len(probabilities)), probabilities] = 0.8
                    dummy_probs += 0.2 / n_classes  # Add some uncertainty
                    return dummy_probs
            except:
                # Fallback for models without probability support
                predictions = model.predict(X)
                n_classes = len(np.unique(predictions))
                dummy_probs = np.zeros((len(predictions), n_classes))
                dummy_probs[np.arange(len(predictions)), predictions] = 0.8
                dummy_probs += 0.2 / n_classes
                return dummy_probs
        else:
            raise ValueError("Model does not support predictions")
    
    def predict_with_enhancement(self, model_ids: List[str], X: np.ndarray,
                               return_detailed: bool = True) -> Union[np.ndarray, EnhancedPrediction]:
        """
        Make enhanced predictions using multiple models with intelligent management
        
        Args:
            model_ids: List of model IDs to use in ensemble
            X: Input features
            return_detailed: Whether to return detailed enhancement results
        
        Returns:
            Enhanced predictions with comprehensive metadata
        """
        
        if not model_ids:
            raise ValueError("No model IDs provided")
        
        # Prepare predictions from each model
        model_predictions = []
        model_probabilities = []
        
        for model_id in model_ids:
            if model_id not in self.model_registry:
                logger.warning(f"Model {model_id} not found in registry, skipping")
                continue
            
            model_info = self.model_registry[model_id]
            model = model_info['model']
            feature_selector = model_info.get('feature_selector')
            
            # Apply feature selection if available
            X_processed = X
            if feature_selector is not None:
                X_processed = feature_selector.transform(X)
            
            # Get predictions with monitoring
            predictions, monitoring_info = self.manager.predict_with_monitoring(
                model_id, X_processed, return_confidence=True
            )
            
            # Get probabilities
            probabilities = self._get_model_probabilities(model, X_processed)
            
            model_predictions.append(predictions)
            model_probabilities.append(probabilities)
        
        if not model_predictions:
            raise ValueError("No valid models found for prediction")
        
        # Get recent performance for weight optimization
        recent_performance = []
        for model_id in model_ids:
            if model_id in self.model_registry:
                health = self.manager.get_model_health(model_id)
                recent_performance.append(health.health_score / 100.0)
            else:
                recent_performance.append(0.5)  # Default performance
        
        # Optimize ensemble weights
        optimized_weights = self.enhancer.optimize_ensemble_weights_realtime(
            model_probabilities, recent_performance
        )
        
        # Enhance predictions
        enhanced_prediction = self.enhancer.enhance_predictions(
            model_predictions, model_probabilities, optimized_weights, return_detailed
        )
        
        logger.info(f"✅ Enhanced predictions generated using {len(model_ids)} models")
        
        return enhanced_prediction
    
    def get_comprehensive_model_report(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive report for a model"""
        
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not registered")
        
        # Get optimization results
        optimization_result = self.optimization_results.get(model_id, None)
        
        # Get health status
        health_status = self.manager.get_model_health(model_id)
        
        # Get adaptation recommendations
        adaptation_recommendations = self.manager.get_adaptation_recommendations(model_id)
        
        # Compile comprehensive report
        report = {
            'model_id': model_id,
            'model_info': self.model_registry[model_id],
            'optimization_results': {
                'best_score': optimization_result.best_score if optimization_result else None,
                'best_params': optimization_result.best_params if optimization_result else {},
                'training_time': optimization_result.training_time if optimization_result else 0,
                'feature_importance': optimization_result.feature_importance if optimization_result else {}
            },
            'health_status': {
                'overall_health': health_status.overall_health,
                'health_score': health_status.health_score,
                'performance_trend': health_status.performance_trend,
                'recommendations': health_status.recommendations,
                'alerts': health_status.alerts
            },
            'adaptation_recommendations': adaptation_recommendations,
            'report_generated': datetime.now().isoformat()
        }
        
        return report
    
    def save_comprehensive_results(self, model_id: str, base_path: str):
        """Save all Phase C results for a model"""
        
        # Create directories
        model_dir = os.path.join(base_path, f"phase_c_results_{model_id}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save optimization results
        if model_id in self.optimization_results:
            self.optimizer.save_optimization_results(
                self.optimization_results[model_id],
                os.path.join(model_dir, "optimization_results.json")
            )
        
        # Save monitoring report
        self.manager.export_monitoring_report(
            model_id, os.path.join(model_dir, "monitoring_report.json")
        )
        
        # Save comprehensive model report
        comprehensive_report = self.get_comprehensive_model_report(model_id)
        with open(os.path.join(model_dir, "comprehensive_report.json"), 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        # Save model
        model_info = self.model_registry[model_id]
        model_path = os.path.join(model_dir, "optimized_model.joblib")
        joblib.dump(model_info['model'], model_path)
        
        # Save feature selector if available
        if model_info.get('feature_selector'):
            selector_path = os.path.join(model_dir, "feature_selector.joblib")
            joblib.dump(model_info['feature_selector'], selector_path)
        
        logger.info(f"💾 Phase C results saved to {model_dir}")

def create_phase_c_optimizer(optimization_config: OptimizationConfig = None,
                           monitoring_window: int = 100,
                           calibration_method: str = 'isotonic') -> PhaseC_IntegratedOptimizer:
    """
    Create a Phase C integrated optimizer with specified configuration
    
    Args:
        optimization_config: Configuration for optimization
        monitoring_window: Window size for model monitoring
        calibration_method: Method for confidence calibration
    
    Returns:
        Configured PhaseC_IntegratedOptimizer instance
    """
    return PhaseC_IntegratedOptimizer(optimization_config, monitoring_window, calibration_method)

if __name__ == "__main__":
    # Example usage
    print("🚀 Phase C: Advanced Model Optimization Integration")
    print("=" * 60)
    
    # Create Phase C optimizer
    config = OptimizationConfig(n_trials=20, timeout=300, feature_selection=True)
    phase_c = create_phase_c_optimizer(config)
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(500, 20)
    y_train = np.random.randint(0, 3, 500)
    X_val = np.random.randn(100, 20)
    y_val = np.random.randint(0, 3, 100)
    
    # Optimize and train XGBoost model
    model, results = phase_c.optimize_and_train_model(
        'xgboost', X_train, y_train, X_val, y_val, 'test_xgb_model'
    )
    
    print(f"✅ Model optimized with score: {results.best_score:.4f}")
    
    # Make enhanced predictions
    X_test = np.random.randn(50, 20)
    enhanced_pred = phase_c.predict_with_enhancement(['test_xgb_model'], X_test)
    
    print(f"🏆 Enhanced predictions quality: {enhanced_pred.quality_assessment.quality_grade}")
    print(f"📊 Confidence: {enhanced_pred.quality_assessment.confidence_score:.3f}")
    
    # Get comprehensive report
    report = phase_c.get_comprehensive_model_report('test_xgb_model')
    print(f"🏥 Model health: {report['health_status']['overall_health']}")
    print(f"📈 Health score: {report['health_status']['health_score']:.1f}/100")
