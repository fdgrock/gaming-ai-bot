"""
AI Engines module for the lottery prediction system.

This module contains the core artificial intelligence components that power
the lottery prediction capabilities. It implements a 4-phase AI enhancement
system with orchestration and model management.

Components:
- Phase 1: Mathematical Engine (statistical analysis)
- Phase 2: Expert Ensemble (multiple model consensus)
- Phase 3: Set Optimizer (coverage optimization)
- Phase 4: Temporal Engine (time-series analysis)
- Prediction Orchestrator (coordination)
- Model Interface (abstraction layer)
- Enhancement Engine (continuous improvement)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging

# Version and metadata
__version__ = "1.0.0"
__author__ = "Lottery AI System"
__description__ = "Advanced AI engines for lottery prediction"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .phase1_mathematical import MathematicalEngine
    from .phase2_expert_ensemble import ExpertEnsemble
    from .phase3_set_optimizer import SetOptimizer
    from .phase4_temporal import TemporalEngine
    from .prediction_orchestrator import PredictionOrchestrator
    from .model_interface import ModelInterface, BaseModel
    from .enhancement_engine import EnhancementEngine
    
    logger.info("✅ All AI engines loaded successfully")
    
    # Export all engines
    __all__ = [
        'MathematicalEngine',
        'ExpertEnsemble', 
        'SetOptimizer',
        'TemporalEngine',
        'PredictionOrchestrator',
        'ModelInterface',
        'BaseModel',
        'EnhancementEngine',
        'create_orchestrator',
        'get_available_engines',
        'validate_engine_compatibility'
    ]
    
except ImportError as e:
    logger.warning(f"⚠️ Some AI engines not available: {e}")
    # Graceful degradation - provide placeholder classes
    
    class MathematicalEngine:
        def __init__(self, *args, **kwargs):
            logger.warning("MathematicalEngine not available - using placeholder")
    
    class ExpertEnsemble:
        def __init__(self, *args, **kwargs):
            logger.warning("ExpertEnsemble not available - using placeholder")
    
    class SetOptimizer:
        def __init__(self, *args, **kwargs):
            logger.warning("SetOptimizer not available - using placeholder")
    
    class TemporalEngine:
        def __init__(self, *args, **kwargs):
            logger.warning("TemporalEngine not available - using placeholder")
    
    class PredictionOrchestrator:
        def __init__(self, *args, **kwargs):
            logger.warning("PredictionOrchestrator not available - using placeholder")
    
    class ModelInterface:
        def __init__(self, *args, **kwargs):
            logger.warning("ModelInterface not available - using placeholder")
    
    class BaseModel:
        def __init__(self, *args, **kwargs):
            logger.warning("BaseModel not available - using placeholder")
    
    class EnhancementEngine:
        def __init__(self, *args, **kwargs):
            logger.warning("EnhancementEngine not available - using placeholder")
    
    __all__ = [
        'MathematicalEngine',
        'ExpertEnsemble',
        'SetOptimizer', 
        'TemporalEngine',
        'PredictionOrchestrator',
        'ModelInterface',
        'BaseModel',
        'EnhancementEngine'
    ]


def create_orchestrator(config: Optional[Dict[str, Any]] = None) -> 'PredictionOrchestrator':
    """
    Create a configured prediction orchestrator.
    
    Args:
        config: Configuration dictionary for the orchestrator
        
    Returns:
        Configured PredictionOrchestrator instance
    """
    try:
        default_config = {
            'enable_mathematical': True,
            'enable_ensemble': True,
            'enable_optimizer': True,
            'enable_temporal': True,
            'orchestration_strategy': 'adaptive',
            'confidence_threshold': 0.5,
            'max_predictions': 20
        }
        
        if config:
            default_config.update(config)
        
        orchestrator = PredictionOrchestrator(default_config)
        logger.info("✅ Prediction orchestrator created successfully")
        return orchestrator
        
    except Exception as e:
        logger.error(f"❌ Failed to create orchestrator: {e}")
        raise


def get_available_engines() -> List[str]:
    """
    Get list of available AI engines.
    
    Returns:
        List of available engine names
    """
    engines = []
    
    try:
        MathematicalEngine()
        engines.append('mathematical')
    except:
        pass
    
    try:
        ExpertEnsemble()
        engines.append('expert_ensemble')
    except:
        pass
    
    try:
        SetOptimizer()
        engines.append('set_optimizer')
    except:
        pass
    
    try:
        TemporalEngine()
        engines.append('temporal')
    except:
        pass
    
    logger.info(f"📊 Available engines: {engines}")
    return engines


def validate_engine_compatibility(engine_name: str, game_config: Dict[str, Any]) -> bool:
    """
    Validate if an engine is compatible with game configuration.
    
    Args:
        engine_name: Name of the engine to validate
        game_config: Game configuration dictionary
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        compatibility_matrix = {
            'mathematical': {
                'min_draws': 20,
                'number_range': (1, 100),
                'supports_bonus': True
            },
            'expert_ensemble': {
                'min_draws': 50,
                'number_range': (1, 200),
                'supports_bonus': True
            },
            'set_optimizer': {
                'min_draws': 30,
                'number_range': (1, 80),
                'supports_bonus': False
            },
            'temporal': {
                'min_draws': 100,
                'number_range': (1, 150),
                'supports_bonus': True
            }
        }
        
        if engine_name not in compatibility_matrix:
            return False
        
        requirements = compatibility_matrix[engine_name]
        
        # Check minimum draws
        if game_config.get('historical_draws', 0) < requirements['min_draws']:
            return False
        
        # Check number range
        game_range = game_config.get('number_range', [1, 49])
        min_range, max_range = requirements['number_range']
        if game_range[1] - game_range[0] < min_range or game_range[1] > max_range:
            return False
        
        # Check bonus number support
        if game_config.get('has_bonus', False) and not requirements['supports_bonus']:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Engine compatibility check failed: {e}")
        return False


# Engine factory
class EngineFactory:
    """Factory class for creating AI engines."""
    
    @staticmethod
    def create_engine(engine_type: str, config: Dict[str, Any]) -> Any:
        """
        Create an AI engine instance.
        
        Args:
            engine_type: Type of engine to create
            config: Engine configuration
            
        Returns:
            Engine instance
        """
        engine_map = {
            'mathematical': MathematicalEngine,
            'expert_ensemble': ExpertEnsemble,
            'set_optimizer': SetOptimizer,
            'temporal': TemporalEngine,
            'orchestrator': PredictionOrchestrator,
            'enhancement': EnhancementEngine
        }
        
        if engine_type not in engine_map:
            raise ValueError(f"Unknown engine type: {engine_type}")
        
        try:
            engine_class = engine_map[engine_type]
            engine = engine_class(config)
            logger.info(f"✅ Created {engine_type} engine")
            return engine
            
        except Exception as e:
            logger.error(f"❌ Failed to create {engine_type} engine: {e}")
            raise


# Engine status and health checking
def check_engine_health() -> Dict[str, Any]:
    """
    Check the health status of all AI engines.
    
    Returns:
        Dictionary with health status information
    """
    health_status = {
        'timestamp': str(logger.handlers[0].formatter.formatTime(logging.LogRecord('', 0, '', 0, '', (), None)) if logger.handlers else 'unknown'),
        'engines': {},
        'overall_status': 'healthy'
    }
    
    engines_to_check = [
        ('mathematical', MathematicalEngine),
        ('expert_ensemble', ExpertEnsemble),
        ('set_optimizer', SetOptimizer),
        ('temporal', TemporalEngine),
        ('orchestrator', PredictionOrchestrator),
        ('enhancement', EnhancementEngine)
    ]
    
    healthy_count = 0
    total_count = len(engines_to_check)
    
    for engine_name, engine_class in engines_to_check:
        try:
            # Try to instantiate with minimal config
            test_config = {'test_mode': True}
            engine = engine_class(test_config)
            
            health_status['engines'][engine_name] = {
                'status': 'healthy',
                'available': True,
                'last_check': 'just_now'
            }
            healthy_count += 1
            
        except Exception as e:
            health_status['engines'][engine_name] = {
                'status': 'unhealthy',
                'available': False,
                'error': str(e),
                'last_check': 'just_now'
            }
    
    # Determine overall status
    health_ratio = healthy_count / total_count
    if health_ratio >= 0.8:
        health_status['overall_status'] = 'healthy'
    elif health_ratio >= 0.5:
        health_status['overall_status'] = 'degraded'
    else:
        health_status['overall_status'] = 'critical'
    
    health_status['healthy_engines'] = healthy_count
    health_status['total_engines'] = total_count
    health_status['health_ratio'] = health_ratio
    
    logger.info(f"🏥 Engine health check completed: {health_status['overall_status']} ({healthy_count}/{total_count})")
    return health_status


# Module initialization
logger.info(f"🚀 AI Engines module initialized (v{__version__})")
logger.info(f"📋 Available engines: {len(__all__)} total")