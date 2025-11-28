"""
Sophisticated Engine Registry for the lottery prediction system.

This module provides advanced engine registry capabilities with sophisticated
engine management, performance tracking, intelligent routing, and ultra-high
accuracy coordination across all AI engines.
"""

import logging
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    """Engine status enumeration"""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    INITIALIZED = "initialized" 
    TRAINED = "trained"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


class EngineCapability(Enum):
    """Engine capability enumeration"""
    MATHEMATICAL_ANALYSIS = "mathematical_analysis"
    EXPERT_ENSEMBLE = "expert_ensemble"
    SET_OPTIMIZATION = "set_optimization"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    ORCHESTRATION = "orchestration"
    VISUALIZATION = "visualization"
    MODEL_INTERFACE = "model_interface"
    ULTRA_ACCURACY = "ultra_accuracy"
    SOPHISTICATED_PREDICTION = "sophisticated_prediction"


@dataclass
class EngineMetadata:
    """Comprehensive engine metadata"""
    engine_id: str
    name: str
    version: str
    engine_type: str
    capabilities: List[str]
    status: str
    performance_score: float
    intelligence_score: float
    registration_time: str
    last_active: str
    configuration: Dict[str, Any]
    dependencies: List[str]
    description: str
    author: str


@dataclass
class EnginePerformanceMetrics:
    """Engine performance metrics"""
    engine_id: str
    total_predictions: int
    successful_predictions: int
    average_confidence: float
    response_time_ms: float
    accuracy_score: float
    intelligence_score: float
    reliability_score: float
    last_updated: str


class SophisticatedEngineIntelligence:
    """Sophisticated engine intelligence for advanced analysis and optimization"""
    
    def __init__(self):
        self.intelligence_cache = {}
        self.performance_analytics = {}
        self.optimization_history = []
    
    def analyze_engine_intelligence(self, engine_metadata: EngineMetadata, 
                                  performance_metrics: EnginePerformanceMetrics,
                                  historical_performance: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze comprehensive engine intelligence"""
        try:
            intelligence_analysis = {}
            
            # Performance consistency analysis
            if historical_performance:
                confidences = [p.get('confidence', 0.5) for p in historical_performance]
                response_times = [p.get('response_time', 1000) for p in historical_performance]
                
                consistency_score = 1.0 - (np.std(confidences) / np.mean(confidences)) if confidences and np.mean(confidences) > 0 else 0.5
                speed_score = max(0.1, 1.0 - (np.mean(response_times) / 5000))  # Normalize to 5 seconds
                
                intelligence_analysis['consistency_intelligence'] = consistency_score
                intelligence_analysis['speed_intelligence'] = speed_score
            
            # Capability intelligence
            capability_count = len(engine_metadata.capabilities)
            capability_score = min(1.0, capability_count / 5.0)  # Normalize to 5 capabilities
            intelligence_analysis['capability_intelligence'] = capability_score
            
            # Performance trend intelligence
            if len(historical_performance) >= 5:
                recent_performance = historical_performance[-5:]
                recent_avg = np.mean([p.get('confidence', 0.5) for p in recent_performance])
                overall_avg = np.mean([p.get('confidence', 0.5) for p in historical_performance])
                
                trend_score = min(1.0, max(0.0, recent_avg / overall_avg)) if overall_avg > 0 else 0.5
                intelligence_analysis['trend_intelligence'] = trend_score
            
            # Reliability intelligence
            success_rate = performance_metrics.successful_predictions / performance_metrics.total_predictions if performance_metrics.total_predictions > 0 else 0
            reliability_factor = success_rate * performance_metrics.reliability_score
            intelligence_analysis['reliability_intelligence'] = reliability_factor
            
            # Calculate overall intelligence score
            intelligence_factors = list(intelligence_analysis.values())
            overall_intelligence = np.mean(intelligence_factors) if intelligence_factors else 0.5
            intelligence_analysis['overall_intelligence'] = overall_intelligence
            
            return intelligence_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing engine intelligence: {e}")
            return {'overall_intelligence': 0.5}
    
    def optimize_engine_parameters(self, engine_id: str, 
                                  performance_history: List[Dict[str, Any]],
                                  current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize engine parameters based on performance analysis"""
        try:
            optimized_config = current_config.copy()
            
            if len(performance_history) < 3:
                return optimized_config
            
            # Analyze performance trends
            recent_performance = performance_history[-5:]
            avg_confidence = np.mean([p.get('confidence', 0.5) for p in recent_performance])
            avg_response_time = np.mean([p.get('response_time', 1000) for p in recent_performance])
            
            # Confidence-based optimization
            if avg_confidence < 0.6:
                # Low confidence - increase conservative parameters
                if 'confidence_threshold' in optimized_config:
                    optimized_config['confidence_threshold'] = min(0.9, optimized_config.get('confidence_threshold', 0.5) + 0.1)
                if 'analysis_depth' in optimized_config:
                    optimized_config['analysis_depth'] = min(10, optimized_config.get('analysis_depth', 5) + 1)
            elif avg_confidence > 0.8:
                # High confidence - can be more aggressive
                if 'confidence_threshold' in optimized_config:
                    optimized_config['confidence_threshold'] = max(0.3, optimized_config.get('confidence_threshold', 0.5) - 0.05)
                if 'prediction_count' in optimized_config:
                    optimized_config['prediction_count'] = min(10, optimized_config.get('prediction_count', 5) + 1)
            
            # Response time optimization
            if avg_response_time > 3000:  # Over 3 seconds
                if 'parallel_processing' in optimized_config:
                    optimized_config['parallel_processing'] = True
                if 'cache_enabled' in optimized_config:
                    optimized_config['cache_enabled'] = True
            
            # Log optimization
            self.optimization_history.append({
                'engine_id': engine_id,
                'timestamp': datetime.now().isoformat(),
                'optimizations_applied': len([k for k in optimized_config.keys() if optimized_config[k] != current_config.get(k)]),
                'avg_confidence': avg_confidence,
                'avg_response_time': avg_response_time
            })
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"Error optimizing engine parameters: {e}")
            return current_config
    
    def calculate_engine_synergy(self, engine_performances: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate synergy between engines for intelligent coordination"""
        try:
            synergy_scores = {}
            
            engine_ids = list(engine_performances.keys())
            
            for i, engine1 in enumerate(engine_ids):
                for engine2 in engine_ids[i+1:]:
                    perf1 = engine_performances[engine1]
                    perf2 = engine_performances[engine2]
                    
                    # Calculate synergy based on complementary strengths
                    conf1 = perf1.get('average_confidence', 0.5)
                    conf2 = perf2.get('average_confidence', 0.5)
                    
                    # Synergy is high when engines complement each other
                    confidence_synergy = 1.0 - abs(conf1 - conf2)  # Similar confidence = good synergy
                    
                    # Capability synergy (different capabilities = better synergy)
                    cap1 = set(perf1.get('capabilities', []))
                    cap2 = set(perf2.get('capabilities', []))
                    capability_overlap = len(cap1 & cap2) / len(cap1 | cap2) if (cap1 | cap2) else 0
                    capability_synergy = 1.0 - capability_overlap  # Less overlap = better synergy
                    
                    # Combined synergy score
                    overall_synergy = (confidence_synergy * 0.6) + (capability_synergy * 0.4)
                    synergy_scores[f"{engine1}_{engine2}"] = overall_synergy
            
            return synergy_scores
            
        except Exception as e:
            logger.error(f"Error calculating engine synergy: {e}")
            return {}


class AdvancedEngineCoordinator:
    """Advanced engine coordinator for sophisticated multi-engine operations"""
    
    def __init__(self):
        self.coordination_cache = {}
        self.active_coordinations = {}
        self.coordination_history = []
    
    def coordinate_ultra_accuracy_operation(self, engines: Dict[str, Any], 
                                          operation_type: str = 'prediction',
                                          coordination_strategy: str = 'intelligent') -> Dict[str, Any]:
        """Coordinate ultra-accuracy operation across multiple engines"""
        try:
            logger.info(f"🚀 Coordinating ultra-accuracy {operation_type} across {len(engines)} engines")
            
            coordination_id = f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Phase 1: Engine Preparation
            prepared_engines = self._prepare_engines_for_coordination(engines)
            
            # Phase 2: Capability Analysis
            capability_matrix = self._analyze_engine_capabilities(prepared_engines)
            
            # Phase 3: Intelligent Routing
            routing_strategy = self._calculate_intelligent_routing(capability_matrix, coordination_strategy)
            
            # Phase 4: Coordinated Execution
            execution_results = self._execute_coordinated_operation(
                prepared_engines, routing_strategy, operation_type
            )
            
            # Phase 5: Result Aggregation
            aggregated_results = self._aggregate_coordination_results(execution_results, routing_strategy)
            
            # Store coordination
            coordination_result = {
                'coordination_id': coordination_id,
                'operation_type': operation_type,
                'strategy': coordination_strategy,
                'engines_coordinated': len(engines),
                'capability_matrix': capability_matrix,
                'routing_strategy': routing_strategy,
                'execution_results': execution_results,
                'aggregated_results': aggregated_results,
                'coordination_intelligence': self._calculate_coordination_intelligence(capability_matrix, execution_results),
                'timestamp': datetime.now().isoformat()
            }
            
            self.coordination_history.append(coordination_result)
            self.coordination_cache[coordination_id] = coordination_result
            
            logger.info(f"✅ Ultra-accuracy coordination complete: {coordination_id}")
            
            return coordination_result
            
        except Exception as e:
            logger.error(f"Error in ultra-accuracy coordination: {e}")
            return {
                'error': str(e),
                'coordination_intelligence': 0.0
            }
    
    def _prepare_engines_for_coordination(self, engines: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Prepare engines for coordinated operation"""
        try:
            prepared_engines = {}
            
            for engine_id, engine in engines.items():
                try:
                    # Check engine readiness
                    engine_status = getattr(engine, 'status', 'unknown')
                    engine_capabilities = getattr(engine, 'capabilities', [])
                    
                    prepared_engines[engine_id] = {
                        'engine': engine,
                        'status': engine_status,
                        'capabilities': engine_capabilities,
                        'ready': engine_status in ['trained', 'active'],
                        'preparation_time': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    logger.warning(f"Error preparing engine {engine_id}: {e}")
                    continue
            
            return prepared_engines
            
        except Exception as e:
            logger.error(f"Error preparing engines: {e}")
            return {}
    
    def _analyze_engine_capabilities(self, prepared_engines: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze engine capabilities for intelligent coordination"""
        try:
            capability_matrix = {
                'engines': {},
                'capability_coverage': defaultdict(list),
                'capability_overlap': {},
                'strength_analysis': {}
            }
            
            for engine_id, engine_info in prepared_engines.items():
                if not engine_info.get('ready', False):
                    continue
                
                capabilities = engine_info.get('capabilities', [])
                
                # Map capabilities
                capability_matrix['engines'][engine_id] = capabilities
                
                for capability in capabilities:
                    capability_matrix['capability_coverage'][capability].append(engine_id)
                
                # Analyze engine strengths
                engine = engine_info.get('engine')
                if engine and hasattr(engine, 'performance_history'):
                    performance_history = getattr(engine, 'performance_history', [])
                    if performance_history:
                        avg_confidence = np.mean([p.get('confidence', 0.5) for p in performance_history])
                        capability_matrix['strength_analysis'][engine_id] = avg_confidence
                    else:
                        capability_matrix['strength_analysis'][engine_id] = 0.5
                else:
                    capability_matrix['strength_analysis'][engine_id] = 0.5
            
            # Calculate capability overlaps
            capabilities = list(capability_matrix['capability_coverage'].keys())
            for i, cap1 in enumerate(capabilities):
                for cap2 in capabilities[i+1:]:
                    engines1 = set(capability_matrix['capability_coverage'][cap1])
                    engines2 = set(capability_matrix['capability_coverage'][cap2])
                    overlap = engines1 & engines2
                    
                    if overlap:
                        capability_matrix['capability_overlap'][f"{cap1}_{cap2}"] = list(overlap)
            
            return capability_matrix
            
        except Exception as e:
            logger.error(f"Error analyzing engine capabilities: {e}")
            return {}
    
    def _calculate_intelligent_routing(self, capability_matrix: Dict[str, Any], 
                                     strategy: str) -> Dict[str, Any]:
        """Calculate intelligent routing strategy"""
        try:
            routing_strategy = {
                'primary_engines': [],
                'secondary_engines': [],
                'capability_assignments': {},
                'load_distribution': {},
                'strategy': strategy
            }
            
            if not capability_matrix.get('engines'):
                return routing_strategy
            
            strength_analysis = capability_matrix.get('strength_analysis', {})
            
            if strategy == 'intelligent':
                # Route based on engine strengths and capabilities
                sorted_engines = sorted(strength_analysis.items(), key=lambda x: x[1], reverse=True)
                
                # Top 70% as primary engines
                primary_count = max(1, int(len(sorted_engines) * 0.7))
                routing_strategy['primary_engines'] = [engine_id for engine_id, _ in sorted_engines[:primary_count]]
                routing_strategy['secondary_engines'] = [engine_id for engine_id, _ in sorted_engines[primary_count:]]
                
            elif strategy == 'balanced':
                # Equal distribution among all engines
                all_engines = list(strength_analysis.keys())
                routing_strategy['primary_engines'] = all_engines
                
            elif strategy == 'best_only':
                # Use only the best performing engine
                best_engine = max(strength_analysis.items(), key=lambda x: x[1])[0]
                routing_strategy['primary_engines'] = [best_engine]
            
            # Calculate load distribution
            total_engines = len(routing_strategy['primary_engines'])
            if total_engines > 0:
                base_load = 1.0 / total_engines
                
                for engine_id in routing_strategy['primary_engines']:
                    strength = strength_analysis.get(engine_id, 0.5)
                    # Adjust load based on strength
                    adjusted_load = base_load * (0.5 + strength)  # 0.5 to 1.5 multiplier
                    routing_strategy['load_distribution'][engine_id] = adjusted_load
                
                # Normalize load distribution
                total_load = sum(routing_strategy['load_distribution'].values())
                if total_load > 0:
                    for engine_id in routing_strategy['load_distribution']:
                        routing_strategy['load_distribution'][engine_id] /= total_load
            
            return routing_strategy
            
        except Exception as e:
            logger.error(f"Error calculating routing strategy: {e}")
            return {'primary_engines': [], 'secondary_engines': []}
    
    def _execute_coordinated_operation(self, prepared_engines: Dict[str, Dict[str, Any]], 
                                     routing_strategy: Dict[str, Any],
                                     operation_type: str) -> Dict[str, Any]:
        """Execute coordinated operation across engines"""
        try:
            execution_results = {
                'successful_executions': {},
                'failed_executions': {},
                'execution_times': {},
                'total_success_rate': 0.0
            }
            
            primary_engines = routing_strategy.get('primary_engines', [])
            load_distribution = routing_strategy.get('load_distribution', {})
            
            # Execute on primary engines
            successful_count = 0
            total_count = 0
            
            for engine_id in primary_engines:
                if engine_id not in prepared_engines:
                    continue
                
                engine_info = prepared_engines[engine_id]
                engine = engine_info.get('engine')
                
                if not engine or not engine_info.get('ready', False):
                    continue
                
                try:
                    start_time = datetime.now()
                    
                    # Execute operation based on type
                    if operation_type == 'prediction':
                        load_factor = load_distribution.get(engine_id, 1.0)
                        num_predictions = max(1, int(5 * load_factor))  # Scale predictions by load
                        
                        if hasattr(engine, 'generate_ultra_high_accuracy_predictions'):
                            result = engine.generate_ultra_high_accuracy_predictions(num_predictions)
                        elif hasattr(engine, 'generate_sophisticated_predictions'):
                            result = engine.generate_sophisticated_predictions(num_predictions)
                        elif hasattr(engine, 'predict'):
                            result = engine.predict(num_predictions)
                        else:
                            raise Exception("No prediction method available")
                        
                        execution_results['successful_executions'][engine_id] = result
                        successful_count += 1
                    
                    execution_time = (datetime.now() - start_time).total_seconds()
                    execution_results['execution_times'][engine_id] = execution_time
                    total_count += 1
                    
                except Exception as e:
                    logger.warning(f"Engine {engine_id} execution failed: {e}")
                    execution_results['failed_executions'][engine_id] = str(e)
                    total_count += 1
            
            # Calculate success rate
            execution_results['total_success_rate'] = successful_count / total_count if total_count > 0 else 0.0
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Error executing coordinated operation: {e}")
            return {'successful_executions': {}, 'total_success_rate': 0.0}
    
    def _aggregate_coordination_results(self, execution_results: Dict[str, Any], 
                                      routing_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from coordinated execution"""
        try:
            successful_executions = execution_results.get('successful_executions', {})
            
            if not successful_executions:
                return {'aggregated_predictions': [], 'confidence_score': 0.0}
            
            # Collect all predictions with weights
            weighted_predictions = []
            load_distribution = routing_strategy.get('load_distribution', {})
            
            for engine_id, result in successful_executions.items():
                engine_weight = load_distribution.get(engine_id, 1.0)
                
                # Extract predictions from result
                predictions = []
                if isinstance(result, dict):
                    if 'optimized_sets' in result:
                        predictions = result['optimized_sets']
                    elif 'predictions' in result:
                        predictions = result['predictions']
                    elif 'numbers' in result:
                        predictions = [result]
                elif isinstance(result, list):
                    predictions = result
                
                # Weight predictions
                for pred in predictions:
                    if isinstance(pred, dict):
                        weighted_pred = pred.copy()
                        base_confidence = pred.get('confidence', 0.5)
                        weighted_confidence = base_confidence * engine_weight
                        weighted_pred['confidence'] = weighted_confidence
                        weighted_pred['source_engine'] = engine_id
                        weighted_pred['weight'] = engine_weight
                        weighted_predictions.append(weighted_pred)
            
            # Sort by weighted confidence
            weighted_predictions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Calculate overall confidence
            confidences = [p.get('confidence', 0) for p in weighted_predictions]
            overall_confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                'aggregated_predictions': weighted_predictions[:10],  # Top 10
                'confidence_score': overall_confidence,
                'total_predictions': len(weighted_predictions),
                'engines_contributed': len(successful_executions)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating coordination results: {e}")
            return {'aggregated_predictions': [], 'confidence_score': 0.0}
    
    def _calculate_coordination_intelligence(self, capability_matrix: Dict[str, Any], 
                                           execution_results: Dict[str, Any]) -> float:
        """Calculate coordination intelligence score"""
        try:
            intelligence_factors = []
            
            # Success rate factor
            success_rate = execution_results.get('total_success_rate', 0)
            intelligence_factors.append(success_rate)
            
            # Capability coverage factor
            capability_coverage = capability_matrix.get('capability_coverage', {})
            coverage_score = min(1.0, len(capability_coverage) / 5.0)  # Normalize to 5 capabilities
            intelligence_factors.append(coverage_score)
            
            # Engine diversity factor
            successful_engines = len(execution_results.get('successful_executions', {}))
            diversity_score = min(1.0, successful_engines / 4.0)  # Normalize to 4 engines
            intelligence_factors.append(diversity_score)
            
            # Performance consistency factor
            execution_times = list(execution_results.get('execution_times', {}).values())
            if execution_times:
                time_consistency = 1.0 - (np.std(execution_times) / np.mean(execution_times)) if np.mean(execution_times) > 0 else 0
                intelligence_factors.append(max(0.0, time_consistency))
            
            return np.mean(intelligence_factors) if intelligence_factors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating coordination intelligence: {e}")
            return 0.0


class UltraHighAccuracyEngineRegistry:
    """Ultra-high accuracy engine registry with sophisticated management capabilities"""
    
    def __init__(self, registry_path: str = "engine_registry.json"):
        """Initialize ultra-high accuracy engine registry"""
        self.registry_path = registry_path
        self.registered_engines = {}
        self.engine_metadata = {}
        self.performance_metrics = {}
        self.engine_intelligence = SophisticatedEngineIntelligence()
        self.engine_coordinator = AdvancedEngineCoordinator()
        
        # Registry configuration
        self.max_engines_per_type = 10
        self.performance_history_limit = 100
        self.intelligence_update_interval = timedelta(hours=1)
        
        # Thread safety
        self._registry_lock = threading.Lock()
        
        # Load existing registry
        self._load_registry()
        
        logger.info("🚀 Ultra-High Accuracy Engine Registry initialized")
    
    def register_engine(self, engine: Any, engine_config: Dict[str, Any]) -> bool:
        """Register engine with sophisticated capabilities analysis"""
        try:
            with self._registry_lock:
                # Generate engine ID
                engine_id = f"{engine_config.get('name', 'unknown')}_{engine_config.get('version', '1.0.0')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Analyze engine capabilities
                capabilities = self._analyze_engine_capabilities(engine)
                
                # Create metadata
                metadata = EngineMetadata(
                    engine_id=engine_id,
                    name=engine_config.get('name', 'Unknown Engine'),
                    version=engine_config.get('version', '1.0.0'),
                    engine_type=engine_config.get('type', 'generic'),
                    capabilities=capabilities,
                    status=EngineStatus.REGISTERED.value,
                    performance_score=0.5,
                    intelligence_score=0.5,
                    registration_time=datetime.now().isoformat(),
                    last_active=datetime.now().isoformat(),
                    configuration=engine_config,
                    dependencies=engine_config.get('dependencies', []),
                    description=engine_config.get('description', ''),
                    author=engine_config.get('author', 'Unknown')
                )
                
                # Create performance metrics
                performance = EnginePerformanceMetrics(
                    engine_id=engine_id,
                    total_predictions=0,
                    successful_predictions=0,
                    average_confidence=0.0,
                    response_time_ms=0.0,
                    accuracy_score=0.0,
                    intelligence_score=0.0,
                    reliability_score=0.0,
                    last_updated=datetime.now().isoformat()
                )
                
                # Register engine
                self.registered_engines[engine_id] = engine
                self.engine_metadata[engine_id] = metadata
                self.performance_metrics[engine_id] = performance
                
                # Save registry
                self._save_registry()
                
                logger.info(f"✅ Engine registered: {engine_id} with capabilities: {capabilities}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering engine: {e}")
            return False
    
    def _analyze_engine_capabilities(self, engine: Any) -> List[str]:
        """Analyze engine capabilities through introspection"""
        try:
            capabilities = []
            
            # Check for sophisticated methods
            if hasattr(engine, 'generate_ultra_high_accuracy_predictions'):
                capabilities.append(EngineCapability.ULTRA_ACCURACY.value)
                capabilities.append(EngineCapability.SOPHISTICATED_PREDICTION.value)
            
            # Check engine type based on methods and attributes
            if hasattr(engine, 'analyze_mathematical_patterns') or hasattr(engine, 'prime_analyzer'):
                capabilities.append(EngineCapability.MATHEMATICAL_ANALYSIS.value)
            
            if hasattr(engine, 'coordinate_experts') or hasattr(engine, 'expert_specialists'):
                capabilities.append(EngineCapability.EXPERT_ENSEMBLE.value)
            
            if hasattr(engine, 'optimize_sets') or hasattr(engine, 'coverage_optimizer'):
                capabilities.append(EngineCapability.SET_OPTIMIZATION.value)
            
            if hasattr(engine, 'analyze_temporal_patterns') or hasattr(engine, 'seasonal_detector'):
                capabilities.append(EngineCapability.TEMPORAL_ANALYSIS.value)
            
            if hasattr(engine, 'coordinate_engines') or hasattr(engine, 'orchestrate_predictions'):
                capabilities.append(EngineCapability.ORCHESTRATION.value)
            
            # Default capability
            if not capabilities:
                capabilities.append('basic_prediction')
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Error analyzing engine capabilities: {e}")
            return ['basic_prediction']
    
    def get_best_engines(self, capability: Optional[str] = None, 
                        count: int = 3) -> List[Tuple[str, Any]]:
        """Get best engines by capability and performance"""
        try:
            candidates = []
            
            for engine_id, metadata in self.engine_metadata.items():
                # Filter by capability if specified
                if capability and capability not in metadata.capabilities:
                    continue
                
                # Only include active engines
                if metadata.status not in [EngineStatus.TRAINED.value, EngineStatus.ACTIVE.value]:
                    continue
                
                # Calculate composite score
                performance = self.performance_metrics.get(engine_id)
                if performance:
                    composite_score = (
                        metadata.performance_score * 0.4 +
                        metadata.intelligence_score * 0.3 +
                        performance.reliability_score * 0.3
                    )
                else:
                    composite_score = metadata.performance_score * 0.5 + metadata.intelligence_score * 0.5
                
                candidates.append((engine_id, composite_score))
            
            # Sort by score and return top engines
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            result = []
            for engine_id, score in candidates[:count]:
                if engine_id in self.registered_engines:
                    result.append((engine_id, self.registered_engines[engine_id]))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting best engines: {e}")
            return []
    
    def coordinate_ultra_accuracy_prediction(self, num_predictions: int = 5) -> Dict[str, Any]:
        """Coordinate ultra-accuracy prediction across all capable engines"""
        try:
            # Get engines with ultra-accuracy capability
            ultra_accuracy_engines = {}
            
            for engine_id, metadata in self.engine_metadata.items():
                if (EngineCapability.ULTRA_ACCURACY.value in metadata.capabilities or
                    EngineCapability.SOPHISTICATED_PREDICTION.value in metadata.capabilities):
                    if metadata.status in [EngineStatus.TRAINED.value, EngineStatus.ACTIVE.value]:
                        ultra_accuracy_engines[engine_id] = self.registered_engines[engine_id]
            
            if not ultra_accuracy_engines:
                logger.warning("No ultra-accuracy capable engines available")
                return {
                    'predictions': [],
                    'coordination_intelligence': 0.0,
                    'error': 'No capable engines available'
                }
            
            # Coordinate prediction
            coordination_result = self.engine_coordinator.coordinate_ultra_accuracy_operation(
                ultra_accuracy_engines, 'prediction', 'intelligent'
            )
            
            # Update engine performance metrics
            self._update_coordination_performance(coordination_result)
            
            return {
                'predictions': coordination_result.get('aggregated_results', {}).get('aggregated_predictions', []),
                'coordination_intelligence': coordination_result.get('coordination_intelligence', 0.0),
                'engines_coordinated': coordination_result.get('engines_coordinated', 0),
                'coordination_id': coordination_result.get('coordination_id', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error in ultra-accuracy prediction coordination: {e}")
            return {
                'predictions': [],
                'coordination_intelligence': 0.0,
                'error': str(e)
            }
    
    def _update_coordination_performance(self, coordination_result: Dict[str, Any]) -> None:
        """Update performance metrics based on coordination results"""
        try:
            execution_results = coordination_result.get('execution_results', {})
            successful_executions = execution_results.get('successful_executions', {})
            execution_times = execution_results.get('execution_times', {})
            
            for engine_id in successful_executions.keys():
                if engine_id in self.performance_metrics:
                    perf = self.performance_metrics[engine_id]
                    
                    # Update metrics
                    perf.total_predictions += 1
                    perf.successful_predictions += 1
                    
                    if engine_id in execution_times:
                        response_time = execution_times[engine_id] * 1000  # Convert to ms
                        perf.response_time_ms = (perf.response_time_ms + response_time) / 2  # Running average
                    
                    perf.last_updated = datetime.now().isoformat()
                    
                    # Update metadata
                    if engine_id in self.engine_metadata:
                        self.engine_metadata[engine_id].last_active = datetime.now().isoformat()
            
            # Handle failed executions
            failed_executions = execution_results.get('failed_executions', {})
            for engine_id in failed_executions.keys():
                if engine_id in self.performance_metrics:
                    self.performance_metrics[engine_id].total_predictions += 1
                    # Note: successful_predictions not incremented for failures
            
        except Exception as e:
            logger.error(f"Error updating coordination performance: {e}")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status"""
        try:
            status = {
                'total_engines': len(self.registered_engines),
                'active_engines': 0,
                'engine_types': Counter(),
                'capability_coverage': Counter(),
                'performance_summary': {
                    'avg_intelligence_score': 0.0,
                    'avg_performance_score': 0.0,
                    'total_predictions': 0,
                    'success_rate': 0.0
                },
                'coordination_history': len(self.engine_coordinator.coordination_history)
            }
            
            total_intelligence = 0
            total_performance = 0
            total_predictions = 0
            total_successful = 0
            
            for engine_id, metadata in self.engine_metadata.items():
                # Count active engines
                if metadata.status in [EngineStatus.TRAINED.value, EngineStatus.ACTIVE.value]:
                    status['active_engines'] += 1
                
                # Count engine types
                status['engine_types'][metadata.engine_type] += 1
                
                # Count capabilities
                for capability in metadata.capabilities:
                    status['capability_coverage'][capability] += 1
                
                # Aggregate performance
                total_intelligence += metadata.intelligence_score
                total_performance += metadata.performance_score
                
                if engine_id in self.performance_metrics:
                    perf = self.performance_metrics[engine_id]
                    total_predictions += perf.total_predictions
                    total_successful += perf.successful_predictions
            
            # Calculate averages
            if len(self.engine_metadata) > 0:
                status['performance_summary']['avg_intelligence_score'] = total_intelligence / len(self.engine_metadata)
                status['performance_summary']['avg_performance_score'] = total_performance / len(self.engine_metadata)
            
            status['performance_summary']['total_predictions'] = total_predictions
            status['performance_summary']['success_rate'] = total_successful / total_predictions if total_predictions > 0 else 0.0
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting registry status: {e}")
            return {'error': str(e)}
    
    def _save_registry(self) -> None:
        """Save registry to persistent storage"""
        try:
            registry_data = {
                'metadata': {engine_id: asdict(metadata) for engine_id, metadata in self.engine_metadata.items()},
                'performance': {engine_id: asdict(perf) for engine_id, perf in self.performance_metrics.items()},
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _load_registry(self) -> None:
        """Load registry from persistent storage"""
        try:
            if Path(self.registry_path).exists():
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                # Load metadata
                metadata_data = registry_data.get('metadata', {})
                for engine_id, metadata_dict in metadata_data.items():
                    self.engine_metadata[engine_id] = EngineMetadata(**metadata_dict)
                
                # Load performance metrics
                performance_data = registry_data.get('performance', {})
                for engine_id, perf_dict in performance_data.items():
                    self.performance_metrics[engine_id] = EnginePerformanceMetrics(**perf_dict)
                
                logger.info(f"📁 Registry loaded: {len(self.engine_metadata)} engines")
                
        except Exception as e:
            logger.warning(f"Error loading registry (starting fresh): {e}")


# Global registry instance
_registry = None

def get_engine_registry() -> UltraHighAccuracyEngineRegistry:
    """Get global engine registry instance"""
    global _registry
    if _registry is None:
        _registry = UltraHighAccuracyEngineRegistry()
    return _registry

def register_engine(engine: Any, config: Dict[str, Any]) -> bool:
    """Register engine with global registry"""
    return get_engine_registry().register_engine(engine, config)

def get_best_engines(capability: Optional[str] = None, count: int = 3) -> List[Tuple[str, Any]]:
    """Get best engines from global registry"""
    return get_engine_registry().get_best_engines(capability, count)

def coordinate_ultra_accuracy_prediction(num_predictions: int = 5) -> Dict[str, Any]:
    """Coordinate ultra-accuracy prediction using global registry"""
    return get_engine_registry().coordinate_ultra_accuracy_prediction(num_predictions)