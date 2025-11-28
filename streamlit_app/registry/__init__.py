"""
Enhanced Registry System - Phase 5 Page Modularization

This package provides advanced registry systems for managing the modular architecture:
- PageRegistry: Dynamic page loading with dependency injection
- ServicesRegistry: Centralized service management
- ComponentsRegistry: UI component registration and management
- AIEnginesRegistry: AI engine lifecycle management

Supports hot-reloading, dependency injection, and advanced navigation.
"""

from .page_registry import EnhancedPageRegistry, PageInfo, NavigationContext
from .services_registry import ServicesRegistry, ServiceInfo
from .components_registry import ComponentsRegistry, ComponentInfo
from .ai_engines_registry import AIEnginesRegistry, EngineInfo, EngineType, EngineStatus, EngineCapability

__all__ = [
    'EnhancedPageRegistry',
    'PageInfo', 
    'NavigationContext',
    'ServicesRegistry',
    'ServiceInfo',
    'ComponentsRegistry',
    'ComponentInfo',
    'AIEnginesRegistry',
    'EngineInfo',
    'EngineType',
    'EngineStatus',
    'EngineCapability'
]