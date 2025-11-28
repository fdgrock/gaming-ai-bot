"""
Cache service module for the lottery prediction system.

This module provides caching services for performance optimization including
prediction caching, data caching, and cache invalidation strategies.
"""

import pickle
import json
import hashlib
import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import sqlite3

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Cache type enumeration."""
    PREDICTION = "prediction"
    DATA = "data"
    MODEL = "model"
    STATISTICS = "statistics"
    TEMPORARY = "temporary"


@dataclass
class CacheEntry:
    """Cache entry structure."""
    key: str
    value: Any
    cache_type: CacheType
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[int]  # Time to live in seconds
    size: int  # Size in bytes


class CacheManager:
    """
    Manages caching for performance optimization.
    
    This class provides a comprehensive caching system with support for
    different cache types, TTL, LRU eviction, and persistent storage.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.cache_dir = Path(self.config.get('cache_dir', 'cache'))
        self.max_memory_mb = self.config.get('max_memory_mb', 500)
        self.max_disk_mb = self.config.get('max_disk_mb', 2000)
        self.default_ttl = self.config.get('default_ttl', 3600)  # 1 hour
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5 minutes
        
        # In-memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0,
            'disk_usage': 0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._cleanup_timer = None
        
        # Cache configuration by type
        self.cache_configs = {
            CacheType.PREDICTION: {
                'ttl': self.config.get('prediction_ttl', 7200),  # 2 hours
                'max_size': self.config.get('prediction_max_size', 100),
                'persist': True
            },
            CacheType.DATA: {
                'ttl': self.config.get('data_ttl', 1800),  # 30 minutes
                'max_size': self.config.get('data_max_size', 50),
                'persist': True
            },
            CacheType.MODEL: {
                'ttl': self.config.get('model_ttl', 86400),  # 24 hours
                'max_size': self.config.get('model_max_size', 20),
                'persist': True
            },
            CacheType.STATISTICS: {
                'ttl': self.config.get('statistics_ttl', 3600),  # 1 hour
                'max_size': self.config.get('statistics_max_size', 30),
                'persist': True
            },
            CacheType.TEMPORARY: {
                'ttl': self.config.get('temporary_ttl', 300),  # 5 minutes
                'max_size': self.config.get('temporary_max_size', 100),
                'persist': False
            }
        }
        
        self.ensure_cache_directory()
        self.init_persistent_cache()
        self.start_cleanup_timer()
    
    def ensure_cache_directory(self) -> None:
        """Ensure cache directory structure exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different cache types
        for cache_type in CacheType:
            if self.cache_configs[cache_type]['persist']:
                (self.cache_dir / cache_type.value).mkdir(exist_ok=True)
        
        logger.info(f"📁 Cache directory: {self.cache_dir}")
    
    def init_persistent_cache(self) -> None:
        """Initialize persistent cache database."""
        try:
            self.db_path = self.cache_dir / 'cache_metadata.db'
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        cache_type TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        last_accessed TIMESTAMP NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        ttl INTEGER,
                        size INTEGER DEFAULT 0,
                        file_path TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("✅ Cache database initialized")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize cache database: {e}")
    
    def put(self, key: str, value: Any, cache_type: CacheType = CacheType.TEMPORARY,
            ttl: Optional[int] = None) -> bool:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache entry
            ttl: Time to live in seconds
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                # Generate cache key
                cache_key = self._generate_cache_key(key, cache_type)
                
                # Calculate size
                size = self._calculate_size(value)
                
                # Get TTL
                if ttl is None:
                    ttl = self.cache_configs[cache_type]['ttl']
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    value=value,
                    cache_type=cache_type,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=0,
                    ttl=ttl,
                    size=size
                )
                
                # Check memory limits
                if not self._can_fit_in_memory(size):
                    self._evict_memory_cache()
                
                # Store in memory
                self.memory_cache[cache_key] = entry
                self.cache_stats['memory_usage'] += size
                
                # Store persistently if configured
                if self.cache_configs[cache_type]['persist']:
                    self._store_persistent(entry)
                
                logger.debug(f"📦 Cached {cache_key} ({size} bytes)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to cache {key}: {e}")
            return False
    
    def get(self, key: str, cache_type: CacheType = CacheType.TEMPORARY) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
            cache_type: Type of cache entry
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            with self._lock:
                cache_key = self._generate_cache_key(key, cache_type)
                
                # Check memory cache first
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    
                    # Check if expired
                    if self._is_expired(entry):
                        self._remove_entry(cache_key)
                        self.cache_stats['misses'] += 1
                        return None
                    
                    # Update access info
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    self.cache_stats['hits'] += 1
                    return entry.value
                
                # Try to load from persistent cache
                if self.cache_configs[cache_type]['persist']:
                    entry = self._load_persistent(cache_key, cache_type)
                    if entry is not None:
                        # Load into memory if not expired
                        if not self._is_expired(entry):
                            if self._can_fit_in_memory(entry.size):
                                self.memory_cache[cache_key] = entry
                                self.cache_stats['memory_usage'] += entry.size
                            
                            entry.last_accessed = datetime.now()
                            entry.access_count += 1
                            self.cache_stats['hits'] += 1
                            return entry.value
                        else:
                            self._remove_persistent(cache_key)
                
                self.cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve {key}: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def invalidate(self, key: str, cache_type: CacheType = CacheType.TEMPORARY) -> bool:
        """
        Invalidate cache entry.
        
        Args:
            key: Cache key
            cache_type: Type of cache entry
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                cache_key = self._generate_cache_key(key, cache_type)
                return self._remove_entry(cache_key)
                
        except Exception as e:
            logger.error(f"❌ Failed to invalidate {key}: {e}")
            return False
    
    def invalidate_by_pattern(self, pattern: str, 
                             cache_type: Optional[CacheType] = None) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Key pattern to match
            cache_type: Specific cache type (None for all types)
            
        Returns:
            Number of entries invalidated
        """
        try:
            with self._lock:
                keys_to_remove = []
                
                for cache_key in self.memory_cache.keys():
                    entry = self.memory_cache[cache_key]
                    
                    # Check cache type filter
                    if cache_type is not None and entry.cache_type != cache_type:
                        continue
                    
                    # Check pattern match
                    if pattern in cache_key:
                        keys_to_remove.append(cache_key)
                
                # Remove matched keys
                removed_count = 0
                for cache_key in keys_to_remove:
                    if self._remove_entry(cache_key):
                        removed_count += 1
                
                logger.info(f"🧹 Invalidated {removed_count} cache entries matching '{pattern}'")
                return removed_count
                
        except Exception as e:
            logger.error(f"❌ Failed to invalidate by pattern: {e}")
            return 0
    
    def clear_cache(self, cache_type: Optional[CacheType] = None) -> bool:
        """
        Clear all cache entries.
        
        Args:
            cache_type: Specific cache type (None for all types)
            
        Returns:
            Success status
        """
        try:
            with self._lock:
                if cache_type is None:
                    # Clear all
                    keys_to_remove = list(self.memory_cache.keys())
                else:
                    # Clear specific type
                    keys_to_remove = [
                        key for key, entry in self.memory_cache.items()
                        if entry.cache_type == cache_type
                    ]
                
                # Remove entries
                for cache_key in keys_to_remove:
                    self._remove_entry(cache_key)
                
                logger.info(f"🧹 Cleared {len(keys_to_remove)} cache entries")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        try:
            with self._lock:
                total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
                hit_rate = self.cache_stats['hits'] / max(total_requests, 1)
                
                stats = {
                    'memory_entries': len(self.memory_cache),
                    'memory_usage_mb': self.cache_stats['memory_usage'] / (1024 * 1024),
                    'memory_limit_mb': self.max_memory_mb,
                    'disk_usage_mb': self._calculate_disk_usage() / (1024 * 1024),
                    'disk_limit_mb': self.max_disk_mb,
                    'hit_rate': hit_rate,
                    'total_hits': self.cache_stats['hits'],
                    'total_misses': self.cache_stats['misses'],
                    'total_evictions': self.cache_stats['evictions'],
                    'cache_type_breakdown': self._get_cache_type_breakdown()
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ Failed to get cache stats: {e}")
            return {}
    
    def optimize_cache(self) -> Dict[str, Any]:
        """
        Optimize cache performance.
        
        Returns:
            Optimization results
        """
        try:
            with self._lock:
                start_time = time.time()
                
                # Clean expired entries
                expired_count = self._cleanup_expired()
                
                # Evict least recently used if needed
                evicted_count = self._evict_memory_cache()
                
                # Cleanup persistent cache
                persistent_cleaned = self._cleanup_persistent_cache()
                
                optimization_time = time.time() - start_time
                
                results = {
                    'expired_removed': expired_count,
                    'memory_evicted': evicted_count,
                    'persistent_cleaned': persistent_cleaned,
                    'optimization_time': optimization_time,
                    'memory_usage_after': self.cache_stats['memory_usage'] / (1024 * 1024)
                }
                
                logger.info(f"⚡ Cache optimized in {optimization_time:.2f}s")
                return results
                
        except Exception as e:
            logger.error(f"❌ Cache optimization failed: {e}")
            return {'error': str(e)}
    
    def _generate_cache_key(self, key: str, cache_type: CacheType) -> str:
        """Generate full cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()[:8]
        return f"{cache_type.value}:{key}:{key_hash}"
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            import sys
            if isinstance(value, (str, bytes)):
                return len(value.encode() if isinstance(value, str) else value)
            else:
                return sys.getsizeof(value)
        except:
            return 1024  # Default size
    
    def _can_fit_in_memory(self, size: int) -> bool:
        """Check if value can fit in memory cache."""
        max_bytes = self.max_memory_mb * 1024 * 1024
        return (self.cache_stats['memory_usage'] + size) <= max_bytes
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if entry.ttl is None:
            return False
        
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl
    
    def _remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry from memory and disk."""
        try:
            # Remove from memory
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                self.cache_stats['memory_usage'] -= entry.size
                del self.memory_cache[cache_key]
            
            # Remove from persistent storage
            self._remove_persistent(cache_key)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove cache entry {cache_key}: {e}")
            return False
    
    def _evict_memory_cache(self) -> int:
        """Evict entries from memory cache using LRU strategy."""
        try:
            max_bytes = self.max_memory_mb * 1024 * 1024
            
            if self.cache_stats['memory_usage'] <= max_bytes:
                return 0
            
            # Sort by last accessed time (LRU)
            entries = list(self.memory_cache.items())
            entries.sort(key=lambda x: x[1].last_accessed)
            
            evicted_count = 0
            for cache_key, entry in entries:
                if self.cache_stats['memory_usage'] <= max_bytes:
                    break
                
                # Remove from memory (keep in persistent storage)
                self.cache_stats['memory_usage'] -= entry.size
                del self.memory_cache[cache_key]
                evicted_count += 1
                self.cache_stats['evictions'] += 1
            
            return evicted_count
            
        except Exception as e:
            logger.error(f"❌ Memory eviction failed: {e}")
            return 0
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries from memory cache."""
        try:
            expired_keys = []
            
            for cache_key, entry in self.memory_cache.items():
                if self._is_expired(entry):
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                self._remove_entry(cache_key)
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"❌ Cleanup expired failed: {e}")
            return 0
    
    def _store_persistent(self, entry: CacheEntry) -> bool:
        """Store entry in persistent cache."""
        try:
            # Save value to file
            cache_type_dir = self.cache_dir / entry.cache_type.value
            file_path = cache_type_dir / f"{entry.key.replace(':', '_')}.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(entry.value, f)
            
            # Save metadata to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_entries
                    (key, cache_type, created_at, last_accessed, access_count, ttl, size, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.key,
                    entry.cache_type.value,
                    entry.created_at.isoformat(),
                    entry.last_accessed.isoformat(),
                    entry.access_count,
                    entry.ttl,
                    entry.size,
                    str(file_path)
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store persistent cache entry: {e}")
            return False
    
    def _load_persistent(self, cache_key: str, cache_type: CacheType) -> Optional[CacheEntry]:
        """Load entry from persistent cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM cache_entries WHERE key = ?
                ''', (cache_key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Load metadata
                (key, cache_type_str, created_at_str, last_accessed_str, 
                 access_count, ttl, size, file_path_str) = row
                
                # Load value from file
                file_path = Path(file_path_str)
                if not file_path.exists():
                    return None
                
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    cache_type=CacheType(cache_type_str),
                    created_at=datetime.fromisoformat(created_at_str),
                    last_accessed=datetime.fromisoformat(last_accessed_str),
                    access_count=access_count,
                    ttl=ttl,
                    size=size
                )
                
                return entry
                
        except Exception as e:
            logger.error(f"❌ Failed to load persistent cache entry: {e}")
            return None
    
    def _remove_persistent(self, cache_key: str) -> bool:
        """Remove entry from persistent cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get file path
                cursor.execute('SELECT file_path FROM cache_entries WHERE key = ?', (cache_key,))
                row = cursor.fetchone()
                
                if row:
                    file_path = Path(row[0])
                    if file_path.exists():
                        file_path.unlink()
                
                # Remove from database
                cursor.execute('DELETE FROM cache_entries WHERE key = ?', (cache_key,))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove persistent cache entry: {e}")
            return False
    
    def _cleanup_persistent_cache(self) -> int:
        """Cleanup expired entries from persistent cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find expired entries
                cursor.execute('''
                    SELECT key, file_path, created_at, ttl FROM cache_entries
                    WHERE ttl IS NOT NULL
                ''')
                
                expired_keys = []
                current_time = datetime.now()
                
                for key, file_path, created_at_str, ttl in cursor.fetchall():
                    created_at = datetime.fromisoformat(created_at_str)
                    age = (current_time - created_at).total_seconds()
                    
                    if age > ttl:
                        expired_keys.append((key, file_path))
                
                # Remove expired entries
                for key, file_path in expired_keys:
                    # Remove file
                    try:
                        Path(file_path).unlink()
                    except:
                        pass
                    
                    # Remove from database
                    cursor.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                
                conn.commit()
                return len(expired_keys)
                
        except Exception as e:
            logger.error(f"❌ Persistent cache cleanup failed: {e}")
            return 0
    
    def _calculate_disk_usage(self) -> int:
        """Calculate total disk usage of cache."""
        try:
            total_size = 0
            for cache_type in CacheType:
                if self.cache_configs[cache_type]['persist']:
                    cache_dir = self.cache_dir / cache_type.value
                    if cache_dir.exists():
                        for file_path in cache_dir.iterdir():
                            if file_path.is_file():
                                total_size += file_path.stat().st_size
            
            return total_size
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate disk usage: {e}")
            return 0
    
    def _get_cache_type_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Get cache breakdown by type."""
        breakdown = {}
        
        for cache_type in CacheType:
            breakdown[cache_type.value] = {
                'count': 0,
                'size': 0
            }
        
        for entry in self.memory_cache.values():
            type_key = entry.cache_type.value
            breakdown[type_key]['count'] += 1
            breakdown[type_key]['size'] += entry.size
        
        return breakdown
    
    def start_cleanup_timer(self) -> None:
        """Start periodic cleanup timer."""
        try:
            if self._cleanup_timer is not None:
                self._cleanup_timer.cancel()
            
            def cleanup_task():
                try:
                    self._cleanup_expired()
                    self._cleanup_persistent_cache()
                except Exception as e:
                    logger.error(f"❌ Cleanup task failed: {e}")
                finally:
                    # Schedule next cleanup
                    self.start_cleanup_timer()
            
            self._cleanup_timer = threading.Timer(self.cleanup_interval, cleanup_task)
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()
            
        except Exception as e:
            logger.error(f"❌ Failed to start cleanup timer: {e}")
    
    def stop_cleanup_timer(self) -> None:
        """Stop periodic cleanup timer."""
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_cleanup_timer()
    
    @staticmethod
    def health_check() -> bool:
        """Check cache manager health."""
        return True


# Convenience functions for common cache operations
def cache_prediction(key: str, prediction_data: Dict[str, Any], 
                    ttl: Optional[int] = None) -> bool:
    """Cache prediction data."""
    from . import get_cache_manager
    cache_manager = get_cache_manager()
    return cache_manager.put(key, prediction_data, CacheType.PREDICTION, ttl)


def get_cached_prediction(key: str) -> Optional[Dict[str, Any]]:
    """Get cached prediction data."""
    from . import get_cache_manager
    cache_manager = get_cache_manager()
    return cache_manager.get(key, CacheType.PREDICTION)


def cache_statistics(key: str, stats_data: Dict[str, Any], 
                    ttl: Optional[int] = None) -> bool:
    """Cache statistics data."""
    from . import get_cache_manager
    cache_manager = get_cache_manager()
    return cache_manager.put(key, stats_data, CacheType.STATISTICS, ttl)


def get_cached_statistics(key: str) -> Optional[Dict[str, Any]]:
    """Get cached statistics data."""
    from . import get_cache_manager
    cache_manager = get_cache_manager()
    return cache_manager.get(key, CacheType.STATISTICS)


def invalidate_predictions() -> int:
    """Invalidate all prediction caches."""
    from . import get_cache_manager
    cache_manager = get_cache_manager()
    return cache_manager.clear_cache(CacheType.PREDICTION)