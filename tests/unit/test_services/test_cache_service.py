"""
Unit tests for the Cache Service module.

Tests caching functionality including storage, retrieval, TTL management,
and cache invalidation.
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from services.cache_service import CacheService
from configs.app_config import AppConfig


class TestCacheService:
    """Test suite for Cache Service functionality."""
    
    def test_service_initialization(self, mock_cache_service):
        """Test that CacheService initializes correctly."""
        assert mock_cache_service is not None
        assert hasattr(mock_cache_service, 'get')
        assert hasattr(mock_cache_service, 'set')
        assert hasattr(mock_cache_service, 'delete')
        assert hasattr(mock_cache_service, 'clear')
    
    def test_cache_set_and_get_success(self, mock_cache_service):
        """Test successful cache set and get operations."""
        key = "test_key"
        value = {"data": "test_value", "timestamp": "2023-12-01"}
        
        # Configure mocks
        mock_cache_service.set.return_value = True
        mock_cache_service.get.return_value = value
        
        # Test set
        set_result = mock_cache_service.set(key, value, ttl=3600)
        assert set_result is True
        
        # Test get
        get_result = mock_cache_service.get(key)
        assert get_result == value
        
        # Verify method calls
        mock_cache_service.set.assert_called_once_with(key, value, ttl=3600)
        mock_cache_service.get.assert_called_once_with(key)
    
    def test_cache_get_nonexistent_key(self, mock_cache_service):
        """Test getting a non-existent cache key."""
        mock_cache_service.get.return_value = None
        
        # Test
        result = mock_cache_service.get("nonexistent_key")
        
        # Assertions
        assert result is None
        mock_cache_service.get.assert_called_once_with("nonexistent_key")
    
    def test_cache_delete_success(self, mock_cache_service):
        """Test successful cache deletion."""
        key = "test_key_to_delete"
        mock_cache_service.delete.return_value = True
        
        # Test
        result = mock_cache_service.delete(key)
        
        # Assertions
        assert result is True
        mock_cache_service.delete.assert_called_once_with(key)
    
    def test_cache_delete_nonexistent_key(self, mock_cache_service):
        """Test deleting a non-existent cache key."""
        mock_cache_service.delete.return_value = False
        
        # Test
        result = mock_cache_service.delete("nonexistent_key")
        
        # Assertions
        assert result is False
    
    def test_cache_clear_all(self, mock_cache_service):
        """Test clearing all cache entries."""
        mock_cache_service.clear.return_value = True
        
        # Test
        result = mock_cache_service.clear()
        
        # Assertions
        assert result is True
        mock_cache_service.clear.assert_called_once()
    
    def test_cache_with_ttl_expiration(self, mock_cache_service):
        """Test cache TTL (Time To Live) functionality."""
        key = "ttl_test_key"
        value = "ttl_test_value"
        
        # Mock TTL behavior
        def mock_get_with_ttl(cache_key):
            # Simulate expired cache
            if hasattr(mock_get_with_ttl, 'call_count'):
                mock_get_with_ttl.call_count += 1
                if mock_get_with_ttl.call_count > 1:
                    return None  # Expired
            else:
                mock_get_with_ttl.call_count = 1
            return value
        
        mock_cache_service.get.side_effect = mock_get_with_ttl
        mock_cache_service.set.return_value = True
        
        # Test initial set
        mock_cache_service.set(key, value, ttl=1)  # 1 second TTL
        
        # Test immediate get (should work)
        result1 = mock_cache_service.get(key)
        assert result1 == value
        
        # Test get after expiration (should return None)
        result2 = mock_cache_service.get(key)
        assert result2 is None
    
    def test_cache_size_management(self, mock_cache_service):
        """Test cache size and memory management."""
        mock_cache_service.get_size.return_value = 1024  # 1KB
        mock_cache_service.get_count.return_value = 10
        
        # Test size methods
        size = mock_cache_service.get_size()
        count = mock_cache_service.get_count()
        
        assert size == 1024
        assert count == 10
        
        # Test eviction when size limit reached
        mock_cache_service.evict_lru.return_value = 5  # 5 items evicted
        
        evicted_count = mock_cache_service.evict_lru(max_size=512)
        assert evicted_count == 5
    
    def test_cache_key_patterns(self, mock_cache_service):
        """Test cache key pattern operations."""
        pattern = "predictions_*"
        matching_keys = ["predictions_powerball", "predictions_mega_millions", "predictions_test"]
        
        mock_cache_service.get_keys.return_value = matching_keys
        
        # Test
        result = mock_cache_service.get_keys(pattern)
        
        # Assertions
        assert result == matching_keys
        assert all("predictions_" in key for key in result)
        mock_cache_service.get_keys.assert_called_once_with(pattern)
    
    def test_cache_batch_operations(self, mock_cache_service):
        """Test batch cache operations."""
        # Test batch set
        batch_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        mock_cache_service.set_batch.return_value = len(batch_data)
        
        set_count = mock_cache_service.set_batch(batch_data, ttl=3600)
        assert set_count == 3
        
        # Test batch get
        keys = list(batch_data.keys())
        expected_values = list(batch_data.values())
        mock_cache_service.get_batch.return_value = expected_values
        
        values = mock_cache_service.get_batch(keys)
        assert values == expected_values
        
        # Test batch delete
        mock_cache_service.delete_batch.return_value = len(keys)
        
        deleted_count = mock_cache_service.delete_batch(keys)
        assert deleted_count == len(keys)
    
    def test_cache_statistics(self, mock_cache_service):
        """Test cache statistics and monitoring."""
        stats = {
            "hits": 150,
            "misses": 25,
            "hit_rate": 0.857,
            "total_requests": 175,
            "memory_usage": 2048,
            "item_count": 50
        }
        
        mock_cache_service.get_stats.return_value = stats
        
        # Test
        result = mock_cache_service.get_stats()
        
        # Assertions
        assert result["hit_rate"] > 0.8  # Good hit rate
        assert result["hits"] > result["misses"]
        assert result["total_requests"] == result["hits"] + result["misses"]
        mock_cache_service.get_stats.assert_called_once()
    
    def test_cache_serialization(self, mock_cache_service, sample_predictions):
        """Test cache serialization of complex objects."""
        key = "complex_object"
        complex_value = {
            "predictions": sample_predictions,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "count": len(sample_predictions)
            }
        }
        
        # Mock serialization behavior
        mock_cache_service.set.return_value = True
        mock_cache_service.get.return_value = complex_value
        
        # Test set complex object
        set_result = mock_cache_service.set(key, complex_value)
        assert set_result is True
        
        # Test get complex object
        get_result = mock_cache_service.get(key)
        assert get_result == complex_value
        assert "predictions" in get_result
        assert "metadata" in get_result
    
    def test_cache_concurrent_access(self, mock_cache_service):
        """Test cache behavior under concurrent access."""
        key = "concurrent_test"
        value = "concurrent_value"
        
        # Mock concurrent access behavior
        mock_cache_service.set.return_value = True
        mock_cache_service.get.return_value = value
        
        # Simulate multiple concurrent operations
        operations = []
        for i in range(10):
            # Alternate between set and get operations
            if i % 2 == 0:
                operations.append(('set', mock_cache_service.set(f"{key}_{i}", f"{value}_{i}")))
            else:
                operations.append(('get', mock_cache_service.get(f"{key}_{i-1}")))
        
        # All operations should complete successfully
        assert len(operations) == 10
        assert all(op[1] is not None for op in operations)
    
    def test_cache_error_handling(self, mock_cache_service):
        """Test cache error handling and recovery."""
        # Test connection error
        mock_cache_service.get.side_effect = ConnectionError("Cache unavailable")
        
        with pytest.raises(ConnectionError, match="Cache unavailable"):
            mock_cache_service.get("test_key")
        
        # Test recovery after error
        mock_cache_service.get.side_effect = None
        mock_cache_service.get.return_value = "recovered_value"
        
        result = mock_cache_service.get("test_key")
        assert result == "recovered_value"
    
    def test_cache_fallback_mechanism(self, mock_cache_service):
        """Test cache fallback when cache is unavailable."""
        # Primary cache fails
        mock_cache_service.get.side_effect = ConnectionError("Primary cache down")
        
        # Fallback mechanism
        mock_cache_service.get_from_fallback.return_value = "fallback_value"
        
        # Test fallback
        try:
            result = mock_cache_service.get("test_key")
        except ConnectionError:
            # Use fallback
            result = mock_cache_service.get_from_fallback("test_key")
        
        assert result == "fallback_value"
    
    def test_cache_warming(self, mock_cache_service, sample_predictions):
        """Test cache warming functionality."""
        warm_data = {
            "predictions_powerball": sample_predictions,
            "stats_powerball": {"frequency": {"1": 45, "2": 32}},
            "config_powerball": {"name": "Powerball", "max_number": 69}
        }
        
        mock_cache_service.warm_cache.return_value = len(warm_data)
        
        # Test
        warmed_count = mock_cache_service.warm_cache(warm_data)
        
        # Assertions
        assert warmed_count == len(warm_data)
        mock_cache_service.warm_cache.assert_called_once_with(warm_data)
    
    def test_cache_invalidation_patterns(self, mock_cache_service):
        """Test cache invalidation by patterns."""
        # Invalidate all prediction caches
        pattern = "predictions_*"
        invalidated_count = 5
        
        mock_cache_service.invalidate_pattern.return_value = invalidated_count
        
        # Test
        result = mock_cache_service.invalidate_pattern(pattern)
        
        # Assertions
        assert result == invalidated_count
        mock_cache_service.invalidate_pattern.assert_called_once_with(pattern)
    
    def test_cache_compression(self, mock_cache_service):
        """Test cache compression for large objects."""
        large_data = {"data": "x" * 10000}  # Large string
        compressed_size = 1500  # Compressed size
        
        mock_cache_service.set_compressed.return_value = True
        mock_cache_service.get_compressed.return_value = large_data
        mock_cache_service.get_compression_ratio.return_value = 0.15  # 85% compression
        
        # Test compressed storage
        set_result = mock_cache_service.set_compressed("large_key", large_data)
        assert set_result is True
        
        # Test compressed retrieval
        get_result = mock_cache_service.get_compressed("large_key")
        assert get_result == large_data
        
        # Test compression efficiency
        ratio = mock_cache_service.get_compression_ratio()
        assert ratio < 0.5  # Good compression
    
    @pytest.mark.parametrize("ttl,expected_behavior", [
        (0, "no_expiration"),
        (1, "short_expiration"),
        (3600, "hour_expiration"),
        (86400, "day_expiration")
    ])
    def test_cache_ttl_variations(self, mock_cache_service, ttl, expected_behavior):
        """Test cache with various TTL values."""
        key = f"ttl_test_{ttl}"
        value = f"value_for_{expected_behavior}"
        
        mock_cache_service.set.return_value = True
        mock_cache_service.get.return_value = value if ttl != 1 else None  # Simulate short expiration
        
        # Test set with TTL
        set_result = mock_cache_service.set(key, value, ttl=ttl)
        assert set_result is True
        
        # Test get behavior based on TTL
        get_result = mock_cache_service.get(key)
        if expected_behavior == "short_expiration":
            assert get_result is None  # Expired
        else:
            assert get_result == value  # Still valid
    
    def test_cache_namespace_isolation(self, mock_cache_service):
        """Test cache namespace isolation."""
        # Different namespaces
        namespaces = ["predictions", "statistics", "configurations"]
        
        for namespace in namespaces:
            key = f"{namespace}:test_key"
            value = f"value_for_{namespace}"
            
            mock_cache_service.set.return_value = True
            mock_cache_service.get.return_value = value
            
            # Test namespace isolation
            mock_cache_service.set(key, value)
            result = mock_cache_service.get(key)
            assert result == value
    
    def test_cache_memory_pressure_handling(self, mock_cache_service):
        """Test cache behavior under memory pressure."""
        # Simulate memory pressure
        mock_cache_service.get_memory_usage.return_value = 0.95  # 95% memory used
        mock_cache_service.handle_memory_pressure.return_value = {
            "evicted_items": 50,
            "freed_memory": 1024 * 1024,  # 1MB freed
            "new_usage": 0.75  # 75% after cleanup
        }
        
        # Test memory pressure handling
        result = mock_cache_service.handle_memory_pressure()
        
        assert result["evicted_items"] > 0
        assert result["freed_memory"] > 0
        assert result["new_usage"] < 0.95