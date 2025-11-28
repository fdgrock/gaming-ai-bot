"""
Performance Testing and Benchmarking Tools

Tools for performance testing, load testing, and benchmarking
the lottery prediction system components.
"""

import time
import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional, Tuple
import statistics
import json
import logging


class PerformanceTester:
    """Performance testing utilities for system components."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize performance tester."""
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        self.results = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def time_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Time the execution of a function."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        
        performance_data = {
            "function_name": func.__name__,
            "execution_time": end_time - start_time,
            "memory_delta": end_memory - start_memory,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "args_count": len(args),
            "kwargs_count": len(kwargs)
        }
        
        self.results.append(performance_data)
        self.logger.info(f"{func.__name__} executed in {performance_data['execution_time']:.3f}s")
        
        return performance_data
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback if psutil not available
            return 0.0
    
    def benchmark_function(self, func: Callable, iterations: int = 100, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark a function over multiple iterations."""
        execution_times = []
        memory_deltas = []
        errors = 0
        
        self.logger.info(f"Benchmarking {func.__name__} over {iterations} iterations")
        
        for i in range(iterations):
            perf_data = self.time_function(func, *args, **kwargs)
            
            if perf_data["success"]:
                execution_times.append(perf_data["execution_time"])
                memory_deltas.append(perf_data["memory_delta"])
            else:
                errors += 1
        
        if not execution_times:
            return {
                "function_name": func.__name__,
                "error": "All iterations failed",
                "total_errors": errors
            }
        
        benchmark_results = {
            "function_name": func.__name__,
            "iterations": iterations,
            "success_rate": (iterations - errors) / iterations,
            "execution_time": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "stdev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                "percentile_95": self._percentile(execution_times, 95),
                "percentile_99": self._percentile(execution_times, 99)
            },
            "memory_usage": {
                "mean_delta": statistics.mean(memory_deltas) if memory_deltas else 0,
                "max_delta": max(memory_deltas) if memory_deltas else 0,
                "min_delta": min(memory_deltas) if memory_deltas else 0
            },
            "total_errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Benchmark completed: {benchmark_results['execution_time']['mean']:.3f}s average")
        return benchmark_results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index % 1)
    
    def stress_test(self, func: Callable, duration_seconds: int = 60, concurrent_calls: int = 10, *args, **kwargs) -> Dict[str, Any]:
        """Perform stress testing on a function."""
        self.logger.info(f"Starting stress test for {func.__name__}: {duration_seconds}s duration, {concurrent_calls} concurrent calls")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        results = {
            "function_name": func.__name__,
            "duration_seconds": duration_seconds,
            "concurrent_calls": concurrent_calls,
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "calls_per_second": 0,
            "response_times": [],
            "errors": [],
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": None
        }
        
        def make_call():
            call_start = time.perf_counter()
            try:
                func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
            
            call_end = time.perf_counter()
            return {
                "success": success,
                "response_time": call_end - call_start,
                "error": error
            }
        
        with ThreadPoolExecutor(max_workers=concurrent_calls) as executor:
            futures = []
            
            while time.time() < end_time:
                # Submit new calls if we have capacity
                while len(futures) < concurrent_calls:
                    future = executor.submit(make_call)
                    futures.append(future)
                
                # Check completed calls
                completed_futures = [f for f in futures if f.done()]
                
                for future in completed_futures:
                    try:
                        call_result = future.result()
                        results["total_calls"] += 1
                        
                        if call_result["success"]:
                            results["successful_calls"] += 1
                        else:
                            results["failed_calls"] += 1
                            results["errors"].append(call_result["error"])
                        
                        results["response_times"].append(call_result["response_time"])
                        
                    except Exception as e:
                        results["failed_calls"] += 1
                        results["errors"].append(str(e))
                
                # Remove completed futures
                futures = [f for f in futures if not f.done()]
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
        
        # Wait for remaining futures
        for future in futures:
            try:
                call_result = future.result(timeout=5)
                results["total_calls"] += 1
                
                if call_result["success"]:
                    results["successful_calls"] += 1
                else:
                    results["failed_calls"] += 1
                    
                results["response_times"].append(call_result["response_time"])
            except Exception:
                results["failed_calls"] += 1
        
        actual_duration = time.time() - start_time
        results["actual_duration"] = actual_duration
        results["calls_per_second"] = results["total_calls"] / actual_duration
        results["end_time"] = datetime.now().isoformat()
        
        # Calculate response time statistics
        if results["response_times"]:
            results["response_time_stats"] = {
                "mean": statistics.mean(results["response_times"]),
                "median": statistics.median(results["response_times"]),
                "min": min(results["response_times"]),
                "max": max(results["response_times"]),
                "stdev": statistics.stdev(results["response_times"]) if len(results["response_times"]) > 1 else 0
            }
        
        self.logger.info(f"Stress test completed: {results['calls_per_second']:.2f} calls/second")
        return results
    
    def load_test(self, func: Callable, users: List[int], duration: int = 30, *args, **kwargs) -> Dict[str, Any]:
        """Perform load testing with varying user counts."""
        self.logger.info(f"Starting load test for {func.__name__} with user counts: {users}")
        
        load_results = {
            "function_name": func.__name__,
            "test_duration": duration,
            "user_scenarios": [],
            "summary": {}
        }
        
        for user_count in users:
            self.logger.info(f"Testing with {user_count} concurrent users")
            
            scenario_result = self.stress_test(func, duration, user_count, *args, **kwargs)
            scenario_result["user_count"] = user_count
            
            load_results["user_scenarios"].append(scenario_result)
        
        # Generate summary
        if load_results["user_scenarios"]:
            calls_per_second = [s["calls_per_second"] for s in load_results["user_scenarios"]]
            success_rates = [s["successful_calls"] / max(s["total_calls"], 1) for s in load_results["user_scenarios"]]
            
            load_results["summary"] = {
                "max_throughput": max(calls_per_second),
                "optimal_users": users[calls_per_second.index(max(calls_per_second))],
                "avg_success_rate": statistics.mean(success_rates),
                "performance_degradation": (max(calls_per_second) - min(calls_per_second)) / max(calls_per_second) if calls_per_second else 0
            }
        
        return load_results
    
    def memory_profiling(self, func: Callable, iterations: int = 10, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage of a function."""
        self.logger.info(f"Memory profiling {func.__name__} over {iterations} iterations")
        
        memory_snapshots = []
        
        # Baseline memory
        baseline_memory = self._get_memory_usage()
        
        for i in range(iterations):
            pre_memory = self._get_memory_usage()
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                result = None
            
            execution_time = time.perf_counter() - start_time
            post_memory = self._get_memory_usage()
            
            memory_snapshots.append({
                "iteration": i + 1,
                "pre_memory": pre_memory,
                "post_memory": post_memory,
                "memory_delta": post_memory - pre_memory,
                "execution_time": execution_time,
                "success": success
            })
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Analysis
        memory_deltas = [s["memory_delta"] for s in memory_snapshots if s["success"]]
        
        profile_result = {
            "function_name": func.__name__,
            "iterations": iterations,
            "baseline_memory": baseline_memory,
            "final_memory": self._get_memory_usage(),
            "total_memory_change": self._get_memory_usage() - baseline_memory,
            "snapshots": memory_snapshots,
            "memory_statistics": {
                "mean_delta": statistics.mean(memory_deltas) if memory_deltas else 0,
                "max_delta": max(memory_deltas) if memory_deltas else 0,
                "min_delta": min(memory_deltas) if memory_deltas else 0,
                "total_allocated": sum(d for d in memory_deltas if d > 0),
                "total_freed": sum(d for d in memory_deltas if d < 0)
            }
        }
        
        return profile_result
    
    def compare_functions(self, functions: List[Callable], iterations: int = 100, *args, **kwargs) -> Dict[str, Any]:
        """Compare performance of multiple functions."""
        self.logger.info(f"Comparing {len(functions)} functions over {iterations} iterations")
        
        comparison_results = {
            "iterations": iterations,
            "functions": {},
            "comparison": {}
        }
        
        # Benchmark each function
        for func in functions:
            self.logger.info(f"Benchmarking {func.__name__}")
            benchmark = self.benchmark_function(func, iterations, *args, **kwargs)
            comparison_results["functions"][func.__name__] = benchmark
        
        # Generate comparison
        function_names = list(comparison_results["functions"].keys())
        if len(function_names) > 1:
            # Find fastest function
            mean_times = {name: data["execution_time"]["mean"] for name, data in comparison_results["functions"].items()}
            fastest_func = min(mean_times, key=mean_times.get)
            
            comparison_results["comparison"] = {
                "fastest_function": fastest_func,
                "fastest_time": mean_times[fastest_func],
                "relative_performance": {
                    name: time / mean_times[fastest_func] 
                    for name, time in mean_times.items()
                },
                "speed_differences": {
                    name: ((time - mean_times[fastest_func]) / mean_times[fastest_func]) * 100
                    for name, time in mean_times.items()
                    if name != fastest_func
                }
            }
        
        return comparison_results
    
    def export_results(self, filename: Optional[str] = None) -> str:
        """Export all performance test results to a file."""
        if not filename:
            filename = f"performance_results_{self.session_id}.json"
        
        export_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {filename}")
        return filename


class SystemBenchmark:
    """System-wide benchmarking for the lottery prediction system."""
    
    def __init__(self):
        """Initialize system benchmark."""
        self.logger = logging.getLogger(f"{__name__}.SystemBenchmark")
        self.performance_tester = PerformanceTester()
    
    def benchmark_data_operations(self, data_service) -> Dict[str, Any]:
        """Benchmark data service operations."""
        self.logger.info("Benchmarking data operations")
        
        results = {}
        
        # Benchmark data retrieval
        if hasattr(data_service, 'get_historical_data'):
            results["get_historical_data"] = self.performance_tester.benchmark_function(
                data_service.get_historical_data, 50, "powerball", days=365
            )
        
        # Benchmark data saving
        if hasattr(data_service, 'save_prediction'):
            sample_prediction = {
                "numbers": [1, 15, 25, 35, 45],
                "confidence": 0.75,
                "strategy": "balanced"
            }
            results["save_prediction"] = self.performance_tester.benchmark_function(
                data_service.save_prediction, 100, sample_prediction
            )
        
        return results
    
    def benchmark_prediction_generation(self, prediction_service) -> Dict[str, Any]:
        """Benchmark prediction generation operations."""
        self.logger.info("Benchmarking prediction generation")
        
        results = {}
        
        if hasattr(prediction_service, 'generate_predictions'):
            # Test different strategies
            strategies = ["conservative", "balanced", "aggressive"]
            
            for strategy in strategies:
                results[f"generate_predictions_{strategy}"] = self.performance_tester.benchmark_function(
                    prediction_service.generate_predictions, 25, 
                    game="powerball", strategy=strategy
                )
        
        return results
    
    def benchmark_ai_engines(self, ai_engines: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark AI engine performance."""
        self.logger.info("Benchmarking AI engines")
        
        results = {}
        
        for engine_name, engine in ai_engines.items():
            if hasattr(engine, 'generate_prediction'):
                results[f"{engine_name}_prediction"] = self.performance_tester.benchmark_function(
                    engine.generate_prediction, 20,
                    game_config={"num_numbers": 5, "max_number": 69},
                    historical_data=[]
                )
        
        return results
    
    def benchmark_cache_operations(self, cache_service) -> Dict[str, Any]:
        """Benchmark cache service operations."""
        self.logger.info("Benchmarking cache operations")
        
        results = {}
        
        # Benchmark cache set
        if hasattr(cache_service, 'set'):
            results["cache_set"] = self.performance_tester.benchmark_function(
                cache_service.set, 1000, "test_key", {"data": "test_value"}, ttl=3600
            )
        
        # Benchmark cache get
        if hasattr(cache_service, 'get'):
            results["cache_get"] = self.performance_tester.benchmark_function(
                cache_service.get, 1000, "test_key"
            )
        
        return results
    
    def full_system_benchmark(self, services: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive system benchmark."""
        self.logger.info("Starting full system benchmark")
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "summary": {}
        }
        
        # Benchmark each service
        if "data_service" in services:
            benchmark_results["components"]["data_service"] = self.benchmark_data_operations(services["data_service"])
        
        if "prediction_service" in services:
            benchmark_results["components"]["prediction_service"] = self.benchmark_prediction_generation(services["prediction_service"])
        
        if "cache_service" in services:
            benchmark_results["components"]["cache_service"] = self.benchmark_cache_operations(services["cache_service"])
        
        # Generate summary
        all_benchmarks = []
        for component, tests in benchmark_results["components"].items():
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict) and "execution_time" in test_result:
                    all_benchmarks.append({
                        "component": component,
                        "test": test_name,
                        "mean_time": test_result["execution_time"]["mean"],
                        "success_rate": test_result["success_rate"]
                    })
        
        if all_benchmarks:
            mean_times = [b["mean_time"] for b in all_benchmarks]
            success_rates = [b["success_rate"] for b in all_benchmarks]
            
            benchmark_results["summary"] = {
                "total_tests": len(all_benchmarks),
                "overall_avg_time": statistics.mean(mean_times),
                "overall_success_rate": statistics.mean(success_rates),
                "fastest_operation": min(all_benchmarks, key=lambda x: x["mean_time"]),
                "slowest_operation": max(all_benchmarks, key=lambda x: x["mean_time"])
            }
        
        self.logger.info("Full system benchmark completed")
        return benchmark_results


# Export performance testing utilities
__all__ = ["PerformanceTester", "SystemBenchmark"]