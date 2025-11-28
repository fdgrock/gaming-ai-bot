"""
Development Tools and Utilities for Lottery Prediction System

This module provides debugging, validation, and development utilities
for the lottery prediction system.
"""

import os
import sys
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
import csv


class DebugUtils:
    """Debugging utilities for development and troubleshooting."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize debug utilities with logging configuration."""
        self.setup_logging(log_level)
        self.debug_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_data = {}
    
    def setup_logging(self, level):
        """Set up comprehensive logging for debugging."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / f"debug_{self.debug_session_id}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Debug session started: {self.debug_session_id}")
    
    def trace_function_calls(self, func):
        """Decorator to trace function calls for debugging."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            self.logger.debug(f"Entering {func.__name__} with args: {args[:2]}... kwargs: {list(kwargs.keys())}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.logger.debug(f"Exiting {func.__name__} successfully in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Error in {func.__name__} after {execution_time:.3f}s: {str(e)}")
                self.logger.error(traceback.format_exc())
                raise
        
        return wrapper
    
    def capture_state(self, component_name: str, state_data: Dict[str, Any]):
        """Capture component state for debugging."""
        timestamp = datetime.now().isoformat()
        
        if component_name not in self.debug_data:
            self.debug_data[component_name] = []
        
        self.debug_data[component_name].append({
            "timestamp": timestamp,
            "state": state_data
        })
        
        self.logger.debug(f"Captured state for {component_name}: {len(state_data)} items")
    
    def dump_debug_data(self, output_file: Optional[str] = None):
        """Dump all captured debug data to file."""
        if not output_file:
            output_file = f"debug_dump_{self.debug_session_id}.json"
        
        debug_dump = {
            "session_id": self.debug_session_id,
            "timestamp": datetime.now().isoformat(),
            "components": self.debug_data,
            "summary": {
                "total_components": len(self.debug_data),
                "total_state_captures": sum(len(states) for states in self.debug_data.values())
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(debug_dump, f, indent=2, default=str)
        
        self.logger.info(f"Debug data dumped to {output_file}")
        return output_file
    
    def analyze_performance(self, performance_data: List[Dict[str, Any]]):
        """Analyze performance metrics for bottlenecks."""
        if not performance_data:
            return {"error": "No performance data provided"}
        
        # Calculate statistics
        execution_times = [item.get("execution_time", 0) for item in performance_data]
        
        analysis = {
            "total_operations": len(performance_data),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "slow_operations": [
                item for item in performance_data 
                if item.get("execution_time", 0) > sum(execution_times) / len(execution_times) * 2
            ],
            "bottlenecks": []
        }
        
        # Identify bottlenecks
        avg_time = analysis["avg_execution_time"]
        for item in performance_data:
            if item.get("execution_time", 0) > avg_time * 3:
                analysis["bottlenecks"].append({
                    "operation": item.get("operation", "unknown"),
                    "execution_time": item.get("execution_time"),
                    "severity": "high" if item.get("execution_time") > avg_time * 5 else "medium"
                })
        
        self.logger.info(f"Performance analysis completed: {len(analysis['bottlenecks'])} bottlenecks found")
        return analysis


class ValidationTools:
    """Validation tools for data integrity and system health."""
    
    def __init__(self):
        """Initialize validation tools."""
        self.logger = logging.getLogger(f"{__name__}.ValidationTools")
        self.validation_results = {}
    
    def validate_lottery_data(self, data: Dict[str, Any], game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate lottery data against game configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "data_quality": "good"
        }
        
        # Validate required fields
        required_fields = ["numbers", "draw_date", "game"]
        for field in required_fields:
            if field not in data:
                validation_result["errors"].append(f"Missing required field: {field}")
                validation_result["valid"] = False
        
        # Validate numbers
        if "numbers" in data:
            numbers = data["numbers"]
            if isinstance(numbers, str):
                try:
                    numbers = json.loads(numbers)
                except json.JSONDecodeError:
                    validation_result["errors"].append("Invalid numbers format")
                    validation_result["valid"] = False
                    return validation_result
            
            # Check number count
            if len(numbers) != game_config.get("num_numbers", 5):
                validation_result["errors"].append(
                    f"Invalid number count: expected {game_config.get('num_numbers', 5)}, got {len(numbers)}"
                )
                validation_result["valid"] = False
            
            # Check number range
            min_num = game_config.get("min_number", 1)
            max_num = game_config.get("max_number", 69)
            
            for num in numbers:
                if not isinstance(num, int) or num < min_num or num > max_num:
                    validation_result["errors"].append(
                        f"Invalid number {num}: must be integer between {min_num} and {max_num}"
                    )
                    validation_result["valid"] = False
            
            # Check for duplicates
            if len(numbers) != len(set(numbers)):
                validation_result["errors"].append("Duplicate numbers found")
                validation_result["valid"] = False
        
        # Validate bonus number
        if game_config.get("bonus_numbers", 0) > 0:
            if "bonus" not in data:
                validation_result["warnings"].append("Bonus number missing")
            elif data["bonus"] is not None:
                bonus = data["bonus"]
                bonus_max = game_config.get("bonus_max", 26)
                if not isinstance(bonus, int) or bonus < 1 or bonus > bonus_max:
                    validation_result["errors"].append(
                        f"Invalid bonus number {bonus}: must be integer between 1 and {bonus_max}"
                    )
                    validation_result["valid"] = False
        
        # Validate date format
        if "draw_date" in data:
            try:
                datetime.strptime(data["draw_date"], "%Y-%m-%d")
            except ValueError:
                validation_result["errors"].append("Invalid date format: expected YYYY-MM-DD")
                validation_result["valid"] = False
        
        # Determine data quality
        if validation_result["errors"]:
            validation_result["data_quality"] = "poor"
        elif validation_result["warnings"]:
            validation_result["data_quality"] = "fair"
        
        self.logger.info(f"Data validation completed: {validation_result['data_quality']} quality")
        return validation_result
    
    def validate_prediction(self, prediction: Dict[str, Any], game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction data structure and values."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "confidence_level": "unknown"
        }
        
        # Check required prediction fields
        required_fields = ["numbers", "confidence", "strategy", "engine"]
        for field in required_fields:
            if field not in prediction:
                validation_result["errors"].append(f"Missing required prediction field: {field}")
                validation_result["valid"] = False
        
        # Validate prediction numbers (same as lottery data)
        if "numbers" in prediction:
            numbers_validation = self.validate_lottery_data(
                {"numbers": prediction["numbers"], "draw_date": "2023-01-01", "game": "test"},
                game_config
            )
            validation_result["errors"].extend(numbers_validation["errors"])
            if not numbers_validation["valid"]:
                validation_result["valid"] = False
        
        # Validate confidence
        if "confidence" in prediction:
            confidence = prediction["confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                validation_result["errors"].append("Confidence must be a number between 0 and 1")
                validation_result["valid"] = False
            else:
                # Determine confidence level
                if confidence >= 0.8:
                    validation_result["confidence_level"] = "high"
                elif confidence >= 0.6:
                    validation_result["confidence_level"] = "medium"
                else:
                    validation_result["confidence_level"] = "low"
        
        # Validate strategy
        valid_strategies = ["conservative", "balanced", "aggressive"]
        if "strategy" in prediction and prediction["strategy"] not in valid_strategies:
            validation_result["warnings"].append(f"Unknown strategy: {prediction['strategy']}")
        
        # Validate engine
        valid_engines = ["ensemble", "neural_network", "pattern", "frequency"]
        if "engine" in prediction and prediction["engine"] not in valid_engines:
            validation_result["warnings"].append(f"Unknown engine: {prediction['engine']}")
        
        return validation_result
    
    def validate_system_health(self, services: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall system health."""
        health_report = {
            "overall_status": "healthy",
            "services": {},
            "issues": [],
            "recommendations": []
        }
        
        for service_name, service in services.items():
            service_health = {
                "status": "healthy",
                "response_time": None,
                "last_error": None,
                "uptime": None
            }
            
            try:
                # Test service responsiveness
                start_time = time.time()
                
                # Mock health check - in real implementation, call actual health check methods
                if hasattr(service, 'health_check'):
                    health_result = service.health_check()
                    service_health["status"] = health_result.get("status", "unknown")
                    service_health["last_error"] = health_result.get("last_error")
                
                service_health["response_time"] = time.time() - start_time
                
                # Check response time
                if service_health["response_time"] > 1.0:
                    service_health["status"] = "slow"
                    health_report["issues"].append(f"{service_name} responding slowly")
                
            except Exception as e:
                service_health["status"] = "error"
                service_health["last_error"] = str(e)
                health_report["issues"].append(f"{service_name} health check failed: {str(e)}")
            
            health_report["services"][service_name] = service_health
        
        # Determine overall status
        service_statuses = [s["status"] for s in health_report["services"].values()]
        if "error" in service_statuses:
            health_report["overall_status"] = "critical"
        elif "slow" in service_statuses:
            health_report["overall_status"] = "degraded"
        
        # Generate recommendations
        if health_report["overall_status"] != "healthy":
            health_report["recommendations"].append("Review service logs for details")
            health_report["recommendations"].append("Check system resources and network connectivity")
        
        return health_report


class DataMigrationTool:
    """Tool for data migration and database management."""
    
    def __init__(self, db_path: str = "lottery_data.db"):
        """Initialize migration tool with database path."""
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.DataMigrationTool")
    
    def create_database_schema(self):
        """Create the database schema for lottery data."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS historical_draws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game TEXT NOT NULL,
            draw_date DATE NOT NULL,
            numbers TEXT NOT NULL,
            bonus INTEGER,
            jackpot INTEGER,
            multiplier INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game TEXT NOT NULL,
            numbers TEXT NOT NULL,
            bonus INTEGER,
            confidence REAL NOT NULL,
            strategy TEXT NOT NULL,
            engine TEXT NOT NULL,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            preferences TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            component TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_historical_game_date ON historical_draws(game, draw_date);
        CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game);
        CREATE INDEX IF NOT EXISTS idx_predictions_generated_at ON predictions(generated_at);
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_sql)
                self.logger.info("Database schema created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create database schema: {str(e)}")
            return False
    
    def import_csv_data(self, csv_file: str, table_name: str, mapping: Dict[str, str]) -> int:
        """Import data from CSV file to database table."""
        imported_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn, open(csv_file, 'r') as file:
                csv_reader = csv.DictReader(file)
                
                for row in csv_reader:
                    # Map CSV columns to database columns
                    mapped_row = {db_col: row.get(csv_col) for db_col, csv_col in mapping.items()}
                    
                    # Insert into database
                    placeholders = ', '.join(['?' for _ in mapped_row])
                    columns = ', '.join(mapped_row.keys())
                    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                    
                    conn.execute(sql, list(mapped_row.values()))
                    imported_count += 1
                
                conn.commit()
                self.logger.info(f"Imported {imported_count} records from {csv_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to import CSV data: {str(e)}")
        
        return imported_count
    
    def export_data(self, table_name: str, output_file: str, format: str = "csv") -> bool:
        """Export data from database table to file."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                if format.lower() == "csv":
                    with open(output_file, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(columns)
                        writer.writerows(rows)
                        
                elif format.lower() == "json":
                    data = [dict(zip(columns, row)) for row in rows]
                    with open(output_file, 'w') as file:
                        json.dump(data, file, indent=2, default=str)
                
                self.logger.info(f"Exported {len(rows)} records to {output_file}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export data: {str(e)}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to backup database: {str(e)}")
            return False


class SystemMonitor:
    """System monitoring and alerting tool."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.logger = logging.getLogger(f"{__name__}.SystemMonitor")
        self.metrics_history = []
        self.alert_thresholds = {
            "response_time": 2.0,
            "error_rate": 0.05,
            "memory_usage": 0.90,
            "cache_hit_rate": 0.70
        }
    
    def collect_metrics(self, services: Dict[str, Any]) -> Dict[str, Any]:
        """Collect system metrics from all services."""
        timestamp = datetime.now()
        metrics = {
            "timestamp": timestamp.isoformat(),
            "services": {},
            "overall": {
                "total_requests": 0,
                "total_errors": 0,
                "avg_response_time": 0
            }
        }
        
        total_response_times = []
        total_requests = 0
        total_errors = 0
        
        for service_name, service in services.items():
            service_metrics = {
                "requests": 0,
                "errors": 0,
                "avg_response_time": 0,
                "memory_usage": 0,
                "status": "unknown"
            }
            
            try:
                # Mock metrics collection - in real implementation, get actual metrics
                if hasattr(service, 'get_metrics'):
                    service_metrics.update(service.get_metrics())
                
                # Accumulate overall metrics
                total_requests += service_metrics["requests"]
                total_errors += service_metrics["errors"]
                total_response_times.append(service_metrics["avg_response_time"])
                
            except Exception as e:
                self.logger.warning(f"Failed to collect metrics for {service_name}: {str(e)}")
                service_metrics["status"] = "error"
            
            metrics["services"][service_name] = service_metrics
        
        # Calculate overall metrics
        metrics["overall"]["total_requests"] = total_requests
        metrics["overall"]["total_errors"] = total_errors
        metrics["overall"]["error_rate"] = total_errors / max(total_requests, 1)
        metrics["overall"]["avg_response_time"] = sum(total_response_times) / max(len(total_response_times), 1)
        
        # Store metrics history
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check metrics against alert thresholds."""
        alerts = []
        
        overall = metrics["overall"]
        
        # Check error rate
        if overall["error_rate"] > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "error_rate",
                "severity": "high",
                "message": f"Error rate {overall['error_rate']:.3f} exceeds threshold {self.alert_thresholds['error_rate']}",
                "timestamp": metrics["timestamp"]
            })
        
        # Check response time
        if overall["avg_response_time"] > self.alert_thresholds["response_time"]:
            alerts.append({
                "type": "response_time",
                "severity": "medium",
                "message": f"Average response time {overall['avg_response_time']:.3f}s exceeds threshold {self.alert_thresholds['response_time']}s",
                "timestamp": metrics["timestamp"]
            })
        
        # Check individual services
        for service_name, service_metrics in metrics["services"].items():
            if service_metrics["status"] == "error":
                alerts.append({
                    "type": "service_error",
                    "severity": "high",
                    "message": f"Service {service_name} is in error state",
                    "timestamp": metrics["timestamp"]
                })
        
        return alerts
    
    def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate monitoring report for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        relevant_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
        ]
        
        if not relevant_metrics:
            return {"error": "No metrics available for specified time period"}
        
        # Calculate summary statistics
        error_rates = [m["overall"]["error_rate"] for m in relevant_metrics]
        response_times = [m["overall"]["avg_response_time"] for m in relevant_metrics]
        
        report = {
            "period": f"Last {hours} hours",
            "metrics_count": len(relevant_metrics),
            "summary": {
                "avg_error_rate": sum(error_rates) / len(error_rates),
                "max_error_rate": max(error_rates),
                "avg_response_time": sum(response_times) / len(response_times),
                "max_response_time": max(response_times),
                "uptime_percentage": (1 - sum(error_rates) / len(error_rates)) * 100
            },
            "trends": {
                "error_rate_trend": "stable",  # Would calculate actual trend
                "response_time_trend": "stable"
            },
            "recommendations": []
        }
        
        # Add recommendations based on metrics
        if report["summary"]["avg_error_rate"] > 0.02:
            report["recommendations"].append("Investigate high error rates")
        
        if report["summary"]["avg_response_time"] > 1.0:
            report["recommendations"].append("Optimize response times")
        
        return report


# Command-line interface for development tools
def main():
    """Main function for command-line usage of development tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lottery Prediction System Development Tools")
    parser.add_argument("command", choices=["debug", "validate", "migrate", "monitor"], 
                       help="Command to execute")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--input", help="Input file path")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    if args.command == "debug":
        debug_utils = DebugUtils(log_level)
        print(f"Debug session started: {debug_utils.debug_session_id}")
        
    elif args.command == "validate":
        validator = ValidationTools()
        print("Validation tools initialized")
        
    elif args.command == "migrate":
        migrator = DataMigrationTool()
        if migrator.create_database_schema():
            print("Database schema created successfully")
        else:
            print("Failed to create database schema")
            
    elif args.command == "monitor":
        monitor = SystemMonitor()
        print("System monitor initialized")


if __name__ == "__main__":
    main()