"""
Background Training Manager
Manages training processes that run in separate console windows
Allows GUI to launch training without blocking or crashing
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BackgroundTrainingManager:
    """Manages background training processes"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.training_log_dir = self.project_root / "logs" / "training"
        self.training_log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: Dict[str, subprocess.Popen] = {}
    
    def start_ensemble_training(
        self,
        game: str = "lotto_max",
        epochs: int = 150,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Start ensemble training in a new console window
        
        Args:
            game: "lotto_max" or "lotto_6_49"
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            dict with process info
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.training_log_dir / f"ensemble_{game}_{timestamp}.log"
        
        # Build command
        cmd = [
            "python",
            "train_models_standalone.py",
            "--model", "ensemble",
            "--game", game,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate)
        ]
        
        try:
            # Start process in new console window (Windows-specific)
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
            )
            
            process_id = f"ensemble_{game}_{timestamp}"
            self.processes[process_id] = process
            
            logger.info(f"Started training process: {process_id}")
            logger.info(f"Log file: {log_file}")
            logger.info(f"Process ID: {process.pid}")
            
            return {
                "status": "started",
                "process_id": process_id,
                "pid": process.pid,
                "game": game,
                "log_file": str(log_file),
                "timestamp": timestamp,
                "command": " ".join(cmd)
            }
        
        except Exception as e:
            logger.error(f"Failed to start training: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def start_single_model_training(
        self,
        model_type: str,
        game: str = "lotto_max",
        epochs: int = 150,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict:
        """Start a single model training in new console"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.training_log_dir / f"{model_type}_{game}_{timestamp}.log"
        
        cmd = [
            "python",
            "train_models_standalone.py",
            "--model", model_type,
            "--game", game,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate)
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
            )
            
            process_id = f"{model_type}_{game}_{timestamp}"
            self.processes[process_id] = process
            
            logger.info(f"Started {model_type} training: {process_id}")
            logger.info(f"Log file: {log_file}")
            
            return {
                "status": "started",
                "process_id": process_id,
                "pid": process.pid,
                "model_type": model_type,
                "game": game,
                "log_file": str(log_file),
                "timestamp": timestamp
            }
        
        except Exception as e:
            logger.error(f"Failed to start {model_type} training: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_process_status(self, process_id: str) -> Dict:
        """Get status of a training process"""
        if process_id not in self.processes:
            return {
                "status": "not_found",
                "process_id": process_id
            }
        
        process = self.processes[process_id]
        return_code = process.poll()
        
        if return_code is None:
            status = "running"
        elif return_code == 0:
            status = "completed"
        else:
            status = "failed"
        
        return {
            "status": status,
            "process_id": process_id,
            "pid": process.pid,
            "return_code": return_code
        }
    
    def list_processes(self) -> list:
        """List all training processes"""
        processes = []
        for process_id, process in self.processes.items():
            return_code = process.poll()
            
            if return_code is None:
                status = "running"
            elif return_code == 0:
                status = "completed"
            else:
                status = "failed"
            
            processes.append({
                "process_id": process_id,
                "pid": process.pid,
                "status": status,
                "return_code": return_code
            })
        
        return processes
    
    def read_log_file(self, process_id: str, tail_lines: int = 50) -> str:
        """Read last N lines from log file"""
        # Find log file based on process_id
        log_files = list(self.training_log_dir.glob(f"{process_id}*.log"))
        
        if not log_files:
            return f"Log file not found for {process_id}"
        
        log_file = log_files[0]
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return "".join(lines[-tail_lines:])
        except Exception as e:
            return f"Error reading log: {e}"

# Global instance
_training_manager = None

def get_training_manager() -> BackgroundTrainingManager:
    """Get or create global training manager instance"""
    global _training_manager
    if _training_manager is None:
        _training_manager = BackgroundTrainingManager()
    return _training_manager
