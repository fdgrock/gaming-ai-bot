"""
Real-Time Log Streaming Infrastructure for Streamlit Training Monitor
======================================================================
Implements non-blocking I/O for subprocess output capture and streaming
with thread-safe queue-based architecture for live log display.

Features:
- Non-blocking readline() with thread-safe Queue
- Line-by-line streaming without buffering
- Progress extraction and tracking
- Proper resource cleanup
"""

import threading
import subprocess
import time
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime
import re
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgress:
    """Tracks training progress metrics extracted from logs."""
    position: int = 0
    total_positions: int = 0
    model_name: str = ""
    trial: int = 0
    total_trials: int = 0
    score: float = 0.0
    is_best: bool = False
    elapsed_time: str = ""
    
    @property
    def position_progress(self) -> float:
        """Progress as fraction (0.0 to 1.0) for position training."""
        if self.total_positions == 0:
            return 0.0
        return self.position / self.total_positions
    
    @property
    def trial_progress(self) -> float:
        """Progress as fraction (0.0 to 1.0) for trial tuning."""
        if self.total_trials == 0:
            return 0.0
        return self.trial / self.total_trials


class RealTimeLogStreamer:
    """
    Manages real-time streaming of subprocess output with progress tracking.
    
    Uses thread-safe Queue to capture output line-by-line without blocking
    the main Streamlit thread. Provides progress extraction and metrics.
    """
    
    def __init__(self, process: subprocess.Popen, log_file: Path, max_buffer_lines: int = 500):
        """
        Initialize the log streamer.
        
        Args:
            process: Running subprocess.Popen object
            log_file: Path to write captured logs
            max_buffer_lines: Maximum lines to keep in memory buffer
        """
        self.process = process
        self.log_file = Path(log_file)
        self.max_buffer_lines = max_buffer_lines
        
        # Thread-safe queue for output
        self.output_queue: Queue = Queue()
        
        # Buffered logs for display
        self.log_buffer = []
        self.log_lock = threading.Lock()
        
        # Progress tracking
        self.progress = TrainingProgress()
        self.progress_lock = threading.Lock()
        
        # State tracking
        self.is_running = True
        self.thread = None
        self.start_time = datetime.now()
        
        # Callbacks for external updates
        self.on_new_line: Optional[Callable[[str], None]] = None
        self.on_progress_update: Optional[Callable[[TrainingProgress], None]] = None
    
    def start(self) -> None:
        """Start the output capture thread."""
        if self.thread is not None:
            return
        
        self.thread = threading.Thread(target=self._capture_output, daemon=True)
        self.thread.start()
        logger.info(f"Log streamer started for {self.log_file.name}")
    
    def stop(self) -> None:
        """Stop the capture thread and wait for it to finish."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info(f"Log streamer stopped for {self.log_file.name}")
    
    def _capture_output(self) -> None:
        """
        Thread target: Read from process stdout line-by-line and queue output.
        Runs in background thread without blocking.
        """
        try:
            with open(str(self.log_file), 'a', buffering=1, encoding='utf-8', errors='ignore') as f:
                line_count = 0
                while self.is_running and self.process.poll() is None:
                    try:
                        # Non-blocking read with timeout
                        line = self.process.stdout.readline()
                        if not line:
                            time.sleep(0.01)
                            continue
                        
                        # Queue the line for display
                        self.output_queue.put(line)
                        
                        # Write to file immediately
                        f.write(line)
                        f.flush()
                        
                        # Add to buffer
                        self._add_to_buffer(line)
                        
                        # Extract progress information
                        self._extract_progress(line)
                        
                        # Trigger callback
                        if self.on_new_line:
                            try:
                                self.on_new_line(line.rstrip())
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                        
                        line_count += 1
                    except Exception as e:
                        logger.warning(f"Read error: {e}")
                        time.sleep(0.01)
                
                # Capture any remaining output after process ends
                remaining = self.process.stdout.read()
                if remaining:
                    f.write(remaining)
                    f.flush()
                
                # Write completion marker
                f.write(f"\n[Process completed - {line_count} lines captured]\n")
                f.flush()
        
        except Exception as e:
            try:
                with open(str(self.log_file), 'a') as f:
                    f.write(f"\n[Error in log capture: {type(e).__name__}: {e}]\n")
            except:
                pass
            logger.error(f"Capture thread error: {e}")
        finally:
            self.is_running = False
    
    def _add_to_buffer(self, line: str) -> None:
        """Add line to buffer, maintaining max size."""
        with self.log_lock:
            self.log_buffer.append(line.rstrip())
            if len(self.log_buffer) > self.max_buffer_lines:
                self.log_buffer.pop(0)
    
    def _extract_progress(self, line: str) -> None:
        """Extract progress information from log line."""
        try:
            with self.progress_lock:
                # Position pattern: "Position 3/6"
                pos_match = re.search(r'Position (\d+)/(\d+)', line)
                if pos_match:
                    self.progress.position = int(pos_match.group(1))
                    self.progress.total_positions = int(pos_match.group(2))
                
                # Model name pattern: "=== XGBoost Position 1 ==="
                model_match = re.search(r'=== (\w+) Position', line)
                if model_match:
                    self.progress.model_name = model_match.group(1)
                
                # Trial pattern: "Trial 5/15: Score=0.1234"
                trial_match = re.search(r'Trial (\d+)/(\d+):', line)
                if trial_match:
                    self.progress.trial = int(trial_match.group(1))
                    self.progress.total_trials = int(trial_match.group(2))
                
                # Score pattern: "Score=0.1234" or "Score: 0.1234"
                score_match = re.search(r'Score[=:]?\s*([\d.]+)', line)
                if score_match:
                    self.progress.score = float(score_match.group(1))
                
                # Best flag pattern: "[BEST]" or "NEW BEST"
                if '[BEST]' in line or 'NEW BEST' in line:
                    self.progress.is_best = True
                
                # Update elapsed time
                self.progress.elapsed_time = self._get_elapsed_time()
                
                # Trigger callback
                if self.on_progress_update:
                    try:
                        self.on_progress_update(self.progress)
                    except Exception as e:
                        logger.error(f"Progress callback error: {e}")
        except Exception as e:
            logger.debug(f"Progress extraction error: {e}")
    
    def _get_elapsed_time(self) -> str:
        """Calculate elapsed time since start."""
        elapsed = datetime.now() - self.start_time
        hours = elapsed.seconds // 3600
        minutes = (elapsed.seconds % 3600) // 60
        seconds = elapsed.seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_logs(self, num_lines: Optional[int] = None) -> list:
        """
        Get buffered logs.
        
        Args:
            num_lines: Return last N lines, or None for all
        
        Returns:
            List of log lines
        """
        with self.log_lock:
            if num_lines is None:
                return self.log_buffer.copy()
            return self.log_buffer[-num_lines:] if self.log_buffer else []
    
    def get_all_logs_str(self, num_lines: Optional[int] = None) -> str:
        """Get logs as joined string."""
        logs = self.get_logs(num_lines)
        return '\n'.join(logs)
    
    def get_progress(self) -> TrainingProgress:
        """Get current progress snapshot."""
        with self.progress_lock:
            # Return a copy to avoid threading issues
            return TrainingProgress(
                position=self.progress.position,
                total_positions=self.progress.total_positions,
                model_name=self.progress.model_name,
                trial=self.progress.trial,
                total_trials=self.progress.total_trials,
                score=self.progress.score,
                is_best=self.progress.is_best,
                elapsed_time=self.progress.elapsed_time
            )
    
    def is_process_running(self) -> bool:
        """Check if subprocess is still running."""
        return self.process.poll() is None
    
    def get_process_returncode(self) -> Optional[int]:
        """Get process return code if finished."""
        return self.process.poll()
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> int:
        """
        Wait for process to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            Process return code
        """
        returncode = self.process.wait(timeout=timeout)
        self.stop()
        return returncode


class RealTimeLogDisplay:
    """Manages real-time display of logs in Streamlit with progress tracking."""
    
    def __init__(self, streamer: RealTimeLogStreamer):
        """
        Initialize display manager.
        
        Args:
            streamer: RealTimeLogStreamer instance
        """
        self.streamer = streamer
        self.last_display_count = 0
    
    def render_logs(self, container, num_lines: int = 100, height: int = 500) -> int:
        """
        Render logs to Streamlit container.
        
        Args:
            container: Streamlit container
            num_lines: Number of lines to display
            height: Height of log display in pixels
        
        Returns:
            Number of new lines added since last render
        """
        logs = self.streamer.get_logs(num_lines)
        new_lines = len(logs) - self.last_display_count
        self.last_display_count = len(logs)
        
        if logs:
            log_text = '\n'.join(logs)
        else:
            log_text = "Waiting for output...\n"
        
        # Escape HTML and render
        escaped = log_text.replace("<", "&lt;").replace(">", "&gt;")
        
        container.markdown(f"""
        <div style="
            height: {height}px; 
            overflow-y: auto; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            padding: 10px; 
            background-color: #f0f0f0;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
        ">
        {escaped}
        </div>
        """, unsafe_allow_html=True)
        
        return new_lines
    
    def render_progress(self, container) -> None:
        """
        Render progress bars and metrics.
        
        Args:
            container: Streamlit container
        """
        progress = self.streamer.get_progress()
        
        cols = container.columns([2, 1, 1, 1])
        
        with cols[0]:
            if progress.total_positions > 0:
                container.progress(progress.position_progress)
                status = f"Position {progress.position}/{progress.total_positions}"
                if progress.model_name:
                    status += f" - {progress.model_name}"
                container.text(status)
        
        with cols[1]:
            if progress.total_trials > 0:
                container.metric("Trial", f"{progress.trial}/{progress.total_trials}")
        
        with cols[2]:
            container.metric("Score", f"{progress.score:.4f}", delta="[BEST]" if progress.is_best else None)
        
        with cols[3]:
            container.metric("Elapsed", progress.elapsed_time)
