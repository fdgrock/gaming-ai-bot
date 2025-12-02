"""
🤖 Advanced ML Training - Phase 2
Comprehensive interface for executing Phase 2A (Tree Models), Phase 2B (Neural Networks), 
and Phase 2C (Ensemble Variants) model training with real-time monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import sys
from datetime import datetime
import subprocess
import json
import time
from dataclasses import dataclass
import logging

try:
    import psutil
except ImportError:
    st.error("psutil not installed. Please run: pip install psutil")
    psutil = None

try:
    from ..core import (
        get_available_games,
        get_session_value,
        set_session_value,
        app_log,
        get_data_dir
    )
except ImportError:
    def get_available_games(): return ["Lotto 6/49", "Lotto Max"]
    def get_session_value(k, d=None): return st.session_state.get(k, d)
    def set_session_value(k, v): st.session_state[k] = v
    def app_log(message: str, level: str = "info"): print(f"[{level.upper()}] {message}")
    def get_data_dir(): return Path("data")

# Import real-time streaming components
try:
    from streamlit_app.realtime_log_streamer import RealTimeLogStreamer, RealTimeLogDisplay
except ImportError:
    try:
        # Fallback for direct module import
        import sys
        from pathlib import Path as PathlibPath
        sys.path.insert(0, str(PathlibPath(__file__).parent.parent))
        from realtime_log_streamer import RealTimeLogStreamer, RealTimeLogDisplay
    except ImportError as e:
        st.error(f"Failed to import RealTimeLogStreamer: {e}")
        RealTimeLogStreamer = None
        RealTimeLogDisplay = None


# ============================================================================
# Constants & Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "advanced"
TOOLS_DIR = PROJECT_ROOT / "tools"

PHASE_2_TRAINERS = {
    "Phase 2A - Tree Models": {
        "script": "advanced_tree_model_trainer.py",
        "description": "Position-specific XGBoost, LightGBM, and CatBoost optimization",
        "models_per_game": "18 (6 positions × 3 architectures)",
        "estimated_time": "60-90 minutes",
        "gpu_required": False,
        "models": ["XGBoost", "LightGBM", "CatBoost"]
    },
    "Phase 2B - LSTM with Attention": {
        "script": "advanced_lstm_model_trainer.py",
        "description": "Encoder-decoder LSTM with Luong-style attention mechanism",
        "models_per_game": "1",
        "estimated_time": "30-45 minutes",
        "gpu_required": True,
        "models": ["LSTM"]
    },
    "Phase 2B - Transformer": {
        "script": "advanced_transformer_model_trainer.py",
        "description": "Decoder-only Transformer (GPT-like) architecture",
        "models_per_game": "1",
        "estimated_time": "45-60 minutes",
        "gpu_required": True,
        "models": ["Transformer"]
    },
    "Phase 2B - CNN": {
        "script": "advanced_cnn_model_trainer.py",
        "description": "1D CNN for detecting local patterns in lottery draws",
        "models_per_game": "1",
        "estimated_time": "15-25 minutes",
        "gpu_required": False,
        "models": ["CNN"]
    }
}

PHASE_2C_VARIANTS = {
    "Ensemble Variants - Transformer": {
        "script": "advanced_transformer_ensemble.py",
        "description": "5 Transformer instances per game with different seeds/bootstrap",
        "models_per_game": "5",
        "estimated_time": "90-120 minutes",
        "gpu_required": True
    },
    "Ensemble Variants - LSTM": {
        "script": "advanced_lstm_ensemble.py",
        "description": "3 LSTM instances per game with different seeds/bootstrap",
        "models_per_game": "3",
        "estimated_time": "60-90 minutes",
        "gpu_required": True
    }
}


# ============================================================================
# Session State Management
# ============================================================================

def initialize_ml_training_session():
    """Initialize session state for ML training page."""
    if "ml_training_status" not in st.session_state:
        st.session_state.ml_training_status = {
            "phase_2a_running": False,
            "phase_2b_running": False,
            "phase_2c_running": False,
            "active_trainers": {},
            "training_history": []
        }
    
    if "training_processes" not in st.session_state:
        st.session_state.training_processes = {}


# ============================================================================
# Model Training Status Functions
# ============================================================================

def get_tree_models_status(game_filter: str = None) -> Dict[str, Any]:
    """Get status of Phase 2A tree models, optionally filtered by game."""
    status = {
        "total_models": 0,
        "trained_models": 0,
        "pending_models": 0,
        "models_by_game": {},
        "training_complete": False,
        "estimated_minutes": 60
    }
    
    try:
        games = get_available_games()
        if game_filter and game_filter != "All Games":
            games = [g for g in games if g == game_filter]
        
        for game in games:
            game_dir = MODELS_DIR / game.lower().replace(" ", "_").replace("/", "_")
            models_by_game = {"xgboost": 0, "lightgbm": 0, "catboost": 0}
            
            if game_dir.exists():
                for arch in ["xgboost", "lightgbm", "catboost"]:
                    arch_dir = game_dir / arch
                    if arch_dir.exists():
                        pkl_files = list(arch_dir.glob("*.pkl"))
                        models_by_game[arch] = len(pkl_files)
                        status["trained_models"] += len(pkl_files)
            
            status["models_by_game"][game] = models_by_game
            status["total_models"] += 18  # 6 positions × 3 architectures per game
        
        # Estimate time based on number of games
        if game_filter and game_filter != "All Games":
            status["estimated_minutes"] = 60  # Single game takes ~60-90 min, use lower estimate
        else:
            status["estimated_minutes"] = 120  # All games take ~120-180 min, use middle estimate
    
    except Exception as e:
        app_log(f"Error getting tree models status: {e}", "error")
    
    status["pending_models"] = status["total_models"] - status["trained_models"]
    status["training_complete"] = status["pending_models"] == 0 and status["trained_models"] > 0
    
    return status


def get_neural_network_models_status() -> Dict[str, Any]:
    """Get status of Phase 2B neural network models."""
    status = {
        "lstm": {"trained": 0, "total": 0},
        "transformer": {"trained": 0, "total": 0},
        "cnn": {"trained": 0, "total": 0},
        "by_game": {}
    }
    
    try:
        for game in get_available_games():
            game_dir = MODELS_DIR / game.lower().replace(" ", "_").replace("/", "_")
            status["by_game"][game] = {}
            
            for model_type in ["lstm", "transformer", "cnn"]:
                model_dir = game_dir / model_type
                status["by_game"][game][model_type] = 0
                
                if model_dir.exists():
                    h5_files = list(model_dir.glob("*.h5"))
                    status[model_type]["trained"] += len(h5_files)
                    status["by_game"][game][model_type] = len(h5_files)
                
                status[model_type]["total"] += 1  # 1 per game
    
    except Exception as e:
        app_log(f"Error getting neural network models status: {e}", "error")
    
    return status


def get_ensemble_variants_status() -> Dict[str, Any]:
    """Get status of Phase 2C ensemble variants."""
    status = {
        "transformer_variants": {"trained": 0, "total": 0},
        "lstm_variants": {"trained": 0, "total": 0},
        "total_variants": 0
    }
    
    try:
        for game in get_available_games():
            game_dir = MODELS_DIR / game.lower().replace(" ", "_").replace("/", "_")
            
            # Check transformer variants
            tf_variants_dir = game_dir / "transformer_variants"
            if tf_variants_dir.exists():
                h5_files = list(tf_variants_dir.glob("*.h5"))
                status["transformer_variants"]["trained"] += len(h5_files)
            status["transformer_variants"]["total"] += 5
            
            # Check LSTM variants
            lstm_variants_dir = game_dir / "lstm_variants"
            if lstm_variants_dir.exists():
                h5_files = list(lstm_variants_dir.glob("*.h5"))
                status["lstm_variants"]["trained"] += len(h5_files)
            status["lstm_variants"]["total"] += 3
    
    except Exception as e:
        app_log(f"Error getting ensemble variants status: {e}", "error")
    
    status["total_variants"] = status["transformer_variants"]["total"] + status["lstm_variants"]["total"]
    return status


# ============================================================================
# Training Execution Functions
# ============================================================================

def start_training(trainer_name: str, game: str = None, resume_mode: bool = False) -> bool:
    """Start a training process in a new terminal/subprocess."""
    import subprocess
    import sys
    
    try:
        trainer_config = PHASE_2_TRAINERS.get(trainer_name) or PHASE_2C_VARIANTS.get(trainer_name)
        if not trainer_config:
            st.error(f"Unknown trainer: {trainer_name}")
            return False
        
        script_path = TOOLS_DIR / trainer_config["script"]
        if not script_path.exists():
            st.error(f"Trainer script not found: {script_path}")
            return False
        
        # Build command with game parameter if specified
        cmd = [sys.executable, str(script_path)]
        if game and game != "All Games":
            cmd.extend(["--game", game])
        
        # Add resume flag if enabled
        if resume_mode:
            cmd.append("--resume")
        
        # Create log file for this training session
        log_dir = PROJECT_ROOT / "logs" / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Replace spaces and slashes to create valid filename
        game_safe = game.replace(' ', '_').replace('/', '_') if game and game != "All Games" else "all_games"
        game_suffix = f"_{game_safe}" if game and game != "All Games" else "_all_games"
        log_file = log_dir / f"{trainer_name.replace(' ', '_')}{game_suffix}_{timestamp}.log"
        
        # Use subprocess.Popen with file redirection for better Windows compatibility
        import os as os_module
        env = os_module.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # Use PIPE instead of file handle - we'll read from it in a thread
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            env=env,
            bufsize=1,  # Line buffered
            creationflags=0x08000000 if os_module.name == 'nt' else 0  # CREATE_NO_WINDOW on Windows
        )
        
        # Create log file with initial message
        with open(str(log_file), 'w', buffering=1) as f:
            f.write(f"{'='*80}\n")
            f.write(f"Training started: {trainer_name}\n")
            f.write(f"Game: {game if game != 'All Games' else 'All Games'}\n")
            f.write(f"Resume Mode: {'ENABLED - Will skip existing models' if resume_mode else 'DISABLED - Will retrain all'}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Starting process: {' '.join(cmd)}\n")
            f.write(f"Working directory: {str(PROJECT_ROOT)}\n")
            f.write(f"Process will output below:\n")
            f.write(f"{'='*80}\n\n")
        
        # Create real-time log streamer for line-by-line output capture
        # (RealTimeLogStreamer handles both stdout capture and log file writing)
        from streamlit_app.realtime_log_streamer import RealTimeLogStreamer
        streamer = RealTimeLogStreamer(process, log_file, max_buffer_lines=500)
        streamer.start()
        
        # Store process info
        if "training_processes" not in st.session_state:
            st.session_state.training_processes = {}
        
        process_key = f"{trainer_name}_{datetime.now().timestamp()}"
        display_name = f"{trainer_name}" + (f" - {game}" if game and game != "All Games" else "")
        
        st.session_state.training_processes[process_key] = {
            "pid": process.pid,
            "trainer_name": trainer_name,
            "display_name": display_name,
            "game": game,
            "started_at": datetime.now().isoformat(),
            "script": str(script_path),
            "log_file": str(log_file),
            "process": process,
            "streamer": streamer  # Real-time streaming object for live log display
        }
        
        # Record in history
        if "training_history" not in st.session_state:
            st.session_state.training_history = []
        
        st.session_state.training_history.append({
            "trainer": display_name,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "process_id": process_key,
            "game": game,
            "pid": process.pid
        })
        
        st.success(f"✅ Started {display_name} training (PID: {process.pid})")
        app_log(f"Started training process: {display_name} (PID: {process.pid})", "info")
        
        return True
    
    except Exception as e:
        st.error(f"Error starting training: {e}")
        app_log(f"Error starting training: {e}", "error")
        return False


def get_process_status(process_key: str) -> Dict[str, Any]:
    """Get the status of a running training process."""
    if "training_processes" not in st.session_state:
        return {"status": "not_found"}
    
    proc_info = st.session_state.training_processes.get(process_key)
    if not proc_info:
        return {"status": "not_found"}
    
    # Try to get status from process object if available (same session)
    if "process" in proc_info and proc_info["process"] is not None:
        try:
            is_running = proc_info["process"].poll() is None
            return {
                "status": "running" if is_running else "completed",
                "trainer": proc_info["trainer_name"],
                "started_at": proc_info.get("started_at"),
                "pid": proc_info["pid"],
                "return_code": proc_info["process"].returncode if not is_running else None
            }
        except:
            pass
    
    # Fall back to checking PID with psutil
    try:
        pid = proc_info.get("pid")
        if pid and psutil.pid_exists(pid):
            try:
                p = psutil.Process(pid)
                return {
                    "status": "running",
                    "trainer": proc_info["trainer_name"],
                    "started_at": proc_info.get("started_at"),
                    "pid": pid,
                    "return_code": None
                }
            except psutil.NoSuchProcess:
                return {
                    "status": "completed",
                    "trainer": proc_info["trainer_name"],
                    "started_at": proc_info.get("started_at"),
                    "pid": pid,
                    "return_code": -1
                }
        else:
            return {
                "status": "completed",
                "trainer": proc_info["trainer_name"],
                "started_at": proc_info.get("started_at"),
                "pid": pid,
                "return_code": -1
            }
    except Exception as e:
        app_log(f"Error checking process status: {e}", "error")
        return {"status": "error", "error": str(e)}


def display_process_monitor(process_key: str):
    """Display monitoring info for a running process."""
    status = get_process_status(process_key)
    
    if status["status"] == "not_found":
        st.warning("Process not found or already completed")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = "🟢" if status["status"] == "running" else "🔴"
        st.metric("Status", f"{status_color} {status['status'].upper()}")
    
    with col2:
        st.metric("Process ID", status["pid"])
    
    with col3:
        # Handle both datetime objects and ISO format strings
        started_at = status["started_at"]
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        elapsed = datetime.now() - started_at
        st.metric("Elapsed", f"{elapsed.seconds // 3600}h {(elapsed.seconds % 3600) // 60}m")


def cleanup_training_process(process_key: str):
    """
    Properly cleanup a training process and clear session state.
    Ensures process is terminated, resources freed, and UI state reset.
    """
    if "training_processes" not in st.session_state:
        return
    
    proc_info = st.session_state.training_processes.get(process_key)
    if not proc_info:
        return
    
    try:
        # Terminate process with timeout and force kill fallback
        process = proc_info.get("process")
        if process and process.poll() is None:  # Only if still running
            try:
                process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    except Exception as e:
        app_log(f"Error terminating process {process_key}: {e}", "error")
    
    # Close streamer if present
    try:
        streamer = proc_info.get("streamer")
        if streamer:
            streamer.stop()
    except Exception as e:
        app_log(f"Error stopping streamer: {e}", "error")
    
    # Remove from training processes
    if process_key in st.session_state.training_processes:
        del st.session_state.training_processes[process_key]
    
    # Clear related session state keys
    keys_to_clear = [
        f"{process_key}_trainer_name",
        f"{process_key}_started_at",
        f"{process_key}_progress_bars",
        f"_autorefresh_timer",
        "training_history"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def display_training_logs_window(process_key: str, height: int = 500):
    """
    Display real-time training logs with live streaming from RealTimeLogStreamer.
    Shows metrics, progress bars, and scrollable logs.
    """
    if "training_processes" not in st.session_state or process_key not in st.session_state.training_processes:
        st.warning("Process information not found")
        return
    
    proc_info = st.session_state.training_processes[process_key]
    streamer = proc_info.get("streamer")
    
    # Show metrics from streamer progress (if available) - otherwise skip metrics
    if streamer:
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            with col1:
                pos_text = f"{streamer.progress.position}/{streamer.progress.total_positions}" if streamer.progress.total_positions > 0 else "N/A"
                st.metric("Position", pos_text)
            
            with col2:
                trial_text = f"{streamer.progress.trial}/{streamer.progress.total_trials}" if streamer.progress.total_trials > 0 else "N/A"
                st.metric("Trial", trial_text)
            
            with col3:
                score_delta = "[BEST]" if streamer.progress.is_best else None
                score_val = getattr(streamer.progress, 'score', 0.0)
                st.metric("Score", f"{score_val:.4f}", delta=score_delta)
            
            with col4:
                best_score_val = getattr(streamer.progress, 'best_score', 0.0)
                st.metric("Best", f"{best_score_val:.4f}")
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
            app_log(f"Metrics display error: {e}", "error")
        
        # Progress bars
        col1, col2 = st.columns(2)
        
        with col1:
            if streamer.progress.total_positions > 0:
                st.progress(
                    streamer.progress.position_progress,
                    text=f"Position: {streamer.progress.position}/{streamer.progress.total_positions}"
                )
        
        with col2:
            if streamer.progress.total_trials > 0:
                st.progress(
                    streamer.progress.trial_progress,
                    text=f"Trial: {streamer.progress.trial}/{streamer.progress.total_trials}"
                )
        
        st.divider()
    else:
        st.info("💡 Real-time metrics not available - showing file-based logs")
    
    # Log display - styled to match data_training.py Model Training tab
    st.markdown("### 📜 Training Output")
    
    # Simple approach: Read directly from log file (which IS being written to)
    # The file is actively being appended to by the streamer thread
    log_file = proc_info.get("log_file")
    all_logs = ""
    
    if log_file:
        try:
            from pathlib import Path
            log_path = Path(log_file)
            if log_path.exists():
                with open(str(log_path), 'r', encoding='utf-8', errors='ignore') as f:
                    all_logs = f.read()
            else:
                all_logs = f"Log file not found: {log_file}"
        except Exception as e:
            all_logs = f"Error reading log file: {str(e)}"
    else:
        all_logs = "No log file path available"
    
    # If still empty, show message
    if not all_logs.strip():
        all_logs = "Initializing... logs will appear here as training progresses"
    
    # Format logs for display - replace HTML special chars
    escaped_logs = all_logs.replace("<", "&lt;").replace(">", "&gt;")
    
    # Use light background styling matching data_training.py
    st.markdown(f"""
    <div style="
        height: {height}px; 
        overflow-y: auto; 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        padding: 10px; 
        background-color: #f0f0f0;
        font-family: monospace;
        font-size: 12px;
        line-height: 1.4;
    ">
    {"<br>".join([line.replace("<", "&lt;").replace(">", "&gt;") for line in escaped_logs.split(chr(10))])}
    </div>
    """, unsafe_allow_html=True)
    
    # Show completion status
    process = proc_info.get("process")
    if process:
        return_code = process.poll()
        if return_code is not None:
            # Process has completed
            if return_code == 0:
                st.success("✅ **Training completed successfully!**")
            else:
                st.error(f"❌ **Training failed (exit code: {return_code})**")
        else:
            # Still running - auto-refresh
            try:
                import time
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.warning(f"⚠️ Auto-refresh error: {e}. Please refresh manually.")


# ============================================================================
# Active Logs Update Function  
# ============================================================================

def update_all_active_training_logs():
    """
    Update all active training logs at page level to ensure they're always displayed.
    This is called once per page render to refresh all active process logs.
    """
    if "training_processes" not in st.session_state:
        return
    
    # Find any active training processes
    for process_key, proc_info in st.session_state.training_processes.items():
        if proc_info.get("process") and proc_info["process"].poll() is None:
            # Process is still running - update its logs
            log_content_key = f"training_log_content_{process_key}"
            log_file = proc_info.get("log_file")
            
            if log_file:
                try:
                    from pathlib import Path
                    log_path = Path(log_file)
                    if log_path.exists():
                        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        st.session_state[log_content_key] = content
                except:
                    pass


# ============================================================================
# UI Rendering Functions
# ============================================================================

def render_phase_2a_section(game_filter: str = None):
    """Render Phase 2A - Tree Models section."""
    st.markdown("## 🌳 Phase 2A: Tree Models")
    st.markdown("Position-specific XGBoost, LightGBM, and CatBoost optimization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    tree_status = get_tree_models_status(game_filter)
    
    with col1:
        st.metric(
            "📊 Models Trained",
            f"{tree_status['trained_models']}/{tree_status['total_models']}",
            help="Completed tree models for selected game(s)"
        )
    
    with col2:
        pct = (tree_status['trained_models'] / tree_status['total_models'] * 100) if tree_status['total_models'] > 0 else 0
        st.metric(
            "📈 Progress",
            f"{pct:.1f}%",
            help="Training completion for selected game(s)"
        )
    
    with col3:
        est_min = tree_status['estimated_minutes']
        st.metric(
            "⏱️ Estimated Time",
            f"{est_min}-{est_min+30} min",
            help="Estimated time for full training"
        )
    
    with col4:
        gpu_info = "CPU Only" if not PHASE_2_TRAINERS["Phase 2A - Tree Models"]["gpu_required"] else "GPU Recommended"
        st.metric(
            "💻 Compute",
            gpu_info,
            help="Required computational resources"
        )
    
    st.divider()
    
    # Check if training is already running for this game
    training_running = False
    running_process_key = None
    if "training_processes" in st.session_state:
        for process_key, proc_info in st.session_state.training_processes.items():
            if "Phase 2A" in proc_info["trainer_name"]:
                process = proc_info["process"]
                poll_result = process.poll()
                
                # If process has exited, check if it was successful or failed
                if poll_result is None:
                    # Process still running
                    if game_filter is None or game_filter == "All Games" or proc_info.get("game") == game_filter:
                        training_running = True
                        running_process_key = process_key
                        break
                else:
                    # Process has ended - check return code
                    was_terminated = proc_info.get("was_terminated", False)
                    
                    if poll_result == 0 and not was_terminated:
                        # Process succeeded - show success message and log file
                        st.success(f"✅ **Training completed successfully!** (Process {process_key})")
                        log_file = proc_info.get("log_file")
                        if log_file:
                            try:
                                with open(log_file, 'r', errors='ignore') as f:
                                    content = f.read()
                                    # Show last 1500 chars for context
                                    success_msg = content[-1500:] if len(content) > 1500 else content
                                    with st.expander("📋 Show full training output"):
                                        st.code(success_msg, language="text")
                            except:
                                pass
                    elif poll_result != 0 and not was_terminated:
                        # Process failed (and wasn't manually stopped) - show error from log file
                        log_file = proc_info.get("log_file")
                        error_msg = ""
                        if log_file:
                            try:
                                with open(log_file, 'r', errors='ignore') as f:
                                    content = f.read()
                                    # Show last 1000 chars for more context
                                    error_msg = content[-1000:] if len(content) > 1000 else content
                            except:
                                pass
                        st.error(f"Previous training failed (exit code: {poll_result})")
                        if error_msg:
                            st.code(error_msg, language="text")
                    elif was_terminated:
                        st.info("✅ Training was stopped by user")
    
    # Training Control
    if not training_running:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 🚀 Start Training")
            st.markdown("Execute Phase 2A tree model training for all games and architectures.")
            
            # Add resume checkbox
            resume_enabled = st.checkbox(
                "📋 Resume Mode: Skip existing models",
                value=False,
                key="phase_2a_resume",
                help="If enabled, will skip already trained models and only train missing ones. Much faster for incomplete training."
            )
        
        with col2:
            if st.button("▶️ Start Phase 2A", key="start_phase_2a", use_container_width=True):
                if start_training("Phase 2A - Tree Models", game_filter, resume_mode=resume_enabled):
                    st.session_state.ml_training_status["phase_2a_running"] = True
                    st.rerun()
    else:
        st.markdown("### 📊 Training in Progress...")
        
        # Check if the process has completed
        process_completed = False
        completion_code = None
        if running_process_key and "training_processes" in st.session_state:
            proc_info = st.session_state.training_processes.get(running_process_key)
            if proc_info:
                process = proc_info.get("process")
                if process:
                    completion_code = process.poll()
                    if completion_code is not None:
                        process_completed = True
        
        if process_completed:
            # Process has finished - show final status instead of progress
            if completion_code == 0:
                st.success("✅ **Training completed successfully!**")
            else:
                st.error(f"❌ **Training failed with exit code: {completion_code}**")
        
        # Add progress indicator
        if running_process_key and "training_processes" in st.session_state:
            proc_info = st.session_state.training_processes.get(running_process_key)
            if proc_info and not process_completed:
                log_file = proc_info.get("log_file")
                if log_file:
                    try:
                        from pathlib import Path
                        log_path = Path(log_file)
                        if log_path.exists():
                            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                log_content = f.read()
                            
                            # Extract position from logs (e.g., "Position 3/6")
                            import re
                            match = re.search(r'Position (\d+)/(\d+)', log_content)
                            if match:
                                current = int(match.group(1))
                                total = int(match.group(2))
                                progress = current / total
                                st.progress(progress, text=f"Training position {current}/{total} ({int(progress*100)}%)")
                    except:
                        pass
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if running_process_key:
                display_process_monitor(running_process_key)
        
        with col2:
            button_label = "⏹️ Stop Training" if not process_completed else "✨ Start New Training"
            if st.button(button_label, key="stop_phase_2a", use_container_width=True):
                if running_process_key:
                    if not process_completed:
                        cleanup_training_process(running_process_key)
                        st.success("✅ Training stopped and cleaned up")
                    else:
                        # Clear completed process and show start button
                        if running_process_key in st.session_state.training_processes:
                            del st.session_state.training_processes[running_process_key]
                    st.rerun()
        
        # Always show training logs - whether still running or completed
        st.divider()
        if running_process_key:
            if not process_completed:
                display_training_logs_window(running_process_key, height=500)
            else:
                # Show final logs for completed training
                log_file = None
                if running_process_key and "training_processes" in st.session_state:
                    proc_info = st.session_state.training_processes.get(running_process_key)
                    if proc_info:
                        log_file = proc_info.get("log_file")
                
                if log_file:
                    try:
                        from pathlib import Path
                        log_path = Path(log_file)
                        if log_path.exists():
                            with open(str(log_path), 'r', encoding='utf-8', errors='ignore') as f:
                                all_logs = f.read()
                            
                            st.markdown("### 📜 Training Output")
                            escaped_logs = all_logs.replace("<", "&lt;").replace(">", "&gt;")
                            st.markdown(f"""
                            <div style="
                                height: 600px; 
                                overflow-y: auto; 
                                border: 2px solid #d32f2f; 
                                border-radius: 5px; 
                                padding: 10px; 
                                background-color: #fff3e0;
                                font-family: monospace;
                                font-size: 12px;
                                line-height: 1.4;
                            ">
                            {"<br>".join([line.replace("<", "&lt;").replace(">", "&gt;") for line in escaped_logs.split(chr(10))])}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show completion info
                            st.info(f"💡 Training process ended with exit code: {completion_code}")
                            if completion_code != 0:
                                st.warning("⚠️ Check the logs above to see where the training failed")
                    except:
                        pass
    
    st.divider()
    
    # Status breakdown by game (filtered) - always show, update live during training
    if tree_status['models_by_game']:
        st.markdown("### 📊 Status by Game")
        
        # Filter games based on selection
        games_to_show = tree_status['models_by_game']
        if game_filter and game_filter != "All Games":
            games_to_show = {k: v for k, v in games_to_show.items() if k == game_filter}
        
        if games_to_show:
            for game, models in games_to_show.items():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(f"🎮 {game}", "", help=f"Game: {game}")
                
                with col2:
                    st.metric("XGBoost", models['xgboost'], help="XGBoost models trained")
                
                with col3:
                    st.metric("LightGBM", models['lightgbm'], help="LightGBM models trained")
                
                with col4:
                    st.metric("CatBoost", models['catboost'], help="CatBoost models trained")
        else:
            st.info(f"No data available for {game_filter}")
    
    # Note: Auto-refresh removed - real-time logs now provide sufficient visibility


def render_phase_2b_section(game_filter: str = None):
    """Render Phase 2B - Neural Networks section."""
    st.markdown("## 🧠 Phase 2B: Neural Networks")
    st.markdown("Advanced deep learning architectures for lottery prediction")
    
    nn_status = get_neural_network_models_status()
    
    # Tab structure for each neural network type
    tab1, tab2, tab3 = st.tabs(["🔗 LSTM with Attention", "🔀 Transformer", "📶 CNN"])
    
    with tab1:
        st.markdown("### Encoder-Decoder LSTM with Luong Attention")
        st.markdown("Multi-task learning with 100-draw lookback sequences")
        
        # Check if LSTM training is running OR recently completed
        lstm_training_key = None
        lstm_process_completed = False
        lstm_completion_code = None
        
        if "training_processes" in st.session_state:
            for key, proc_info in st.session_state.training_processes.items():
                if "Phase 2B - LSTM" in key:
                    process = proc_info.get("process")
                    if process:
                        poll_result = process.poll()
                        if poll_result is None:
                            # Still running
                            lstm_training_key = key
                            break
                        else:
                            # Recently completed - still show logs
                            lstm_training_key = key
                            lstm_process_completed = True
                            lstm_completion_code = poll_result
                            break
        
        if not lstm_training_key:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "✅ Trained",
                    f"{nn_status['lstm']['trained']}/{nn_status['lstm']['total']}",
                    help="LSTM models completed"
                )
            
            with col2:
                st.metric("⏱️ Time", "30-45 min", help="Training duration")
            
            with col3:
                st.metric("💾 Size", "~150MB", help="Per model size")
            
            with col4:
                st.metric("💻 GPU", "Required", help="GPU memory needed")
            
            st.divider()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("Start LSTM training for all games")
            with col2:
                if st.button("▶️ Start LSTM", key="start_lstm", use_container_width=True):
                    if start_training("Phase 2B - LSTM with Attention", game_filter):
                        st.rerun()
        else:
            if lstm_process_completed:
                st.markdown("### 📊 Training Completed")
                if lstm_completion_code == 0:
                    st.success("✅ **Training completed successfully!**")
                else:
                    st.error(f"❌ **Training failed with exit code: {lstm_completion_code}**")
            else:
                st.markdown("### 📊 Training in Progress...")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                display_process_monitor(lstm_training_key)
            
            with col2:
                button_label = "🔄 Reset" if lstm_process_completed else "⏹️ Stop Training"
                if st.button(button_label, key="stop_lstm", use_container_width=True):
                    if lstm_training_key:
                        if not lstm_process_completed:
                            cleanup_training_process(lstm_training_key)
                            st.success("✅ Training stopped and cleaned up")
                        else:
                            if lstm_training_key in st.session_state.training_processes:
                                del st.session_state.training_processes[lstm_training_key]
                        st.rerun()
            
            # Show training logs (persist even after completion)
            st.divider()
            if lstm_training_key:
                display_training_logs_window(lstm_training_key, height=500)
    
    with tab2:
        st.markdown("### Decoder-Only Transformer (GPT-like)")
        st.markdown("4-layer transformer with 8-head multi-attention")
        
        # Check if Transformer training is running OR recently completed
        transformer_training_key = None
        transformer_process_completed = False
        transformer_completion_code = None
        
        if "training_processes" in st.session_state:
            for key, proc_info in st.session_state.training_processes.items():
                if "Phase 2B - Transformer" in key:
                    process = proc_info.get("process")
                    if process:
                        poll_result = process.poll()
                        if poll_result is None:
                            # Still running
                            transformer_training_key = key
                            break
                        else:
                            # Recently completed - still show logs
                            transformer_training_key = key
                            transformer_process_completed = True
                            transformer_completion_code = poll_result
                            break
        
        if not transformer_training_key:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "✅ Trained",
                    f"{nn_status['transformer']['trained']}/{nn_status['transformer']['total']}",
                    help="Transformer models completed"
                )
            
            with col2:
                st.metric("⏱️ Time", "45-60 min", help="Training duration")
            
            with col3:
                st.metric("💾 Size", "~180MB", help="Per model size")
            
            with col4:
                st.metric("💻 GPU", "Required", help="GPU memory needed")
            
            st.divider()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("Start Transformer training for all games")
            with col2:
                if st.button("▶️ Start Transformer", key="start_transformer", use_container_width=True):
                    if start_training("Phase 2B - Transformer", game_filter):
                        st.rerun()
        else:
            if transformer_process_completed:
                st.markdown("### 📊 Training Completed")
                if transformer_completion_code == 0:
                    st.success("✅ **Training completed successfully!**")
                else:
                    st.error(f"❌ **Training failed with exit code: {transformer_completion_code}**")
            else:
                st.markdown("### 📊 Training in Progress...")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                display_process_monitor(transformer_training_key)
            
            with col2:
                button_label = "🔄 Reset" if transformer_process_completed else "⏹️ Stop Training"
                if st.button(button_label, key="stop_transformer", use_container_width=True):
                    if transformer_training_key:
                        if not transformer_process_completed:
                            cleanup_training_process(transformer_training_key)
                            st.success("✅ Training stopped and cleaned up")
                        else:
                            if transformer_training_key in st.session_state.training_processes:
                                del st.session_state.training_processes[transformer_training_key]
                        st.rerun()
            
            # Show training logs (persist even after completion)
            st.divider()
            if transformer_training_key:
                display_training_logs_window(transformer_training_key, height=500)
    
    with tab3:
        st.markdown("### 1D Convolutional Neural Network")
        st.markdown("Progressive convolution blocks (64→128→256 filters)")
        
        # Check if CNN training is running OR recently completed
        cnn_training_key = None
        cnn_process_completed = False
        cnn_completion_code = None
        
        if "training_processes" in st.session_state:
            for key, proc_info in st.session_state.training_processes.items():
                if "Phase 2B - CNN" in key:
                    process = proc_info.get("process")
                    if process:
                        poll_result = process.poll()
                        if poll_result is None:
                            # Still running
                            cnn_training_key = key
                            break
                        else:
                            # Recently completed - still show logs
                            cnn_training_key = key
                            cnn_process_completed = True
                            cnn_completion_code = poll_result
                            break
        
        if not cnn_training_key:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "✅ Trained",
                    f"{nn_status['cnn']['trained']}/{nn_status['cnn']['total']}",
                    help="CNN models completed"
                )
            
            with col2:
                st.metric("⏱️ Time", "15-25 min", help="Training duration")
            
            with col3:
                st.metric("💾 Size", "~120MB", help="Per model size")
            
            with col4:
                st.metric("💻 CPU", "Sufficient", help="GPU optional")
            
            st.divider()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("Start CNN training for all games")
            with col2:
                if st.button("▶️ Start CNN", key="start_cnn", use_container_width=True):
                    if start_training("Phase 2B - CNN", game_filter):
                        st.rerun()
        else:
            if cnn_process_completed:
                st.markdown("### 📊 Training Completed")
                if cnn_completion_code == 0:
                    st.success("✅ **Training completed successfully!**")
                else:
                    st.error(f"❌ **Training failed with exit code: {cnn_completion_code}**")
            else:
                st.markdown("### 📊 Training in Progress...")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                display_process_monitor(cnn_training_key)
            
            with col2:
                button_label = "🔄 Reset" if cnn_process_completed else "⏹️ Stop Training"
                if st.button(button_label, key="stop_cnn", use_container_width=True):
                    if cnn_training_key:
                        if not cnn_process_completed:
                            cleanup_training_process(cnn_training_key)
                            st.success("✅ Training stopped and cleaned up")
                        else:
                            if cnn_training_key in st.session_state.training_processes:
                                del st.session_state.training_processes[cnn_training_key]
                        st.rerun()
            
            # Show training logs (persist even after completion)
            st.divider()
            if cnn_training_key:
                display_training_logs_window(cnn_training_key, height=500)


def render_phase_2c_section(game_filter: str = None):
    """Render Phase 2C - Ensemble Variants section."""
    st.markdown("## 🎯 Phase 2C: Ensemble Variants")
    st.markdown("Multiple instances with different seeds and bootstrap sampling")
    st.divider()
    
    ensemble_status = get_ensemble_variants_status()
    
    # Transformer Variants
    with st.expander("🔀 **Transformer Variants** - 5 instances per game", expanded=True):
        # Check if Transformer Variants training is running OR recently completed
        tf_variants_training_key = None
        tf_variants_process_completed = False
        tf_variants_completion_code = None
        
        if "training_processes" in st.session_state:
            for key, proc_info in st.session_state.training_processes.items():
                if "Ensemble Variants - Transformer" in key:
                    process = proc_info.get("process")
                    if process:
                        poll_result = process.poll()
                        if poll_result is None:
                            # Still running
                            tf_variants_training_key = key
                            break
                        else:
                            # Recently completed - still show logs
                            tf_variants_training_key = key
                            tf_variants_process_completed = True
                            tf_variants_completion_code = poll_result
                            break
        
        if not tf_variants_training_key:
            st.markdown("**Description:** 5 Transformer instances with different random seeds and bootstrap sampling")
            
            tf_col1, tf_col2, tf_col3, tf_col4 = st.columns(4)
            
            with tf_col1:
                st.metric(
                    "✅ Trained",
                    f"{ensemble_status['transformer_variants']['trained']}/{ensemble_status['transformer_variants']['total']}",
                )
            
            with tf_col2:
                st.metric("⏱️ Estimated Time", "90-120 min")
            
            with tf_col3:
                st.metric("💾 Total Size", "~900MB")
            
            with tf_col4:
                st.metric("💻 Compute", "GPU Required")
            
            st.write("")
            if st.button("▶️ Start Transformer Variants Training", key="start_tf_variants", use_container_width=True):
                if start_training("Ensemble Variants - Transformer", game_filter):
                    st.rerun()
        else:
            if tf_variants_process_completed:
                st.markdown("### 📊 Training Completed")
                if tf_variants_completion_code == 0:
                    st.success("✅ **Training completed successfully!**")
                else:
                    st.error(f"❌ **Training failed with exit code: {tf_variants_completion_code}**")
            else:
                st.markdown("### 📊 Training in Progress...")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                display_process_monitor(tf_variants_training_key)
            
            with col2:
                button_label = "🔄 Reset" if tf_variants_process_completed else "⏹️ Stop Training"
                if st.button(button_label, key="stop_tf_variants", use_container_width=True):
                    if tf_variants_training_key:
                        if not tf_variants_process_completed:
                            cleanup_training_process(tf_variants_training_key)
                            st.success("✅ Training stopped and cleaned up")
                        else:
                            if tf_variants_training_key in st.session_state.training_processes:
                                del st.session_state.training_processes[tf_variants_training_key]
                        st.rerun()
            
            # Show training logs (persist even after completion)
            st.divider()
            if tf_variants_training_key:
                display_training_logs_window(tf_variants_training_key, height=500)
    
    st.write("")
    
    # LSTM Variants
    with st.expander("🔗 **LSTM Variants** - 3 instances per game", expanded=True):
        # Check if LSTM Variants training is running OR recently completed
        lstm_variants_training_key = None
        lstm_variants_process_completed = False
        lstm_variants_completion_code = None
        
        if "training_processes" in st.session_state:
            for key, proc_info in st.session_state.training_processes.items():
                if "Ensemble Variants - LSTM" in key:
                    process = proc_info.get("process")
                    if process:
                        poll_result = process.poll()
                        if poll_result is None:
                            # Still running
                            lstm_variants_training_key = key
                            break
                        else:
                            # Recently completed - still show logs
                            lstm_variants_training_key = key
                            lstm_variants_process_completed = True
                            lstm_variants_completion_code = poll_result
                            break
        
        if not lstm_variants_training_key:
            st.markdown("**Description:** 3 LSTM instances with different random seeds and bootstrap sampling")
            
            lstm_col1, lstm_col2, lstm_col3, lstm_col4 = st.columns(4)
            
            with lstm_col1:
                st.metric(
                    "✅ Trained",
                    f"{ensemble_status['lstm_variants']['trained']}/{ensemble_status['lstm_variants']['total']}",
                )
            
            with lstm_col2:
                st.metric("⏱️ Estimated Time", "60-90 min")
            
            with lstm_col3:
                st.metric("💾 Total Size", "~450MB")
            
            with lstm_col4:
                st.metric("💻 Compute", "GPU Required")
            
            st.write("")
            if st.button("▶️ Start LSTM Variants Training", key="start_lstm_variants", use_container_width=True):
                if start_training("Ensemble Variants - LSTM", game_filter):
                    st.rerun()
        else:
            if lstm_variants_process_completed:
                st.markdown("### 📊 Training Completed")
                if lstm_variants_completion_code == 0:
                    st.success("✅ **Training completed successfully!**")
                else:
                    st.error(f"❌ **Training failed with exit code: {lstm_variants_completion_code}**")
            else:
                st.markdown("### 📊 Training in Progress...")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                display_process_monitor(lstm_variants_training_key)
            
            with col2:
                button_label = "🔄 Reset" if lstm_variants_process_completed else "⏹️ Stop Training"
                if st.button(button_label, key="stop_lstm_variants", use_container_width=True):
                    if lstm_variants_training_key:
                        if not lstm_variants_process_completed:
                            cleanup_training_process(lstm_variants_training_key)
                            st.success("✅ Training stopped and cleaned up")
                        else:
                            if lstm_variants_training_key in st.session_state.training_processes:
                                del st.session_state.training_processes[lstm_variants_training_key]
                        st.rerun()
            
            # Show training logs (persist even after completion)
            st.divider()
            if lstm_variants_training_key:
                display_training_logs_window(lstm_variants_training_key, height=500)


def render_phase_2d_section(game_filter: str = None):
    """Render Phase 2D - Model Leaderboard & Analysis section."""
    st.markdown("## 🏆 Phase 2D - Model Leaderboard & Analysis")
    st.markdown("*Comprehensive evaluation and ranking of all trained models for production deployment*")
    st.markdown("*Using top-level game filter from page header*")
    
    st.divider()
    
    try:
        # Import Phase 2D module
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
        from phase_2d_leaderboard import Phase2DLeaderboard
        
        # Action Buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Leaderboard", key="phase2d_leaderboard", use_container_width=True):
                with st.spinner("Scanning and evaluating all models..."):
                    leaderboard = Phase2DLeaderboard()
                    
                    # Determine game filter
                    game_param = None
                    if game_filter and game_filter != "All Games":
                        game_param = game_filter.lower().replace(" ", "_")
                    
                    df = leaderboard.generate_leaderboard(game_param)
                    
                    if not df.empty:
                        set_session_value("phase2d_leaderboard_df", df)
                        set_session_value("phase2d_promoted_models", [])  # Initialize promoted models list
                        st.success(f"✅ Leaderboard generated with {len(df)} models")
                        st.rerun()
                    else:
                        st.warning("⚠️ No models found. Train models first in Phase 2A/2B/2C")
        
        with col2:
            if st.button("🎫 Generate Model Cards", key="phase2d_cards", use_container_width=True):
                promoted_models = get_session_value("phase2d_promoted_models", [])
                
                if not promoted_models:
                    st.warning("⚠️ Please promote models first using the 'Model Ranking' tab")
                else:
                    with st.spinner("Generating detailed model cards for promoted models..."):
                        leaderboard = Phase2DLeaderboard()
                        
                        game_param = None
                        if game_filter and game_filter != "All Games":
                            game_param = game_filter.lower().replace(" ", "_")
                        
                        df = leaderboard.generate_leaderboard(game_param)
                        
                        if not df.empty:
                            # Filter to only promoted models
                            promoted_df = df[df['model_name'].isin(promoted_models)]
                            
                            if not promoted_df.empty:
                                cards = leaderboard.generate_model_cards(promoted_df, top_n=len(promoted_df))
                                set_session_value("phase2d_model_cards", cards)
                                leaderboard.save_model_cards(cards, game_param or "all")
                                st.success(f"✅ Generated {len(cards)} model cards for promoted models")
                                st.rerun()
                            else:
                                st.warning("⚠️ No promoted models found in leaderboard")
        
        with col3:
            if st.button("💾 Export Results", key="phase2d_export", use_container_width=True):
                promoted_models = get_session_value("phase2d_promoted_models", [])
                
                if not promoted_models:
                    st.warning("⚠️ Please promote models first before exporting")
                else:
                    with st.spinner("Exporting leaderboard and promoted model cards..."):
                        leaderboard = Phase2DLeaderboard()
                        
                        game_param = None
                        if game_filter and game_filter != "All Games":
                            game_param = game_filter.lower().replace(" ", "_")
                        
                        df = leaderboard.generate_leaderboard(game_param)
                        
                        if not df.empty:
                            leaderboard.save_leaderboard(df, game_param or "all")
                            
                            promoted_df = df[df['model_name'].isin(promoted_models)]
                            if not promoted_df.empty:
                                cards = leaderboard.generate_model_cards(promoted_df, top_n=len(promoted_df))
                                leaderboard.save_model_cards(cards, game_param or "all")
                                st.success(f"✅ Results exported:\n  📁 models/advanced/leaderboards/\n  📁 models/advanced/model_cards/\n  📊 Promoted {len(promoted_models)} models for prediction engine")
                            st.rerun()
        
        st.divider()
        
        # Display Comprehensive Leaderboard
        leaderboard_df = get_session_value("phase2d_leaderboard_df")
        if leaderboard_df is not None and not leaderboard_df.empty:
            st.markdown("### 📊 Comprehensive Model Leaderboard")
            
            # Overall Statistics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Models", len(leaderboard_df))
            with col2:
                phase_2a_count = len(leaderboard_df[leaderboard_df['phase'] == '2A'])
                st.metric("Phase 2A (Trees)", phase_2a_count)
            with col3:
                phase_2b_count = len(leaderboard_df[leaderboard_df['phase'] == '2B'])
                st.metric("Phase 2B (Neural)", phase_2b_count)
            with col4:
                phase_2c_count = len(leaderboard_df[leaderboard_df['phase'] == '2C'])
                st.metric("Phase 2C (Variants)", phase_2c_count)
            with col5:
                st.metric("Top Score", f"{leaderboard_df['composite_score'].max():.4f}")
            
            st.divider()
            
            # Phase 2A - Tree Models Section
            st.markdown("#### 🌳 Phase 2A - Tree Models")
            phase_2a_df = leaderboard_df[leaderboard_df['phase'] == '2A'].copy()
            if not phase_2a_df.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tree Models", len(phase_2a_df))
                with col2:
                    st.metric("Avg Score", f"{phase_2a_df['composite_score'].mean():.4f}")
                with col3:
                    st.metric("Best Score", f"{phase_2a_df['composite_score'].max():.4f}")
                
                # Display table
                display_cols = ['rank', 'model_name', 'model_type', 'composite_score', 'top_5_accuracy', 'ensemble_weight']
                display_df = phase_2a_df[display_cols].copy()
                display_df.columns = ['Rank', 'Model', 'Type', 'Score', 'Top-5', 'Weight']
                display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.4f}")
                display_df['Top-5'] = display_df['Top-5'].apply(lambda x: f"{x:.1%}")
                display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.4f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No Phase 2A tree models found")
            
            st.divider()
            
            # Phase 2B - Neural Networks Section
            st.markdown("#### 🧠 Phase 2B - Neural Networks")
            phase_2b_df = leaderboard_df[leaderboard_df['phase'] == '2B'].copy()
            if not phase_2b_df.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Neural Models", len(phase_2b_df))
                with col2:
                    st.metric("Avg Score", f"{phase_2b_df['composite_score'].mean():.4f}")
                with col3:
                    st.metric("Best Score", f"{phase_2b_df['composite_score'].max():.4f}")
                
                # Display table
                display_cols = ['rank', 'model_name', 'model_type', 'composite_score', 'top_5_accuracy', 'ensemble_weight']
                display_df = phase_2b_df[display_cols].copy()
                display_df.columns = ['Rank', 'Model', 'Type', 'Score', 'Top-5', 'Weight']
                display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.4f}")
                display_df['Top-5'] = display_df['Top-5'].apply(lambda x: f"{x:.1%}")
                display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.4f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No Phase 2B neural network models found")
            
            st.divider()
            
            # Phase 2C - Ensemble Variants Section
            st.markdown("#### 🎯 Phase 2C - Ensemble Variants")
            phase_2c_df = leaderboard_df[leaderboard_df['phase'] == '2C'].copy()
            if not phase_2c_df.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Variant Models", len(phase_2c_df))
                with col2:
                    st.metric("Avg Score", f"{phase_2c_df['composite_score'].mean():.4f}")
                with col3:
                    st.metric("Best Score", f"{phase_2c_df['composite_score'].max():.4f}")
                
                # Display table
                display_cols = ['rank', 'model_name', 'model_type', 'composite_score', 'top_5_accuracy', 'ensemble_weight']
                display_df = phase_2c_df[display_cols].copy()
                display_df.columns = ['Rank', 'Model', 'Type', 'Score', 'Top-5', 'Weight']
                display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.4f}")
                display_df['Top-5'] = display_df['Top-5'].apply(lambda x: f"{x:.1%}")
                display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.4f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No Phase 2C ensemble variant models found")
            
            st.divider()
            
            # Model Details & Analysis with Hierarchical Selectors
            st.markdown("### 📋 Model Details & Analysis")
            
            analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["🔍 Model Explorer", "📊 Comparison", "📈 Model Ranking"])
            
            with analysis_tab1:
                st.markdown("#### Hierarchical Model Explorer")
                
                # Group selector
                group_options = ["All", "Tree Models (2A)", "Neural Networks (2B)", "Ensemble Variants (2C)"]
                selected_group = st.selectbox("Select Model Group:", group_options, key="group_selector")
                
                # Filter by group
                if selected_group == "Tree Models (2A)":
                    group_df = leaderboard_df[leaderboard_df['phase'] == '2A']
                elif selected_group == "Neural Networks (2B)":
                    group_df = leaderboard_df[leaderboard_df['phase'] == '2B']
                elif selected_group == "Ensemble Variants (2C)":
                    group_df = leaderboard_df[leaderboard_df['phase'] == '2C']
                else:
                    group_df = leaderboard_df
                
                if not group_df.empty:
                    # Type selector
                    type_options = sorted(group_df['model_type'].unique().tolist())
                    selected_type = st.selectbox("Select Model Type:", ["All"] + type_options, key="type_selector")
                    
                    # Filter by type
                    if selected_type != "All":
                        type_df = group_df[group_df['model_type'] == selected_type]
                    else:
                        type_df = group_df
                    
                    if not type_df.empty:
                        # Model selector
                        model_options = sorted(type_df['model_name'].tolist())
                        selected_model = st.selectbox("Select Model:", model_options, key="model_selector")
                        selected_row = type_df[type_df['model_name'] == selected_model].iloc[0]
                        
                        # Display detailed information
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Model Info**")
                            st.write(f"Rank: #{int(selected_row['rank'])}")
                            st.write(f"Phase: {selected_row['phase']}")
                            st.write(f"Type: {selected_row['model_type'].upper()}")
                            st.write(f"Architecture: {selected_row['architecture']}")
                            st.write(f"Game: {selected_row['game']}")
                        
                        with col2:
                            st.markdown("**Performance Metrics**")
                            st.metric("Composite Score", f"{selected_row['composite_score']:.4f}")
                            st.metric("Top-5 Accuracy", f"{selected_row['top_5_accuracy']:.2%}")
                            st.metric("Top-10 Accuracy", f"{selected_row['top_10_accuracy']:.2%}")
                            st.metric("KL Divergence", f"{selected_row['kl_divergence']:.4f}")
                        
                        with col3:
                            st.markdown("**Production Metrics**")
                            st.metric("Health Score", f"{selected_row['health_score']:.4f}")
                            st.metric("Ensemble Weight", f"{selected_row['ensemble_weight']:.4f}")
                            if pd.notna(selected_row['seed']):
                                st.metric("Seed", int(selected_row['seed']))
                        
                        st.divider()
                        
                        st.markdown("**💪 Strength**")
                        st.info(selected_row['strength'])
                        
                        st.markdown("**⚠️ Known Bias**")
                        st.warning(selected_row['known_bias'])
                        
                        st.markdown("**🎯 Recommended Use**")
                        st.success(selected_row['recommended_use'])
                    else:
                        st.info("No models found for selected type")
                else:
                    st.info("No models found for selected group")
            
            with analysis_tab2:
                st.markdown("#### Model Comparison: Tree vs Neural vs Variants")
                
                comparison_data = {
                    'Tree Models (2A)': leaderboard_df[leaderboard_df['phase'] == '2A'],
                    'Neural Networks (2B)': leaderboard_df[leaderboard_df['phase'] == '2B'],
                    'Ensemble Variants (2C)': leaderboard_df[leaderboard_df['phase'] == '2C']
                }
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Tree Models (2A)**")
                    if not comparison_data['Tree Models (2A)'].empty:
                        st.write(f"Count: {len(comparison_data['Tree Models (2A)'])}")
                        st.write(f"Avg Score: {comparison_data['Tree Models (2A)']['composite_score'].mean():.4f}")
                        st.write(f"Best: {comparison_data['Tree Models (2A)']['composite_score'].max():.4f}")
                        st.write(f"Worst: {comparison_data['Tree Models (2A)']['composite_score'].min():.4f}")
                    else:
                        st.write("No models")
                
                with col2:
                    st.markdown("**Neural Networks (2B)**")
                    if not comparison_data['Neural Networks (2B)'].empty:
                        st.write(f"Count: {len(comparison_data['Neural Networks (2B)'])}")
                        st.write(f"Avg Score: {comparison_data['Neural Networks (2B)']['composite_score'].mean():.4f}")
                        st.write(f"Best: {comparison_data['Neural Networks (2B)']['composite_score'].max():.4f}")
                        st.write(f"Worst: {comparison_data['Neural Networks (2B)']['composite_score'].min():.4f}")
                    else:
                        st.write("No models")
                
                with col3:
                    st.markdown("**Ensemble Variants (2C)**")
                    if not comparison_data['Ensemble Variants (2C)'].empty:
                        st.write(f"Count: {len(comparison_data['Ensemble Variants (2C)'])}")
                        st.write(f"Avg Score: {comparison_data['Ensemble Variants (2C)']['composite_score'].mean():.4f}")
                        st.write(f"Best: {comparison_data['Ensemble Variants (2C)']['composite_score'].max():.4f}")
                        st.write(f"Worst: {comparison_data['Ensemble Variants (2C)']['composite_score'].min():.4f}")
                    else:
                        st.write("No models")
                
                st.divider()
                
                # Score Distribution Comparison
                st.markdown("**Score Distribution by Phase**")
                comparison_df = leaderboard_df[['phase', 'composite_score']].copy()
                st.bar_chart(comparison_df.set_index('phase')['composite_score'].value_counts().sort_index())
                
                # Accuracy Comparison
                st.markdown("**Top-5 Accuracy by Phase**")
                accuracy_data = {
                    '2A': leaderboard_df[leaderboard_df['phase'] == '2A']['top_5_accuracy'].mean(),
                    '2B': leaderboard_df[leaderboard_df['phase'] == '2B']['top_5_accuracy'].mean(),
                    '2C': leaderboard_df[leaderboard_df['phase'] == '2C']['top_5_accuracy'].mean()
                }
                st.bar_chart(pd.Series(accuracy_data))
            
            with analysis_tab3:
                st.markdown("#### 📈 Model Ranking - Promotion for Production")
                
                promoted_models = get_session_value("phase2d_promoted_models", [])
                
                # Display all models ranked from best to worst
                ranking_df = leaderboard_df[['rank', 'model_name', 'model_type', 'phase', 'composite_score', 'top_5_accuracy']].copy()
                ranking_df = ranking_df.sort_values('rank').reset_index(drop=True)
                
                st.markdown("**All Models Ranked by Composite Score**")
                
                # Create a container for the ranking display
                for idx, row in ranking_df.iterrows():
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 1.5, 1, 1])
                    
                    with col1:
                        st.markdown(f"### #{int(row['rank'])}")
                    
                    with col2:
                        st.markdown(f"**{row['model_name']}**")
                        st.caption(f"{row['phase']} | {row['model_type'].upper()}")
                    
                    with col3:
                        st.metric("Score", f"{row['composite_score']:.4f}")
                    
                    with col4:
                        st.metric("Top-5", f"{row['top_5_accuracy']:.1%}")
                    
                    with col5:
                        # Promote/Demote button
                        model_name = row['model_name']
                        is_promoted = model_name in promoted_models
                        
                        if is_promoted:
                            if st.button("❌ Demote", key=f"demote_{idx}"):
                                promoted_models.remove(model_name)
                                set_session_value("phase2d_promoted_models", promoted_models)
                                st.success(f"Demoted: {model_name}")
                                st.rerun()
                        else:
                            if st.button("✅ Promote", key=f"promote_{idx}"):
                                promoted_models.append(model_name)
                                set_session_value("phase2d_promoted_models", promoted_models)
                                st.success(f"Promoted: {model_name}")
                                st.rerun()
                    
                    st.divider()
                
                # Summary of promoted models
                st.markdown("#### ✅ Promoted Models for Production Engine")
                if promoted_models:
                    promoted_df = leaderboard_df[leaderboard_df['model_name'].isin(promoted_models)].sort_values('rank')
                    
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.success(f"**Total Promoted: {len(promoted_models)} models**")
                        for i, model in enumerate(promoted_models, 1):
                            model_row = leaderboard_df[leaderboard_df['model_name'] == model].iloc[0]
                            st.write(f"{i}. {model} - Score: {model_row['composite_score']:.4f}")
                    
                    with col2:
                        st.info(f"**Statistics**")
                        st.write(f"Count: {len(promoted_df)}")
                        st.write(f"Avg Score: {promoted_df['composite_score'].mean():.4f}")
                        st.write(f"Best: {promoted_df['composite_score'].max():.4f}")
                else:
                    st.warning("No models promoted yet. Click 'Promote' buttons above to select models for the prediction engine.")
        
        else:
            st.info("ℹ️ Click '📊 Generate Leaderboard' to evaluate all trained models")
    
    except ImportError as e:
        st.error(f"❌ Failed to import Phase 2D module: {e}")
    except Exception as e:
        st.error(f"❌ Error in Phase 2D: {e}")
        import traceback
        st.write(traceback.format_exc())


def render_training_summary(game_filter: str = None):
    """Render overall training summary."""
    st.markdown("## 📊 Complete Training Summary")
    
    tree_status = get_tree_models_status()
    nn_status = get_neural_network_models_status()
    ensemble_status = get_ensemble_variants_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_phase_2a = tree_status['trained_models']
        st.metric("🌳 Phase 2A", f"{total_phase_2a} models", help="Tree models trained")
    
    with col2:
        total_phase_2b = (nn_status['lstm']['trained'] + 
                         nn_status['transformer']['trained'] + 
                         nn_status['cnn']['trained'])
        st.metric("🧠 Phase 2B", f"{total_phase_2b} models", help="Neural networks trained")
    
    with col3:
        total_phase_2c = (ensemble_status['transformer_variants']['trained'] + 
                         ensemble_status['lstm_variants']['trained'])
        st.metric("🎯 Phase 2C", f"{total_phase_2c} variants", help="Ensemble variants trained")
    
    with col4:
        total_all = total_phase_2a + total_phase_2b + total_phase_2c
        st.metric("✨ Total", f"{total_all} models", help="All models across all phases")
    
    st.divider()
    
    # Training timeline
    st.markdown("### ⏱️ Estimated Total Timeline")
    
    timeline_data = {
        "Phase": ["2A - Tree Models", "2B - LSTM", "2B - Transformer", "2B - CNN", 
                  "2C - Transformer Variants", "2C - LSTM Variants"],
        "Estimated Time": ["60-90 min", "30-45 min", "45-60 min", "15-25 min", "90-120 min", "60-90 min"],
        "Status": ["⏳ Pending" if not tree_status['training_complete'] else "✅ Complete",
                  "⏳ Pending",
                  "⏳ Pending",
                  "⏳ Pending",
                  "⏳ Pending",
                  "⏳ Pending"]
    }
    
    df_timeline = pd.DataFrame(timeline_data)
    st.dataframe(df_timeline, use_container_width=True, hide_index=True)
    
    st.markdown("**⚠️ Note:** Phases can run in parallel if GPU resources are sufficient.")


def render_phase_1_section(game_filter: str = None):
    """Render Phase 1 - Feature Generation section."""
    st.markdown("## ⚙️ Phase 1: Advanced Feature Generation")
    
    st.markdown("""
    Advanced features have been generated for all lottery games. These include:
    - **Temporal Features**: Time-series patterns and dependencies
    - **Global Features**: Statistical aggregations and distributions
    - **Distribution Targets**: Target encodings for distribution positions
    - **Skipgram Targets**: Sequence prediction targets
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📊 Feature Status")
        st.markdown("Checking feature availability for all games...")
    
    with col2:
        # Check if features exist for all games
        games_to_check = {
            "lotto_6_49": "Lotto 6/49",
            "lotto_max": "Lotto Max"
        }
        
        features_status = {}
        all_features = ["temporal_features.parquet", "global_features.parquet", 
                       "distribution_targets.parquet", "skipgram_targets.parquet"]
        
        for game_short, game_name in games_to_check.items():
            features_dir = PROJECT_ROOT / "data" / "features" / "advanced" / game_short
            all_exist = all((features_dir / f).exists() for f in all_features)
            features_status[game_name] = all_exist
        
        all_complete = all(features_status.values())
        if all_complete:
            st.success("✅ All features exist")
        else:
            missing = [g for g, exists in features_status.items() if not exists]
            st.warning(f"⚠️ Missing features for: {', '.join(missing)}")
    
    st.divider()
    
    # Feature status table
    st.markdown("### 📋 Feature Details")
    
    status_data = []
    for game_short, game_name in games_to_check.items():
        features_dir = PROJECT_ROOT / "data" / "features" / "advanced" / game_short
        status_data.append({
            "Game": game_name,
            "Temporal": "✅" if (features_dir / "temporal_features.parquet").exists() else "❌",
            "Global": "✅" if (features_dir / "global_features.parquet").exists() else "❌",
            "Distribution": "✅" if (features_dir / "distribution_targets.parquet").exists() else "❌",
            "Skipgram": "✅" if (features_dir / "skipgram_targets.parquet").exists() else "❌",
            "Status": "🟢 Ready" if all((features_dir / f).exists() for f in all_features) else "🔴 Incomplete"
        })
    
    df_status = pd.DataFrame(status_data)
    st.dataframe(df_status, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Training control
    if "feature_generation_processes" not in st.session_state:
        st.session_state.feature_generation_processes = {}
    
    feature_gen_running = False
    running_process_key = None
    
    if "feature_generation_processes" in st.session_state:
        for process_key, proc_info in st.session_state.feature_generation_processes.items():
            process = proc_info["process"]
            poll_result = process.poll()
            
            if poll_result is None:
                feature_gen_running = True
                running_process_key = process_key
                break
            else:
                if poll_result != 0:
                    log_file = proc_info.get("log_file")
                    error_msg = ""
                    if log_file and Path(log_file).exists():
                        try:
                            with open(log_file, 'r', errors='ignore') as f:
                                error_msg = f.read()[-500:]
                        except:
                            pass
                    st.error(f"Feature generation failed (exit code: {poll_result})")
                    if error_msg:
                        st.code(error_msg, language="text")
    
    if not feature_gen_running:
        # Show regenerate button if features exist, else show "Generate"
        all_complete = all(features_status.values())
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if all_complete:
                st.markdown("### ✅ Features Ready")
                st.markdown("All advanced features have been generated. You can proceed to training phases.")
            else:
                st.markdown("### 🚀 Generate Features")
                st.markdown("Generate missing advanced features for model training.")
        
        with col2:
            button_text = "🔄 Regenerate" if all_complete else "▶️ Generate"
            if st.button(button_text, key="start_feature_gen", use_container_width=True):
                # Generate for the selected game or all games
                games_to_gen = ["Lotto 6/49", "Lotto Max"] if game_filter == "All Games" else [game_filter]
                
                cmd = [sys.executable, str(PROJECT_ROOT / "regenerate_features.py")]
                
                log_dir = PROJECT_ROOT / "logs" / "training"
                log_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"feature_generation_{timestamp}.log"
                
                import os as os_module
                env = os_module.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                
                process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    bufsize=1,
                    creationflags=0x08000000 if os_module.name == 'nt' else 0
                )
                
                import threading
                
                def capture_output():
                    try:
                        with open(str(log_file), 'w', buffering=1) as f:
                            for line in process.stdout:
                                f.write(line)
                                f.flush()
                    except Exception as e:
                        try:
                            with open(str(log_file), 'a') as f:
                                f.write(f"\n[Error: {e}]\n")
                        except:
                            pass
                
                capture_thread = threading.Thread(target=capture_output, daemon=True)
                capture_thread.start()
                
                if "feature_generation_processes" not in st.session_state:
                    st.session_state.feature_generation_processes = {}
                
                process_key = f"features_{datetime.now().timestamp()}"
                st.session_state.feature_generation_processes[process_key] = {
                    "pid": process.pid,
                    "process": process,
                    "log_file": str(log_file),
                    "started_at": datetime.now().isoformat(),
                    "command": " ".join(cmd)
                }
                
                st.success("✅ Feature generation started")
                st.rerun()
    
    else:
        st.markdown("### 📊 Feature Generation in Progress...")
        
        if running_process_key:
            proc_info = st.session_state.feature_generation_processes[running_process_key]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", "🟢 Running")
            with col2:
                st.metric("Process ID", proc_info["pid"])
            with col3:
                started_at = datetime.fromisoformat(proc_info["started_at"])
                elapsed = datetime.now() - started_at
                st.metric("Elapsed", f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s")
            
            # Display logs
            log_file = proc_info.get("log_file")
            if log_file and Path(log_file).exists():
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if content.strip():
                        lines = content.split('\n')
                        log_output = '\n'.join(lines[-300:]) if len(lines) > 300 else content
                    else:
                        log_output = "Waiting for output..."
                except:
                    log_output = "Reading log..."
            else:
                log_output = "No log file yet..."
            
            st.text_area(
                "Feature Generation Output",
                value=log_output,
                height=400,
                disabled=True,
                key=f"feature_log_{running_process_key}_{int(time.time() * 1000) % 100000}"
            )
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("⏹️ Stop Generation", key="stop_feature_gen", use_container_width=True):
                    try:
                        proc_info["process"].terminate()
                        proc_info["process"].wait(timeout=5)
                        st.success("✅ Generation stopped")
                    except:
                        proc_info["process"].kill()
                        st.success("✅ Generation force-stopped")
                    st.rerun()


def render_advanced_ml_training_page(services_registry=None, ai_engines=None, components=None) -> None:
    """Main render function for Advanced ML Training page."""
    try:
        initialize_ml_training_session()
        
        st.title("🤖 Advanced ML Training - Phase 2")
        st.markdown("*Execute and monitor Phase 2 advanced model training pipeline*")
        
        # Update all active training logs at page level
        update_all_active_training_logs()
        
        # Game Selection
        if "selected_ml_game" not in st.session_state:
            st.session_state.selected_ml_game = "All Games"
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### Select Game Type")
        with col2:
            game_options = ["All Games", "Lotto 6/49", "Lotto Max"]
            st.session_state.selected_ml_game = st.selectbox(
                "Game:",
                game_options,
                index=game_options.index(st.session_state.selected_ml_game),
                key="game_selector"
            )
        
        st.divider()
        
        # Main tabs
        tab_overview, tab_phase1, tab_phase2a, tab_phase2b, tab_phase2c, tab_phase2d, tab_monitor = st.tabs([
            "📊 Overview",
            "⚙️ Phase 1",
            "🌳 Phase 2A",
            "🧠 Phase 2B",
            "🎯 Phase 2C",
            "🏆 Phase 2D",
            "📈 Monitor"
        ])
        
        with tab_overview:
            render_training_summary(st.session_state.selected_ml_game)
        
        with tab_phase1:
            render_phase_1_section(st.session_state.selected_ml_game)
        
        with tab_phase2a:
            render_phase_2a_section(st.session_state.selected_ml_game)
        
        with tab_phase2b:
            render_phase_2b_section(st.session_state.selected_ml_game)
        
        with tab_phase2c:
            render_phase_2c_section(st.session_state.selected_ml_game)
        
        with tab_phase2d:
            # Use top-level game filter
            current_game = None if st.session_state.selected_ml_game == "All Games" else st.session_state.selected_ml_game
            render_phase_2d_section(current_game)
        
        with tab_monitor:
            st.markdown("## 📈 Training Monitor")
            st.markdown("Real-time monitoring of active training processes")
            st.divider()
            
            # Debug info
            with st.expander("🔍 Debug Info"):
                st.write(f"Session State Keys: {list(st.session_state.keys())}")
                if "training_processes" in st.session_state:
                    st.write(f"Training Processes: {len(st.session_state.training_processes)} total")
                    for key, proc in st.session_state.training_processes.items():
                        st.write(f"  - {key}: Status={proc['process'].poll() is None} (running={proc['process'].poll() is None})")
                else:
                    st.write("No training_processes in session state")
            
            # Check for active processes
            has_training_processes = "training_processes" in st.session_state and st.session_state.training_processes
            
            if has_training_processes:
                active_processes = {
                    k: v for k, v in st.session_state.training_processes.items()
                    if v["process"].poll() is None
                }
                
                if active_processes:
                    st.markdown("### 🟢 Active Training Sessions")
                    
                    for process_key, proc_info in active_processes.items():
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            display_process_monitor(process_key)
                        
                        with col2:
                            if st.button(
                                "⏹️ Stop",
                                key=f"stop_{process_key}",
                                use_container_width=True
                            ):
                                try:
                                    proc_info["process"].terminate()
                                    proc_info["process"].wait(timeout=5)
                                    st.success("✅ Process stopped")
                                except:
                                    proc_info["process"].kill()
                                    st.success("✅ Process force-stopped")
                                st.rerun()
                        
                        st.divider()
                else:
                    st.info("ℹ️ No active training processes at this moment (check history below)")
            else:
                st.warning("⚠️ No training session data in memory. Start a training from one of the tabs above.")
            
            # Show history
            if st.session_state.get("training_history"):
                st.markdown("### 📋 Recent Training History")
                history_df = pd.DataFrame(st.session_state.training_history)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Auto-refresh mechanism for real-time log streaming
        # With RealTimeLogStreamer, we can use more aggressive polling without flickering
        # because logs are read from queue, not from files
        if "training_processes" in st.session_state and st.session_state.training_processes:
            has_running = any(
                proc_info.get("process") and proc_info["process"].poll() is None
                for proc_info in st.session_state.training_processes.values()
            )
            if has_running:
                # Rerun every 1 second (every ~40 iterations at 25ms per iteration)
                # With streamer, this is safe because we're reading from queue, not files
                st.session_state._autorefresh_timer = st.session_state.get("_autorefresh_timer", 0) + 1
                
                if st.session_state._autorefresh_timer >= 40:
                    st.session_state._autorefresh_timer = 0
                    st.rerun()
        
        app_log("Advanced ML Training page rendered")
    
    except Exception as e:
        st.error(f"Error rendering Advanced ML Training page: {e}")
        app_log(f"Error rendering Advanced ML Training page: {e}", "error")
if __name__ == "__main__":
    render_advanced_ml_training_page()
