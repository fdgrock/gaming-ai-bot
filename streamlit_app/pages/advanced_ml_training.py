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
from datetime import datetime
import subprocess
import json
import time
from dataclasses import dataclass
import logging

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

def get_tree_models_status() -> Dict[str, Any]:
    """Get status of Phase 2A tree models."""
    status = {
        "total_models": 0,
        "trained_models": 0,
        "pending_models": 0,
        "models_by_game": {},
        "training_complete": False
    }
    
    try:
        for game in get_available_games():
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

def start_training(trainer_name: str, game: str = None) -> bool:
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
        
        # Start process via subprocess
        # On Windows, this will run in background
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Store process info
        if "training_processes" not in st.session_state:
            st.session_state.training_processes = {}
        
        process_key = f"{trainer_name}_{datetime.now().timestamp()}"
        st.session_state.training_processes[process_key] = {
            "process": process,
            "trainer_name": trainer_name,
            "started_at": datetime.now(),
            "script": str(script_path)
        }
        
        # Record in history
        if "training_history" not in st.session_state:
            st.session_state.training_history = []
        
        st.session_state.training_history.append({
            "trainer": trainer_name,
            "started_at": datetime.now(),
            "status": "running",
            "process_id": process_key
        })
        
        st.success(f"✅ Started {trainer_name} training (PID: {process.pid})")
        app_log(f"Started training process: {trainer_name} (PID: {process.pid})", "info")
        
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
    
    process = proc_info["process"]
    
    return {
        "status": "running" if process.poll() is None else "completed",
        "trainer": proc_info["trainer_name"],
        "started_at": proc_info["started_at"],
        "pid": process.pid,
        "return_code": process.returncode if process.poll() is not None else None
    }


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
        elapsed = datetime.now() - status["started_at"]
        st.metric("Elapsed", f"{elapsed.seconds // 3600}h {(elapsed.seconds % 3600) // 60}m")


# ============================================================================
# UI Rendering Functions
# ============================================================================

def render_phase_2a_section():
    """Render Phase 2A - Tree Models section."""
    st.markdown("## 🌳 Phase 2A: Tree Models")
    st.markdown("Position-specific XGBoost, LightGBM, and CatBoost optimization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    tree_status = get_tree_models_status()
    
    with col1:
        st.metric(
            "📊 Models Trained",
            f"{tree_status['trained_models']}/{tree_status['total_models']}",
            help="Completed tree models across all games and architectures"
        )
    
    with col2:
        pct = (tree_status['trained_models'] / tree_status['total_models'] * 100) if tree_status['total_models'] > 0 else 0
        st.metric(
            "📈 Progress",
            f"{pct:.1f}%",
            help="Overall training completion"
        )
    
    with col3:
        st.metric(
            "⏱️ Estimated Time",
            "60-90 min",
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
    
    # Check if training is already running
    training_running = False
    running_process_key = None
    if "training_history" in st.session_state:
        for process_key, proc_info in st.session_state.training_processes.items():
            if "Phase 2A" in proc_info["trainer_name"] and proc_info["process"].poll() is None:
                training_running = True
                running_process_key = process_key
                break
    
    # Training Control
    if not training_running:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 🚀 Start Training")
            st.markdown("Execute Phase 2A tree model training for all games and architectures.")
        
        with col2:
            if st.button("▶️ Start Phase 2A", key="start_phase_2a", use_container_width=True):
                if start_training("Phase 2A - Tree Models"):
                    st.session_state.ml_training_status["phase_2a_running"] = True
                    st.rerun()
    else:
        st.markdown("### 📊 Training in Progress...")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if running_process_key:
                display_process_monitor(running_process_key)
        
        with col2:
            if st.button("⏹️ Stop Training", key="stop_phase_2a", use_container_width=True):
                if running_process_key and "training_processes" in st.session_state:
                    proc_info = st.session_state.training_processes.get(running_process_key)
                    if proc_info:
                        try:
                            proc_info["process"].terminate()
                            st.session_state.training_processes[running_process_key]["process"].wait(timeout=5)
                            st.success("✅ Training stopped")
                        except:
                            proc_info["process"].kill()
                            st.success("✅ Training forcefully stopped")
                        st.rerun()
    
    st.divider()
    
    # Auto-refresh every 5 seconds while training
    if training_running:
        st.markdown("*Refreshing every 5 seconds...* ⟳")
        time.sleep(5)
        st.rerun()
    
    # Status breakdown by game
    if tree_status['models_by_game']:
        st.markdown("### 📊 Status by Game")
        
        for game, models in tree_status['models_by_game'].items():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"🎮 {game}", "", help=f"Game: {game}")
            
            with col2:
                st.metric("XGBoost", models['xgboost'], help="XGBoost models trained")
            
            with col3:
                st.metric("LightGBM", models['lightgbm'], help="LightGBM models trained")
            
            with col4:
                st.metric("CatBoost", models['catboost'], help="CatBoost models trained")


def render_phase_2b_section():
    """Render Phase 2B - Neural Networks section."""
    st.markdown("## 🧠 Phase 2B: Neural Networks")
    st.markdown("Advanced deep learning architectures for lottery prediction")
    
    nn_status = get_neural_network_models_status()
    
    # Tab structure for each neural network type
    tab1, tab2, tab3 = st.tabs(["🔗 LSTM with Attention", "🔀 Transformer", "📶 CNN"])
    
    with tab1:
        st.markdown("### Encoder-Decoder LSTM with Luong Attention")
        st.markdown("Multi-task learning with 100-draw lookback sequences")
        
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
                if start_training("Phase 2B - LSTM with Attention"):
                    st.rerun()
    
    with tab2:
        st.markdown("### Decoder-Only Transformer (GPT-like)")
        st.markdown("4-layer transformer with 8-head multi-attention")
        
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
                if start_training("Phase 2B - Transformer"):
                    st.rerun()
    
    with tab3:
        st.markdown("### 1D Convolutional Neural Network")
        st.markdown("Progressive convolution blocks (64→128→256 filters)")
        
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
                if start_training("Phase 2B - CNN"):
                    st.rerun()


def render_phase_2c_section():
    """Render Phase 2C - Ensemble Variants section."""
    st.markdown("## 🎯 Phase 2C: Ensemble Variants")
    st.markdown("Multiple instances with different seeds and bootstrap sampling")
    
    ensemble_status = get_ensemble_variants_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔀 Transformer Variants")
        st.markdown("5 instances per game with different random seeds")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric(
                "✅ Trained",
                f"{ensemble_status['transformer_variants']['trained']}/{ensemble_status['transformer_variants']['total']}",
                help="Transformer variants completed"
            )
        
        with col_b:
            st.metric("⏱️ Time", "90-120 min", help="Training duration")
        
        with col_c:
            st.metric("💾 Size", "~900MB", help="Total for 10 models")
        
        with col_d:
            st.metric("💻 GPU", "Required", help="GPU memory needed")
        
        st.divider()
        
        if st.button("▶️ Start Transformer Variants", key="start_tf_variants", use_container_width=True):
            if start_training("Ensemble Variants - Transformer"):
                st.rerun()
    
    with col2:
        st.markdown("### 🔗 LSTM Variants")
        st.markdown("3 instances per game with different random seeds")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric(
                "✅ Trained",
                f"{ensemble_status['lstm_variants']['trained']}/{ensemble_status['lstm_variants']['total']}",
                help="LSTM variants completed"
            )
        
        with col_b:
            st.metric("⏱️ Time", "60-90 min", help="Training duration")
        
        with col_c:
            st.metric("💾 Size", "~450MB", help="Total for 6 models")
        
        with col_d:
            st.metric("💻 GPU", "Required", help="GPU memory needed")
        
        st.divider()
        
        if st.button("▶️ Start LSTM Variants", key="start_lstm_variants", use_container_width=True):
            if start_training("Ensemble Variants - LSTM"):
                st.rerun()


def render_training_summary():
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


def render_advanced_ml_training_page(services_registry=None, ai_engines=None, components=None) -> None:
    """Main render function for Advanced ML Training page."""
    try:
        initialize_ml_training_session()
        
        st.title("🤖 Advanced ML Training - Phase 2")
        st.markdown("*Execute and monitor Phase 2 advanced model training pipeline*")
        
        # Main tabs
        tab_overview, tab_phase2a, tab_phase2b, tab_phase2c, tab_monitor = st.tabs([
            "📊 Overview",
            "🌳 Phase 2A",
            "🧠 Phase 2B",
            "🎯 Phase 2C",
            "📈 Monitor"
        ])
        
        with tab_overview:
            render_training_summary()
        
        with tab_phase2a:
            render_phase_2a_section()
        
        with tab_phase2b:
            render_phase_2b_section()
        
        with tab_phase2c:
            render_phase_2c_section()
        
        with tab_monitor:
            st.markdown("## 📈 Training Monitor")
            st.markdown("Real-time monitoring of active training processes")
            
            if st.session_state.get("training_history"):
                st.markdown("### Recent Training Sessions")
                history_df = pd.DataFrame(st.session_state.training_history)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("No training sessions started yet")
        
        app_log("Advanced ML Training page rendered")
    
    except Exception as e:
        st.error(f"Error rendering Advanced ML Training page: {e}")
        app_log(f"Error rendering Advanced ML Training page: {e}", "error")


# This allows the page to be called directly
if __name__ == "__main__":
    render_advanced_ml_training_page()
