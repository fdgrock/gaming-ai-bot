"""
Training Status UI Components
Streamlit components for displaying real-time training progress and logs
"""

import streamlit as st
from typing import Dict, List, Any
from datetime import datetime


def display_training_progress_panel(progress_summary: Dict[str, Any], container=None):
    """
    Display overall training progress with metrics and per-model status.
    
    Args:
        progress_summary: Output from TrainingProgressTracker.get_summary()
        container: Optional st.container() to render in
    """
    target = container if container else st
    
    with target.container():
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Overall Progress",
                f"{progress_summary['overall_progress_pct']:.1f}%"
            )
        
        with col2:
            st.metric(
                "✅ Completed",
                f"{progress_summary['completed_models']}/{progress_summary['total_models']}"
            )
        
        with col3:
            elapsed = progress_summary['elapsed_seconds']
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            st.metric("⏱️ Elapsed", f"{hours}h {mins}m")
        
        with col4:
            current = progress_summary['current_model']
            if current:
                st.metric("🔄 Current", current)
            else:
                st.metric("🔄 Current", "None")
        
        st.divider()
        
        # Progress bar
        st.progress(progress_summary['overall_progress_pct'] / 100.0)


def display_per_model_progress(model_details: Dict[str, Dict]):
    """
    Display progress for each individual model.
    
    Args:
        model_details: Dict mapping model names to their progress info
    """
    if not model_details:
        st.info("No models started yet")
        return
    
    with st.expander("📋 **Per-Model Progress Details**", expanded=False):
        for model_name, details in model_details.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{model_name}**")
            
            with col2:
                status = details.get("status", "unknown")
                if status == "completed":
                    st.success("✅ Completed")
                elif status == "in_progress":
                    st.info("🔄 Training")
                elif status == "failed":
                    st.error("❌ Failed")
                else:
                    st.write(status)
            
            with col3:
                progress_pct = details.get("progress", 0)
                st.write(f"{progress_pct:.1f}%")
            
            # Progress bar for this model
            st.progress(progress_pct / 100.0)


def display_training_logs(
    logs: List[Dict[str, Any]],
    height: int = 400,
    max_logs: int = 100
):
    """
    Display training logs in a scrollable container.
    
    Args:
        logs: List of log entries from TrainingLogHandler
        height: Height of the log container in pixels
        max_logs: Maximum number of recent logs to display
    """
    if not logs:
        st.info("No logs yet")
        return
    
    # Show latest logs
    recent_logs = logs[-max_logs:] if len(logs) > max_logs else logs
    
    # Create formatted log display
    log_text = ""
    for log in recent_logs:
        timestamp = log.get("timestamp", "")[:19]  # HH:MM:SS format
        level = log.get("level", "INFO")
        message = log.get("message", "")
        
        # Color code by level
        if level == "ERROR":
            log_text += f"🔴 [{timestamp}] {level}: {message}\n"
        elif level == "WARNING":
            log_text += f"🟡 [{timestamp}] {level}: {message}\n"
        elif level == "SUCCESS" or "completed" in message.lower():
            log_text += f"🟢 [{timestamp}] {level}: {message}\n"
        else:
            log_text += f"⚪ [{timestamp}] {level}: {message}\n"
    
    # Display in scrollable text area
    st.text_area(
        "📜 Training Log Output",
        value=log_text,
        height=height,
        disabled=True,
        key=f"training_logs_{datetime.now().timestamp()}"
    )


def display_training_status_window(
    progress_summary: Dict[str, Any],
    logs: List[Dict[str, Any]],
    model_details: Dict[str, Dict] = None
):
    """
    Display complete training status window with all components.
    
    Args:
        progress_summary: From TrainingProgressTracker.get_summary()
        logs: From TrainingLogHandler.get_logs()
        model_details: Optional detailed model progress
    """
    # Overall progress panel
    display_training_progress_panel(progress_summary)
    
    # Per-model progress if available
    if model_details:
        display_per_model_progress(model_details)
    
    # Training logs
    st.markdown("---")
    st.markdown("### 📜 Training Output Log")
    display_training_logs(logs, height=500, max_logs=200)
