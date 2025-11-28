"""
Model card components for the lottery prediction system.

This module provides comprehensive model information display components
extracted from legacy app.py for consistent, professional presentation across all pages.
Includes model cards, grids, comparison tables, champion displays, and performance charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Check for plotly availability
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ModelCardComponents:
    """
    Comprehensive model card components extracted from legacy app.py.
    Provides reusable model display functionality for consistent presentation.
    """
    
    @staticmethod
    def render_model_card(model_info: Dict, layout: str = "card", 
                         show_actions: bool = True) -> None:
        """
        Render individual model information card with metadata.
        
        Args:
            model_info: Model information dictionary
            layout: Layout style ("card", "compact", "detailed")  
            show_actions: Whether to show action buttons
        """
        try:
            name = model_info.get('name', 'Unknown Model')
            version = model_info.get('version', '1.0.0')
            status = model_info.get('status', 'unknown')
            description = model_info.get('description', 'No description available')
            performance = model_info.get('performance_metrics', {})
            last_updated = model_info.get('last_updated', 'Unknown')
            model_type = model_info.get('type', 'Unknown')
            
            # Container with styling based on layout
            container_style = ModelCardComponents._get_card_style(layout, status)
            
            with st.container():
                st.markdown(container_style, unsafe_allow_html=True)
                
                # Card header
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### 🤖 {name}")
                    if layout != "compact":
                        st.markdown(f"*{model_type} • Version {version}*")
                        st.markdown(f"📝 {description}")
                
                with col2:
                    ModelCardComponents._render_status_badge(status)
                
                # Performance metrics section
                if layout == "detailed" and performance:
                    st.markdown("#### 📊 Performance Metrics")
                    
                    metrics_cols = st.columns(min(4, len(performance)))
                    for i, (metric, value) in enumerate(performance.items()):
                        with metrics_cols[i % 4]:
                            ModelCardComponents._render_metric_card(metric, value)
                
                elif layout == "card" and performance:
                    # Show key metrics in compact format
                    key_metrics = ['accuracy', 'confidence', 'last_score']
                    shown_metrics = {k: v for k, v in performance.items() if k in key_metrics}
                    
                    if shown_metrics:
                        cols = st.columns(len(shown_metrics))
                        for i, (metric, value) in enumerate(shown_metrics.items()):
                            with cols[i]:
                                ModelCardComponents._render_metric_card(metric, value)
                
                # Model metadata
                if layout == "detailed":
                    with st.expander("🔍 Model Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Last Updated:** {last_updated}")
                            st.markdown(f"**Model Type:** {model_type}")
                            st.markdown(f"**Status:** {status}")
                        with col2:
                            st.markdown(f"**Version:** {version}")
                            if 'file_size' in model_info:
                                st.markdown(f"**File Size:** {model_info['file_size']}")
                            if 'training_date' in model_info:
                                st.markdown(f"**Trained:** {model_info['training_date']}")
                
                # Action buttons
                if show_actions:
                    ModelCardComponents._render_action_buttons(model_info)
                
                if layout != "compact":
                    st.markdown("---")
        
        except Exception as e:
            logger.error(f"Error rendering model card: {e}")
            st.error(f"Error displaying model card: {str(e)}")
    
    @staticmethod
    def render_model_grid(models: List[Dict], columns: int = 3,
                         filter_options: Dict = None) -> None:
        """
        Display multiple models in responsive grid layout.
        
        Args:
            models: List of model dictionaries
            columns: Number of columns in grid
            filter_options: Filtering options
        """
        try:
            if not models:
                st.info("🤖 No models available to display")
                return
            
            # Apply filters if provided
            filtered_models = ModelCardComponents._apply_filters(models, filter_options)
            
            st.markdown(f"### 🤖 Model Library ({len(filtered_models)} models)")
            
            # Create grid layout
            for i in range(0, len(filtered_models), columns):
                cols = st.columns(columns)
                
                for j in range(columns):
                    if i + j < len(filtered_models):
                        with cols[j]:
                            ModelCardComponents.render_model_card(
                                filtered_models[i + j], 
                                layout="card", 
                                show_actions=False
                            )
        
        except Exception as e:
            logger.error(f"Error rendering model grid: {e}")
            st.error("Error displaying model grid")
    
    @staticmethod  
    def render_model_comparison_table(models: List[Dict],
                                    comparison_fields: List[str] = None) -> None:
        """
        Compare multiple models in detailed table format.
        
        Args:
            models: List of model dictionaries to compare
            comparison_fields: Fields to compare
        """
        try:
            if not models:
                st.info("No models available for comparison")
                return
            
            # Default comparison fields
            if not comparison_fields:
                comparison_fields = ['name', 'type', 'status', 'accuracy', 'last_updated']
            
            st.subheader("📋 Model Comparison")
            
            # Prepare comparison data
            comparison_data = []
            for model in models:
                row = {}
                
                for field in comparison_fields:
                    if field == 'name':
                        row['Model Name'] = model.get('name', 'Unknown')
                    elif field == 'type':
                        row['Type'] = model.get('type', 'Unknown')
                    elif field == 'status':
                        row['Status'] = model.get('status', 'Unknown')
                    elif field == 'accuracy':
                        performance = model.get('performance_metrics', {})
                        accuracy = performance.get('accuracy', 0)
                        row['Accuracy'] = f"{accuracy:.1%}" if isinstance(accuracy, (int, float)) else str(accuracy)
                    elif field == 'last_updated':
                        row['Last Updated'] = model.get('last_updated', 'Unknown')
                    elif field == 'version':
                        row['Version'] = model.get('version', '1.0.0')
                    else:
                        # Generic field handling
                        row[field.title()] = model.get(field, 'N/A')
                
                comparison_data.append(row)
            
            # Create and display comparison table
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Add download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison",
                    data=csv,
                    file_name="model_comparison.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            logger.error(f"Error rendering model comparison table: {e}")
            st.error("Error displaying model comparison")
    
    @staticmethod
    def render_champion_model_display(champion_info: Dict,
                                    show_promotion_history: bool = True) -> None:
        """
        Special display for champion model with promotion details.
        
        Args:
            champion_info: Champion model information
            show_promotion_history: Whether to show promotion history
        """
        try:
            if not champion_info:
                st.info("🏆 No champion model selected")
                return
            
            st.markdown("### 🏆 Champion Model")
            
            # Champion model card with special styling
            with st.container():
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 15px;
                    color: white;
                    margin-bottom: 20px;
                ">
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    name = champion_info.get('name', 'Unknown Champion')
                    st.markdown(f"## 🏆 {name}")
                    st.markdown(f"**Type:** {champion_info.get('type', 'Unknown')}")
                    st.markdown(f"**Promoted:** {champion_info.get('promotion_date', 'Unknown')}")
                
                with col2:
                    # Champion metrics
                    performance = champion_info.get('performance_metrics', {})
                    if 'accuracy' in performance:
                        st.metric("🎯 Accuracy", f"{performance['accuracy']:.1%}")
                    if 'confidence' in performance:
                        st.metric("💪 Confidence", f"{performance['confidence']:.1%}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Promotion history
            if show_promotion_history and 'promotion_history' in champion_info:
                st.markdown("#### 📈 Promotion History")
                
                history = champion_info['promotion_history']
                for i, promotion in enumerate(history[-5:]):  # Show last 5 promotions
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{promotion.get('date', 'Unknown')}**")
                        st.markdown(f"*{promotion.get('reason', 'No reason provided')}*")
                    
                    with col2:
                        st.markdown(f"Accuracy: {promotion.get('accuracy', 0):.1%}")
                    
                    with col3:
                        st.markdown(f"Score: {promotion.get('score', 0):.3f}")
        
        except Exception as e:
            logger.error(f"Error rendering champion model display: {e}")
            st.error("Error displaying champion model")
    
    @staticmethod
    def render_model_performance_chart(model_id: str, performance_data: Dict,
                                     chart_type: str = "line") -> None:
        """
        Performance visualization for individual model.
        
        Args:
            model_id: Model identifier  
            performance_data: Historical performance data
            chart_type: Type of chart ("line", "bar", "area")
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("📊 Performance charts require plotly. Install with: pip install plotly")
                return
            
            st.subheader(f"📊 Performance History: {model_id}")
            
            if not performance_data or 'history' not in performance_data:
                st.info("No performance history available")
                return
            
            history = performance_data['history']
            
            # Prepare data
            dates = [entry.get('date') for entry in history]
            accuracies = [entry.get('accuracy', 0) for entry in history]
            confidences = [entry.get('confidence', 0) for entry in history]
            
            # Create chart based on type
            fig = go.Figure()
            
            if chart_type == "line":
                fig.add_trace(go.Scatter(
                    x=dates, y=accuracies,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='#28a745')
                ))
                fig.add_trace(go.Scatter(
                    x=dates, y=confidences,
                    mode='lines+markers', 
                    name='Confidence',
                    line=dict(color='#007bff')
                ))
            
            elif chart_type == "bar":
                fig.add_trace(go.Bar(
                    x=dates, y=accuracies,
                    name='Accuracy',
                    marker_color='#28a745'
                ))
                fig.add_trace(go.Bar(
                    x=dates, y=confidences,
                    name='Confidence',
                    marker_color='#007bff'
                ))
            
            elif chart_type == "area":
                fig.add_trace(go.Scatter(
                    x=dates, y=accuracies,
                    fill='tonexty',
                    mode='lines',
                    name='Accuracy',
                    line=dict(color='#28a745')
                ))
                fig.add_trace(go.Scatter(
                    x=dates, y=confidences,
                    fill='tozeroy',
                    mode='lines',
                    name='Confidence', 
                    line=dict(color='#007bff')
                ))
            
            fig.update_layout(
                title=f"{model_id} Performance Over Time",
                xaxis_title="Date",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error rendering model performance chart: {e}")
            st.error("Error displaying performance chart")
    
    @staticmethod
    def render_model_health_status(model_info: Dict,
                                 validation_results: Dict = None) -> None:
        """
        Health and validation status display with indicators.
        
        Args:
            model_info: Model information dictionary
            validation_results: Validation test results
        """
        try:
            st.subheader("🏥 Model Health Status")
            
            # Basic health indicators
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = model_info.get('status', 'unknown')
                if status == 'active':
                    st.success("🟢 Status: Active")
                elif status == 'training':
                    st.warning("🟡 Status: Training")
                elif status == 'error':
                    st.error("🔴 Status: Error")
                else:
                    st.info(f"ℹ️ Status: {status.title()}")
            
            with col2:
                last_check = model_info.get('last_health_check', 'Never')
                st.markdown(f"**Last Check:** {last_check}")
            
            with col3:
                uptime = model_info.get('uptime', 0)
                st.markdown(f"**Uptime:** {uptime} hours")
            
            # Validation results
            if validation_results:
                st.markdown("#### 🧪 Validation Results")
                
                for test_name, result in validation_results.items():
                    if result.get('passed', False):
                        st.success(f"✅ {test_name}: Passed")
                    else:
                        st.error(f"❌ {test_name}: Failed")
                        if 'message' in result:
                            st.markdown(f"   *{result['message']}*")
        
        except Exception as e:
            logger.error(f"Error rendering model health status: {e}")
            st.error("Error displaying health status")
    
    @staticmethod
    def render_model_actions_panel(model_info: Dict, available_actions: List[str],
                                 on_action_callback: Callable = None) -> None:
        """
        Action buttons and controls for model operations.
        
        Args:
            model_info: Model information
            available_actions: List of available actions
            on_action_callback: Callback function for actions
        """
        try:
            st.markdown("#### ⚡ Model Actions")
            
            # Action mapping
            action_config = {
                'train': {'label': '🔄 Train Model', 'color': 'primary'},
                'test': {'label': '🧪 Test Model', 'color': 'secondary'},
                'promote': {'label': '🏆 Promote to Champion', 'color': 'success'},
                'archive': {'label': '📦 Archive Model', 'color': 'warning'},
                'delete': {'label': '🗑️ Delete Model', 'color': 'error'},
                'export': {'label': '💾 Export Model', 'color': 'info'},
                'duplicate': {'label': '📋 Duplicate Model', 'color': 'secondary'}
            }
            
            # Render action buttons
            action_cols = st.columns(min(3, len(available_actions)))
            
            for i, action in enumerate(available_actions):
                with action_cols[i % 3]:
                    config = action_config.get(action, {'label': action.title(), 'color': 'primary'})
                    
                    if st.button(config['label'], key=f"action_{action}_{model_info.get('name', 'unknown')}"):
                        if on_action_callback:
                            on_action_callback(action, model_info)
                        else:
                            st.info(f"Action '{action}' triggered for {model_info.get('name')}")
        
        except Exception as e:
            logger.error(f"Error rendering model actions panel: {e}")
            st.error("Error displaying actions panel")
    
    @staticmethod
    def render_model_metadata_panel(metadata: Dict, 
                                  collapsible: bool = True) -> None:
        """
        Detailed metadata display with organized sections.
        
        Args:
            metadata: Model metadata dictionary
            collapsible: Whether to make sections collapsible
        """
        try:
            if collapsible:
                with st.expander("🔍 Detailed Metadata"):
                    ModelCardComponents._render_metadata_content(metadata)
            else:
                st.subheader("🔍 Model Metadata")
                ModelCardComponents._render_metadata_content(metadata)
        
        except Exception as e:
            logger.error(f"Error rendering model metadata panel: {e}")
            st.error("Error displaying metadata")
    
    @staticmethod
    def render_training_history_display(training_data: Dict,
                                      show_metrics: bool = True) -> None:
        """
        Training history and metrics visualization.
        
        Args:
            training_data: Training history data
            show_metrics: Whether to show detailed metrics
        """
        try:
            st.subheader("📚 Training History")
            
            if not training_data or 'sessions' not in training_data:
                st.info("No training history available")
                return
            
            sessions = training_data['sessions']
            
            # Training sessions timeline
            for i, session in enumerate(sessions[-5:]):  # Show last 5 sessions
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        start_time = session.get('start_time', 'Unknown')
                        duration = session.get('duration', 0)
                        st.markdown(f"**Session {i+1}** - {start_time}")
                        st.markdown(f"Duration: {duration} minutes")
                    
                    with col2:
                        final_loss = session.get('final_loss', 0)
                        st.metric("Final Loss", f"{final_loss:.4f}")
                    
                    with col3:
                        final_accuracy = session.get('final_accuracy', 0)
                        st.metric("Final Accuracy", f"{final_accuracy:.1%}")
                    
                    # Training metrics chart
                    if show_metrics and 'metrics_history' in session:
                        ModelCardComponents._render_training_metrics_chart(
                            session['metrics_history'], f"session_{i+1}"
                        )
                    
                    st.markdown("---")
        
        except Exception as e:
            logger.error(f"Error rendering training history display: {e}")
            st.error("Error displaying training history")
    
    # Utility methods
    @staticmethod
    def _get_card_style(layout: str, status: str) -> str:
        """Get CSS styling for model card based on layout and status."""
        base_style = """
        <style>
        .model-card {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 5px solid;
        }
        .model-card-compact { padding: 10px; }
        .model-card-detailed { padding: 20px; }
        </style>
        """
        
        status_colors = {
            'active': '#28a745',
            'training': '#ffc107',
            'inactive': '#6c757d',
            'error': '#dc3545',
            'ready': '#17a2b8'
        }
        
        border_color = status_colors.get(status.lower(), '#6c757d')
        
        return f"{base_style}<div class='model-card model-card-{layout}' style='border-left-color: {border_color};'>"
    
    @staticmethod
    def _render_status_badge(status: str) -> None:
        """Render status badge with appropriate color."""
        status_config = {
            'active': {'color': '#28a745', 'icon': '🟢'},
            'training': {'color': '#ffc107', 'icon': '🟡'}, 
            'inactive': {'color': '#6c757d', 'icon': '⚫'},
            'error': {'color': '#dc3545', 'icon': '🔴'},
            'ready': {'color': '#17a2b8', 'icon': '🔵'}
        }
        
        config = status_config.get(status.lower(), {'color': '#6c757d', 'icon': '⚪'})
        
        st.markdown(f"""
        <div style="
            background-color: {config['color']};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            text-align: center;
            font-size: 0.875rem;
            font-weight: bold;
        ">
            {config['icon']} {status.upper()}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _render_metric_card(metric: str, value: Any) -> None:
        """Render individual metric card."""
        try:
            # Format value based on type
            if isinstance(value, float):
                if 0 <= value <= 1:
                    formatted_value = f"{value:.1%}"
                else:
                    formatted_value = f"{value:.3f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            
            # Determine metric icon
            metric_icons = {
                'accuracy': '🎯',
                'confidence': '💪',
                'loss': '📉',
                'score': '⭐',
                'precision': '🔍',
                'recall': '🎪',
                'f1_score': '⚖️'
            }
            
            icon = metric_icons.get(metric.lower(), '📊')
            
            st.metric(f"{icon} {metric.title()}", formatted_value)
        
        except Exception as e:
            logger.error(f"Error rendering metric card: {e}")
            st.metric(metric.title(), "Error")
    
    @staticmethod
    def _render_action_buttons(model_info: Dict) -> None:
        """Render default action buttons for a model."""
        col1, col2, col3 = st.columns(3)
        
        model_name = model_info.get('name', 'Unknown')
        
        with col1:
            if st.button("📊 View Details", key=f"details_{model_name}"):
                st.info(f"Viewing details for {model_name}")
        
        with col2:
            if st.button("🧪 Test Model", key=f"test_{model_name}"):
                st.info(f"Testing {model_name}")
        
        with col3:
            if st.button("📥 Export", key=f"export_{model_name}"):
                st.info(f"Exporting {model_name}")
    
    @staticmethod
    def _apply_filters(models: List[Dict], filter_options: Dict) -> List[Dict]:
        """Apply filtering to model list."""
        if not filter_options:
            return models
        
        filtered = models.copy()
        
        # Status filter
        if 'status' in filter_options and filter_options['status'] != 'all':
            filtered = [m for m in filtered if m.get('status') == filter_options['status']]
        
        # Type filter
        if 'type' in filter_options and filter_options['type'] != 'all':
            filtered = [m for m in filtered if m.get('type') == filter_options['type']]
        
        # Performance threshold
        if 'min_accuracy' in filter_options:
            min_acc = filter_options['min_accuracy']
            filtered = [
                m for m in filtered 
                if m.get('performance_metrics', {}).get('accuracy', 0) >= min_acc
            ]
        
        return filtered
    
    @staticmethod
    def _render_metadata_content(metadata: Dict) -> None:
        """Render metadata content in organized sections."""
        # Basic information
        st.markdown("**Basic Information**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"- **Name:** {metadata.get('name', 'Unknown')}")
            st.markdown(f"- **Type:** {metadata.get('type', 'Unknown')}")
            st.markdown(f"- **Version:** {metadata.get('version', '1.0.0')}")
        
        with col2:
            st.markdown(f"- **Created:** {metadata.get('creation_date', 'Unknown')}")
            st.markdown(f"- **Modified:** {metadata.get('last_modified', 'Unknown')}")
            st.markdown(f"- **Size:** {metadata.get('file_size', 'Unknown')}")
        
        # Training information
        if 'training_config' in metadata:
            st.markdown("**Training Configuration**")
            training_config = metadata['training_config']
            for key, value in training_config.items():
                st.markdown(f"- **{key.title()}:** {value}")
        
        # Performance metrics
        if 'performance_metrics' in metadata:
            st.markdown("**Performance Metrics**")
            performance = metadata['performance_metrics']
            cols = st.columns(min(3, len(performance)))
            
            for i, (metric, value) in enumerate(performance.items()):
                with cols[i % 3]:
                    if isinstance(value, float) and 0 <= value <= 1:
                        st.metric(metric.title(), f"{value:.1%}")
                    else:
                        st.metric(metric.title(), str(value))
    
    @staticmethod
    def _render_training_metrics_chart(metrics_history: Dict, session_id: str) -> None:
        """Render training metrics chart for a session."""
        try:
            if not PLOTLY_AVAILABLE:
                st.info("Training metrics visualization requires plotly")
                return
            
            fig = go.Figure()
            
            # Add loss curve
            if 'loss' in metrics_history:
                epochs = list(range(1, len(metrics_history['loss']) + 1))
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=metrics_history['loss'],
                    mode='lines',
                    name='Loss',
                    line=dict(color='red')
                ))
            
            # Add accuracy curve
            if 'accuracy' in metrics_history:
                epochs = list(range(1, len(metrics_history['accuracy']) + 1))
                fig.add_trace(go.Scatter(
                    x=epochs,
                    y=metrics_history['accuracy'],
                    mode='lines',
                    name='Accuracy',
                    line=dict(color='blue'),
                    yaxis='y2'
                ))
            
            fig.update_layout(
                title=f"Training Metrics - {session_id}",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                yaxis2=dict(
                    title="Accuracy",
                    overlaying='y',
                    side='right'
                ),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error rendering training metrics chart: {e}")
            st.error("Error displaying training metrics")


# Backward compatibility class - maintains original interface
class ModelCard:
    """
    Legacy ModelCard class for backward compatibility.
    
    This class maintains the original simple interface while delegating
    to the enhanced ModelCardComponents class for actual functionality.
    """
    
    @staticmethod
    def render_basic_card(model_info: Dict) -> None:
        """Render basic model card (legacy method)."""
        ModelCardComponents.render_model_card(model_info)
    
    @staticmethod
    def render_model_grid(models: List[Dict]) -> None:
        """Render model grid (legacy method)."""
        ModelCardComponents.render_model_grid(models)
    
    @staticmethod
    def show_model_comparison(models: List[Dict]) -> None:
        """Show model comparison (legacy method)."""
        ModelCardComponents.render_model_comparison_table(models)


# Export classes for easy importing
__all__ = ['ModelCardComponents', 'ModelCard']


class ModelCard:
    """Individual model card component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model card.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.theme = self.config.get('theme', 'default')
    
    def render(self, model_data: Dict[str, Any], 
               show_actions: bool = True,
               compact: bool = False) -> None:
        """
        Render model card.
        
        Args:
            model_data: Model information dictionary
            show_actions: Whether to show action buttons
            compact: Whether to show compact view
        """
        try:
            name = model_data.get('name', 'Unknown Model')
            version = model_data.get('version', '1.0.0')
            status = model_data.get('status', 'unknown')
            description = model_data.get('description', 'No description available')
            performance = model_data.get('performance_metrics', {})
            
            with st.container():
                # Card header
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### {name}")
                    if not compact:
                        st.markdown(f"*Version: {version}*")
                        st.markdown(description)
                
                with col2:
                    self._render_status_badge(status)
                
                # Performance metrics (if not compact)
                if not compact and performance:
                    st.markdown("**Performance Metrics:**")
                    
                    metrics_cols = st.columns(len(performance))
                    for i, (metric, value) in enumerate(performance.items()):
                        with metrics_cols[i]:
                            self._render_metric_card(metric, value)
                
                # Action buttons
                if show_actions:
                    self._render_action_buttons(model_data)
                
                st.markdown("---")
                
        except Exception as e:
            logger.error(f"❌ Failed to render model card: {e}")
            st.error(f"Failed to display model card: {e}")
    
    def _render_status_badge(self, status: str) -> None:
        """Render status badge."""
        status_colors = {
            'active': '#28a745',
            'training': '#ffc107',
            'inactive': '#6c757d',
            'error': '#dc3545',
            'ready': '#17a2b8'
        }
        
        color = status_colors.get(status.lower(), '#6c757d')
        
        st.markdown(f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
            text-align: center;
        ">{status}</div>
        """, unsafe_allow_html=True)
    
    def _render_metric_card(self, metric_name: str, value: float) -> None:
        """Render individual metric card."""
        # Format value based on metric type
        if 'accuracy' in metric_name.lower() or 'confidence' in metric_name.lower():
            formatted_value = f"{value:.1%}"
        elif 'time' in metric_name.lower():
            formatted_value = f"{value:.2f}s"
        else:
            formatted_value = f"{value:.3f}"
        
        st.markdown(f"""
        <div style="
            text-align: center;
            padding: 0.5rem;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        ">
            <div style="font-size: 1.2rem; font-weight: bold; color: #0066cc;">
                {formatted_value}
            </div>
            <div style="font-size: 0.8rem; color: #6c757d;">
                {metric_name.replace('_', ' ').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_action_buttons(self, model_data: Dict[str, Any]) -> None:
        """Render action buttons for the model."""
        status = model_data.get('status', 'unknown')
        model_id = model_data.get('id', model_data.get('name', 'unknown'))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if status.lower() in ['inactive', 'ready']:
                if st.button("▶️ Activate", key=f"activate_{model_id}"):
                    st.session_state[f"action_{model_id}"] = "activate"
                    st.rerun()
        
        with col2:
            if status.lower() == 'active':
                if st.button("⏸️ Pause", key=f"pause_{model_id}"):
                    st.session_state[f"action_{model_id}"] = "pause"
                    st.rerun()
        
        with col3:
            if st.button("🔄 Retrain", key=f"retrain_{model_id}"):
                st.session_state[f"action_{model_id}"] = "retrain"
                st.rerun()
        
        with col4:
            if st.button("⚙️ Configure", key=f"config_{model_id}"):
                st.session_state[f"action_{model_id}"] = "configure"
                st.rerun()
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class ModelGrid:
    """Grid layout for multiple model cards."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize model grid."""
        self.config = config or {}
        self.columns_per_row = self.config.get('columns_per_row', 2)
    
    def render(self, models: List[Dict[str, Any]], 
               title: str = "AI Models",
               filter_by_status: Optional[str] = None,
               sort_by: str = 'name') -> None:
        """
        Render model grid.
        
        Args:
            models: List of model data dictionaries
            title: Grid title
            filter_by_status: Filter models by status
            sort_by: Sort models by field
        """
        try:
            if not models:
                st.warning("⚠️ No models to display")
                return
            
            st.subheader(title)
            
            # Filter models if specified
            if filter_by_status:
                models = [m for m in models if m.get('status', '').lower() == filter_by_status.lower()]
            
            # Sort models
            if sort_by in ['name', 'version', 'status']:
                models = sorted(models, key=lambda x: x.get(sort_by, ''))
            elif sort_by == 'performance':
                models = sorted(models, 
                              key=lambda x: x.get('performance_metrics', {}).get('accuracy', 0), 
                              reverse=True)
            
            # Render grid
            for i in range(0, len(models), self.columns_per_row):
                cols = st.columns(self.columns_per_row)
                
                for j in range(self.columns_per_row):
                    if i + j < len(models):
                        with cols[j]:
                            model_card = ModelCard(self.config)
                            model_card.render(models[i + j], compact=True)
            
            # Summary statistics
            self._render_grid_summary(models)
            
        except Exception as e:
            logger.error(f"❌ Failed to render model grid: {e}")
            st.error(f"Failed to display model grid: {e}")
    
    def _render_grid_summary(self, models: List[Dict[str, Any]]) -> None:
        """Render summary statistics for the model grid."""
        try:
            total_models = len(models)
            status_counts = {}
            
            for model in models:
                status = model.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            st.markdown("### Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Models", total_models)
            
            with col2:
                active_count = status_counts.get('active', 0)
                st.metric("Active Models", active_count)
            
            with col3:
                if total_models > 0:
                    active_percentage = (active_count / total_models) * 100
                    st.metric("Active %", f"{active_percentage:.1f}%")
            
            # Status breakdown
            if status_counts:
                st.markdown("**Status Breakdown:**")
                for status, count in status_counts.items():
                    st.write(f"- {status.title()}: {count}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to render grid summary: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class ModelComparison:
    """Component for comparing multiple models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize model comparison."""
        self.config = config or {}
    
    def render(self, models: List[Dict[str, Any]], 
               title: str = "Model Comparison",
               metrics_to_compare: List[str] = None) -> None:
        """
        Render model comparison.
        
        Args:
            models: List of model data
            title: Comparison title
            metrics_to_compare: List of metrics to compare
        """
        try:
            if len(models) < 2:
                st.warning("⚠️ At least 2 models required for comparison")
                return
            
            st.subheader(title)
            
            if not metrics_to_compare:
                metrics_to_compare = ['accuracy', 'confidence', 'training_time']
            
            # Create comparison table
            self._render_comparison_table(models, metrics_to_compare)
            
            # Create comparison charts
            self._render_comparison_charts(models, metrics_to_compare)
            
        except Exception as e:
            logger.error(f"❌ Failed to render model comparison: {e}")
            st.error(f"Failed to display model comparison: {e}")
    
    def _render_comparison_table(self, models: List[Dict[str, Any]], 
                               metrics: List[str]) -> None:
        """Render comparison table."""
        try:
            table_data = []
            
            for model in models:
                row = {
                    'Model': model.get('name', 'Unknown'),
                    'Version': model.get('version', '1.0.0'),
                    'Status': model.get('status', 'unknown')
                }
                
                performance = model.get('performance_metrics', {})
                for metric in metrics:
                    value = performance.get(metric, 0)
                    if 'accuracy' in metric or 'confidence' in metric:
                        row[metric.title()] = f"{value:.1%}"
                    elif 'time' in metric:
                        row[metric.title()] = f"{value:.2f}s"
                    else:
                        row[metric.title()] = f"{value:.3f}"
                
                table_data.append(row)
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to render comparison table: {e}")
    
    def _render_comparison_charts(self, models: List[Dict[str, Any]], 
                                metrics: List[str]) -> None:
        """Render comparison charts."""
        try:
            for metric in metrics:
                values = []
                names = []
                
                for model in models:
                    performance = model.get('performance_metrics', {})
                    if metric in performance:
                        values.append(performance[metric])
                        names.append(model.get('name', 'Unknown'))
                
                if values:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=names,
                            y=values,
                            name=metric.title(),
                            marker_color='rgb(0, 102, 204)'
                        )
                    ])
                    
                    # Format y-axis based on metric type
                    if 'accuracy' in metric or 'confidence' in metric:
                        fig.update_layout(yaxis=dict(tickformat='.0%'))
                    
                    fig.update_layout(
                        title=f"{metric.title()} Comparison",
                        xaxis_title="Models",
                        yaxis_title=metric.title(),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            logger.error(f"❌ Failed to render comparison charts: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class ModelMetrics:
    """Component for displaying detailed model metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize model metrics."""
        self.config = config or {}
    
    def render(self, model_data: Dict[str, Any], 
               title: str = None,
               show_history: bool = True) -> None:
        """
        Render model metrics.
        
        Args:
            model_data: Model data dictionary
            title: Metrics title
            show_history: Whether to show performance history
        """
        try:
            model_name = model_data.get('name', 'Unknown Model')
            if not title:
                title = f"{model_name} Metrics"
            
            st.subheader(title)
            
            performance = model_data.get('performance_metrics', {})
            
            if not performance:
                st.warning("⚠️ No performance metrics available")
                return
            
            # Display current metrics
            self._render_current_metrics(performance)
            
            # Display performance history if available
            if show_history:
                history = model_data.get('performance_history', [])
                if history:
                    self._render_performance_history(history)
                else:
                    st.info("📊 No performance history available")
            
        except Exception as e:
            logger.error(f"❌ Failed to render model metrics: {e}")
            st.error(f"Failed to display model metrics: {e}")
    
    def _render_current_metrics(self, metrics: Dict[str, float]) -> None:
        """Render current performance metrics."""
        st.markdown("**Current Performance:**")
        
        # Create columns for metrics
        cols = st.columns(min(len(metrics), 4))
        
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % len(cols)]:
                if 'accuracy' in metric.lower() or 'confidence' in metric.lower():
                    st.metric(metric.replace('_', ' ').title(), f"{value:.1%}")
                elif 'time' in metric.lower():
                    st.metric(metric.replace('_', ' ').title(), f"{value:.2f}s")
                elif 'count' in metric.lower():
                    st.metric(metric.replace('_', ' ').title(), f"{int(value)}")
                else:
                    st.metric(metric.replace('_', ' ').title(), f"{value:.3f}")
    
    def _render_performance_history(self, history: List[Dict[str, Any]]) -> None:
        """Render performance history chart."""
        try:
            st.markdown("**Performance History:**")
            
            if not history:
                st.info("No historical data available")
                return
            
            # Extract time series data
            timestamps = []
            metrics_data = {}
            
            for entry in history:
                if 'timestamp' in entry and 'metrics' in entry:
                    timestamps.append(entry['timestamp'])
                    
                    for metric, value in entry['metrics'].items():
                        if metric not in metrics_data:
                            metrics_data[metric] = []
                        metrics_data[metric].append(value)
            
            if not timestamps or not metrics_data:
                st.info("Insufficient historical data for visualization")
                return
            
            # Create line chart for each metric
            for metric, values in metrics_data.items():
                if len(values) == len(timestamps):
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines+markers',
                        name=metric.title(),
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Format y-axis based on metric type
                    if 'accuracy' in metric.lower() or 'confidence' in metric.lower():
                        fig.update_layout(yaxis=dict(tickformat='.0%'))
                    
                    fig.update_layout(
                        title=f"{metric.replace('_', ' ').title()} Over Time",
                        xaxis_title="Time",
                        yaxis_title=metric.title(),
                        showlegend=False,
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            logger.error(f"❌ Failed to render performance history: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


# Utility functions for model cards
def create_sample_model_data() -> List[Dict[str, Any]]:
    """Create sample model data for testing."""
    return [
        {
            'id': 'math_engine_v1',
            'name': 'Mathematical Engine',
            'version': '1.0.0',
            'status': 'active',
            'description': 'Statistical analysis engine with frequency and trend analysis',
            'performance_metrics': {
                'accuracy': 0.78,
                'confidence': 0.82,
                'training_time': 45.2,
                'prediction_count': 1250
            }
        },
        {
            'id': 'ensemble_v2',
            'name': 'Expert Ensemble',
            'version': '2.1.0',
            'status': 'training',
            'description': 'Machine learning ensemble with multiple algorithms',
            'performance_metrics': {
                'accuracy': 0.85,
                'confidence': 0.79,
                'training_time': 128.7,
                'prediction_count': 980
            }
        },
        {
            'id': 'temporal_v1',
            'name': 'Temporal Engine',
            'version': '1.2.0',
            'status': 'inactive',
            'description': 'Time-series analysis with LSTM and ARIMA models',
            'performance_metrics': {
                'accuracy': 0.72,
                'confidence': 0.76,
                'training_time': 203.1,
                'prediction_count': 567
            }
        }
    ]


def get_model_status_color(status: str) -> str:
    """Get color for model status."""
    status_colors = {
        'active': '#28a745',
        'training': '#ffc107',
        'inactive': '#6c757d',
        'error': '#dc3545',
        'ready': '#17a2b8'
    }
    
    return status_colors.get(status.lower(), '#6c757d')