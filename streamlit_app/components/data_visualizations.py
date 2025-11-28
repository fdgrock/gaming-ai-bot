"""
Data visualization components for the lottery prediction system.

This module provides comprehensive chart and visualization components
for displaying lottery data, trends, analytics, and AI model insights
with consistent styling and advanced interactivity.

Enhanced Components:
- DataVisualizationComponents: Comprehensive chart library for all visualization needs
- SophisticatedVisualizationIntelligence: Legacy advanced visualization system
- ChartBuilder: Backward compatibility interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import logging

# Plotly imports with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    ff = None
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataVisualizationComponents:
    """
    Comprehensive data visualization component library for lottery prediction system.
    
    This class provides a complete set of reusable chart and visualization components
    extracted from the legacy application UI patterns. All components maintain
    consistent styling, theming, and interactive capabilities.
    
    Key Features:
    - Performance Charts: Model accuracy, confidence, and training metrics
    - Analytics Dashboards: Comprehensive data analysis visualizations
    - Comparison Charts: Model and prediction comparison visualizations
    - Distribution Plots: Number frequency, pattern analysis charts
    - Timeline Charts: Historical data and trend visualization
    - Custom Layouts: Multi-panel dashboards and subplots
    - Interactive Elements: Filters, zoom, hover, and selection
    - Consistent Theming: Unified color schemes and styling
    
    Chart Categories:
    1. Performance & Metrics Charts
    2. Data Analytics & Distribution Charts  
    3. Comparison & Evaluation Charts
    4. Timeline & Trend Charts
    5. Interactive Dashboards
    6. Custom Visualization Layouts
    """
    
    # Color palettes for consistent theming
    COLOR_PALETTES = {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'performance': ['#003049', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7'],
        'accuracy': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
        'confidence': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
        'success': ['#28a745', '#20c997', '#17a2b8', '#6610f2', '#e83e8c'],
        'warning': ['#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#6c757d']
    }
    
    @staticmethod
    def render_performance_dashboard(performance_data: Dict,
                                   title: str = "Model Performance Dashboard") -> None:
        """
        Comprehensive performance dashboard with multiple metrics panels.
        
        Args:
            performance_data: Dictionary containing performance metrics
            title: Dashboard title
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("🔧 Performance Dashboard requires plotly. Install with: pip install plotly")
                return
            
            st.markdown(f"## 📊 {title}")
            
            # Extract metrics
            metrics = performance_data.get('metrics', {})
            model_data = performance_data.get('models', [])
            
            # Top-level metrics
            DataVisualizationComponents._render_summary_metrics(metrics)
            
            # Performance comparison chart
            if model_data:
                st.markdown("### 📈 Model Performance Comparison")
                DataVisualizationComponents.render_model_comparison_chart(model_data)
            
            # Confidence distribution
            if 'confidence_data' in performance_data:
                st.markdown("### 🎯 Confidence Distribution")
                DataVisualizationComponents.render_confidence_distribution_chart(
                    performance_data['confidence_data']
                )
            
            # Training history
            if 'training_history' in performance_data:
                st.markdown("### 📚 Training Progress")
                DataVisualizationComponents.render_training_history_chart(
                    performance_data['training_history']
                )
        
        except Exception as e:
            logger.error(f"Error rendering performance dashboard: {e}")
            st.error("Error displaying performance dashboard")
    
    @staticmethod
    def render_confidence_scoring_chart(confidence_data: Dict,
                                      chart_type: str = 'bar') -> None:
        """
        Render confidence scoring visualization with multiple chart types.
        
        Args:
            confidence_data: Dictionary with confidence scores and metadata
            chart_type: Type of chart ('bar', 'gauge', 'radar')
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("📊 Confidence Scoring requires plotly. Install with: pip install plotly")
                return
            
            phases = confidence_data.get('phases', [])
            confidences = confidence_data.get('confidences', [])
            
            if not phases or not confidences:
                st.info("No confidence data available")
                return
            
            if chart_type == 'bar':
                DataVisualizationComponents._render_confidence_bar_chart(phases, confidences)
            elif chart_type == 'gauge':
                DataVisualizationComponents._render_confidence_gauge_chart(phases, confidences)
            elif chart_type == 'radar':
                DataVisualizationComponents._render_confidence_radar_chart(phases, confidences)
            
            # Summary metrics
            avg_confidence = sum(confidences) / len(confidences)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            with col2:
                max_conf = max(confidences)
                st.metric("Highest Phase", f"{max_conf:.1%}")
            with col3:
                active_phases = sum(1 for c in confidences if c > 0.1)
                st.metric("Active Phases", f"{active_phases}/{len(phases)}")
        
        except Exception as e:
            logger.error(f"Error rendering confidence scoring chart: {e}")
            st.error("Error displaying confidence chart")
    
    @staticmethod
    def render_model_comparison_chart(models_data: List[Dict],
                                    metrics: List[str] = None) -> None:
        """
        Render comprehensive model comparison chart.
        
        Args:
            models_data: List of model dictionaries with performance data
            metrics: List of metrics to compare
        """
        try:
            if not PLOTLY_AVAILABLE or not models_data:
                st.info("Model comparison requires plotly and model data")
                return
            
            if metrics is None:
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            # Extract data for comparison
            model_names = [model.get('name', f"Model {i+1}") for i, model in enumerate(models_data)]
            
            fig = go.Figure()
            
            colors = DataVisualizationComponents.COLOR_PALETTES['performance']
            
            for i, metric in enumerate(metrics):
                values = []
                for model in models_data:
                    model_metrics = model.get('performance_metrics', {})
                    values.append(model_metrics.get(metric, 0))
                
                fig.add_trace(go.Bar(
                    name=metric.title(),
                    x=model_names,
                    y=values,
                    yaxis='y',
                    offsetgroup=i,
                    marker_color=colors[i % len(colors)],
                    text=[f"{v:.1%}" if v <= 1 else f"{v:.3f}" for v in values],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Performance Score",
                barmode='group',
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error rendering model comparison chart: {e}")
            st.error("Error displaying model comparison")
    
    @staticmethod
    def render_analytics_dashboard(analytics_data: Dict,
                                 layout: str = 'multi_panel') -> None:
        """
        Comprehensive analytics dashboard with multiple visualization panels.
        
        Args:
            analytics_data: Dictionary containing various analytics data
            layout: Layout style ('multi_panel', 'tabbed', 'stacked')
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("🔧 Analytics Dashboard requires plotly. Install with: pip install plotly")
                return
            
            st.markdown("## 📊 Analytics Dashboard")
            
            if layout == 'tabbed':
                DataVisualizationComponents._render_tabbed_analytics(analytics_data)
            elif layout == 'stacked':
                DataVisualizationComponents._render_stacked_analytics(analytics_data)
            else:
                DataVisualizationComponents._render_multi_panel_analytics(analytics_data)
        
        except Exception as e:
            logger.error(f"Error rendering analytics dashboard: {e}")
            st.error("Error displaying analytics dashboard")
    
    @staticmethod
    def render_distribution_analysis_chart(data: Union[List, np.ndarray, pd.Series],
                                         chart_type: str = 'histogram',
                                         title: str = "Distribution Analysis") -> None:
        """
        Render distribution analysis with multiple visualization options.
        
        Args:
            data: Data for distribution analysis
            chart_type: Type of chart ('histogram', 'kde', 'box', 'violin')
            title: Chart title
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info(f"📊 {title} requires plotly. Install with: pip install plotly")
                return
            
            if isinstance(data, pd.Series):
                data_values = data.values
            elif isinstance(data, list):
                data_values = np.array(data)
            else:
                data_values = data
            
            if chart_type == 'histogram':
                fig = px.histogram(x=data_values, nbins=30, title=title)
            elif chart_type == 'box':
                fig = px.box(y=data_values, title=title)
            elif chart_type == 'violin':
                fig = px.violin(y=data_values, title=title)
            else:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=data_values, nbinsx=30))
                fig.update_layout(title=title)
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            DataVisualizationComponents._render_distribution_stats(data_values)
        
        except Exception as e:
            logger.error(f"Error rendering distribution analysis: {e}")
            st.error("Error displaying distribution analysis")
    
    @staticmethod
    def render_timeline_chart(timeline_data: Dict,
                            chart_type: str = 'line') -> None:
        """
        Render timeline/trend visualization with multiple chart types.
        
        Args:
            timeline_data: Dictionary with timestamps and values
            chart_type: Type of chart ('line', 'area', 'scatter')
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("📈 Timeline Chart requires plotly. Install with: pip install plotly")
                return
            
            timestamps = timeline_data.get('timestamps', [])
            values = timeline_data.get('values', [])
            labels = timeline_data.get('labels', ['Value'])
            
            if not timestamps or not values:
                st.info("No timeline data available")
                return
            
            fig = go.Figure()
            
            if chart_type == 'area':
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    fill='tonexty',
                    name=labels[0] if labels else 'Value'
                ))
            elif chart_type == 'scatter':
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='markers',
                    name=labels[0] if labels else 'Value'
                ))
            else:  # line chart
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=labels[0] if labels else 'Value'
                ))
            
            fig.update_layout(
                title="Timeline Analysis",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error rendering timeline chart: {e}")
            st.error("Error displaying timeline chart")
    
    @staticmethod
    def render_heatmap_visualization(data: Union[pd.DataFrame, np.ndarray],
                                   title: str = "Heatmap Analysis") -> None:
        """
        Render heatmap visualization for correlation and pattern analysis.
        
        Args:
            data: 2D data for heatmap (DataFrame or array)
            title: Heatmap title
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info(f"📊 {title} requires plotly. Install with: pip install plotly")
                return
            
            if isinstance(data, pd.DataFrame):
                fig = px.imshow(data, title=title, color_continuous_scale='RdYlBu')
                fig.update_layout(height=500)
            else:
                fig = go.Figure(data=go.Heatmap(z=data, colorscale='RdYlBu'))
                fig.update_layout(title=title, height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error rendering heatmap: {e}")
            st.error("Error displaying heatmap")
    
    @staticmethod
    def render_interactive_scatter_plot(scatter_data: Dict,
                                      enable_selection: bool = True) -> Optional[List]:
        """
        Render interactive scatter plot with selection capabilities.
        
        Args:
            scatter_data: Dictionary with x, y data and optional metadata
            enable_selection: Whether to enable point selection
            
        Returns:
            List of selected point indices if selection enabled
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("📊 Interactive Scatter Plot requires plotly. Install with: pip install plotly")
                return None
            
            x_values = scatter_data.get('x', [])
            y_values = scatter_data.get('y', [])
            labels = scatter_data.get('labels', [])
            colors = scatter_data.get('colors', None)
            
            if not x_values or not y_values:
                st.info("No scatter data available")
                return None
            
            fig = go.Figure()
            
            scatter_kwargs = {
                'x': x_values,
                'y': y_values,
                'mode': 'markers',
                'marker': {'size': 8},
                'text': labels if labels else None,
                'hovertemplate': '<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>' if labels else None
            }
            
            if colors:
                scatter_kwargs['marker']['color'] = colors
                scatter_kwargs['marker']['colorscale'] = 'viridis'
            
            fig.add_trace(go.Scatter(**scatter_kwargs))
            
            fig.update_layout(
                title="Interactive Scatter Analysis",
                height=500,
                dragmode='select' if enable_selection else 'zoom'
            )
            
            # Render with event handling for selection
            selected_points = st.plotly_chart(fig, use_container_width=True, key="interactive_scatter")
            
            return selected_points
        
        except Exception as e:
            logger.error(f"Error rendering interactive scatter plot: {e}")
            st.error("Error displaying scatter plot")
            return None
    
    # Utility methods for internal chart rendering
    @staticmethod
    def _render_summary_metrics(metrics: Dict) -> None:
        """Render summary metrics in columns."""
        if not metrics:
            return
        
        cols = st.columns(min(4, len(metrics)))
        
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i % 4]:
                if isinstance(value, float) and 0 <= value <= 1:
                    st.metric(metric.title(), f"{value:.1%}")
                else:
                    st.metric(metric.title(), str(value))
    
    @staticmethod
    def _render_confidence_bar_chart(phases: List[str], confidences: List[float]) -> None:
        """Render confidence scores as bar chart."""
        colors = DataVisualizationComponents.COLOR_PALETTES['confidence'][:len(phases)]
        
        fig = go.Figure(data=[
            go.Bar(
                x=phases,
                y=confidences,
                marker_color=colors,
                text=[f"{c:.1%}" for c in confidences],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Phase Confidence Levels",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_confidence_gauge_chart(phases: List[str], confidences: List[float]) -> None:
        """Render confidence scores as gauge charts."""
        cols = st.columns(min(3, len(phases)))
        
        for i, (phase, confidence) in enumerate(zip(phases, confidences)):
            with cols[i % 3]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': phase},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_confidence_radar_chart(phases: List[str], confidences: List[float]) -> None:
        """Render confidence scores as radar chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=confidences,
            theta=phases,
            fill='toself',
            name='Confidence Levels'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Confidence Radar Analysis",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_tabbed_analytics(analytics_data: Dict) -> None:
        """Render analytics in tabbed layout."""
        tab_names = list(analytics_data.keys())
        tabs = st.tabs(tab_names)
        
        for tab, (key, data) in zip(tabs, analytics_data.items()):
            with tab:
                if key == 'performance':
                    DataVisualizationComponents.render_performance_dashboard(data, title=key.title())
                elif key == 'distribution':
                    DataVisualizationComponents.render_distribution_analysis_chart(
                        data.get('values', []), title=f"{key.title()} Analysis"
                    )
                elif key == 'timeline':
                    DataVisualizationComponents.render_timeline_chart(data)
                else:
                    st.info(f"Analytics data for {key}")
                    st.json(data)
    
    @staticmethod
    def _render_stacked_analytics(analytics_data: Dict) -> None:
        """Render analytics in stacked layout."""
        for key, data in analytics_data.items():
            st.markdown(f"### 📊 {key.title()} Analysis")
            
            if key == 'performance' and isinstance(data, dict):
                DataVisualizationComponents.render_performance_dashboard(data, title="")
            elif key == 'distribution' and isinstance(data, dict):
                values = data.get('values', [])
                if values:
                    DataVisualizationComponents.render_distribution_analysis_chart(values, title="")
            elif key == 'timeline' and isinstance(data, dict):
                DataVisualizationComponents.render_timeline_chart(data)
            else:
                st.info(f"Data for {key}")
                if isinstance(data, dict) and data:
                    # Display key metrics if available
                    for subkey, subvalue in data.items():
                        if isinstance(subvalue, (int, float, str)):
                            st.metric(subkey.title(), str(subvalue))
    
    @staticmethod
    def _render_multi_panel_analytics(analytics_data: Dict) -> None:
        """Render analytics in multi-panel layout."""
        # Create dynamic column layout based on data
        num_panels = len(analytics_data)
        if num_panels <= 2:
            cols = st.columns(num_panels)
        else:
            # Use 2-column layout for more than 2 panels
            cols = st.columns(2)
        
        for i, (key, data) in enumerate(analytics_data.items()):
            with cols[i % len(cols)]:
                st.markdown(f"#### 📊 {key.title()}")
                
                if key == 'performance' and isinstance(data, dict):
                    # Render simplified performance metrics
                    metrics = data.get('metrics', {})
                    DataVisualizationComponents._render_summary_metrics(metrics)
                elif key == 'distribution' and isinstance(data, dict):
                    values = data.get('values', [])
                    if values:
                        # Simple histogram
                        if len(values) > 0:
                            fig = go.Figure(go.Histogram(x=values))
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                elif isinstance(data, dict) and 'values' in data:
                    values = data['values']
                    if isinstance(values, list) and len(values) > 0:
                        st.line_chart(values)
                else:
                    st.info(f"Data available for {key}")
    
    @staticmethod
    def _render_distribution_stats(data_values: np.ndarray) -> None:
        """Render statistical summary for distribution data."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{np.mean(data_values):.3f}")
        with col2:
            st.metric("Std Dev", f"{np.std(data_values):.3f}")
        with col3:
            st.metric("Min", f"{np.min(data_values):.3f}")
        with col4:
            st.metric("Max", f"{np.max(data_values):.3f}")
    
    @staticmethod
    def render_training_history_chart(training_data: Dict) -> None:
        """Render training history with loss and accuracy curves."""
        try:
            if not PLOTLY_AVAILABLE:
                st.info("Training history chart requires plotly")
                return
            
            epochs = training_data.get('epochs', [])
            loss = training_data.get('loss', [])
            accuracy = training_data.get('accuracy', [])
            
            if not epochs:
                st.info("No training history available")
                return
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add loss curve
            if loss:
                fig.add_trace(
                    go.Scatter(x=epochs, y=loss, name="Loss", line=dict(color='red')),
                    secondary_y=False,
                )
            
            # Add accuracy curve
            if accuracy:
                fig.add_trace(
                    go.Scatter(x=epochs, y=accuracy, name="Accuracy", line=dict(color='blue')),
                    secondary_y=True,
                )
            
            # Update layout
            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Loss", secondary_y=False)
            fig.update_yaxes(title_text="Accuracy", secondary_y=True)
            
            fig.update_layout(
                title="Training History",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error rendering training history chart: {e}")
            st.error("Error displaying training history")
    
    @staticmethod
    def render_confidence_distribution_chart(confidence_data: Dict) -> None:
        """Render confidence score distribution analysis."""
        try:
            if not PLOTLY_AVAILABLE:
                st.info("Confidence distribution chart requires plotly")
                return
            
            scores = confidence_data.get('scores', [])
            if not scores:
                st.info("No confidence data available")
                return
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=scores, nbinsx=20, name="Confidence Distribution"))
            
            fig.update_layout(
                title="Confidence Score Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average", f"{np.mean(scores):.3f}")
            with col2:
                st.metric("Median", f"{np.median(scores):.3f}")
            with col3:
                st.metric("Std Dev", f"{np.std(scores):.3f}")
        
        except Exception as e:
            logger.error(f"Error rendering confidence distribution: {e}")
            st.error("Error displaying confidence distribution")


class SophisticatedVisualizationIntelligence:
    """Advanced visualization intelligence for sophisticated data analysis and presentation"""
    
    def __init__(self):
        self.visualization_cache = {}
        self.intelligence_metrics = {}
        self.color_palettes = {
            'ultra_accuracy': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'sophistication': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
            'intelligence': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51'],
            'performance': ['#003049', '#D62828', '#F77F00', '#FCBF49', '#EAE2B7']
        }
    
    def generate_ultra_accuracy_dashboard(self, orchestration_result: Dict[str, Any], 
                                        historical_data: pd.DataFrame) -> None:
        """Generate ultra-accuracy dashboard with comprehensive visualizations"""
        try:
            st.markdown("## 🚀 Ultra-Accuracy Analysis Dashboard")
            
            # Header metrics
            self._render_ultra_accuracy_metrics(orchestration_result)
            
            # Engine synergy analysis
            synergy_data = orchestration_result.get('engine_synergy', {})
            if synergy_data:
                st.markdown("### 🔗 Engine Synergy Analysis")
                self._render_synergy_network(synergy_data)
            
            # Dynamic weights visualization
            weights_data = orchestration_result.get('dynamic_weights', {})
            if weights_data:
                st.markdown("### ⚖️ Dynamic Engine Weights")
                self._render_dynamic_weights(weights_data)
            
            # Prediction confidence distribution
            predictions = orchestration_result.get('ultra_accuracy_predictions', [])
            if predictions:
                st.markdown("### 📊 Prediction Confidence Distribution")
                self._render_confidence_distribution(predictions)
            
            # Temporal intelligence analysis
            if historical_data is not None and not historical_data.empty:
                st.markdown("### 🕐 Temporal Intelligence Patterns")
                self._render_temporal_intelligence(historical_data, predictions)
            
        except Exception as e:
            logger.error(f"Error generating ultra-accuracy dashboard: {e}")
            st.error(f"Failed to generate dashboard: {e}")
    
    def _render_ultra_accuracy_metrics(self, orchestration_result: Dict[str, Any]) -> None:
        """Render ultra-accuracy metrics"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            # Orchestration Intelligence
            intelligence = orchestration_result.get('orchestration_intelligence', 0)
            with col1:
                st.metric(
                    "🧠 Intelligence Score",
                    f"{intelligence:.3f}",
                    delta=f"{intelligence - 0.5:.3f}" if intelligence > 0.5 else None
                )
            
            # Models Coordinated
            models_count = orchestration_result.get('models_coordinated', 0)
            with col2:
                st.metric("🤖 Models", str(models_count))
            
            # Prediction Sets
            predictions = orchestration_result.get('ultra_accuracy_predictions', [])
            with col3:
                st.metric("🎯 Predictions", str(len(predictions)))
            
            # Average Confidence
            if predictions:
                avg_confidence = np.mean([p.get('confidence', 0) for p in predictions])
                with col4:
                    st.metric(
                        "🎪 Avg Confidence",
                        f"{avg_confidence:.3f}",
                        delta=f"{avg_confidence - 0.6:.3f}" if avg_confidence > 0.6 else None
                    )
            
        except Exception as e:
            logger.error(f"Error rendering ultra-accuracy metrics: {e}")
    
    def _render_synergy_network(self, synergy_data: Dict[str, float]) -> None:
        """Render engine synergy network visualization"""
        try:
            if not synergy_data:
                st.info("No synergy data available")
                return
            
            # Create network-style visualization
            fig = go.Figure()
            
            # Extract engine pairs and synergy scores
            engines = set()
            for pair in synergy_data.keys():
                if '_' in pair:
                    engine1, engine2 = pair.split('_', 1)
                    engines.add(engine1)
                    engines.add(engine2)
            
            engines = list(engines)
            
            # Position engines in a circle
            n_engines = len(engines)
            if n_engines == 0:
                st.info("No engine data for synergy visualization")
                return
            
            angles = np.linspace(0, 2 * np.pi, n_engines, endpoint=False)
            positions = {engine: (np.cos(angle), np.sin(angle)) for engine, angle in zip(engines, angles)}
            
            # Draw connections (edges)
            for pair, synergy_score in synergy_data.items():
                if '_' in pair and synergy_score > 0.1:  # Only show significant synergy
                    engine1, engine2 = pair.split('_', 1)
                    if engine1 in positions and engine2 in positions:
                        x1, y1 = positions[engine1]
                        x2, y2 = positions[engine2]
                        
                        # Line thickness based on synergy strength
                        line_width = max(1, synergy_score * 10)
                        line_color = 'rgba(50, 168, 82, 0.8)' if synergy_score > 0.5 else 'rgba(255, 127, 14, 0.6)'
                        
                        fig.add_trace(go.Scatter(
                            x=[x1, x2], y=[y1, y2],
                            mode='lines',
                            line=dict(width=line_width, color=line_color),
                            showlegend=False,
                            hovertemplate=f"{engine1} ↔ {engine2}<br>Synergy: {synergy_score:.3f}<extra></extra>"
                        ))
            
            # Draw nodes (engines)
            x_coords = [positions[engine][0] for engine in engines]
            y_coords = [positions[engine][1] for engine in engines]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers+text',
                marker=dict(size=30, color=self.color_palettes['sophistication'][:len(engines)]),
                text=engines,
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                showlegend=False,
                hovertemplate="%{text}<extra></extra>"
            ))
            
            fig.update_layout(
                title="Engine Synergy Network",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering synergy network: {e}")
            st.error("Failed to render synergy network")
    
    def _render_dynamic_weights(self, weights_data: Dict[str, float]) -> None:
        """Render dynamic weights visualization"""
        try:
            if not weights_data:
                st.info("No weight data available")
                return
            
            engines = list(weights_data.keys())
            weights = list(weights_data.values())
            
            # Create pie chart for weights
            fig = go.Figure(data=[go.Pie(
                labels=engines,
                values=weights,
                hole=0.3,
                textinfo='label+percent',
                textfont=dict(size=12),
                marker=dict(colors=self.color_palettes['performance'][:len(engines)])
            )])
            
            fig.update_layout(
                title="Dynamic Engine Weight Distribution",
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display weight table
            weight_df = pd.DataFrame({
                'Engine': engines,
                'Weight': [f"{w:.3f}" for w in weights],
                'Influence %': [f"{w*100:.1f}%" for w in weights]
            })
            
            st.dataframe(weight_df, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering dynamic weights: {e}")
            st.error("Failed to render dynamic weights")
    
    def _render_confidence_distribution(self, predictions: List[Dict[str, Any]]) -> None:
        """Render confidence distribution visualization"""
        try:
            confidences = [p.get('confidence', 0) for p in predictions if 'confidence' in p]
            
            if not confidences:
                st.info("No confidence data available")
                return
            
            # Create histogram
            fig = go.Figure(data=[go.Histogram(
                x=confidences,
                nbinsx=20,
                marker_color=self.color_palettes['intelligence'][0],
                opacity=0.7
            )])
            
            fig.update_layout(
                title="Prediction Confidence Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Frequency",
                height=400
            )
            
            # Add mean line
            mean_confidence = np.mean(confidences)
            fig.add_vline(x=mean_confidence, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_confidence:.3f}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Confidence", f"{mean_confidence:.3f}")
            with col2:
                st.metric("Max Confidence", f"{max(confidences):.3f}")
            with col3:
                st.metric("Min Confidence", f"{min(confidences):.3f}")
            
        except Exception as e:
            logger.error(f"Error rendering confidence distribution: {e}")
            st.error("Failed to render confidence distribution")
    
    def _render_temporal_intelligence(self, historical_data: pd.DataFrame, 
                                    predictions: List[Dict[str, Any]]) -> None:
        """Render temporal intelligence analysis"""
        try:
            if historical_data.empty:
                st.info("No historical data for temporal analysis")
                return
            
            # Analyze temporal patterns in historical data
            dates = pd.to_datetime(historical_data['date']) if 'date' in historical_data.columns else None
            
            if dates is None:
                st.info("No date information for temporal analysis")
                return
            
            # Create timeline visualization
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=("Historical Draw Frequency", "Prediction Confidence Timeline"),
                              vertical_spacing=0.1)
            
            # Historical timeline
            daily_counts = dates.dt.date.value_counts().sort_index()
            fig.add_trace(go.Scatter(
                x=daily_counts.index,
                y=daily_counts.values,
                mode='lines',
                name='Draw Frequency',
                line=dict(color=self.color_palettes['ultra_accuracy'][0])
            ), row=1, col=1)
            
            # Prediction confidence timeline (if predictions have timestamps)
            if predictions and any('timestamp' in p for p in predictions):
                pred_times = []
                pred_confidences = []
                
                for p in predictions:
                    if 'timestamp' in p:
                        try:
                            timestamp = pd.to_datetime(p['timestamp'])
                            pred_times.append(timestamp)
                            pred_confidences.append(p.get('confidence', 0))
                        except:
                            continue
                
                if pred_times:
                    fig.add_trace(go.Scatter(
                        x=pred_times,
                        y=pred_confidences,
                        mode='markers+lines',
                        name='Prediction Confidence',
                        line=dict(color=self.color_palettes['ultra_accuracy'][1])
                    ), row=2, col=1)
            
            fig.update_layout(
                title="Temporal Intelligence Analysis",
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering temporal intelligence: {e}")
            st.error("Failed to render temporal intelligence")


class AdvancedPerformanceVisualizer:
    """Advanced performance visualization for AI engine analysis"""
    
    def __init__(self):
        self.performance_cache = {}
        self.benchmark_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    
    def render_engine_performance_dashboard(self, engine_performances: Dict[str, Dict[str, Any]]) -> None:
        """Render comprehensive engine performance dashboard"""
        try:
            st.markdown("## 📊 Engine Performance Dashboard")
            
            if not engine_performances:
                st.info("No performance data available")
                return
            
            # Performance overview
            self._render_performance_overview(engine_performances)
            
            # Individual engine analysis
            st.markdown("### 🔍 Individual Engine Analysis")
            self._render_individual_engine_analysis(engine_performances)
            
            # Performance comparison
            st.markdown("### ⚖️ Performance Comparison")
            self._render_performance_comparison(engine_performances)
            
            # Performance trends
            st.markdown("### 📈 Performance Trends")
            self._render_performance_trends(engine_performances)
            
        except Exception as e:
            logger.error(f"Error rendering performance dashboard: {e}")
            st.error("Failed to render performance dashboard")
    
    def _render_performance_overview(self, engine_performances: Dict[str, Dict[str, Any]]) -> None:
        """Render performance overview metrics"""
        try:
            # Calculate overall metrics
            all_scores = []
            engine_count = len(engine_performances)
            
            for engine_data in engine_performances.values():
                scores = engine_data.get('scores', [])
                if scores:
                    all_scores.extend(scores)
            
            if not all_scores:
                st.info("No performance scores available")
                return
            
            avg_performance = np.mean(all_scores)
            max_performance = max(all_scores)
            min_performance = min(all_scores)
            performance_std = np.std(all_scores)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 Average Performance",
                    f"{avg_performance:.3f}",
                    delta=f"{avg_performance - 0.5:.3f}"
                )
            
            with col2:
                st.metric("🏆 Best Performance", f"{max_performance:.3f}")
            
            with col3:
                st.metric("📉 Lowest Performance", f"{min_performance:.3f}")
            
            with col4:
                consistency = 1 - (performance_std / avg_performance) if avg_performance > 0 else 0
                st.metric("🎪 Consistency", f"{consistency:.3f}")
            
        except Exception as e:
            logger.error(f"Error rendering performance overview: {e}")
    
    def _render_individual_engine_analysis(self, engine_performances: Dict[str, Dict[str, Any]]) -> None:
        """Render individual engine analysis"""
        try:
            for engine_name, engine_data in engine_performances.items():
                with st.expander(f"🤖 {engine_name} Analysis"):
                    scores = engine_data.get('scores', [])
                    
                    if not scores:
                        st.info("No scores available for this engine")
                        continue
                    
                    # Engine metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Average Score", f"{np.mean(scores):.3f}")
                    
                    with col2:
                        st.metric("Best Score", f"{max(scores):.3f}")
                    
                    with col3:
                        st.metric("Consistency", f"{1 - np.std(scores)/np.mean(scores) if np.mean(scores) > 0 else 0:.3f}")
                    
                    # Performance chart
                    fig = go.Figure(data=[go.Scatter(
                        x=list(range(len(scores))),
                        y=scores,
                        mode='lines+markers',
                        name=engine_name,
                        line=dict(width=2)
                    )])
                    
                    fig.update_layout(
                        title=f"{engine_name} Performance Over Time",
                        xaxis_title="Iteration",
                        yaxis_title="Performance Score",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering individual engine analysis: {e}")
    
    def _render_performance_comparison(self, engine_performances: Dict[str, Dict[str, Any]]) -> None:
        """Render performance comparison visualization"""
        try:
            engine_names = []
            avg_scores = []
            
            for engine_name, engine_data in engine_performances.items():
                scores = engine_data.get('scores', [])
                if scores:
                    engine_names.append(engine_name)
                    avg_scores.append(np.mean(scores))
            
            if not engine_names:
                st.info("No data for performance comparison")
                return
            
            # Create comparison bar chart
            fig = go.Figure(data=[go.Bar(
                x=engine_names,
                y=avg_scores,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(engine_names)],
                text=[f"{score:.3f}" for score in avg_scores],
                textposition='auto'
            )])
            
            # Add performance threshold lines
            for threshold_name, threshold_value in self.benchmark_thresholds.items():
                fig.add_hline(y=threshold_value, line_dash="dash", 
                             annotation_text=threshold_name.title())
            
            fig.update_layout(
                title="Engine Performance Comparison",
                xaxis_title="Engines",
                yaxis_title="Average Performance Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering performance comparison: {e}")
    
    def _render_performance_trends(self, engine_performances: Dict[str, Dict[str, Any]]) -> None:
        """Render performance trends visualization"""
        try:
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for i, (engine_name, engine_data) in enumerate(engine_performances.items()):
                scores = engine_data.get('scores', [])
                if scores:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(scores))),
                        y=scores,
                        mode='lines+markers',
                        name=engine_name,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            fig.update_layout(
                title="Performance Trends Over Time",
                xaxis_title="Iteration",
                yaxis_title="Performance Score",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering performance trends: {e}")


class NumberFrequencyChart:
    """Chart component for number frequency analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize number frequency chart."""
        self.config = config or {}
        self.color_scheme = self.config.get('color_scheme', 'blues')
    
    def render(self, data: pd.DataFrame, 
               title: str = "Number Frequency Analysis",
               number_range: Tuple[int, int] = (1, 49),
               chart_type: str = 'bar') -> None:
        """
        Render number frequency chart.
        
        Args:
            data: Historical lottery data
            title: Chart title
            number_range: Range of numbers to analyze
            chart_type: Type of chart ('bar', 'heatmap', 'scatter')
        """
        try:
            st.subheader(title)
            
            # Calculate number frequencies
            frequencies = self._calculate_frequencies(data, number_range)
            
            if not frequencies:
                st.warning("⚠️ No frequency data available")
                return
            
            if chart_type == 'bar':
                self._render_bar_chart(frequencies, title)
            elif chart_type == 'heatmap':
                self._render_heatmap(frequencies, number_range, title)
            elif chart_type == 'scatter':
                self._render_scatter_chart(frequencies, title)
            else:
                st.error(f"Unknown chart type: {chart_type}")
            
            # Display statistics
            self._display_frequency_stats(frequencies)
            
        except Exception as e:
            logger.error(f"❌ Failed to render frequency chart: {e}")
            st.error(f"Failed to display frequency chart: {e}")
    
    def _calculate_frequencies(self, data: pd.DataFrame, 
                             number_range: Tuple[int, int]) -> Dict[int, int]:
        """Calculate number frequencies from data."""
        frequencies = {i: 0 for i in range(number_range[0], number_range[1] + 1)}
        
        for _, row in data.iterrows():
            numbers = row.get('numbers', [])
            
            if isinstance(numbers, str):
                number_list = [int(x.strip()) for x in numbers.split(',')]
            elif isinstance(numbers, list):
                number_list = [int(x) for x in numbers]
            else:
                continue
            
            for num in number_list:
                if number_range[0] <= num <= number_range[1]:
                    frequencies[num] += 1
        
        return frequencies
    
    def _render_bar_chart(self, frequencies: Dict[int, int], title: str) -> None:
        """Render bar chart."""
        numbers = list(frequencies.keys())
        counts = list(frequencies.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=numbers,
                y=counts,
                marker_color='rgb(0, 102, 204)',
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Numbers",
            yaxis_title="Frequency",
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_heatmap(self, frequencies: Dict[int, int], 
                       number_range: Tuple[int, int], title: str) -> None:
        """Render heatmap visualization."""
        # Create matrix for heatmap
        numbers_per_row = 10
        max_num = number_range[1]
        rows = (max_num // numbers_per_row) + 1
        
        matrix = []
        labels = []
        
        for row in range(rows):
            matrix_row = []
            label_row = []
            
            for col in range(numbers_per_row):
                num = row * numbers_per_row + col + 1
                if num <= max_num:
                    matrix_row.append(frequencies.get(num, 0))
                    label_row.append(str(num))
                else:
                    matrix_row.append(0)
                    label_row.append("")
            
            matrix.append(matrix_row)
            labels.append(label_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            text=labels,
            texttemplate="%{text}<br>%{z}",
            textfont={"size": 10},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            height=400,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_scatter_chart(self, frequencies: Dict[int, int], title: str) -> None:
        """Render scatter plot."""
        numbers = list(frequencies.keys())
        counts = list(frequencies.values())
        
        fig = go.Figure(data=go.Scatter(
            x=numbers,
            y=counts,
            mode='markers',
            marker=dict(
                size=8,
                color=counts,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Frequency")
            ),
            text=[f"Number: {num}<br>Frequency: {count}" 
                  for num, count in zip(numbers, counts)],
            hovertemplate="%{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Numbers",
            yaxis_title="Frequency",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_frequency_stats(self, frequencies: Dict[int, int]) -> None:
        """Display frequency statistics."""
        counts = list(frequencies.values())
        
        if not counts:
            return
        
        most_frequent = max(frequencies, key=frequencies.get)
        least_frequent = min(frequencies, key=frequencies.get)
        avg_frequency = np.mean(counts)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Most Frequent", f"{most_frequent} ({frequencies[most_frequent]})")
        
        with col2:
            st.metric("Least Frequent", f"{least_frequent} ({frequencies[least_frequent]})")
        
        with col3:
            st.metric("Average Frequency", f"{avg_frequency:.1f}")
        
        with col4:
            st.metric("Total Draws", sum(counts) // 6)  # Assuming 6 numbers per draw
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class TrendAnalysisChart:
    """Chart component for trend analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize trend analysis chart."""
        self.config = config or {}
    
    def render(self, data: pd.DataFrame, 
               title: str = "Trend Analysis",
               analysis_type: str = 'moving_average') -> None:
        """
        Render trend analysis chart.
        
        Args:
            data: Historical lottery data with dates
            title: Chart title
            analysis_type: Type of analysis ('moving_average', 'seasonal', 'anomaly')
        """
        try:
            st.subheader(title)
            
            if 'date' not in data.columns:
                st.error("❌ Date column required for trend analysis")
                return
            
            if analysis_type == 'moving_average':
                self._render_moving_average(data, title)
            elif analysis_type == 'seasonal':
                self._render_seasonal_analysis(data, title)
            elif analysis_type == 'anomaly':
                self._render_anomaly_detection(data, title)
            else:
                st.error(f"Unknown analysis type: {analysis_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to render trend analysis: {e}")
            st.error(f"Failed to display trend analysis: {e}")
    
    def _render_moving_average(self, data: pd.DataFrame, title: str) -> None:
        """Render moving average trend."""
        # Calculate sum of numbers for each draw
        data = data.copy()
        data['number_sum'] = data['numbers'].apply(self._calculate_sum)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Calculate moving averages
        data['ma_7'] = data['number_sum'].rolling(window=7).mean()
        data['ma_30'] = data['number_sum'].rolling(window=30).mean()
        
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['number_sum'],
            mode='markers',
            name='Draw Sum',
            opacity=0.6,
            marker=dict(size=4, color='lightblue')
        ))
        
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['ma_7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['ma_30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"{title} - Moving Averages",
            xaxis_title="Date",
            yaxis_title="Sum of Numbers",
            height=500,
            legend=dict(x=0, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_seasonal_analysis(self, data: pd.DataFrame, title: str) -> None:
        """Render seasonal analysis."""
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        data['number_sum'] = data['numbers'].apply(self._calculate_sum)
        
        # Monthly analysis
        monthly_stats = data.groupby('month')['number_sum'].agg(['mean', 'std']).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Patterns', 'Day of Week Patterns'),
            vertical_spacing=0.1
        )
        
        # Monthly pattern
        fig.add_trace(
            go.Bar(
                x=monthly_stats['month'],
                y=monthly_stats['mean'],
                name='Monthly Average',
                marker_color='rgb(0, 102, 204)'
            ),
            row=1, col=1
        )
        
        # Day of week pattern
        weekly_stats = data.groupby('day_of_week')['number_sum'].mean().reset_index()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig.add_trace(
            go.Bar(
                x=[day_names[i] for i in weekly_stats['day_of_week']],
                y=weekly_stats['number_sum'],
                name='Weekly Average',
                marker_color='rgb(255, 107, 53)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{title} - Seasonal Patterns",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_anomaly_detection(self, data: pd.DataFrame, title: str) -> None:
        """Render anomaly detection chart."""
        data = data.copy()
        data['number_sum'] = data['numbers'].apply(self._calculate_sum)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        
        # Simple anomaly detection using z-score
        mean_sum = data['number_sum'].mean()
        std_sum = data['number_sum'].std()
        data['z_score'] = (data['number_sum'] - mean_sum) / std_sum
        data['is_anomaly'] = np.abs(data['z_score']) > 2
        
        fig = go.Figure()
        
        # Normal points
        normal_data = data[~data['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['number_sum'],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6)
        ))
        
        # Anomaly points
        anomaly_data = data[data['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomaly_data['date'],
            y=anomaly_data['number_sum'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=8, symbol='diamond')
        ))
        
        # Add threshold lines
        fig.add_hline(y=mean_sum + 2*std_sum, line_dash="dash", line_color="red", 
                     annotation_text="Upper Threshold")
        fig.add_hline(y=mean_sum - 2*std_sum, line_dash="dash", line_color="red", 
                     annotation_text="Lower Threshold")
        
        fig.update_layout(
            title=f"{title} - Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Sum of Numbers",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display anomaly statistics
        anomaly_count = len(anomaly_data)
        st.metric("Anomalies Detected", f"{anomaly_count} ({anomaly_count/len(data)*100:.1f}%)")
    
    def _calculate_sum(self, numbers) -> int:
        """Calculate sum of numbers in a draw."""
        if isinstance(numbers, str):
            number_list = [int(x.strip()) for x in numbers.split(',')]
        elif isinstance(numbers, list):
            number_list = [int(x) for x in numbers]
        else:
            return 0
        
        return sum(number_list)
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class PerformanceChart:
    """Chart component for model performance visualization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance chart."""
        self.config = config or {}
    
    def render(self, performance_data: Dict[str, List[Dict[str, Any]]], 
               title: str = "Model Performance",
               metric: str = 'accuracy') -> None:
        """
        Render performance chart.
        
        Args:
            performance_data: Dictionary with model names as keys and performance history as values
            title: Chart title
            metric: Metric to visualize
        """
        try:
            st.subheader(title)
            
            if not performance_data:
                st.warning("⚠️ No performance data available")
                return
            
            fig = go.Figure()
            
            colors = ['rgb(0, 102, 204)', 'rgb(255, 107, 53)', 'rgb(46, 204, 113)', 
                     'rgb(155, 89, 182)', 'rgb(241, 196, 15)']
            
            for i, (model_name, history) in enumerate(performance_data.items()):
                if not history:
                    continue
                
                timestamps = []
                values = []
                
                for entry in history:
                    if 'timestamp' in entry and 'metrics' in entry:
                        if metric in entry['metrics']:
                            timestamps.append(entry['timestamp'])
                            values.append(entry['metrics'][metric])
                
                if timestamps and values:
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines+markers',
                        name=model_name,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6)
                    ))
            
            # Format y-axis based on metric type
            if 'accuracy' in metric.lower() or 'confidence' in metric.lower():
                fig.update_layout(yaxis=dict(tickformat='.0%'))
            
            fig.update_layout(
                title=f"{title} - {metric.title()} Over Time",
                xaxis_title="Time",
                yaxis_title=metric.title(),
                height=500,
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to render performance chart: {e}")
            st.error(f"Failed to display performance chart: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class HeatmapChart:
    """Heatmap chart component for correlation analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize heatmap chart."""
        self.config = config or {}
    
    def render(self, data: pd.DataFrame, 
               title: str = "Correlation Heatmap",
               correlation_method: str = 'pearson') -> None:
        """
        Render correlation heatmap.
        
        Args:
            data: Data for correlation analysis
            title: Chart title
            correlation_method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        try:
            st.subheader(title)
            
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                st.warning("⚠️ No numeric data available for correlation analysis")
                return
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr(method=correlation_method)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=True,
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title=f"{title} ({correlation_method.title()})",
                height=500,
                xaxis=dict(side="bottom"),
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display correlation insights
            self._display_correlation_insights(corr_matrix)
            
        except Exception as e:
            logger.error(f"❌ Failed to render heatmap: {e}")
            st.error(f"Failed to display heatmap: {e}")
    
    def _display_correlation_insights(self, corr_matrix: pd.DataFrame) -> None:
        """Display correlation insights."""
        try:
            # Find strongest correlations
            correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value):
                        correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if correlations:
                # Sort by absolute correlation
                correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                st.markdown("**Top Correlations:**")
                for i, corr in enumerate(correlations[:5]):
                    st.write(f"{i+1}. {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to display correlation insights: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class TimeSeriesChart:
    """Time series chart component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize time series chart."""
        self.config = config or {}
    
    def render(self, data: pd.DataFrame, 
               title: str = "Time Series Analysis",
               value_column: str = 'value',
               date_column: str = 'date') -> None:
        """
        Render time series chart.
        
        Args:
            data: Time series data
            title: Chart title
            value_column: Column containing values
            date_column: Column containing dates
        """
        try:
            st.subheader(title)
            
            if date_column not in data.columns or value_column not in data.columns:
                st.error(f"❌ Required columns missing: {date_column}, {value_column}")
                return
            
            # Prepare data
            chart_data = data.copy()
            chart_data[date_column] = pd.to_datetime(chart_data[date_column])
            chart_data = chart_data.sort_values(date_column)
            
            # Create time series chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=chart_data[date_column],
                y=chart_data[value_column],
                mode='lines+markers',
                name='Values',
                line=dict(color='rgb(0, 102, 204)', width=2),
                marker=dict(size=4)
            ))
            
            # Add trend line
            if len(chart_data) > 1:
                x_numeric = np.arange(len(chart_data))
                z = np.polyfit(x_numeric, chart_data[value_column], 1)
                trend_line = np.poly1d(z)(x_numeric)
                
                fig.add_trace(go.Scatter(
                    x=chart_data[date_column],
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title=value_column.title(),
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            self._display_timeseries_stats(chart_data, value_column)
            
        except Exception as e:
            logger.error(f"❌ Failed to render time series chart: {e}")
            st.error(f"Failed to display time series chart: {e}")
    
    def _display_timeseries_stats(self, data: pd.DataFrame, value_column: str) -> None:
        """Display time series statistics."""
        try:
            values = data[value_column]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{values.mean():.2f}")
            
            with col2:
                st.metric("Std Dev", f"{values.std():.2f}")
            
            with col3:
                st.metric("Min", f"{values.min():.2f}")
            
            with col4:
                st.metric("Max", f"{values.max():.2f}")
                
        except Exception as e:
            logger.error(f"❌ Failed to display time series stats: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


# Utility functions for visualizations
def create_sample_visualization_data() -> pd.DataFrame:
    """Create sample data for testing visualizations."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    data = []
    
    for date in dates:
        numbers = sorted(np.random.choice(range(1, 50), 6, replace=False))
        data.append({
            'date': date,
            'numbers': numbers,
            'sum': sum(numbers),
            'even_count': sum(1 for n in numbers if n % 2 == 0),
            'odd_count': sum(1 for n in numbers if n % 2 == 1)
        })
    
    return pd.DataFrame(data)


def get_color_palette(palette_name: str = 'default') -> List[str]:
    """Get color palette for visualizations."""
    palettes = {
        'default': ['#0066cc', '#ff6b35', '#2ecc71', '#9b59b6', '#f1c40f'],
        'blues': ['#0066cc', '#3498db', '#5dade2', '#85c1e9', '#aed6f1'],
        'warm': ['#e74c3c', '#f39c12', '#f1c40f', '#e67e22', '#d35400'],
        'cool': ['#2ecc71', '#1abc9c', '#16a085', '#27ae60', '#229954']
    }
    
    return palettes.get(palette_name, palettes['default'])


# Backward compatibility classes - maintain original interfaces
class ChartBuilder:
    """
    Legacy ChartBuilder class for backward compatibility.
    
    This class maintains the original simple interface while delegating
    to the enhanced DataVisualizationComponents class for actual functionality.
    """
    
    @staticmethod
    def render_bar_chart(data: Dict, title: str = "Bar Chart") -> None:
        """Render bar chart (legacy method)."""
        if 'x' in data and 'y' in data:
            # Convert to confidence format for enhanced rendering
            confidence_data = {
                'phases': data['x'],
                'confidences': data['y']
            }
            DataVisualizationComponents.render_confidence_scoring_chart(confidence_data, 'bar')
        else:
            st.error("Data must contain 'x' and 'y' keys")
    
    @staticmethod
    def render_line_chart(data: Dict, title: str = "Line Chart") -> None:
        """Render line chart (legacy method)."""
        if 'timestamps' in data and 'values' in data:
            DataVisualizationComponents.render_timeline_chart(data, 'line')
        else:
            # Try to convert generic data format
            timeline_data = {
                'timestamps': data.get('x', list(range(len(data.get('y', []))))),
                'values': data.get('y', [])
            }
            DataVisualizationComponents.render_timeline_chart(timeline_data, 'line')
    
    @staticmethod
    def render_scatter_plot(data: Dict, title: str = "Scatter Plot") -> None:
        """Render scatter plot (legacy method)."""
        scatter_data = {
            'x': data.get('x', []),
            'y': data.get('y', []),
            'labels': data.get('labels', [])
        }
        DataVisualizationComponents.render_interactive_scatter_plot(scatter_data, enable_selection=False)


class VisualizationHelper:
    """
    Legacy VisualizationHelper class for backward compatibility.
    
    This class provides helper methods that maintain the original interface.
    """
    
    @staticmethod
    def create_performance_chart(performance_metrics: Dict) -> None:
        """Create performance chart (legacy method)."""
        # Convert to new format
        performance_data = {
            'metrics': performance_metrics,
            'models': []
        }
        DataVisualizationComponents.render_performance_dashboard(performance_data)
    
    @staticmethod
    def show_analytics_dashboard(analytics_data: Dict) -> None:
        """Show analytics dashboard (legacy method)."""
        DataVisualizationComponents.render_analytics_dashboard(analytics_data, 'multi_panel')
    
    @staticmethod
    def render_heatmap(data, title: str = "Heatmap") -> None:
        """Render heatmap (legacy method)."""
        DataVisualizationComponents.render_heatmap_visualization(data, title)


# Export classes for easy importing
__all__ = [
    'DataVisualizationComponents', 
    'SophisticatedVisualizationIntelligence', 
    'ChartBuilder', 
    'VisualizationHelper'
]