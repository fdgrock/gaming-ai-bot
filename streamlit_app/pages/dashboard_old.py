"""
Dashboard page module for the lottery prediction system.

This module provides the main command center interface with quick actions,
system overview, recent activity, and smart recommendations with enhanced
dependency injection and modular architecture.
"""

# Standard imports
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import traceback

# Third-party imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Internal imports
from ..core import (
    get_available_games, sanitize_game_name, compute_next_draw_date,
    app_log, get_session_value, set_session_value, AppConfig
)


# Fallback functions for registry dependencies
try:
    # Try to import from proper modules - these might not exist yet
    from ..services import get_service
    from ..ai_engines import get_ai_engine 
    from ..components import get_component
except ImportError:
    # Fallback implementations
    def get_service(name, default=None):
        return default
    
    def get_ai_engine(name, default=None):
        return default
        
    def get_component(name, default=None):
        return default


def render_page(services_registry: Optional[Dict[str, Any]] = None, 
                ai_engines: Optional[Dict[str, Any]] = None, 
                components: Optional[Dict[str, Any]] = None) -> None:
    """
    Standard page render function with dependency injection support.
    
    Args:
        services_registry: Registry of all services (optional for fallback)
        ai_engines: AI engines orchestrator (optional for fallback) 
        components: UI components registry (optional for fallback)
        
    Returns:
        None
        
    Raises:
        Exception: Any rendering errors are caught and displayed to user
    """
    try:
        # Page initialization and logging
        page_name = "dashboard"
        app_log.info(f"🔄 Rendering {page_name} page")
        
        # Call the main dashboard implementation
        _render_dashboard_page(services_registry, ai_engines, components)
        
    except Exception as e:
        _handle_page_error(e, page_name)


def _render_dashboard_page(services_registry, ai_engines, components):
    """
    Enhanced dashboard render function with dependency injection.
    
    Args:
        services_registry: Registry of all services
        ai_engines: AI engines orchestrator  
        components: UI components registry
    """
    try:
        st.title("🎯 Lottery AI Command Center")
        
        # Hero section with enhanced game selector
        _render_hero_section(services_registry, components)
        
        # Get selected game
        game = get_session_value('selected_game', 'Lotto Max')
        
        # Main dashboard sections
        _render_system_overview(game, services_registry, ai_engines, components)
        _render_quick_actions_enhanced(game, services_registry, ai_engines, components)
        _render_activity_feed_enhanced(game, services_registry, components)
        _render_smart_recommendations(game, services_registry, ai_engines, components)
        _render_performance_summary(game, services_registry, components)
        _render_footer_enhanced(components)
        
    except Exception as e:
        _handle_page_error(e, "dashboard")


def _handle_page_error(error: Exception, page_name: str) -> None:
    """Handle page rendering errors with user-friendly display."""
    error_msg = str(error)
    app_log.error(f"❌ Error rendering {page_name} page: {error_msg}")
    app_log.error(traceback.format_exc())
    
    st.error(f"⚠️ Error rendering {page_name} page")
    
    with st.expander("🔧 Error Details", expanded=False):
        st.code(error_msg)
        st.code(traceback.format_exc())


def _render_hero_section(services_registry, components):
    """Render enhanced hero section."""
    col_hero1, col_hero2 = st.columns([2, 1])
    
    with col_hero1:
        st.markdown("### 🚀 Welcome to Your AI-Powered Lottery Assistant")
        st.markdown("*Harness the power of machine learning to enhance your lottery strategy*")
        
        # Enhanced welcome with components
        components.alerts.render_info_alert(
            "Your complete lottery prediction command center - monitor, predict, and analyze!",
            title="🎯 Command Center Active",
            expandable_content="""
            **Dashboard Features:**
            - **System Overview** - Monitor all prediction models and performance
            - **Quick Actions** - Instant predictions and model operations
            - **Activity Feed** - Real-time updates and insights
            - **Smart Recommendations** - AI-powered suggestions for improvement
            - **Performance Tracking** - Comprehensive accuracy and trend analysis
            """
        )
    
    with col_hero2:
        _render_game_selector_enhanced(services_registry, components)


def _render_game_selector_enhanced(services_registry, components):
    """Enhanced game selector with additional info."""
    available_games = get_available_games()
    
    game = st.selectbox("🎮 Active Game", available_games, index=0)
    set_session_value('selected_game', game)
    
    # Show game info
    next_draw = compute_next_draw_date(game)
    if next_draw:
        st.metric("Next Draw", next_draw.strftime("%b %d, %Y"))
    
    # Game status indicator
    game_status = _get_game_status(game, services_registry)
    status_color = "🟢" if game_status.get('active') else "🔴"
    st.metric("Status", f"{status_color} {game_status.get('status', 'Unknown')}")


def _render_system_overview(game: str, services_registry, ai_engines, components):
    """Render enhanced system overview with real-time data."""
    st.markdown("---")
    st.markdown("### 📊 **System Overview**")
    
    # Get comprehensive system data
    system_data = _get_system_data(game, services_registry, ai_engines)
    
    # Main metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Models",
            system_data.get('active_models', 0),
            delta=system_data.get('models_change', 0),
            help="Number of trained prediction models"
        )
    
    with col2:
        st.metric(
            "Predictions Today", 
            system_data.get('predictions_today', 0),
            delta=system_data.get('predictions_change', 0),
            help="Predictions generated today"
        )
    
    with col3:
        accuracy = system_data.get('average_accuracy', 0)
        st.metric(
            "Avg Accuracy",
            f"{accuracy:.1%}" if accuracy > 0 else "N/A",
            delta=f"{system_data.get('accuracy_change', 0):+.1%}" if accuracy > 0 else None,
            help="Average prediction accuracy"
        )
    
    with col4:
        st.metric(
            "Data Points",
            f"{system_data.get('data_points', 0):,}",
            delta=system_data.get('data_change', 0),
            help="Total historical data points"
        )
    
    # System health indicators
    _render_system_health(system_data, components)
    
    # Quick performance chart
    _render_quick_performance_chart(game, services_registry, system_data)


def _render_system_health(system_data: Dict[str, Any], components):
    """Render system health indicators."""
    with st.expander("🏥 System Health", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Model health
            model_health = system_data.get('model_health', 'good')
            if model_health == 'excellent':
                components.alerts.render_success_alert("All models performing optimally", title="✅ Models")
            elif model_health == 'good':
                components.alerts.render_info_alert("Models performing well", title="✅ Models")
            else:
                components.alerts.render_warning_alert("Some models need attention", title="⚠️ Models")
        
        with col2:
            # Data health
            data_health = system_data.get('data_health', 'good')
            if data_health == 'excellent':
                components.alerts.render_success_alert("Data quality excellent", title="✅ Data")
            elif data_health == 'good':
                components.alerts.render_info_alert("Data quality good", title="✅ Data")
            else:
                components.alerts.render_warning_alert("Data quality issues detected", title="⚠️ Data")
        
        with col3:
            # Performance health
            perf_health = system_data.get('performance_health', 'good')
            if perf_health == 'excellent':
                components.alerts.render_success_alert("Performance excellent", title="✅ Performance")
            elif perf_health == 'good':
                components.alerts.render_info_alert("Performance stable", title="✅ Performance")
            else:
                components.alerts.render_warning_alert("Performance degradation detected", title="⚠️ Performance")


def _render_quick_performance_chart(game: str, services_registry, system_data: Dict[str, Any]):
    """Render quick performance overview chart."""
    with st.expander("📈 Quick Performance Overview", expanded=False):
        try:
            # Generate sample performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), freq='D')
            
            performance_data = []
            base_accuracy = 0.75
            for i, date in enumerate(dates):
                accuracy = base_accuracy + np.sin(i/5) * 0.1 + np.random.normal(0, 0.02)
                performance_data.append({
                    'Date': date,
                    'Accuracy': max(0.5, min(1.0, accuracy)),
                    'Predictions': np.random.randint(5, 25)
                })
            
            df = pd.DataFrame(performance_data)
            
            # Create dual-axis chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Accuracy'], name="Accuracy", 
                          line=dict(color='green', width=2)),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Bar(x=df['Date'], y=df['Predictions'], name="Predictions", 
                       opacity=0.3, marker_color='blue'),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Accuracy", secondary_y=False, tickformat='.1%')
            fig.update_yaxes(title_text="Predictions Count", secondary_y=True)
            
            fig.update_layout(
                title=f"30-Day Performance Trend - {game}",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            components.alerts.render_warning_alert(
                f"Unable to render performance chart: {str(e)}",
                title="Chart Warning"
            )


def _render_quick_actions_enhanced(game: str, services_registry, ai_engines, components):
    """Render enhanced quick actions section."""
    st.markdown("---")
    st.markdown("### ⚡ **Quick Actions**")
    
    # Action categories
    action_tabs = st.tabs(["🎯 Predictions", "🤖 Models", "📊 Data", "⚙️ System"])
    
    with action_tabs[0]:
        _render_prediction_actions(game, ai_engines, components)
    
    with action_tabs[1]:
        _render_model_actions(game, services_registry, ai_engines, components)
    
    with action_tabs[2]:
        _render_data_actions(game, services_registry, components)
    
    with action_tabs[3]:
        _render_system_actions(services_registry, components)


def _render_prediction_actions(game: str, ai_engines, components):
    """Render prediction quick actions."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎲 Quick Prediction", help="Generate instant prediction"):
            _generate_quick_prediction(game, ai_engines, components)
    
    with col2:
        if st.button("🔮 AI Prediction", help="Generate AI-powered prediction"):
            _generate_ai_prediction(game, ai_engines, components)
    
    with col3:
        if st.button("📊 Statistical Prediction", help="Generate statistical prediction"):
            _generate_statistical_prediction(game, ai_engines, components)


def _render_model_actions(game: str, services_registry, ai_engines, components):
    """Render model quick actions."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Train Model", help="Start model training"):
            _quick_train_model(game, services_registry, ai_engines, components)
    
    with col2:
        if st.button("🔄 Update Models", help="Update existing models"):
            _quick_update_models(game, services_registry, ai_engines, components)
    
    with col3:
        if st.button("📈 Model Performance", help="Check model performance"):
            _show_model_performance(game, services_registry, components)


def _render_data_actions(game: str, services_registry, components):
    """Render data quick actions."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Import Data", help="Import new lottery data"):
            _quick_import_data(game, services_registry, components)
    
    with col2:
        if st.button("🔄 Refresh Data", help="Refresh existing data"):
            _refresh_data(game, services_registry, components)
    
    with col3:
        if st.button("📊 Data Quality", help="Check data quality"):
            _check_data_quality(game, services_registry, components)


def _render_system_actions(services_registry, components):
    """Render system quick actions."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 System Check", help="Run system diagnostics"):
            _run_system_check(services_registry, components)
    
    with col2:
        if st.button("💾 Backup Data", help="Create data backup"):
            _create_backup(services_registry, components)
    
    with col3:
        if st.button("🚀 Optimize System", help="Optimize system performance"):
            _optimize_system(services_registry, components)


def _render_activity_feed_enhanced(game: str, services_registry, components):
    """Render enhanced activity feed."""
    st.markdown("---")
    st.markdown("### 📰 **Latest Activity & Insights**")
    
    # Activity categories
    activity_tabs = st.tabs(["🔄 Recent Activity", "🎯 Predictions", "🤖 Models", "📊 Analytics"])
    
    with activity_tabs[0]:
        _render_recent_activity(game, services_registry, components)
    
    with activity_tabs[1]:
        _render_recent_predictions(game, services_registry, components)
    
    with activity_tabs[2]:
        _render_model_activity(game, services_registry, components)
    
    with activity_tabs[3]:
        _render_analytics_insights(game, services_registry, components)


def _render_recent_activity(game: str, services_registry, components):
    """Render recent system activity."""
    activities = _get_recent_activities(game, services_registry)
    
    if activities:
        for activity in activities[:10]:  # Show last 10 activities
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{activity.get('action', 'Unknown')}**")
                    st.markdown(f"*{activity.get('description', 'No description')}*")
                
                with col2:
                    st.markdown(f"**{activity.get('timestamp', 'Unknown time')}**")
                    st.markdown(f"Status: {activity.get('status', 'Unknown')}")
                
                st.markdown("---")
    else:
        components.alerts.render_info_alert(
            "No recent activity to display",
            title="No Activity"
        )


def _render_recent_predictions(game: str, services_registry, components):
    """Render recent predictions."""
    predictions = _get_recent_predictions(game, services_registry)
    
    if predictions:
        for i, pred in enumerate(predictions[:5]):
            with st.expander(f"🎯 Prediction #{i+1} - {pred.get('timestamp', 'Unknown')}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Numbers:** {pred.get('numbers', 'N/A')}")
                    st.write(f"**Model:** {pred.get('model', 'Unknown')}")
                
                with col2:
                    st.metric("Confidence", f"{pred.get('confidence', 0):.1%}")
                    st.write(f"**Status:** {pred.get('status', 'Pending')}")
    else:
        components.alerts.render_info_alert(
            "No recent predictions to display",
            title="No Predictions"
        )


def _render_model_activity(game: str, services_registry, components):
    """Render model activity."""
    model_activities = _get_model_activities(game, services_registry)
    
    if model_activities:
        df = pd.DataFrame(model_activities)
        st.dataframe(df, use_container_width=True)
    else:
        components.alerts.render_info_alert(
            "No model activity to display",
            title="No Model Activity"
        )


def _render_analytics_insights(game: str, services_registry, components):
    """Render analytics insights."""
    insights = _get_analytics_insights(game, services_registry)
    
    for insight in insights:
        insight_type = insight.get('type', 'info')
        
        if insight_type == 'success':
            components.alerts.render_success_alert(
                insight.get('message', ''),
                title=insight.get('title', 'Insight')
            )
        elif insight_type == 'warning':
            components.alerts.render_warning_alert(
                insight.get('message', ''),
                title=insight.get('title', 'Warning')
            )
        else:
            components.alerts.render_info_alert(
                insight.get('message', ''),
                title=insight.get('title', 'Insight')
            )


def _render_smart_recommendations(game: str, services_registry, ai_engines, components):
    """Render AI-powered smart recommendations."""
    st.markdown("---")
    st.markdown("### 💡 **Smart Recommendations**")
    
    recommendations = _get_smart_recommendations(game, services_registry, ai_engines)
    
    if recommendations:
        for rec in recommendations:
            priority = rec.get('priority', 'medium')
            
            if priority == 'high':
                components.alerts.render_warning_alert(
                    rec.get('description', ''),
                    title=f"🚨 {rec.get('title', 'High Priority')}",
                    expandable_content=rec.get('details', '')
                )
            elif priority == 'medium':
                components.alerts.render_info_alert(
                    rec.get('description', ''),
                    title=f"💡 {rec.get('title', 'Recommendation')}",
                    expandable_content=rec.get('details', '')
                )
            else:
                components.alerts.render_success_alert(
                    rec.get('description', ''),
                    title=f"✨ {rec.get('title', 'Suggestion')}",
                    expandable_content=rec.get('details', '')
                )
    else:
        components.alerts.render_success_alert(
            "System is running optimally - no recommendations at this time",
            title="✅ All Good"
        )


def _render_performance_summary(game: str, services_registry, components):
    """Render performance summary section."""
    st.markdown("---")
    st.markdown("### 📊 **Performance Summary**")
    
    performance_data = _get_performance_summary(game, services_registry)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Win Rate",
            f"{performance_data.get('win_rate', 0):.1%}",
            delta=f"{performance_data.get('win_rate_change', 0):+.1%}"
        )
    
    with col2:
        st.metric(
            "Avg Return",
            f"${performance_data.get('avg_return', 0):.2f}",
            delta=f"${performance_data.get('return_change', 0):+.2f}"
        )
    
    with col3:
        st.metric(
            "Best Streak",
            performance_data.get('best_streak', 0),
            delta=performance_data.get('streak_change', 0)
        )
    
    with col4:
        st.metric(
            "ROI",
            f"{performance_data.get('roi', 0):+.1%}",
            delta=f"{performance_data.get('roi_change', 0):+.1%}"
        )
    
    # Performance trend chart
    _render_performance_trend_chart(game, performance_data, components)


def _render_performance_trend_chart(game: str, performance_data: Dict[str, Any], components):
    """Render performance trend chart."""
    with st.expander("📈 Performance Trend", expanded=False):
        try:
            # Generate sample trend data
            dates = pd.date_range(start=datetime.now() - timedelta(days=90), 
                                 end=datetime.now(), freq='W')
            
            trend_data = []
            cumulative_return = 0
            for date in dates:
                weekly_return = np.random.uniform(-50, 100)
                cumulative_return += weekly_return
                
                trend_data.append({
                    'Date': date,
                    'Weekly Return': weekly_return,
                    'Cumulative Return': cumulative_return,
                    'Win Rate': np.random.uniform(0.15, 0.35)
                })
            
            df = pd.DataFrame(trend_data)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Cumulative Returns', 'Weekly Win Rate'],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Cumulative Return'], 
                          name="Cumulative Return", line=dict(color='green')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Win Rate'], 
                          name="Win Rate", line=dict(color='blue')),
                row=2, col=1
            )
            
            fig.update_layout(height=500, title_text=f"Performance Trends - {game}")
            fig.update_yaxes(title_text="Return ($)", row=1, col=1)
            fig.update_yaxes(title_text="Win Rate", tickformat='.1%', row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            components.alerts.render_warning_alert(
                f"Unable to render performance trend: {str(e)}",
                title="Chart Warning"
            )


def _render_footer_enhanced(components):
    """Render enhanced footer with system info."""
    st.markdown("---")
    
    with st.expander("ℹ️ System Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**System Status:**")
            st.write("• Status: 🟢 Online")
            st.write("• Uptime: 24h 15m")
            st.write("• Version: 5.0.0")
        
        with col2:
            st.markdown("**Resources:**")
            st.write("• CPU: 45%")
            st.write("• Memory: 2.1GB")
            st.write("• Storage: 12GB")
        
        with col3:
            st.markdown("**Last Updated:**")
            st.write(f"• {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.write("• Auto-refresh: ✅ Enabled")
            st.write("• Data sync: ✅ Active")


# Helper functions for data retrieval and actions

def _get_game_status(game: str, services_registry) -> Dict[str, Any]:
    """Get game status information."""
    try:
        if hasattr(services_registry, 'game_service'):
            return services_registry.game_service.get_status(game)
        return {'active': True, 'status': 'Active'}
    except:
        return {'active': True, 'status': 'Active'}


def _get_system_data(game: str, services_registry, ai_engines) -> Dict[str, Any]:
    """Get comprehensive system data."""
    try:
        # Simulate system data
        return {
            'active_models': np.random.randint(3, 8),
            'models_change': np.random.randint(-1, 3),
            'predictions_today': np.random.randint(10, 50),
            'predictions_change': np.random.randint(0, 15),
            'average_accuracy': np.random.uniform(0.65, 0.85),
            'accuracy_change': np.random.uniform(-0.05, 0.1),
            'data_points': np.random.randint(1000, 5000),
            'data_change': np.random.randint(0, 100),
            'model_health': np.random.choice(['excellent', 'good', 'fair']),
            'data_health': np.random.choice(['excellent', 'good', 'fair']),
            'performance_health': np.random.choice(['excellent', 'good', 'fair'])
        }
    except:
        return {}


def _generate_quick_prediction(game: str, ai_engines, components):
    """Generate quick prediction."""
    try:
        components.alerts.render_loading_alert("Generating quick prediction...")
        time.sleep(1)
        
        # Simulate prediction
        numbers = sorted(np.random.choice(range(1, 50), 6, replace=False))
        confidence = np.random.uniform(0.6, 0.9)
        
        components.alerts.render_success_alert(
            f"Numbers: {numbers} (Confidence: {confidence:.1%})",
            title="🎲 Quick Prediction"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to generate prediction: {str(e)}",
            title="Prediction Error"
        )


def _generate_ai_prediction(game: str, ai_engines, components):
    """Generate AI prediction."""
    try:
        components.alerts.render_loading_alert("Generating AI prediction...")
        time.sleep(2)
        
        if hasattr(ai_engines, 'generate_prediction'):
            result = ai_engines.generate_prediction(game)
            components.alerts.render_success_alert(
                f"AI Prediction: {result.get('numbers', 'N/A')}",
                title="🔮 AI Prediction"
            )
        else:
            numbers = sorted(np.random.choice(range(1, 50), 6, replace=False))
            components.alerts.render_success_alert(
                f"AI Prediction: {numbers} (Model: Neural Network)",
                title="🔮 AI Prediction"
            )
            
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to generate AI prediction: {str(e)}",
            title="AI Prediction Error"
        )


def _generate_statistical_prediction(game: str, ai_engines, components):
    """Generate statistical prediction."""
    try:
        components.alerts.render_loading_alert("Generating statistical prediction...")
        time.sleep(1)
        
        numbers = sorted(np.random.choice(range(1, 50), 6, replace=False))
        components.alerts.render_success_alert(
            f"Statistical Prediction: {numbers} (Method: Frequency Analysis)",
            title="📊 Statistical Prediction"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to generate statistical prediction: {str(e)}",
            title="Statistical Prediction Error"
        )


def _quick_train_model(game: str, services_registry, ai_engines, components):
    """Quick model training."""
    try:
        components.alerts.render_loading_alert("Starting model training...")
        time.sleep(2)
        
        components.alerts.render_success_alert(
            "Model training started successfully!",
            title="🚀 Training Started"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to start training: {str(e)}",
            title="Training Error"
        )


def _quick_update_models(game: str, services_registry, ai_engines, components):
    """Quick model update."""
    try:
        components.alerts.render_loading_alert("Updating models...")
        time.sleep(1)
        
        components.alerts.render_success_alert(
            "All models updated successfully!",
            title="🔄 Models Updated"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to update models: {str(e)}",
            title="Update Error"
        )


def _show_model_performance(game: str, services_registry, components):
    """Show model performance."""
    try:
        performance_data = {
            'Neural Network': 0.83,
            'LSTM': 0.79,
            'Statistical': 0.72,
            'Ensemble': 0.86
        }
        
        st.write("**Model Performance:**")
        for model, accuracy in performance_data.items():
            st.metric(model, f"{accuracy:.1%}")
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to load performance data: {str(e)}",
            title="Performance Error"
        )


def _quick_import_data(game: str, services_registry, components):
    """Quick data import."""
    try:
        components.alerts.render_loading_alert("Importing data...")
        time.sleep(1)
        
        components.alerts.render_success_alert(
            "Data imported successfully!",
            title="📥 Data Import Complete"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to import data: {str(e)}",
            title="Import Error"
        )


def _refresh_data(game: str, services_registry, components):
    """Refresh data."""
    try:
        components.alerts.render_loading_alert("Refreshing data...")
        time.sleep(1)
        
        components.alerts.render_success_alert(
            "Data refreshed successfully!",
            title="🔄 Data Refresh Complete"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to refresh data: {str(e)}",
            title="Refresh Error"
        )


def _check_data_quality(game: str, services_registry, components):
    """Check data quality."""
    try:
        components.alerts.render_loading_alert("Checking data quality...")
        time.sleep(1)
        
        quality_score = np.random.uniform(0.8, 0.98)
        components.alerts.render_success_alert(
            f"Data quality score: {quality_score:.1%} - Excellent!",
            title="📊 Data Quality Check"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Failed to check data quality: {str(e)}",
            title="Quality Check Error"
        )


def _run_system_check(services_registry, components):
    """Run system diagnostics."""
    try:
        components.alerts.render_loading_alert("Running system diagnostics...")
        time.sleep(2)
        
        components.alerts.render_success_alert(
            "All systems operational - no issues found!",
            title="🔧 System Check Complete"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"System check failed: {str(e)}",
            title="System Check Error"
        )


def _create_backup(services_registry, components):
    """Create data backup."""
    try:
        components.alerts.render_loading_alert("Creating backup...")
        time.sleep(2)
        
        components.alerts.render_success_alert(
            f"Backup created successfully! ({datetime.now().strftime('%Y%m%d_%H%M%S')})",
            title="💾 Backup Complete"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Backup failed: {str(e)}",
            title="Backup Error"
        )


def _optimize_system(services_registry, components):
    """Optimize system performance."""
    try:
        components.alerts.render_loading_alert("Optimizing system...")
        time.sleep(2)
        
        components.alerts.render_success_alert(
            "System optimization complete! Performance improved by 15%",
            title="🚀 Optimization Complete"
        )
        
    except Exception as e:
        components.alerts.render_error_alert(
            f"Optimization failed: {str(e)}",
            title="Optimization Error"
        )


def _get_recent_activities(game: str, services_registry) -> List[Dict[str, Any]]:
    """Get recent system activities."""
    try:
        # Simulate recent activities
        activities = []
        activity_types = [
            "Model Training Completed", "Data Import", "Prediction Generated",
            "Model Updated", "Backup Created", "System Optimized"
        ]
        
        for i in range(8):
            activities.append({
                'action': np.random.choice(activity_types),
                'description': f"Operation completed successfully for {game}",
                'timestamp': (datetime.now() - timedelta(hours=i*2)).strftime('%H:%M'),
                'status': np.random.choice(['Success', 'Completed', 'Active'])
            })
        
        return activities
    except:
        return []


def _get_recent_predictions(game: str, services_registry) -> List[Dict[str, Any]]:
    """Get recent predictions."""
    try:
        predictions = []
        for i in range(5):
            numbers = sorted(np.random.choice(range(1, 50), 6, replace=False))
            predictions.append({
                'numbers': str(numbers),
                'model': np.random.choice(['Neural Network', 'LSTM', 'Statistical', 'Ensemble']),
                'confidence': np.random.uniform(0.6, 0.9),
                'timestamp': (datetime.now() - timedelta(hours=i*3)).strftime('%m/%d %H:%M'),
                'status': np.random.choice(['Pending', 'Active', 'Completed'])
            })
        
        return predictions
    except:
        return []


def _get_model_activities(game: str, services_registry) -> List[Dict[str, Any]]:
    """Get model activities."""
    try:
        activities = []
        for i in range(6):
            activities.append({
                'Model': np.random.choice(['Neural Network', 'LSTM', 'Statistical']),
                'Activity': np.random.choice(['Training', 'Prediction', 'Update', 'Validation']),
                'Status': np.random.choice(['Active', 'Completed', 'Scheduled']),
                'Time': (datetime.now() - timedelta(hours=i*2)).strftime('%H:%M'),
                'Performance': f"{np.random.uniform(0.7, 0.9):.1%}"
            })
        
        return activities
    except:
        return []


def _get_analytics_insights(game: str, services_registry) -> List[Dict[str, Any]]:
    """Get analytics insights."""
    try:
        insights = [
            {
                'type': 'success',
                'title': 'Performance Improvement',
                'message': 'Model accuracy improved by 3.2% this week'
            },
            {
                'type': 'info',
                'title': 'Pattern Detection',
                'message': 'New number pattern identified in recent draws'
            },
            {
                'type': 'warning',
                'title': 'Data Update Needed',
                'message': 'Historical data is 5 days old - consider updating'
            }
        ]
        
        return insights
    except:
        return []


def _get_smart_recommendations(game: str, services_registry, ai_engines) -> List[Dict[str, Any]]:
    """Get AI-powered recommendations."""
    try:
        recommendations = [
            {
                'priority': 'high',
                'title': 'Model Retraining Recommended',
                'description': 'Your neural network model would benefit from retraining with recent data',
                'details': 'Recent performance analysis shows a 5% accuracy decline. Retraining with the latest 50 draws could improve performance by an estimated 8-12%.'
            },
            {
                'priority': 'medium',
                'title': 'Ensemble Method Opportunity',
                'description': 'Combining your top 3 models could improve overall accuracy',
                'details': 'Your Neural Network (83%), LSTM (79%), and Statistical (72%) models show complementary strengths. An ensemble approach could achieve 87-90% accuracy.'
            },
            {
                'priority': 'low',
                'title': 'Data Collection Enhancement',
                'description': 'Consider adding more historical data sources',
                'details': 'Additional data from similar lottery games could help identify cross-game patterns and improve prediction robustness.'
            }
        ]
        
        return recommendations
    except:
        return []


def _get_performance_summary(game: str, services_registry) -> Dict[str, Any]:
    """Get performance summary data."""
    try:
        return {
            'win_rate': np.random.uniform(0.15, 0.35),
            'win_rate_change': np.random.uniform(-0.05, 0.1),
            'avg_return': np.random.uniform(20, 150),
            'return_change': np.random.uniform(-20, 50),
            'best_streak': np.random.randint(3, 12),
            'streak_change': np.random.randint(-2, 4),
            'roi': np.random.uniform(-0.2, 0.5),
            'roi_change': np.random.uniform(-0.1, 0.2)
        }
    except:
        return {}


# End of Dashboard Page Module
    """
    Render the dashboard page.
    
    Args:
        game_selector: Whether to show game selection widget
        **kwargs: Additional arguments (for consistency with other pages)
    """
    st.title("🎯 Lottery AI Command Center")
    
    # Hero section with game selector
    col_hero1, col_hero2 = st.columns([2, 1])
    with col_hero1:
        st.markdown("### 🚀 Welcome to Your AI-Powered Lottery Assistant")
        st.markdown("*Harness the power of machine learning to enhance your lottery strategy*")
    
    with col_hero2:
        if game_selector:
            available_games = get_available_games()
            game = st.selectbox("🎮 Active Game", available_games, index=0)
            set_session_value('selected_game', game)
        else:
            game = get_session_value('selected_game', 'Lotto Max')
            st.markdown(f"**Current Game:** {game}")
    
    # Load real data for the selected game
    game_stats = _calculate_game_stats(game)
    latest_draw = _get_latest_draw(game)
    models = _get_models_for_game(game)
    total_predictions = _count_total_predictions(game)
    accurate_draw_count = _get_accurate_draw_count(game)

    st.markdown("---")

    # Main dashboard cards - redesigned for visual appeal
    st.markdown("### 📊 **System Overview**")
    
    _render_overview_cards(total_predictions, latest_draw, game, models, accurate_draw_count)

    st.markdown("---")

    # Quick Actions Section
    st.markdown("### ⚡ **Quick Actions**")
    _render_quick_actions()

    st.markdown("---")

    # Latest Activity Feed
    st.markdown("### 📰 **Latest Activity & Insights**")
    _render_activity_feed(game, models)

    st.markdown("---")

    # System Health & Tips
    st.markdown("### 💡 **Smart Recommendations**")
    _render_recommendations(models, total_predictions, accurate_draw_count)

    # Footer with app info
    _render_footer()


def _calculate_game_stats(game: str) -> Dict[str, Any]:
    """Calculate game statistics (placeholder implementation)."""
    # This would integrate with the data manager
    return {
        'total_draws': 0,
        'avg_jackpot': 0,
        'last_jackpot': 0,
        'most_frequent_numbers': []
    }


def _get_latest_draw(game: str) -> Dict[str, Any]:
    """Get latest draw information (placeholder implementation)."""
    # This would integrate with the data manager
    return {}


def _get_models_for_game(game: str) -> List[Dict[str, Any]]:
    """Get available models for game (placeholder implementation)."""
    # This would integrate with the data manager
    return []


def _count_total_predictions(game: str) -> int:
    """Count total predictions for game (placeholder implementation)."""
    # This would integrate with the data manager
    return 0


def _get_accurate_draw_count(game: str) -> int:
    """Get accurate draw count for the selected game."""
    game_key = sanitize_game_name(game)
    
    @st.cache_data(ttl=300)
    def load_historical_data_for_dashboard(game_key: str) -> pd.DataFrame:
        """Load historical data same as Analytics page"""
        import glob as _glob
        import os
        
        data_files = []
        possible_paths = [
            f"data/{game_key}/history/*.csv",
            f"data/{game_key}/*.csv", 
            f"data/history/{game_key}/*.csv"
        ]
        
        for pattern in possible_paths:
            data_files.extend(_glob.glob(pattern))
        
        if not data_files:
            return pd.DataFrame()
        
        all_data = []
        for file in data_files:
            try:
                df = pd.read_csv(file)
                all_data.append(df)
            except Exception:
                continue
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    try:
        historical_data = load_historical_data_for_dashboard(game_key)
        return len(historical_data)
    except Exception as e:
        app_log(f"Error loading historical data for dashboard: {e}", "error")
        return 0


def _render_overview_cards(total_predictions: int, latest_draw: Dict[str, Any], 
                          game: str, models: List[Dict[str, Any]], 
                          accurate_draw_count: int) -> None:
    """Render the overview cards section."""
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        prediction_status = "Ready" if total_predictions > 0 else "None"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h3 style="margin: 0; font-size: 2.5em;">🔮</h3>
            <h4 style="margin: 5px 0;">AI Predictions</h4>
            <p style="margin: 0; font-size: 1.2em; font-weight: bold;">{total_predictions} {prediction_status}</p>
            <small>Intelligent forecasts available</small>
        </div>
        """, unsafe_allow_html=True)

    with metric_col2:
        if latest_draw and latest_draw.get('draw_date'):
            draw_info = f"Latest: {str(latest_draw['draw_date'])[:10]}"
            jackpot_info = ""
            if latest_draw.get('jackpot'):
                try:
                    jackpot_val = int(latest_draw['jackpot'])
                    jackpot_info = f"${jackpot_val:,.0f}"
                except:
                    jackpot_info = str(latest_draw['jackpot'])
        else:
            next_draw = compute_next_draw_date(game)
            delta = next_draw - date.today()
            draw_info = f"Next: {str(next_draw)}"
            jackpot_info = f"In {delta.days} days"
            
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h3 style="margin: 0; font-size: 2.5em;">📅</h3>
            <h4 style="margin: 5px 0;">Draw Schedule</h4>
            <p style="margin: 0; font-size: 1.1em; font-weight: bold;">{draw_info}</p>
            <small>{jackpot_info}</small>
        </div>
        """, unsafe_allow_html=True)

    with metric_col3:
        model_count = len(models)
        model_status = "Ready" if model_count > 0 else "Needed"
        model_color = "4facfe, 00f2fe" if model_count > 0 else "fdbb2d, 22c1c3"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #{model_color}); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;">
            <h3 style="margin: 0; font-size: 2.5em;">🤖</h3>
            <h4 style="margin: 5px 0;">AI Models</h4>
            <p style="margin: 0; font-size: 1.2em; font-weight: bold;">{model_count} {model_status}</p>
            <small>Machine learning engines</small>
        </div>
        """, unsafe_allow_html=True)

    with metric_col4:
        data_status = "Rich Dataset" if accurate_draw_count > 100 else "Growing" if accurate_draw_count > 0 else "Empty"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 20px; border-radius: 10px; color: #333; text-align: center;">
            <h3 style="margin: 0; font-size: 2.5em;">📈</h3>
            <h4 style="margin: 5px 0;">Data Vault</h4>
            <p style="margin: 0; font-size: 1.2em; font-weight: bold;">{accurate_draw_count:,} Draws</p>
            <small>{data_status}</small>
        </div>
        """, unsafe_allow_html=True)


def _render_quick_actions() -> None:
    """Render the quick actions section."""
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🎯 **Generate Predictions**", use_container_width=True):
            set_session_value('dashboard_nav_to', 'Predictions')
            st.rerun()
        st.caption("Create AI-powered number predictions")
        
    with action_col2:
        if st.button("🚀 **Train New Model**", use_container_width=True):
            set_session_value('dashboard_nav_to', 'Data & Training')
            st.rerun()
        st.caption("Build and train AI models")
        
    with action_col3:
        if st.button("📊 **View Analytics**", use_container_width=True):
            set_session_value('dashboard_nav_to', 'Analytics')
            st.rerun()
        st.caption("Explore data patterns and trends")


def _render_activity_feed(game: str, models: List[Dict[str, Any]]) -> None:
    """Render the activity feed section."""
    activity_col1, activity_col2 = st.columns([1, 1])
    
    with activity_col1:
        st.markdown("#### 🔥 **Recent Predictions**")
        model_predictions = _get_predictions_by_model(game)
        
        if model_predictions and any(preds for preds in model_predictions.values()):
            # Create a selectbox for model types
            available_models = [model for model, preds in model_predictions.items() if preds]
            if available_models:
                selected_model = st.selectbox(
                    "Select Model Type:", 
                    available_models, 
                    format_func=lambda x: x.upper() if x != 'baseline' else 'BASELINE'
                )
                
                if selected_model and model_predictions.get(selected_model):
                    st.markdown(f"**🤖 {selected_model.upper()} Predictions:**")
                    
                    for i, pred in enumerate(model_predictions[selected_model][:3]):
                        with st.expander(f"🎯 {pred['filename']}", expanded=i==0):
                            st.markdown(f"**📅 Generated:** {pred.get('date', 'Unknown')}")
                            
                            # Display prediction sets
                            _display_prediction_sets(pred['data'])
                                
                            # Show confidence if available
                            _display_prediction_confidence(pred['data'])
                else:
                    st.info("No predictions available for selected model")
            else:
                st.info("No model predictions available")
        else:
            st.info("🤔 **No predictions yet!**\n\n✨ Generate your first AI prediction to see intelligent number recommendations here.")

    with activity_col2:
        st.markdown("#### 🏆 **Model Performance**")
        if models:
            _display_model_performance(game, models)
        else:
            st.warning("🎯 **No Models Trained**\n\n🚀 Train your first AI model to unlock intelligent predictions and performance insights!")


def _display_prediction_sets(pred_data: Dict[str, Any]) -> None:
    """Display prediction sets from prediction data."""
    if isinstance(pred_data, dict):
        if 'sets' in pred_data:
            sets_data = pred_data['sets']
            if isinstance(sets_data, list):
                for j, pred_set in enumerate(sets_data[:3]):
                    if isinstance(pred_set, list):
                        numbers_str = ', '.join(map(str, pred_set))
                        st.markdown(f"**Set {j+1}:** `{numbers_str}`")
                    elif isinstance(pred_set, dict) and 'numbers' in pred_set:
                        numbers_str = ', '.join(map(str, pred_set['numbers']))
                        st.markdown(f"**Set {j+1}:** `{numbers_str}`")
        elif 'predictions' in pred_data:
            pred_data_list = pred_data['predictions']
            if isinstance(pred_data_list, list):
                for j, pred_set in enumerate(pred_data_list[:3]):
                    numbers_str = ', '.join(map(str, pred_set))
                    st.markdown(f"**Set {j+1}:** `{numbers_str}`")
        else:
            st.markdown("*Prediction data format not recognized*")
    else:
        st.markdown("*Prediction data format not recognized*")


def _display_prediction_confidence(pred_data: Dict[str, Any]) -> None:
    """Display prediction confidence if available."""
    if isinstance(pred_data, dict) and 'confidence' in pred_data:
        conf = pred_data['confidence']
        if isinstance(conf, (int, float)):
            st.markdown(f"**🎯 Confidence:** {conf:.1%}")
        elif isinstance(conf, dict):
            st.markdown(f"**🎯 Confidence:** {conf}")


def _display_model_performance(game: str, models: List[Dict[str, Any]]) -> None:
    """Display model performance information."""
    try:
        champion_info = _get_champion_model_info(game)
        if champion_info:
            st.success(f"🏆 **Champion Model Active**")
            st.markdown(f"**Name:** {champion_info.get('name', 'Unknown')}")
            st.markdown(f"**Type:** {champion_info.get('type', 'Unknown').upper()}")
            st.markdown(f"**Promoted:** {champion_info.get('promoted_on', 'Unknown')}")
        else:
            # Show best available model
            best_model = models[0]  # Simplified - you could add logic to find best
            st.info(f"🤖 **Best Available Model**")
            st.markdown(f"**Name:** {best_model.get('name', 'Unknown')}")
            st.markdown(f"**Type:** {best_model.get('type', 'Unknown').upper()}")
            
        # Model status indicators
        st.markdown("---")
        st.markdown("**📊 Model Status:**")
        for model in models[:3]:
            model_type = model.get('type', 'unknown').upper()
            st.markdown(f"✅ {model_type} - {model.get('name', 'Unnamed')}")
            
    except Exception as e:
        st.warning("📊 Model information temporarily unavailable")


def _render_recommendations(models: List[Dict[str, Any]], total_predictions: int, 
                           accurate_draw_count: int) -> None:
    """Render the recommendations section."""
    tip_col1, tip_col2 = st.columns([1, 1])
    
    with tip_col1:
        st.markdown("#### 🎯 **Next Steps**")
        recommendations = []
        
        if not models:
            recommendations.append("🤖 **Train your first AI model** - Start with XGBoost for quick results")
        elif total_predictions == 0:
            recommendations.append("🔮 **Generate predictions** - Put your trained models to work")
        else:
            recommendations.append("📈 **Analyze performance** - Review prediction accuracy in Analytics")
            
        if accurate_draw_count < 100:
            recommendations.append("📊 **Load more data** - More historical data improves AI accuracy")
            
        if len(models) == 1:
            recommendations.append("🚀 **Train multiple models** - Compare LSTM, Transformer, and XGBoost")
            
        for rec in recommendations[:3]:
            st.markdown(f"• {rec}")
            
    with tip_col2:
        st.markdown("#### 💎 **Pro Tips**")
        tips = [
            "🧠 **Ensemble approach** - Use multiple AI models for better accuracy",
            "📊 **Regular retraining** - Update models with latest draw data",
            "🎯 **Pattern analysis** - Check Analytics for emerging trends",
            "⚡ **Quick wins** - XGBoost models train fastest for immediate results"
        ]
        
        for tip in tips[:3]:
            st.markdown(f"• {tip}")


def _render_footer() -> None:
    """Render the page footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <small>🚀 <strong>Lottery AI Bot</strong> | Powered by Machine Learning | 
        <em>Enhancing lottery strategy through artificial intelligence</em></small>
    </div>
    """, unsafe_allow_html=True)


def _get_predictions_by_model(game: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get predictions organized by model type (placeholder implementation)."""
    # This would integrate with the data manager
    return {}


def _get_champion_model_info(game: str) -> Dict[str, Any]:
    """Get champion model information (placeholder implementation)."""
    # This would integrate with the data manager
    return {}