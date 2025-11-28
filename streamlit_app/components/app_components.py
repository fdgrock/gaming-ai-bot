"""
Application Components for Streamlit App

This module provides reusable UI components for the lottery prediction system.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime


def create_header(title: str, version: str = "1.0.0", environment: str = "development") -> None:
    """Create the application header."""
    st.markdown(f"""
    <div style="padding: 1rem 0; border-bottom: 1px solid #e0e0e0; margin-bottom: 2rem;">
        <h1 style="margin: 0; color: #1f77b4;">🎰 {title}</h1>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
            Version {version} | Environment: {environment}
        </p>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar(pages: List[str], current_page: str, services: Any = None, config: Any = None) -> str:
    """Create the application sidebar with navigation and system info."""
    st.sidebar.markdown("### 🎯 Navigation")
    
    # Page selection
    selected_page = st.sidebar.radio("Select Page", pages, index=pages.index(current_page) if current_page in pages else 0)
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.markdown("### 📊 System Status")
    
    if services:
        try:
            # Show service status
            services_status = getattr(services, 'health_check', lambda: {'healthy': True, 'services': {}})()
            
            if services_status.get('healthy', True):
                st.sidebar.success("✅ All Systems Operational")
            else:
                st.sidebar.warning("⚠️ Some Services Down")
            
            # Show individual service status
            service_details = services_status.get('services', {})
            for service_name, status in service_details.items():
                status_icon = "✅" if status.get('status') == 'healthy' else "❌"
                st.sidebar.text(f"{status_icon} {service_name.title()}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Status Check Failed: {str(e)[:50]}...")
    else:
        st.sidebar.info("📡 Services Initializing...")
    
    # Cache status if available
    if services and hasattr(services, 'get_service'):
        try:
            cache_service = services.get_service('cache')
            if cache_service and hasattr(cache_service, 'get_cache_stats'):
                cache_stats = cache_service.get_cache_stats()
                hit_rate = cache_stats.get('hit_rate', 0) * 100
                st.sidebar.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        except Exception:
            pass
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🎲 Quick Predict", key="quick_predict"):
            st.session_state.quick_action = "predict"
    
    with col2:
        if st.button("📈 View Stats", key="quick_stats"):
            st.session_state.quick_action = "stats"
    
    # Recent activity
    st.sidebar.markdown("### 🕒 Recent Activity")
    
    recent_predictions = st.session_state.get('prediction_history', [])
    if recent_predictions:
        # Show last 3 predictions
        for i, pred in enumerate(recent_predictions[-3:]):
            timestamp = pred.get('timestamp', 'Unknown')
            confidence = pred.get('confidence', 0)
            st.sidebar.text(f"🎯 {confidence:.1%} | {timestamp}")
    else:
        st.sidebar.text("No recent predictions")
    
    return selected_page


def create_footer(version: str = "1.0.0", environment: str = "development") -> None:
    """Create the application footer."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Version:** {version}")
    
    with col2:
        st.markdown(f"**Environment:** {environment}")
    
    with col3:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.markdown(f"**Last Updated:** {current_time}")
    
    # Copyright and links
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666; font-size: 0.8rem;">
        <p>© 2025 Lottery Prediction System | Built with Streamlit</p>
        <p>
            <a href="#" style="color: #1f77b4; text-decoration: none;">Documentation</a> | 
            <a href="#" style="color: #1f77b4; text-decoration: none;">Support</a> | 
            <a href="#" style="color: #1f77b4; text-decoration: none;">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_loading_spinner(message: str = "Loading...") -> None:
    """Show a loading spinner with message."""
    with st.spinner(message):
        st.empty()


def show_success_message(message: str, duration: int = 3) -> None:
    """Show a success message."""
    st.success(f"✅ {message}")


def show_error_message(message: str, details: str = None) -> None:
    """Show an error message with optional details."""
    st.error(f"❌ {message}")
    if details:
        with st.expander("Error Details"):
            st.code(details)


def show_warning_message(message: str) -> None:
    """Show a warning message."""
    st.warning(f"⚠️ {message}")


def show_info_message(message: str) -> None:
    """Show an info message."""
    st.info(f"ℹ️ {message}")


def create_metric_card(title: str, value: Any, delta: Any = None, delta_color: str = "normal") -> None:
    """Create a metric card display."""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color
    )


def create_progress_bar(value: float, text: str = "") -> None:
    """Create a progress bar."""
    st.progress(value, text=text)


def create_status_indicator(status: str, message: str = "") -> None:
    """Create a status indicator."""
    if status == "success":
        st.success(f"✅ {message}")
    elif status == "error":
        st.error(f"❌ {message}")
    elif status == "warning":
        st.warning(f"⚠️ {message}")
    elif status == "info":
        st.info(f"ℹ️ {message}")
    else:
        st.text(f"📌 {message}")


def create_collapsible_section(title: str, content_func, expanded: bool = False):
    """Create a collapsible section."""
    with st.expander(title, expanded=expanded):
        content_func()


def create_tabs(tab_names: List[str], tab_contents: List[callable]) -> None:
    """Create tabbed interface."""
    tabs = st.tabs(tab_names)
    
    for tab, content_func in zip(tabs, tab_contents):
        with tab:
            content_func()


def create_data_table(data: Any, title: str = None) -> None:
    """Create a data table display."""
    if title:
        st.subheader(title)
    
    if hasattr(data, 'empty') and data.empty:
        st.info("No data available")
    else:
        st.dataframe(data, use_container_width=True)


def create_download_button(data: Any, filename: str, label: str = "Download") -> None:
    """Create a download button for data."""
    if hasattr(data, 'to_csv'):
        # DataFrame
        csv_data = data.to_csv(index=False)
        st.download_button(
            label=f"📥 {label}",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
    else:
        # JSON or other
        import json
        json_data = json.dumps(data, indent=2, default=str)
        st.download_button(
            label=f"📥 {label}",
            data=json_data,
            file_name=filename,
            mime="application/json"
        )


def create_confirmation_dialog(message: str, key: str = None) -> bool:
    """Create a confirmation dialog."""
    if st.button(f"⚠️ {message}", key=key):
        return st.checkbox("Yes, I'm sure", key=f"{key}_confirm" if key else "confirm")
    return False


def create_number_input_grid(num_slots: int, min_val: int = 1, max_val: int = 69, key_prefix: str = "num") -> List[int]:
    """Create a grid of number inputs for lottery numbers."""
    st.markdown("### Select Your Numbers")
    
    numbers = []
    cols = st.columns(num_slots)
    
    for i in range(num_slots):
        with cols[i]:
            num = st.number_input(
                f"#{i+1}",
                min_value=min_val,
                max_value=max_val,
                value=min_val,
                key=f"{key_prefix}_{i}"
            )
            numbers.append(num)
    
    return numbers


def create_game_selector() -> str:
    """Create a game selection dropdown."""
    games = ["Powerball", "Mega Millions", "Lucky for Life", "Cash 5"]
    selected_game = st.selectbox("🎮 Select Game", games)
    return selected_game


def create_strategy_selector() -> str:
    """Create a strategy selection component."""
    strategies = {
        "Conservative": "Lower risk, higher probability numbers",
        "Balanced": "Balanced approach with mixed strategies", 
        "Aggressive": "Higher risk, potentially higher reward"
    }
    
    selected = st.radio("🎯 Prediction Strategy", list(strategies.keys()))
    st.caption(strategies[selected])
    
    return selected.lower()


def create_prediction_display(prediction: Dict[str, Any]) -> None:
    """Display a prediction result."""
    st.markdown("### 🎯 Your Prediction")
    
    # Main numbers
    numbers = prediction.get('numbers', [])
    if numbers:
        # Display numbers as badges
        number_badges = " ".join([f"<span style='background: #1f77b4; color: white; padding: 0.5rem; margin: 0.2rem; border-radius: 50%; display: inline-block; width: 3rem; text-align: center; font-weight: bold;'>{num}</span>" for num in numbers])
        st.markdown(f"<div style='text-align: center; margin: 2rem 0;'>{number_badges}</div>", unsafe_allow_html=True)
    
    # Bonus number if available
    bonus = prediction.get('bonus')
    if bonus:
        st.markdown(f"<div style='text-align: center; margin: 1rem 0;'><span style='background: #ff6b6b; color: white; padding: 0.5rem 1rem; border-radius: 25px; font-weight: bold;'>Bonus: {bonus}</span></div>", unsafe_allow_html=True)
    
    # Prediction details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = prediction.get('confidence', 0)
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col2:
        strategy = prediction.get('strategy', 'Unknown')
        st.metric("Strategy", strategy.title())
    
    with col3:
        engine = prediction.get('engine', 'Unknown')
        st.metric("AI Engine", engine.replace('_', ' ').title())


# Export all components
__all__ = [
    'create_header', 'create_sidebar', 'create_footer',
    'show_loading_spinner', 'show_success_message', 'show_error_message',
    'show_warning_message', 'show_info_message', 'create_metric_card',
    'create_progress_bar', 'create_status_indicator', 'create_collapsible_section',
    'create_tabs', 'create_data_table', 'create_download_button',
    'create_confirmation_dialog', 'create_number_input_grid', 'create_game_selector',
    'create_strategy_selector', 'create_prediction_display'
]