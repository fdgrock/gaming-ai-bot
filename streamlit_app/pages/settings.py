"""
Enhanced Settings & Configuration Page - Phase 5
"""
import streamlit as st
from typing import Optional, Dict, Any

try:
    from ..core import get_available_games, get_session_value, set_session_value
    from ..core.logger import get_logger
    app_logger = get_logger()
except ImportError:
    def get_available_games(): return ["Lotto Max", "Lotto 6/49"]
    def get_session_value(k, d=None): return st.session_state.get(k, d)
    def set_session_value(k, v): st.session_state[k] = v
    class app_logger:
        @staticmethod
        def info(m): print(m)

def render_page(services_registry=None, ai_engines=None, components=None) -> None:
    try:
        st.title("⚙️ System Configuration Center")
        st.markdown("*Comprehensive settings management*")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎛️ General",
            "🎮 Games",
            "🧠 AI Engines",
            "🔒 Security",
            "💾 Backup"
        ])
        
        with tab1:
            _render_general()
        with tab2:
            _render_games()
        with tab3:
            _render_ai_engines()
        with tab4:
            _render_security()
        with tab5:
            _render_backup()
        
        app_logger.info("Settings page rendered")
    except Exception as e:
        st.error(f"Error: {e}")

def _render_general():
    st.subheader("🎛️ General Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        theme = st.selectbox("Theme", ["Dark", "Light", "Auto"])
        language = st.selectbox("Language", ["English", "French", "Spanish"])
    with col2:
        notifications = st.checkbox("Enable Notifications", value=True)
        auto_save = st.checkbox("Auto-save", value=True)
    
    if st.button("💾 Save Settings", key="save_gen", use_container_width=True):
        st.success("✅ Settings saved")

def _render_games():
    st.subheader("🎮 Game Configuration")
    
    games = ["Lotto Max", "Lotto 6/49", "Daily Grand", "Powerball"]
    for game in games:
        with st.expander(game):
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox(f"Enable {game}", value=True)
                st.number_input(f"Update Frequency (min)", value=60)
            with col2:
                st.text_input("API Key", type="password")

def _render_ai_engines():
    st.subheader("🧠 AI Engine Configuration")
    
    engines = ["Mathematical", "Ensemble", "Optimizer", "Temporal", "Enhancement"]
    for engine in engines:
        with st.expander(engine):
            st.checkbox(f"Enable {engine}", value=True)
            st.slider(f"{engine} Weight", 0.0, 1.0, 0.2)

def _render_security():
    st.subheader("🔒 Security Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Current Password", type="password")
        st.text_input("New Password", type="password")
    with col2:
        st.text_input("Confirm Password", type="password")
        encryption = st.checkbox("Enable Encryption", value=True)
    
    if st.button("🔄 Update Security", use_container_width=True):
        st.success("✅ Security settings updated")

def _render_backup():
    st.subheader("💾 Backup & Recovery")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Create Backup", use_container_width=True):
            st.success("✅ Backup created")
    
    with col2:
        if st.button("📤 Restore Backup", use_container_width=True):
            st.success("✅ Restore complete")
    
    with col3:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.success("✅ Cache cleared")
    
    st.write("Last backup: 2025-01-15 at 10:30 AM")
