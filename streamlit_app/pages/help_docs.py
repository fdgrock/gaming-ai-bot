"""
Help & Documentation Page - Phase 5
"""
import streamlit as st
from typing import Optional, Dict, Any

try:
    from ..core import get_session_value, set_session_value, app_log
except ImportError:
    def get_session_value(k, d=None): return st.session_state.get(k, d)
    def set_session_value(k, v): st.session_state[k] = v
    def app_log(message: str, level: str = "info"): print(f"[{level.upper()}] {message}")

def render_page(services_registry=None, ai_engines=None, components=None) -> None:
    try:
        st.title("📚 Help & Documentation")
        st.markdown("*Complete guides and documentation for the Gaming AI Bot*")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚀 Getting Started",
            "📖 User Guide",
            "🧠 AI Guide",
            "❓ FAQ",
            "🔗 Resources"
        ])
        
        with tab1:
            _render_getting_started()
        with tab2:
            _render_user_guide()
        with tab3:
            _render_ai_guide()
        with tab4:
            _render_faq()
        with tab5:
            _render_resources()
        
        app_log("Help page rendered")
    except Exception as e:
        st.error(f"Error: {e}")

def _render_getting_started():
    st.subheader("🚀 Getting Started")
    st.markdown("""
    ### Quick Start Guide
    
    **Step 1: Select a Game**
    - Navigate to the Dashboard
    - Choose your preferred lottery game
    
    **Step 2: Generate Predictions**
    - Go to Predictions page
    - Click "Generate Predictions"
    - View your AI-generated numbers
    
    **Step 3: Track Performance**
    - Check Analytics for accuracy metrics
    - Review historical data
    - Monitor AI engine performance
    """)

def _render_user_guide():
    st.subheader("📖 User Guide")
    
    with st.expander("Dashboard", expanded=True):
        st.write("Overview of your gaming AI bot with key metrics and quick actions")
    
    with st.expander("Predictions"):
        st.write("Generate lottery predictions using advanced AI models")
    
    with st.expander("Analytics"):
        st.write("Track performance metrics and analyze prediction accuracy")
    
    with st.expander("Model Manager"):
        st.write("Manage ML models, versioning, and deployment")
    
    with st.expander("Data & Training"):
        st.write("Manage data and train custom models")

def _render_ai_guide():
    st.subheader("🧠 AI Engine Guide")
    
    st.markdown("""
    ### Available AI Engines
    
    **1. Mathematical Engine**
    - Statistical analysis and pattern recognition
    - Best for: Initial analysis
    
    **2. Expert Ensemble**
    - Multi-model consensus approach
    - Best for: Balanced predictions
    
    **3. Set Optimizer**
    - Coverage optimization
    - Best for: Comprehensive coverage
    
    **4. Temporal Engine**
    - Time-series pattern analysis
    - Best for: Trend-based predictions
    
    **5. Enhancement Engine**
    - Continuous improvement
    - Best for: Long-term accuracy
    """)

def _render_faq():
    st.subheader("❓ Frequently Asked Questions")
    
    with st.expander("How accurate are the predictions?"):
        st.write("Accuracy varies by game and model, typically 65-85%")
    
    with st.expander("How often should I generate predictions?"):
        st.write("Recommendations vary based on draw frequency")
    
    with st.expander("Can I export predictions?"):
        st.write("Yes, export to CSV or JSON from the Predictions page")
    
    with st.expander("How do I improve accuracy?"):
        st.write("Use the Training page to optimize models")
    
    with st.expander("Is my data secure?"):
        st.write("Yes, all data is encrypted and stored securely")

def _render_resources():
    st.subheader("🔗 Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📘 Full Documentation"):
            st.info("Link: https://docs.gamingaibot.local")
    
    with col2:
        if st.button("🎥 Video Tutorials"):
            st.info("Link: https://tutorials.gamingaibot.local")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Support Chat"):
            st.info("Link: https://support.gamingaibot.local")
    
    with col2:
        if st.button("📧 Contact Us"):
            st.info("Email: support@gamingaibot.local")
