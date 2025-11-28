"""
Debug script to test dashboard loading and identify the error source.
"""

import streamlit as st
import traceback

st.set_page_config(page_title="Dashboard Debug", page_icon="🔧")

st.title("🔧 Dashboard Debug Test")

try:
    st.write("Testing streamlit import and basic functionality...")
    
    # Test basic streamlit functions
    st.write("✅ st.write works")
    st.info("✅ st.info works") 
    st.success("✅ st.success works")
    
    # Test column layout
    col1, col2 = st.columns(2)
    with col1:
        st.write("Column 1 works")
    with col2:
        st.write("Column 2 works")
    
    st.write("✅ Basic streamlit functions are working")
    
    # Now try to import dashboard components
    st.write("Testing dashboard imports...")
    
    try:
        from streamlit_app.core import get_available_games, app_log
        st.success("✅ Core imports successful")
        
        games = get_available_games()
        st.write(f"Available games: {games}")
        
    except Exception as e:
        st.error(f"❌ Core import failed: {e}")
        st.code(traceback.format_exc())
    
    # Try navigation context simulation
    st.write("Testing navigation context...")
    
    try:
        from streamlit_app.registry.page_registry import NavigationContext
        
        # Create a mock navigation context
        nav_context = NavigationContext(
            current_page="dashboard",
            services_registry=None,
            ai_engines_registry=None,
            components_registry=None
        )
        
        st.success("✅ Navigation context created successfully")
        st.write(f"Navigation context: {nav_context}")
        
        # Try calling the dashboard render function
        from streamlit_app.pages.dashboard import render_page
        
        st.write("About to call dashboard render_page...")
        render_page(nav_context)
        
    except Exception as e:
        st.error(f"❌ Dashboard render failed: {e}")
        st.code(traceback.format_exc())
        
except Exception as e:
    st.error(f"❌ Debug script failed: {e}")
    st.code(traceback.format_exc())