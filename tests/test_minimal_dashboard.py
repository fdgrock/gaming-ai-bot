"""
Minimal test to isolate the dashboard shadowing issue.
"""

def test_imports():
    """Test importing just the core components that dashboard uses."""
    print("Testing core imports...")
    
    # Test individual imports
    try:
        import streamlit as st
        print(f"✅ streamlit imported: {type(st)} = {st}")
    except Exception as e:
        print(f"❌ Error importing streamlit: {e}")
        
    try:
        from streamlit_app.core import (
            get_available_games, sanitize_game_name, compute_next_draw_date,
            app_log, get_session_value, set_session_value, AppConfig
        )
        print("✅ Core imports successful")
        
        # Check if st is still streamlit
        import streamlit as st
        print(f"✅ After core imports, streamlit is: {type(st)} = {st}")
        
    except Exception as e:
        print(f"❌ Error with core imports: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Try the st_error function specifically
        from streamlit_app.core import st_error
        print(f"✅ st_error imported: {type(st_error)} = {st_error}")
        
        # Check if st is still streamlit after importing st_error
        import streamlit as st
        print(f"✅ After st_error import, streamlit is: {type(st)} = {st}")
        
    except Exception as e:
        print(f"❌ Error importing st_error: {e}")

if __name__ == "__main__":
    test_imports()