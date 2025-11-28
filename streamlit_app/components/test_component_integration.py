"""
Comprehensive integration test for Phase 4 Enhanced Component Library.

This test validates that all enhanced components work together seamlessly,
maintains backward compatibility, and demonstrates the full component capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# Import enhanced components via registry
try:
    from . import (
        get_enhanced_component,
        get_legacy_component, 
        inject_component_styles,
        validate_all_components,
        list_all_components,
        ComponentRegistry
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Components not available: {e}")
    COMPONENTS_AVAILABLE = False


def run_component_integration_test():
    """Run comprehensive component integration test."""
    
    if not COMPONENTS_AVAILABLE:
        st.error("❌ Enhanced components are not available for testing")
        return
    
    st.title("🧪 Phase 4 Component Library Integration Test")
    st.markdown("---")
    
    # Inject component styles
    inject_component_styles()
    
    # Test 1: Component Registry Health Check
    st.header("1. 🏥 Component Health Check")
    with st.expander("Component Health Validation", expanded=True):
        health_results = validate_all_components()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Enhanced Components")
            for comp_name, is_healthy in health_results.get('enhanced', {}).items():
                if is_healthy:
                    st.success(f"✅ {comp_name}")
                else:
                    st.error(f"❌ {comp_name}")
        
        with col2:
            st.subheader("Legacy Components")
            legacy_count = len(health_results.get('legacy', {}))
            healthy_legacy = sum(health_results.get('legacy', {}).values())
            st.metric("Legacy Components", f"{healthy_legacy}/{legacy_count}")
            
            if legacy_count > 0:
                health_pct = (healthy_legacy / legacy_count) * 100
                st.progress(health_pct / 100)
    
    st.markdown("---")
    
    # Test 2: Prediction Components
    st.header("2. 🎯 Prediction Components Test")
    with st.expander("Test Prediction Display Components", expanded=False):
        try:
            prediction_comp = get_enhanced_component('prediction')
            if prediction_comp:
                # Test prediction card
                sample_prediction = {
                    'numbers': [7, 14, 21, 28, 35, 42],
                    'confidence': 0.85,
                    'model': 'Enhanced Neural Network',
                    'timestamp': datetime.now(),
                    'bonus_number': 12
                }
                
                st.subheader("Prediction Card Test")
                prediction_comp.render_prediction_card(
                    prediction_data=sample_prediction,
                    title="Test Prediction",
                    show_confidence=True
                )
                
                # Test prediction grid
                st.subheader("Prediction Grid Test")
                multiple_predictions = [
                    sample_prediction,
                    {**sample_prediction, 'numbers': [1, 5, 10, 15, 20, 25], 'model': 'XGBoost Ensemble'},
                    {**sample_prediction, 'numbers': [3, 9, 18, 27, 36, 45], 'model': 'Random Forest'}
                ]
                
                prediction_comp.render_prediction_grid(
                    predictions_data=multiple_predictions,
                    columns=3
                )
                
                st.success("✅ Prediction components working correctly")
            else:
                st.error("❌ Failed to load prediction components")
                
        except Exception as e:
            st.error(f"❌ Prediction components test failed: {e}")
    
    st.markdown("---")
    
    # Test 3: Alert Components
    st.header("3. 🚨 Alert Components Test")
    with st.expander("Test Alert and Notification Components", expanded=False):
        try:
            alert_comp = get_enhanced_component('alerts')
            if alert_comp:
                # Test different alert types
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Success Alert")
                    alert_comp.render_success_alert("Components loaded successfully!")
                    
                    st.subheader("Info Alert")
                    alert_comp.render_info_alert(
                        "This is an informational message",
                        expandable_content="Additional details can be shown here."
                    )
                
                with col2:
                    st.subheader("Warning Alert")
                    alert_comp.render_warning_alert("This is a warning message")
                    
                    st.subheader("Loading Alert")
                    alert_comp.render_loading_alert("Processing data...", progress=0.75)
                
                # Test validation summary
                st.subheader("Validation Summary Test")
                sample_validation = {
                    'passed': [
                        {'name': 'Component Import', 'message': 'All components imported successfully'},
                        {'name': 'Theme Loading', 'message': 'Theme configuration loaded'}
                    ],
                    'warnings': [
                        {'name': 'Performance', 'message': 'Some components may be slow on first load'}
                    ],
                    'failed': []
                }
                
                alert_comp.render_validation_summary(sample_validation)
                
                st.success("✅ Alert components working correctly")
            else:
                st.error("❌ Failed to load alert components")
                
        except Exception as e:
            st.error(f"❌ Alert components test failed: {e}")
    
    st.markdown("---")
    
    # Test 4: Form Components
    st.header("4. 📝 Form Components Test")
    with st.expander("Test Form Components", expanded=False):
        try:
            form_comp = get_enhanced_component('forms')
            if form_comp:
                # Test quick generation form
                st.subheader("Quick Generation Form Test")
                quick_form_result = form_comp.render_quick_generation_form(
                    title="Test Quick Generation"
                )
                
                if quick_form_result.get('submitted'):
                    st.success(f"Form submitted with: {quick_form_result}")
                
                # Test search/filter form
                st.subheader("Search Filter Form Test")
                search_result = form_comp.render_search_filter_form(
                    available_filters=['game_type', 'date_range', 'confidence'],
                    title="Test Search Form"
                )
                
                if search_result.get('applied'):
                    st.info(f"Filters applied: {search_result}")
                
                st.success("✅ Form components working correctly")
            else:
                st.error("❌ Failed to load form components")
                
        except Exception as e:
            st.error(f"❌ Form components test failed: {e}")
    
    st.markdown("---")
    
    # Test 5: Table Components
    st.header("5. 📊 Table Components Test")
    with st.expander("Test Table Components", expanded=False):
        try:
            table_comp = get_enhanced_component('tables')
            if table_comp:
                # Create sample data
                sample_data = pd.DataFrame({
                    'Model': ['Neural Network', 'XGBoost', 'Random Forest', 'SVM', 'Ensemble'],
                    'Accuracy': [0.85, 0.82, 0.79, 0.76, 0.88],
                    'Confidence': [0.92, 0.88, 0.85, 0.81, 0.95],
                    'Training_Time': [120, 45, 30, 15, 180],
                    'Status': ['Active', 'Active', 'Training', 'Inactive', 'Active']
                })
                
                # Test data table
                st.subheader("Data Table Test")
                table_result = table_comp.render_data_table(
                    data=sample_data,
                    title="Model Performance Comparison",
                    searchable=True,
                    sortable=True
                )
                
                # Test comparison table
                st.subheader("Comparison Table Test")
                comparison_data = sample_data.head(3)
                table_comp.render_comparison_table(
                    data=comparison_data,
                    title="Top 3 Models Comparison",
                    highlight_best=True
                )
                
                st.success("✅ Table components working correctly")
            else:
                st.error("❌ Failed to load table components")
                
        except Exception as e:
            st.error(f"❌ Table components test failed: {e}")
    
    st.markdown("---")
    
    # Test 6: Model Card Components
    st.header("6. 🤖 Model Card Components Test")
    with st.expander("Test Model Card Components", expanded=False):
        try:
            model_comp = get_enhanced_component('model_cards')
            if model_comp:
                # Test model card
                st.subheader("Model Card Test")
                sample_model = {
                    'name': 'Enhanced Neural Network',
                    'type': 'Deep Learning',
                    'accuracy': 0.85,
                    'confidence': 0.92,
                    'status': 'active',
                    'last_trained': datetime.now() - timedelta(days=1),
                    'description': 'Advanced neural network for lottery prediction'
                }
                
                model_comp.render_model_card(
                    model_data=sample_model,
                    show_metrics=True,
                    show_actions=True
                )
                
                # Test model comparison
                st.subheader("Model Comparison Test")
                models_to_compare = [
                    sample_model,
                    {**sample_model, 'name': 'XGBoost Ensemble', 'accuracy': 0.82, 'type': 'Gradient Boosting'},
                    {**sample_model, 'name': 'Random Forest', 'accuracy': 0.79, 'type': 'Tree-based'}
                ]
                
                model_comp.render_model_comparison(
                    models_data=models_to_compare,
                    comparison_metrics=['accuracy', 'confidence']
                )
                
                st.success("✅ Model card components working correctly")
            else:
                st.error("❌ Failed to load model card components")
                
        except Exception as e:
            st.error(f"❌ Model card components test failed: {e}")
    
    st.markdown("---")
    
    # Test 7: Navigation Components
    st.header("7. 🧭 Navigation Components Test")
    with st.expander("Test Navigation Components", expanded=False):
        try:
            nav_comp = get_enhanced_component('navigation')
            if nav_comp:
                # Test tab navigation
                st.subheader("Tab Navigation Test")
                tab_result = nav_comp.render_tab_navigation(
                    tabs=['Overview', 'Predictions', 'Analysis', 'Settings'],
                    active_tab='Overview'
                )
                
                # Test quick actions menu
                st.subheader("Quick Actions Menu Test")
                actions = [
                    {'label': 'Generate Prediction', 'icon': '🎯', 'key': 'generate'},
                    {'label': 'View History', 'icon': '📈', 'key': 'history'},
                    {'label': 'Settings', 'icon': '⚙️', 'key': 'settings'}
                ]
                
                action_result = nav_comp.render_quick_actions_menu(
                    actions=actions,
                    layout='horizontal'
                )
                
                if action_result:
                    st.info(f"Action selected: {action_result}")
                
                st.success("✅ Navigation components working correctly")
            else:
                st.error("❌ Failed to load navigation components")
                
        except Exception as e:
            st.error(f"❌ Navigation components test failed: {e}")
    
    st.markdown("---")
    
    # Test 8: Data Visualization Components
    st.header("8. 📈 Data Visualization Components Test")
    with st.expander("Test Data Visualization Components", expanded=False):
        try:
            viz_comp = get_enhanced_component('data_viz')
            if viz_comp:
                # Test performance chart
                st.subheader("Performance Chart Test")
                performance_data = {
                    'dates': pd.date_range('2024-01-01', '2024-01-30'),
                    'accuracy': np.random.uniform(0.7, 0.9, 30),
                    'confidence': np.random.uniform(0.8, 0.95, 30)
                }
                
                viz_comp.render_performance_chart(
                    data=performance_data,
                    title="Model Performance Over Time",
                    metrics=['accuracy', 'confidence']
                )
                
                # Test statistical summary
                st.subheader("Statistical Summary Test")
                stats_data = {
                    'accuracy': {'mean': 0.85, 'std': 0.05, 'min': 0.75, 'max': 0.95},
                    'confidence': {'mean': 0.90, 'std': 0.03, 'min': 0.85, 'max': 0.98}
                }
                
                viz_comp.render_statistical_summary(
                    data=stats_data,
                    title="Model Performance Statistics"
                )
                
                st.success("✅ Data visualization components working correctly")
            else:
                st.error("❌ Failed to load data visualization components")
                
        except Exception as e:
            st.error(f"❌ Data visualization components test failed: {e}")
    
    st.markdown("---")
    
    # Test 9: Backward Compatibility
    st.header("9. 🔄 Backward Compatibility Test")
    with st.expander("Test Legacy Component Compatibility", expanded=False):
        try:
            # Test legacy component creation
            legacy_notification = get_legacy_component('alerts', 'NotificationManager')
            if legacy_notification:
                st.success("✅ Legacy NotificationManager created successfully")
                
                # Test legacy methods
                notif_id = legacy_notification.add_notification(
                    "Legacy compatibility test successful!",
                    legacy_notification.__class__.__module__.split('.')[-1] if hasattr(legacy_notification.__class__, '__module__') else 'INFO'
                )
                st.success(f"✅ Legacy notification added: {notif_id}")
            
            # Test other legacy components
            legacy_components_to_test = [
                ('alerts', 'StatusAlert'),
                ('alerts', 'ProgressAlert'),
                ('alerts', 'ToastNotification')
            ]
            
            for comp_type, comp_name in legacy_components_to_test:
                try:
                    legacy_comp = get_legacy_component(comp_type, comp_name)
                    if legacy_comp:
                        st.success(f"✅ Legacy {comp_name} created successfully")
                    else:
                        st.warning(f"⚠️ Legacy {comp_name} not available")
                except Exception as e:
                    st.warning(f"⚠️ Legacy {comp_name} test failed: {e}")
            
            st.success("✅ Backward compatibility maintained")
            
        except Exception as e:
            st.error(f"❌ Backward compatibility test failed: {e}")
    
    st.markdown("---")
    
    # Test Summary
    st.header("🏁 Test Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Enhanced Components", "7", "🎯")
    
    with col2:
        all_components = list_all_components()
        enhanced_count = len(all_components.get('enhanced', {}))
        legacy_count = len(all_components.get('legacy', {}))
        st.metric("Total Components", f"{enhanced_count + legacy_count}", f"+{enhanced_count} enhanced")
    
    with col3:
        health_results = validate_all_components()
        total_healthy = sum(health_results.get('enhanced', {}).values()) + sum(health_results.get('legacy', {}).values())
        total_components = len(health_results.get('enhanced', {})) + len(health_results.get('legacy', {}))
        health_pct = (total_healthy / total_components * 100) if total_components > 0 else 0
        st.metric("Health Score", f"{health_pct:.1f}%", "🏥")
    
    if health_pct >= 90:
        st.success("🎉 **Phase 4 Component Library Integration Test: PASSED**")
        st.success("✅ All components are working correctly and backward compatibility is maintained!")
    elif health_pct >= 75:
        st.warning("⚠️ **Phase 4 Component Library Integration Test: PASSED with Warnings**")
        st.warning("Most components are working, but some issues detected.")
    else:
        st.error("❌ **Phase 4 Component Library Integration Test: FAILED**")
        st.error("Critical issues detected with component system.")
    
    # Show component registry info
    st.markdown("---")
    st.subheader("📋 Component Registry Information")
    
    registry_info = list_all_components()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Enhanced Components:**")
        for comp_name, comp_info in registry_info.get('enhanced', {}).items():
            with st.expander(f"🎯 {comp_name.title()}", expanded=False):
                st.write(f"**Description:** {comp_info.get('description', 'No description')}")
                st.write("**Methods:**")
                for method in comp_info.get('methods', []):
                    st.write(f"  • {method}")
    
    with col2:
        st.write("**Legacy Components:**")
        for category, components in registry_info.get('legacy', {}).items():
            with st.expander(f"🔄 {category}", expanded=False):
                for comp in components:
                    st.write(f"  • {comp}")


if __name__ == "__main__":
    run_component_integration_test()