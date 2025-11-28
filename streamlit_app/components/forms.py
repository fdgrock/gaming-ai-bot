"""
Form components for the lottery prediction system.

This module provides comprehensive form components for data input,
configuration, validation, and user interactions extracted from
the legacy application UI patterns.

Enhanced Components:
- FormComponents: Complete form library for all input and validation needs
- ModelConfigForm: Legacy model configuration system (backward compatibility)
- FormBuilder: Utility functions for form creation and management
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, date
import logging
import json
import re

logger = logging.getLogger(__name__)


class FormComponents:
    """
    Comprehensive form component library for lottery prediction system.
    
    This class provides a complete set of reusable form components
    extracted from the legacy application UI patterns. All components maintain
    consistent styling, validation, and error handling capabilities.
    
    Key Features:
    - Input Forms: Text, number, selection, and file input components
    - Validation Systems: Real-time validation with error messaging
    - Multi-step Wizards: Guided form workflows with progress tracking
    - Configuration Forms: Model and system configuration interfaces
    - Data Entry Forms: Structured data input with validation
    - Search Forms: Advanced search and filter interfaces
    - Upload Forms: File upload with validation and processing
    - Settings Forms: Application settings and preferences
    
    Form Categories:
    1. Basic Input Forms (text, number, selection)
    2. Advanced Input Forms (multi-select, sliders, dates)
    3. Configuration Forms (model, system settings)
    4. Data Entry Forms (structured input, validation)
    5. Upload Forms (file handling, processing)
    6. Search & Filter Forms (advanced queries)
    7. Multi-step Wizards (guided workflows)
    8. Validation Systems (real-time feedback)
    """
    
    @staticmethod
    def render_model_configuration_form(model_type: str = "lstm",
                                      current_config: Dict = None,
                                      title: str = "Model Configuration") -> Dict[str, Any]:
        """
        Render comprehensive model configuration form with validation.
        
        Args:
            model_type: Type of model to configure
            current_config: Current configuration values
            title: Form title
            
        Returns:
            Updated configuration dictionary
        """
        try:
            st.subheader(f"🤖 {title}")
            
            config = current_config or {}
            updated_config = {}
            
            # Basic model settings
            with st.expander("⚙️ Basic Settings", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    updated_config['model_name'] = st.text_input(
                        "Model Name",
                        value=config.get('model_name', f"{model_type.upper()}_Model"),
                        help="Unique identifier for this model"
                    )
                    
                    updated_config['description'] = st.text_area(
                        "Description",
                        value=config.get('description', ''),
                        help="Brief description of the model purpose"
                    )
                
                with col2:
                    updated_config['priority'] = st.selectbox(
                        "Priority Level",
                        options=["Low", "Medium", "High", "Critical"],
                        index=["Low", "Medium", "High", "Critical"].index(
                            config.get('priority', 'Medium')
                        ),
                        help="Model execution priority"
                    )
                    
                    updated_config['enabled'] = st.checkbox(
                        "Enable Model",
                        value=config.get('enabled', True),
                        help="Whether this model is active"
                    )
            
            # Advanced model parameters
            with st.expander("🔧 Advanced Parameters", expanded=False):
                FormComponents._render_advanced_model_params(
                    model_type, config, updated_config
                )
            
            # Training configuration
            with st.expander("📚 Training Configuration", expanded=False):
                FormComponents._render_training_config_form(
                    config, updated_config
                )
            
            # Validation and submission
            if st.button("✅ Save Configuration", type="primary"):
                validation_result = FormComponents._validate_model_config(updated_config)
                if validation_result['valid']:
                    st.success("✅ Configuration saved successfully!")
                    return updated_config
                else:
                    st.error("❌ Configuration validation failed:")
                    for error in validation_result['errors']:
                        st.error(f"  • {error}")
            
            return updated_config
        
        except Exception as e:
            logger.error(f"Error rendering model configuration form: {e}")
            st.error("Error displaying configuration form")
            return current_config or {}
    
    @staticmethod
    def render_strategy_comparison_form(title: str = "Strategy Comparison") -> Dict[str, Any]:
        """
        Render strategy comparison form with multiple strategy inputs.
        
        Args:
            title: Form title
            
        Returns:
            Strategy comparison configuration
        """
        try:
            st.subheader(f"⚔️ {title}")
            
            strategies = {}
            
            col1, col2 = st.columns(2)
            
            # Strategy 1 - Conservative
            with col1:
                st.markdown("**🛡️ Strategy 1: Conservative**")
                strategies['strategy1'] = {
                    'name': 'Conservative Strategy',
                    'target_probability': st.selectbox(
                        "Target Probability 1:",
                        [0.95, 0.99],
                        key="s1_prob",
                        format_func=lambda x: f"{x*100:.0f}%",
                        help="Conservative approach with higher confidence"
                    ),
                    'estimated_sets': st.number_input(
                        "Estimated Sets 1:",
                        min_value=1,
                        max_value=100,
                        value=15,
                        key="s1_sets",
                        help="Number of prediction sets to generate"
                    ),
                    'risk_level': st.selectbox(
                        "Risk Level 1:",
                        ["Low", "Medium", "High"],
                        key="s1_risk",
                        index=0,
                        help="Risk tolerance for this strategy"
                    )
                }
            
            # Strategy 2 - Aggressive
            with col2:
                st.markdown("**⚡ Strategy 2: Aggressive**")
                strategies['strategy2'] = {
                    'name': 'Aggressive Strategy',
                    'target_probability': st.selectbox(
                        "Target Probability 2:",
                        [0.90, 0.95],
                        key="s2_prob",
                        format_func=lambda x: f"{x*100:.0f}%",
                        help="Aggressive approach with moderate confidence"
                    ),
                    'estimated_sets': st.number_input(
                        "Estimated Sets 2:",
                        min_value=1,
                        max_value=100,
                        value=8,
                        key="s2_sets",
                        help="Number of prediction sets to generate"
                    ),
                    'risk_level': st.selectbox(
                        "Risk Level 2:",
                        ["Low", "Medium", "High"],
                        key="s2_risk",
                        index=2,
                        help="Risk tolerance for this strategy"
                    )
                }
            
            # Comparison controls
            st.markdown("---")
            comparison_enabled = st.checkbox(
                "🔄 Enable Real-time Comparison",
                value=True,
                help="Compare strategies as you modify parameters"
            )
            
            if st.button("⚔️ Compare Strategies", type="primary"):
                strategies['comparison_requested'] = True
                strategies['comparison_enabled'] = comparison_enabled
                st.success("🚀 Strategy comparison initiated!")
            
            return strategies
        
        except Exception as e:
            logger.error(f"Error rendering strategy comparison form: {e}")
            st.error("Error displaying strategy form")
            return {}
    
    @staticmethod
    def render_quick_generation_form(title: str = "Quick Generation") -> Dict[str, Any]:
        """
        Render quick generation form with intelligence and probability controls.
        
        Args:
            title: Form title
            
        Returns:
            Quick generation configuration
        """
        try:
            st.subheader(f"⚡ {title}")
            
            config = {}
            
            # Quick generation controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                config['intelligence_score'] = st.slider(
                    "🧠 Intelligence Score:",
                    min_value=50.0,
                    max_value=115.0,
                    value=85.0,
                    step=5.0,
                    help="AI intelligence level (50=Basic, 115=Maximum)"
                )
            
            with col2:
                config['target_probability'] = st.selectbox(
                    "🎯 Target Probability:",
                    [0.90, 0.95, 0.99],
                    index=1,
                    key="quick_target",
                    format_func=lambda x: f"{x*100:.0f}%",
                    help="Desired confidence level"
                )
            
            with col3:
                config['generation_mode'] = st.selectbox(
                    "🎮 Generation Mode:",
                    ["Standard", "Enhanced", "Ultra"],
                    index=1,
                    help="Generation complexity level"
                )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                config['use_ensemble'] = st.checkbox(
                    "🤖 Use Ensemble Models",
                    value=True,
                    help="Combine multiple models for better accuracy"
                )
                
                config['include_bonus'] = st.checkbox(
                    "🎁 Include Bonus Numbers",
                    value=True,
                    help="Generate bonus/extra numbers where applicable"
                )
                
                config['optimization_level'] = st.slider(
                    "⚙️ Optimization Level:",
                    min_value=1,
                    max_value=10,
                    value=7,
                    help="Computational intensity (1=Fast, 10=Maximum)"
                )
            
            # Generation button
            if st.button("🚀 Generate Quick Predictions", type="primary"):
                config['generate_requested'] = True
                st.success("⚡ Quick generation initiated!")
            
            return config
        
        except Exception as e:
            logger.error(f"Error rendering quick generation form: {e}")
            st.error("Error displaying generation form")
            return {}
    
    @staticmethod
    def render_data_upload_form(title: str = "Data Upload",
                              accepted_types: List[str] = None) -> Dict[str, Any]:
        """
        Render comprehensive data upload form with validation.
        
        Args:
            title: Form title
            accepted_types: List of accepted file types
            
        Returns:
            Upload configuration and file data
        """
        try:
            st.subheader(f"📁 {title}")
            
            if accepted_types is None:
                accepted_types = ["csv", "json", "xlsx", "txt", "xml"]
            
            upload_data = {}
            
            # Game configuration
            col1, col2 = st.columns(2)
            
            with col1:
                upload_data['game'] = st.selectbox(
                    "🎮 Select Game",
                    ["Lotto Max", "Lotto 6/49", "Daily Grand"],
                    index=0,
                    help="Target game for the uploaded data"
                )
            
            with col2:
                upload_data['data_type'] = st.selectbox(
                    "📊 Data Type",
                    ["Historical Draws", "Training Data", "Configuration", "Results"],
                    help="Type of data being uploaded"
                )
            
            # Pool size configuration
            st.markdown("#### 🎱 Number Pool Configuration")
            col3, col4 = st.columns(2)
            
            with col3:
                default_pool = 49 if upload_data['game'] == "Lotto 6/49" else 50
                upload_data['pool_size'] = st.selectbox(
                    'Number pool size',
                    [49, 50],
                    index=0 if default_pool == 49 else 1,
                    help="Maximum number in the pool"
                )
            
            with col4:
                default_main = 6 if upload_data['game'] == "Lotto 6/49" else 7
                upload_data['main_count'] = st.number_input(
                    'Main numbers per draw',
                    min_value=1,
                    max_value=10,
                    value=default_main,
                    help="How many main numbers per draw"
                )
            
            # File upload
            st.markdown("#### 📤 File Upload")
            uploaded_file = st.file_uploader(
                f"Upload {'/'.join(accepted_types).upper()} file",
                type=accepted_types,
                help="Select your data file to upload"
            )
            
            upload_data['file'] = uploaded_file
            
            # URL scraping option
            st.markdown("#### 🌐 Web Scraping (Alternative)")
            scraping_enabled = st.checkbox("Enable web scraping", help="Extract data from web pages")
            
            if scraping_enabled:
                upload_data['scrape_url'] = st.text_input(
                    "URL to scrape (table pages)",
                    help="Web page containing tabular data"
                )
                upload_data['css_selector'] = st.text_input(
                    "Optional CSS selector",
                    value="table",
                    help="CSS selector to target specific elements"
                )
                upload_data['scrape_year'] = st.number_input(
                    "Year (optional)",
                    min_value=1900,
                    max_value=2100,
                    value=date.today().year,
                    help="Filter data by year"
                )
            
            # Processing options
            with st.expander("⚙️ Processing Options"):
                upload_data['auto_validate'] = st.checkbox(
                    "Auto-validate data",
                    value=True,
                    help="Automatically validate uploaded data"
                )
                
                upload_data['create_backup'] = st.checkbox(
                    "Create backup",
                    value=True,
                    help="Create backup before processing"
                )
                
                upload_data['merge_existing'] = st.checkbox(
                    "Merge with existing data",
                    value=False,
                    help="Merge with existing dataset"
                )
            
            # Process button
            if uploaded_file or (scraping_enabled and upload_data.get('scrape_url')):
                if st.button("🔄 Process Data", type="primary"):
                    upload_data['process_requested'] = True
                    st.success("📊 Data processing initiated!")
            
            return upload_data
        
        except Exception as e:
            logger.error(f"Error rendering data upload form: {e}")
            st.error("Error displaying upload form")
            return {}
    
    @staticmethod
    def render_search_filter_form(title: str = "Search & Filter",
                                search_fields: List[str] = None) -> Dict[str, Any]:
        """
        Render advanced search and filter form.
        
        Args:
            title: Form title
            search_fields: Available search fields
            
        Returns:
            Search and filter configuration
        """
        try:
            st.subheader(f"🔍 {title}")
            
            if search_fields is None:
                search_fields = ["All Fields", "Model Name", "Performance", "Date", "Type"]
            
            search_config = {}
            
            # Main search
            search_config['search_term'] = st.text_input(
                "🔍 Search for topics, features, or parameters...",
                placeholder="Enter search terms",
                help="Search across all available content"
            )
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_config['search_field'] = st.selectbox(
                    "Search Field",
                    search_fields,
                    help="Specific field to search in"
                )
            
            with col2:
                search_config['date_range'] = st.selectbox(
                    "Date Range",
                    ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"],
                    help="Filter by date range"
                )
            
            with col3:
                search_config['result_limit'] = st.number_input(
                    "Result Limit",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10,
                    help="Maximum results to return"
                )
            
            # Advanced filters
            with st.expander("🔧 Advanced Filters"):
                search_config['include_inactive'] = st.checkbox(
                    "Include inactive items",
                    help="Include disabled or inactive items in results"
                )
                
                search_config['sort_by'] = st.selectbox(
                    "Sort By",
                    ["Relevance", "Date", "Name", "Performance", "Type"],
                    help="Sort results by specified criteria"
                )
                
                search_config['sort_order'] = st.radio(
                    "Sort Order",
                    ["Ascending", "Descending"],
                    index=1,
                    help="Sort direction"
                )
            
            # Search execution
            if st.button("🔍 Execute Search", type="primary"):
                search_config['search_requested'] = True
                st.success("🚀 Search initiated!")
            
            return search_config
        
        except Exception as e:
            logger.error(f"Error rendering search filter form: {e}")
            st.error("Error displaying search form")
            return {}
    
    @staticmethod
    def render_multi_step_wizard(steps: List[Dict], 
                               current_step: int = 0,
                               title: str = "Setup Wizard") -> Dict[str, Any]:
        """
        Render multi-step wizard with progress tracking.
        
        Args:
            steps: List of step configurations
            current_step: Current active step
            title: Wizard title
            
        Returns:
            Wizard state and collected data
        """
        try:
            st.subheader(f"🧙‍♂️ {title}")
            
            # Progress indicator
            progress = (current_step + 1) / len(steps)
            st.progress(progress)
            st.caption(f"Step {current_step + 1} of {len(steps)}: {steps[current_step]['title']}")
            
            wizard_data = {
                'current_step': current_step,
                'total_steps': len(steps),
                'step_data': {}
            }
            
            # Render current step
            step_config = steps[current_step]
            step_data = FormComponents._render_wizard_step(step_config, current_step)
            wizard_data['step_data'][current_step] = step_data
            
            # Navigation buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if current_step > 0:
                    if st.button("⬅️ Previous", key="wizard_prev"):
                        wizard_data['action'] = 'previous'
                        wizard_data['current_step'] = current_step - 1
            
            with col2:
                if current_step < len(steps) - 1:
                    if st.button("Next ➡️", key="wizard_next", type="primary"):
                        wizard_data['action'] = 'next'
                        wizard_data['current_step'] = current_step + 1
                else:
                    if st.button("✅ Complete", key="wizard_complete", type="primary"):
                        wizard_data['action'] = 'complete'
                        wizard_data['completed'] = True
            
            with col3:
                if st.button("❌ Cancel", key="wizard_cancel"):
                    wizard_data['action'] = 'cancel'
                    wizard_data['cancelled'] = True
            
            return wizard_data
        
        except Exception as e:
            logger.error(f"Error rendering multi-step wizard: {e}")
            st.error("Error displaying wizard")
            return {'error': True}
    
    # Utility methods for form rendering
    @staticmethod
    def _render_advanced_model_params(model_type: str, config: Dict, updated_config: Dict) -> None:
        """Render advanced model parameters based on model type."""
        try:
            if model_type.lower() in ['lstm', 'neural', 'deep']:
                col1, col2 = st.columns(2)
                
                with col1:
                    updated_config['hidden_layers'] = st.number_input(
                        "Hidden Layers",
                        min_value=1,
                        max_value=10,
                        value=config.get('hidden_layers', 3)
                    )
                    
                    updated_config['neurons_per_layer'] = st.number_input(
                        "Neurons per Layer",
                        min_value=32,
                        max_value=512,
                        value=config.get('neurons_per_layer', 128),
                        step=32
                    )
                
                with col2:
                    updated_config['dropout_rate'] = st.slider(
                        "Dropout Rate",
                        min_value=0.0,
                        max_value=0.8,
                        value=config.get('dropout_rate', 0.2),
                        step=0.1
                    )
                    
                    updated_config['learning_rate'] = st.selectbox(
                        "Learning Rate",
                        [0.001, 0.01, 0.1],
                        index=[0.001, 0.01, 0.1].index(config.get('learning_rate', 0.001))
                    )
            
            elif model_type.lower() in ['xgboost', 'tree', 'ensemble']:
                col1, col2 = st.columns(2)
                
                with col1:
                    updated_config['n_estimators'] = st.number_input(
                        "Number of Estimators",
                        min_value=50,
                        max_value=1000,
                        value=config.get('n_estimators', 100),
                        step=50
                    )
                    
                    updated_config['max_depth'] = st.number_input(
                        "Max Depth",
                        min_value=3,
                        max_value=20,
                        value=config.get('max_depth', 6)
                    )
                
                with col2:
                    updated_config['learning_rate'] = st.slider(
                        "Learning Rate",
                        min_value=0.01,
                        max_value=0.3,
                        value=config.get('learning_rate', 0.1),
                        step=0.01
                    )
                    
                    updated_config['subsample'] = st.slider(
                        "Subsample",
                        min_value=0.5,
                        max_value=1.0,
                        value=config.get('subsample', 0.8),
                        step=0.1
                    )
        
        except Exception as e:
            logger.error(f"Error rendering advanced parameters: {e}")
    
    @staticmethod
    def _render_training_config_form(config: Dict, updated_config: Dict) -> None:
        """Render training configuration form section."""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                updated_config['batch_size'] = st.selectbox(
                    "Batch Size",
                    [16, 32, 64, 128],
                    index=[16, 32, 64, 128].index(config.get('batch_size', 32))
                )
                
                updated_config['epochs'] = st.number_input(
                    "Training Epochs",
                    min_value=10,
                    max_value=1000,
                    value=config.get('epochs', 100),
                    step=10
                )
            
            with col2:
                updated_config['validation_split'] = st.slider(
                    "Validation Split",
                    min_value=0.1,
                    max_value=0.5,
                    value=config.get('validation_split', 0.2),
                    step=0.05
                )
                
                updated_config['early_stopping'] = st.checkbox(
                    "Early Stopping",
                    value=config.get('early_stopping', True)
                )
            
            # Advanced training options
            updated_config['use_gpu'] = st.checkbox(
                "Use GPU (if available)",
                value=config.get('use_gpu', False)
            )
            
            updated_config['save_checkpoints'] = st.checkbox(
                "Save Checkpoints",
                value=config.get('save_checkpoints', True)
            )
        
        except Exception as e:
            logger.error(f"Error rendering training config: {e}")
    
    @staticmethod
    def _validate_model_config(config: Dict) -> Dict[str, Any]:
        """Validate model configuration."""
        try:
            validation_result = {'valid': True, 'errors': []}
            
            # Required fields
            required_fields = ['model_name', 'priority']
            for field in required_fields:
                if not config.get(field):
                    validation_result['errors'].append(f"Missing required field: {field}")
            
            # Model name validation
            model_name = config.get('model_name', '')
            if model_name and not re.match(r'^[a-zA-Z0-9_\-]+$', model_name):
                validation_result['errors'].append("Model name can only contain letters, numbers, underscores, and hyphens")
            
            # Numeric validations
            numeric_validations = {
                'epochs': (1, 10000),
                'batch_size': (1, 1024),
                'hidden_layers': (1, 20),
                'neurons_per_layer': (1, 2048)
            }
            
            for field, (min_val, max_val) in numeric_validations.items():
                if field in config:
                    value = config[field]
                    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                        validation_result['errors'].append(f"{field} must be between {min_val} and {max_val}")
            
            validation_result['valid'] = len(validation_result['errors']) == 0
            return validation_result
        
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return {'valid': False, 'errors': ['Validation error occurred']}
    
    @staticmethod
    def _render_wizard_step(step_config: Dict, step_index: int) -> Dict[str, Any]:
        """Render individual wizard step."""
        try:
            st.markdown(f"### {step_config['title']}")
            
            if 'description' in step_config:
                st.info(step_config['description'])
            
            step_data = {}
            
            # Render step fields based on configuration
            if 'fields' in step_config:
                for field in step_config['fields']:
                    field_type = field.get('type', 'text')
                    field_key = f"step_{step_index}_{field['name']}"
                    
                    if field_type == 'text':
                        step_data[field['name']] = st.text_input(
                            field['label'],
                            value=field.get('default', ''),
                            key=field_key,
                            help=field.get('help', '')
                        )
                    elif field_type == 'number':
                        step_data[field['name']] = st.number_input(
                            field['label'],
                            min_value=field.get('min', 0),
                            max_value=field.get('max', 100),
                            value=field.get('default', 0),
                            key=field_key,
                            help=field.get('help', '')
                        )
                    elif field_type == 'select':
                        step_data[field['name']] = st.selectbox(
                            field['label'],
                            field.get('options', []),
                            key=field_key,
                            help=field.get('help', '')
                        )
                    elif field_type == 'checkbox':
                        step_data[field['name']] = st.checkbox(
                            field['label'],
                            value=field.get('default', False),
                            key=field_key,
                            help=field.get('help', '')
                        )
            
            return step_data
        
        except Exception as e:
            logger.error(f"Error rendering wizard step: {e}")
            return {}


class ModelConfigForm:
    """Form component for model configuration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model configuration form.
        
        Args:
            config: Form configuration
        """
        self.config = config or {}
    
    def render(self, model_type: str = "mathematical",
               current_config: Dict[str, Any] = None,
               title: str = "Model Configuration") -> Dict[str, Any]:
        """
        Render model configuration form.
        
        Args:
            model_type: Type of model to configure
            current_config: Current model configuration
            title: Form title
            
        Returns:
            Updated model configuration
        """
        try:
            st.subheader(title)
            
            if not current_config:
                current_config = self._get_default_config(model_type)
            
            # General settings
            with st.expander("General Settings", expanded=True):
                config = self._render_general_settings(current_config)
            
            # Model-specific settings
            with st.expander("Model-Specific Settings", expanded=True):
                config.update(self._render_model_specific_settings(model_type, current_config))
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                config.update(self._render_advanced_settings(current_config))
            
            # Validation and save
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Validate Config", use_container_width=True):
                    is_valid, errors = self._validate_config(config)
                    if is_valid:
                        st.success("✅ Configuration is valid")
                    else:
                        st.error(f"❌ Configuration errors: {', '.join(errors)}")
            
            with col2:
                if st.button("Reset to Default", use_container_width=True):
                    st.session_state.model_config_reset = True
                    st.rerun()
            
            with col3:
                if st.button("Save Configuration", use_container_width=True):
                    st.session_state.model_config_saved = config
                    st.success("✅ Configuration saved")
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to render model config form: {e}")
            st.error(f"Failed to display model configuration form: {e}")
            return {}
    
    def _render_general_settings(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render general model settings."""
        config = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['model_name'] = st.text_input(
                "Model Name:",
                value=current_config.get('model_name', 'My Model'),
                key="model_name"
            )
            
            config['description'] = st.text_area(
                "Description:",
                value=current_config.get('description', ''),
                key="model_description"
            )
            
            config['training_data_size'] = st.number_input(
                "Training Data Size:",
                min_value=10,
                max_value=10000,
                value=current_config.get('training_data_size', 1000),
                key="training_data_size"
            )
        
        with col2:
            config['prediction_strategy'] = st.selectbox(
                "Prediction Strategy:",
                options=['balanced', 'conservative', 'aggressive', 'experimental'],
                index=['balanced', 'conservative', 'aggressive', 'experimental'].index(
                    current_config.get('prediction_strategy', 'balanced')
                ),
                key="prediction_strategy"
            )
            
            config['confidence_threshold'] = st.slider(
                "Confidence Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get('confidence_threshold', 0.5),
                step=0.05,
                key="confidence_threshold"
            )
            
            config['enable_ensemble'] = st.checkbox(
                "Enable Ensemble Predictions:",
                value=current_config.get('enable_ensemble', True),
                key="enable_ensemble"
            )
        
        return config
    
    def _render_model_specific_settings(self, model_type: str, 
                                      current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render model-specific settings."""
        config = {}
        
        if model_type == 'mathematical':
            config.update(self._render_mathematical_settings(current_config))
        elif model_type == 'ensemble':
            config.update(self._render_ensemble_settings(current_config))
        elif model_type == 'temporal':
            config.update(self._render_temporal_settings(current_config))
        elif model_type == 'optimizer':
            config.update(self._render_optimizer_settings(current_config))
        
        return config
    
    def _render_mathematical_settings(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render mathematical model settings."""
        st.markdown("**Mathematical Engine Settings:**")
        
        config = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['frequency_weight'] = st.slider(
                "Frequency Analysis Weight:",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get('frequency_weight', 0.4),
                step=0.05,
                key="freq_weight"
            )
            
            config['gap_weight'] = st.slider(
                "Gap Analysis Weight:",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get('gap_weight', 0.3),
                step=0.05,
                key="gap_weight"
            )
        
        with col2:
            config['trend_weight'] = st.slider(
                "Trend Analysis Weight:",
                min_value=0.0,
                max_value=1.0,
                value=current_config.get('trend_weight', 0.3),
                step=0.05,
                key="trend_weight"
            )
            
            config['smoothing_factor'] = st.slider(
                "Smoothing Factor:",
                min_value=0.1,
                max_value=0.9,
                value=current_config.get('smoothing_factor', 0.5),
                step=0.05,
                key="smoothing_factor"
            )
        
        return config
    
    def _render_ensemble_settings(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render ensemble model settings."""
        st.markdown("**Ensemble Engine Settings:**")
        
        config = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['n_estimators'] = st.number_input(
                "Number of Estimators:",
                min_value=10,
                max_value=1000,
                value=current_config.get('n_estimators', 100),
                key="n_estimators"
            )
            
            config['max_depth'] = st.number_input(
                "Maximum Depth:",
                min_value=1,
                max_value=50,
                value=current_config.get('max_depth', 10),
                key="max_depth"
            )
        
        with col2:
            config['learning_rate'] = st.number_input(
                "Learning Rate:",
                min_value=0.001,
                max_value=1.0,
                value=current_config.get('learning_rate', 0.1),
                step=0.001,
                format="%.3f",
                key="learning_rate"
            )
            
            config['voting_method'] = st.selectbox(
                "Voting Method:",
                options=['hard', 'soft'],
                index=['hard', 'soft'].index(current_config.get('voting_method', 'soft')),
                key="voting_method"
            )
        
        return config
    
    def _render_temporal_settings(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render temporal model settings."""
        st.markdown("**Temporal Engine Settings:**")
        
        config = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['lstm_units'] = st.number_input(
                "LSTM Units:",
                min_value=16,
                max_value=512,
                value=current_config.get('lstm_units', 64),
                key="lstm_units"
            )
            
            config['sequence_length'] = st.number_input(
                "Sequence Length:",
                min_value=5,
                max_value=200,
                value=current_config.get('sequence_length', 30),
                key="sequence_length"
            )
        
        with col2:
            config['dropout_rate'] = st.slider(
                "Dropout Rate:",
                min_value=0.0,
                max_value=0.8,
                value=current_config.get('dropout_rate', 0.2),
                step=0.05,
                key="dropout_rate"
            )
            
            config['epochs'] = st.number_input(
                "Training Epochs:",
                min_value=10,
                max_value=1000,
                value=current_config.get('epochs', 100),
                key="epochs"
            )
        
        return config
    
    def _render_optimizer_settings(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render optimizer model settings."""
        st.markdown("**Optimizer Engine Settings:**")
        
        config = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['population_size'] = st.number_input(
                "Population Size:",
                min_value=20,
                max_value=1000,
                value=current_config.get('population_size', 100),
                key="population_size"
            )
            
            config['mutation_rate'] = st.slider(
                "Mutation Rate:",
                min_value=0.01,
                max_value=0.5,
                value=current_config.get('mutation_rate', 0.1),
                step=0.01,
                key="mutation_rate"
            )
        
        with col2:
            config['crossover_rate'] = st.slider(
                "Crossover Rate:",
                min_value=0.1,
                max_value=1.0,
                value=current_config.get('crossover_rate', 0.8),
                step=0.05,
                key="crossover_rate"
            )
            
            config['max_generations'] = st.number_input(
                "Maximum Generations:",
                min_value=10,
                max_value=1000,
                value=current_config.get('max_generations', 100),
                key="max_generations"
            )
        
        return config
    
    def _render_advanced_settings(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render advanced settings."""
        config = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['random_seed'] = st.number_input(
                "Random Seed:",
                min_value=0,
                max_value=999999,
                value=current_config.get('random_seed', 42),
                key="random_seed"
            )
            
            config['cross_validation_folds'] = st.number_input(
                "Cross Validation Folds:",
                min_value=3,
                max_value=20,
                value=current_config.get('cross_validation_folds', 5),
                key="cv_folds"
            )
        
        with col2:
            config['early_stopping'] = st.checkbox(
                "Enable Early Stopping:",
                value=current_config.get('early_stopping', True),
                key="early_stopping"
            )
            
            config['patience'] = st.number_input(
                "Patience (epochs):",
                min_value=5,
                max_value=100,
                value=current_config.get('patience', 10),
                key="patience"
            )
        
        return config
    
    def _get_default_config(self, model_type: str) -> Dict[str, Any]:
        """Get default configuration for model type."""
        defaults = {
            'mathematical': {
                'model_name': 'Mathematical Engine',
                'description': 'Statistical analysis with frequency and trend analysis',
                'training_data_size': 1000,
                'prediction_strategy': 'balanced',
                'confidence_threshold': 0.5,
                'enable_ensemble': True,
                'frequency_weight': 0.4,
                'gap_weight': 0.3,
                'trend_weight': 0.3,
                'smoothing_factor': 0.5
            },
            'ensemble': {
                'model_name': 'Ensemble Engine',
                'description': 'Machine learning ensemble with multiple algorithms',
                'training_data_size': 1000,
                'prediction_strategy': 'balanced',
                'confidence_threshold': 0.6,
                'enable_ensemble': True,
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'voting_method': 'soft'
            },
            'temporal': {
                'model_name': 'Temporal Engine',
                'description': 'Time-series analysis with LSTM and ARIMA',
                'training_data_size': 1000,
                'prediction_strategy': 'balanced',
                'confidence_threshold': 0.5,
                'enable_ensemble': True,
                'lstm_units': 64,
                'sequence_length': 30,
                'dropout_rate': 0.2,
                'epochs': 100
            },
            'optimizer': {
                'model_name': 'Optimizer Engine',
                'description': 'Combinatorial optimization for number selection',
                'training_data_size': 1000,
                'prediction_strategy': 'balanced',
                'confidence_threshold': 0.5,
                'enable_ensemble': True,
                'population_size': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'max_generations': 100
            }
        }
        
        base_config = {
            'random_seed': 42,
            'cross_validation_folds': 5,
            'early_stopping': True,
            'patience': 10
        }
        
        config = defaults.get(model_type, defaults['mathematical']).copy()
        config.update(base_config)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate model configuration."""
        errors = []
        
        # Check required fields
        required_fields = ['model_name', 'training_data_size', 'confidence_threshold']
        for field in required_fields:
            if field not in config or config[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric ranges
        if 'confidence_threshold' in config:
            if not (0 <= config['confidence_threshold'] <= 1):
                errors.append("Confidence threshold must be between 0 and 1")
        
        if 'training_data_size' in config:
            if config['training_data_size'] < 10:
                errors.append("Training data size must be at least 10")
        
        # Validate weights sum to 1 for mathematical model
        weight_fields = ['frequency_weight', 'gap_weight', 'trend_weight']
        if all(field in config for field in weight_fields):
            weight_sum = sum(config[field] for field in weight_fields)
            if abs(weight_sum - 1.0) > 0.01:
                errors.append("Mathematical engine weights must sum to 1.0")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class PredictionForm:
    """Form component for prediction parameters."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize prediction form."""
        self.config = config or {}
    
    def render(self, title: str = "Prediction Parameters") -> Dict[str, Any]:
        """
        Render prediction form.
        
        Args:
            title: Form title
            
        Returns:
            Prediction parameters
        """
        try:
            st.subheader(title)
            
            params = {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                params['num_predictions'] = st.number_input(
                    "Number of Predictions:",
                    min_value=1,
                    max_value=50,
                    value=5,
                    key="num_predictions"
                )
                
                params['strategy'] = st.selectbox(
                    "Prediction Strategy:",
                    options=['balanced', 'conservative', 'aggressive', 'experimental'],
                    index=0,
                    key="pred_strategy"
                )
                
                params['use_ai_engines'] = st.multiselect(
                    "Use AI Engines:",
                    options=['Mathematical', 'Ensemble', 'Temporal', 'Optimizer'],
                    default=['Mathematical', 'Ensemble'],
                    key="use_engines"
                )
            
            with col2:
                params['confidence_filter'] = st.slider(
                    "Minimum Confidence:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    key="confidence_filter"
                )
                
                params['diversification'] = st.slider(
                    "Diversification Level:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="diversification"
                )
                
                params['include_bonus'] = st.checkbox(
                    "Include Bonus Ball Predictions:",
                    value=True,
                    key="include_bonus"
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                col3, col4 = st.columns(2)
                
                with col3:
                    params['seed_numbers'] = st.text_input(
                        "Seed Numbers (comma-separated):",
                        value="",
                        help="Force certain numbers to be included",
                        key="seed_numbers"
                    )
                    
                    params['avoid_numbers'] = st.text_input(
                        "Numbers to Avoid (comma-separated):",
                        value="",
                        help="Numbers to exclude from predictions",
                        key="avoid_numbers"
                    )
                
                with col4:
                    params['max_consecutive'] = st.number_input(
                        "Max Consecutive Numbers:",
                        min_value=1,
                        max_value=6,
                        value=3,
                        key="max_consecutive"
                    )
                    
                    params['balance_odd_even'] = st.checkbox(
                        "Balance Odd/Even Numbers:",
                        value=True,
                        key="balance_odd_even"
                    )
            
            # Process text inputs
            if params['seed_numbers']:
                try:
                    params['seed_numbers'] = [int(x.strip()) for x in params['seed_numbers'].split(',') if x.strip()]
                except ValueError:
                    st.warning("⚠️ Invalid seed numbers format")
                    params['seed_numbers'] = []
            else:
                params['seed_numbers'] = []
            
            if params['avoid_numbers']:
                try:
                    params['avoid_numbers'] = [int(x.strip()) for x in params['avoid_numbers'].split(',') if x.strip()]
                except ValueError:
                    st.warning("⚠️ Invalid avoid numbers format")
                    params['avoid_numbers'] = []
            else:
                params['avoid_numbers'] = []
            
            # Generate button
            if st.button("🎯 Generate Predictions", type="primary", use_container_width=True):
                params['generate_clicked'] = True
            else:
                params['generate_clicked'] = False
            
            return params
            
        except Exception as e:
            logger.error(f"❌ Failed to render prediction form: {e}")
            st.error(f"Failed to display prediction form: {e}")
            return {}
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class DataUploadForm:
    """Form component for data upload and import."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize data upload form."""
        self.config = config or {}
    
    def render(self, title: str = "Data Upload") -> Dict[str, Any]:
        """
        Render data upload form.
        
        Args:
            title: Form title
            
        Returns:
            Upload results and data
        """
        try:
            st.subheader(title)
            
            result = {}
            
            # Upload method selection
            upload_method = st.radio(
                "Choose Upload Method:",
                options=["File Upload", "Manual Entry", "URL Import"],
                horizontal=True,
                key="upload_method"
            )
            
            if upload_method == "File Upload":
                result.update(self._render_file_upload())
            elif upload_method == "Manual Entry":
                result.update(self._render_manual_entry())
            elif upload_method == "URL Import":
                result.update(self._render_url_import())
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to render data upload form: {e}")
            st.error(f"Failed to display data upload form: {e}")
            return {}
    
    def _render_file_upload(self) -> Dict[str, Any]:
        """Render file upload section."""
        result = {}
        
        st.markdown("**Upload CSV File:**")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file with lottery draw data",
            key="csv_upload"
        )
        
        if uploaded_file is not None:
            try:
                # Read file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully: {len(df)} rows")
                
                # Data preview
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column mapping
                st.markdown("**Column Mapping:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    date_column = st.selectbox(
                        "Date Column:",
                        options=df.columns.tolist(),
                        key="date_col_mapping"
                    )
                
                with col2:
                    numbers_column = st.selectbox(
                        "Numbers Column:",
                        options=df.columns.tolist(),
                        key="numbers_col_mapping"
                    )
                
                # Data validation
                if st.button("Validate Data", key="validate_upload"):
                    validation_result = self._validate_uploaded_data(df, date_column, numbers_column)
                    
                    if validation_result['is_valid']:
                        st.success("✅ Data validation passed")
                        result['data'] = df
                        result['date_column'] = date_column
                        result['numbers_column'] = numbers_column
                        result['is_valid'] = True
                    else:
                        st.error(f"❌ Data validation failed: {validation_result['errors']}")
                        result['is_valid'] = False
                
            except Exception as e:
                st.error(f"❌ Failed to read file: {e}")
                result['is_valid'] = False
        
        return result
    
    def _render_manual_entry(self) -> Dict[str, Any]:
        """Render manual data entry section."""
        result = {}
        
        st.markdown("**Manual Data Entry:**")
        
        # Number of entries
        num_entries = st.number_input(
            "Number of Draw Entries:",
            min_value=1,
            max_value=100,
            value=5,
            key="num_manual_entries"
        )
        
        # Manual entry form
        entries = []
        
        with st.form("manual_entry_form"):
            st.markdown("**Enter Draw Data:**")
            
            for i in range(num_entries):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    draw_date = st.date_input(
                        f"Date {i+1}:",
                        value=date.today(),
                        key=f"manual_date_{i}"
                    )
                
                with col2:
                    numbers_text = st.text_input(
                        f"Numbers {i+1} (comma-separated):",
                        placeholder="1,15,23,34,42,49",
                        key=f"manual_numbers_{i}"
                    )
                
                if draw_date and numbers_text:
                    try:
                        numbers = [int(x.strip()) for x in numbers_text.split(',')]
                        entries.append({
                            'date': draw_date,
                            'numbers': numbers
                        })
                    except ValueError:
                        st.warning(f"⚠️ Invalid numbers format in entry {i+1}")
            
            submitted = st.form_submit_button("Save Manual Entries")
            
            if submitted and entries:
                df = pd.DataFrame(entries)
                st.success(f"✅ {len(entries)} entries saved")
                
                result['data'] = df
                result['date_column'] = 'date'
                result['numbers_column'] = 'numbers'
                result['is_valid'] = True
        
        return result
    
    def _render_url_import(self) -> Dict[str, Any]:
        """Render URL import section."""
        result = {}
        
        st.markdown("**Import from URL:**")
        
        url = st.text_input(
            "Data URL:",
            placeholder="https://example.com/lottery-data.csv",
            key="import_url"
        )
        
        if url and st.button("Import from URL", key="url_import"):
            try:
                df = pd.read_csv(url)
                st.success(f"✅ Data imported successfully: {len(df)} rows")
                
                # Preview and validation similar to file upload
                st.dataframe(df.head(10), use_container_width=True)
                
                result['data'] = df
                result['is_valid'] = True
                
            except Exception as e:
                st.error(f"❌ Failed to import from URL: {e}")
                result['is_valid'] = False
        
        return result
    
    def _validate_uploaded_data(self, df: pd.DataFrame, 
                               date_column: str, numbers_column: str) -> Dict[str, Any]:
        """Validate uploaded data."""
        errors = []
        
        try:
            # Check if columns exist
            if date_column not in df.columns:
                errors.append(f"Date column '{date_column}' not found")
            
            if numbers_column not in df.columns:
                errors.append(f"Numbers column '{numbers_column}' not found")
            
            if errors:
                return {'is_valid': False, 'errors': errors}
            
            # Validate date column
            try:
                pd.to_datetime(df[date_column])
            except Exception:
                errors.append("Date column contains invalid dates")
            
            # Validate numbers column (sample check)
            sample_size = min(10, len(df))
            for i in range(sample_size):
                numbers_str = str(df.iloc[i][numbers_column])
                try:
                    if ',' in numbers_str:
                        numbers = [int(x.strip()) for x in numbers_str.split(',')]
                    else:
                        # Try space-separated
                        numbers = [int(x.strip()) for x in numbers_str.split()]
                    
                    if len(numbers) < 3 or len(numbers) > 10:
                        errors.append(f"Invalid number count in row {i+1}")
                        break
                        
                except Exception:
                    errors.append(f"Invalid numbers format in row {i+1}")
                    break
            
            return {'is_valid': len(errors) == 0, 'errors': errors}
            
        except Exception as e:
            return {'is_valid': False, 'errors': [f"Validation error: {e}"]}
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class SettingsForm:
    """Form component for application settings."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize settings form."""
        self.config = config or {}
    
    def render(self, title: str = "Application Settings") -> Dict[str, Any]:
        """
        Render settings form.
        
        Args:
            title: Form title
            
        Returns:
            Settings configuration
        """
        try:
            st.subheader(title)
            
            settings = {}
            
            # General settings
            with st.expander("General Settings", expanded=True):
                settings.update(self._render_general_settings())
            
            # UI settings
            with st.expander("User Interface"):
                settings.update(self._render_ui_settings())
            
            # Performance settings
            with st.expander("Performance"):
                settings.update(self._render_performance_settings())
            
            # Data settings
            with st.expander("Data Management"):
                settings.update(self._render_data_settings())
            
            # Save settings
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                st.session_state.app_settings = settings
                st.success("✅ Settings saved successfully")
                settings['saved'] = True
            
            return settings
            
        except Exception as e:
            logger.error(f"❌ Failed to render settings form: {e}")
            st.error(f"Failed to display settings form: {e}")
            return {}
    
    def _render_general_settings(self) -> Dict[str, Any]:
        """Render general settings."""
        settings = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            settings['app_name'] = st.text_input(
                "Application Name:",
                value=st.session_state.get('app_name', 'Lottery Prediction System'),
                key="app_name_setting"
            )
            
            settings['default_game'] = st.selectbox(
                "Default Game:",
                options=['Lotto 6/49', 'Powerball', 'Mega Millions', 'EuroMillions'],
                index=0,
                key="default_game_setting"
            )
        
        with col2:
            settings['auto_save'] = st.checkbox(
                "Auto-save Predictions:",
                value=True,
                key="auto_save_setting"
            )
            
            settings['enable_notifications'] = st.checkbox(
                "Enable Notifications:",
                value=True,
                key="notifications_setting"
            )
        
        return settings
    
    def _render_ui_settings(self) -> Dict[str, Any]:
        """Render UI settings."""
        settings = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            settings['theme'] = st.selectbox(
                "Color Theme:",
                options=['Default', 'Dark', 'Light', 'Blue'],
                index=0,
                key="theme_setting"
            )
            
            settings['language'] = st.selectbox(
                "Language:",
                options=['English', 'Spanish', 'French', 'German'],
                index=0,
                key="language_setting"
            )
        
        with col2:
            settings['show_advanced_options'] = st.checkbox(
                "Show Advanced Options:",
                value=False,
                key="advanced_options_setting"
            )
            
            settings['compact_view'] = st.checkbox(
                "Compact View Mode:",
                value=False,
                key="compact_view_setting"
            )
        
        return settings
    
    def _render_performance_settings(self) -> Dict[str, Any]:
        """Render performance settings."""
        settings = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            settings['cache_size'] = st.number_input(
                "Cache Size (MB):",
                min_value=50,
                max_value=1000,
                value=200,
                key="cache_size_setting"
            )
            
            settings['max_concurrent_predictions'] = st.number_input(
                "Max Concurrent Predictions:",
                min_value=1,
                max_value=10,
                value=3,
                key="max_concurrent_setting"
            )
        
        with col2:
            settings['enable_gpu'] = st.checkbox(
                "Enable GPU Acceleration:",
                value=False,
                key="gpu_setting"
            )
            
            settings['parallel_processing'] = st.checkbox(
                "Enable Parallel Processing:",
                value=True,
                key="parallel_setting"
            )
        
        return settings
    
    def _render_data_settings(self) -> Dict[str, Any]:
        """Render data settings."""
        settings = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            settings['data_retention_days'] = st.number_input(
                "Data Retention (days):",
                min_value=30,
                max_value=3650,
                value=365,
                key="retention_setting"
            )
            
            settings['backup_frequency'] = st.selectbox(
                "Backup Frequency:",
                options=['Daily', 'Weekly', 'Monthly', 'Never'],
                index=1,
                key="backup_setting"
            )
        
        with col2:
            settings['auto_update_data'] = st.checkbox(
                "Auto-update Data:",
                value=True,
                key="auto_update_setting"
            )
            
            settings['compress_backups'] = st.checkbox(
                "Compress Backups:",
                value=True,
                key="compress_setting"
            )
        
        return settings
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


# Utility functions for forms
def validate_form_data(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate form data.
    
    Args:
        data: Form data to validate
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            errors.append(f"Field '{field}' is required")
    
    return len(errors) == 0, errors


def create_form_schema(form_type: str) -> Dict[str, Any]:
    """
    Create form schema for validation.
    
    Args:
        form_type: Type of form
        
    Returns:
        Form schema dictionary
    """
    schemas = {
        'model_config': {
            'required_fields': ['model_name', 'training_data_size', 'confidence_threshold'],
            'field_types': {
                'model_name': str,
                'training_data_size': int,
                'confidence_threshold': float
            }
        },
        'prediction': {
            'required_fields': ['num_predictions', 'strategy'],
            'field_types': {
                'num_predictions': int,
                'strategy': str
            }
        },
        'settings': {
            'required_fields': ['app_name', 'default_game'],
            'field_types': {
                'app_name': str,
                'default_game': str
            }
        }
    }
    
    return schemas.get(form_type, {})


# Backward compatibility classes - maintain original interfaces
class ModelConfigForm:
    """
    Legacy ModelConfigForm class for backward compatibility.
    
    This class maintains the original simple interface while delegating
    to the enhanced FormComponents class for actual functionality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize model configuration form (legacy constructor)."""
        self.config = config or {}
    
    def render(self, model_type: str = "mathematical",
               current_config: Dict[str, Any] = None,
               title: str = "Model Configuration") -> Dict[str, Any]:
        """Render model configuration form (legacy method)."""
        return FormComponents.render_model_configuration_form(
            model_type, current_config, title
        )
    
    def _get_default_config(self, model_type: str) -> Dict[str, Any]:
        """Get default configuration (legacy method)."""
        defaults = {
            'mathematical': {'priority': 'Medium', 'enabled': True},
            'lstm': {'hidden_layers': 3, 'neurons_per_layer': 128},
            'xgboost': {'n_estimators': 100, 'max_depth': 6}
        }
        return defaults.get(model_type, {})
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration (legacy method)."""
        validation_result = FormComponents._validate_model_config(config)
        return validation_result['valid']


class FormBuilder:
    """
    Legacy FormBuilder class for backward compatibility.
    
    This class provides helper methods that maintain the original interface.
    """
    
    @staticmethod
    def create_input_form(fields: List[Dict], title: str = "Input Form") -> Dict[str, Any]:
        """Create input form (legacy method)."""
        st.subheader(title)
        
        form_data = {}
        for field in fields:
            field_type = field.get('type', 'text')
            field_name = field['name']
            field_label = field.get('label', field_name.title())
            
            if field_type == 'text':
                form_data[field_name] = st.text_input(field_label)
            elif field_type == 'number':
                form_data[field_name] = st.number_input(
                    field_label,
                    min_value=field.get('min', 0),
                    max_value=field.get('max', 100)
                )
            elif field_type == 'select':
                form_data[field_name] = st.selectbox(field_label, field.get('options', []))
        
        return form_data
    
    @staticmethod
    def create_search_form(search_fields: List[str] = None) -> Dict[str, Any]:
        """Create search form (legacy method)."""
        return FormComponents.render_search_filter_form("Search", search_fields)
    
    @staticmethod
    def create_upload_form(accepted_types: List[str] = None) -> Dict[str, Any]:
        """Create upload form (legacy method)."""
        return FormComponents.render_data_upload_form("File Upload", accepted_types)


class FormValidator:
    """
    Legacy FormValidator class for backward compatibility.
    
    This class provides validation methods that maintain the original interface.
    """
    
    @staticmethod
    def validate_form_data(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate form data (legacy method)."""
        validation_result = {'valid': True, 'errors': []}
        
        # Check required fields
        required_fields = schema.get('required_fields', [])
        for field in required_fields:
            if field not in data or not data[field]:
                validation_result['errors'].append(f"Missing required field: {field}")
        
        # Check field types
        field_types = schema.get('field_types', {})
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                validation_result['errors'].append(f"Invalid type for field {field}")
        
        validation_result['valid'] = len(validation_result['errors']) == 0
        return validation_result
    
    @staticmethod
    def get_validation_schema(form_type: str) -> Dict[str, Any]:
        """Get validation schema (legacy method)."""
        # Define schemas inline for backward compatibility
        schemas = {
            'model': {
                'required_fields': ['model_name', 'priority'],
                'field_types': {
                    'model_name': str,
                    'priority': str,
                    'enabled': bool
                }
            },
            'prediction': {
                'required_fields': ['num_predictions', 'strategy'],
                'field_types': {
                    'num_predictions': int,
                    'strategy': str
                }
            },
            'settings': {
                'required_fields': ['app_name', 'default_game'],
                'field_types': {
                    'app_name': str,
                    'default_game': str
                }
            }
        }
        return schemas.get(form_type, {})


# Export classes for easy importing
__all__ = [
    'FormComponents', 
    'ModelConfigForm', 
    'FormBuilder', 
    'FormValidator'
]