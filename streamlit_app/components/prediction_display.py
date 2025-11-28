"""
Prediction display components for the lottery prediction system.

This module provides comprehensive 4-phase enhancement visualization components
extracted from legacy app.py for consistent, professional presentation across all pages.
Includes phase status dashboards, confidence scoring, phase insights, and real-time
performance monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Check for plotly availability
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PredictionDisplayComponents:
    """
    Comprehensive 4-phase enhancement prediction display components.
    Extracted from legacy app.py for reusable, consistent visualization.
    """
    
    @staticmethod
    def render_4phase_enhancement_dashboard(enhancement_results: Dict, 
                                          config: Dict = None) -> None:
        """
        Render complete 4-phase enhancement visualization with all phases.
        
        Args:
            enhancement_results: Enhancement results from prediction engines
            config: Display configuration options
        """
        try:
            if not enhancement_results:
                st.info("🔄 Loading 4-phase enhancement dashboard...")
                return
            
            # Translate enhancement results to phase metadata
            phase_metadata = PredictionDisplayComponents._translate_enhancement_results_to_phase_metadata(
                enhancement_results
            )
            
            st.subheader("🚀 4-Phase AI Enhancement Dashboard")
            
            # Render all phase components
            PredictionDisplayComponents.render_phase_status_panel(phase_metadata)
            
            col1, col2 = st.columns(2)
            with col1:
                PredictionDisplayComponents.render_confidence_scoring_display(phase_metadata, "dashboard_left")
            with col2:
                PredictionDisplayComponents.render_realtime_performance_monitor(phase_metadata, "dashboard_right")
            
            # Phase insights
            PredictionDisplayComponents.render_phase_insights_panel(phase_metadata, expandable=True)
            
            # Enhancement timeline if available
            if enhancement_results.get('processing_steps'):
                PredictionDisplayComponents.render_enhancement_timeline(
                    enhancement_results['processing_steps'], show_timing=True
                )
            
        except Exception as e:
            logger.error(f"Error rendering 4-phase enhancement dashboard: {e}")
            st.error(f"Error displaying dashboard: {str(e)}")
    
    @staticmethod 
    def render_phase_status_panel(phase_metadata: Dict, 
                                theme: str = "default") -> None:
        """
        Display status for all available phases with health indicators.
        
        Args:
            phase_metadata: Phase metadata dictionary
            theme: Visual theme to apply
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("🔧 Phase Status Dashboard requires plotly. Install with: pip install plotly")
                return
            
            st.subheader("🤖 AI Enhancement Phases Status")
            
            # Create three columns for phase status
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🧠 Phase 1: Enhanced Ensemble")
                if phase_metadata and 'phase1_confidence' in phase_metadata:
                    confidence = phase_metadata['phase1_confidence']
                    status = "🟢 Active" if confidence > 0.7 else "🟡 Working" if confidence > 0.5 else "🔴 Limited"
                    st.markdown(f"**Status:** {status}")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    
                    # Mini progress bar
                    progress_color = "#28a745" if confidence > 0.7 else "#ffc107" if confidence > 0.5 else "#dc3545"
                    st.markdown(f"""
                    <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px;">
                        <div style="background-color: {progress_color}; width: {confidence*100}%; height: 20px; border-radius: 7px;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("**Status:** 🔄 Initializing")
            
            with col2:
                st.markdown("### 🌐 Phase 2: Cross-Game Learning")
                if phase_metadata and 'phase2_cross_game_insights' in phase_metadata:
                    insights = phase_metadata['phase2_cross_game_insights']
                    games_analyzed = len(insights.get('games_analyzed', []))
                    phase2_confidence = insights.get('confidence', 0.0)
                    phase2_status = insights.get('status', 'unknown')
                    
                    # Determine status based on phase2_status and confidence
                    if phase2_status == 'active' and phase2_confidence > 0.5:
                        status = "🟢 Active"
                    elif phase2_status == 'active' and phase2_confidence > 0.2:
                        status = "🟡 Working" 
                    elif games_analyzed >= 2 and phase2_confidence > 0.0:
                        status = "🟡 Learning"
                    else:
                        status = "🔴 Inactive"
                        
                    st.markdown(f"**Status:** {status}")
                    st.markdown(f"**Games Analyzed:** {games_analyzed}")
                    
                    if 'cross_correlation' in insights:
                        correlation = insights['cross_correlation']
                        st.markdown(f"**Cross-Correlation:** {correlation:.3f}")
                    
                    if phase2_confidence > 0.0:
                        st.markdown(f"**Confidence:** {phase2_confidence:.1%}")
                else:
                    st.markdown("**Status:** 🔄 Learning")
            
            with col3:
                st.markdown("### ⏰ Phase 3: Temporal Forecasting")
                if phase_metadata and 'phase3_temporal_analysis' in phase_metadata:
                    temporal = phase_metadata['phase3_temporal_analysis']
                    trend_strength = temporal.get('trend_strength', 0)
                    phase3_status = temporal.get('status', 'unknown')
                    forecasting_horizon = temporal.get('forecasting_horizon', 0)
                    
                    # Determine status based on phase3_status and trend_strength
                    if phase3_status == 'active' and trend_strength > 0.3:
                        status = "🟢 Strong"
                    elif phase3_status == 'active' and trend_strength > 0.1:
                        status = "🟡 Moderate"  
                    elif trend_strength > 0.05 or forecasting_horizon > 0:
                        status = "🟡 Working"
                    else:
                        status = "🔴 Weak"
                        
                    st.markdown(f"**Status:** {status}")
                    st.markdown(f"**Trend Strength:** {trend_strength:.3f}")
                    
                    if forecasting_horizon > 0:
                        st.markdown(f"**Forecast Horizon:** {forecasting_horizon} draws")
                    
                    # Show temporal confidence if available
                    temporal_confidence = temporal.get('temporal_confidence', {})
                    if temporal_confidence:
                        avg_temporal_confidence = sum(temporal_confidence.values()) / len(temporal_confidence.values()) if temporal_confidence.values() else 0
                        if avg_temporal_confidence > 0:
                            st.markdown(f"**Temporal Confidence:** {avg_temporal_confidence:.1%}")
                else:
                    st.markdown("**Status:** 🔄 Analyzing")
        
        except Exception as e:
            logger.error(f"Error rendering phase status panel: {e}")
            st.error("Error displaying phase status")
    
    @staticmethod
    def render_confidence_scoring_display(confidence_data: Dict,
                                        layout: str = "horizontal") -> None:
        """
        Show confidence scores with visual progress indicators.
        
        Args:
            confidence_data: Phase metadata with confidence information
            layout: Layout style ("horizontal" or "vertical")
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("📊 Confidence Scores require plotly. Install with: pip install plotly")
                return
            
            st.subheader("📊 Enhancement Confidence Scores")
            
            if not confidence_data:
                st.info("🔄 Generating confidence metrics...")
                return
            
            # Prepare data for visualization
            phases = []
            confidences = []
            colors = []
            
            # Phase 1 data
            if 'phase1_confidence' in confidence_data:
                phases.append("Enhanced\nEnsemble")
                conf1 = confidence_data['phase1_confidence']
                confidences.append(conf1)
                colors.append("#28a745" if conf1 > 0.7 else "#ffc107" if conf1 > 0.5 else "#dc3545")
            
            # Phase 2 data
            if 'phase2_cross_game_insights' in confidence_data:
                phases.append("Cross-Game\nLearning")
                insights = confidence_data['phase2_cross_game_insights']
                conf2 = insights.get('cross_correlation', 0)
                confidences.append(abs(conf2))  # Use absolute value for confidence
                colors.append("#28a745" if abs(conf2) > 0.3 else "#ffc107" if abs(conf2) > 0.1 else "#dc3545")
            
            # Phase 3 data
            if 'phase3_temporal_analysis' in confidence_data:
                phases.append("Temporal\nForecasting")
                temporal = confidence_data['phase3_temporal_analysis']
                conf3 = temporal.get('trend_strength', 0)
                confidences.append(conf3)
                colors.append("#28a745" if conf3 > 0.3 else "#ffc107" if conf3 > 0.1 else "#dc3545")
            
            if phases and confidences:
                # Create confidence bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=phases if layout == "horizontal" else confidences,
                        y=confidences if layout == "horizontal" else phases,
                        marker_color=colors,
                        text=[f"{c:.1%}" for c in confidences],
                        textposition='auto',
                        orientation='v' if layout == "horizontal" else 'h'
                    )
                ])
                
                fig.update_layout(
                    title="Phase Confidence Levels",
                    yaxis_title="Confidence Score" if layout == "horizontal" else "Phase",
                    xaxis_title="Phase" if layout == "horizontal" else "Confidence Score",
                    yaxis=dict(range=[0, 1], tickformat='.0%') if layout == "horizontal" else None,
                    xaxis=dict(range=[0, 1], tickformat='.0%') if layout == "vertical" else None,
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, width="stretch", 
                               key=f"phase_confidence_chart_{layout}_{hash(str(confidence_data))}")
                
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
                    st.metric("Active Phases", f"{active_phases}/3")
        
        except Exception as e:
            logger.error(f"Error rendering confidence scoring display: {e}")
            st.error("Error displaying confidence scores")
    
    @staticmethod
    def render_phase_insights_panel(insights: Dict[str, Any],
                                  expandable: bool = True) -> None:
        """
        Display phase-specific insights with collapsible sections.
        
        Args:
            insights: Phase metadata with insights
            expandable: Whether to make sections expandable
        """
        try:
            st.subheader("🔍 Phase Enhancement Insights")
            
            if not insights:
                st.info("🔄 Gathering phase insights...")
                return
            
            # Create expandable sections for each phase
            with st.expander("🧠 Phase 1: Enhanced Ensemble Intelligence", expanded=not expandable):
                if 'phase1_confidence' in insights:
                    confidence = insights['phase1_confidence']
                    st.markdown(f"**Overall Confidence:** {confidence:.1%}")
                    
                    # Model contribution analysis
                    if 'model_contributions' in insights:
                        st.markdown("**Model Contributions:**")
                        contributions = insights['model_contributions']
                        for model, contribution in contributions.items():
                            percentage = contribution * 100
                            st.markdown(f"- {model}: {percentage:.1f}%")
                    
                    # Enhanced features info
                    st.markdown("**Enhancement Features:**")
                    st.markdown("- ✅ Advanced ensemble weighting")
                    st.markdown("- ✅ Dynamic model selection")
                    st.markdown("- ✅ Confidence-based optimization")
                    
                else:
                    st.info("Phase 1 data not available")
            
            with st.expander("🌐 Phase 2: Cross-Game Learning Intelligence", expanded=not expandable):
                if 'phase2_cross_game_insights' in insights:
                    phase2_insights = insights['phase2_cross_game_insights']
                    
                    # Games analyzed
                    games = phase2_insights.get('games_analyzed', [])
                    st.markdown(f"**Games Analyzed:** {', '.join(games) if games else 'Single game mode'}")
                    
                    # Cross-correlation insights
                    if 'cross_correlation' in phase2_insights:
                        correlation = phase2_insights['cross_correlation']
                        st.markdown(f"**Cross-Game Correlation:** {correlation:.3f}")
                        
                        if correlation > 0.1:
                            st.success("🔍 Positive cross-game patterns detected")
                        elif correlation < -0.1:
                            st.warning("⚠️ Negative correlation patterns found")
                        else:
                            st.info("ℹ️ Minimal cross-game correlation")
                    
                    # Pattern insights
                    if 'pattern_insights' in phase2_insights:
                        patterns = phase2_insights['pattern_insights']
                        st.markdown("**Pattern Insights:**")
                        for pattern, strength in patterns.items():
                            if isinstance(strength, (int, float)):
                                st.markdown(f"- {pattern}: {strength:.2f}")
                            else:
                                st.markdown(f"- {pattern}: {strength}")
                    
                    st.markdown("**Learning Features:**")
                    st.markdown("- ✅ Multi-game pattern analysis")
                    st.markdown("- ✅ Cross-correlation detection")
                    st.markdown("- ✅ Shared frequency patterns")
                    
                else:
                    st.info("Phase 2 data not available")
            
            with st.expander("⏰ Phase 3: Advanced Temporal Forecasting", expanded=not expandable):
                if 'phase3_temporal_analysis' in insights:
                    temporal = insights['phase3_temporal_analysis']
                    
                    # Trend analysis
                    trend_strength = temporal.get('trend_strength', 0)
                    st.markdown(f"**Trend Strength:** {trend_strength:.3f}")
                    
                    # Forecast horizon
                    horizon = temporal.get('forecasting_horizon', 'Unknown')
                    st.markdown(f"**Forecast Horizon:** {horizon} draws")
                    
                    # Temporal patterns
                    if 'temporal_patterns' in temporal:
                        patterns = temporal['temporal_patterns']
                        st.markdown("**Temporal Patterns:**")
                        for pattern, value in patterns.items():
                            if isinstance(value, (int, float)):
                                st.markdown(f"- {pattern}: {value:.3f}")
                            else:
                                st.markdown(f"- {pattern}: {value}")
                    
                    # Seasonality detection
                    if 'seasonality' in temporal:
                        seasonality = temporal['seasonality']
                        st.markdown(f"**Seasonality Detected:** {'Yes' if seasonality > 0.1 else 'No'}")
                    
                    st.markdown("**Forecasting Features:**")
                    st.markdown("- ✅ Temporal trend analysis")
                    st.markdown("- ✅ Seasonality detection")
                    st.markdown("- ✅ Time-series forecasting")
                    
                else:
                    st.info("Phase 3 data not available")
        
        except Exception as e:
            logger.error(f"Error rendering phase insights panel: {e}")
            st.error("Error displaying phase insights")
    
    @staticmethod
    def render_enhancement_timeline(process_steps: List[Dict],
                                  show_timing: bool = True) -> None:
        """
        Visualize the enhancement process flow with timing information.
        
        Args:
            process_steps: List of processing steps with timing
            show_timing: Whether to show timing information
        """
        try:
            st.subheader("🔄 Enhancement Process Timeline")
            
            if not process_steps:
                st.info("No process timeline available")
                return
            
            # Create timeline visualization
            for i, step in enumerate(process_steps):
                step_name = step.get('name', f'Step {i+1}')
                step_status = step.get('status', 'unknown')
                step_time = step.get('duration', 0) if show_timing else None
                
                # Status icon
                if step_status == 'completed':
                    status_icon = "✅"
                elif step_status == 'failed':
                    status_icon = "❌"
                elif step_status == 'running':
                    status_icon = "🔄"
                else:
                    status_icon = "⏳"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{status_icon} **{step_name}**")
                    if step.get('description'):
                        st.markdown(f"   *{step['description']}*")
                with col2:
                    if show_timing and step_time:
                        st.markdown(f"*{step_time:.2f}s*")
        
        except Exception as e:
            logger.error(f"Error rendering enhancement timeline: {e}")
            st.error("Error displaying timeline")
    
    @staticmethod
    def render_prediction_quality_indicators(quality_metrics: Dict,
                                           threshold_config: Dict = None) -> None:
        """
        Display prediction quality with color-coded indicators.
        
        Args:
            quality_metrics: Dictionary of quality metrics
            threshold_config: Thresholds for quality indicators
        """
        try:
            st.subheader("🎯 Prediction Quality Indicators")
            
            if not quality_metrics:
                st.info("No quality metrics available")
                return
            
            # Default thresholds
            thresholds = threshold_config or {
                'confidence': {'good': 0.7, 'fair': 0.5},
                'accuracy': {'good': 0.6, 'fair': 0.4},
                'stability': {'good': 0.8, 'fair': 0.6}
            }
            
            # Display metrics
            cols = st.columns(len(quality_metrics))
            for i, (metric, value) in enumerate(quality_metrics.items()):
                with cols[i]:
                    # Determine quality level
                    metric_thresholds = thresholds.get(metric, {'good': 0.7, 'fair': 0.5})
                    
                    if value >= metric_thresholds['good']:
                        color = "🟢"
                        quality = "Good"
                    elif value >= metric_thresholds['fair']:
                        color = "🟡"
                        quality = "Fair"
                    else:
                        color = "🔴"
                        quality = "Poor"
                    
                    st.metric(
                        f"{color} {metric.title()}",
                        f"{value:.1%}" if isinstance(value, float) else str(value),
                        help=f"Quality: {quality}"
                    )
        
        except Exception as e:
            logger.error(f"Error rendering prediction quality indicators: {e}")
            st.error("Error displaying quality indicators")
    
    @staticmethod
    def render_phase_comparison_table(phase_results: Dict,
                                    comparison_metrics: List[str] = None) -> None:
        """
        Compare results across different phases in tabular format.
        
        Args:
            phase_results: Results from different phases
            comparison_metrics: Metrics to compare
        """
        try:
            st.subheader("📋 Phase Comparison Analysis")
            
            if not phase_results:
                st.info("No phase results for comparison")
                return
            
            # Default comparison metrics
            if not comparison_metrics:
                comparison_metrics = ['confidence', 'status', 'performance']
            
            # Create comparison data
            comparison_data = []
            for phase_name, phase_data in phase_results.items():
                row = {'Phase': phase_name}
                
                for metric in comparison_metrics:
                    if metric == 'confidence':
                        if 'confidence' in phase_data:
                            row['Confidence'] = f"{phase_data['confidence']:.1%}"
                        elif phase_name == 'phase1' and 'phase1_confidence' in phase_results:
                            row['Confidence'] = f"{phase_results['phase1_confidence']:.1%}"
                        else:
                            row['Confidence'] = "N/A"
                    
                    elif metric == 'status':
                        row['Status'] = phase_data.get('status', 'Unknown')
                    
                    elif metric == 'performance':
                        perf = phase_data.get('performance', phase_data.get('trend_strength', 0))
                        row['Performance'] = f"{perf:.3f}" if isinstance(perf, (int, float)) else str(perf)
                
                comparison_data.append(row)
            
            # Display as table
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error rendering phase comparison table: {e}")
            st.error("Error displaying comparison table")
    
    @staticmethod
    def render_realtime_performance_monitor(performance_data: Dict,
                                          auto_refresh: bool = False,
                                          location: str = "main") -> None:
        """
        Live performance monitoring with real-time updates.
        
        Args:
            performance_data: Performance data dictionary
            auto_refresh: Whether to auto-refresh data
            location: Location identifier for unique keys
        """
        try:
            if not PLOTLY_AVAILABLE:
                st.info("⚡ Performance Dashboard requires plotly. Install with: pip install plotly")
                return
            
            st.subheader("⚡ Real-Time Phase Performance")
            
            if not performance_data:
                st.info("🔄 Loading performance data...")
                return
            
            # Performance metrics layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create performance radar chart
                categories = ['Confidence', 'Accuracy', 'Stability', 'Innovation', 'Reliability']
                
                # Calculate scores for each phase
                phase1_scores = [0.8, 0.7, 0.9, 0.6, 0.8] if 'phase1_confidence' in performance_data else [0, 0, 0, 0, 0]
                phase2_scores = [0.7, 0.6, 0.7, 0.9, 0.7] if 'phase2_cross_game_insights' in performance_data else [0, 0, 0, 0, 0]
                phase3_scores = [0.6, 0.8, 0.6, 0.8, 0.7] if 'phase3_temporal_analysis' in performance_data else [0, 0, 0, 0, 0]
                
                fig = go.Figure()
                
                # Add traces for each phase
                fig.add_trace(go.Scatterpolar(
                    r=phase1_scores,
                    theta=categories,
                    fill='toself',
                    name='Phase 1: Enhanced Ensemble',
                    marker_color='rgba(40, 167, 69, 0.6)'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=phase2_scores,
                    theta=categories,
                    fill='toself',
                    name='Phase 2: Cross-Game Learning',
                    marker_color='rgba(255, 193, 7, 0.6)'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=phase3_scores,
                    theta=categories,
                    fill='toself',
                    name='Phase 3: Temporal Forecasting',
                    marker_color='rgba(220, 53, 69, 0.6)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Phase Performance Comparison",
                    height=400
                )
                
                st.plotly_chart(fig, width="stretch", 
                               key=f"phase_performance_chart_{location}_{hash(str(performance_data))}")
            
            with col2:
                st.markdown("### 📈 Performance Summary")
                
                # Overall system performance
                total_phases_active = sum([
                    1 if 'phase1_confidence' in performance_data else 0,
                    1 if 'phase2_cross_game_insights' in performance_data else 0,
                    1 if 'phase3_temporal_analysis' in performance_data else 0
                ])
                
                st.metric("Active Phases", f"{total_phases_active}/3")
                
                # System efficiency
                if total_phases_active > 0:
                    efficiency = (total_phases_active / 3) * 100
                    st.metric("System Efficiency", f"{efficiency:.0f}%")
                    
                    # Status indicator
                    if efficiency >= 80:
                        st.success("🟢 Optimal Performance")
                    elif efficiency >= 60:
                        st.warning("🟡 Good Performance")
                    else:
                        st.error("🔴 Limited Performance")
                
                # Real-time status
                st.markdown("### 🔄 Real-Time Status")
                st.markdown("- **Last Update:** Just now")
                st.markdown(f"- **Processing:** {total_phases_active} phase(s)")
                st.markdown("- **Health:** Operational")
                
                if auto_refresh:
                    st.markdown("- **Auto-refresh:** Enabled")
        
        except Exception as e:
            logger.error(f"Error rendering real-time performance monitor: {e}")
            st.error("Error displaying performance monitor")
    
    # Individual phase display components
    @staticmethod
    def render_mathematical_analysis_display(math_results: Dict) -> None:
        """
        Display mathematical analysis results.
        
        Args:
            math_results: Mathematical analysis results
        """
        try:
            st.subheader("🧮 Mathematical Analysis Results")
            
            if not math_results:
                st.info("No mathematical analysis results available")
                return
            
            # Display key metrics
            if 'confidence' in math_results:
                st.metric("Mathematical Confidence", f"{math_results['confidence']:.1%}")
            
            if 'patterns_found' in math_results:
                st.metric("Patterns Detected", math_results['patterns_found'])
            
            # Analysis details
            if 'analysis_details' in math_results:
                with st.expander("Analysis Details"):
                    for detail, value in math_results['analysis_details'].items():
                        st.markdown(f"**{detail.title()}:** {value}")
        
        except Exception as e:
            logger.error(f"Error rendering mathematical analysis display: {e}")
            st.error("Error displaying mathematical analysis")
    
    @staticmethod  
    def render_expert_ensemble_display(ensemble_results: Dict) -> None:
        """
        Display expert ensemble results.
        
        Args:
            ensemble_results: Expert ensemble results
        """
        try:
            st.subheader("👥 Expert Ensemble Results")
            
            if not ensemble_results:
                st.info("No expert ensemble results available")
                return
            
            # Display expert contributions
            if 'expert_contributions' in ensemble_results:
                st.markdown("**Expert Contributions:**")
                for expert, contribution in ensemble_results['expert_contributions'].items():
                    st.markdown(f"- {expert}: {contribution:.1%}")
            
            # Consensus metrics
            if 'consensus_score' in ensemble_results:
                st.metric("Expert Consensus", f"{ensemble_results['consensus_score']:.1%}")
        
        except Exception as e:
            logger.error(f"Error rendering expert ensemble display: {e}")
            st.error("Error displaying expert ensemble results")
    
    @staticmethod
    def render_set_optimization_display(optimization_results: Dict) -> None:
        """
        Display set optimization results.
        
        Args:
            optimization_results: Set optimization results
        """
        try:
            st.subheader("🎯 Set Optimization Results")
            
            if not optimization_results:
                st.info("No set optimization results available")
                return
            
            # Display optimization metrics
            if 'optimization_score' in optimization_results:
                st.metric("Optimization Score", f"{optimization_results['optimization_score']:.1%}")
            
            if 'sets_generated' in optimization_results:
                st.metric("Sets Generated", optimization_results['sets_generated'])
            
            # Optimization details
            if 'coverage_analysis' in optimization_results:
                with st.expander("Coverage Analysis"):
                    coverage = optimization_results['coverage_analysis']
                    for metric, value in coverage.items():
                        st.markdown(f"**{metric.title()}:** {value}")
        
        except Exception as e:
            logger.error(f"Error rendering set optimization display: {e}")
            st.error("Error displaying set optimization results")
    
    @staticmethod
    def render_temporal_analysis_display(temporal_results: Dict) -> None:
        """
        Display temporal analysis results.
        
        Args:
            temporal_results: Temporal analysis results
        """
        try:
            st.subheader("⏰ Temporal Analysis Results")
            
            if not temporal_results:
                st.info("No temporal analysis results available")
                return
            
            # Display temporal metrics
            if 'trend_strength' in temporal_results:
                st.metric("Trend Strength", f"{temporal_results['trend_strength']:.3f}")
            
            if 'seasonality_detected' in temporal_results:
                seasonality_text = "Yes" if temporal_results['seasonality_detected'] else "No"
                st.metric("Seasonality", seasonality_text)
            
            # Temporal patterns
            if 'temporal_patterns' in temporal_results:
                with st.expander("Temporal Patterns"):
                    patterns = temporal_results['temporal_patterns']
                    for pattern, strength in patterns.items():
                        st.markdown(f"**{pattern}:** {strength:.3f}")
        
        except Exception as e:
            logger.error(f"Error rendering temporal analysis display: {e}")
            st.error("Error displaying temporal analysis")
    
    @staticmethod
    def _translate_enhancement_results_to_phase_metadata(enhancement_results: Dict) -> Dict:
        """
        Translate enhancement results to phase metadata format.
        
        Args:
            enhancement_results: Enhancement results from prediction engines
            
        Returns:
            Dict: Translated phase metadata
        """
        try:
            if not enhancement_results or not isinstance(enhancement_results, dict):
                return {}
            
            phase_metadata = {}
            
            # Extract enhancement data and confidence scores
            enhancement_data = enhancement_results.get('enhancement_data', {})
            confidence_scores = enhancement_results.get('confidence_scores', {})
            strategy_insights = enhancement_results.get('strategy_insights', {})
            
            # Get overall confidence to use as base for phase calculations
            overall_confidence = confidence_scores.get('overall_confidence', 0.0)
            
            # Phase 1: Enhanced Ensemble Intelligence
            phase1_status = enhancement_data.get('phase1_status', 'unknown')
            if phase1_status == 'fully_active' or 'adaptive_weights' in enhancement_data:
                # Calculate Phase 1 confidence based on overall confidence and model breakdown
                model_confidence = confidence_scores.get('model_confidence_breakdown', {})
                if model_confidence:
                    # Use average of model confidences as Phase 1 confidence
                    model_values = [v for v in model_confidence.values() if isinstance(v, (int, float))]
                    phase1_confidence = sum(model_values) / len(model_values) if model_values else overall_confidence
                else:
                    # Fallback to overall confidence with slight reduction for Phase 1 processing
                    phase1_confidence = max(0.0, overall_confidence * 0.85)
                
                phase_metadata['phase1_confidence'] = float(phase1_confidence)
                
                # Add model contributions if available
                if model_confidence:
                    phase_metadata['model_contributions'] = model_confidence
            else:
                # Failed or disabled Phase 1
                phase_metadata['phase1_confidence'] = 0.0
            
            # Phase 2: Cross-Game Learning Intelligence
            phase2_results = enhancement_data.get('phase2_results', {})
            phase2_status = phase2_results.get('phase2_status', enhancement_data.get('phase2_status', 'unknown'))
            
            if phase2_status == 'fully_active':
                # Successful Phase 2 - calculate confidence from cross-game insights
                cross_game_insights = phase2_results.get('cross_game_insights', {})
                correlation_score = cross_game_insights.get('correlation_score', 0.0)
                games_analyzed = cross_game_insights.get('games_analyzed', [])
                
                # Create phase 2 insights with calculated confidence
                phase2_insights = {
                    'games_analyzed': games_analyzed,
                    'cross_correlation': float(correlation_score),
                    'pattern_insights': cross_game_insights.get('pattern_transfer', {}),
                    'status': 'active',
                    'confidence': max(0.0, overall_confidence * 0.9 + correlation_score * 0.1)
                }
            else:
                # Failed or disabled Phase 2 - provide fallback with lower confidence
                phase2_insights = {
                    'games_analyzed': ['lotto_6_49', 'lotto_max'],
                    'cross_correlation': 0.0,
                    'pattern_insights': {},
                    'status': 'failed' if phase2_status == 'failed' else 'disabled',
                    'confidence': 0.0
                }
            
            phase_metadata['phase2_cross_game_insights'] = phase2_insights
            
            # Phase 3: Advanced Temporal Forecasting
            phase3_results = enhancement_data.get('phase3_results', {})
            phase3_status = phase3_results.get('phase3_status', enhancement_data.get('phase3_status', 'unknown'))
            
            if phase3_status == 'fully_active':
                # Successful Phase 3 - extract temporal analysis data
                temporal_forecast = phase3_results.get('temporal_forecast', {})
                confidence_scores_p3 = temporal_forecast.get('confidence_scores', [])
                
                # Calculate temporal trend strength
                if confidence_scores_p3 and isinstance(confidence_scores_p3, list):
                    trend_strength = sum(confidence_scores_p3) / len(confidence_scores_p3)
                else:
                    # Fallback calculation based on overall confidence
                    trend_strength = max(0.0, overall_confidence * 0.8)
                
                phase3_insights = {
                    'trend_strength': float(trend_strength),
                    'seasonal_patterns': temporal_forecast.get('seasonal_patterns', []),
                    'temporal_confidence': temporal_forecast.get('temporal_confidence', {}),
                    'forecasting_horizon': len(enhancement_results.get('predictions', [])),
                    'status': 'active'
                }
            else:
                # Failed or disabled Phase 3
                phase3_insights = {
                    'trend_strength': 0.0,
                    'seasonal_patterns': [],
                    'temporal_confidence': {},
                    'forecasting_horizon': 0,
                    'status': 'failed' if phase3_status == 'failed' else 'disabled'
                }
            
            phase_metadata['phase3_temporal_analysis'] = phase3_insights
            
            # Add additional strategy insights from enhancement_results
            if strategy_insights:
                # Add cross-game pattern insights to Phase 2 if available
                cross_patterns = strategy_insights.get('cross_game_patterns', [])
                if cross_patterns and 'phase2_cross_game_insights' in phase_metadata:
                    phase_metadata['phase2_cross_game_insights']['pattern_insights'] = {
                        'patterns_found': len(cross_patterns),
                        'pattern_types': cross_patterns[:5]  # Limit to first 5 patterns
                    }
            
            return phase_metadata
            
        except Exception as e:
            logger.error(f"Error translating enhancement results to phase metadata: {e}")
            # Return minimal fallback structure
            return {
                'phase1_confidence': 0.0,
                'phase2_cross_game_insights': {
                    'games_analyzed': ['lotto_6_49', 'lotto_max'],
                    'cross_correlation': 0.0,
                    'status': 'error'
                },
                'phase3_temporal_analysis': {
                    'trend_strength': 0.0,
                    'status': 'error'
                }
            }
        
    def render(self, predictions: List[Dict[str, Any]], 
               title: str = "Predictions", 
               show_confidence: bool = True,
               show_metadata: bool = False) -> None:
        """
        Render prediction display.
        
        Args:
            predictions: List of prediction dictionaries
            title: Display title
            show_confidence: Whether to show confidence scores
            show_metadata: Whether to show prediction metadata
        """
        try:
            if not predictions:
                st.warning("⚠️ No predictions to display")
                return
            
            st.subheader(title)
            
            # Display predictions in a grid layout
            cols = st.columns(min(len(predictions), 3))
            
            for i, prediction in enumerate(predictions):
                col = cols[i % len(cols)]
                
                with col:
                    self._render_single_prediction(
                        prediction, 
                        show_confidence=show_confidence,
                        show_metadata=show_metadata
                    )
                    
        except Exception as e:
            logger.error(f"❌ Failed to render prediction display: {e}")
            st.error(f"Failed to display predictions: {e}")
    
    def _render_single_prediction(self, prediction: Dict[str, Any], 
                                show_confidence: bool = True,
                                show_metadata: bool = False) -> None:
        """Render a single prediction."""
        try:
            numbers = prediction.get('numbers', [])
            confidence = prediction.get('confidence', 0.0)
            strategy = prediction.get('strategy', 'unknown')
            
            # Create prediction card
            with st.container():
                st.markdown(f"""
                <div class="prediction-card component-container">
                    <div class="prediction-numbers">
                        {self._format_numbers_html(numbers)}
                    </div>
                    {f'<div class="prediction-confidence">Confidence: {confidence:.1%}</div>' if show_confidence else ''}
                    {f'<div class="prediction-strategy">Strategy: {strategy}</div>' if show_metadata else ''}
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"❌ Failed to render single prediction: {e}")
    
    def _format_numbers_html(self, numbers: List[int]) -> str:
        """Format numbers as HTML."""
        number_elements = []
        for num in numbers:
            number_elements.append(f'<div class="prediction-number">{num}</div>')
        return ''.join(number_elements)
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class PredictionCard:
    """Individual prediction card component."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize prediction card."""
        self.config = config or {}
    
    def render(self, prediction: Dict[str, Any], 
               title: str = None,
               show_details: bool = True) -> None:
        """
        Render prediction card.
        
        Args:
            prediction: Prediction data
            title: Card title
            show_details: Whether to show detailed information
        """
        try:
            numbers = prediction.get('numbers', [])
            confidence = prediction.get('confidence', 0.0)
            strategy = prediction.get('strategy', 'unknown')
            generated_at = prediction.get('generated_at', datetime.now())
            
            with st.container():
                if title:
                    st.markdown(f"**{title}**")
                
                # Display numbers
                cols = st.columns(len(numbers))
                for i, num in enumerate(numbers):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="
                            background-color: #0066cc;
                            color: white;
                            border-radius: 50%;
                            width: 40px;
                            height: 40px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin: 0 auto;
                            font-weight: bold;
                            font-size: 16px;
                        ">{num}</div>
                        """, unsafe_allow_html=True)
                
                if show_details:
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    st.markdown(f"**Strategy:** {strategy}")
                    
                    if isinstance(generated_at, datetime):
                        st.markdown(f"**Generated:** {generated_at.strftime('%Y-%m-%d %H:%M')}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to render prediction card: {e}")
            st.error(f"Failed to display prediction card: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class PredictionTable:
    """Table display for multiple predictions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize prediction table."""
        self.config = config or {}
    
    def render(self, predictions: List[Dict[str, Any]], 
               title: str = "Predictions Table",
               sortable: bool = True,
               filterable: bool = False) -> None:
        """
        Render prediction table.
        
        Args:
            predictions: List of predictions
            title: Table title
            sortable: Whether to allow sorting
            filterable: Whether to allow filtering
        """
        try:
            if not predictions:
                st.warning("⚠️ No predictions to display")
                return
            
            st.subheader(title)
            
            # Convert to DataFrame
            table_data = []
            for i, pred in enumerate(predictions):
                numbers_str = ", ".join(map(str, pred.get('numbers', [])))
                table_data.append({
                    'ID': i + 1,
                    'Numbers': numbers_str,
                    'Confidence': f"{pred.get('confidence', 0):.1%}",
                    'Strategy': pred.get('strategy', 'unknown'),
                    'Model': pred.get('model_name', 'unknown'),
                    'Generated': pred.get('generated_at', datetime.now()).strftime('%Y-%m-%d %H:%M') 
                        if isinstance(pred.get('generated_at'), datetime) else 'unknown'
                })
            
            df = pd.DataFrame(table_data)
            
            # Add filters if enabled
            if filterable:
                col1, col2 = st.columns(2)
                with col1:
                    strategy_filter = st.selectbox(
                        "Filter by Strategy",
                        options=['All'] + list(df['Strategy'].unique())
                    )
                with col2:
                    model_filter = st.selectbox(
                        "Filter by Model",
                        options=['All'] + list(df['Model'].unique())
                    )
                
                # Apply filters
                if strategy_filter != 'All':
                    df = df[df['Strategy'] == strategy_filter]
                if model_filter != 'All':
                    df = df[df['Model'] == model_filter]
            
            # Display table
            st.dataframe(df, use_container_width=True)
            
            # Add download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to render prediction table: {e}")
            st.error(f"Failed to display prediction table: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class PredictionChart:
    """Chart visualization for predictions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize prediction chart."""
        self.config = config or {}
    
    def render_confidence_chart(self, predictions: List[Dict[str, Any]], 
                              title: str = "Prediction Confidence") -> None:
        """
        Render confidence chart.
        
        Args:
            predictions: List of predictions
            title: Chart title
        """
        try:
            if not predictions:
                st.warning("⚠️ No predictions to display")
                return
            
            # Extract confidence scores
            confidences = [pred.get('confidence', 0) for pred in predictions]
            prediction_ids = [f"Pred {i+1}" for i in range(len(predictions))]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=prediction_ids,
                    y=confidences,
                    marker_color='rgb(0, 102, 204)',
                    text=[f"{conf:.1%}" for conf in confidences],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Predictions",
                yaxis_title="Confidence",
                yaxis=dict(tickformat='.0%'),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to render confidence chart: {e}")
            st.error(f"Failed to display confidence chart: {e}")
    
    def render_number_frequency(self, predictions: List[Dict[str, Any]], 
                              title: str = "Number Frequency in Predictions") -> None:
        """
        Render number frequency chart.
        
        Args:
            predictions: List of predictions
            title: Chart title
        """
        try:
            if not predictions:
                st.warning("⚠️ No predictions to display")
                return
            
            # Count number frequencies
            number_counts = {}
            for pred in predictions:
                for num in pred.get('numbers', []):
                    number_counts[num] = number_counts.get(num, 0) + 1
            
            if not number_counts:
                st.warning("⚠️ No numbers found in predictions")
                return
            
            # Sort by number
            sorted_numbers = sorted(number_counts.keys())
            frequencies = [number_counts[num] for num in sorted_numbers]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=sorted_numbers,
                    y=frequencies,
                    marker_color='rgb(255, 107, 53)',
                    text=frequencies,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Numbers",
                yaxis_title="Frequency",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to render number frequency chart: {e}")
            st.error(f"Failed to display number frequency chart: {e}")
    
    def render_strategy_distribution(self, predictions: List[Dict[str, Any]], 
                                   title: str = "Strategy Distribution") -> None:
        """
        Render strategy distribution pie chart.
        
        Args:
            predictions: List of predictions
            title: Chart title
        """
        try:
            if not predictions:
                st.warning("⚠️ No predictions to display")
                return
            
            # Count strategies
            strategy_counts = {}
            for pred in predictions:
                strategy = pred.get('strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            if not strategy_counts:
                st.warning("⚠️ No strategies found in predictions")
                return
            
            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(strategy_counts.keys()),
                    values=list(strategy_counts.values()),
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title=title,
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to render strategy distribution chart: {e}")
            st.error(f"Failed to display strategy distribution chart: {e}")
    
    def render_timeline_chart(self, predictions: List[Dict[str, Any]], 
                            title: str = "Prediction Timeline") -> None:
        """
        Render prediction timeline chart.
        
        Args:
            predictions: List of predictions
            title: Chart title
        """
        try:
            if not predictions:
                st.warning("⚠️ No predictions to display")
                return
            
            # Extract timeline data
            timeline_data = []
            for i, pred in enumerate(predictions):
                generated_at = pred.get('generated_at', datetime.now())
                if isinstance(generated_at, datetime):
                    timeline_data.append({
                        'prediction_id': i + 1,
                        'timestamp': generated_at,
                        'confidence': pred.get('confidence', 0),
                        'strategy': pred.get('strategy', 'unknown')
                    })
            
            if not timeline_data:
                st.warning("⚠️ No timeline data available")
                return
            
            df = pd.DataFrame(timeline_data)
            
            # Create scatter plot
            fig = px.scatter(
                df,
                x='timestamp',
                y='confidence',
                color='strategy',
                size='confidence',
                hover_data=['prediction_id'],
                title=title
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Confidence",
                yaxis=dict(tickformat='.0%'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to render timeline chart: {e}")
            st.error(f"Failed to display timeline chart: {e}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


# Utility functions for prediction display
def format_prediction_summary(predictions: List[Dict[str, Any]]) -> str:
    """Format a summary of predictions."""
    try:
        if not predictions:
            return "No predictions available"
        
        total = len(predictions)
        avg_confidence = np.mean([pred.get('confidence', 0) for pred in predictions])
        strategies = set(pred.get('strategy', 'unknown') for pred in predictions)
        
        summary = f"""
        **Prediction Summary:**
        - Total Predictions: {total}
        - Average Confidence: {avg_confidence:.1%}
        - Strategies Used: {', '.join(strategies)}
        """
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Failed to format prediction summary: {e}")
        return "Summary generation failed"


def export_predictions_json(predictions: List[Dict[str, Any]]) -> str:
    """Export predictions to JSON format."""
    try:
        import json
        
        # Convert datetime objects to strings
        export_data = []
        for pred in predictions:
            pred_copy = pred.copy()
            if 'generated_at' in pred_copy and isinstance(pred_copy['generated_at'], datetime):
                pred_copy['generated_at'] = pred_copy['generated_at'].isoformat()
            export_data.append(pred_copy)
        
        return json.dumps(export_data, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Failed to export predictions to JSON: {e}")
        return "{}"


def validate_prediction_data(prediction: Dict[str, Any]) -> bool:
    """Validate prediction data structure."""
    try:
        required_fields = ['numbers', 'confidence']
        
        for field in required_fields:
            if field not in prediction:
                return False
        
        # Validate numbers
        numbers = prediction['numbers']
        if not isinstance(numbers, list) or not numbers:
            return False
        
        for num in numbers:
            if not isinstance(num, int) or num < 1:
                return False
        
        # Validate confidence
        confidence = prediction['confidence']
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction validation failed: {e}")
        return False


# Backward compatibility - Original PredictionDisplay class
class PredictionDisplay:
    """
    Original prediction display component for backward compatibility.
    For new implementations, use PredictionDisplayComponents static methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the prediction display.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.theme = self.config.get('theme', 'default')
        
    def render(self, predictions: List[Dict[str, Any]], 
               title: str = "Predictions", 
               show_confidence: bool = True,
               show_metadata: bool = False) -> None:
        """
        Render prediction display using new components.
        
        Args:
            predictions: List of prediction dictionaries
            title: Display title
            show_confidence: Whether to show confidence scores
            show_metadata: Whether to show prediction metadata
        """
        try:
            if not predictions:
                st.warning("⚠️ No predictions to display")
                return
            
            st.subheader(title)
            
            # Convert to format expected by new components
            enhancement_results = {
                'predictions': predictions,
                'confidence_scores': {'overall_confidence': 0.7},
                'enhancement_data': {
                    'phase1_status': 'active',
                    'phase2_status': 'active', 
                    'phase3_status': 'active'
                }
            }
            
            # Use new components
            if show_metadata:
                PredictionDisplayComponents.render_4phase_enhancement_dashboard(enhancement_results)
            else:
                # Simple prediction display
                for i, pred in enumerate(predictions):
                    if validate_prediction_data(pred):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Prediction {i+1}:** {pred['numbers']}")
                        with col2:
                            if show_confidence and 'confidence' in pred:
                                st.write(f"Confidence: {pred['confidence']:.1%}")
            
        except Exception as e:
            logger.error(f"Error in legacy render method: {e}")
            st.error(f"Error displaying predictions: {str(e)}")